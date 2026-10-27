"""Read a CSV recording of the sensor line in chunks, as a stream of sensor × time blocks.

The format is detected from the first rows and printed; when it is ambiguous the
reader stops and names the option to set, rather than guessing. A row with a missing
value or a time step larger than `gap_factor` × the usual step ends a segment: the
recording is never interpolated or stitched across such a place.

Stream coordinates: a sample is the 0-based index of its data row (or column of a
transposed file), counted over the whole file, dropped rows included.
"""

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..artifacts import file_sha256
from .settings import AUTO

DELIMITERS = (",", ";", "\t", " ")
TIME_NAMES = re.compile(r"^(t|time|timestamp|ts|time_?s|time_?ms|sec|seconds|millis)$", re.IGNORECASE)
DETECTION_ROWS = 2000
MIN_TIME_ROWS = 3
# Bad rows listed in the report; the count covers all of them.
LISTED_ROWS = 20


@dataclass
class Block:
    segment: int
    start: int            # stream sample of the first column
    values: np.ndarray    # [sensors, n] float64, as read


@dataclass
class SegmentEnd:
    segment: int
    end: int              # stream sample after the last one of the segment


def pooled(items, pool: int):
    """Means of `pool` consecutive polls, as a recording polled `pool` times slower would hold.

    The grid starts at the first row of the file; polls before the first whole group of
    a segment and an incomplete group at its end are dropped. Coordinates become
    indices of pooled polls.
    """
    carry, carry_start = None, None
    for item in items:
        if isinstance(item, SegmentEnd):
            carry, carry_start = None, None
            yield SegmentEnd(item.segment, item.end // pool)
            continue
        values, start = item.values, item.start
        if carry is None:
            carry_start = -(-start // pool) * pool
            values = values[:, carry_start - start:]
        else:
            values = np.concatenate((carry, values), axis=1)
        whole = values.shape[1] // pool * pool
        if whole:
            sensors = values.shape[0]
            yield Block(item.segment, carry_start // pool,
                        values[:, :whole].reshape(sensors, whole // pool, pool).mean(axis=2))
        carry, carry_start = values[:, whole:], carry_start + whole


@dataclass
class SourceFormat:
    delimiter: str
    header: list | None
    orientation: str
    time_col: int | None
    channel_cols: list
    notes: list = field(default_factory=list)


def split_line(line: str, delimiter: str) -> list[str]:
    if delimiter == " ":
        return line.split()
    return next(csv.reader([line], delimiter=delimiter))


def is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def sample_lines(path: Path, count: int) -> list[str]:
    lines = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for line in stream:
            if line.strip():
                lines.append(line.rstrip("\r\n"))
            if len(lines) == count:
                break
    if not lines:
        raise ValueError(f"{path} is empty")
    return lines


def count_data_lines(path: Path) -> int:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return sum(1 for line in stream if line.strip())


def detect_delimiter(lines: list[str], configured) -> str:
    if configured != AUTO:
        return configured
    consistent = []
    for delimiter in DELIMITERS:
        widths = {len(split_line(line, delimiter)) for line in lines}
        if len(widths) == 1 and widths.pop() > 1:
            consistent.append(delimiter)
    # A comma or semicolon file may also split on spaces inside a field; prefer real separators.
    if len(consistent) > 1 and " " in consistent:
        consistent.remove(" ")
    if len(consistent) != 1:
        found = [repr(d) for d in consistent] or ["none"]
        raise ValueError(f"CSV delimiter is ambiguous (consistent: {', '.join(found)}); set data.delimiter")
    return consistent[0]


def column_index(value, header, name: str) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str) and header is not None and value in header:
        return header.index(value)
    raise ValueError(f"{name}: column {value!r} not found (header: {header})")


def strictly_increasing(values: np.ndarray) -> bool:
    return len(values) >= MIN_TIME_ROWS and bool(np.all(np.diff(values) > 0))


def detect_columns(rows: list[list[str]], header, settings: dict, channels: int) -> tuple[int | None, list, list]:
    """Time column and channel columns of a file whose rows are polls."""
    notes = []
    width = len(rows[0])
    numeric = np.array([[float(cell) if is_number(cell) else np.nan for cell in row] for row in rows])
    numeric_cols = [index for index in range(width) if not np.isnan(numeric[:, index]).all()]

    time_col = settings["time_col"]
    if time_col == AUTO:
        named = [index for index, name in enumerate(header or []) if TIME_NAMES.match(name.strip())]
        if len(named) == 1:
            time_col = named[0]
            notes.append(f"time column {header[time_col]!r} found by its name")
        elif len(numeric_cols) == channels:
            time_col = None
            notes.append("no time column: the file has exactly as many numeric columns as sensors")
        elif len(numeric_cols) == channels + 1:
            increasing = [index for index in numeric_cols if strictly_increasing(numeric[:, index])]
            if len(increasing) != 1:
                raise ValueError(
                    f"{len(numeric_cols)} numeric columns for {channels} sensors, and {len(increasing)} of them "
                    "increase strictly; set data.time_col (a name, an index or null)"
                )
            time_col = increasing[0]
            notes.append(f"time column {time_col} found as the only strictly increasing column")
        else:
            raise ValueError(
                f"{len(numeric_cols)} numeric columns for {channels} sensors; set data.time_col and data.channel_cols"
            )
    elif time_col is not None:
        time_col = column_index(time_col, header, "data.time_col")

    channel_cols = settings["channel_cols"]
    if channel_cols == AUTO:
        channel_cols = [index for index in numeric_cols if index != time_col]
    else:
        channel_cols = [column_index(value, header, "data.channel_cols") for value in channel_cols]
    if len(channel_cols) != channels:
        raise ValueError(f"{len(channel_cols)} channel columns, the model expects {channels} sensors; set data.channel_cols")
    if time_col is not None and time_col in channel_cols:
        raise ValueError("data.time_col is also listed in data.channel_cols")
    if time_col is not None and not strictly_increasing(numeric[:, time_col]):
        raise ValueError(f"Time column {time_col} does not increase strictly in the first rows")
    return time_col, channel_cols, notes


def detect_format(path: Path, settings: dict, channels: int) -> SourceFormat:
    lines = sample_lines(path, DETECTION_ROWS)
    delimiter = detect_delimiter(lines, settings["delimiter"])
    rows = [split_line(line, delimiter) for line in lines]
    header = settings["header"]
    if header == AUTO:
        header = not all(is_number(cell) for cell in rows[0])
    header_row = [cell.strip() for cell in rows[0]] if header else None
    data_rows = rows[1:] if header else rows

    orientation = settings["orientation"]
    if orientation == AUTO:
        # A transposed file: one row per sensor, many columns of samples.
        lines_total = count_data_lines(path) - (1 if header else 0)
        orientation = "rows_are_channels" if lines_total == channels and len(rows[0]) > channels + 1 else "rows_are_polls"
    if orientation == "rows_are_channels":
        if settings["time_col"] not in (AUTO, None):
            raise ValueError("A transposed file (rows_are_channels) is read without a time column")
        first = data_rows[0]
        label = 0 if is_number(first[0]) else 1
        return SourceFormat(delimiter, header_row, orientation, None, list(range(label, len(first))),
                            ["one row per sensor: the file is transposed and is read whole"])
    if not data_rows:
        raise ValueError(f"{path} has a header and no data rows")
    time_col, channel_cols, notes = detect_columns(data_rows, header_row, settings, channels)
    return SourceFormat(delimiter, header_row, orientation, time_col, channel_cols, notes)


def parse_rows(rows: list[list[str]], columns: list) -> np.ndarray:
    """[rows, columns] float64; a missing or unparsable cell becomes NaN."""
    try:
        return np.array([[row[index] for index in columns] for row in rows], dtype=np.float64)
    except (ValueError, IndexError):
        values = np.full((len(rows), len(columns)), np.nan)
        for number, row in enumerate(rows):
            for position, index in enumerate(columns):
                if index < len(row) and is_number(row[index]):
                    values[number, position] = float(row[index])
        return values


class RecordSource:
    """Yield Block and SegmentEnd items of one CSV recording; `report` describes what was read."""

    def __init__(self, path, settings: dict, channels: int, value_range):
        self.path = Path(path)
        self.settings = settings
        self.channels = channels
        self.value_range = value_range
        self.format = detect_format(self.path, settings, channels)
        self.report = {
            "path": str(self.path.resolve()),
            "sha256": file_sha256(self.path),
            "format": {
                "delimiter": self.format.delimiter,
                "header": self.format.header is not None,
                "orientation": self.format.orientation,
                "time_col": self.describe_column(self.format.time_col),
                "channel_cols": [self.describe_column(index) for index in self.format.channel_cols],
                "notes": self.format.notes,
            },
            "samples": 0,
            "missing_rows": 0,
            "missing_row_samples": [],
            "out_of_range_values": 0,
            "time_gaps": [],
            "segments": [],
            "time_step_median": self.initial_time_step(),
            "pool": settings["pool"],
        }

    def initial_time_step(self):
        """Median step of the time column over the first rows, or None without one."""
        if self.format.time_col is None:
            return None
        rows = [split_line(line, self.format.delimiter) for line in sample_lines(self.path, DETECTION_ROWS)]
        rows = rows[1:] if self.format.header is not None else rows
        times = parse_rows(rows, [self.format.time_col])[:, 0]
        steps = np.diff(times[~np.isnan(times)])
        return float(np.median(steps)) if len(steps) else None

    def describe_column(self, index):
        if index is None or self.format.header is None:
            return index
        return self.format.header[index]

    def describe(self) -> str:
        fmt = self.report["format"]
        return (f"CSV {self.path.name}: delimiter {fmt['delimiter']!r}, header {fmt['header']}, "
                f"{fmt['orientation']}, time column {fmt['time_col']!r}, {len(fmt['channel_cols'])} channels"
                + "".join(f"; {note}" for note in fmt["notes"]))

    def rows(self):
        """Lists of cells of the data rows, in chunks of chunk_rows."""
        chunk = []
        with self.path.open(encoding="utf-8-sig", newline="") as stream:
            if self.format.delimiter == " ":
                reader = (line.split() for line in stream)
            else:
                reader = csv.reader(stream, delimiter=self.format.delimiter)
            skip_header = self.format.header is not None
            for row in reader:
                if not row or not any(cell.strip() for cell in row):
                    continue
                if skip_header:
                    skip_header = False
                    continue
                chunk.append(row)
                if len(chunk) == self.settings["chunk_rows"]:
                    yield chunk
                    chunk = []
        if chunk:
            yield chunk

    def count_out_of_range(self, values: np.ndarray) -> None:
        low, high = self.value_range
        self.report["out_of_range_values"] += int(((values < low) | (values > high)).sum())

    def __iter__(self):
        items = self.transposed() if self.format.orientation == "rows_are_channels" else self.polls()
        pool = self.settings["pool"]
        yield from (items if pool == 1 else pooled(items, pool))

    def transposed(self):
        rows = [row for chunk in self.rows() for row in chunk]
        if len(rows) != self.channels:
            raise ValueError(f"A transposed file needs {self.channels} rows, found {len(rows)}")
        width = len(self.format.channel_cols)
        values = np.full((self.channels, width), np.nan)
        for index, row in enumerate(rows):
            values[index] = parse_rows([row], self.format.channel_cols)[0]
        missing = np.isnan(values).any(axis=0)
        if missing.any():
            raise ValueError(f"A transposed file has {int(missing.sum())} samples with missing values")
        self.count_out_of_range(values)
        self.report["samples"] = width
        self.report["segments"] = [[0, width]]
        yield Block(0, 0, values)
        yield SegmentEnd(0, width)

    def polls(self):
        time_col, channel_cols = self.format.time_col, self.format.channel_cols
        gap_factor = self.settings["gap_factor"]
        sample = 0
        segment, segment_start = 0, None
        previous_time, step = None, None
        segments = self.report["segments"]

        def end_segment(end):
            nonlocal segment, segment_start
            if segment_start is not None:
                segments.append([segment_start, end])
                item = SegmentEnd(segment, end)
                segment += 1
                segment_start = None
                return item
            return None

        for chunk in self.rows():
            columns = channel_cols + ([time_col] if time_col is not None else [])
            parsed = parse_rows(chunk, columns)
            values, times = parsed[:, :len(channel_cols)], parsed[:, len(channel_cols):]
            if step is None and time_col is not None:
                step = self.report["time_step_median"]
                if step is None:
                    raise ValueError("Too few rows with a time to find the time step")
            run_start = 0
            for index in range(len(parsed)):
                breaks = False
                if np.isnan(parsed[index]).any():
                    self.report["missing_rows"] += 1
                    if len(self.report["missing_row_samples"]) < LISTED_ROWS:
                        self.report["missing_row_samples"].append(sample + index)
                    breaks = True
                elif time_col is not None:
                    time = times[index, 0]
                    if previous_time is not None:
                        delta = time - previous_time
                        if delta <= 0:
                            raise ValueError(f"Time does not increase at sample {sample + index} "
                                             f"({previous_time} then {time})")
                        if delta > gap_factor * step:
                            self.report["time_gaps"].append({"sample": sample + index, "step": float(delta)})
                            breaks = True
                    previous_time = time
                if breaks:
                    if index > run_start:
                        yield self.block(segment, sample + run_start, values[run_start:index])
                        if segment_start is None:
                            segment_start = sample + run_start
                    item = end_segment(sample + index)
                    if item is not None:
                        yield item
                    run_start = index + 1 if np.isnan(parsed[index]).any() else index
            if len(parsed) > run_start:
                if segment_start is None:
                    segment_start = sample + run_start
                yield self.block(segment, sample + run_start, values[run_start:])
            sample += len(parsed)
        self.report["samples"] = sample
        item = end_segment(sample)
        if item is not None:
            yield item

    def block(self, segment: int, start: int, values: np.ndarray) -> Block:
        values = values.T.copy()
        self.count_out_of_range(values)
        return Block(segment, start, values)
