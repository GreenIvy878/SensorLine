"""One tracking run: CSV -> preprocessing -> windows -> detector -> tracker -> event bus.

The tracker neither draws nor writes files; it publishes WindowResult and Event to the
bus, and the run logs, a live screen or anything else subscribe to it. Offline runs
may group windows into batches for the model (model.batch_windows); the tracker still
receives them one by one in order of k. With batches of one window an offline run logs
exactly the events of a streaming run.
"""

import csv
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

from ..artifacts import write_json
from ..checkpoint import load_checkpoint
from ..runtime import choose_device
from .detector import WindowDetector
from .passport import git_commit, load_passport
from .preprocess import Preprocessor, input_range
from .settings import AUTO, resolve_model_settings
from .source import Block, RecordSource, SegmentEnd
from .tracker import Tracker
from .windows import Windower, resolve_stride


class EventBus:
    """Subscribers get every WindowResult with its events and the tracker after the step.

    `result` is None for the events of closing a segment. The tracker is passed for
    reading only (a live screen shows the active tracks).
    """

    def __init__(self, subscribers=()):
        self.subscribers = list(subscribers)

    def publish(self, result, events, tracker=None) -> None:
        for subscriber in self.subscribers:
            subscriber.on_window(result, events, tracker)

    def finish(self, tracker, summary) -> None:
        for subscriber in self.subscribers:
            subscriber.on_finish(tracker, summary)


DETECTION_COLUMNS = ("k", "segment", "t0", "raw_index", "score", "nms_kept", "accepted",
                     "p1", "t1_local", "p2", "t2_local", "t1", "t2", "cut_left", "cut_right")
EVENT_COLUMNS = ("t_glob", "k", "type", "track_id", "payload")
TIMELINE_COLUMNS = ("k", "segment", "t0", "last", "skip", "peak", "candidates", "detections", "count",
                    "active", "infer_ms", "window_ms")


class RunLogs:
    """detections.csv, events.csv and timeline.csv while the run goes, tracks.csv at its end."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.files, self.writers = {}, {}
        for name, columns in (("detections", DETECTION_COLUMNS), ("events", EVENT_COLUMNS),
                              ("timeline", TIMELINE_COLUMNS)):
            stream = (directory / f"{name}.csv").open("w", newline="", encoding="utf-8")
            self.files[name] = stream
            self.writers[name] = csv.DictWriter(stream, fieldnames=columns)
            self.writers[name].writeheader()
        self.count = 0
        self.active = 0

    def on_window(self, result, events, tracker=None) -> None:
        for event in events:
            self.writers["events"].writerow(event.row())
            if event.type == "counted":
                self.count = event.payload["count"]
        if result is None:
            return
        for detection, (kept, accepted) in zip(result.detections_pre_nms, result.flags):
            local, glob = detection.box_local, detection.box_global
            self.writers["detections"].writerow({
                "k": result.k, "segment": result.segment, "t0": result.t0, "raw_index": detection.raw_index,
                "score": detection.score, "nms_kept": int(kept), "accepted": int(accepted),
                "p1": local.p1, "t1_local": local.t1, "p2": local.p2, "t2_local": local.t2,
                "t1": glob.t1, "t2": glob.t2, "cut_left": int(detection.cut_left), "cut_right": int(detection.cut_right),
            })
        self.writers["timeline"].writerow({
            "k": result.k, "segment": result.segment, "t0": result.t0, "last": int(result.last),
            "skip": result.extra.get("skip") or "", "peak": result.extra.get("peak"),
            "candidates": len(result.detections_pre_nms), "detections": len(result.detections),
            "count": self.count, "active": result.extra.get("active", 0),
            "infer_ms": round(result.infer_ms, 4), "window_ms": round(result.window_ms, 4),
        })

    def close(self) -> None:
        for stream in self.files.values():
            stream.close()

    def on_finish(self, tracker, summary) -> None:
        self.close()
        rows = tracker.track_rows()
        with (self.directory / "tracks.csv").open("w", newline="", encoding="utf-8") as stream:
            if rows:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)


def run_directory(runs_root, csv_path, checkpoint_path) -> Path:
    model = f"{Path(checkpoint_path).parent.name}-{Path(checkpoint_path).stem}"
    name = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{Path(csv_path).stem}_{model}"
    directory = Path(runs_root) / name
    suffix = 1
    while directory.exists():
        suffix += 1
        directory = Path(runs_root) / f"{name}_{suffix}"
    directory.mkdir(parents=True)
    return directory


def passport_summary(passport: dict) -> dict:
    """The passport without its long list of box lengths, for the resolved config."""
    summary = {key: value for key, value in passport.items() if key != "trace_lengths_time"}
    summary["trace_lengths_time"] = {"count": len(passport["trace_lengths_time"])}
    return summary


def sample_rate(settings: dict, source: RecordSource) -> float | None:
    data = settings["data"]
    if data["sample_rate_hz"] != AUTO:
        return float(data["sample_rate_hz"])
    step = source.report["time_step_median"]
    if step is None or data["time_unit_s"] is None:
        return None
    return 1.0 / (step * data["time_unit_s"])


class SegmentDone:
    """Marker of the end of a segment in the stream of windows."""


def stream_windows(source: RecordSource, preprocessor: Preprocessor, windower: Windower):
    """Windows of a recording in order of k, with a SegmentDone after each segment."""
    segment = None
    for item in source:
        if isinstance(item, Block):
            if item.segment != segment:
                segment = item.segment
                windower.start_segment(segment, preprocessor.start_segment(item.start))
            first, samples = preprocessor.push(item.start, item.values)
            if samples.shape[1]:
                yield from windower.push(first, samples)
        elif isinstance(item, SegmentEnd):
            if item.segment == segment:
                yield from windower.finish()
            yield SegmentDone()


def recording_parts(csv_path, settings: dict, passport: dict, log=print):
    """Source, preprocessor and windower of a recording, from resolved settings and a passport."""
    sensors, width = passport["input_shape"]
    data = settings["data"]
    source = RecordSource(csv_path, data, sensors, input_range(passport, data))
    log(source.describe())
    recording_rate = sample_rate(settings, source)
    # Pooling simulates a slower poll rate; the preprocessor sees pooled polls.
    poll_rate = recording_rate / data["pool"] if recording_rate else None
    preprocessor = Preprocessor(passport, data, settings["windows"], poll_rate)
    windower = Windower(sensors, width, settings["windows"]["stride"], settings["windows"]["tail_mode"],
                        render=preprocessor.window)
    return source, preprocessor, windower, recording_rate, poll_rate


class TrackingRun:
    """Everything a run needs, resolved and checked before the first window."""

    def __init__(self, csv_path, checkpoint_path, settings: dict, *, streaming=False, log=print):
        self.log = log
        self.csv_path, self.checkpoint_path = Path(csv_path), Path(checkpoint_path)
        self.device = choose_device(settings["model"]["device"])
        self.checkpoint = load_checkpoint(self.checkpoint_path, self.device)
        self.passport = load_passport(self.checkpoint, self.checkpoint_path, log=log)
        self.settings = resolve_model_settings(settings, self.passport)
        self.geometry = resolve_stride(self.settings, self.passport)
        self.settings["windows"]["stride"] = self.geometry["stride"]
        log(f"Window W = {self.geometry['width']}, stride S = {self.geometry['stride']} ({self.geometry['reason']})")
        if self.geometry["warning"]:
            log("Warning: " + self.geometry["warning"])
        self.batch = 1 if streaming else self.settings["model"]["batch_windows"]
        self.settings["model"]["batch_windows"] = self.batch
        (self.source, self.preprocessor, self.windower,
         self.recording_rate, self.poll_rate) = recording_parts(self.csv_path, self.settings, self.passport, log)
        log(self.preprocessor.describe())

    def resolved(self) -> dict:
        return {
            "settings": self.settings,
            "windows": self.geometry,
            "model": {"checkpoint": str(self.checkpoint_path.resolve()),
                      "checkpoint_sha256": self.passport["checkpoint_sha256"],
                      "device": str(self.device), "passport": passport_summary(self.passport)},
            "csv": {"path": str(self.csv_path.resolve()), "sha256": self.source.report["sha256"],
                    "format": self.source.report["format"]},
            "input": {"kind": self.preprocessor.kind, "polls_per_sample": self.preprocessor.factor,
                      "recording_rate_hz": self.recording_rate, "pool": self.settings["data"]["pool"],
                      "poll_rate_hz": self.poll_rate, "model_rate_hz": self.preprocessor.model_rate_hz,
                      "empty_peak_adc": self.preprocessor.empty_peak, "description": self.preprocessor.describe()},
            "git_commit": git_commit(Path(__file__).resolve().parent),
            "date": datetime.now().astimezone().isoformat(timespec="seconds"),
        }

    def run(self, subscribers=()) -> dict:
        bus = EventBus(subscribers)
        detector = WindowDetector(self.checkpoint, self.passport, self.settings["model"],
                                  self.settings["tracker"]["edge_tol"], self.device)
        tracker = Tracker(self.settings["tracker"])
        pending, timings = [], []
        started = time.perf_counter()

        def flush(all_of_them):
            while pending and (all_of_them or len(pending) >= self.batch):
                group = pending[:self.batch]
                del pending[:self.batch]
                for result in detector(group):
                    if not self.settings["output"]["save_frames"]:
                        result.frame = result.raw = None
                    step_started = time.perf_counter()
                    events = tracker.step(result)
                    result.extra["active"] = len(tracker.active)
                    result.window_ms = result.infer_ms + (time.perf_counter() - step_started) * 1000
                    timings.append((result.infer_ms, result.window_ms, result.extra.get("skip")))
                    bus.publish(result, events, tracker)

        with torch.inference_mode():
            for item in stream_windows(self.source, self.preprocessor, self.windower):
                if isinstance(item, SegmentDone):
                    flush(True)
                    bus.publish(None, tracker.finish_segment(), tracker)
                else:
                    pending.append(item)
                    flush(False)
        elapsed = time.perf_counter() - started
        summary = self.summary(tracker, timings, elapsed)
        bus.finish(tracker, summary)
        return summary

    def summary(self, tracker, timings, elapsed) -> dict:
        report = self.source.report
        rate = self.recording_rate
        duration = report["samples"] / rate if rate else None
        states = {}
        for track in tracker.tracks:
            states[track.state] = states.get(track.state, 0) + 1
        window_ms = np.array([t[1] for t in timings]) if timings else np.zeros(1)
        infer_ms = np.array([t[0] for t in timings]) if timings else np.zeros(1)
        return {
            "count": tracker.count,
            "tracks": states,
            "windows": len(timings),
            "empty_windows": sum(1 for t in timings if t[2] == "empty"),
            "stationary_windows": sum(1 for t in timings if t[2] == "stationary"),
            "polls_per_sample": self.preprocessor.factor,
            "segments": len(report["segments"]),
            "polls": report["samples"],
            "sample_rate_hz": rate,
            "pool": self.settings["data"]["pool"],
            "model_rate_hz": self.preprocessor.model_rate_hz,
            "duration_s": duration,
            "processing_s": elapsed,
            "rtf": elapsed / duration if duration else None,
            # Processing alone, without waiting for real-time playback of a live run.
            "compute_rtf": sum(t[1] for t in timings) / 1000 / duration if duration and timings else None,
            "infer_ms_mean": float(infer_ms.mean()),
            "window_ms_mean": float(window_ms.mean()),
            "window_ms_p95": float(np.percentile(window_ms, 95)),
            "source": report,
        }


def track_recording(csv_path, checkpoint_path, settings, *, run_dir=None, streaming=False, subscribers=(),
                    log=print, prepared=None) -> tuple[Path, dict]:
    """Run tracking on one recording and write the run folder; returns it and the summary.

    `prepared` is a TrackingRun already built from the same arguments (the live screen
    reads its settings before the first window).
    """
    run = prepared or TrackingRun(csv_path, checkpoint_path, settings, streaming=streaming, log=log)
    directory = Path(run_dir) if run_dir else run_directory(settings["output"]["runs_root"], csv_path, checkpoint_path)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.resolved.yaml").write_text(
        yaml.safe_dump(run.resolved(), sort_keys=False, allow_unicode=True), encoding="utf-8")
    logs = RunLogs(directory)
    try:
        summary = run.run([logs, *subscribers])
    finally:
        # A cancelled or failed run never reaches on_finish; its logs must not stay open.
        logs.close()
    write_json(directory / "summary.json", summary)
    return directory, summary

