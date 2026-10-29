"""Run track.py on every distinct CSV recording of a folder and summarize the counts.

A file identical to an earlier one (same SHA-256) is skipped and listed.
"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sensorline.artifacts import file_sha256, write_json, write_rows
from sensorline.tracking.pipeline import track_recording
from sensorline.tracking.settings import load_settings

DEFAULT_CONFIG = ROOT / "configs" / "tracking" / "default.yaml"


def recording_order(path: Path):
    """DATA_2 before DATA_10: by the number in the name, then by the name."""
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return (int(digits) if digits else 0, path.name)


def unique_recordings(folder: Path) -> tuple[list[Path], dict]:
    """Recordings in name order; a file identical to an earlier one is listed as its duplicate."""
    seen, unique, duplicates = {}, [], {}
    for path in sorted(folder.glob("*.csv"), key=recording_order):
        digest = file_sha256(path)
        if digest in seen:
            duplicates[path.name] = seen[digest]
            continue
        seen[digest] = path.name
        unique.append(path)
    return unique, duplicates


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--recordings", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True, help="New folder: one run folder per recording")
    parser.add_argument("--config")
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    args = parser.parse_args()

    settings = load_settings(DEFAULT_CONFIG, args.config, args.set)
    recordings, duplicates = unique_recordings(args.recordings)
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for path in recordings:
        directory, summary = track_recording(path, args.model, settings, run_dir=args.output / path.stem,
                                             log=lambda *_: None)
        row = {"recording": path.name, "polls": summary["polls"], "windows": summary["windows"],
               "empty_windows": summary["empty_windows"], "stationary_windows": summary["stationary_windows"],
               "count": summary["count"],
               "rejected": summary["tracks"].get("rejected", 0), "merged": summary["tracks"].get("merged", 0),
               "window_ms_mean": round(summary["window_ms_mean"], 3)}
        rows.append(row)
        print(", ".join(f"{key} {value}" for key, value in row.items()), flush=True)
    columns = []
    for row in rows:
        columns += [key for key in row if key not in columns]
    write_rows(args.output / "summary.csv", [{key: row.get(key, "") for key in columns} for row in rows])
    write_json(args.output / "duplicates.json", duplicates)
    print(f"Summary: {(args.output / 'summary.csv').resolve()}")


if __name__ == "__main__":
    main()
