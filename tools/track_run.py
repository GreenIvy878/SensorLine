"""Count objects in one CSV recording without the screen, and print the result.

For scripts and checks; day to day use `python track.py`.
"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sensorline.tracking.pipeline import track_recording
from sensorline.tracking.settings import load_settings

DEFAULT_CONFIG = ROOT / "configs" / "tracking" / "default.yaml"


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--csv", required=True, help="Recording: one row per poll of the sensor line")
    parser.add_argument("--model", required=True, help="Weights (.pt checkpoint); its passport is kept next to it")
    parser.add_argument("--config", help=f"Changes to the defaults of {DEFAULT_CONFIG.relative_to(ROOT)}")
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                        help="Override one option, e.g. --set windows.stride=10; may repeat")
    parser.add_argument("--run-dir", help="Folder for the run logs (default: a new one under output.runs_root)")
    parser.add_argument("--streaming", action="store_true",
                        help="One window per forward pass, as on the device; the results are the same")
    return parser, parser.parse_args()


def main():
    parser, args = parse_arguments()
    try:
        settings = load_settings(DEFAULT_CONFIG, args.config, args.set)
        directory, summary = track_recording(args.csv, args.model, settings, run_dir=args.run_dir,
                                             streaming=args.streaming)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    rtf = f", RTF {summary['rtf']:.4f}" if summary["rtf"] is not None else ""
    print(f"Counted {summary['count']} objects in {summary['windows']} windows, "
          f"{summary['segments']} segment(s); tracks {summary['tracks']}; "
          f"window {summary['window_ms_mean']:.2f} ms mean, {summary['window_ms_p95']:.2f} ms p95{rtf}")
    print(f"Run: {directory.resolve()}")


if __name__ == "__main__":
    main()
