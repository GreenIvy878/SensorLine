"""Write a small synthetic dataset in the format of the real one (see sensorline.toydata)."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sensorline.toydata import DEFAULT_BRIGHTNESS, DEFAULT_LENGTHS, make_toy_dataset


def main():
    parser = argparse.ArgumentParser(description="Write a synthetic sensor-line dataset with YOLO labels")
    parser.add_argument("--output", default="toy_data", help="New or empty folder (default: toy_data)")
    parser.add_argument("--train", type=int, default=300, help="Training frames (default 300)")
    parser.add_argument("--val", type=int, default=80, help="Validation frames (default 80)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sensors", type=int, default=40, help="Sensors per frame (default 40)")
    parser.add_argument("--lengths", type=float, nargs=2, metavar=("LOW", "HIGH"), default=DEFAULT_LENGTHS,
                        help="Object length range in time steps (default 4 10.5)")
    parser.add_argument("--brightness", type=float, nargs=2, metavar=("LOW", "HIGH"), default=DEFAULT_BRIGHTNESS,
                        help="Object peak brightness range, of 1 (default 0.55 1)")
    args = parser.parse_args()
    if args.train < 1 or args.val < 1:
        parser.error("--train and --val must be positive")
    try:
        root = make_toy_dataset(args.output, args.train, args.val, args.seed, args.sensors,
                                tuple(args.lengths), tuple(args.brightness))
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"Toy dataset: {root.resolve()} ({args.train} train, {args.val} val frames)")


if __name__ == "__main__":
    main()
