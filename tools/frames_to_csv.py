"""Join the frames of a split folder along time into one CSV recording.

The repository holds frames, not the long recordings they were cut from. Joined in
name order, the frames give a recording to run track.py on: the CSV has a time
column `t` (sample index) and one column per sensor holding the stored pixel values
(data.input_kind: pixels). Frames that were not consecutive in the original recording
meet at a seam that did not exist there.
"""

import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from PIL import Image

from sensorline.data import find_images


def join_frames(split: Path, output: Path, sensor_axis: str = "x", limit: int | None = None) -> int:
    paths = find_images(split / "images")[:limit]
    output.mkdir(parents=True, exist_ok=True)
    start = 0
    with (output / "recording.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = None
        for path in paths:
            with Image.open(path) as image:
                values = np.array(image)
            # Rows of the CSV are polls: [time, sensor].
            polls = values if sensor_axis == "x" else values.T
            if writer is None:
                writer = csv.writer(stream)
                writer.writerow(["t", *(f"s{index:02d}" for index in range(polls.shape[1]))])
            for row in polls:
                writer.writerow([start, *row.tolist()])
                start += 1
    return start


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("split", type=Path, help="A split folder with images/")
    parser.add_argument("output", type=Path, help="Folder for recording.csv")
    parser.add_argument("--sensor-axis", choices=("x", "y"), default="x")
    parser.add_argument("--limit", type=int, help="Only the first N frames")
    args = parser.parse_args()
    samples = join_frames(args.split, args.output, args.sensor_axis, args.limit)
    print(f"{samples} samples: {args.output.resolve()}")


if __name__ == "__main__":
    main()
