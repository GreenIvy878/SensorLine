"""Find where the training frames were cut from the raw recordings, and check the transform.

The training frames are 8-bit images; the conveyor recordings are CSV files of 12-bit
ADC codes, one row per poll of the 40 sensors. The transform between them was not kept
with the data; it was reconstructed from the data itself:

  1. mean of every `decimate` polls, on a grid aligned to the first row of the file;
  2. frames are consecutive tiles of `window` such samples;
  3. minus the median of all values of the tile, times 255 / the tile maximum,
     rounded, clipped to [0, 255]  (sensorline.tracking.preprocess.render_window).

This tool renders every tile of every recording with that transform and matches it to
the frames: a frame is located when one tile reproduces it within --max-error pixel
values. It writes, into the dataset root:

  recording_transform.json  the transform with its verification; the model passport
                            reads it (data.input_kind: raw_adc)

Tracking does not use the located frames or their labels: it only needs the transform.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from PIL import Image

from sensorline.artifacts import file_sha256, write_json
from sensorline.data import find_images
from sensorline.tracking.preprocess import TRANSFORM_VERSION, decimate, render_window

CLIP = [0, 255]
CHUNK = 64


def read_recording(path: Path, sensors: int) -> np.ndarray:
    """[polls, sensors] ADC codes of a recording (40 values per row, trailing comma allowed)."""
    return np.loadtxt(path, delimiter=",", usecols=range(sensors), dtype=np.float64, ndmin=2)


def recording_order(path: Path):
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


def tiles(polls: np.ndarray, factor: int, window: int, transform: dict):
    """Rendered tiles [n, time, sensor], their start in model samples and their peaks."""
    count = len(polls) // (factor * window)
    if count == 0:
        return np.empty((0, window, polls.shape[1])), [], []
    samples = decimate(polls[:count * factor * window].T, factor)   # [sensors, count·window]
    rendered, peaks = [], []
    for index in range(count):
        pixels, peak = render_window(samples[:, index * window:(index + 1) * window], transform)
        rendered.append(pixels.T)
        peaks.append(peak)
    return np.stack(rendered), [index * window for index in range(count)], peaks


def locate(recordings, frames, factor, window, sensors, transform, max_error):
    """For every frame the best tile over all recordings: (error, recording, start, peak, exact share)."""
    best = [(np.inf, None, None, None, None)] * len(frames)
    flat = frames.reshape(len(frames), -1).astype(np.int16)
    for path in recordings:
        rendered, starts, peaks = tiles(read_recording(path, sensors), factor, window, transform)
        if not len(starts):
            continue
        flat_tiles = rendered.reshape(len(starts), -1).astype(np.int16)
        for begin in range(0, len(starts), CHUNK):
            block = flat_tiles[begin:begin + CHUNK]
            difference = np.abs(block[:, None, :] - flat[None, :, :])     # [tiles, frames, pixels]
            errors = difference.max(axis=2)
            for offset, row in enumerate(errors):
                index = begin + offset
                for frame in np.flatnonzero(row <= max_error):
                    if row[frame] < best[frame][0]:
                        exact = float((difference[offset, frame] == 0).mean())
                        best[frame] = (int(row[frame]), path.name, starts[index], peaks[index], exact)
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--recordings", type=Path, required=True, help="Folder of CSV recordings")
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset root with train/ and val/")
    parser.add_argument("--decimate", type=int, default=10, help="Polls per model sample")
    parser.add_argument("--poll-rate", type=float, default=200.0, help="Polls per second of the recordings")
    parser.add_argument("--max-error", type=int, default=4, help="Largest pixel difference of a located frame")
    parser.add_argument("--sensors", type=int, default=40)
    args = parser.parse_args()

    splits = [split for split in ("train", "val", "test") if (args.dataset / split / "images").is_dir()]
    paths = [(split, path) for split in splits for path in find_images(args.dataset / split / "images")]
    frames = np.stack([np.array(Image.open(path)) for _, path in paths])        # [n, time, sensor]
    window = frames.shape[1]
    transform = {
        "version": TRANSFORM_VERSION,
        "decimate": args.decimate,
        "poll_rate_hz": args.poll_rate,
        "decimate_mode": "mean of consecutive polls, grid aligned to the first row of the file",
        "baseline": "median of all values of the window",
        "normalize": "255 / maximum of the window above the baseline",
        "round": "nearest",
        "clip": CLIP,
        "window": window,
    }
    recordings, duplicates = unique_recordings(args.recordings)
    best = locate(recordings, frames, args.decimate, window, args.sensors, transform, args.max_error)

    located = [(split, path, found) for (split, path), found in zip(paths, best) if found[1] is not None]
    adc_max = max(float(read_recording(path, args.sensors).max()) for path in recordings)
    errors = np.array([found[0] for _, _, found in located])
    transform.update({
        "adc_range": [0, 4095 if adc_max <= 4095 else adc_max],
        "min_peak": min(found[3] for _, _, found in located),
        "verification": {
            "frames": len(paths), "located": len(located), "max_error_allowed": args.max_error,
            "error_counts": {str(value): int((errors == value).sum()) for value in range(args.max_error + 1)},
            "exact_pixel_share": float(np.mean([found[4] for _, _, found in located])),
            "recordings": [path.name for path in recordings],
            "duplicate_recordings": duplicates,
            "frames_by_recording": {name: sum(1 for *_, f in located if f[1] == name)
                                    for name in sorted({f[1] for *_, f in located}, key=lambda n: recording_order(Path(n)))},
        },
    })
    write_json(args.dataset / "recording_transform.json", transform)

    # Identical frames of the dataset land on the same tile: the first is kept, the others are listed as copies.
    tiles_seen, copies = {}, {}
    for split, path, found in located:
        key = (found[1], found[2])
        if key in tiles_seen:
            copies[path.name] = tiles_seen[key]
        else:
            tiles_seen[key] = path.name
    transform["verification"]["copies_of_frames"] = copies
    write_json(args.dataset / "recording_transform.json", transform)
    if copies:
        print(f"Identical frames on one tile, kept once: {copies}")
    print(f"Located {len(located)} of {len(paths)} frames in {len(transform['verification']['frames_by_recording'])} "
          f"recordings; pixel errors {transform['verification']['error_counts']}, exact pixels "
          f"{transform['verification']['exact_pixel_share']:.1%}; weakest frame peak {transform['min_peak']:.1f} ADC")
    if duplicates:
        print(f"Identical recordings skipped: {duplicates}")
    print(f"Written to {args.dataset.resolve()}")


if __name__ == "__main__":
    main()
