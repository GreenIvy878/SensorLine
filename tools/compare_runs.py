"""Compare models with the baseline by a paired bootstrap over the frames they were evaluated on."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sensorline.comparison import METRICS, paired_bootstrap, read_samples, samples_path


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Paired bootstrap of a metric difference against a baseline. Each argument is a "
                    "training run, a test.py report or an Optuna study (its best trial)."
    )
    parser.add_argument("baseline", help="Run, report or study folder of the baseline")
    parser.add_argument("others", nargs="+", metavar="other", help="Folders to compare with it")
    parser.add_argument("--metric", choices=METRICS, default="f1")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    return parser, parser.parse_args()


def name_of(path: Path) -> str:
    return path.resolve().name


def main():
    parser, args = parse_arguments()
    try:
        baseline = read_samples(samples_path(args.baseline))
        rows = []
        for other in args.others:
            result = paired_bootstrap(baseline, read_samples(samples_path(other)), args.metric,
                                      args.resamples, args.confidence, args.seed)
            rows.append((name_of(Path(other)), result))
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))

    first = rows[0][1]
    print(f"{args.metric} on {first['frames']} frames; baseline {name_of(Path(args.baseline))}: "
          f"{first['baseline']:.4f}, {first['errors'][0]} errors (FP+FN)")
    print(f"paired bootstrap: {args.resamples} resamples, {args.confidence:.0%} interval, seed {args.seed}")
    width = max(len(name) for name, _ in rows)
    for name, result in rows:
        low, high = result["interval"]
        verdict = "differs" if result["distinguishable"] else "not distinguishable"
        print(f"{name:<{width}}  {result['other']:.4f}  errors {result['errors'][1]:>4}  "
              f"delta {result['difference']:+.4f}  [{low:+.4f}, {high:+.4f}]  {verdict}")


if __name__ == "__main__":
    main()
