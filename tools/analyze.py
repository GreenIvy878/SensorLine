"""Write the dataset statistics or a run summary as JSON; `python dashboard.py` shows them."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sensorline.analysis import load_or_run
from sensorline.analysis.experiments.experiment_analysis import prepare_experiment
from sensorline.config import load_config


def parse_arguments():
    parser = argparse.ArgumentParser(description="Read-only sensor dataset analyzer")
    parser.add_argument("--config", help="Model YAML whose data section is analyzed")
    parser.add_argument("--output-dir", default="analysis_results")
    parser.add_argument("--refresh", action="store_true", help="Recompute even if cache matches")
    parser.add_argument("--experiment", help="Prepare the model summary of this run without training")
    args = parser.parse_args()
    if not args.config and not args.experiment:
        parser.error("--config or --experiment is required")
    return parser, args


def run(args) -> None:
    if args.experiment:
        output = Path(args.output_dir)
        prepare_experiment(args.experiment, output)
        print(f"Experiment analysis: {(output / 'experiment_analysis.json').resolve()}")
    if not args.config:
        return
    config = load_config(args.config)
    report = load_or_run(config["data"], args.output_dir, config["training"]["seed"], args.refresh)
    print(f"Analysis: {(Path(args.output_dir) / 'analysis.json').resolve()}")
    print(f"Valid samples: {report['splits']['all']['overview']['samples']}; "
          f"input issues: {len(report['issues'])}")


def main():
    parser, args = parse_arguments()
    try:
        run(args)
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
