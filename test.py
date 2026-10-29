"""Evaluate a checkpoint or export predictions for unlabeled signals."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import torch

from sensorline.checkpoint import load_checkpoint, require_same_model_and_signal
from sensorline.config import DEVICES, load_config, validate_border_ignore, with_defaults
from sensorline.inference import evaluate_checkpoint, save_report
from sensorline.runtime import choose_device


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate/infer a saved bbox detector; default split is val"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--data-dir", help="YOLO split root with images/ and labels/")
    parser.add_argument(
        "--unlabeled", action="store_true",
        help="Inference only; ignore labels and skip metrics",
    )
    parser.add_argument(
        "--border-ignore", type=float, metavar="SAMPLES",
        help="Score with this evaluation.border_ignore instead of the checkpoint's own: objects cut "
             "by a time edge of the frame to fewer samples are counted neither as found nor as missed "
             "(0 counts everything)",
    )
    parser.add_argument("--device", choices=DEVICES)
    parser.add_argument("--output-dir", help="New directory for report and predictions")
    return parser, parser.parse_args()


# Not a threshold of the model but a rule of scoring: the checkpoint's own value is used,
# and --border-ignore replaces it.
SCORING_RULES = ("border_ignore",)


def thresholds(evaluation: dict) -> dict:
    """The evaluation section as it applies, without the scoring rules."""
    return {key: value for key, value in with_defaults("evaluation", evaluation).items() if key not in SCORING_RULES}


def validate_checkpoint_config(config: dict, saved: dict) -> None:
    require_same_model_and_signal(config, saved)
    if thresholds(config["evaluation"]) != thresholds(saved["evaluation"]):
        raise ValueError(
            "Evaluation thresholds differ from the checkpoint; "
            "keep training thresholds for independent testing"
        )


def output_directory(args) -> Path:
    """Where the report goes; refused before the evaluation if it already holds files."""
    if args.output_dir:
        directory = Path(args.output_dir)
    else:
        name = "inference" if args.unlabeled else f"evaluation_{args.split}"
        directory = Path(args.checkpoint).parent / name
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(
            f"Output directory already contains files: {directory}; "
            "choose a new --output-dir"
        )
    return directory


def run_evaluation(args) -> Path:
    config = load_config(args.config)
    if args.border_ignore is not None:
        validate_border_ignore(args.border_ignore, "--border-ignore")
    torch.set_num_threads(config["training"]["cpu_threads"])
    device = choose_device(args.device or config["training"]["device"])

    checkpoint = load_checkpoint(args.checkpoint, device)
    validate_checkpoint_config(config, checkpoint["config"])
    if not args.data_dir and args.split not in config["data"]:
        raise ValueError(
            f"No data.{args.split}; supply --data-dir for an independent test dataset"
        )
    directory = output_directory(args)

    # The YAML names the splits and stores its own folders; a --data-dir is read
    # as its own data.yaml declares, otherwise as the YAML stores its data.
    labeled = not args.unlabeled
    result = evaluate_checkpoint(
        checkpoint, args.checkpoint, device, config["data"], args.split,
        runtime=config["training"], folder=args.data_dir or None, owner=str(Path(args.config).resolve()),
        labeled=labeled, with_loss=labeled and args.split == "val", border_ignore=args.border_ignore,
    )
    directory.mkdir(parents=True, exist_ok=True)
    save_report(directory, result)
    print(result["report"]["metrics"])
    print(f"Report: {directory.resolve()}")
    return directory


def main():
    parser, args = parse_arguments()
    try:
        run_evaluation(args)
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
