import argparse
import os
import sys
from pathlib import Path

# Required by deterministic CUDA matrix operations; set before importing torch.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from sensorline.config import DEVICES, load_config, load_data_description, validate_config
from sensorline.finetune import finetune, finetune_config
from sensorline.search_config import load_best_parameters, load_search_config, read_json_object
from sensorline.trainer import train


# Command-line names of the fine-tuning options.
FINETUNE_FLAGS = {
    "lr_scale": "learning_rate_scale",
    "replay_ratio": "replay_ratio",
    "max_f1_drop": "max_f1_drop",
    "max_count_mae_rise": "max_count_mae_rise",
}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train a YAML-defined SensorLine bbox detector: once, with an Optuna "
                    "search, or by fine-tuning a trained checkpoint on new data"
    )
    parser.add_argument("--config", required=True, help="Model YAML")
    parser.add_argument(
        "--data", metavar="DATASET",
        help="Train on this dataset instead of the data section of the YAML: a folder "
             "with data.yaml or that file",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--optuna", metavar="SEARCH.json", help="Run a validation-only parameter search")
    mode.add_argument("--best-params", metavar="BEST.json", help="Train once using saved best parameters")
    parser.add_argument("--trials", type=int, help="Override search trial count; additional trials with --resume")
    parser.add_argument("--resume", action="store_true", help="Continue an existing Optuna study")
    parser.add_argument(
        "--finetune", metavar="BASE.pt",
        help="Fine-tune this checkpoint cautiously on the data of --config; loss, thresholds "
             "and optimisation settings are taken from the checkpoint, not from the YAML",
    )
    parser.add_argument(
        "--lr-scale", type=float,
        help="With --finetune: multiplier of the YAML learning rate (default 0.1)",
    )
    parser.add_argument(
        "--replay-ratio", type=float,
        help="With --finetune: samples of the base training data replayed per new "
             "training sample in each epoch (default 1; 0 disables)",
    )
    parser.add_argument(
        "--max-f1-drop", type=float,
        help="With --finetune: accepted loss of F1 on the base validation data (default 0.005)",
    )
    parser.add_argument(
        "--max-count-mae-rise", type=float,
        help="With --finetune: accepted rise of count MAE on the base validation data (default 0.01)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Override YAML epochs; incompatible with searching training.epochs",
    )
    parser.add_argument("--output-dir", help="Run directory, or study directory with --optuna")
    parser.add_argument("--device", choices=DEVICES)
    parser.add_argument(
        "--seed", type=int,
        help="Override YAML seed, to repeat one configuration with several seeds",
    )
    parser.add_argument(
        "--name",
        help="Optional experiment display name",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the plan without training")
    args = parser.parse_args()
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.trials is not None and args.trials < 1:
        parser.error("--trials must be positive")
    if not args.optuna and (args.trials is not None or args.resume):
        parser.error("--trials and --resume require --optuna")
    if args.finetune and args.optuna:
        parser.error("--finetune cannot be combined with --optuna")
    if not args.finetune and any(getattr(args, flag) is not None for flag in FINETUNE_FLAGS):
        parser.error("--lr-scale, --replay-ratio, --max-f1-drop and --max-count-mae-rise require --finetune")
    return args


def apply_overrides(config: dict, args) -> None:
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.output_dir:
        config["training"]["output_dir"] = str(Path(args.output_dir).resolve())
    if args.device:
        config["training"]["device"] = args.device
    if args.seed is not None:
        config["training"]["seed"] = args.seed


def prepare_search(args, config):
    """Load the search JSON and decide where its study lives."""
    search = load_search_config(args.optuna, config)
    if args.epochs is not None and "training.epochs" in search["parameters"]:
        raise ValueError("--epochs cannot be combined with searching training.epochs")
    if args.output_dir:
        search["output_dir"] = str(Path(args.output_dir).resolve())
    return search


def prepare_finetune(args, config):
    """Fine-tuning options and the config that continues the base run.

    The base checkpoint must fit the YAML. Its loss, thresholds and optimisation
    settings then replace those of the YAML, and every replaced value is printed.
    """
    options = {"checkpoint": args.finetune}
    for flag, option in FINETUNE_FLAGS.items():
        if getattr(args, flag) is not None:
            options[option] = getattr(args, flag)
    config, options, replaced = finetune_config(config, options)
    if replaced:
        print("Hyperparameters of the base checkpoint replace those of the YAML:", flush=True)
        for path, (written, inherited) in replaced.items():
            print(f"  {path}: {written!r} -> {inherited!r}", flush=True)
    else:
        print("Hyperparameters of the base checkpoint are those of the YAML.", flush=True)
    return config, options


def check_best_parameters(args) -> None:
    if args.epochs is not None and "training.epochs" in read_json_object(args.best_params).get("params", {}):
        raise ValueError("--epochs cannot be combined with best parameters that set training.epochs")


def prepare_run(args):
    """Validated config and what the chosen mode adds: a search, fine-tuning options, a selection.

    Later inputs replace earlier ones: the YAML, the dataset of --data, the base
    checkpoint of --finetune, --best-params, then the single command-line overrides.
    """
    config = load_config(args.config)
    if args.data:
        # The dataset brings its own paths and storage; the model YAML stays as written.
        config["data"] = load_data_description(args.data)
    tuning = None
    if args.finetune:
        config, tuning = prepare_finetune(args, config)
    selection = None
    if args.best_params:
        check_best_parameters(args)
        config, selection = load_best_parameters(args.best_params, config)
    apply_overrides(config, args)
    validate_config(config)
    search = prepare_search(args, config) if args.optuna else None
    return config, search, tuning, selection


def execute_run(config, search, tuning, selection, args):
    if search is not None:
        from sensorline.optuna_search import run_search
        return run_search(config, search, n_trials=args.trials, resume=args.resume, name=args.name)
    if tuning is not None:
        return finetune(config, tuning, name=args.name, selection=selection)
    return train(config, name=args.name, selection=selection)


def print_plan(config, search, tuning, args):
    directory = search["output_dir"] if search else config["training"]["output_dir"]
    budget = f"epochs={config['training']['epochs']}"
    if search:
        budget += f", trials={args.trials or search['n_trials']}, resume={args.resume}"
    if tuning:
        rate = config["training"]["learning_rate"] * tuning["learning_rate_scale"]
        budget += (
            f", finetune={Path(tuning['checkpoint']).parent.name}/{Path(tuning['checkpoint']).name}"
            f", learning_rate={rate:g}, replay_ratio={tuning['replay_ratio']:g}"
        )
    print(f"{Path(args.config).name} -> {directory} ({budget})", flush=True)


def main():
    args = parse_arguments()
    # Validate the model and the mode's inputs before the expensive training starts.
    config, search, tuning, selection = prepare_run(args)
    if args.dry_run:
        print_plan(config, search, tuning, args)
        return
    return execute_run(config, search, tuning, selection, args)


if __name__ == "__main__":
    main()
