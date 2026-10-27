"""Cautious fine-tuning of a trained detector on a new dataset.

The run starts from the weights of a base checkpoint and trains on the data of
the YAML: the new, typically hard, samples. Three precautions protect what the
base model already does well:

- the learning rate is a fraction of the YAML value;
- every epoch replays samples of the data the base checkpoint was trained on;
- an epoch may become the checkpoint only if the validation data of the base
  checkpoint (the "reference") did not degrade beyond the tolerances.

The base weights are the first checkpoint candidate (epoch 0), so an epoch has
to beat them on the new validation data to replace them.
"""

from copy import deepcopy
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .artifacts import file_sha256, update_experiment, write_json
from .augmentation import augmentation_options
from .checkpoint import load_checkpoint, require_same_model_and_signal
from .config import SECTIONS
from .data import build_dataset, dataset_fingerprint, dataset_signal_hashes
from .evaluation import evaluate, evaluation_settings
from .models import Detector
from .runtime import choose_device, configure_reproducibility, make_loader
from .trainer import build_trainer, open_run, record_failure, record_run_inputs
from .validation import finite_number, require_mapping


FINETUNE_DEFAULTS = {
    # Multiplier of training.learning_rate.
    "learning_rate_scale": 0.1,
    # Reference training samples replayed per new training sample in each epoch.
    "replay_ratio": 1.0,
    # Largest accepted loss of F1 and rise of count MAE on the reference validation.
    "max_f1_drop": 0.005,
    "max_count_mae_rise": 0.01,
}


def finetune_options(options: dict) -> dict:
    """Validated options with defaults; `checkpoint` is the only required key."""
    require_mapping(options, "finetune")
    unknown = set(options) - set(FINETUNE_DEFAULTS) - {"checkpoint"}
    if unknown:
        raise ValueError(f"Unknown finetune options: {sorted(unknown)}")
    checkpoint = options.get("checkpoint")
    if not isinstance(checkpoint, str) or not checkpoint.strip():
        raise ValueError("finetune.checkpoint must be the path of a base checkpoint")
    if not Path(checkpoint).is_file():
        raise ValueError(f"Base checkpoint not found: {checkpoint}")

    result = {**FINETUNE_DEFAULTS, **options}
    scale = finite_number(result["learning_rate_scale"], "finetune.learning_rate_scale")
    if not 0 < scale <= 1:
        raise ValueError("finetune.learning_rate_scale must be in (0, 1]")
    for key in ("replay_ratio", "max_f1_drop", "max_count_mae_rise"):
        if finite_number(result[key], f"finetune.{key}") < 0:
            raise ValueError(f"finetune.{key} must be nonnegative")
    result["checkpoint"] = str(Path(checkpoint).resolve())
    return result


def load_base(config: dict, options: dict, device) -> dict:
    """The base checkpoint, once its weights are known to fit the YAML."""
    base = load_checkpoint(options["checkpoint"], device)
    require_same_model_and_signal(config, base["config"])
    return base


# What made the base model. A fine-tuning run continues with these settings of the
# base run, not with those of the YAML: a model found by a parameter search would
# otherwise silently go on with the untuned values. The YAML still gives the data,
# the number of epochs and where and how to run.
BASE_TRAINING_KEYS = (
    "optimizer", "learning_rate", "weight_decay", "momentum", "batch_size",
    "gradient_clip", "augmentation", *SECTIONS["training"]["unsaved"],
)


def unscaled_learning_rate(base: dict) -> float:
    """Learning rate the base run started from.

    A fine-tuned base stores the scaled rate in its config; taking that one would
    shrink the rate again with every run of a chain.
    """
    saved = base["provenance"].get("finetune", {})
    return saved.get("unscaled_learning_rate", base["config"]["training"]["learning_rate"])


def flat_parameters(config: dict) -> dict:
    """Loss, thresholds and the inherited training settings by dotted path."""
    training = {key: config["training"][key] for key in BASE_TRAINING_KEYS if key in config["training"]}
    flat = {}
    for section, values in (("loss", config["loss"]), ("evaluation", config["evaluation"]),
                            ("training", training)):
        for key, value in values.items():
            if isinstance(value, dict):
                flat.update({f"{section}.{key}.{name}": item for name, item in value.items()})
            else:
                flat[f"{section}.{key}"] = value
    return flat


def inherit_base_parameters(config: dict, saved: dict, learning_rate: float) -> tuple[dict, dict]:
    """`config` with the hyperparameters of the base run, and what that replaced.

    `saved` is the config of the base run and `learning_rate` its unscaled rate.
    The second value maps a dotted path to (value of the YAML, value of the base);
    None stands for an option that one of them does not set.
    """
    result = deepcopy(config)
    result["loss"] = deepcopy(saved["loss"])
    result["evaluation"] = deepcopy(saved["evaluation"])
    for key in BASE_TRAINING_KEYS:
        if key in saved["training"]:
            result["training"][key] = deepcopy(saved["training"][key])
        else:
            result["training"].pop(key, None)
    result["training"]["learning_rate"] = learning_rate
    if "augmentation" in result["training"]:
        # Read as a YAML would be, so options of older saved runs compare like current ones.
        result["training"]["augmentation"] = augmentation_options(result["training"]["augmentation"])
    before, after = flat_parameters(config), flat_parameters(result)
    replaced = {
        path: (before.get(path), after.get(path))
        for path in sorted(before.keys() | after.keys())
        if before.get(path) != after.get(path)
    }
    return result, replaced


def finetune_config(config: dict, options: dict) -> tuple[dict, dict, dict]:
    """Validated options and `config` continued from the base run; train.py and the dashboard share it.

    The base checkpoint must fit the YAML before anything is trained. The third
    value maps each replaced hyperparameter to (value of the YAML, value of the base).
    """
    options = finetune_options(options)
    base = load_base(config, options, "cpu")
    config, replaced = inherit_base_parameters(config, base["config"], unscaled_learning_rate(base))
    return config, options, replaced


def load_reference(base: dict, with_train: bool) -> dict:
    """Datasets the base checkpoint was trained and validated on, checked against its fingerprints."""
    data = base["config"]["data"]
    reference = {}
    for split in ("train", "val") if with_train else ("val",):
        try:
            dataset = build_dataset(data, split)
        except (OSError, ValueError) as error:
            # The path comes from the checkpoint, not from the YAML the user is looking at.
            raise ValueError(
                f"Reference {split} data of the base checkpoint cannot be read: {error}"
            ) from error
        if dataset_fingerprint(dataset) != base["provenance"].get(f"{split}_fingerprint"):
            raise ValueError(
                f"Reference {split} data differ from the data of the base checkpoint: {data[split]}"
            )
        reference[split] = dataset
    return reference


def shared_signal_count(first, second) -> int:
    """Number of distinct signals that occur in both datasets."""
    first_hashes, _ = dataset_signal_hashes(first)
    second_hashes, _ = dataset_signal_hashes(second)
    return len(first_hashes.keys() & second_hashes.keys())


class ReplayDataset(Dataset):
    """The new training samples followed by random samples of the reference data.

    Every index past the new data returns a random reference sample, so each
    epoch replays a different part of the data the base checkpoint was trained on.
    """

    def __init__(self, new, reference, ratio: float):
        self.new = new
        self.reference = reference
        # More replay than the reference holds would only repeat its samples.
        self.replay_count = min(round(ratio * len(new)), len(reference))

    def __len__(self):
        return len(self.new) + self.replay_count

    def __getitem__(self, index):
        if index < len(self.new):
            return self.new[index]
        return self.reference[int(torch.randint(len(self.reference), ()))]


class ReferenceGuard:
    """Accept an epoch only if the reference validation kept its quality.

    The limits are relative to the base weights measured by this run, with the
    thresholds of this run, not to the metrics stored in the base checkpoint.
    """

    def __init__(self, loader, device, evaluation: dict, options: dict):
        self.loader = loader
        self.device = device
        self.evaluation = evaluation
        self.max_f1_drop = options["max_f1_drop"]
        self.max_count_mae_rise = options["max_count_mae_rise"]
        self.limits = None

    def evaluate(self, model) -> dict:
        metrics, _, _ = evaluate(model, self.loader, self.device, self.evaluation)
        return metrics

    def set_baseline(self, model) -> dict:
        baseline = self.evaluate(model)
        self.limits = {
            "f1": baseline["f1"] - self.max_f1_drop,
            "count_mae": baseline["count_mae"] + self.max_count_mae_rise,
        }
        return baseline

    def check(self, model, epoch: int) -> dict:
        """History columns of one epoch: its reference metrics and the verdict."""
        metrics = self.evaluate(model)
        accepted = (
            metrics["f1"] >= self.limits["f1"]
            and metrics["count_mae"] <= self.limits["count_mae"]
        )
        print(
            f"epoch={epoch} reference_f1={metrics['f1']:.4f} (limit {self.limits['f1']:.4f}) "
            f"reference_count_mae={metrics['count_mae']:.4f} (limit {self.limits['count_mae']:.4f}) "
            f"{'accepted' if accepted else 'rejected'}",
            flush=True,
        )
        row = {f"reference_{name}": value for name, value in metrics.items()}
        row["accepted"] = accepted
        return row


def check_leaks(train_dataset, val_dataset, reference: dict) -> dict:
    """Identical signals that weaken the guard or flatter the new validation."""
    leaks = {"train_in_reference_val": shared_signal_count(train_dataset, reference["val"])}
    if leaks["train_in_reference_val"]:
        print(
            "WARNING: new training data contain signals of the reference validation; "
            "the guard against forgetting is weaker than it looks.",
            flush=True,
        )
    if "train" in reference:
        leaks["reference_train_in_val"] = shared_signal_count(reference["train"], val_dataset)
        if leaks["reference_train_in_val"]:
            print(
                "WARNING: replayed reference training data contain signals of the new "
                "validation; its metrics are optimistic.",
                flush=True,
            )
    return leaks


def check_count_limit(max_count_mae_rise: float, reference_frames: int) -> bool:
    """Warn when one more miscounted frame of the reference validation exceeds the limit."""
    step = 1 / reference_frames
    if max_count_mae_rise >= step:
        return True
    print(
        f"WARNING: max_count_mae_rise {max_count_mae_rise:g} is below the count MAE of one frame "
        f"miscounted by one object on the {reference_frames} reference frames ({step:.4f}); "
        "almost every epoch will be rejected and the base weights kept.",
        flush=True,
    )
    return False


def unprefixed(row: dict, prefix: str) -> dict:
    return {key[len(prefix):]: value for key, value in row.items() if key.startswith(prefix)}


def finetune_result(trainer, baseline: dict) -> dict:
    """Which weights were kept, with their metrics on both validation sets."""
    best_epoch = trainer.best_epoch
    kept = baseline
    if best_epoch > 0:
        row = trainer.history[best_epoch - 1]
        kept = {"val": unprefixed(row, "val_"), "reference": unprefixed(row, "reference_")}
    return {
        "best_epoch": best_epoch,
        "base_kept": best_epoch == 0,
        "epochs": len(trainer.history),
        "accepted_epochs": [row["epoch"] for row in trainer.history if row["accepted"]],
        **kept,
    }


def print_baseline(report: dict) -> None:
    baseline, limits = report["baseline"], report["limits"]
    print(f"Base checkpoint: {report['base']['checkpoint']} (epoch {report['base']['epoch']})", flush=True)
    print(f"Replayed reference samples per epoch: {report['reference']['replay_per_epoch']}", flush=True)
    for name, metrics in (("New validation", baseline["val"]), ("Reference validation", baseline["reference"])):
        print(
            f"{name} with the base weights: f1={metrics['f1']:.4f} "
            f"count_mae={metrics['count_mae']:.4f}",
            flush=True,
        )
    print(
        f"An epoch is accepted with reference f1 >= {limits['f1']:.4f} "
        f"and count_mae <= {limits['count_mae']:.4f}",
        flush=True,
    )


def print_result(result: dict) -> None:
    if result["base_kept"]:
        print(
            "No accepted epoch improved the new validation; best.pt holds the base weights.",
            flush=True,
        )
        return
    print(
        f"Fine-tuned checkpoint: epoch {result['best_epoch']}, "
        f"{len(result['accepted_epochs'])} of {result['epochs']} epochs accepted; "
        f"new validation f1={result['val']['f1']:.4f}, "
        f"reference validation f1={result['reference']['f1']:.4f}",
        flush=True,
    )


def finetune(config: dict, options: dict, name=None, selection: dict | None = None) -> Path:
    """Fine-tune the base checkpoint on the data of `config`; returns the run directory.

    The run trains `config` as given. The command line and the dashboard put the
    hyperparameters of the base run into it first (inherit_base_parameters).
    """
    options = finetune_options(options)
    config = deepcopy(config)
    unscaled = config["training"]["learning_rate"]
    # The saved config and checkpoint record the rate that was actually used.
    config["training"]["learning_rate"] *= options["learning_rate_scale"]
    artifacts, selection = open_run(config, name, selection)
    update_experiment(artifacts.directory, finetune={**options, "unscaled_learning_rate": unscaled})
    try:
        return _finetune(config, options, artifacts, selection, unscaled)
    except BaseException as error:
        record_failure(artifacts, error)
        raise


def _finetune(config: dict, options: dict, artifacts, selection: dict | None, unscaled: float) -> Path:
    training = config["training"]
    configure_reproducibility(training)
    device = choose_device(training["device"])

    base = load_base(config, options, device)
    train_dataset = build_dataset(config["data"], "train")
    val_dataset = build_dataset(config["data"], "val")
    reference = load_reference(base, with_train=options["replay_ratio"] > 0)

    model = Detector(config["model"]).to(device)
    model.load_state_dict(base["model"])

    provenance, parameter_count = record_run_inputs(
        config, artifacts, model, device, train_dataset, val_dataset
    )
    # train_fingerprint describes the new data only; replay is recorded here.
    provenance["finetune"] = {
        "base_sha256": file_sha256(options["checkpoint"]),
        "unscaled_learning_rate": unscaled,
        "reference_train_fingerprint": base["provenance"]["train_fingerprint"] if "train" in reference else None,
        "reference_val_fingerprint": base["provenance"]["val_fingerprint"],
    }
    leaks = check_leaks(train_dataset, val_dataset, reference)

    training_samples = train_dataset
    replay_count = 0
    if "train" in reference:
        training_samples = ReplayDataset(train_dataset, reference["train"], options["replay_ratio"])
        replay_count = training_samples.replay_count
    evaluation = evaluation_settings(config["evaluation"], config["data"])
    guard = ReferenceGuard(make_loader(reference["val"], training), device, evaluation, options)
    check_count_limit(options["max_count_mae_rise"], len(reference["val"]))
    trainer = build_trainer(
        config, artifacts, model, device, training_samples, val_dataset,
        provenance, selection, guard,
    )

    baseline = {"val": trainer.validate()[0], "reference": guard.set_baseline(model)}
    # The base weights are the first candidate; an interrupted run keeps them too.
    trainer.save_best_checkpoint(0, baseline["val"])
    update_experiment(artifacts.directory, best_epoch=0, best_metrics=baseline["val"])
    report = {
        "schema_version": 1,
        "options": options,
        "base": {
            "checkpoint": options["checkpoint"],
            "sha256": provenance["finetune"]["base_sha256"],
            "epoch": base["epoch"],
        },
        "reference": {
            "train": base["config"]["data"]["train"] if "train" in reference else None,
            "val": base["config"]["data"]["val"],
            "replay_per_epoch": replay_count,
            "identical_signals": leaks,
        },
        "baseline": baseline,
        "limits": guard.limits,
    }
    write_json(artifacts.directory / "finetune.json", report)
    print_baseline(report)

    directory = trainer.fit(training["epochs"], parameter_count)
    report["result"] = finetune_result(trainer, baseline)
    write_json(directory / "finetune.json", report)
    print_result(report["result"])
    return directory
