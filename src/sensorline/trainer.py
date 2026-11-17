"""Training orchestration and the epoch loop."""

import json
from pathlib import Path
from time import perf_counter

import torch

from .augmentation import AugmentedDataset, augmentation_options
from .artifacts import RunArtifacts, start_experiment, update_experiment
from .batching import forward_groups
from .checkpoint import load_checkpoint
from .config import training_option
from .data import audit_sources, build_dataset, dataset_fingerprint
from .ema import ModelEma, build_ema
from .evaluation import evaluate, evaluation_settings
from .losses import DetectionLoss
from .models import Detector, model_info
from .search_config import validate_selection
from .runtime import (
    build_optimizer,
    build_scheduler,
    choose_device,
    configure_reproducibility,
    environment_info,
    make_loader,
)


def build_test_dataset(config: dict):
    """The test split of a config, or None when it has none.

    The test split is monitored: it is evaluated after every epoch and reported, but it
    never chooses the checkpoint, the Optuna trial or any parameter.
    """
    return build_dataset(config["data"], "test") if "test" in config["data"] else None


def build_provenance(config: dict, train_dataset, val_dataset, test_dataset=None) -> dict:
    provenance = {
        "train_fingerprint": dataset_fingerprint(train_dataset),
        "val_fingerprint": dataset_fingerprint(val_dataset),
        "signal": {key: config["data"][key] for key in ("type", "sensor_axis", "scale", "offset")},
        "evaluation": config["evaluation"],
    }
    if test_dataset is not None:
        provenance["test_fingerprint"] = dataset_fingerprint(test_dataset)
    return provenance


def build_history_row(epoch: int, train_losses: dict, val_metrics: dict, test_metrics: dict | None = None) -> dict:
    row = {"epoch": epoch}
    for name, value in train_losses.items():
        row[f"train_loss_{name}"] = value
    for name, value in val_metrics.items():
        row[f"val_{name}"] = value
    for name, value in (test_metrics or {}).items():
        row[f"test_{name}"] = value
    return row


def print_run_summary(audit: dict, info: dict) -> None:
    print(json.dumps(audit, ensure_ascii=False), flush=True)
    if audit["cross_split_duplicates"]:
        print(
            "WARNING: identical signals occur in train and val; supplied split is preserved.",
            flush=True,
        )
    for key, pair in (("train_test_duplicates", "train and test"), ("val_test_duplicates", "val and test")):
        if audit.get(key):
            print(f"WARNING: identical signals occur in {pair}; the test metrics are optimistic.", flush=True)
    print(info["structure"], flush=True)
    shapes = {key: value for key, value in info.items() if key != "structure"}
    print(json.dumps(shapes), flush=True)


class Trainer:
    """Train one model and select its checkpoint using validation metrics.

    A test loader, when given, is evaluated after every epoch next to the validation
    and reported; it takes no part in choosing the checkpoint.
    """

    def __init__(
        self,
        model,
        criterion,
        optimizer,
        train_loader,
        val_loader,
        device,
        config: dict,
        artifacts: RunArtifacts,
        provenance: dict,
        selection: dict | None = None,
        scheduler=None,
        ema: ModelEma | None = None,
        guard=None,
        test_loader=None,
    ):
        self.model = model
        # Validation, checkpoints and reports use the averaged weights when EMA is on.
        self.evaluated_model = ema.model if ema is not None else model
        self.ema = ema
        self.scheduler = scheduler
        # Fine-tuning only: decides whether an epoch may become the checkpoint.
        self.guard = guard
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.config = config
        self.artifacts = artifacts
        self.provenance = provenance
        self.selection = selection

        self.gradient_clip = config["training"]["gradient_clip"]
        self.eval_config = evaluation_settings(config["evaluation"], config["data"])
        self.best_rank = None
        self.best_epoch = None
        self.history = []

    def batch_losses(self, signal, targets) -> dict:
        """Average original-length group losses before a single optimizer step."""
        combined = None
        for indices, output, group_targets, image_shape in forward_groups(
            self.model, signal, targets
        ):
            losses = self.criterion(output, group_targets, image_shape)
            if len(indices) == len(targets):
                return losses
            weight = len(indices) / len(targets)
            weighted = {name: value * weight for name, value in losses.items()}
            if combined is None:
                combined = weighted
            else:
                combined = {name: combined[name] + weighted[name] for name in combined}
        return combined

    def train_epoch(self) -> dict:
        self.model.train()
        loss_sums = {"total": 0.0, "bbox": 0.0, "objectness": 0.0}

        for signal, targets in self.train_loader:
            signal = signal.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)

            losses = self.batch_losses(signal, targets)
            if not torch.isfinite(losses["total"]):
                raise RuntimeError("Nonfinite training loss")

            losses["total"].backward()
            if self.gradient_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip
                )
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            if self.ema is not None:
                self.ema.update(self.model)

            batch_size = len(targets)
            for name in loss_sums:
                loss_sums[name] += float(losses[name].detach()) * batch_size

        sample_count = len(self.train_loader.dataset)
        return {name: total / sample_count for name, total in loss_sums.items()}

    def validate(self):
        return evaluate(
            self.evaluated_model,
            self.val_loader,
            self.device,
            self.eval_config,
            criterion=self.criterion,
        )

    def validate_test(self):
        """Metrics, frames and predictions of the test split; the model is evaluated, not changed."""
        return evaluate(
            self.evaluated_model,
            self.test_loader,
            self.device,
            self.eval_config,
            criterion=self.criterion,
        )

    def save_best_checkpoint(self, epoch: int, metrics: dict) -> None:
        rank = (metrics["count_mae"], -metrics["map50"], metrics["loss_total"])
        if self.selection is not None:
            value = metrics[self.selection["metric"]]
            if self.selection["direction"] == "maximize":
                value = -value
            rank = (value, *rank)
        if self.best_rank is not None and rank >= self.best_rank:
            return

        self.best_rank = rank
        self.best_epoch = epoch
        checkpoint = {
            "format_version": 1,
            "model": self.evaluated_model.state_dict(),
            "config": self.config,
            "epoch": epoch,
            "validation_metrics": metrics,
            "provenance": self.provenance,
        }
        torch.save(checkpoint, self.artifacts.checkpoint_path)

    def record_progress(
        self, epoch: int, metrics: dict, parameter_count: int, test_metrics: dict | None = None
    ) -> None:
        progress = {
            "best_epoch": self.best_epoch,
            "final_metrics": metrics,
            "parameters": parameter_count,
            "training_seconds": sum(
                row["train_seconds"] + row["val_seconds"] + row.get("test_seconds", 0)
                for row in self.history
            ),
        }
        if self.best_epoch == epoch:
            progress["best_metrics"] = metrics
            if test_metrics is not None:
                progress["best_test_metrics"] = test_metrics
        update_experiment(self.artifacts.directory, **progress)

    def print_epoch(
        self, epoch: int, train_losses: dict, metrics: dict, test_metrics: dict | None = None
    ) -> None:
        print(
            f"epoch={epoch} "
            f"train_loss={train_losses['total']:.4f} "
            f"val_loss={metrics['loss_total']:.4f} "
            f"val_count_mae={metrics['count_mae']:.4f} "
            f"val_mAP50={metrics['map50']:.4f} "
            f"best_epoch={self.best_epoch}",
            flush=True,
        )
        if test_metrics is not None:
            print(
                f"  test_loss={test_metrics['loss_total']:.4f} "
                f"test_count_mae={test_metrics['count_mae']:.4f} "
                f"test_mAP50={test_metrics['map50']:.4f}",
                flush=True,
            )

    def save_final_validation(self, parameter_count: int) -> None:
        checkpoint = load_checkpoint(self.artifacts.checkpoint_path, self.device)
        self.evaluated_model.load_state_dict(checkpoint["model"])
        metrics, sample_rows, predictions = self.validate()
        report = {
            "split": "val",
            "best_epoch": self.best_epoch,
            "trainable_parameters": parameter_count,
            "metrics": metrics,
            "provenance": self.provenance,
        }
        self.artifacts.save_validation(report, sample_rows, predictions)
        update_experiment(
            self.artifacts.directory, best_metrics=metrics, best_epoch=self.best_epoch
        )
        if self.test_loader is not None:
            # The weights of the best epoch are loaded already: the same ones the report above used.
            test_metrics, test_rows, _ = self.validate_test()
            self.artifacts.save_test({**report, "split": "test", "metrics": test_metrics}, test_rows)
            update_experiment(self.artifacts.directory, best_test_metrics=test_metrics)

    def fit(self, epochs: int, parameter_count: int) -> Path:
        for epoch in range(1, epochs + 1):
            started = perf_counter()
            train_losses = self.train_epoch()
            train_seconds = perf_counter() - started
            started = perf_counter()
            val_metrics, _, _ = self.validate()
            verdict = None
            if self.guard is not None:
                verdict = self.guard.check(self.evaluated_model, epoch)
            val_seconds = perf_counter() - started
            test_metrics = None
            if self.test_loader is not None:
                started = perf_counter()
                test_metrics, _, _ = self.validate_test()
                test_seconds = perf_counter() - started

            row = build_history_row(epoch, train_losses, val_metrics, test_metrics)
            row.update(
                train_seconds=train_seconds,
                val_seconds=val_seconds,
                learning_rate=self.optimizer.param_groups[0]["lr"],
            )
            if test_metrics is not None:
                row["test_seconds"] = test_seconds
            if verdict is not None:
                row.update(verdict)
            self.history.append(row)
            self.artifacts.save_history(self.history)
            # A rejected epoch stays in the history but cannot become the checkpoint.
            if verdict is None or verdict["accepted"]:
                self.save_best_checkpoint(epoch, val_metrics)
            self.record_progress(epoch, val_metrics, parameter_count, test_metrics)
            self.print_epoch(epoch, train_losses, val_metrics, test_metrics)
            print(
                f"  train={train_seconds:.2f}s val={val_seconds:.2f}s"
                + (f" test={test_seconds:.2f}s" if test_metrics is not None else ""),
                flush=True,
            )

        self.save_final_validation(parameter_count)
        update_experiment(
            self.artifacts.directory, status="completed", provenance=self.provenance
        )
        print(f"Best checkpoint: {self.artifacts.checkpoint_path}", flush=True)
        return self.artifacts.directory


def open_run(config: dict, name, selection: dict | None) -> tuple[RunArtifacts, dict | None]:
    """Create the run directory and its record, so a later failure has a place to be reported."""
    if selection is not None:
        selection = validate_selection(selection)
    artifacts = RunArtifacts(config["training"]["output_dir"])
    start_experiment(artifacts.directory, config, name)
    if selection is not None:
        update_experiment(artifacts.directory, checkpoint_selection=selection)
    return artifacts, selection


def record_failure(artifacts: RunArtifacts, error: BaseException) -> None:
    update_experiment(
        artifacts.directory,
        status="failed",
        error=f"{type(error).__name__}: {error}",
    )


def train(config: dict, name=None, selection: dict | None = None) -> Path:
    artifacts, selection = open_run(config, name, selection)
    try:
        return _train(config, artifacts, selection)
    except BaseException as error:
        record_failure(artifacts, error)
        raise


def record_run_inputs(
    config: dict, artifacts: RunArtifacts, model, device, train_dataset, val_dataset, test_dataset=None
) -> tuple[dict, int]:
    """Describe the model, data and environment of a run; returns provenance and parameter count."""
    sensors, time_steps = train_dataset[0]["signal"].shape[-2:]
    info = model_info(model, (1, 1, sensors, time_steps))
    audit = audit_sources(train_dataset, val_dataset, test_dataset)
    provenance = build_provenance(config, train_dataset, val_dataset, test_dataset)
    print_run_summary(audit, info)

    environment = environment_info(device, config["training"]["seed"])
    print(f"Device: {environment['device']} ({environment['device_name']})", flush=True)
    artifacts.save_metadata(config, info, audit, environment)
    update_experiment(
        artifacts.directory,
        parameters=info["trainable_parameters"],
        provenance=provenance,
    )
    return provenance, info["trainable_parameters"]


def build_trainer(
    config: dict,
    artifacts: RunArtifacts,
    model,
    device,
    training_samples,
    val_dataset,
    provenance: dict,
    selection: dict | None = None,
    guard=None,
    test_dataset=None,
) -> Trainer:
    """Loss, optimizer, loaders and optional schedule and EMA around a model that is ready to train."""
    training = config["training"]
    criterion = DetectionLoss(config["loss"])
    optimizer = build_optimizer(model, training)

    augmentation = augmentation_options(training.get("augmentation"))
    print("Training augmentation: " + json.dumps(augmentation), flush=True)
    if augmentation["enabled"]:
        background = -config["data"]["offset"] / config["data"]["scale"]
        training_samples = AugmentedDataset(training_samples, augmentation, background)
    train_loader = make_loader(training_samples, training, shuffle=True)
    val_loader = make_loader(val_dataset, training)
    # Its own loader and generator: the test pass cannot change the random stream of training.
    test_loader = make_loader(test_dataset, training) if test_dataset is not None else None

    # Both are None by default; training then follows the constant-rate path.
    scheduler = build_scheduler(optimizer, training, len(train_loader))
    ema = build_ema(model, training_option(training, "ema_decay"))
    print(
        f"Learning rate schedule: {training_option(training, 'lr_schedule')}, "
        f"warmup epochs: {training_option(training, 'warmup_epochs')}, "
        f"EMA decay: {training_option(training, 'ema_decay')}",
        flush=True,
    )

    return Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config,
        artifacts=artifacts,
        provenance=provenance,
        selection=selection,
        scheduler=scheduler,
        ema=ema,
        guard=guard,
        test_loader=test_loader,
    )


def _train(config: dict, artifacts: RunArtifacts, selection: dict | None = None) -> Path:
    training = config["training"]
    configure_reproducibility(training)
    device = choose_device(training["device"])

    train_dataset = build_dataset(config["data"], "train")
    val_dataset = build_dataset(config["data"], "val")
    test_dataset = build_test_dataset(config)
    model = Detector(config["model"]).to(device)

    provenance, parameter_count = record_run_inputs(
        config, artifacts, model, device, train_dataset, val_dataset, test_dataset
    )
    trainer = build_trainer(
        config, artifacts, model, device, train_dataset, val_dataset, provenance, selection,
        test_dataset=test_dataset,
    )
    return trainer.fit(training["epochs"], parameter_count)
