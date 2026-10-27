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


def build_provenance(config: dict, train_dataset, val_dataset) -> dict:
    return {
        "train_fingerprint": dataset_fingerprint(train_dataset),
        "val_fingerprint": dataset_fingerprint(val_dataset),
        "signal": {key: config["data"][key] for key in ("type", "sensor_axis", "scale", "offset")},
        "evaluation": config["evaluation"],
    }


def build_history_row(epoch: int, train_losses: dict, val_metrics: dict) -> dict:
    row = {"epoch": epoch}
    for name, value in train_losses.items():
        row[f"train_loss_{name}"] = value
    for name, value in val_metrics.items():
        row[f"val_{name}"] = value
    return row


def print_run_summary(audit: dict, info: dict) -> None:
    print(json.dumps(audit, ensure_ascii=False), flush=True)
    if audit["cross_split_duplicates"]:
        print(
            "WARNING: identical signals occur in train and val; supplied split is preserved.",
            flush=True,
        )
    print(info["structure"], flush=True)
    shapes = {key: value for key, value in info.items() if key != "structure"}
    print(json.dumps(shapes), flush=True)


class Trainer:
    """Train one model and select its checkpoint using validation metrics."""

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

    def record_progress(self, epoch: int, metrics: dict, parameter_count: int) -> None:
        progress = {
            "best_epoch": self.best_epoch,
            "final_metrics": metrics,
            "parameters": parameter_count,
            "training_seconds": sum(
                row["train_seconds"] + row["val_seconds"] for row in self.history
            ),
        }
        if self.best_epoch == epoch:
            progress["best_metrics"] = metrics
        update_experiment(self.artifacts.directory, **progress)

    def print_epoch(self, epoch: int, train_losses: dict, metrics: dict) -> None:
        print(
            f"epoch={epoch} "
            f"train_loss={train_losses['total']:.4f} "
            f"val_loss={metrics['loss_total']:.4f} "
            f"val_count_mae={metrics['count_mae']:.4f} "
            f"val_mAP50={metrics['map50']:.4f} "
            f"best_epoch={self.best_epoch}",
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

            row = build_history_row(epoch, train_losses, val_metrics)
            row.update(
                train_seconds=train_seconds,
                val_seconds=val_seconds,
                learning_rate=self.optimizer.param_groups[0]["lr"],
            )
            if verdict is not None:
                row.update(verdict)
            self.history.append(row)
            self.artifacts.save_history(self.history)
            # A rejected epoch stays in the history but cannot become the checkpoint.
            if verdict is None or verdict["accepted"]:
                self.save_best_checkpoint(epoch, val_metrics)
            self.record_progress(epoch, val_metrics, parameter_count)
            self.print_epoch(epoch, train_losses, val_metrics)
            print(
                f"  train={train_seconds:.2f}s val={val_seconds:.2f}s",
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
    config: dict, artifacts: RunArtifacts, model, device, train_dataset, val_dataset
) -> tuple[dict, int]:
    """Describe the model, data and environment of a run; returns provenance and parameter count."""
    sensors, time_steps = train_dataset[0]["signal"].shape[-2:]
    info = model_info(model, (1, 1, sensors, time_steps))
    audit = audit_sources(train_dataset, val_dataset)
    provenance = build_provenance(config, train_dataset, val_dataset)
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
    )


def _train(config: dict, artifacts: RunArtifacts, selection: dict | None = None) -> Path:
    training = config["training"]
    configure_reproducibility(training)
    device = choose_device(training["device"])

    train_dataset = build_dataset(config["data"], "train")
    val_dataset = build_dataset(config["data"], "val")
    model = Detector(config["model"]).to(device)

    provenance, parameter_count = record_run_inputs(
        config, artifacts, model, device, train_dataset, val_dataset
    )
    trainer = build_trainer(
        config, artifacts, model, device, train_dataset, val_dataset, provenance, selection
    )
    return trainer.fit(training["epochs"], parameter_count)
