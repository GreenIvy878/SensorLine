"""Evaluate a saved checkpoint on a dataset and save the report; test.py and the dashboard share it.

The checkpoint owns the model, the thresholds and the loss. The caller chooses the
data (a split of a data section, or another folder), how to run it (device and the
loader settings) and where the report goes.
"""

from pathlib import Path

from .artifacts import write_json, write_rows
from .checkpoint import load_model
from .config import STORAGE_OPTIONS, declared_storage
from .data import build_dataset, dataset_fingerprint
from .evaluation import evaluate, evaluation_settings
from .losses import DetectionLoss
from .runtime import make_loader


class CountingLoader:
    """Report how many frames the evaluator has consumed."""

    def __init__(self, loader, report):
        self.loader = loader
        self.report = report

    def __iter__(self):
        done = 0
        for signal, targets in self.loader:
            yield signal, targets
            done += len(targets)
            self.report(done)


def folder_data(folder: Path, data: dict, owner="checkpoint") -> tuple[dict, str]:
    """How the images of a folder are stored, and where that description comes from.

    `data` is the data section the evaluation starts from, named by `owner`: the
    checkpoint's own, or the model YAML given to test.py. A folder whose dataset has
    a data.yaml (in the folder or at the dataset root that lists it) is read as that
    file says; any other folder is read as `data` stores its own frames.
    """
    declared = declared_storage(folder.resolve())
    if declared is None:
        return data, owner
    storage, source = declared
    return {**data, **storage}, str(source)


def evaluate_checkpoint(
    checkpoint: dict, path, device, data: dict, split: str, *, runtime: dict, folder=None,
    owner="checkpoint", labeled=True, with_loss=False, progress=None,
) -> dict:
    """Run a loaded checkpoint on `data[split]`, or on `folder` when it is given.

    `runtime` is a training section that supplies batch_size, num_workers and seed to
    the loader. `progress(done, total)` follows the evaluated frames. Returns the
    report, the per-frame rows, the predictions and the dataset they come from.
    """
    described_by = owner
    if folder is not None:
        data, described_by = folder_data(Path(folder), data, owner)
    dataset = build_dataset(data, split, root=folder, require_labels=labeled)
    loader = make_loader(dataset, runtime)
    if progress is not None:
        progress(0, len(dataset))
        loader = CountingLoader(loader, lambda done: progress(done, len(dataset)))
    saved = checkpoint["config"]
    model = load_model(checkpoint, device)
    criterion = DetectionLoss(saved["loss"]) if with_loss else None
    metrics, rows, predictions = evaluate(
        model, loader, device, evaluation_settings(saved["evaluation"], data),
        criterion=criterion, labeled=labeled,
    )
    report = {
        "split": split,
        "data_root": str(Path(dataset.root).resolve()),
        "labeled": labeled,
        "checkpoint": str(Path(path).resolve()),
        "checkpoint_epoch": checkpoint["epoch"],
        "metrics": metrics,
        "dataset_fingerprint": dataset_fingerprint(dataset),
        "evaluation": saved["evaluation"],
        "preprocessing": {**{key: data[key] for key in STORAGE_OPTIONS}, "source": described_by},
    }
    return {"report": report, "rows": rows, "predictions": predictions, "dataset": dataset}


def save_report(directory: Path, result: dict) -> None:
    """The files of one evaluation: predictions, per-frame rows and the report."""
    write_json(directory / "predictions.json", result["predictions"])
    write_rows(directory / "samples.csv", result["rows"])
    write_json(directory / "metrics.json", result["report"])
