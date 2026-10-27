"""Read-only dataset diagnostics; independent of model training and rendering."""

from .dataset.runner import analyze_dataset, load_or_run

__all__ = ["analyze_dataset", "load_or_run"]
