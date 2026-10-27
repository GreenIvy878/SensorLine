"""Validate search JSON and apply saved parameter values without importing Optuna."""

from copy import deepcopy
import json
from pathlib import Path

from .augmentation import TUNABLE_AUGMENTATION, augmentation_options
from .config import SECTIONS, validate_config
from .validation import finite_number, require_integer, require_mapping


VALIDATION_METRICS = {
    "samples", "objects", "tp", "fp", "fn", "precision", "recall", "f1",
    "map50", "map75", "map50_95", "count_mae", "count_rmse", "count_bias",
    "exact_count_accuracy", "loss_total", "loss_bbox", "loss_objectness",
}
TUNABLE_SECTIONS = {section: set(table["tunable"]) for section, table in SECTIONS.items()}


def read_json_object(path: str | Path) -> dict:
    """A JSON object written by a person (a search or saved parameters): no NaN or Infinity."""
    def reject_constant(value):
        raise ValueError(f"JSON contains nonfinite number: {value}")

    value = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject_constant)
    require_mapping(value, "JSON root")
    return value


def validate_selection(selection: dict, allow_target=False) -> dict:
    require_mapping(selection, "objective")
    allowed = {"metric", "direction"}
    if allow_target:
        allowed.add("target")
    if set(selection) - allowed or not {"metric", "direction"} <= set(selection):
        raise ValueError(f"objective requires metric and direction; allowed: {sorted(allowed)}")
    if not isinstance(selection["metric"], str) or selection["metric"] not in VALIDATION_METRICS:
        raise ValueError(f"objective.metric must be a validation metric: {sorted(VALIDATION_METRICS)}")
    if selection["direction"] not in ("minimize", "maximize"):
        raise ValueError("objective.direction must be minimize or maximize")
    if "target" in selection:
        finite_number(selection["target"], "objective.target")
    return dict(selection)


def selection_from_objective(objective: dict) -> dict:
    validate_selection(objective, allow_target=True)
    return {key: objective[key] for key in ("metric", "direction")}


def is_augmentation_path(parts: list[str]) -> bool:
    return len(parts) == 3 and parts[:2] == ["training", "augmentation"]


def parameter_parts(path: str) -> list[str]:
    """Split a tunable path: section.option or training.augmentation.option."""
    if not isinstance(path, str) or not path or any(not part for part in path.split(".")):
        raise ValueError("Parameter paths must be nonempty dotted paths")
    parts = path.split(".")
    if is_augmentation_path(parts) and parts[2] in TUNABLE_AUGMENTATION:
        return parts
    if len(parts) == 2 and parts[1] in TUNABLE_SECTIONS.get(parts[0], ()):
        return parts
    if parts[0] == "model":
        raise ValueError(
            f"Model architecture is fixed by YAML; Optuna/best-params cannot change {path}"
        )
    raise ValueError(f"Parameter path is not tunable: {path}")


def validate_parameter_paths(parameters: dict) -> None:
    require_mapping(parameters, "parameters")
    if not parameters:
        raise ValueError("parameters must not be empty")
    for path in parameters:
        parameter_parts(path)


def path_parent(config: dict, parts: list[str]) -> dict:
    """Mapping that holds the parameter; augmentation keys and unsaved options may be absent."""
    parent = config
    for part in parts[:-1]:
        if not isinstance(parent, dict) or part not in parent:
            raise ValueError(f"Parameter path has no existing parent: {'.'.join(parts)}")
        parent = parent[part]
    optional = is_augmentation_path(parts) or parts[-1] in SECTIONS[parts[0]]["unsaved"]
    if parts[-1] not in parent and not optional:
        raise ValueError(f"Unknown parameter key: {'.'.join(parts)}")
    return parent


def apply_parameters(config: dict, parameters: dict) -> dict:
    """Return an independent validated config; never mutate the base YAML values."""
    result = deepcopy(config)
    for path, value in parameters.items():
        parts = parameter_parts(path)
        path_parent(result, parts)[parts[-1]] = deepcopy(value)
    validate_config(result)
    return result


def validate_augmentation_values(path, values):
    if path.startswith("training.augmentation."):
        key = parameter_parts(path)[-1]
        for value in values:
            augmentation_options({key: value})


def validate_distribution(path: str, spec: dict) -> None:
    require_mapping(spec, f"parameters.{path}")
    kind = spec.get("type")
    if kind == "categorical":
        if set(spec) != {"type", "choices"}:
            raise ValueError(f"{path}: categorical accepts type and choices only")
        choices = spec["choices"]
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"{path}: choices must be a nonempty list")
        # Python equality also rejects ambiguous choices such as 1 and true.
        if any(choice == previous for index, choice in enumerate(choices) for previous in choices[:index]):
            raise ValueError(f"{path}: choices must be unique")
        validate_augmentation_values(path, choices)
        return
    if kind not in ("int", "float"):
        raise ValueError(f"{path}: type must be int, float or categorical")
    if set(spec) - {"type", "low", "high", "log", "step"} or not {"low", "high"} <= set(spec):
        raise ValueError(f"{path}: numeric distributions require low/high and accept log/step")
    if kind == "int":
        for key in ("low", "high"):
            if not isinstance(spec[key], int) or isinstance(spec[key], bool):
                raise ValueError(f"{path}.{key} must be an integer")
    low = finite_number(spec["low"], f"{path}.low")
    high = finite_number(spec["high"], f"{path}.high")
    if low > high:
        raise ValueError(f"{path}: low must not exceed high")
    log = spec.get("log", False)
    if not isinstance(log, bool):
        raise ValueError(f"{path}.log must be boolean")
    if log and low <= 0:
        raise ValueError(f"{path}: log requires low > 0")
    if "step" in spec:
        step = finite_number(spec["step"], f"{path}.step")
        if kind == "int":
            require_integer(spec["step"], f"{path}.step", minimum=1)
        elif step <= 0:
            raise ValueError(f"{path}.step must be positive")
        if log and (kind == "float" or step != 1):
            raise ValueError(f"{path}: log is incompatible with this step")

    validate_augmentation_values(path, (spec["low"], spec["high"]))


def load_search_config(path: str | Path, base_config: dict | None = None) -> dict:
    path = Path(path).resolve()
    search = read_json_object(path)
    required = {"schema_version", "study_name", "output_dir", "n_trials", "sampler_seed", "objective", "parameters"}
    if set(search) - (required | {"timeout"}) or not required <= set(search):
        raise ValueError(f"Search JSON requires {sorted(required)}; optional: timeout")
    require_integer(search["schema_version"], "schema_version", minimum=1, maximum=1)
    for key in ("study_name", "output_dir"):
        if not isinstance(search[key], str) or not search[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    require_integer(search["n_trials"], "n_trials", minimum=1)
    require_integer(search["sampler_seed"], "sampler_seed", maximum=2**32 - 1)
    if "timeout" in search and finite_number(search["timeout"], "timeout") <= 0:
        raise ValueError("timeout must be positive seconds")
    validate_selection(search["objective"], allow_target=True)
    validate_parameter_paths(search["parameters"])
    for parameter, spec in search["parameters"].items():
        validate_distribution(parameter, spec)
        if base_config is not None:
            path_parent(base_config, parameter_parts(parameter))
    search["output_dir"] = str((path.parent / search["output_dir"]).resolve())
    return search


def load_best_parameters(path: str | Path, base_config: dict) -> tuple[dict, dict]:
    saved = read_json_object(path)
    require_integer(saved.get("schema_version"), "schema_version", minimum=1, maximum=1)
    if "objective" not in saved or "params" not in saved:
        raise ValueError("Best parameter JSON requires objective and params")
    selection = selection_from_objective(saved["objective"])
    return apply_parameters(base_config, saved["params"]), selection
