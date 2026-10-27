"""The run config of tracking: defaults from configs/tracking/default.yaml, a user file and CLI overrides.

The default file names every option; a user file and `section.key=value` overrides may
change only options that exist there, so a typo is an error rather than an ignored line.
"""

import copy
from pathlib import Path

import yaml

from ..validation import finite_number, require_integer, require_mapping

AUTO = "auto"
MATCH_MODES = ("overlap_clipped", "plain_iou", "iomin")
TAIL_MODES = ("align_end", "pad_background")
COUNT_ON = ("close", "confirm")
INPUT_KINDS = ("pixels", "raw_adc")
ORIENTATIONS = (AUTO, "rows_are_polls", "rows_are_channels")
# Options holding paths, resolved against the file that sets them.
PATH_OPTIONS = {("output", "runs_root")}


def read_yaml(path: Path) -> dict:
    values = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    require_mapping(values, str(path))
    return values


def resolve_file_paths(values: dict, directory: Path) -> None:
    for section, key in PATH_OPTIONS:
        value = values.get(section, {}).get(key)
        if isinstance(value, str):
            values[section][key] = str((directory / value).resolve())


def merge(base: dict, changes: dict, source: str) -> None:
    for section, options in changes.items():
        if section not in base:
            raise ValueError(f"{source}: unknown section {section!r}; known: {sorted(base)}")
        require_mapping(options, f"{source}: {section}")
        unknown = set(options) - set(base[section])
        if unknown:
            raise ValueError(f"{source}: unknown options {section}.{sorted(unknown)}; known: {sorted(base[section])}")
        base[section].update(options)


def parse_override(text: str) -> tuple[str, str, object]:
    """`section.key=value`, with the value read as YAML (numbers, null, lists)."""
    name, separator, value = text.partition("=")
    section, dot, key = name.strip().partition(".")
    if not separator or not dot or not key:
        raise ValueError(f"Override {text!r} must look like section.key=value")
    return section, key, yaml.safe_load(value)


def load_settings(default_path, user_path=None, overrides=()) -> dict:
    default_path = Path(default_path).resolve()
    settings = read_yaml(default_path)
    resolve_file_paths(settings, default_path.parent)
    if user_path is not None:
        user_path = Path(user_path).resolve()
        user = read_yaml(user_path)
        resolve_file_paths(user, user_path.parent)
        merge(settings, user, str(user_path))
    for text in overrides:
        section, key, value = parse_override(text)
        changes = {section: {key: value}}
        if (section, key) in PATH_OPTIONS and isinstance(value, str):
            changes[section][key] = str(Path(value).resolve())
        merge(settings, changes, f"override {text!r}")
    validate_settings(settings)
    return settings


def unit_interval(value, name: str) -> float:
    if not 0 < finite_number(value, name) <= 1:
        raise ValueError(f"{name} must be in (0, 1]")
    return value


def auto_or(value, check) -> None:
    if value != AUTO:
        check(value)


def validate_settings(settings: dict) -> None:
    model, data, windows, tracker = (settings[key] for key in ("model", "data", "windows", "tracker"))
    if model["device"] not in (AUTO, "cpu", "cuda"):
        raise ValueError("model.device must be auto, cpu or cuda")
    require_integer(model["batch_windows"], "model.batch_windows", minimum=1)
    auto_or(model["score_thr"], lambda v: unit_interval(v, "model.score_thr"))
    auto_or(model["nms_iou"], lambda v: unit_interval(v, "model.nms_iou"))
    unit_interval(model["nms_iou_max"], "model.nms_iou_max")

    if data["input_kind"] not in INPUT_KINDS:
        raise ValueError(f"data.input_kind must be one of {INPUT_KINDS}")
    if data["orientation"] not in ORIENTATIONS:
        raise ValueError(f"data.orientation must be one of {ORIENTATIONS}")
    if data["delimiter"] != AUTO and (not isinstance(data["delimiter"], str) or len(data["delimiter"]) != 1):
        raise ValueError("data.delimiter must be auto or one character")
    if data["header"] not in (AUTO, True, False):
        raise ValueError("data.header must be auto, true or false")
    if finite_number(data["gap_factor"], "data.gap_factor") <= 1:
        raise ValueError("data.gap_factor must be greater than 1")
    require_integer(data["chunk_rows"], "data.chunk_rows", minimum=1)
    require_integer(data["pool"], "data.pool", minimum=1)
    if data["time_unit_s"] is not None and finite_number(data["time_unit_s"], "data.time_unit_s") <= 0:
        raise ValueError("data.time_unit_s must be positive or null")
    if data["sample_rate_hz"] != AUTO and finite_number(data["sample_rate_hz"], "data.sample_rate_hz") <= 0:
        raise ValueError("data.sample_rate_hz must be positive")

    auto_or(windows["stride"], lambda v: require_integer(v, "windows.stride", minimum=1))
    unit_interval(windows["lmax_quantile"], "windows.lmax_quantile")
    if finite_number(windows["margin"], "windows.margin") < 0:
        raise ValueError("windows.margin must be nonnegative")
    if not 0 <= finite_number(windows["empty_peak"], "windows.empty_peak") < 1:
        raise ValueError("windows.empty_peak must be in [0, 1)")
    if windows["tail_mode"] not in TAIL_MODES:
        raise ValueError(f"windows.tail_mode must be one of {TAIL_MODES}")

    unit_interval(tracker["match_iou"], "tracker.match_iou")
    unit_interval(tracker["dedup_iou"], "tracker.dedup_iou")
    if tracker["match_mode"] not in MATCH_MODES:
        raise ValueError(f"tracker.match_mode must be one of {MATCH_MODES}")
    require_integer(tracker["min_hits"], "tracker.min_hits", minimum=1)
    auto_or(tracker["strong_score"], lambda v: unit_interval(v, "tracker.strong_score"))
    finite_number(tracker["strong_score_margin"], "tracker.strong_score_margin")
    if finite_number(tracker["edge_tol"], "tracker.edge_tol") < 0:
        raise ValueError("tracker.edge_tol must be nonnegative")
    if tracker["count_on"] not in COUNT_ON:
        raise ValueError(f"tracker.count_on must be one of {COUNT_ON}")


def resolve_model_settings(settings: dict, passport: dict) -> dict:
    """Replace the model and tracker `auto` values that the passport decides; returns a copy."""
    resolved = copy.deepcopy(settings)
    model, tracker = resolved["model"], resolved["tracker"]
    if model["score_thr"] == AUTO:
        model["score_thr"] = passport["score_thr_default"]
    if model["nms_iou"] == AUTO:
        model["nms_iou"] = passport["postprocess"]["nms_iou"]
    if model["nms_iou"] > model["nms_iou_max"]:
        raise ValueError(
            f"NMS IoU {model['nms_iou']} exceeds model.nms_iou_max {model['nms_iou_max']}: "
            "neighbouring objects overlap by IoU 0.1-0.2, so two touching objects would suppress each other; "
            "set model.nms_iou explicitly"
        )
    if tracker["strong_score"] == AUTO:
        tracker["strong_score"] = min(model["score_thr"] + tracker["strong_score_margin"], 1.0)
    return resolved
