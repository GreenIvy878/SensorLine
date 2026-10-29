"""Read YAML, apply defaults, resolve paths and validate user-facing options."""

from pathlib import Path

import yaml

from .augmentation import augmentation_options
from .models.builder import validate_model_config
from .validation import finite_number, require_integer, require_mapping


REQUIRED_SECTIONS = {"data", "model", "loss", "training", "evaluation"}

# Values that options accept, in the order the dashboard offers them.
DEVICES = ("auto", "cuda", "cpu")
OPTIMIZERS = ("adamw", "adam", "sgd")
LR_SCHEDULES = ("constant", "cosine")
BOX_LOSSES = ("ciou", "diou", "giou", "iou")
OBJECTNESS_LOSSES = ("bce", "focal")
DEFAULT_SENSORS = 40

# Every option of the data, training, evaluation and loss sections. The model section
# is described by models.builder, training.augmentation by augmentation.
#   required  the YAML writes it. Thresholds and loss decide what a run measures and
#             learns, so they have no silent default.
#   optional  may be left out, and then nothing is assumed (no test split, no augmentation).
#   defaults  filled in when a YAML is loaded, and saved with the run.
#   unsaved   assumed when absent but never filled in, so that configs, checkpoints and
#             Optuna manifests saved before the option existed load and compare unchanged.
#   tunable   what an Optuna search and --best-params may change.
SECTIONS = {
    "data": {
        "required": ("type", "train", "val"),
        "optional": ("test",),
        "defaults": {"sensor_axis": "x", "scale": 255.0, "offset": 0.0},
        "unsaved": {"sensors": DEFAULT_SENSORS},
        "tunable": (),
    },
    "training": {
        "required": ("batch_size", "epochs", "optimizer", "learning_rate", "output_dir"),
        "optional": ("augmentation",),
        "defaults": {
            "weight_decay": 0.0001, "momentum": 0.9, "device": "auto", "seed": 17,
            "num_workers": 0, "gradient_clip": 5.0, "cpu_threads": 4,
        },
        "unsaved": {"lr_schedule": "constant", "warmup_epochs": 0, "ema_decay": 0.0},
        "tunable": (
            "optimizer", "learning_rate", "weight_decay", "momentum", "batch_size", "epochs",
            "gradient_clip", "lr_schedule", "warmup_epochs", "ema_decay",
        ),
    },
    "evaluation": {
        "required": (
            "confidence", "ap_min_confidence", "nms_iou", "match_iou", "max_detections", "pre_nms_topk",
        ),
        "optional": (),
        "defaults": {},
        "unsaved": {"border_ignore": 0},
        "tunable": ("confidence", "nms_iou", "max_detections", "pre_nms_topk"),
    },
    # A model YAML writes every loss option that applies to it (require_written_loss);
    # these values only complete older saved configs and the partial mappings of tests.
    "loss": {
        "required": (),
        "optional": (),
        "defaults": {},
        "unsaved": {
            "bbox": "ciou", "bbox_weight": 5.0, "objectness": "bce", "objectness_weight": 1.0,
            "negative_weight": 1.0, "hard_negative_ratio": 0, "focal_alpha": 0.25, "focal_gamma": 2.0,
            "border_ignore": 0,
        },
        "tunable": (
            "bbox", "bbox_weight", "objectness", "objectness_weight", "negative_weight",
            "hard_negative_ratio", "focal_alpha", "focal_gamma",
        ),
    },
}
FOCAL_OPTIONS = {"focal_alpha", "focal_gamma"}
# Loss options that an old config lacks and a model YAML may leave out: they are off by default.
OFF_BY_DEFAULT_LOSS_OPTIONS = {"border_ignore"}
PATH_OPTIONS = {
    "data": ("train", "val", "test"),
    "training": ("output_dir",),
}


def allowed_options(section: str) -> set:
    table = SECTIONS[section]
    return {*table["required"], *table["optional"], *table["defaults"], *table["unsaved"]}


def with_defaults(section: str, values: dict) -> dict:
    """The options of a section, completed with the defaults that configs do not save."""
    return {**SECTIONS[section]["unsaved"], **values}


def training_option(training: dict, key: str):
    return training.get(key, SECTIONS["training"]["unsaved"][key])


def sensor_count(data: dict) -> int:
    return data.get("sensors", DEFAULT_SENSORS)


def validate_sections(config: dict) -> None:
    """Section names and option names only; values are checked afterwards."""
    require_mapping(config, "config")
    if set(config) != REQUIRED_SECTIONS:
        raise ValueError(f"YAML sections must be {sorted(REQUIRED_SECTIONS)}")
    for section in REQUIRED_SECTIONS:
        require_mapping(config[section], section)
    # Unknown loss options are reported by validate_loss_config, which also checks a loss alone.
    for section in ("data", "training", "evaluation"):
        allowed = allowed_options(section)
        if set(config[section]) - allowed:
            raise ValueError(f"Invalid options in {section}; allowed: {sorted(allowed)}")


def apply_defaults(config: dict) -> None:
    for section, table in SECTIONS.items():
        if table["defaults"]:
            config[section] = {**table["defaults"], **config[section]}
    training = config["training"]
    if "augmentation" in training:
        require_mapping(training["augmentation"], "training.augmentation")
        training["augmentation"] = augmentation_options(training["augmentation"])


def validate_required_options(config: dict) -> None:
    for section, table in SECTIONS.items():
        missing = set(table["defaults"]) - set(config[section])
        if missing:
            raise ValueError(f"Missing {section} options: {sorted(missing)}")
    for section, table in SECTIONS.items():
        for key in table["required"]:
            if key not in config[section]:
                raise ValueError(f"Missing {section}.{key}")


def require_written_loss(loss: dict) -> None:
    """A model YAML writes every loss option that applies to it.

    The unsaved loss defaults complete older saved configs, and they differ from the
    values of the series; they must not reach a new run through an omitted line.
    """
    options = set(SECTIONS["loss"]["unsaved"]) - OFF_BY_DEFAULT_LOSS_OPTIONS
    required = options if loss.get("objectness") == "focal" else options - FOCAL_OPTIONS
    missing = required - set(loss)
    if missing:
        raise ValueError(f"Missing loss options: {sorted(missing)}; a model YAML sets them explicitly")


def resolve_paths(config: dict, directory: Path) -> None:
    for section, keys in PATH_OPTIONS.items():
        for key in keys:
            if key in config[section]:
                config[section][key] = str((directory / config[section][key]).resolve())


def validate_data(data: dict) -> None:
    if data["type"] != "yolo_images":
        raise ValueError("data.type must be yolo_images")
    if data["sensor_axis"] not in ("x", "y"):
        raise ValueError("data.sensor_axis must be x or y")
    if finite_number(data["scale"], "data.scale") <= 0:
        raise ValueError("data.scale must be positive")
    finite_number(data["offset"], "data.offset")
    require_integer(sensor_count(data), "data.sensors", minimum=1)
    for key in ("train", "val", "test"):
        if key in data and (not isinstance(data[key], str) or not data[key].strip()):
            raise ValueError(f"data.{key} must be a nonempty path string")


def validate_training(training: dict) -> None:
    if "augmentation" in training:
        require_mapping(training["augmentation"], "training.augmentation")
        augmentation_options(training["augmentation"])
    for key in ("epochs", "batch_size", "cpu_threads"):
        require_integer(training[key], f"training.{key}", minimum=1)
    require_integer(training["num_workers"], "training.num_workers")
    require_integer(training["seed"], "training.seed", maximum=2**32 - 1)
    if training["optimizer"] not in OPTIMIZERS:
        raise ValueError("training.optimizer must be adam, adamw or sgd")
    if finite_number(training["learning_rate"], "training.learning_rate") <= 0:
        raise ValueError("training.learning_rate must be positive")
    for key in ("weight_decay", "gradient_clip"):
        if finite_number(training[key], f"training.{key}") < 0:
            raise ValueError(f"training.{key} must be nonnegative")
    if not 0 <= finite_number(training["momentum"], "training.momentum") < 1:
        raise ValueError("training.momentum must be in [0, 1)")
    if training_option(training, "lr_schedule") not in LR_SCHEDULES:
        raise ValueError("training.lr_schedule must be constant or cosine")
    warmup_epochs = require_integer(training_option(training, "warmup_epochs"), "training.warmup_epochs")
    if warmup_epochs and warmup_epochs >= training["epochs"]:
        # The rate would still be rising when training ends.
        raise ValueError("training.warmup_epochs must be less than training.epochs")
    ema_decay = finite_number(training_option(training, "ema_decay"), "training.ema_decay")
    if not 0 <= ema_decay < 1:
        raise ValueError("training.ema_decay must be in [0, 1); 0 disables averaging")
    if training["device"] not in DEVICES:
        raise ValueError("training.device must be auto, cpu or cuda")
    if not isinstance(training["output_dir"], str) or not training["output_dir"].strip():
        raise ValueError("training.output_dir must be a nonempty path string")


def validate_border_ignore(value, name: str) -> None:
    if finite_number(value, name) < 0:
        raise ValueError(f"{name} must be nonnegative; 0 disables it")


def validate_evaluation(evaluation: dict) -> None:
    for key in ("confidence", "ap_min_confidence", "nms_iou", "match_iou"):
        if not 0 < finite_number(evaluation[key], f"evaluation.{key}") <= 1:
            raise ValueError(f"evaluation.{key} must be in (0, 1]")
    if evaluation["ap_min_confidence"] > evaluation["confidence"]:
        raise ValueError("ap_min_confidence must not exceed confidence")
    for key in ("max_detections", "pre_nms_topk"):
        require_integer(evaluation[key], f"evaluation.{key}", minimum=1)
    if evaluation["pre_nms_topk"] < evaluation["max_detections"]:
        raise ValueError("pre_nms_topk must be >= max_detections")
    validate_border_ignore(with_defaults("evaluation", evaluation)["border_ignore"], "evaluation.border_ignore")


def validate_loss_config(loss: dict) -> None:
    require_mapping(loss, "loss")
    if set(loss) - allowed_options("loss"):
        raise ValueError("Unknown loss options")
    options = with_defaults("loss", loss)
    if options["bbox"] not in BOX_LOSSES:
        raise ValueError("loss.bbox must be iou, giou, diou or ciou")
    if options["objectness"] not in OBJECTNESS_LOSSES:
        raise ValueError("loss.objectness must be bce or focal")
    if options["objectness"] != "focal" and FOCAL_OPTIONS & set(loss):
        raise ValueError("focal_alpha/focal_gamma require objectness: focal")
    alpha = finite_number(options["focal_alpha"], "loss.focal_alpha")
    if alpha != -1 and not 0 <= alpha <= 1:
        raise ValueError("loss.focal_alpha must be in [0, 1] or -1 to disable balancing")
    if finite_number(options["focal_gamma"], "loss.focal_gamma") < 0:
        raise ValueError("loss.focal_gamma must be nonnegative")
    require_integer(options["hard_negative_ratio"], "loss.hard_negative_ratio")
    validate_border_ignore(options["border_ignore"], "loss.border_ignore")
    for key in ("bbox_weight", "objectness_weight", "negative_weight"):
        if finite_number(options[key], f"loss.{key}") <= 0:
            raise ValueError(f"loss.{key} must be positive")


def validate_config(config: dict) -> None:
    validate_sections(config)
    validate_required_options(config)
    validate_data(config["data"])
    validate_training(config["training"])
    validate_evaluation(config["evaluation"])
    validate_loss_config(config["loss"])
    validate_model_config(config["model"])


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        # A syntax error is a bad config like any other; callers handle ValueError.
        raise ValueError(f"{path}: {error}") from error
    validate_sections(config)
    require_written_loss(config["loss"])
    apply_defaults(config)
    validate_config(config)
    resolve_paths(config, path.parent)
    check_declared_storage(config["data"])
    return config


# A dataset may describe its own storage in this file at its root, in the same form as
# the `data` section of a model YAML. Storage may
# differ between datasets: a float32 TIFF already in signal units has scale 1,
# an 8-bit PNG has scale 255.
DATA_DESCRIPTION = "data.yaml"
STORAGE_OPTIONS = ("sensor_axis", "scale", "offset")


def read_data_description(path: Path) -> dict | None:
    """The `data` section of a data.yaml; None for a file of another tool, such as a YOLO export."""
    try:
        described = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"{path}: {error}") from error
    if not isinstance(described, dict) or not isinstance(described.get("data"), dict):
        return None
    data = described["data"]
    allowed = allowed_options("data")
    if set(data) - allowed:
        raise ValueError(f"{path}: invalid data options; allowed: {sorted(allowed)}")
    if "type" not in data:
        raise ValueError(f"{path}: missing data.type")
    data = {**SECTIONS["data"]["defaults"], **data}
    try:
        validate_data(data)
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error
    for key in PATH_OPTIONS["data"]:
        if key in data:
            data[key] = str((path.parent / data[key]).resolve())
    return data


def load_data_description(path: str | Path) -> dict:
    """The `data` section of a dataset, given its folder or its data.yaml; replaces the one of a model YAML."""
    path = Path(path).resolve()
    if path.is_dir():
        path = path / DATA_DESCRIPTION
    if not path.is_file():
        raise ValueError(
            f"No dataset description: {path}. A dataset names its train and val folders "
            f"and its pixel scale in a {DATA_DESCRIPTION}"
        )
    data = read_data_description(path)
    if data is None:
        raise ValueError(f"{path} has no data section")
    for key in SECTIONS["data"]["required"]:
        if key not in data:
            raise ValueError(f"{path}: missing data.{key}")
    return data


def declared_storage(folder: str | Path) -> tuple[dict, Path] | None:
    """Storage that a split folder declares: its own data.yaml, or the one of its dataset root."""
    folder = Path(folder).resolve()
    for path in (folder / DATA_DESCRIPTION, folder.parent / DATA_DESCRIPTION):
        if not path.is_file():
            continue
        described = read_data_description(path)
        if described is None:
            continue
        # A dataset root describes its splits; an unrelated neighbour folder is not one of them.
        splits = [Path(described[key]) for key in PATH_OPTIONS["data"] if key in described]
        if path.parent == folder or folder in splits:
            return {key: described[key] for key in STORAGE_OPTIONS}, path
    return None


def check_declared_storage(data: dict) -> None:
    """Reading a dataset with another scale than it declares gives the model a wrong signal."""
    for split in PATH_OPTIONS["data"]:
        declared = declared_storage(data[split]) if split in data else None
        if declared is None:
            continue
        storage, source = declared
        for key in STORAGE_OPTIONS:
            if data[key] != storage[key]:
                raise ValueError(
                    f"data.{key} is {data[key]!r}, but data.{split} is stored with "
                    f"{key}: {storage[key]!r} according to {source}"
                )


def storage_is_declared(data: dict) -> bool:
    """Whether the training and validation data both come with their own description."""
    return all(declared_storage(data[split]) is not None for split in ("train", "val"))
