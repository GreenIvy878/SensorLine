"""Sequential Optuna search using the existing train/validation artifact pipeline."""

from copy import deepcopy
from pathlib import Path

from .artifacts import write_json
from .augmentation import augmentation_options
from .data import build_dataset, dataset_fingerprint
from .search_config import apply_parameters, read_json_object, selection_from_objective
from .trainer import train


RUNTIME_TRAINING_KEYS = {"output_dir", "device", "num_workers", "cpu_threads"}


def compatible_base_config(config: dict) -> dict:
    result = deepcopy(config)
    for key in RUNTIME_TRAINING_KEYS:
        result["training"].pop(key, None)
    if "augmentation" in result["training"]:
        # Missing new keys mean disabled transforms, matching old study manifests.
        result["training"]["augmentation"] = augmentation_options(result["training"]["augmentation"])
    # Independent test is never loaded or used by a search.
    result["data"].pop("test", None)
    return result


def search_definition(search: dict) -> dict:
    result = {key: deepcopy(value) for key, value in search.items()
              if key not in ("output_dir", "n_trials", "timeout")}
    result["objective"].pop("target", None)
    return result


def data_fingerprints(config: dict) -> dict:
    return {split: dataset_fingerprint(build_dataset(config["data"], split))
            for split in ("train", "val")}


def prepare_study_directory(config: dict, search: dict, resume: bool) -> tuple[Path, dict]:
    directory = Path(search["output_dir"]).resolve()
    occupied = directory.exists() and any(directory.iterdir())
    if occupied and not resume:
        raise ValueError(f"Study output is not empty; use --resume: {directory}")
    if resume and not (directory / "study.db").is_file():
        raise ValueError("--resume requires an existing study.db")

    fingerprints = data_fingerprints(config)
    manifest = {
        "schema_version": 1,
        "study_name": search["study_name"],
        "base_config": deepcopy(config),
        "search_definition": search_definition(search),
        "data_fingerprints": fingerprints,
    }
    if resume:
        stored = read_json_object(directory / "study_manifest.json")
        if stored.get("schema_version") != 1:
            raise ValueError("Unsupported study manifest schema_version")
        if stored.get("search_definition") != manifest["search_definition"]:
            raise ValueError("Cannot resume: search parameters, sampler or objective changed")
        if compatible_base_config(stored["base_config"]) != compatible_base_config(config):
            raise ValueError("Cannot resume: base model, training or data configuration changed")
        if stored.get("data_fingerprints") != fingerprints:
            raise ValueError("Cannot resume: train/val data fingerprints changed")
        manifest = stored
    else:
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / "study_manifest.json", manifest)
    write_json(directory / "search_config.json", search)
    return directory, manifest


def suggest_parameters(trial, specifications: dict) -> dict:
    parameters = {}
    for path, spec in specifications.items():
        kind = spec["type"]
        if kind == "categorical":
            choices = spec["choices"]
            if any(isinstance(value, (list, dict)) for value in choices):
                # Optuna persists scalar categories; exports retain the real JSON value.
                index = trial.suggest_categorical(path, list(range(len(choices))))
                value = choices[index]
            else:
                value = trial.suggest_categorical(path, choices)
        else:
            options = {key: spec[key] for key in ("log", "step") if key in spec}
            suggest = trial.suggest_int if kind == "int" else trial.suggest_float
            value = suggest(path, spec["low"], spec["high"], **options)
        parameters[path] = deepcopy(value)
    return parameters


def target_reached(value: float, objective: dict) -> bool:
    if "target" not in objective:
        return False
    if objective["direction"] == "minimize":
        return value <= objective["target"]
    return value >= objective["target"]


def export_best_parameters(study, directory: Path, manifest: dict, objective: dict) -> dict | None:
    # best_trial raises ValueError until the first trial completes successfully.
    completed = [trial for trial in study.trials if trial.state.name == "COMPLETE"]
    if not completed:
        return None
    best = study.best_trial
    saved = {
        "schema_version": 1,
        "study_name": study.study_name,
        "objective": deepcopy(objective),
        "params": best.user_attrs["decoded_params"],
        "best_value": best.value,
        "trial_number": best.number,
        "run_dir": best.user_attrs["run_dir"],
        "checkpoint": best.user_attrs["checkpoint"],
        "best_epoch": best.user_attrs["best_epoch"],
        "base_config": manifest["base_config"],
        "data_fingerprints": manifest["data_fingerprints"],
    }
    write_json(directory / "best_params.json", saved)
    return saved


def import_optuna():
    """Optional dependency remains absent from normal training and saved-parameter reuse."""
    try:
        import optuna
        from sqlalchemy.pool import NullPool
    except ImportError as error:
        raise ImportError('Optuna search requires: pip install -e ".[optuna]"') from error
    return optuna, NullPool


def open_study(optuna, storage, search: dict, direction: str, resume: bool):
    study = optuna.create_study(
        study_name=search["study_name"],
        storage=storage,
        sampler=optuna.samplers.TPESampler(seed=search["sampler_seed"]),
        direction=direction,
        load_if_exists=resume,
    )
    if resume:
        # A fresh seeded startup RNG would otherwise repeat previous suggestions.
        seed = (search["sampler_seed"] + len(study.trials)) % 2**32
        study.sampler = optuna.samplers.TPESampler(seed=seed)
    if study.direction.name.lower() != direction:
        raise ValueError("Stored study direction does not match the objective")
    return study


def run_trial(trial, config: dict, search: dict, selection: dict, directory: Path, name) -> float:
    """Train one suggested configuration and return its validation objective."""
    parameters = suggest_parameters(trial, search["parameters"])
    trial.set_user_attr("decoded_params", parameters)
    trial_config = apply_parameters(config, parameters)
    run_dir = directory / "trials" / f"trial_{trial.number:05d}"
    trial_config["training"]["output_dir"] = str(run_dir)
    trial.set_user_attr("run_dir", str(run_dir))
    trial_name = f"{name or search['study_name']} / trial {trial.number}"
    actual_run = train(trial_config, name=trial_name, selection=selection)
    report = read_json_object(actual_run / "metrics.json")
    value = report["metrics"][selection["metric"]]
    trial.set_user_attr("run_dir", str(actual_run))
    trial.set_user_attr("checkpoint", str(actual_run / "best.pt"))
    trial.set_user_attr("best_epoch", report["best_epoch"])
    return value


def run_search(config: dict, search: dict, *, n_trials=None, resume=False, name=None) -> Path:
    """Run additional trials, or stop immediately if a saved best meets the target.

    The config and the search come validated from load_config and load_search_config.
    """
    optuna, NullPool = import_optuna()
    trials = search["n_trials"] if n_trials is None else n_trials
    objective = search["objective"]
    selection = selection_from_objective(objective)
    directory, manifest = prepare_study_directory(config, search, resume)

    def export_best(study):
        return export_best_parameters(study, directory, manifest, objective)

    def stop_at_target(study, trial):
        export_best(study)
        completed = trial.state == optuna.trial.TrialState.COMPLETE
        if completed and target_reached(trial.value, objective):
            study.stop()

    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{(directory / 'study.db').as_posix()}",
        engine_kwargs={"poolclass": NullPool, "connect_args": {"timeout": 60}},
    )
    study = None
    try:
        study = open_study(optuna, storage, search, selection["direction"], resume)
        best = export_best(study)
        if best is not None and target_reached(best["best_value"], objective):
            print(f"Target already reached: {best['best_value']}; no new trials", flush=True)
            return directory

        study.optimize(
            lambda trial: run_trial(trial, config, search, selection, directory, name),
            n_trials=trials,
            timeout=search.get("timeout"),
            n_jobs=1,
            callbacks=[stop_at_target],
            gc_after_trial=True,
        )
        if export_best(study) is None:
            print("No completed trials; best_params.json not created", flush=True)
        else:
            print(f"Best parameters: {directory / 'best_params.json'}", flush=True)
        return directory
    finally:
        # Also reached on interruption: keep the best completed trial on disk.
        try:
            if study is not None:
                export_best(study)
        finally:
            storage.remove_session()
            storage.engine.dispose()
