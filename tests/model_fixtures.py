"""Architecture variants kept in tests, independent of optional example YAMLs."""

from sensorline.config import load_config


def model_config(root, name):
    config = load_config(root / "tests/fixtures/baseline.yaml")
    if name == "example_tcn":
        config["model"]["backbone"][1] = {
            "type": "tcn",
            "channels": 24,
            "kernel": [1, 5],
            "stride": [1, 2],
        }
        config["model"]["backbone"][2] = {
            "type": "conv",
            "channels": 32,
            "kernel": [3, 5],
            "dilation": [1, 2],
        }
    elif name == "example_fpn":
        config["model"]["backbone"][2]["stride"] = [2, 2]
        config["model"]["neck"] = {"type": "fpn", "features": [1, 2], "channels": 24}
    elif name != "baseline":
        raise ValueError(name)
    return config
