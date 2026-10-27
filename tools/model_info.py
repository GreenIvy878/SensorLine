import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sensorline.config import load_config, sensor_count
from sensorline.models import Detector, model_info


def main():
    parser = argparse.ArgumentParser(description="Inspect architecture and dummy forward shapes")
    parser.add_argument("--config", required=True)
    parser.add_argument("--time", type=int, default=40, help="Time steps of the frame (default 40)")
    args = parser.parse_args()
    if args.time < 1:
        parser.error("--time must be positive")
    config = load_config(args.config)
    shape = (1, 1, sensor_count(config["data"]), args.time)
    info = model_info(Detector(config["model"]), shape)
    print(info.pop("structure"))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
