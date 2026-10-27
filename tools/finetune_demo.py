"""Fine-tuning on shifted synthetic data, with and without the protection against forgetting.

A base model is trained on the quick-start toy data (A) and fine-tuned on frames whose
objects are longer and dimmer (B): plainly at the full rate, with replay at a tenth of the
rate, and with replay plus the reference check. Every model is evaluated on the
validation frames of A and B; the intervals compare F1 on A with the base model.
"""

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sensorline.artifacts import read_json
from sensorline.comparison import paired_bootstrap, read_samples

CONFIG = ROOT / "configs" / "toy.yaml"
SHIFTED = ["--train", "150", "--val", "80", "--seed", "1", "--lengths", "12", "20", "--brightness", "0.25", "0.5"]
# Name, fine-tuning options. The plain run turns every precaution off.
VARIANTS = [
    ("plain", ["--lr-scale", "1", "--replay-ratio", "0", "--max-f1-drop", "1", "--max-count-mae-rise", "1000"]),
    ("replay", ["--max-f1-drop", "1", "--max-count-mae-rise", "1000"]),
    # 0.05 allows four of the 80 reference frames to be miscounted by one more object.
    ("replay + check", ["--max-count-mae-rise", "0.05"]),
]


def run(*arguments):
    subprocess.run([sys.executable, "-B", *map(str, arguments)], check=True, cwd=ROOT,
                   stdout=subprocess.DEVNULL)


def evaluate(output: Path, name: str, checkpoint: Path, device: str) -> dict:
    """Metrics and per-frame rows of a checkpoint on the validation frames of A and B."""
    result = {}
    for data in ("A", "B"):
        report = output / "evaluation" / f"{name}_{data}".replace(" + ", "_")
        run(ROOT / "test.py", "--config", CONFIG, "--checkpoint", checkpoint, "--device", device,
            "--data-dir", output / data / "val", "--output-dir", report)
        result[data] = (read_json(report / "metrics.json")["metrics"], read_samples(report / "samples.csv"))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", help="New or empty folder for the data, runs and reports")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f"{output} is not empty")

    print("data, base model and three fine-tuning runs; a few minutes on a CPU", flush=True)
    run(ROOT / "tools/make_toy_data.py", "--output", output / "A")
    run(ROOT / "tools/make_toy_data.py", "--output", output / "B", *SHIFTED)
    base = output / "runs" / "base"
    run(ROOT / "train.py", "--config", CONFIG, "--data", output / "A", "--device", args.device, "--output-dir", base)
    models = {"base": base}
    for name, options in VARIANTS:
        models[name] = output / "runs" / name.replace(" + ", "_")
        run(ROOT / "train.py", "--config", CONFIG, "--data", output / "B", "--finetune", base / "best.pt",
            "--epochs", 15, "--device", args.device, "--output-dir", models[name], *options)

    results = {name: evaluate(output, name, directory / "best.pt", args.device) for name, directory in models.items()}
    print(f"\n{'model':<15} {'F1 on A':>8} {'MAE on A':>9} {'F1 on B':>8} {'MAE on B':>9}  delta F1 on A, 95% CI")
    for name, result in results.items():
        (a, a_frames), (b, _) = result["A"], result["B"]
        interval = ""
        if name != "base":
            compared = paired_bootstrap(results["base"]["A"][1], a_frames)
            low, high = compared["interval"]
            interval = f"{compared['difference']:+.4f} [{low:+.4f}, {high:+.4f}]"
        print(f"{name:<15} {a['f1']:8.4f} {a['count_mae']:9.3f} {b['f1']:8.4f} {b['count_mae']:9.3f}  {interval}")
    print(f"\nRuns and reports: {output}")


if __name__ == "__main__":
    main()
