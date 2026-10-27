"""Open the local dashboard: dataset statistics, inference, runs and models.

    python dashboard.py --config configs/toy.yaml

The data section of the model YAML is the dataset whose statistics are shown;
the YAMLs next to it and the runs under --runs-root can be inspected and run.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from sensorline.analysis.web import AnalysisState, make_server
from sensorline.config import load_config
from sensorline.local_page import first_free_server, serve_page


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--config", required=True, help="Model YAML whose data section is analyzed")
    parser.add_argument("--runs-root", default="runs", help="Folder with the training runs")
    parser.add_argument("--output-dir", default="analysis_results", help="Cache of the statistics and inference results")
    parser.add_argument("--refresh", action="store_true", help="Recompute the statistics even if the cache matches")
    parser.add_argument("--port", type=int, default=8008, help="First port to try on 127.0.0.1")
    parser.add_argument("--no-browser", action="store_true", help="Only print the address")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        state = AnalysisState(config["data"], args.output_dir, config["training"]["seed"], args.runs_root,
                              config_path=args.config)
        server = first_free_server(lambda port: make_server(state, port), args.port)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    state.start(args.refresh)
    serve_page(server, "SensorLine dashboard", open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
