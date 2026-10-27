"""Count objects in CSV recordings of the sensor line with trained weights.

    python track.py

opens the screen in the browser: choose the recording and the weights, start, watch.
"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from sensorline.local_page import first_free_server, serve_page
from sensorline.tracking.live import Screen, make_server
from sensorline.tracking.settings import load_settings

DEFAULT_CONFIG = ROOT / "configs" / "tracking" / "default.yaml"


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--port", type=int, default=8010, help="First port to try on 127.0.0.1")
    parser.add_argument("--no-browser", action="store_true", help="Only print the address")
    args = parser.parse_args()
    try:
        load_settings(DEFAULT_CONFIG)
        screen = Screen(DEFAULT_CONFIG)
        server = first_free_server(lambda port: make_server(screen, port), args.port)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    serve_page(server, "SensorLine tracking", open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
