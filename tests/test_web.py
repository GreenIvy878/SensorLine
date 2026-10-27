"""Routing of the dashboard server: page files, API dispatch, downloads and refusals."""

from http.server import BaseHTTPRequestHandler
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sensorline.analysis.web import STATIC, AnalysisState, make_server
from sensorline.config import load_data_description
from sensorline.local_page import STATIC_TYPES, LocalServer, first_free_server
from sensorline.toydata import make_toy_dataset


class DashboardRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="sensorline_web_")
        root = Path(cls.temporary.name)
        data = load_data_description(make_toy_dataset(root / "toy", train=4, val=2, seed=2))
        cls.state = AnalysisState(data, root / "analysis", 17, runs_root=root / "runs")
        cls.server = make_server(cls.state, 0)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join(timeout=5)
        cls.server.server_close()
        cls.temporary.cleanup()

    def status(self, path, method="GET", headers=None):
        request = Request(self.base + path, method=method, headers=headers or {})
        try:
            with urlopen(request) as response:
                return response.status, response.headers, response.read()
        except HTTPError as error:
            return error.code, error.headers, error.read()

    def test_every_page_file_is_served_with_its_fixed_type(self):
        files = [path for path in STATIC.iterdir() if path.suffix in STATIC_TYPES]
        self.assertTrue(files)
        for path in files:
            with self.subTest(file=path.name):
                code, headers, body = self.status("/" + path.name)
                self.assertEqual(code, 200)
                self.assertEqual(headers["Content-Type"], STATIC_TYPES[path.suffix] + "; charset=utf-8")
                self.assertEqual(body, path.read_bytes())
        code, headers, body = self.status("/")
        self.assertEqual((code, body), (200, (STATIC / "dashboard.html").read_bytes()))
        for missing in ("/missing.js", "/../web.py", "/api/unknown", "/api/experiment-prepare"):
            with self.subTest(path=missing):
                self.assertEqual(self.status(missing)[0], 404)

    def test_analysis_download_and_api_dispatch(self):
        self.state.start()
        deadline = time.time() + 60
        while not self.state.status()["available"] and time.time() < deadline:
            time.sleep(0.1)
        code, headers, body = self.status("/analysis.json")
        self.assertEqual(code, 200)
        self.assertEqual(headers["Content-Disposition"], 'attachment; filename="analysis.json"')
        code, headers, _ = self.status("/api/analysis")
        self.assertEqual((code, headers["Content-Disposition"]), (200, None))
        self.assertEqual(json.loads(self.status("/api/experiments")[2]), [])
        self.assertIn("configs", json.loads(self.status("/api/model/options")[2]))
        self.assertEqual(json.loads(self.status("/api/inference/checkpoints")[2]), [])
        self.assertEqual(json.loads(self.status("/api/inference/status")[2]), {"inference": None})
        self.assertEqual(self.status("/api/inference/unknown")[0], 400)
        # Errors of a view come back as JSON with status 400.
        code, _, body = self.status("/api/sample?split=train&id=absent.png")
        self.assertEqual(code, 400)
        self.assertIn("error", json.loads(body))

    def test_posts_reach_only_their_endpoints(self):
        self.assertEqual(self.status("/api/experiments", "POST")[0], 404)
        self.assertEqual(self.status("/api/unknown", "POST")[0], 404)
        self.assertEqual(self.status("/api/experiment-prepare?id=absent", "POST")[0], 400)
        self.assertEqual(self.status("/api/model/benchmark", "POST")[0], 400)  # a body is required
        self.assertEqual(self.status("/api/inference/start", "POST")[0], 400)  # a body is required
        self.assertEqual(self.status("/api/inference/start", "POST", {"Host": "evil.test"})[0], 403)
        self.assertEqual(self.status("/api/inference/start", "POST", {"Origin": "http://evil.test"})[0], 403)

    def test_every_request_must_name_the_local_host(self):
        # A rebound DNS name reaches 127.0.0.1 but carries its own name in Host.
        foreign = {"Host": "evil.test"}
        for path in ("/", "/dashboard.js", "/api/status", "/api/analysis", "/analysis.json",
                     "/api/sample?split=train&id=toy_0000.png", "/api/experiments",
                     "/api/inference/checkpoints", "/api/model/options"):
            with self.subTest(method="GET", path=path):
                self.assertEqual(self.status(path, "GET", foreign)[0], 403)
        running = self.state.status()["running"]
        for path in ("/api/run", "/api/refresh", "/api/experiment-prepare?id=absent", "/api/model/benchmark"):
            with self.subTest(method="POST", path=path):
                self.assertEqual(self.status(path, "POST", foreign)[0], 403)
        self.assertEqual(self.state.status()["running"], running)
        port = self.server.server_port
        for host in (f"127.0.0.1:{port}", f"localhost:{port}"):
            with self.subTest(host=host):
                self.assertEqual(self.status("/api/status", "GET", {"Host": host})[0], 200)


class LocalPageTests(unittest.TestCase):
    def test_a_served_port_is_skipped(self):
        # On Windows a plain HTTP server would bind the served port a second time.
        def make(port):
            return LocalServer(("127.0.0.1", port), BaseHTTPRequestHandler)

        taken = make(0)
        try:
            with self.assertRaises(OSError):
                make(taken.server_port)
            server = first_free_server(make, taken.server_port)
            self.assertNotEqual(server.server_port, taken.server_port)
            server.server_close()
        finally:
            taken.server_close()


if __name__ == "__main__":
    unittest.main()
