"""The screen of tracking: a local web page fed by the event bus.

`python track.py` starts it and opens the page. The page begins with forms: choose a
recording (CSV) and the weights and start, or open a finished run folder. Then it shows
the signal strip, the windows, the count and the tracks. Two modes share one picture:

  live    the model runs in a background thread at real time × speed; a LiveRecorder
          subscribed to the bus keeps the history of every window;
  replay  the run logs are read back without the model: the windows are rebuilt from
          the CSV, the saved candidates go through the same NMS, threshold and tracker,
          so the events equal the logged ones (checked and shown). Changing τ or the
          NMS IoU recomputes the whole recording from the saved candidates.

The page only reads the history; stepping back never runs anything again. Rendering
runs in the browser, so drawing cannot slow the processing down; the server only
answers requests for windows it already has. Uses only the standard library, like the
dashboard, and listens on 127.0.0.1 with the same Host check.
"""

import base64
import csv
import json
import pickle
import string
import threading
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch
import yaml
from torchvision.ops import nms

from ..local_page import LocalHandler, LocalServer
from .detector import Box, Detection, WindowResult
from .pipeline import EventBus, SegmentDone, TrackingRun, recording_parts, stream_windows, track_recording
from .settings import INPUT_KINDS, load_settings, validate_settings
from .tracker import CLOSED, MERGED, REJECTED, Tracker

STATIC = Path(__file__).parent / "static"
# Windows per /api/windows answer.
PAGE = 200
# The files last used in the forms, so they need to be chosen once.
PREFS_PATH = Path.home() / ".sensorline_track.json"
RECENT = 8
LISTED = 3000
# Errors a request may end in; their text is shown to the user in the form.
REQUEST_ERRORS = (ValueError, KeyError, TypeError, OSError, RuntimeError, EOFError, pickle.UnpicklingError,
                  yaml.YAMLError)
SLEEP_SLICE_S = 0.2


class RunCancelled(Exception):
    """Raised inside the run of a live session that the user left."""


def encode(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array, dtype=np.uint8).tobytes()).decode()


def track_row(track) -> list:
    box = track.box_best
    return [track.id, track.state, box.p1, box.t1, box.p2, box.t2, track.hits, track.misses,
            round(track.best_score, 4), sum(o.cut for o in track.obs), track.merged_into]


class Pacer:
    """Holds the producer to real time × speed; speed 0 runs as fast as possible."""

    def __init__(self, speed: float, model_rate_hz: float):
        self.speed = speed
        self.rate = model_rate_hz
        self.running = threading.Event()
        self.running.set()
        self.cancelled = False
        self.origin = None          # (wall time, stream seconds) when pacing last (re)started

    def set_speed(self, speed: float) -> None:
        self.speed = speed
        self.origin = None

    def pause(self) -> None:
        self.running.clear()

    def play(self) -> None:
        self.origin = None
        self.running.set()

    def cancel(self) -> None:
        self.cancelled = True
        self.running.set()

    def wait(self, stream_samples: float) -> None:
        self.running.wait()
        if self.cancelled:
            raise RunCancelled()
        if not self.speed or not self.rate:
            return
        seconds = stream_samples / self.rate
        if self.origin is None:
            self.origin = (time.perf_counter(), seconds)
        wall, start = self.origin
        while True:
            delay = wall + (seconds - start) / self.speed - time.perf_counter()
            if delay <= 0:
                return
            time.sleep(min(delay, SLEEP_SLICE_S))
            if self.cancelled:
                raise RunCancelled()


class LiveRecorder:
    """Bus subscriber keeping what the page shows, window by window."""

    def __init__(self, display_peak: float, kind: str, pacer: Pacer | None = None):
        self.display_peak = display_peak
        self.kind = kind
        self.pacer = pacer
        self.lock = threading.Lock()
        self.windows, self.details = [], []
        self.last_column = None
        self.count = 0
        self.pending_events = []
        self.finished = False
        self.summary = None

    def display(self, raw: np.ndarray, window: np.ndarray) -> np.ndarray:
        """Stream samples -> 0..255 for the strip, on one scale for the whole recording.

        For display only, each sensor is taken from its own median in the window, so
        that sensors idling a little higher do not draw stripes; the model input is
        not affected.
        """
        if self.kind == "pixels":
            return np.clip(raw, 0, 255)
        excess = np.clip(raw - np.median(window, axis=1, keepdims=True), 0, None)
        return np.clip(np.round(255 * np.sqrt(excess / self.display_peak)), 0, 255)

    def on_window(self, result, events, tracker=None) -> None:
        rows = [[round(e.t_glob, 3), e.type, e.track_id, e.payload] for e in events]
        for event in events:
            if event.type == "counted":
                self.count = event.payload["count"]
        if result is None:
            # Closing a segment: its events belong to the last window shown.
            with self.lock:
                if self.windows:
                    self.windows[-1]["events"] += rows
                    self.windows[-1]["count"] = self.count
                    if tracker is not None:
                        self.windows[-1]["tracks"] += [track_row(t) for t in tracker.tracks
                                                       if t.state in (CLOSED, REJECTED)
                                                       and t.last_k == self.windows[-1]["k"]]
            return
        start = result.t0 if self.last_column is None else max(result.t0, self.last_column + 1)
        end = result.t0 + result.valid
        columns = ""
        if result.raw is not None and end > start:
            shown = self.display(result.raw[:, start - result.t0:end - result.t0], result.raw[:, :result.valid])
            columns = encode(shown.T)          # column by column: 40 values per sample
            self.last_column = end - 1
        touched = {e.track_id for e in events}
        tracks = [track_row(t) for t in tracker.active] if tracker else []
        if tracker:
            tracks += [track_row(t) for t in tracker.tracks
                       if t.id in touched and t.state in (CLOSED, REJECTED, MERGED)]
        window = {
            "k": result.k, "segment": result.segment, "t0": result.t0, "last": result.last,
            "valid": result.valid, "skip": result.extra.get("skip"), "peak": result.extra.get("peak"),
            "infer_ms": round(result.infer_ms, 3), "window_ms": round(result.window_ms, 3),
            "count": self.count, "columns_start": start, "columns": columns,
            "detections": [[d.box_global.p1, d.box_global.t1, d.box_global.p2, d.box_global.t2, round(d.score, 4),
                            int(d.cut_left), int(d.cut_right)] for d in result.detections],
            "events": rows, "tracks": tracks,
        }
        frame = None if result.frame is None else encode(np.clip(np.round(np.asarray(result.frame) * 255), 0, 255))
        candidates = [[d.box_local.p1, d.box_local.t1, d.box_local.p2, d.box_local.t2, round(d.score, 4),
                       int(kept), int(accepted)] for d, (kept, accepted) in zip(result.detections_pre_nms, result.flags)]
        with self.lock:
            self.windows.append(window)
            self.details.append({"k": result.k, "frame": frame, "candidates": candidates})
        if self.pacer is not None:
            self.pacer.wait(result.t0 + result.width)

    def on_finish(self, tracker, summary) -> None:
        with self.lock:
            self.finished = True
            self.summary = summary

    def page(self, start: int) -> dict:
        with self.lock:
            return {"windows": self.windows[start:start + PAGE], "produced": len(self.windows),
                    "finished": self.finished}

    def detail(self, index: int) -> dict:
        with self.lock:
            if not 0 <= index < len(self.details):
                raise ValueError(f"No window {index}")
            return self.details[index]


class Session:
    """State of one screen: the history, the run behind it and its controls."""

    def __init__(self, mode: str, meta: dict, recorder: LiveRecorder, pacer: Pacer | None = None):
        self.mode, self.meta, self.recorder, self.pacer = mode, meta, recorder, pacer
        self.failure = None
        self.version = 0
        self.thread = None
        self.lock = threading.Lock()

    def status(self) -> dict:
        summary = self.recorder.summary or {}
        return {"mode": self.mode, "version": self.version, "produced": len(self.recorder.windows),
                "finished": self.recorder.finished, "paused": bool(self.pacer and not self.pacer.running.is_set()),
                "speed": self.pacer.speed if self.pacer else None, "count": self.recorder.count,
                "rtf": summary.get("compute_rtf", summary.get("rtf")), "failure": self.failure}

    def cancel(self) -> None:
        """Stop the run behind a live session; a replay has none."""
        if self.pacer is not None:
            self.pacer.cancel()

    def control(self, body: dict) -> dict:
        action = body.get("action")
        if self.pacer is None:
            raise ValueError("Replay is played in the page; only a live run has server controls")
        if action == "pause":
            self.pacer.pause()
        elif action == "play":
            self.pacer.play()
        elif action == "speed":
            speed = float(body["value"])
            if speed < 0:
                raise ValueError("speed must be nonnegative")
            self.pacer.set_speed(speed)
        else:
            raise ValueError(f"Unknown action {action!r}")
        return self.status()


def display_peak(settings: dict) -> float:
    """The signal level drawn at full brightness in the strip: raw ADC units or pixel values."""
    return settings["output"]["display_peak_adc"] if settings["data"]["input_kind"] == "raw_adc" else 255.0


def describe(settings, geometry, passport, preprocessor, csv_path: Path, checkpoint: Path, recording_rate) -> dict:
    sensors, width = passport["input_shape"]
    return {
        "recording": csv_path.name, "model": f"{checkpoint.parent.name}/{checkpoint.name}",
        "sensors": sensors, "width": width, "stride": geometry["stride"],
        "model_rate_hz": preprocessor.model_rate_hz, "recording_rate_hz": recording_rate,
        "polls_per_sample": preprocessor.factor, "pool": settings["data"]["pool"],
        "score_thr": settings["model"]["score_thr"], "nms_iou": settings["model"]["nms_iou"],
        "max_detections": passport["postprocess"]["max_detections"],
        "match_mode": settings["tracker"]["match_mode"], "count_on": settings["tracker"]["count_on"],
        "input": preprocessor.describe(),
    }


def start_live(csv_path, checkpoint, settings: dict, *, run_dir=None, speed=1.0, log=print) -> Session:
    """Run tracking in a background thread, paced, and return the session the page reads."""
    settings = json.loads(json.dumps(settings))
    settings["output"]["save_frames"] = True
    prepared = TrackingRun(csv_path, checkpoint, settings, log=log)
    pacer = Pacer(speed, prepared.preprocessor.model_rate_hz)
    recorder = LiveRecorder(display_peak(prepared.settings), prepared.preprocessor.kind, pacer)
    meta = describe(prepared.settings, prepared.geometry, prepared.passport, prepared.preprocessor,
                    Path(csv_path), Path(checkpoint), prepared.recording_rate)
    session = Session("live", meta, recorder, pacer)

    def run():
        try:
            directory, _ = track_recording(csv_path, checkpoint, settings, run_dir=run_dir,
                                           subscribers=[recorder], log=log, prepared=prepared)
            session.meta["run_dir"] = str(directory.resolve())
        except RunCancelled:
            pass
        except Exception as error:     # shown on the page; the server keeps running
            session.failure = str(error)
            recorder.finished = True
            raise

    session.thread = threading.Thread(target=run, daemon=True)
    session.thread.start()
    return session


class Replay:
    """A finished run read back from its folder; tracks are recomputed from the logged candidates."""

    def __init__(self, run_dir, log=print):
        self.run_dir = Path(run_dir)
        if not (self.run_dir / "config.resolved.yaml").is_file():
            raise ValueError(f"{self.run_dir} is not a run folder: no config.resolved.yaml in it")
        for name in ("detections.csv", "timeline.csv", "events.csv"):
            if not (self.run_dir / name).is_file():
                raise ValueError(f"{self.run_dir} has no {name}")
        self.log = log
        self.resolved = yaml.safe_load((self.run_dir / "config.resolved.yaml").read_text(encoding="utf-8"))
        self.settings = self.resolved["settings"]
        try:
            validate_settings(self.settings)
        except KeyError as error:
            raise ValueError(f"{self.run_dir} was made by an older version of the tracker (no option {error}); "
                             "run the recording again") from error
        self.passport = self.resolved["model"]["passport"]
        self.csv_path = Path(self.resolved["csv"]["path"])
        if not self.csv_path.is_file():
            raise ValueError(f"The recording of this run is gone: {self.csv_path}")
        self.candidates = {}
        with (self.run_dir / "detections.csv").open(encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                self.candidates.setdefault(int(row["k"]), []).append(row)
        with (self.run_dir / "timeline.csv").open(encoding="utf-8") as stream:
            self.timeline = {int(row["k"]): row for row in csv.DictReader(stream)}
        with (self.run_dir / "events.csv").open(encoding="utf-8") as stream:
            self.logged_events = [dict(row) for row in csv.DictReader(stream)]

    def result(self, window, score_thr=None, nms_iou=None) -> WindowResult:
        rows = self.candidates.get(window.k, [])
        detections = []
        for row in rows:
            local = Box(float(row["p1"]), float(row["t1_local"]), float(row["p2"]), float(row["t2_local"]))
            detections.append(Detection(window.k, local, Box(local.p1, float(row["t1"]), local.p2, float(row["t2"])),
                                        float(row["score"]), row["cut_left"] == "1", row["cut_right"] == "1",
                                        int(row["raw_index"])))
        if score_thr is None:
            flags = [(row["nms_kept"] == "1", row["accepted"] == "1") for row in rows]
            accepted = [d for d, (_, a) in zip(detections, flags) if a]
            # The detector lists accepted boxes in NMS order (descending score).
            order = sorted(range(len(accepted)), key=lambda i: -accepted[i].score)
            accepted = [accepted[i] for i in order]
        else:
            flags, accepted = self.postprocess(detections, window.valid, score_thr, nms_iou)
        timing = self.timeline.get(window.k, {})
        return WindowResult(window.k, window.segment, window.t0, window.raw.shape[1], window.first, window.last,
                            detections, flags, accepted, float(timing.get("infer_ms") or 0),
                            float(timing.get("window_ms") or 0), frame=window.frame, valid=window.valid,
                            raw=window.raw, extra={"skip": window.skip, "peak": window.peak})

    def postprocess(self, detections, valid, score_thr, nms_iou):
        """NMS and τ as in WindowDetector, over the logged candidates (boxes in window coordinates)."""
        if not detections:
            return [], []
        # Candidates were decoded with x = time; NMS is symmetric in the axes.
        boxes = torch.tensor([[d.box_local.t1, d.box_local.p1, d.box_local.t2, d.box_local.p2] for d in detections],
                             dtype=torch.float64)
        scores = torch.tensor([d.score for d in detections], dtype=torch.float64)
        kept = nms(boxes, scores, nms_iou)[:self.passport["postprocess"]["max_detections"]].tolist()
        flags = [(False, False)] * len(detections)
        accepted = []
        for index in kept:
            flags[index] = (True, False)
            detection = detections[index]
            if detection.score >= score_thr and detection.box_local.t1 < valid:
                flags[index] = (True, True)
                accepted.append(detection)
        return flags, accepted

    def run(self, recorder: LiveRecorder, score_thr=None, nms_iou=None) -> tuple:
        """Feed the rebuilt windows through the tracker into the recorder; returns the event rows and the tracker."""
        settings = json.loads(json.dumps(self.settings))
        source, preprocessor, windower, _, _ = recording_parts(self.csv_path, settings, self.passport, self.log)
        tracker = Tracker(settings["tracker"])
        bus = EventBus([recorder])
        rows = []
        for item in stream_windows(source, preprocessor, windower):
            if isinstance(item, SegmentDone):
                events = tracker.finish_segment()
                bus.publish(None, events, tracker)
            else:
                result = self.result(item, score_thr, nms_iou)
                events = tracker.step(result)
                bus.publish(result, events, tracker)
            rows += [{key: str(value) for key, value in event.row().items()} for event in events]
        recorder.on_finish(tracker, {"count": tracker.count})
        return rows, tracker

    def session(self, score_thr=None, nms_iou=None) -> Session:
        kind = self.settings["data"]["input_kind"]
        recorder = LiveRecorder(display_peak(self.settings), kind)
        meta = {**self.resolved_meta(), "score_thr": score_thr or self.settings["model"]["score_thr"],
                "nms_iou": nms_iou or self.settings["model"]["nms_iou"]}
        session = Session("replay", meta, recorder)
        rows, tracker = self.run(recorder, score_thr, nms_iou)
        # Logged events hold floats as written; compare in the same text form.
        logged = [{key: row[key] for key in ("t_glob", "k", "type", "track_id", "payload")} for row in self.logged_events]
        session.meta["replay_matches_log"] = rows == logged if score_thr is None else None
        summary = self.run_dir / "summary.json"
        if summary.is_file():
            recorder.summary = json.loads(summary.read_text(encoding="utf-8"))
        return session

    def resolved_meta(self) -> dict:
        settings, windows, model = self.settings, self.resolved["windows"], self.resolved["model"]
        sensors, width = self.passport["input_shape"]
        rate = self.resolved.get("input", {})
        meta = {
            "recording": self.csv_path.name, "model": Path(model["checkpoint"]).name, "run_dir": str(self.run_dir),
            "sensors": sensors, "width": width, "stride": windows["stride"],
            "model_rate_hz": rate.get("model_rate_hz"), "recording_rate_hz": rate.get("recording_rate_hz"),
            "polls_per_sample": rate.get("polls_per_sample"), "pool": settings["data"]["pool"],
            "max_detections": self.passport["postprocess"]["max_detections"],
            "match_mode": settings["tracker"]["match_mode"], "count_on": settings["tracker"]["count_on"],
            "input": rate.get("description"),
        }
        return meta


def read_prefs(path) -> dict:
    """The saved choices of the forms; each file field holds a list of the last files, newest first."""
    try:
        saved = json.loads(Path(path).read_text(encoding="utf-8")) if path else {}
    except (OSError, ValueError):
        return {}
    if not isinstance(saved, dict):
        return {}
    for key in ("csv", "weights", "runs"):
        value = saved.get(key)
        saved[key] = [item for item in value if isinstance(item, str)] if isinstance(value, list) else []
    return saved


def drives() -> list:
    letters = [f"{letter}:\\" for letter in string.ascii_uppercase if Path(f"{letter}:\\").exists()]
    return letters or ["/"]


def browse(path, kind: str, extensions: str) -> dict:
    """A folder listing for the file chooser of the forms.

    kind "file" lists folders and the files with one of the comma-separated extensions;
    kind "run" lists folders only and says whether the folder itself is a run folder.
    """
    wanted = {e.strip().lower() for e in extensions.split(",") if e.strip()}
    folder = Path(path).expanduser() if path else Path.home()
    while not folder.is_dir():
        if folder.parent == folder:
            folder = Path.home()
            break
        folder = folder.parent
    folder = folder.resolve()
    try:
        children = sorted(folder.iterdir(), key=lambda child: (not child.is_dir(), child.name.lower()))
    except OSError as error:
        raise ValueError(f"Cannot read {folder}: {error}") from error
    entries = []
    for child in children:
        if child.name.startswith(".") or len(entries) >= LISTED:
            continue
        try:
            is_dir = child.is_dir()
            if is_dir:
                entries.append({"name": child.name, "path": str(child), "dir": True})
            elif kind == "file" and (not wanted or child.suffix.lower() in wanted):
                entries.append({"name": child.name, "path": str(child), "dir": False, "size": child.stat().st_size})
        except OSError:
            continue
    return {"path": str(folder), "parent": str(folder.parent) if folder.parent != folder else None,
            "entries": entries, "drives": drives(), "is_run": (folder / "config.resolved.yaml").is_file()}


class Screen:
    """The server side of the page: the current session, the forms and recomputing a replay."""

    def __init__(self, default_config, user_config=None, *, overrides=(), prefs_path=PREFS_PATH):
        self.default_config, self.user_config, self.overrides = default_config, user_config, list(overrides)
        self.prefs_path = prefs_path
        self.session = None
        self.replay = None
        self.version = 0
        self.lock = threading.RLock()

    def status(self) -> dict:
        session = self.session
        if session is None:
            return {"active": False, "version": self.version}
        return {"active": True, **session.status()}

    def set_session(self, session: Session | None, replay: Replay | None = None) -> None:
        with self.lock:
            if self.session is not None and self.session is not session:
                self.session.cancel()
            self.version += 1
            if session is not None:
                session.version = self.version
            self.session, self.replay = session, replay

    def remember(self, **entries) -> None:
        if not self.prefs_path:
            return
        prefs = read_prefs(self.prefs_path)
        for key, value in entries.items():
            if value is None:
                continue
            if key in ("rate", "speed", "input_kind"):
                prefs[key] = value
            elif value:
                prefs[key] = [value] + [old for old in prefs.get(key, []) if old != value][:RECENT - 1]
        try:
            Path(self.prefs_path).write_text(json.dumps(prefs), encoding="utf-8")
        except OSError:
            pass

    def form(self) -> dict:
        prefs = read_prefs(self.prefs_path)
        recent = {key: [p for p in prefs.get(key, []) if isinstance(p, str) and Path(p).exists()]
                  for key in ("csv", "weights", "runs")}
        defaults = load_settings(self.default_config, self.user_config, self.overrides)
        return {"csv": (recent["csv"] or [""])[0], "weights": (recent["weights"] or [""])[0],
                "run": (recent["runs"] or [""])[0], "rate": str(prefs.get("rate", defaults["data"]["sample_rate_hz"])),
                "input_kind": prefs.get("input_kind", defaults["data"]["input_kind"]),
                "speed": prefs.get("speed", 1.0), "recent": recent}

    @staticmethod
    def existing(value, label: str, hint: str = "", kind: str = "file") -> Path:
        """The file (or folder) a form field names; a message for the form if there is none."""
        text = str(value or "").strip()
        if not text:
            raise ValueError(f"Choose the {label}" + (f": {hint}" if hint else ""))
        path = Path(text).expanduser()
        if not (path.is_file() if kind == "file" else path.is_dir()):
            raise ValueError(f"{label[0].upper() + label[1:]} not found: {path}")
        return path

    def start(self, body: dict) -> dict:
        csv_path = self.existing(body.get("csv"), "recording", "a CSV file")
        weights = self.existing(body.get("weights"), "weights", "a .pt file")
        rate = str(body.get("rate", "")).strip() or "auto"
        speed = float(body.get("speed", 1.0))
        if speed < 0:
            raise ValueError("Speed must be nonnegative")
        overrides = [*self.overrides, f"data.sample_rate_hz={rate}"]
        kind = body.get("input_kind")
        if kind is not None:
            if kind not in INPUT_KINDS:
                raise ValueError(f"Input must be one of {INPUT_KINDS}")
            overrides.append(f"data.input_kind={kind}")
        settings = load_settings(self.default_config, self.user_config, overrides)
        with self.lock:
            self.set_session(None)
            session = start_live(csv_path, weights, settings, speed=speed)
            self.set_session(session)
        self.remember(csv=str(csv_path), weights=str(weights), rate=rate, speed=speed, input_kind=kind)
        return self.status()

    def open_run(self, body: dict) -> dict:
        run = self.existing(body.get("run"), "run folder", kind="dir")
        replay = Replay(run)
        session = replay.session()
        with self.lock:
            self.set_session(session, replay)
        self.remember(runs=str(run))
        return self.status()

    def reset(self) -> dict:
        with self.lock:
            self.set_session(None)
        return self.status()

    def recompute(self, body: dict) -> dict:
        if self.replay is None or self.session is None or not self.session.recorder.finished:
            raise ValueError("Thresholds can be changed in a replay (open a saved run)")
        score_thr, nms_iou = float(body["score_thr"]), float(body["nms_iou"])
        if not (0 < score_thr <= 1 and 0 < nms_iou <= 1):
            raise ValueError("score_thr and nms_iou must be in (0, 1]")
        if nms_iou > self.replay.settings["model"]["nms_iou_max"]:
            raise ValueError(f"NMS IoU above {self.replay.settings['model']['nms_iou_max']} would suppress touching neighbours")
        with self.lock:
            same = (score_thr == self.replay.settings["model"]["score_thr"]
                    and nms_iou == self.replay.settings["model"]["nms_iou"])
            session = self.replay.session(None if same else score_thr, None if same else nms_iou)
            self.set_session(session, self.replay)
        return self.status()

    def require_session(self) -> Session:
        session = self.session
        if session is None:
            raise ValueError("No run is open: choose a recording and the weights first")
        return session


def make_server(screen: Screen, port: int) -> LocalServer:
    actions = {"/api/start": screen.start, "/api/open": screen.open_run, "/api/reset": lambda _: screen.reset(),
               "/api/control": lambda body: screen.require_session().control(body),
               "/api/recompute": screen.recompute}

    class Handler(LocalHandler):
        static = STATIC
        index = "live.html"

        def do_GET(self):
            if self.refuse_foreign_request():
                return
            url = urlparse(self.path)
            query = parse_qs(url.query)
            try:
                if url.path == "/api/status":
                    self.send_json(screen.status())
                elif url.path == "/api/form":
                    self.send_json(screen.form())
                elif url.path == "/api/browse":
                    self.send_json(browse(query.get("path", [""])[0], query.get("kind", ["file"])[0],
                                          query.get("ext", [""])[0]))
                elif url.path == "/api/meta":
                    session = screen.require_session()
                    self.send_json({**session.meta, "mode": session.mode})
                elif url.path == "/api/windows":
                    session = screen.require_session()
                    self.send_json({**session.recorder.page(int(query.get("from", ["0"])[0])),
                                    "version": session.version})
                elif url.path == "/api/window":
                    self.send_json(screen.require_session().recorder.detail(int(query.get("index", ["0"])[0])))
                elif not self.send_page_file(url.path):
                    self.send_json({"error": "Not found"}, 404)
            except REQUEST_ERRORS as error:
                self.send_json({"error": str(error)}, 400)

        def do_POST(self):
            if self.refuse_foreign_request():
                return
            if self.path not in actions:
                self.send_json({"error": "Not found"}, 404)
                return
            try:
                self.send_json(actions[self.path](self.json_body()))
            except REQUEST_ERRORS as error:
                self.send_json({"error": str(error)}, 400)

    return LocalServer(("127.0.0.1", port), Handler)
