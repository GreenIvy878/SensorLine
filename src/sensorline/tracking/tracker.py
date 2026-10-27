"""Link window detections into tracks in stream coordinates and count every object once.

Tracking by detection with IoU and no motion model: in stream coordinates an object does
not move between windows, only the window edge crops it. The tracker sees only
WindowResult (t0, W and boxes), never the model or the stride, and only the current
window: a closed track is never reopened.

Step order for window k (fixed, so that the event log does not depend on the order of detections):
  1 close tracks the window has moved past (box_best.t2 <= t0)
  2-3 match active tracks to detections, greedily by descending similarity
  4 update matched tracks;  5 count misses of tracks lying wholly inside the window
  6 merge duplicates into an active or just closed track;  7 open new tracks
  8 confirm tentative tracks
"""

import json
from dataclasses import dataclass, field

from .detector import Box

TENTATIVE, CONFIRMED, CLOSED, REJECTED, MERGED = "tentative", "confirmed", "closed", "rejected", "merged"
EVENT_TYPES = ("track_new", "track_confirmed", "counted", "track_closed", "track_rejected", "merged", "miss")


@dataclass
class Observation:
    k: int
    t0: int          # start of the window that saw it
    width: int
    box: Box         # stream coordinates
    score: float
    cut_left: bool
    cut_right: bool

    @property
    def cut(self) -> bool:
        return self.cut_left or self.cut_right


@dataclass
class Track:
    id: int
    segment: int
    state: str = TENTATIVE
    obs: list = field(default_factory=list)
    hits: int = 0
    misses: int = 0
    first_k: int = -1
    last_k: int = -1
    counted: bool = False
    count_t: float | None = None
    merged_into: int | None = None

    @property
    def box_best(self) -> Box:
        """The uncut observation with the highest score; if all are cut, the longest one in t."""
        uncut = [o for o in self.obs if not o.cut]
        if uncut:
            best = max(uncut, key=lambda o: (o.score, -o.k))
        else:
            best = max(self.obs, key=lambda o: (o.box.t2 - o.box.t1, o.score, -o.k))
        return best.box

    @property
    def best_score(self) -> float:
        return max(o.score for o in self.obs)

    def add(self, observation: Observation) -> None:
        if not self.obs:
            self.first_k = observation.k
        self.obs.append(observation)
        self.hits += 1
        self.last_k = observation.k


@dataclass
class Event:
    t_glob: float
    k: int
    type: str
    track_id: int
    payload: dict

    def row(self) -> dict:
        return {"t_glob": self.t_glob, "k": self.k, "type": self.type, "track_id": self.track_id,
                "payload": json.dumps(self.payload, sort_keys=True)}


def area(box: Box) -> float:
    return max(box.p2 - box.p1, 0.0) * max(box.t2 - box.t1, 0.0)


def intersection(a: Box, b: Box) -> float:
    return max(min(a.p2, b.p2) - max(a.p1, b.p1), 0.0) * max(min(a.t2, b.t2) - max(a.t1, b.t1), 0.0)


def iou(a: Box, b: Box) -> float:
    inter = intersection(a, b)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def clip_time(box: Box, start: float, end: float) -> Box | None:
    t1, t2 = max(box.t1, start), min(box.t2, end)
    return Box(box.p1, t1, box.p2, t2) if t2 > t1 else None


def similarity(mode: str, observation: Observation, box: Box, t0: int) -> float:
    """How well a detection of the window starting at t0 continues an earlier observation."""
    if mode == "plain_iou":
        return iou(observation.box, box)
    if mode == "iomin":
        smaller = min(area(observation.box), area(box))
        return intersection(observation.box, box) / smaller if smaller > 0 else 0.0
    # overlap_clipped: compare inside the interval both windows see.
    start, end = t0, observation.t0 + observation.width
    a, b = clip_time(observation.box, start, end), clip_time(box, start, end)
    if a is None or b is None:
        return 0.0
    return iou(a, b)


def centre_t(box: Box) -> float:
    return (box.t1 + box.t2) / 2


def detection_order(detection) -> tuple:
    """Canonical order of a window's detections, independent of how the model listed them."""
    box = detection.box_global
    return (box.t1, box.p1, box.t2, box.p2, -detection.score, detection.raw_index)


class Tracker:
    def __init__(self, settings: dict):
        self.match_iou = settings["match_iou"]
        self.match_mode = settings["match_mode"]
        self.min_hits = settings["min_hits"]
        self.strong_score = settings["strong_score"]
        self.dedup_iou = settings["dedup_iou"]
        self.count_on = settings["count_on"]
        self.tracks: list[Track] = []        # every track ever opened, by id
        self.active: list[Track] = []
        self.recent: list[Track] = []        # closed confirmed tracks a later window may still overlap
        self.count = 0
        self.segment = None
        self.last_k = -1

    def event(self, events, k, kind, track, payload=None, t=None) -> None:
        payload = dict(payload or {})
        events.append(Event(centre_t(track.box_best) if t is None else t, k, kind, track.id, payload))

    def count_track(self, track: Track, k: int, events: list) -> None:
        # One count per track, never for a merged one.
        if track.counted or track.merged_into is not None:
            return
        track.counted = True
        track.count_t = centre_t(track.box_best)
        self.count += 1
        self.event(events, k, "counted", track, {"count": self.count, "hits": track.hits, "misses": track.misses,
                                                 "score": track.best_score, "box": track.box_best.as_list()})

    def close(self, track: Track, k: int, events: list) -> None:
        self.active.remove(track)
        if track.state == CONFIRMED:
            if self.count_on == "close":
                self.count_track(track, k, events)
            track.state = CLOSED
            self.recent.append(track)
            self.event(events, k, "track_closed", track, {"hits": track.hits, "misses": track.misses})
        else:
            track.state = REJECTED
            self.event(events, k, "track_rejected", track, {"hits": track.hits, "score": track.best_score})

    def new_track(self, segment: int) -> Track:
        track = Track(len(self.tracks) + 1, segment)
        self.tracks.append(track)
        return track

    def observation(self, result, detection) -> Observation:
        return Observation(result.k, result.t0, result.width, detection.box_global, detection.score,
                           detection.cut_left, detection.cut_right)

    def step(self, result) -> list[Event]:
        if result.k <= self.last_k:
            raise ValueError(f"Windows must arrive in order: {result.k} after {self.last_k}")
        if self.segment is not None and result.segment != self.segment:
            raise ValueError("A new segment starts only after finish_segment()")
        self.segment, self.last_k = result.segment, result.k
        k, t0 = result.k, result.t0
        events = []

        # 1 close
        for track in sorted(self.active, key=lambda t: t.id):
            if track.box_best.t2 <= t0:
                self.close(track, k, events)
        self.recent = [t for t in self.recent if t.obs[-1].t0 + t.obs[-1].width > t0]

        # 2-3 match
        detections = sorted(result.detections, key=detection_order)
        active = sorted(self.active, key=lambda t: t.id)
        pairs = []
        for track in active:
            for index, detection in enumerate(detections):
                value = similarity(self.match_mode, track.obs[-1], detection.box_global, t0)
                if value >= self.match_iou:
                    pairs.append((-value, track.id, index, track))
        pairs.sort(key=lambda pair: pair[:3])
        matched_tracks, matched_detections = set(), set()
        for _, track_id, index, track in pairs:
            if track_id in matched_tracks or index in matched_detections:
                continue
            matched_tracks.add(track_id)
            matched_detections.add(index)
            # 4 update
            track.add(self.observation(result, detections[index]))

        # 5 miss
        for track in active:
            if track.id in matched_tracks:
                continue
            box = track.box_best
            if box.t1 >= t0 and box.t2 <= t0 + result.width:
                track.misses += 1
                self.event(events, k, "miss", track, {"misses": track.misses})

        # 6 dedup, 7 new
        candidates = sorted(self.active + self.recent, key=lambda t: t.id)
        for index, detection in enumerate(detections):
            if index in matched_detections:
                continue
            best = None
            for track in candidates:
                value = similarity(self.match_mode, track.obs[-1], detection.box_global, t0)
                if value >= self.dedup_iou and (best is None or value > best[0]):
                    best = (value, track)
            if best is not None:
                duplicate = self.new_track(result.segment)
                duplicate.add(self.observation(result, detection))
                duplicate.state, duplicate.merged_into = MERGED, best[1].id
                self.event(events, k, "merged", duplicate, {"into": best[1].id, "similarity": best[0],
                                                            "score": detection.score})
                continue
            track = self.new_track(result.segment)
            track.add(self.observation(result, detection))
            self.active.append(track)
            self.event(events, k, "track_new", track, {"score": detection.score, "cut_left": detection.cut_left,
                                                      "cut_right": detection.cut_right})

        # 8 confirm
        for track in sorted(self.active, key=lambda t: t.id):
            if track.state != TENTATIVE:
                continue
            strong = any(not o.cut and o.score >= self.strong_score for o in track.obs)
            if track.hits >= self.min_hits or strong:
                track.state = CONFIRMED
                self.event(events, k, "track_confirmed", track, {"hits": track.hits, "strong": strong})
                if self.count_on == "confirm":
                    self.count_track(track, k, events)
        return events

    def finish_segment(self) -> list[Event]:
        """Close every active track at the end of a segment or recording; the next segment starts clean."""
        events = []
        for track in sorted(self.active, key=lambda t: t.id):
            self.close(track, self.last_k, events)
        self.recent = []
        self.segment = None
        return events

    def track_rows(self) -> list[dict]:
        rows = []
        for track in self.tracks:
            box = track.box_best
            rows.append({
                "id": track.id, "segment": track.segment, "state": track.state,
                "p1": box.p1, "t1": box.t1, "p2": box.p2, "t2": box.t2,
                "hits": track.hits, "misses": track.misses, "best_score": track.best_score,
                "cut_observations": sum(o.cut for o in track.obs),
                "first_k": track.first_k, "last_k": track.last_k,
                "counted": track.counted, "count_t": track.count_t, "merged_into": track.merged_into,
            })
        return rows
