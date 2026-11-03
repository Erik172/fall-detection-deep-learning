"""Per-person fall logic: keypoints over time -> alerts. No video or model code, so it is unit-testable."""
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .features import observe, is_low
from .rules import fall_score

NORMAL, FALLING, DOWN, ALERTED = 'NORMAL', 'FALLING', 'DOWN', 'ALERTED'


@dataclass
class Track:
    obs: deque = field(default_factory=deque)
    upright_torso: deque = field(default_factory=lambda: deque(maxlen=90))  # stable scale: foreshortening shrinks it when lying
    state: str = NORMAL
    t_state: float = 0.0
    t_up: float = None          # when the person started looking upright again
    last_seen: float = 0.0
    bbox: tuple = None
    last_alert: float = float('-inf')
    score: float = 0.0
    peak: float = 0.0


class FallLogic:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tracks = {}

    def step(self, t, detections):
        """detections: iterable of (track_id, xy[17,2], conf[17], bbox xyxy) for one frame at time t (s).
        Returns [(track_id, peak_score)] for alerts fired at this frame."""
        c, alerts, seen = self.cfg, [], set()
        detections = list(detections)
        ids_now = {d[0] for d in detections}
        for tid, xy, conf, bbox in detections:
            seen.add(tid)
            if tid not in self.tracks:
                self._relink(tid, bbox, t, ids_now)
            tr = self.tracks.setdefault(tid, Track())
            tr.bbox = tuple(bbox)
            o = observe(t, xy, conf, bbox, c.kp_conf)
            tr.obs.append(o)
            while t - tr.obs[0].t > c.history_secs:
                tr.obs.popleft()
            if np.isfinite(o.torso_len) and o.angle < 30:
                tr.upright_torso.append(o.torso_len)
            tr.score = fall_score(tr.obs, self._scale(tr), c)
            tr.last_seen = t
            if self._advance(tr, t, tr.score, is_low(o, c.low_angle, c.low_aspect)):
                alerts.append((tid, tr.peak))

        for tid, tr in list(self.tracks.items()):
            if tid in seen:
                continue
            if t - tr.last_seen > c.lost_secs:
                del self.tracks[tid]
            # unseen: posture unknown. A person lost by the pose model while on the floor still alerts.
            elif self._advance(tr, t, 0.0, None):
                alerts.append((tid, tr.peak))
        return alerts

    def _relink(self, new_id, bbox, t, ids_now):
        """The tracker often switches ID when a falling person's box goes from tall to wide.
        Hand a recently lost mid-fall track over to a new ID that appears where it was."""
        cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        best, best_d = None, float('inf')
        for old_id, tr in self.tracks.items():
            if old_id in ids_now or tr.state == NORMAL or tr.bbox is None or t - tr.last_seen > self.cfg.relink_secs:
                continue
            x1, y1, x2, y2 = tr.bbox
            d = np.hypot(cx - (x1 + x2) / 2, cy - (y1 + y2) / 2)
            if d <= max(x2 - x1, y2 - y1) and d < best_d:  # within one body length of where it was lost
                best, best_d = old_id, d
        if best is not None:
            self.tracks[new_id] = self.tracks.pop(best)

    @staticmethod
    def _scale(tr):
        if tr.upright_torso:
            return float(np.median(tr.upright_torso))
        lens = [o.torso_len for o in tr.obs if np.isfinite(o.torso_len)]
        return float(np.median(lens)) if lens else float('nan')

    def _advance(self, tr, t, score, low):
        c = self.cfg
        if tr.state == NORMAL:
            if score >= c.fall_thr:
                tr.state, tr.t_state, tr.peak = FALLING, t, score
        elif tr.state == FALLING:
            tr.peak = max(tr.peak, score)
            if low:
                tr.state, tr.t_state, tr.t_up = DOWN, t, None
            elif t - tr.t_state > c.fall_to_down_secs:
                tr.state = NORMAL
        else:  # DOWN or ALERTED
            if low is False:
                tr.t_up = t if tr.t_up is None else tr.t_up
                if t - tr.t_up >= c.get_up_secs:  # hysteresis: one noisy upright frame must not cancel
                    tr.state = NORMAL
                return False
            tr.t_up = None
            if tr.state == DOWN and t - tr.t_state >= c.down_secs:
                tr.state = ALERTED
                if t - tr.last_alert >= c.cooldown_secs:
                    tr.last_alert = t
                    return True
        return False
