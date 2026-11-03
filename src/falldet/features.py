"""Per-frame pose features. Pure numpy, so live inference and future training share one code path."""
from typing import NamedTuple

import numpy as np

NOSE, L_SH, R_SH, L_HIP, R_HIP = 0, 5, 6, 11, 12
NAN = float('nan')


class Obs(NamedTuple):
    t: float           # seconds
    hip_y: float       # image y of the hip center (px, grows downward)
    angle: float       # torso tilt from vertical in degrees: 0 upright, 90 horizontal
    aspect: float      # bbox width / height
    torso_len: float   # shoulder center to hip center (px)


def _mid(xy, conf, a, b, thr):
    ok = conf[[a, b]] >= thr
    return xy[[a, b]][ok].mean(axis=0) if ok.any() else None


def observe(t, xy, conf, bbox, thr=0.3):
    """One person in one frame -> Obs. Missing parts become nan."""
    hip = _mid(xy, conf, L_HIP, R_HIP, thr)
    sh = _mid(xy, conf, L_SH, R_SH, thr)
    angle = torso_len = NAN
    if hip is not None and sh is not None:
        v = sh - hip
        torso_len = float(np.hypot(*v))
        angle = float(np.degrees(np.arctan2(abs(v[0]), -v[1])))  # >90 means head below hips
    x1, y1, x2, y2 = bbox
    aspect = float((x2 - x1) / (y2 - y1)) if y2 > y1 else NAN
    return Obs(t, float(hip[1]) if hip is not None else NAN, angle, aspect, torso_len)


def normalize(xy, conf, scale, thr=0.3):
    """Body-relative keypoints: centered on the hips, in torso lengths. Missing -> nan.
    Independent of resolution and position in the frame (input format for the phase-2 model)."""
    out = np.full(xy.shape, np.nan, dtype=np.float32)
    center = _mid(xy, conf, L_HIP, R_HIP, thr)
    if center is None:
        center = _mid(xy, conf, L_SH, R_SH, thr)
    if center is None or not scale or not np.isfinite(scale):
        return out
    ok = conf >= thr
    out[ok] = (xy[ok] - center) / scale
    return out


def is_low(o, low_angle, low_aspect):
    """True if the person looks on the ground, False if upright, None if unknown."""
    # Torso angle wins when visible: bbox shape alone flags people sitting on the floor with legs stretched out.
    # shortcut: a person lying along the camera axis looks short and upright; needs depth or a learned model (phase 2)
    if np.isfinite(o.angle):
        return bool(o.angle >= low_angle)
    if np.isfinite(o.aspect):
        return bool(o.aspect >= low_aspect)
    return None
