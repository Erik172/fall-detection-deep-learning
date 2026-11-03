"""Rule-based fall score. Phase 2 replaces this with a learned model behind the same signature."""
import numpy as np


def fall_score(obs, scale, cfg):
    """obs: recent Obs, oldest first. scale: torso length in px.
    Returns 0..1, high when the hips drop fast AND the body tips over."""
    if not scale or not np.isfinite(scale) or len(obs) < 3:
        return 0.0
    now = obs[-1].t
    win = [o for o in obs if now - o.t <= cfg.window_secs]
    t = np.array([o.t for o in win])
    y = np.array([o.hip_y for o in win])

    # downward hip speed over ~vel_dt, in torso lengths per second (resolution and FPS independent)
    j = np.searchsorted(t, t - cfg.vel_dt, side='right') - 1
    i = np.nonzero(j >= 0)[0]
    v = (y[i] - y[j[i]]) / (t[i] - t[j[i]]) / scale
    v = np.nanmax(v) if np.isfinite(v).any() else 0.0

    tip = max(_rise(np.array([o.angle for o in win])) / cfg.angle_fall,
              _rise(np.array([o.aspect for o in win])) / cfg.aspect_jump)
    v_term = float(np.clip(v / cfg.v_fall, 0, 1))
    return v_term * (0.5 + 0.5 * float(np.clip(tip, 0, 1)))


def _rise(x):
    """Latest value minus the window minimum (0 if unknown)."""
    if not np.isfinite(x[-1]) or not np.isfinite(x).any():
        return 0.0
    return float(x[-1] - np.nanmin(x))
