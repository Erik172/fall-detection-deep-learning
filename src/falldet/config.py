from dataclasses import dataclass
from typing import Optional

import yaml


@dataclass
class Config:
    # pose model
    model: str = 'yolo11m-pose.pt'
    det_conf: float = 0.4        # person detection confidence
    kp_conf: float = 0.3         # keypoints below this count as missing

    # fall score (rules.py)
    window_secs: float = 1.0     # look-back window for the score
    vel_dt: float = 0.2          # time step used to measure hip speed
    v_fall: float = 1.5          # downward hip speed (torso lengths/s) that counts as a fall
    angle_fall: float = 60.0     # torso tilt change (deg) that counts as tipping over
    aspect_jump: float = 0.8     # bbox width/height increase that counts as tipping over
    fall_thr: float = 0.5        # score that starts the FALLING state

    # event state machine (events.py)
    low_angle: float = 60.0      # torso tilt from vertical that counts as "on the ground"
    low_aspect: float = 1.0      # bbox wider than tall counts as "on the ground"
    fall_to_down_secs: float = 2.0
    down_secs: float = 3.0       # time on the ground before alerting
    get_up_secs: float = 1.0     # time upright needed to cancel a fall
    cooldown_secs: float = 30.0  # min time between alerts for the same person
    lost_secs: float = 10.0      # forget a track after this long unseen
    relink_secs: float = 1.0     # a new ID appearing this soon near a lost mid-fall track inherits it
    history_secs: float = 3.0

    # I/O
    camera_id: str = 'cam0'
    out_dir: str = 'falldet_out'
    webhook: Optional[str] = None
    reconnect_secs: float = 2.0


def load_config(path=None, **overrides):
    """Defaults <- YAML file <- non-None overrides. Unknown keys raise TypeError."""
    values = {}
    if path:
        with open(path, encoding='utf-8') as f:
            values = yaml.safe_load(f) or {}
    values.update({k: v for k, v in overrides.items() if v is not None})
    return Config(**values)
