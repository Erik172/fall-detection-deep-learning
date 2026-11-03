"""Synthetic-skeleton tests for the fall logic (no video, no YOLO)."""
import time

import numpy as np
import pytest

from src.falldet import Config, FallLogic
from src.falldet.camera import LatestFrameReader, ip_camera_url, mask_url, parse_source
from src.falldet.features import normalize

FPS = 30
# distance of each COCO keypoint from the feet along the body axis (torso lengths), and sideways offset
ALONG = [3.2, 3.25, 3.25, 3.2, 3.2, 2.5, 2.5, 2.0, 2.0, 1.6, 1.6, 1.5, 1.5, 0.75, 0.75, 0.0, 0.0]
SIDE = [0, -.1, .1, -.15, .15, -.2, .2, -.25, .25, -.25, .25, -.15, .15, -.15, .15, -.15, .15]


def skeleton(theta_deg, L=100.0, foot=(320.0, 400.0)):
    """Stick person rotated about the feet by theta (0 upright, 90 lying). Torso length = L px."""
    th = np.radians(theta_deg)
    d, n = np.array([np.sin(th), -np.cos(th)]), np.array([np.cos(th), np.sin(th)])
    xy = np.array([foot]) + L * (np.array(ALONG)[:, None] * d + np.array(SIDE)[:, None] * n)
    bbox = (*xy.min(0) - 5, *xy.max(0) + 5)
    return xy, np.ones(17), bbox


def ramp(t, t0, t1, a, b):
    return a + (b - a) * np.clip((t - t0) / (t1 - t0), 0, 1)


def fall(t):          # stands 1 s, falls in 0.5 s, stays down
    return ramp(t, 1.0, 1.5, 0, 90)


def run(people, duration, cfg=None):
    """people: {track_id: (angle_fn, L, foot)}; angle_fn returning None = person not detected."""
    logic, alerts = FallLogic(cfg or Config()), []
    for k in range(int(duration * FPS)):
        t = k / FPS
        dets = []
        for tid, (fn, L, foot) in people.items():
            a = fn(t)
            if a is not None:
                dets.append((tid, *skeleton(a, L, foot)))
        alerts += [(tid, t) for tid, _ in logic.step(t, dets)]
    return alerts


def test_fall_alerts_once_after_down_secs():
    alerts = run({1: (fall, 100, (320, 400))}, 10)
    assert len(alerts) == 1
    tid, t = alerts[0]
    assert tid == 1 and 4.0 <= t <= 5.0  # ~1.33 s on the ground + 3 s down_secs


def test_lying_down_slowly_does_not_alert():
    assert run({1: (lambda t: ramp(t, 1, 5, 0, 90), 100, (320, 400))}, 12) == []


def test_quick_lean_without_lying_does_not_alert():
    # fast 45 deg lean held 5 s: bbox gets wide, but the torso never reaches "on the ground"
    assert run({1: (lambda t: ramp(t, 1, 1.5, 0, 45), 100, (320, 400))}, 10) == []


def test_standing_still_does_not_alert():
    assert run({1: (lambda t: 0.0, 100, (320, 400))}, 10) == []


def test_fall_then_getting_up_does_not_alert():
    def stumble(t):
        return ramp(t, 1.0, 1.5, 0, 90) if t < 2.5 else ramp(t, 2.5, 3.0, 90, 0)
    assert run({1: (stumble, 100, (320, 400))}, 10) == []


@pytest.mark.parametrize('L,foot', [(40, (100, 150)), (250, (900, 1000))])
def test_alert_timing_is_resolution_and_position_independent(L, foot):
    ref = run({1: (fall, 100, (320, 400))}, 10)
    got = run({1: (fall, L, foot)}, 10)
    assert len(got) == 1 and abs(got[0][1] - ref[0][1]) <= 2 / FPS


def test_tracks_are_independent():
    alerts = run({1: (fall, 100, (200, 400)), 2: (lambda t: 0.0, 100, (500, 400))}, 10)
    assert [tid for tid, _ in alerts] == [1]


def test_person_lost_on_the_floor_still_alerts():
    # pose model loses the person 0.5 s after they hit the ground
    alerts = run({1: (lambda t: fall(t) if t < 2.0 else None, 100, (320, 400))}, 10)
    assert len(alerts) == 1


def test_tracker_id_switch_mid_fall_still_alerts_once():
    # ByteTrack drops ID 1 as the box turns wide and starts ID 2 for the same person
    alerts = run({1: (lambda t: fall(t) if t < 1.4 else None, 100, (320, 400)),
                  2: (lambda t: fall(t) if t >= 1.4 else None, 100, (320, 400))}, 10)
    assert [tid for tid, _ in alerts] == [2]


def test_new_person_far_away_does_not_inherit_a_fall():
    alerts = run({1: (lambda t: fall(t) if t < 1.4 else None, 100, (100, 400)),
                  2: (lambda t: 90.0 if t >= 1.4 else None, 100, (1500, 400))}, 10)
    assert [tid for tid, _ in alerts] == [1]  # 1 was lost while on the ground; 2 never fell


def test_cooldown_blocks_repeat_alerts():
    def two_falls(t):  # falls, gets up at 6 s, falls again at 8 s
        if t < 6:
            return fall(t)
        return ramp(t, 6, 6.5, 90, 0) if t < 8 else ramp(t, 8, 8.5, 0, 90)
    assert len(run({1: (two_falls, 100, (320, 400))}, 15)) == 1
    assert len(run({1: (two_falls, 100, (320, 400))}, 15, Config(cooldown_secs=1))) == 2


@pytest.mark.parametrize('given,expected', [
    ('192.168.1.5', 'http://192.168.1.5:8080/video'),
    (' 192.168.1.5:8081 ', 'http://192.168.1.5:8081/video'),
    ('http://10.0.0.2:8080/video', 'http://10.0.0.2:8080/video'),
])
def test_ip_camera_url(given, expected):
    assert ip_camera_url(given) == expected


def test_mask_url_and_source_kinds():
    assert mask_url('rtsp://admin:s3cret@10.0.0.9:554/ch1') == 'rtsp://***@10.0.0.9:554/ch1'
    assert parse_source('0') == (0, True)
    assert parse_source('http://1.2.3.4:8080/video')[1] and parse_source('rtsp://x/y')[1]
    assert parse_source('clip.mp4') == ('clip.mp4', False)


class FakeCap:
    """Yields numbered frames; read() fails once `fail_after` frames are served."""

    def __init__(self, counter, fail_after, delay=0.002):
        self.counter, self.left, self.delay = counter, fail_after, delay

    def isOpened(self):
        return True

    def read(self):
        time.sleep(self.delay)
        if self.left == 0:
            return False, None
        self.left -= 1
        self.counter[0] += 1
        return True, np.full((2, 2, 3), self.counter[0] % 256, np.uint8)

    def release(self):
        pass


def test_latest_frame_reader_skips_stale_frames_and_reconnects():
    counter, opens = [0], []

    def opener(spec):
        opens.append(spec)
        return FakeCap(counter, fail_after=20)  # stream "drops" every 20 frames

    LatestFrameReader.RECONNECT_MAX = 0.05
    reader = LatestFrameReader('http://cam/video', opener)
    try:
        ids, last = [], 0
        while counter[0] < 70:
            got = reader.latest(last, timeout=2)
            assert got is not None, 'reader stopped delivering frames'
            last = got[0]
            ids.append(last)
            time.sleep(0.02)  # slow consumer: the reader must drop frames, not queue them
        assert ids == sorted(set(ids)), 'a frame was repeated or went backwards'
        assert len(ids) < counter[0], 'no frames were skipped for the slow consumer'
        assert len(opens) >= 3, 'did not reconnect after the stream dropped'
    finally:
        reader.stop()
        LatestFrameReader.RECONNECT_MAX = 5.0


def test_normalize_is_scale_and_shift_invariant():
    xy, conf, _ = skeleton(20)
    a = normalize(xy, conf, 100.0)
    b = normalize(xy * 2.5 + [40, -70], conf, 250.0)
    np.testing.assert_allclose(a, b, atol=1e-5)
    assert np.isnan(normalize(xy, np.zeros(17), 100.0)).all()
