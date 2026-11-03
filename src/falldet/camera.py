"""Frame sources: video files (exact, replayable) and live cameras (webcam, IP camera, RTSP).

A live camera must never queue frames: if processing is slower than the camera, a queue makes the alarm act on
ever older video. LatestFrameReader keeps only the newest frame (same idea as cad_interpretation/cadrec/camera.py).
"""
import logging
import re
import sys
import threading
import time

import cv2

log = logging.getLogger(__name__)
URL_SCHEMES = ('http://', 'https://', 'rtsp://', 'rtsps://', 'rtmp://', 'udp://', 'tcp://')


def mask_url(source):
    """Hide credentials before logging: rtsp://user:pass@host -> rtsp://***@host."""
    return re.sub(r'(://)[^/@\s]+@', r'\1***@', str(source))


def ip_camera_url(host):
    """IP Webcam (Android app) shortcut: '192.168.1.5' -> 'http://192.168.1.5:8080/video'. Full URLs pass through."""
    host = host.strip().rstrip('/')
    if '://' in host:
        return host
    return f"http://{host if ':' in host else host + ':8080'}/video"


def parse_source(source):
    """'0' -> webcam 0. Returns (spec, is_live)."""
    spec = int(source) if str(source).strip().isdigit() else source
    return spec, isinstance(spec, int) or str(spec).lower().startswith(URL_SCHEMES)


def open_capture(spec):
    if isinstance(spec, int):
        # DirectShow + MJPG: the Windows default backend opens slowly and gives ~10 fps at 720p on many webcams
        cap = cv2.VideoCapture(spec, cv2.CAP_DSHOW if sys.platform == 'win32' else cv2.CAP_ANY)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    else:
        cap = cv2.VideoCapture(str(spec))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep the driver's own buffer small too
    return cap


class LatestFrameReader:
    """Reads a live source in a background thread, keeps only the newest frame, reconnects forever."""

    RECONNECT_MAX = 5.0

    def __init__(self, spec, opener=open_capture):
        self.spec, self._open = spec, opener
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._id, self._frame, self._t = 0, None, 0.0
        self._thread = threading.Thread(target=self._run, name='falldet-camera', daemon=True)
        self._thread.start()

    def _run(self):
        cap, backoff, failing = None, 0.5, False
        while not self._stop.is_set():
            if cap is None:
                cap = self._open(self.spec)
                if not cap.isOpened():
                    cap.release()
                    cap = None
                    if not failing:
                        log.error('cannot open %s - retrying until it answers', mask_url(self.spec))
                        failing = True
                    self._stop.wait(backoff)
                    backoff = min(backoff * 2, self.RECONNECT_MAX)
                    continue
            ok, frame = cap.read()
            if not ok or frame is None:
                if not failing:
                    log.error('lost %s - reconnecting', mask_url(self.spec))
                    failing = True
                cap.release()
                cap = None
                self._stop.wait(backoff)
                backoff = min(backoff * 2, self.RECONNECT_MAX)
                continue
            if failing:
                log.info('connected to %s', mask_url(self.spec))
                failing = False
            backoff = 0.5
            with self._cond:
                self._id, self._frame, self._t = self._id + 1, frame, time.monotonic()
                self._cond.notify_all()
        if cap is not None:
            cap.release()

    def latest(self, after_id=0, timeout=1.0):
        """(id, frame, capture_time) of the newest frame newer than after_id, or None after timeout."""
        with self._cond:
            self._cond.wait_for(lambda: self._id > after_id or self._stop.is_set(), timeout)
            return (self._id, self._frame, self._t) if self._id > after_id else None

    def stop(self):
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout=2)


def file_frames(spec):
    """Every frame of a video file with its exact video time: (frame, t)."""
    cap = open_capture(spec)
    if not cap.isOpened():
        raise IOError(f'cannot open video source {mask_url(spec)!r}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            yield frame, n / fps
            n += 1
    finally:
        cap.release()


def live_frames(spec, opener=open_capture):
    """Newest frames of a live source, skipping any that arrived while we were busy: (frame, capture_time)."""
    reader, last, t0 = LatestFrameReader(spec, opener), 0, None
    try:
        while True:
            got = reader.latest(last)
            if got is None:
                continue  # camera silent; the reader is reconnecting
            last, frame, t = got
            t0 = t if t0 is None else t0
            yield frame, t - t0
    finally:
        reader.stop()
