import logging
import time

import cv2
import numpy as np
from ultralytics import YOLO

from src.utils.body import BODY_CONNECTIONS
from .alerts import AlertSink
from .camera import parse_source, file_frames, live_frames
from .events import FallLogic, NORMAL, FALLING, DOWN

log = logging.getLogger(__name__)
COLORS = {NORMAL: (0, 200, 0), FALLING: (0, 220, 255), DOWN: (0, 140, 255)}  # anything else (ALERTED) is red
MIN_KP = 8          # confident keypoints needed to count a frame as "pose visible"
FPS_PROBE = 30      # frames used to measure the recording frame rate


class FallDetector:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = YOLO(cfg.model)
        # warm up (CUDA init, ~2 s) before the camera starts, or a live stream loses its first seconds
        self.model.track(np.zeros((480, 640, 3), np.uint8), persist=True, tracker='bytetrack.yaml', verbose=False)
        self.logic = FallLogic(cfg)
        self.sink = AlertSink(cfg.out_dir, cfg.webhook, cfg.camera_id)

    def process(self, frame, t):
        """One frame -> (detections, alert events)."""
        r = self.model.track(frame, persist=True, tracker='bytetrack.yaml', conf=self.cfg.det_conf, verbose=False)[0]
        dets = []
        if r.boxes is not None and r.boxes.id is not None and r.keypoints is not None:
            xy = r.keypoints.xy.cpu().numpy()
            conf = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else np.ones(xy.shape[:2])
            dets = list(zip(r.boxes.id.int().tolist(), xy, conf, r.boxes.xyxy.cpu().numpy()))
        boxes = {tid: bbox for tid, _, _, bbox in dets}
        events = []
        for tid, score in self.logic.step(t, dets):
            # a track lost on the floor has no box this frame: snapshot the whole frame
            bbox = boxes.get(tid, (0, 0, frame.shape[1] - 1, frame.shape[0] - 1))
            events.append(self.sink.emit(tid, t, score, frame, bbox))
        return dets, events

    def draw(self, frame, dets):
        thr = self.cfg.kp_conf
        for tid, xy, conf, bbox in dets:
            tr = self.logic.tracks.get(tid)
            state = tr.state if tr else NORMAL
            color = COLORS.get(state, (0, 0, 255))
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'#{tid} {state} {tr.score if tr else 0:.2f}', (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            for a, b in BODY_CONNECTIONS:
                if conf[a] >= thr and conf[b] >= thr:
                    cv2.line(frame, tuple(map(int, xy[a])), tuple(map(int, xy[b])), color, 2)
        return frame

    def run(self, source, show=False, record=None, max_secs=None):
        """Process a file, webcam, IP camera or stream until it ends, 'q' is pressed, Ctrl+C, or max_secs.
        Files: every frame, video time (exact, replayable). Live: newest frame only, capture time."""
        spec, live = parse_source(source)
        frames = live_frames(spec) if live else file_frames(spec)
        writer, pending, n, pose_frames, events, t = None, [], 0, 0, [], 0.0
        start = time.monotonic()
        try:
            for frame, t in frames:
                n += 1
                dets, ev = self.process(frame, t)
                events += ev
                pose_frames += any((c >= self.cfg.kp_conf).sum() >= MIN_KP for _, _, c, _ in dets)
                if show or record:
                    self.draw(frame, dets)
                if record:
                    writer = self._record(writer, pending, record, frame, t)
                if show:
                    cv2.imshow(f'falldet {self.cfg.camera_id} (q to quit)', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                if max_secs and t >= max_secs:
                    break
        except KeyboardInterrupt:
            log.info('stopped by user')
        finally:
            frames.close()  # stops the live reader thread / releases the file
            if record and writer is None and pending:  # short run: write what we have
                writer = self._record(writer, pending, record, None, t, force=True)
            if writer:
                writer.release()
            if show:
                cv2.destroyAllWindows()
        elapsed = time.monotonic() - start
        return {'frames': n, 'duration_s': t, 'pose_frames': pose_frames,
                'proc_fps': n / elapsed if elapsed else 0.0, 'events': events}

    @staticmethod
    def _record(writer, pending, path, frame, t, force=False):
        """Measure the real frame rate over the first frames, then open the writer.
        Live cameras report bogus FPS and drop frames, so the camera's own number would replay at the wrong speed."""
        if writer is not None:
            writer.write(frame)
            return writer
        if frame is not None:
            pending.append((frame.copy(), t))
        if len(pending) < FPS_PROBE and not force:
            return None
        # processed frames per second, skipping the first gap (one-off warm-up / connect pause)
        ts = [p[1] for p in pending[1:]]
        span = ts[-1] - ts[0] if len(ts) > 1 else 0.0
        fps = float(np.clip((len(ts) - 1) / span, 5, 60)) if span > 0 else 15.0
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, pending[0][0].shape[1::-1])
        for f, _ in pending:
            writer.write(f)
        pending.clear()
        return writer
