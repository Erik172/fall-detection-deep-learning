import json
import logging
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import cv2

log = logging.getLogger(__name__)


class AlertSink:
    """Writes each fall event to <out_dir>/events.jsonl + a snapshot, and POSTs it to an optional webhook."""

    def __init__(self, out_dir, webhook=None, camera_id='cam0'):
        if webhook and urlparse(webhook).scheme not in ('http', 'https'):
            raise ValueError(f'webhook must be an http(s) URL, got {webhook!r}')
        self.dir = Path(out_dir)
        (self.dir / 'snapshots').mkdir(parents=True, exist_ok=True)
        self.webhook, self.camera_id = webhook, camera_id

    def emit(self, track_id, t, score, frame, bbox):
        snap = self.dir / 'snapshots' / f'{self.camera_id}_{track_id}_{int(time.time() * 1000)}.jpg'
        img = frame.copy()
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.imwrite(str(snap), img)
        event = {
            'event': 'fall', 'camera_id': self.camera_id, 'track_id': int(track_id),
            'ts': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'video_t': round(float(t), 2), 'score': round(float(score), 3), 'snapshot': str(snap),
        }
        with open(self.dir / 'events.jsonl', 'a', encoding='utf-8') as f:
            f.write(json.dumps(event) + '\n')
        log.warning('FALL camera=%s track=%s t=%.1fs score=%.2f', self.camera_id, track_id, t, score)
        if self.webhook:
            # background thread so a slow endpoint never stalls video; non-daemon so it finishes on exit
            threading.Thread(target=self._post, args=(event,)).start()
        return event

    def _post(self, event):
        req = urllib.request.Request(self.webhook, data=json.dumps(event).encode(),
                                     headers={'Content-Type': 'application/json'}, method='POST')
        for attempt in range(2):
            try:
                urllib.request.urlopen(req, timeout=5).close()
                return
            except Exception as e:  # network errors must not kill the detector; the event is already in events.jsonl
                err = e
        log.error('webhook POST failed after 2 attempts: %s', err)
