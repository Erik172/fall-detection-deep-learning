# falldet: fall detection module

Pose (YOLO11-pose) → per-person tracking (ByteTrack) → fall rules → alert state machine → JSONL + snapshot + webhook.

## Run

```bash
# file, webcam (0) or RTSP stream
python -m src.falldet --source video.mp4 --show
python -m src.falldet --source rtsp://user:pass@cam/stream --camera-id hall --webhook https://example/fall
python -m src.falldet --source video.mp4 --record annotated.mp4 --config site.yaml
```

Each fall produces one line in `falldet_out/events.jsonl` and a snapshot in `falldet_out/snapshots/`.
With `--webhook`, the same JSON is POSTed:

```json
{"event": "fall", "camera_id": "hall", "track_id": 3, "ts": "2026-11-03T08:09:34+00:00",
 "video_t": 12.4, "score": 0.91, "snapshot": "falldet_out/snapshots/hall_3_1793693374595.jpg"}
```

## Live test with your phone (IP Webcam)

This uses the same setup as the CAD project: the phone streams video and the laptop does the detection on its GPU.

1. **Phone:** install the **IP Webcam** app. Under *Video preferences*, set the resolution to **1280x720** and the quality to **~50**. Tap **Start server** and note the IP address it shows, for example `192.168.1.35`.
2. **Network:** connect the phone and the laptop to the **same Wi-Fi** and turn the phone's **mobile data off**. If the Wi-Fi isolates devices from each other, use the phone's own hotspot instead. The phone is then usually `192.168.43.1`.
3. **Check:** `http://<phone-ip>:8080` must load in the laptop's browser. If it doesn't, nothing below will work.
4. **Run** (from the repo root, with the shared venv):
   ```bash
   python -m src.falldet --ip-camera 192.168.1.35 --show --record live.mp4
   ```
   Stop with `q` in the window, or Ctrl+C. Alerts go to `falldet_out/events.jsonl` and `falldet_out/snapshots/`, as with files.
5. **Staging a fall:** put the phone at chest height or higher, pointing at an open floor area. Fall onto a **mattress** and **stay down for at least 3 s**. The box turns yellow (falling), then orange (down), then red (alert).

Notes:
- A real IP or CCTV camera works with `--source "rtsp://user:pass@ip:554/..."`. Passwords are masked in logs. A laptop webcam works with `--source 0`.
- Live input always processes the **newest** frame and drops frames if it falls behind, so alerts never lag behind reality. If the stream drops, it reconnects on its own.
- Troubleshooting:
  - "cannot open … retrying": step 3 fails, so check the IP, the Wi-Fi and that the app's server is running.
  - Laggy video: lower the resolution in the app.
  - No alert: you got up too soon (it needs at least 3 s down), or the pose wasn't visible. Make sure your whole body is in frame.
  - For quicker demos, put `down_secs: 1.5` in a YAML file and pass it with `--config`.

## How a fall is decided (per person)

`NORMAL → FALLING → DOWN → ALERT`

1. **FALLING**: hips drop fast (in torso lengths/s, so camera resolution and distance don't matter) while the body tips over.
2. **DOWN**: within 2 s, the torso is past 60° from vertical. If the torso is not visible, the bounding box being wider than tall counts instead.
3. **ALERT**: still down after `down_secs` (3 s). Standing up for 1 s cancels it. Each person has a 30 s cooldown.

Other behavior:
- A person lost by the pose model while on the floor still alerts.
- A tracker ID switch during the fall is re-linked to the same person.
- Lying down slowly, sitting, and bending don't alert, because there is no fast drop.

All thresholds are in `config.py`. Override them with a YAML file passed through `--config`.

## Evaluate

```bash
python -m src.falldet.evaluate --labels data/videos/urfall/labels.csv
```

`labels.csv` has the columns `video,has_fall,fall_time_s`. Video paths are relative to the CSV, and `fall_time_s` is optional. The report gives:
- event recall
- false alarms per hour
- alert delay
- pose coverage (low coverage means the camera angle doesn't suit pose estimation)
- processing FPS

## Tests

```bash
python -m pytest tests/test_falldet.py
```

These use synthetic skeletons, so no video or GPU is needed.

## Known limits (phase 1)
- Rule thresholds are untuned. Tune them on UR Fall, Le2i and client footage.
- A person lying along the camera's line of sight can look upright. Phase 2 (a learned model on `features.normalize`) addresses this.
- Ceiling or fisheye cameras: check pose coverage first.
