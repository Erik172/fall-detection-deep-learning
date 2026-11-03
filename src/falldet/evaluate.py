"""Event-level evaluation on labelled videos.

labels.csv columns: video (path relative to the CSV), has_fall (1/0), fall_time_s (optional, seconds into the video)

python -m src.falldet.evaluate --labels data/videos/urfall/labels.csv [--model yolo11s-pose.pt]
"""
import argparse
import csv
import logging
from pathlib import Path

import numpy as np

from .config import load_config
from .pipeline import FallDetector

HIT_BEFORE, HIT_AFTER = 1.0, 10.0  # an alert in [fall - 1 s, fall + 10 s] counts as detecting that fall


def score_video(has_fall, fall_t, alert_ts):
    """-> (hit, delay or None, false_alarms) for one video."""
    if not has_fall:
        return False, None, len(alert_ts)
    if fall_t is None:  # no timestamp: any alert is a hit, extras are false alarms
        return bool(alert_ts), None, max(len(alert_ts) - 1, 0)
    inside = [a for a in alert_ts if fall_t - HIT_BEFORE <= a <= fall_t + HIT_AFTER]
    return bool(inside), (inside[0] - fall_t if inside else None), len(alert_ts) - len(inside)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', required=True)
    ap.add_argument('--config')
    ap.add_argument('--model')
    ap.add_argument('--out', default='falldet_eval')
    a = ap.parse_args()
    logging.basicConfig(level=logging.ERROR)

    cfg = load_config(a.config, model=a.model, out_dir=a.out)
    base = Path(a.labels).parent
    with open(a.labels, newline='', encoding='utf-8-sig') as f:  # -sig: Excel/PowerShell CSVs start with a BOM
        rows = list(csv.DictReader(f))

    hits, n_falls, fas, delays, dur, frames, pose, fps = 0, 0, 0, [], 0.0, 0, 0, []
    for row in rows:
        has_fall = row['has_fall'].strip() == '1'
        fall_t = float(row['fall_time_s']) if row.get('fall_time_s', '').strip() else None
        stats = FallDetector(cfg).run(str(base / row['video']))  # fresh detector = fresh tracker per video
        alert_ts = [e['video_t'] for e in stats['events']]
        hit, delay, fa = score_video(has_fall, fall_t, alert_ts)
        n_falls += has_fall
        hits += hit
        fas += fa
        delays += [delay] if delay is not None else []
        dur += stats['duration_s']
        frames += stats['frames']
        pose += stats['pose_frames']
        fps.append(stats['proc_fps'])
        tag = ('HIT' if hit else 'MISS') if has_fall else ('FA' if fa else 'ok')
        print(f"{tag:4s} {row['video']}: alerts at {alert_ts} (fall_t={fall_t}) "
              f"pose={stats['pose_frames'] / max(stats['frames'], 1):.0%}")

    hours = dur / 3600
    print('\n=== summary ===')
    print(f'videos: {len(rows)} ({n_falls} with falls), footage {dur / 60:.1f} min')
    print(f'recall: {hits}/{n_falls} = {hits / max(n_falls, 1):.2f}')
    print(f'false alarms: {fas} -> {fas / hours if hours else 0:.1f} per hour of footage')
    if delays:
        print(f'alert delay after fall: mean {np.mean(delays):.1f}s, p90 {np.percentile(delays, 90):.1f}s')
    print(f'pose coverage: {pose / max(frames, 1):.0%} of frames | processing speed: {np.mean(fps):.1f} FPS')


if __name__ == '__main__':
    main()
