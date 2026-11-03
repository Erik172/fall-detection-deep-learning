"""python -m src.falldet (--source <video|rtsp://...|http://...|0> | --ip-camera <phone-ip>) [--show] [--webhook URL]"""
import argparse
import logging

from .camera import ip_camera_url, mask_url, parse_source
from .config import load_config
from .pipeline import FallDetector


def main():
    ap = argparse.ArgumentParser(description='Detect falls in a video file, webcam, IP camera or RTSP stream.')
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--source', help='video path, stream URL (rtsp://, http://), or webcam index (0)')
    src.add_argument('--ip-camera', metavar='HOST[:PORT]',
                     help='phone running the IP Webcam app, e.g. 192.168.1.35 (-> http://HOST:8080/video)')
    ap.add_argument('--config', help='YAML file overriding Config defaults')
    ap.add_argument('--model', help='YOLO pose weights, e.g. yolo11s-pose.pt')
    ap.add_argument('--webhook', help='http(s) URL that receives each fall event as JSON')
    ap.add_argument('--camera-id')
    ap.add_argument('--out', help='output folder for events.jsonl and snapshots')
    ap.add_argument('--show', action='store_true', help='display the annotated video')
    ap.add_argument('--record', help='save the annotated video to this .mp4')
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    source = ip_camera_url(a.ip_camera) if a.ip_camera else a.source
    if parse_source(source)[1]:
        print(f'live source: {mask_url(source)} - stop with q in the window or Ctrl+C')
    cfg = load_config(a.config, model=a.model, webhook=a.webhook, camera_id=a.camera_id, out_dir=a.out)
    stats = FallDetector(cfg).run(source, show=a.show, record=a.record)
    print(f"frames={stats['frames']} falls={len(stats['events'])} proc_fps={stats['proc_fps']:.1f} "
          f"pose_coverage={stats['pose_frames'] / max(stats['frames'], 1):.0%} -> {cfg.out_dir}/events.jsonl")


if __name__ == '__main__':
    main()
