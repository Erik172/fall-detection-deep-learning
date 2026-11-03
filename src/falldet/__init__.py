"""Fall detection module: pose -> per-person tracking -> fall rules -> event alerts.
Import FallDetector from src.falldet.pipeline (kept out of here so tests don't load YOLO)."""
from .config import Config, load_config
from .events import FallLogic
