import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from config import config

_TRACKER_CONFIG = Path(__file__).parent / "tracker.yaml"

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    # Absolute pixel coordinates in the output frame
    bbox_xyxy: tuple[float, float, float, float]


class BoatDetector:
    def __init__(self) -> None:
        self._device = config.device
        if self._device == "mps" and not torch.backends.mps.is_available():
            logger.warning("MPS not available — falling back to CPU")
            self._device = "cpu"

        self.model = YOLO(config.model_name)
        logger.info("Loaded %s on device=%s", config.model_name, self._device)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.track(
            frame,
            persist=True,
            device=self._device,
            conf=config.confidence_threshold,
            classes=config.target_classes,
            tracker=str(_TRACKER_CONFIG),
            verbose=False,
        )

        detections: list[Detection] = []
        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            return detections

        for box, track_id, cls_id, conf in zip(
            boxes.xyxy.cpu().numpy(),
            boxes.id.cpu().numpy().astype(int),
            boxes.cls.cpu().numpy().astype(int),
            boxes.conf.cpu().numpy(),
        ):
            detections.append(
                Detection(
                    track_id=int(track_id),
                    class_id=int(cls_id),
                    class_name=self.model.names[int(cls_id)],
                    confidence=float(conf),
                    bbox_xyxy=tuple(float(v) for v in box),
                )
            )

        return detections
