import logging

import numpy as np
from sort.tracker import SortTracker

from src.detection.object_detector import DetectionResult


class ObjectTracker:
    def __init__(self, config):
        self.algorithm = config["algorithm"]
        self.max_age = config["max_age"]
        self.min_hits = config["min_hits"]
        self.logger = logging.getLogger("autonomous_perception.tracking")
        self.tracker = SortTracker(
            max_age=self.max_age, min_hits=self.min_hits
        )  # Initialize SORT tracker

    def update(self, detections: list[DetectionResult]):
        dets = np.array([det.bbox + [det.confidence] for det in detections])
        tracked_objects = self.tracker.update(dets, None)
        # Convert tracked_objects to desired format
        self.logger.debug(f"Tracked objects count: {len(tracked_objects)}")
        return tracked_objects
