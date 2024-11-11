# src/visualization/visualizer.py
import cv2
import logging
from typing import Tuple, List
from src.detection.object_detector import DetectionResult
from src.classification.classifier import ClassificationResult

class Visualizer:
    def __init__(self):
        self.logger = logging.getLogger('autonomous_perception.visualization')

    def display(self, frame, tracked_objects: np.ndarray, classifications: List[ClassificationResult]):
        img_left, img_right = frame.images
        for obj, cls in zip(tracked_objects, classifications):
            x1, y1, x2, y2, obj_id = obj[:5].astype(int)
            cv2.rectangle(img_left, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"ID: {obj_id} Class: {cls.class_id} Conf: {cls.confidence:.2f}"
            cv2.putText(img_left, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow('Visualization - Left Camera', img_left)
        cv2.imshow('Visualization - Right Camera', img_right)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
