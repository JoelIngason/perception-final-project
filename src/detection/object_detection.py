# src/detection/object_detector.py
import torch
import cv2
import numpy as np
import logging
from typing import List, Tuple

class DetectionResult:
    def __init__(self, bbox: List[float], confidence: float, class_id: int):
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.confidence = confidence
        self.class_id = class_id

class ObjectDetector:
    def __init__(self, config):
        self.model_name = config['model']
        self.conf_threshold = config['confidence_threshold']
        self.nms_threshold = config['nms_threshold']
        self.logger = logging.getLogger('autonomous_perception.detection')
        self.model = self._load_model()

    def _load_model(self):
        self.logger.info(f"Loading detection model: {self.model_name}")
        if self.model_name.lower() == 'yolov5':
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            model.conf = self.conf_threshold
            model.iou = self.nms_threshold
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        self.logger.info("Detection model loaded successfully")
        return model

    def detect(self, images: Tuple[np.ndarray, np.ndarray]) -> List[DetectionResult]:
        # Perform detection on the left rectified image
        img_left, _ = images
        self.logger.debug("Performing detection on left image")
        results = self.model(img_left)
        detections = self._process_results(results)
        return detections

    def _process_results(self, results) -> List[DetectionResult]:
        detections = []
        for *box, conf, cls in results.xyxy[0].tolist():
            if conf >= self.conf_threshold:
                detections.append(DetectionResult(bbox=box, confidence=conf, class_id=int(cls)))
        self.logger.debug(f"Detections found: {len(detections)}")
        return detections
