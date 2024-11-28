import logging

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.detection.detection_result import DetectionResult


class ObjectDetector:
    """Performs object detection, segmentation, depth estimation, and tracking."""

    SUPPORTED_LABELS = {"person", "car", "bicycle"}

    def __init__(self, config: dict, calib_params: dict[str, np.ndarray]) -> None:
        """Initialize the ObjectDetector."""
        self.conf_threshold = config["detection"].get("confidence_threshold", 0.5)
        self.nms_threshold = config["detection"].get("nms_threshold", 0.4)
        self.logger = logging.getLogger("autonomous_perception.detection")
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(config)
        self.model.to(device)

        self.calib_params = calib_params

    def _load_model(self, config: dict) -> YOLO:
        """Load the YOLO model based on the provided configuration."""
        model_path = config["detection"]["classification"].get("model_path", "yolov8n-seg.pt")
        try:
            model = YOLO(model_path)
            self.logger.info(f"Loaded YOLO model from {model_path}")
            return model
        except Exception as e:
            self.logger.exception(f"Failed to load YOLO model from {model_path}: {e}")
            raise

    def detect(self, image: np.ndarray, camera: str) -> list[DetectionResult]:
        """
        Detect objects in the image and obtain segmentation masks.

        Args:
            image (np.ndarray): Input image.
            camera (str): Camera identifier ("left" or "right").

        Returns:
            List[DetectionResult]: List of detection results.

        """
        try:
            results = self.model(image)
        except Exception as e:
            self.logger.exception(f"Model inference failed: {e}")
            return []

        detections: list[DetectionResult] = []
        for result in results:
            boxes = result.boxes
            masks = result.masks

            # Check if masks are available and contain data
            if masks is not None and masks.xy is not None:
                mask_list = masks.xy
            else:
                # Initialize empty masks if not available
                mask_list = [np.array([]) for _ in boxes]

            # Handle cases where number of masks doesn't match number of boxes
            if len(mask_list) != len(boxes):
                self.logger.warning(
                    f"Number of masks ({len(mask_list)}) does not match number of boxes ({len(boxes)}). Filling missing masks with empty arrays.",
                )
                while len(mask_list) < len(boxes):
                    mask_list.append(np.array([]))

            for box, mask in zip(boxes, mask_list, strict=False):
                cls = int(box.cls.item())
                label = self.model.names[cls] if cls < len(self.model.names) else "unknown"
                if label.lower() not in self.SUPPORTED_LABELS:
                    continue
                confidence = box.conf.item()
                if confidence < self.conf_threshold:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                left, top, right, bottom = int(x1), int(y1), int(x2), int(y2)
                # Retrieve camera intrinsics
                if camera.lower() == "left":
                    fx = self.calib_params.get("P_rect_02")[0, 0]  # type: ignore
                    fy = self.calib_params.get("P_rect_02")[1, 1]  # type: ignore
                    cx = self.calib_params.get("P_rect_02")[0, 2]  # type: ignore
                    cy = self.calib_params.get("P_rect_02")[1, 2]  # type: ignore
                elif camera.lower() == "right":
                    fx = self.calib_params.get("P_rect_03")[0, 0]  # type: ignore
                    fy = self.calib_params.get("P_rect_03")[1, 1]  # type: ignore
                    cx = self.calib_params.get("P_rect_03")[0, 2]  # type: ignore
                    cy = self.calib_params.get("P_rect_03")[1, 2]  # type: ignore
                else:
                    self.logger.error(f"Unknown camera identifier: {camera}. Skipping detection.")
                    continue

                # Initialize DetectionResult with consistent bbox format
                detection = DetectionResult(
                    bbox=[left, top, right, bottom],
                    confidence=confidence,
                    class_id=cls,
                    label=label,
                    mask=mask.astype(int) if mask.size > 0 else np.array([]),
                    appearance_feature=self._extract_appearance_feature(
                        mask,
                        image,
                        [left, top, right, bottom],
                    ),
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                )
                detections.append(detection)
                self.logger.debug(
                    f"Detected {label} with confidence {confidence:.2f} at [{left}, {top}, {right}, {bottom}]",
                )
        return detections

    def _extract_appearance_feature(
        self,
        mask: np.ndarray,
        image: np.ndarray,
        bbox: list[float],
    ) -> np.ndarray:
        """Extract appearance features using the segmentation mask."""
        x1, y1, x2, y2 = map(int, bbox)
        if mask is None or mask.size == 0:
            return np.array([])

        # Ensure bounding box is within image boundaries
        img_h, img_w = image.shape[:2]
        x_end = min(x2, img_w)
        y_end = min(y2, img_h)
        x = max(x1, 0)
        y = max(y2, 0)
        # Extract the mask ROI corresponding to the bounding box
        mask_roi = mask[y:y_end, x:x_end]
        roi = image[y:y_end, x:x_end]
        if roi.size == 0 or mask_roi.size == 0:
            return np.array([])
        masked_roi = cv2.bitwise_and(roi, roi, mask=mask_roi.astype(np.uint8))
        hist = cv2.calcHist(
            [masked_roi],
            [0, 1, 2],
            mask_roi.astype(np.uint8),
            [8, 8, 8],
            [0, 256, 0, 256, 0, 256],
        )
        cv2.normalize(hist, hist)
        return hist.flatten()
