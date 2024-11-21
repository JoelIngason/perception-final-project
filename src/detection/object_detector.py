# src/detection/object_detector.py

import logging

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.data_loader.dataset import TrackedObject
from src.depth_estimation.depth_estimator import DepthEstimator
from src.detection.detection_result import DetectionResult
from src.tracking.tracker import ObjectTracker


class ObjectDetector:
    """Performs object detection, segmentation, depth estimation, and tracking."""

    SUPPORTED_LABELS = {"person", "car", "bicycle"}

    def __init__(self, config: dict, calib_params: dict[str, np.ndarray]) -> None:
        """Initialize the ObjectDetector."""
        self.conf_threshold = config["detection"].get("confidence_threshold", 0.5)
        self.nms_threshold = config["detection"].get("nms_threshold", 0.4)
        self.logger = logging.getLogger("autonomous_perception.detection")
        self.model = self._load_model(config)
        self.depth_estimator = DepthEstimator(calib_params, config)
        config["projection_matrix"] = calib_params["P_rect_02"]
        self.object_tracker = ObjectTracker(config, self.logger)
        self.calib_params = calib_params
        self.fx = self.calib_params["P_rect_02"][0, 0]
        self.fy = self.calib_params["P_rect_02"][1, 1]
        self.cx = self.calib_params["P_rect_02"][0, 2]
        self.cy = self.calib_params["P_rect_02"][1, 2]

        # Log calibration parameters
        self.logger.debug(
            f"Calibration Parameters - fx: {self.fx}, fy: {self.fy}, cx: {self.cx}, cy: {self.cy}"
        )

    def _load_model(self, config):
        model_path = config["detection"]["classification"].get("model_path", "yolov8n-seg.pt")
        try:
            model = YOLO(model_path)
            self.logger.info(f"Loaded model from {model_path}")
            return model
        except Exception as e:
            self.logger.exception(f"Failed to load model: {e}")
            raise

    def reset(self) -> None:
        """Reset the ObjectDetector state by reinitializing the tracker."""
        self.object_tracker.reset()
        self.logger.info("ObjectTracker has been reset.")

    def detect_and_track(
        self,
        images: tuple[np.ndarray, np.ndarray],
    ) -> tuple[list[DetectionResult], list[TrackedObject], np.ndarray]:
        """Perform detection and tracking on the provided stereo images."""
        img_left, img_right = images

        # Verify image dimensions
        self.logger.debug(f"Image Left Shape: {img_left.shape}")
        self.logger.debug(f"Image Right Shape: {img_right.shape}")

        disparity_map = self.depth_estimator.compute_disparity_map(img_left, img_right)
        depth_map = self.depth_estimator.compute_depth_map(disparity_map)

        # Verify depth map dimensions
        self.logger.debug(f"Depth Map Shape: {depth_map.shape}")

        detections_left = self._detect(img_left)

        for det in detections_left:
            depth = self._compute_object_depth(det, depth_map)
            if depth > 0:
                center_x, center_y = self._get_center(det.bbox)
                X = (center_x - self.cx) * depth / self.fx
                Y = (center_y - self.cy) * depth / self.fy
                Z = depth
                det.add_3d_position((X, Y, Z))
                det.estimate_dimensions(depth_map)
                det.appearance_feature = self._extract_appearance_feature(
                    det.mask,
                    img_left,
                    det.bbox,
                )
            else:
                self.logger.warning(f"Invalid depth for detection {det.label} at bbox {det.bbox}")
                det.add_3d_position((0.0, 0.0, 0.0))

        tracked_objects = self.object_tracker.update_tracks(detections_left)
        return detections_left, tracked_objects, depth_map

    def _detect(self, image: np.ndarray) -> list[DetectionResult]:
        """Detect objects and obtain segmentation masks."""
        try:
            results = self.model(image)
        except Exception as e:
            self.logger.exception(f"Model inference failed: {e}")
            return []

        detections = []
        for result in results:
            print(result)
            boxes = result.boxes
            masks = result.masks

            # Check if masks are available and contain data
            if masks is not None and masks.data is not None:
                # masks.data is a list or tensor containing one mask per detection
                mask_list = masks.data
                orig_shape = masks.orig_shape  # Original image shape (height, width)
            else:
                mask_list = [None] * len(boxes)
                orig_shape = image.shape[:2]  # (height, width)

            for box, mask in zip(boxes, mask_list):
                cls = int(box.cls.item())
                label = self.model.names[cls] if cls < len(self.model.names) else "unknown"
                if label.lower() not in self.SUPPORTED_LABELS:
                    continue
                confidence = box.conf.item()
                if confidence < self.conf_threshold:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x, y, w, h = x1, y1, x2 - x1, y2 - y1

                # Handle mask conversion and resizing
                if mask is not None:
                    if isinstance(mask, torch.Tensor):
                        mask_np = mask.cpu().numpy()
                    elif isinstance(mask, np.ndarray):
                        mask_np = mask
                    else:
                        self.logger.warning(f"Unknown mask type: {type(mask)}")
                        mask_np = np.zeros((int(h), int(w)), dtype=np.uint8)

                    # **Resize mask to match original image size**
                    try:
                        mask_np = cv2.resize(
                            mask_np, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST
                        )
                        # **Ensure binary mask**
                        mask_np = (mask_np > 0).astype(np.uint8)
                        self.logger.debug(
                            f"Resized mask shape: {mask_np.shape}, Expected: {orig_shape}"
                        )
                    except Exception as e:
                        self.logger.exception(f"Error resizing mask: {e}")
                        mask_np = np.zeros(orig_shape, dtype=np.uint8)
                else:
                    mask_np = np.zeros(orig_shape, dtype=np.uint8)

                detection = DetectionResult(
                    bbox=[x, y, w, h],
                    confidence=confidence,
                    class_id=cls,
                    label=label,
                    mask=mask_np,
                    fx=self.fx,
                    fy=self.fy,
                    cx=self.cx,
                    cy=self.cy,
                )
                detections.append(detection)
        return detections

    def _get_center(self, bbox: list[float]) -> tuple[float, float]:
        """Calculate the center of a bounding box."""
        x, y, w, h = bbox
        return x + w / 2, y + h / 2

    def _extract_appearance_feature(
        self,
        mask: np.ndarray,
        image: np.ndarray,
        bbox: list[float],
    ) -> np.ndarray:
        """Extract appearance features using the segmentation mask."""
        x, y, w, h = map(int, bbox)
        if mask is None or mask.size == 0:
            return np.array([])

        # Ensure bounding box is within image boundaries
        img_h, img_w = image.shape[:2]
        x_end = min(x + w, img_w)
        y_end = min(y + h, img_h)
        x = max(x, 0)
        y = max(y, 0)
        w = x_end - x
        h = y_end - y

        # Extract the mask ROI corresponding to the bounding box
        mask_roi = mask[y : y + h, x : x + w]
        roi = image[y : y + h, x : x + w]
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
