import logging
from dataclasses import dataclass, field

import numpy as np


@dataclass
class DetectionResult:
    """DetectionResult class to store detection information."""

    bbox: list[int]  # [left, top, right, bottom]
    confidence: float
    class_id: int
    label: str
    mask: np.ndarray | None = field(default_factory=lambda: np.array([]))  # Full image size mask
    position_3d: tuple[float, float, float] | None = None
    dimensions: tuple[float, float, float] | None = None  # height, width, length
    appearance_feature: np.ndarray | None = field(default_factory=lambda: np.array([]))
    oriented_bbox: np.ndarray | None = field(default_factory=lambda: np.array([]))
    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None

    # Initialize a logger for the class
    def __post_init__(self):
        self.logger = logging.getLogger("autonomous_perception.detection_result")
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def add_3d_position(self, position_3d: tuple[float, float, float]) -> None:
        """Add 3D position to the detection result."""
        self.position_3d = position_3d
        self.logger.debug(f"Added 3D position: {self.position_3d}")

    def estimate_dimensions(self, depth: float) -> None:
        """
        Estimate object dimensions using depth map and camera intrinsics.

        Args:
            depth (float): Depth value of the object in meters.

        """
        # Estimate object dimensions by using bounding box area and depth
        if self.bbox is None or len(self.bbox) == 0:
            self.logger.debug("No bounding box available for dimensions estimation.")
            return

        width = self.bbox[2] - self.bbox[0]
        height = self.bbox[3] - self.bbox[1]

        self.dimensions = (height, width, depth)

    def to_supported_label(self) -> str:
        """Return the supported label for the object."""
        MAP = {
            "car": "Car",
            "person": "Pedestrian",
            "bicycle": "Cyclist",
        }
        return MAP.get(self.label.lower(), "Unknown")

    def compute_position_3d(self, depth: float) -> None:
        """
        Compute and add the 3D position based on the center of the mask and the provided depth.

        Args:
            depth (float): The depth (Z-coordinate) of the object in meters.

        """
        if self.mask is None or self.mask.size == 0:
            self.logger.debug("No mask available for position_3d computation.")
            return

        # Get indices where mask is true
        ys, xs = np.where(self.mask)
        self.logger.debug(f"Mask True Indices - ys: {ys.shape}, xs: {xs.shape}")

        if xs.size == 0 or ys.size == 0:
            self.logger.debug("No valid mask regions found for position_3d computation.")
            return

        # Compute the centroid of the mask
        centroid_x = np.mean(xs)
        centroid_y = np.mean(ys)
        self.logger.debug(f"Computed Centroid - x: {centroid_x}, y: {centroid_y}")

        if self.fx is None or self.fy is None or self.cx is None or self.cy is None:
            self.logger.error("Camera intrinsics (fx, fy, cx, cy) are not set.")
            return

        # Compute 3D position using camera intrinsics
        X = float((centroid_x - self.cx) * depth / self.fx)
        Y = float((centroid_y - self.cy) * depth / self.fy)
        Z = float(depth)

        self.position_3d = (X, Y, Z)
        self.logger.debug(f"Computed 3D Position: {self.position_3d}")
