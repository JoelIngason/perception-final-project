# src/detection/detection_result.py

import logging
from dataclasses import dataclass, field

import numpy as np


@dataclass
class DetectionResult:
    """DetectionResult class to store detection information."""

    logger = logging.getLogger("autonomous_perception.detection")
    bbox: list[float]  # [x, y, w, h]
    confidence: float
    class_id: int
    label: str
    mask: np.ndarray = field(default_factory=lambda: np.array([]))  # Full image size mask
    position_3d: tuple[float, float, float] = (0.0, 0.0, 0.0)
    dimensions: tuple[float, float, float] = (0.0, 0.0, 0.0)  # height, width, length
    appearance_feature: np.ndarray = field(default_factory=lambda: np.array([]))
    oriented_bbox: np.ndarray = field(default_factory=lambda: np.array([]))
    fx: float = None
    fy: float = None
    cx: float = None
    cy: float = None

    def add_3d_position(self, position_3d: tuple[float, float, float]) -> None:
        """Add 3D position to the detection result."""
        self.position_3d = position_3d

    def estimate_dimensions(self, depth_map: np.ndarray) -> None:
        """Estimate object dimensions using depth map and camera intrinsics."""
        if self.mask is None or self.mask.size == 0:
            self.logger.debug("No mask available for dimension estimation.")
            return

        # Get indices where mask is true
        ys, xs = np.where(self.mask)
        self.logger.debug(f"Mask True Indices - ys: {ys.shape}, xs: {xs.shape}")
        if xs.size == 0 or ys.size == 0:
            self.logger.debug("No valid mask regions found.")
            return

        # Extract depth values at mask locations
        depth_values = depth_map[ys, xs]

        # Create a valid mask for depth values
        valid_mask = np.isfinite(depth_values) & (depth_values > 0)
        self.logger.debug(f"Valid Depth Values Count: {valid_mask.sum()} / {valid_mask.size}")

        # Apply the valid mask to xs, ys, and depth_values
        xs_valid = xs[valid_mask]
        ys_valid = ys[valid_mask]
        zs_valid = depth_values[valid_mask]

        if zs_valid.size == 0:
            self.logger.debug("No valid depth values after masking.")
            return

        # Convert pixel coordinates to camera coordinates
        xs_camera = (xs_valid - self.cx) * zs_valid / self.fx
        ys_camera = (ys_valid - self.cy) * zs_valid / self.fy

        self.logger.debug(f"xs_camera shape: {xs_camera.shape}")
        self.logger.debug(f"ys_camera shape: {ys_camera.shape}")
        self.logger.debug(f"zs_valid shape: {zs_valid.shape}")

        # Validate depth range (example: 0.5m to 100m)
        median_depth = np.median(zs_valid)
        if not (0.5 <= median_depth <= 100.0):
            self.logger.warning(
                f"Estimated depth {median_depth:.2f}m for {self.label} is out of realistic range."
            )
            return

        # Calculate dimensions
        width = xs_camera.max() - xs_camera.min()
        height = ys_camera.max() - ys_camera.min()
        length = zs_valid.max() - zs_valid.min()

        self.logger.debug(
            f"Estimated Dimensions - Height: {height:.2f}m, Width: {width:.2f}m, Length: {length:.2f}m"
        )

        self.dimensions = (height, width, length)
