import logging

import cv2
import numpy as np
from cv2.typing import MatLike  # Import MatLike for type annotations

from src.data_loader.dataset import Frame  # Ensure correct import path


class Rectifier:
    def __init__(self, calibration_params: dict[str, np.ndarray]):
        """
        Initialize the Rectifier with calibration parameters.

        Args:
            calibration_params (dict[str, np.ndarray]): Dictionary containing calibration matrices and parameters.

        """
        self.calibration_params = calibration_params
        self.logger = logging.getLogger("autonomous_perception.calibration")

        # Configure logger if not already configured
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        # Compute rectification maps during initialization
        self.left_map1, self.left_map2, self.right_map1, self.right_map2 = (
            self._compute_rectification_maps()
        )

    def _compute_rectification_maps(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute rectification maps for left and right images.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Rectification maps for left and right images.

        """
        try:
            # Extract necessary calibration parameters
            K_left = self.calibration_params["K_02"]
            K_right = self.calibration_params["K_03"]
            D_left = self.calibration_params["D_02"]
            D_right = self.calibration_params["D_03"]
            R_rect = self.calibration_params["R_rect_02"]
            P_left = self.calibration_params["P_rect_02"]
            P_right = self.calibration_params["P_rect_03"]
            S_rect_02 = self.calibration_params["S_rect_02"]

            self.logger.info("Computing rectification maps")

            # Define image size (width, height)
            image_size = (
                int(S_rect_02[0]),
                int(S_rect_02[1]),
            )

            # Compute rectification maps for left image
            left_map1, left_map2 = cv2.initUndistortRectifyMap(
                cameraMatrix=K_left,
                distCoeffs=D_left,
                R=R_rect,
                newCameraMatrix=P_left,
                size=image_size,
                m1type=cv2.CV_16SC2,
            )

            # Compute rectification maps for right image
            right_map1, right_map2 = cv2.initUndistortRectifyMap(
                cameraMatrix=K_right,
                distCoeffs=D_right,
                R=R_rect,
                newCameraMatrix=P_right,
                size=image_size,
                m1type=cv2.CV_16SC2,
            )

            self.logger.info("Rectification maps computed successfully")
            return left_map1, left_map2, right_map1, right_map2
        except KeyError as e:
            self.logger.error(f"Missing calibration parameter: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error computing rectification maps: {e}")
            raise

    def rectify_images(
        self,
        img_left: MatLike,
        img_right: MatLike,
    ) -> tuple[MatLike, MatLike]:
        """
        Rectify a pair of left and right images.

        Args:
            img_left (MatLike): Left image as a NumPy array or compatible type.
            img_right (MatLike): Right image as a NumPy array or compatible type.

        Returns:
            Tuple[MatLike, MatLike]: Rectified left and right images.

        """
        self.logger.debug("Rectifying image pair")
        try:
            rectified_left = cv2.remap(
                src=img_left,
                map1=self.left_map1,
                map2=self.left_map2,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),  # Black border
            )
            rectified_right = cv2.remap(
                src=img_right,
                map1=self.right_map1,
                map2=self.right_map2,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),  # Black border
            )
            return rectified_left, rectified_right
        except Exception as e:
            self.logger.error(f"Error during image rectification: {e}")
            raise

    def rectify_data(self, frames: list[Frame]) -> list[Frame]:
        """
        Rectify a list of Frame objects containing raw images.

        Args:
            frames (List[Frame]): List of Frame objects with raw left and right images.

        Returns:
            List[Frame]: List of Frame objects with rectified left and right images.

        """
        self.logger.info(f"Rectifying {len(frames)} frames")
        rectified_frames = []
        for idx, frame in enumerate(frames, 1):
            try:
                rectified_left, rectified_right = self.rectify_images(
                    frame.images[0],
                    frame.images[1],
                )
                rectified_frame = Frame(
                    image_left=rectified_left,
                    image_right=rectified_right,
                    timestamp=frame.timestamp,
                    labels=frame.labels,
                )
                rectified_frames.append(rectified_frame)
                self.logger.debug(f"Rectified frame {idx}/{len(frames)}")
            except Exception as e:
                self.logger.error(f"Failed to rectify frame {idx}: {e}")
                continue
        self.logger.info(f"Rectified {len(rectified_frames)} out of {len(frames)} frames")
        return rectified_frames
