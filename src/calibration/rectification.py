import logging

import cv2
import numpy as np
from cv2.typing import MatLike  # Import MatLike for type annotations

from src.data_loader.dataset import Frame


class Rectifier:
    """Rectify stereo image pairs using calibration parameters."""

    def __init__(self, calibration_params: dict[str, np.ndarray]) -> None:
        """
        Initialize the Rectifier with calibration parameters.

        Args:
            calibration_params (Dict[str, np.ndarray]): Dictionary containing calibration matrices
                and parameters.

        """
        self.calibration_params = calibration_params
        self.logger = logging.getLogger("autonomous_perception.calibration")
        self.logger.setLevel(logging.INFO)
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Compute rectification maps during initialization
        self.left_map1, self.left_map2, self.right_map1, self.right_map2 = (
            self._compute_rectification_maps()
        )

    def _compute_rectification_maps(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute rectification maps for left and right images.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Rectification maps for left
                and right images.

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
            image_size = (int(S_rect_02[0]), int(S_rect_02[1]))

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
        except KeyError:
            self.logger.exception("Missing calibration parameters for rectification")
            raise
        except Exception:
            self.logger.exception("Error during rectification map computation")
            raise
        else:
            return left_map1, left_map2, right_map1, right_map2

    def rectify_images(self, img_left: MatLike, img_right: MatLike) -> tuple[MatLike, MatLike]:
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
        except Exception:
            self.logger.exception("Error during image rectification")
            raise
        else:
            return rectified_left, rectified_right

    def rectify_data(self, frames: dict[str, list[Frame]]) -> dict[str, list[Frame]]:
        """
        Rectify a dictionary of Frame objects containing raw images.

        Args:
            frames (Dict[str, list[Frame]]): Dictionary of sequences with Frame objects.

        Returns:
            Dict[str, list[Frame]]: Dictionary with rectified Frame objects.

        """
        self.logger.info(f"Rectifying {len(frames)} sequences")
        rectified_frames = {}
        for seq_id, frame_list in frames.items():
            rectified_seq = []
            for idx, frame in enumerate(frame_list):
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
                    rectified_seq.append(rectified_frame)
                    self.logger.debug(
                        f"Rectified frame {idx}/{len(frame_list)} in sequence {seq_id}",
                    )
                except Exception:
                    self.logger.exception(f"Failed to rectify frame {idx} in sequence {seq_id}")
                    continue
            rectified_frames[seq_id] = rectified_seq
            self.logger.info(f"Rectified {len(rectified_seq)} frames in sequence {seq_id}")
        return rectified_frames
