import logging

import cv2
import numpy as np


class Rectifier:
    def __init__(self, calibration_params: dict[str, np.ndarray]):
        self.calibration_params = calibration_params
        self.logger = logging.getLogger('autonomous_perception.calibration')
        self.left_map, self.right_map = self._compute_rectification_maps()

    def _compute_rectification_maps(self) -> tuple[np.ndarray, np.ndarray]:
        K_left = self.calibration_params['K_02']
        K_right = self.calibration_params['K_03']
        D_left = self.calibration_params['D_02']
        D_right = self.calibration_params['D_03']
        R = self.calibration_params['R_02']
        T = self.calibration_params['T_02']

        # Compute rectification transforms
        self.logger.info("Computing rectification maps")
        R_rect = self.calibration_params['R_rect_02']
        P_left = self.calibration_params['P_rect_02']
        P_right = self.calibration_params['P_rect_03']

        image_size = (int(self.calibration_params['S_rect_02'][0]), int(self.calibration_params['S_rect_02'][1]))

        left_map, _ = cv2.initUndistortRectifyMap(
            K_left, D_left, R_rect, P_left, image_size, cv2.CV_16SC2,
        )
        right_map, _ = cv2.initUndistortRectifyMap(
            K_right, D_right, R_rect, P_right, image_size, cv2.CV_16SC2,
        )
        self.logger.info("Rectification maps computed successfully")
        return left_map, right_map

    def rectify_images(self, img_left: np.ndarray, img_right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.logger.debug("Rectifying image pair")
        rectified_left = cv2.remap(img_left, self.left_map, None, cv2.INTER_LINEAR)
        rectified_right = cv2.remap(img_right, self.right_map, None, cv2.INTER_LINEAR)
        return rectified_left, rectified_right
