# src/calibration/stereo_calibration.py
import cv2
import numpy as np
import logging
from typing import Dict

class StereoCalibrator:
    def __init__(self, config):
        self.chessboard_size = tuple(config['chessboard_size'])
        self.square_size = config['square_size']
        self.logger = logging.getLogger('autonomous_perception.calibration')

    def calibrate(self, calibration_file: str) -> Dict[str, np.ndarray]:
        self.logger.info("Starting stereo calibration")
        # Load calibration parameters from file
        calib_params = self._load_calibration_file(calibration_file)
        self.logger.info("Stereo calibration completed")
        return calib_params

    def _load_calibration_file(self, filepath: str) -> Dict[str, np.ndarray]:
        calib_params = {}
        with open(filepath, 'r') as f:
            for line in f:
                if ':' not in line:
                    continue
                key, value = line.strip().split(':', 1)
                values = list(map(float, value.strip().split()))
                if key.startswith('K_') or key.startswith('R_') or key.startswith('P_'):
                    calib_params[key] = np.array(values).reshape((3, 3))
                elif key.startswith('D_'):
                    calib_params[key] = np.array(values)
                elif key.startswith('T_'):
                    calib_params[key] = np.array(values).reshape((3, 1))
                elif key.startswith('S_') or key.startswith('R_rect_'):
                    calib_params[key] = np.array(values)
        return calib_params
