import logging

import numpy as np


class StereoCalibrator:
    def __init__(self, config: dict):
        self.chessboard_size = tuple(config.get("chessboard_size", (9, 6)))  # Default size
        self.square_size = config.get("square_size", 0.025)  # Default square size in meters
        self.logger = logging.getLogger("autonomous_perception.calibration")

        # Configure logger if not already configured
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def calibrate(self, calibration_file: str) -> dict[str, np.ndarray]:
        """
        Perform stereo calibration by loading calibration parameters from a file.

        Args:
            calibration_file (str): Path to the calibration parameters file.

        Returns:
            Dict[str, np.ndarray]: Dictionary containing calibration parameters.

        """
        self.logger.info("Starting stereo calibration")
        try:
            # Load calibration parameters from file
            calib_params = self._load_calibration_file(calibration_file)

            # Validate essential calibration parameters
            required_keys = {
                "K_02",
                "K_03",
                "D_02",
                "D_03",
                "R_02",
                "T_02",
                "R_rect_02",
                "P_rect_02",
                "P_rect_03",
                "S_rect_02",
            }
            missing_keys = required_keys - calib_params.keys()
            if missing_keys:
                self.logger.error(f"Missing calibration parameters: {missing_keys}")
                raise KeyError(f"Missing calibration parameters: {missing_keys}")

            self.logger.info("Stereo calibration completed successfully")
            return calib_params
        except Exception as e:
            self.logger.error(f"Stereo calibration failed: {e}")
            raise

    def _load_calibration_file(self, filepath: str) -> dict[str, np.ndarray]:
        """
        Load calibration parameters from a text file.

        Args:
            filepath (str): Path to the calibration file.

        Returns:
            Dict[str, np.ndarray]: Dictionary containing calibration parameters.

        """
        calib_params = {}
        try:
            with open(filepath) as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or ":" not in line:
                        self.logger.debug(
                            f"Skipping line {line_num}: Invalid format or empty line.",
                        )
                        continue
                    key, value = line.split(":", 1)
                    key = key.strip()
                    value = value.strip()
                    if not value:
                        self.logger.warning(
                            f"Line {line_num}: No values found for key '{key}'. Skipping.",
                        )
                        continue

                    # Handle specific keys
                    if key == "calib_time":
                        self.logger.info(f"Line {line_num}: Calibration time '{value}' ignored.")
                        continue  # Skip non-numeric calibration time
                    if key == "corner_dist":
                        try:
                            calib_params[key] = float(value)
                            self.logger.debug(f"Loaded {key} as float.")
                        except ValueError as e:
                            self.logger.error(
                                f"Line {line_num}: Could not convert value to float for key '{key}': {e}",
                            )
                        continue

                    try:
                        values = list(map(float, value.split()))
                    except ValueError as e:
                        self.logger.error(
                            f"Line {line_num}: Could not convert values to floats for key '{key}': {e}",
                        )
                        continue

                    # Handle matrix and vector keys
                    if key.startswith("K_") or key.startswith("R_") or key.startswith("P_"):
                        if len(values) == 9:
                            calib_params[key] = np.array(values).reshape((3, 3))
                            self.logger.debug(f"Loaded {key} as 3x3 matrix.")
                        elif len(values) == 12 and key.startswith("P_"):
                            calib_params[key] = np.array(values).reshape((3, 4))
                            self.logger.debug(f"Loaded {key} as 3x4 projection matrix.")
                        else:
                            self.logger.warning(
                                f"Key '{key}' has unexpected number of values ({len(values)}). Skipping.",
                            )
                    elif key.startswith("D_"):
                        calib_params[key] = np.array(values)
                        self.logger.debug(
                            f"Loaded {key} as distortion coefficients: {calib_params[key]}",
                        )
                    elif key.startswith("T_"):
                        if len(values) == 3:
                            calib_params[key] = np.array(values).reshape((3, 1))
                            self.logger.debug(
                                f"Loaded {key} as 3x1 translation vector: {calib_params[key]}",
                            )
                        else:
                            self.logger.warning(
                                f"Key '{key}' has unexpected number of values ({len(values)}). Expected 3 for T_. Skipping.",
                            )
                    elif key.startswith("S_") or key.startswith("R_rect_"):
                        calib_params[key] = np.array(values)
                        self.logger.debug(f"Loaded {key} as 1D array.")
                    else:
                        self.logger.warning(
                            f"Unrecognized key '{key}' at line {line_num}. Skipping.",
                        )
        except FileNotFoundError:
            self.logger.error(f"Calibration file not found: {filepath}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error while loading calibration file: {e}")
            raise

        self.logger.debug(f"Loaded calibration parameters: {list(calib_params.keys())}")
        return calib_params
