import logging

import numpy as np


class StereoCalibrator:
    """Calibrate stereo camera using chessboard calibration images."""

    def __init__(self, config: dict) -> None:
        """
        Initialize the StereoCalibrator with calibration configuration.

        Args:
            config (Dict): Calibration configuration dictionary.

        """
        self.chessboard_size = tuple(config.get("chessboard_size", (9, 6)))
        self.square_size = config.get("square_size", 0.01)  # Default square size in meters
        self.logger = logging.getLogger("stereo_calibration")
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

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
                "K_02",  # Intrinsic matrix for Camera 2
                "K_03",  # Intrinsic matrix for Camera 3
                "T_02",  # Translation vector for Camera 2
                "T_03",  # Translation vector for Camera 3
                "P_rect_02",  # Rectified projection matrix for Camera 2
                "P_rect_03",  # Rectified projection matrix for Camera 3
            }
            missing_keys = required_keys - calib_params.keys()
            if missing_keys:
                self.logger.error(f"Missing calibration parameters: {missing_keys}")
                raise KeyError(f"Missing calibration parameters: {missing_keys}")

            # Extract translation vectors
            T_02 = calib_params["T_02"]  # Translation vector for Camera 2
            T_03 = calib_params["T_03"]  # Translation vector for Camera 3

            if T_02.shape != (3, 1) or T_03.shape != (3, 1):
                raise ValueError("T_02 or T_03 have unexpected shapes. Expected (3, 1).")

            # Compute baseline: distance between Camera 2 and Camera 3 along x-axis
            self.baseline = np.abs(T_03[0, 0] - T_02[0, 0])  # Baseline in meters

            # Alternatively, infer baseline from projection matrices (as validation)
            P_rect_02 = calib_params["P_rect_02"]
            P_rect_03 = calib_params["P_rect_03"]
            focal_length = P_rect_02[0, 0]  # fx from rectified left projection matrix
            self.focal_length = focal_length
            Tx_left = P_rect_02[0, 3] / focal_length
            Tx_right = P_rect_03[0, 3] / focal_length
            inferred_baseline = np.abs(Tx_left - Tx_right)

            # Validate baseline consistency
            if not np.isclose(self.baseline, inferred_baseline, atol=1e-4):
                self.logger.warning(
                    f"Baseline from translation vectors ({self.baseline:.4f} m) "
                    f"differs from inferred baseline ({inferred_baseline:.4f} m).",
                )
                self.baseline = inferred_baseline  # Use the more reliable value

            # Store baseline and focal length in calibration parameters
            calib_params["baseline"] = self.baseline
            calib_params["focal_length"] = self.focal_length
            self.logger.info(
                f"Stereo calibration completed successfully with baseline: {self.baseline:.4f} meters and focal length: {self.focal_length:.4f}",
            )
        except Exception as e:
            self.logger.exception("Stereo calibration failed")
            raise e
        else:
            return calib_params

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
                    try:
                        values = list(map(float, value.split()))
                    except ValueError:
                        self.logger.warning(
                            f"Line {line_num}: Could not convert values to floats for key '{key}'",
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
                        self.logger.debug(f"Loaded {key} as distortion coefficients.")
                    elif key.startswith("T_"):
                        if len(values) == 3:
                            calib_params[key] = np.array(values).reshape((3, 1))
                            self.logger.debug(f"Loaded {key} as 3x1 translation vector.")
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
            self.logger.exception(f"Calibration file not found: {filepath}")
            raise
        except Exception as e:
            self.logger.exception("Unexpected error while loading calibration file")
            raise e
        else:
            self.logger.debug(f"Loaded calibration parameters: {list(calib_params.keys())}")
            return calib_params
