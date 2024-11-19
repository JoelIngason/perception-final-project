import logging

import cv2
import numpy as np


def calculate_reprojection_error(
    objpoints: list,
    imgpoints_left: list,
    imgpoints_right: list,
    rvecs_left,
    tvecs_left,
    rvecs_right,
    tvecs_right,
    camera_matrix_left,
    dist_coeffs_left,
    camera_matrix_right,
    dist_coeffs_right,
    R,
    T,
) -> float:
    """
    Calculate the reprojection error for stereo calibration.

    Args:
        objpoints (List): 3D points in real world space.
        imgpoints_left (List): 2D points in left image.
        imgpoints_right (List): 2D points in right image.
        rvecs_left, tvecs_left (List): Rotation and translation vectors for left camera.
        rvecs_right, tvecs_right (List): Rotation and translation vectors for right camera.
        camera_matrix_left, dist_coeffs_left (np.ndarray): Left camera matrix and distortion coefficients.
        camera_matrix_right, dist_coeffs_right (np.ndarray): Right camera matrix and distortion coefficients.
        R, T (np.ndarray): Rotation and translation between the cameras.

    Returns:
        float: Average reprojection error.

    """
    total_error = 0
    total_points = 0
    for i in range(len(objpoints)):
        # Project points onto left image
        imgpoints_left_proj, _ = cv2.projectPoints(
            objpoints[i],
            rvecs_left[i],
            tvecs_left[i],
            camera_matrix_left,
            dist_coeffs_left,
        )
        error_left = cv2.norm(imgpoints_left[i], imgpoints_left_proj, cv2.NORM_L2) / len(
            imgpoints_left_proj,
        )

        # Project points onto right image
        imgpoints_right_proj, _ = cv2.projectPoints(
            objpoints[i],
            rvecs_right[i],
            tvecs_right[i],
            camera_matrix_right,
            dist_coeffs_right,
        )
        error_right = cv2.norm(imgpoints_right[i], imgpoints_right_proj, cv2.NORM_L2) / len(
            imgpoints_right_proj,
        )

        total_error += error_left + error_right
        total_points += 2

    mean_error = total_error / total_points if total_points > 0 else 0
    return mean_error


class StereoCalibrator:
    """Calibrate stereo camera using chessboard calibration images."""

    def __init__(self, config: dict) -> None:
        """
        Initialize the StereoCalibrator with calibration configuration.

        Args:
            config (Dict): Calibration configuration dictionary.

        """
        self.chessboard_size = tuple(config.get("chessboard_size", (9, 6)))
        self.square_size = config.get("square_size", 0.025)  # Default square size in meters
        self.logger = logging.getLogger("autonomous_perception.calibration")
        self.logger.setLevel(logging.INFO)
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
                msg = f"Missing calibration parameters: {missing_keys}"
                raise KeyError(msg)

            # Calculate baseline from projection matrices
            P_rect_02 = calib_params["P_rect_02"]  # 3x4 matrix
            P_rect_03 = calib_params["P_rect_03"]  # 3x4 matrix

            self.focal_length = P_rect_02[0, 0]  # fx from rectified left projection matrix
            Tx_left = P_rect_02[0, 3]
            Tx_right = P_rect_03[0, 3]
            self.baseline = np.abs(Tx_right - Tx_left) / self.focal_length

            # Store baseline in calibration parameters
            calib_params["baseline"] = self.baseline

            # Calculate reprojection error
            # Assuming objpoints, imgpoints_left, imgpoints_right are available
            objpoints = calib_params.get("objpoints", [])
            if isinstance(objpoints, np.ndarray):
                objpoints = objpoints.tolist()
            # Convert image points to list if they are numpy arrays
            imgpoints_left = calib_params.get("imgpoints_left", [])
            imgpoints_right = calib_params.get("imgpoints_right", [])
            if isinstance(imgpoints_left, np.ndarray):
                imgpoints_left = imgpoints_left.tolist()
            if isinstance(imgpoints_right, np.ndarray):
                imgpoints_right = imgpoints_right.tolist()

            reproj_error = calculate_reprojection_error(
                objpoints=objpoints,
                imgpoints_left=imgpoints_left,
                imgpoints_right=imgpoints_right,
                rvecs_left=calib_params.get("rvecs_left", []),
                tvecs_left=calib_params.get("tvecs_left", []),
                rvecs_right=calib_params.get("rvecs_right", []),
                tvecs_right=calib_params.get("tvecs_right", []),
                camera_matrix_left=calib_params["K_02"],
                dist_coeffs_left=calib_params["D_02"],
                camera_matrix_right=calib_params["K_03"],
                dist_coeffs_right=calib_params["D_03"],
                R=calib_params["R_02"],
                T=calib_params["T_02"],
            )
            self.logger.info(f"Stereo calibration reprojection error: {reproj_error:.4f} pixels")

            if reproj_error > 1.0:
                self.logger.warning(
                    "High reprojection error detected. Consider recalibrating the stereo cameras.",
                )

            self.logger.info("Stereo calibration completed successfully")
        except Exception:
            self.logger.exception("Stereo calibration failed")
            raise
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
                    if key == "calib_time":
                        self.logger.info(f"Line {line_num}: Calibration time '{value}' ignored.")
                        continue
                    if key == "corner_dist":
                        try:
                            calib_params[key] = float(value)
                            self.logger.debug(f"Loaded {key} as float.")
                        except ValueError:
                            self.logger.exception(
                                f"Line {line_num}: Could not convert value to float for key '{key}'",
                            )
                        continue
                    try:
                        values = list(map(float, value.split()))
                    except ValueError:
                        self.logger.exception(
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
            self.logger.exception(f"Calibration file not found: {filepath}")
            raise
        except Exception:
            self.logger.exception("Unexpected error while loading calibration file")
            raise
        else:
            self.logger.debug(f"Loaded calibration parameters: {list(calib_params.keys())}")
            return calib_params
