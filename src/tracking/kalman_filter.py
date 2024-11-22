import cv2
import numpy as np


class KalmanFilter3D:
    def __init__(self, dt: float = 1.0):
        """
        Initialize the Kalman Filter for 3D tracking with a constant velocity model.

        Args:
            dt (float): Time step between measurements.

        """
        # Define the number of states and measurements
        self.kf = cv2.KalmanFilter(6, 3)

        # State vector: [x, y, z, vx, vy, vz]
        self.kf.transitionMatrix = np.array(
            [
                [1, 0, 0, dt, 0, 0],
                [0, 1, 0, 0, dt, 0],
                [0, 0, 1, 0, 0, dt],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

        # Measurement matrix: we measure [x, y, z]
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]],
            dtype=np.float32,
        )

        # Process noise covariance
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-4

        # Measurement noise covariance
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1e-1

        # Error covariance post
        self.kf.errorCovPost = np.eye(6, dtype=np.float32)

        # Initial state (will be set with the first detection)
        self.kf.statePost = np.zeros((6, 1), dtype=np.float32)

    def predict(self):
        """Predict the next state."""
        self.kf.predict()

    def update(self, measurement: np.ndarray):
        """
        Update the Kalman Filter with a new measurement.

        Args:
            measurement (np.ndarray): Measurement vector [x, y, z].

        """
        self.kf.correct(measurement.astype(np.float32).reshape(3, 1))

    def get_position(self) -> np.ndarray:
        """
        Get the current estimated position.

        Returns:
            np.ndarray: Estimated position [x, y, z].

        """
        return self.kf.statePost[:3].flatten()
