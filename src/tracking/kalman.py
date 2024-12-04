# Ultralytics YOLO 🚀, AGPL-3.0 license

import numpy as np
import scipy.linalg


class KalmanFilterXYZAH:
    """
    A KalmanFilterXYZAH class for tracking bounding boxes in 3D image space using a Kalman filter.

    Implements a Kalman filter for tracking bounding boxes with state space
    (x, y, z, a, h, vx, vy, vz, va, vh), where (x, y, z) represents the bounding box center position,
    'a' is the aspect ratio, 'h' is the height, and their respective velocities. Object motion follows
    a constant velocity model, and bounding box location (x, y, z, a, h) is taken as a direct
    observation of the state space (linear observation model).

    Attributes:
        _motion_mat (np.ndarray): The motion matrix for the Kalman filter.
        _update_mat (np.ndarray): The update matrix for the Kalman filter.
        _std_weight_position (float): Standard deviation weight for position.
        _std_weight_velocity (float): Standard deviation weight for velocity.

    Methods:
        initiate: Creates a track from an unassociated measurement.
        predict: Runs the Kalman filter prediction step.
        project: Projects the state distribution to measurement space.
        multi_predict: Runs the Kalman filter prediction step (vectorized version).
        update: Runs the Kalman filter correction step.
        gating_distance: Computes the gating distance between state distribution and measurements.

    Examples:
        Initialize the Kalman filter and create a track from a measurement
        >>> kf = KalmanFilterXYZAH()
        >>> measurement = np.array([100, 200, 50, 1.5, 50])
        >>> mean, covariance = kf.initiate(measurement)
        >>> print(mean)
        >>> print(covariance)

    """

    def __init__(self):
        """
        Initialize Kalman filter model matrices with motion and observation uncertainty weights.

        The Kalman filter is initialized with a 10-dimensional state space
        (x, y, z, a, h, vx, vy, vz, va, vh), where (x, y, z) represents the bounding box center position,
        'a' is the aspect ratio, 'h' is the height, and their respective velocities. The filter uses a constant
        velocity model for object motion and a linear observation model for bounding box location.

        Examples:
            Initialize a Kalman filter for tracking:
            >>> kf = KalmanFilterXYZAH()

        """
        ndim = 5  # x, y, z, a, h
        dt = 1.0  # Time step

        # Create Kalman filter model matrices
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty weights
        self._std_weight_position = 1.0 / 40
        self._std_weight_velocity = 1.0 / 160

        # Increase uncertainty in z-axis
        self._std_weight_position_z = 0.9
        self._std_weight_velocity_z = 0.9

    def initiate(self, measurement: np.ndarray) -> tuple:
        """
        Create a track from an unassociated measurement.

        Args:
            measurement (ndarray): Bounding box coordinates (x, y, z, a, h) with center position (x, y, z),
                aspect ratio a, and height h.

        Returns:
            (tuple[ndarray, ndarray]): Returns the mean vector (10-dimensional) and covariance matrix (10x10 dimensional)
                of the new track. Unobserved velocities are initialized to 0 mean.

        Examples:
            >>> kf = KalmanFilterXYZAH()
            >>> measurement = np.array([100, 50, 30, 1.5, 200])
            >>> mean, covariance = kf.initiate(measurement)

        """
        mean_pos = measurement  # [x, y, z, a, h]
        mean_vel = np.zeros_like(mean_pos)  # [vx, vy, vz, va, vh]
        mean = np.r_[mean_pos, mean_vel]  # [x, y, z, a, h, vx, vy, vz, va, vh]

        std = [
            2 * self._std_weight_position * measurement[4],  # x
            2 * self._std_weight_position * measurement[4],  # y
            4 * self._std_weight_position * measurement[4],  # z
            1e-2,  # a
            2 * self._std_weight_position * measurement[4],  # h
            10 * self._std_weight_velocity * measurement[4],  # vx
            10 * self._std_weight_velocity * measurement[4],  # vy
            20 * self._std_weight_velocity * measurement[4],  # vz
            1e-5,  # va
            10 * self._std_weight_velocity * measurement[4],  # vh
        ]
        covariance = np.diag(np.square(std))
        # Increase z-axis covariance
        covariance[2, 2] = np.square(4 * self._std_weight_position_z * measurement[4])
        covariance[7, 7] = np.square(20 * self._std_weight_velocity_z * measurement[4])
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple:
        """
        Run Kalman filter prediction step.

        Args:
            mean (ndarray): The 10-dimensional mean vector of the object state at the previous time step.
            covariance (ndarray): The 10x10-dimensional covariance matrix of the object state at the previous time step.

        Returns:
            (tuple[ndarray, ndarray]): Returns the mean vector and covariance matrix of the predicted state.

        Examples:
            >>> kf = KalmanFilterXYZAH()
            >>> mean = np.array([0, 0, 0, 1, 1, 0, 0, 0, 0, 0])
            >>> covariance = np.eye(10)
            >>> predicted_mean, predicted_covariance = kf.predict(mean, covariance)

        """
        std_pos = [
            self._std_weight_position * mean[4],  # x
            self._std_weight_position * mean[4],  # y
            self._std_weight_position * mean[4],  # z
            1e-2,  # a
            self._std_weight_position * mean[4],  # h
        ]
        std_vel = [
            self._std_weight_velocity * mean[4],  # vx
            self._std_weight_velocity * mean[4],  # vy
            self._std_weight_velocity * mean[4],  # vz
            1e-5,  # va
            self._std_weight_velocity * mean[4],  # vh
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(self._motion_mat, mean)
        covariance = (
            np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        )

        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> tuple:
        """
        Project state distribution to measurement space.

        Args:
            mean (ndarray): The state's mean vector (10-dimensional array).
            covariance (ndarray): The state's covariance matrix (10x10-dimensional).

        Returns:
            (tuple[ndarray, ndarray]): Returns the projected mean and covariance matrix of the given state estimate.

        Examples:
            >>> kf = KalmanFilterXYZAH()
            >>> mean = np.array([0, 0, 0, 1, 1, 0, 0, 0, 0, 0])
            >>> covariance = np.eye(10)
            >>> projected_mean, projected_covariance = kf.project(mean, covariance)

        """
        std = [
            self._std_weight_position * mean[4],  # x
            self._std_weight_position * mean[4],  # y
            self._std_weight_position * mean[4],  # z
            1e-1,  # a
            self._std_weight_position * mean[4],  # h
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple:
        """
        Run Kalman filter prediction step for multiple object states (Vectorized version).

        Args:
            mean (ndarray): The Nx10 dimensional mean matrix of the object states at the previous time step.
            covariance (ndarray): The Nx10x10 covariance matrix of the object states at the previous time step.

        Returns:
            (tuple[ndarray, ndarray]): Returns the mean matrix and covariance matrix of the predicted states.
                The mean matrix has shape (N, 10) and the covariance matrix has shape (N, 10, 10).

        Examples:
            >>> mean = np.random.rand(10, 10)  # 10 object states
            >>> covariance = np.random.rand(10, 10, 10)  # Covariance matrices for 10 object states
            >>> predicted_mean, predicted_covariance = kalman_filter.multi_predict(mean, covariance)

        """
        std_pos = [
            self._std_weight_position * mean[:, 4],  # x
            self._std_weight_position * mean[:, 4],  # y
            self._std_weight_position * mean[:, 4],  # z
            1e-2 * np.ones_like(mean[:, 4]),  # a
            self._std_weight_position * mean[:, 4],  # h
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 4],  # vx
            self._std_weight_velocity * mean[:, 4],  # vy
            self._std_weight_velocity * mean[:, 4],  # vz
            1e-5 * np.ones_like(mean[:, 4]),  # va
            self._std_weight_velocity * mean[:, 4],  # vh
        ]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = np.array([np.diag(sqr[i]) for i in range(len(mean))])

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
        measurement_mask: np.ndarray = None,
    ) -> tuple:
        """
        Run Kalman filter correction step with optional partial measurements.

        Args:
            mean (ndarray): The predicted state's mean vector (10-dimensional).
            covariance (ndarray): The state's covariance matrix (10x10-dimensional).
            measurement (ndarray): The 5-dimensional measurement vector (x, y, z, a, h).
            measurement_mask (ndarray): A boolean array indicating which measurements are available.

        Returns:
            (tuple[ndarray, ndarray]): Returns the measurement-corrected state distribution.

        """
        if measurement_mask is None:
            measurement_mask = np.array([True, True, True, True, True])

        # Select only the available measurements
        available_indices = np.where(measurement_mask)[0]
        if len(available_indices) == 0:
            # No measurements available; return the prediction
            return mean, covariance

        # Construct the measurement matrix for available measurements
        H = self._update_mat[available_indices]

        # Project the state distribution to measurement space for available measurements
        mean_proj, covariance_proj = self.project(mean, covariance)
        mean_proj = mean_proj[available_indices]
        covariance_proj = covariance_proj[np.ix_(available_indices, available_indices)]

        # Measurement vector with only available measurements
        z = measurement[available_indices]

        # Compute Kalman Gain
        S = (
            covariance_proj
            + np.diag(
                [self._std_weight_position * mean[4] if i < 4 else 1e-2 for i in available_indices],
            )
            ** 2
        )
        K = np.dot(covariance, H.T) @ np.linalg.inv(S)

        # Innovation
        y = z - mean_proj

        # Update mean and covariance
        new_mean = mean + np.dot(K, y)
        new_covariance = covariance - np.dot(K, H) @ covariance

        return new_mean, new_covariance

    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        only_position: bool = False,
        metric: str = "maha",
    ) -> np.ndarray:
        """
        Compute gating distance between state distribution and measurements.

        A suitable distance threshold can be obtained from `chi2inv95`. If `only_position` is False, the chi-square
        distribution has 5 degrees of freedom, otherwise 3.

        Args:
            mean (ndarray): Mean vector over the state distribution (10-dimensional).
            covariance (ndarray): Covariance of the state distribution (10x10-dimensional).
            measurements (ndarray): An (N, 5) matrix of N measurements, each in format (x, y, z, a, h) where (x, y, z) is the
                bounding box center position, a the aspect ratio, and h the height.
            only_position (bool): If True, distance computation is done with respect to box center position only.
            metric (str): The metric to use for calculating the distance. Options are 'gaussian' for the squared
                Euclidean distance and 'maha' for the squared Mahalanobis distance.

        Returns:
            (np.ndarray): Returns an array of length N, where the i-th element contains the squared distance between
                (mean, covariance) and `measurements[i]`.

        Examples:
            Compute gating distance using Mahalanobis metric:
            >>> kf = KalmanFilterXYZAH()
            >>> mean = np.array([0, 0, 0, 1, 1, 0, 0, 0, 0, 0])
            >>> covariance = np.eye(10)
            >>> measurements = np.array([[1, 1, 1, 1, 1], [2, 2, 2, 1, 1]])
            >>> distances = kf.gating_distance(mean, covariance, measurements, only_position=False, metric="maha")

        """
        mean_proj, covariance_proj = self.project(mean, covariance)
        if only_position:
            mean_proj, covariance_proj = mean_proj[:3], covariance_proj[:3, :3]
            measurements = measurements[:, :3]

        d = measurements - mean_proj
        if metric == "gaussian":
            return np.sum(d * d, axis=1)
        if metric == "maha":
            # try:
            cholesky_factor = np.linalg.cholesky(covariance_proj)
            z = scipy.linalg.solve_triangular(
                cholesky_factor,
                d.T,
                lower=True,
                check_finite=False,
                overwrite_b=True,
            )
            return np.sum(z * z, axis=0)  # square Mahalanobis
        # except np.linalg.LinAlgError:
        #    # If covariance is not positive definite, fallback to pseudo-inverse
        #    inv_cov = np.linalg.pinv(covariance_proj)
        #    return np.einsum("ij,ij->i", d, np.dot(d, inv_cov))
        raise ValueError("Invalid distance metric")
