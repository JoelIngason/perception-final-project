import logging
import os
from datetime import datetime

import cv2
import numpy as np


class Frame:
    def __init__(
        self,
        image_left: np.ndarray,
        image_right: np.ndarray,
        timestamp: tuple[float, float],
        labels: list[dict] | None = None,
    ):
        self.images = (image_left, image_right)
        self.timestamp = timestamp
        self.labels = labels if labels is not None else []


class DataLoader:
    def __init__(self, config):
        self.raw_sequences_paths = config["raw_sequences_path"]
        self.rect_sequences_paths = config["rect_sequences_path"]
        self.calibration_file = config["calibration_file"]
        self.rectified_images_path = config["rectified_images_path"]
        self.logger = logging.getLogger("autonomous_perception.data_loader")

        # Configure logger if not already configured
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def load_sequences_rect(self, calib_params) -> dict[str, list[Frame]]:
        """
        Load all rectified sequences of frames based on rectified paths.

        Args:
            calib_params (dict[str, np.ndarray]): Calibration parameters.

        Returns:
            Dict[str, List[Frame]]: A dictionary of rectified sequences with Frame objects.

        """
        self.logger.info("Loading rectified sequences")
        all_frames = {}

        num_sequences = len(self.rect_sequences_paths)

        for idx in range(num_sequences):
            rect_seq_path = self.rect_sequences_paths[idx]
            self.logger.info(f"Loading rectified sequence {idx + 1}: {rect_seq_path}")
            frames = self._load_sequence_rectified(rect_seq_path)
            all_frames[str(idx + 1)] = frames

        self.logger.info(f"Loaded a total of {len(all_frames)} rectified sequences.")
        return all_frames

    def load_sequences_raw(self, calib_params) -> dict[str, list[Frame]]:
        """
        Load all raw sequences of frames based on raw paths.

        Args:
            calib_params (dict[str, np.ndarray]): Calibration parameters.

        Returns:
            Dict[str, List[Frame]]: A dictionary of raw sequences with Frame objects.

        """
        self.logger.info("Loading raw sequences")
        all_frames = {}

        num_sequences = len(self.raw_sequences_paths)

        for idx in range(num_sequences):
            raw_seq_path = self.raw_sequences_paths[idx]
            rect_seq_path = (
                self.rect_sequences_paths[idx] if idx < len(self.rect_sequences_paths) else None
            )

            if rect_seq_path is None:
                self.logger.warning(
                    f"No rectified sequence path for raw sequence {raw_seq_path}. Skipping.",
                )
                continue

            self.logger.info(f"Loading raw sequence {idx + 1}: {raw_seq_path}")
            frames = self._load_sequence_raw(raw_seq_path, rect_seq_path)
            all_frames[str(idx + 1)] = frames

        self.logger.info(f"Loaded a total of {len(all_frames)} raw frames.")
        return all_frames

    def _load_sequence_rectified(self, rect_seq_path: str) -> list[Frame]:
        """
        Load a single rectified sequence of frames.

        Args:
            rect_seq_path (str): Path to the rectified sequence.

        Returns:
            List[Frame]: A list of Frame objects for the rectified sequence.

        """
        self.logger.debug(f"Loading rectified sequence from {rect_seq_path}")

        # Paths for rectified images
        rect_image_02_path = os.path.join(rect_seq_path, "image_02", "data")
        rect_image_03_path = os.path.join(rect_seq_path, "image_03", "data")
        rect_timestamps02_path = os.path.join(rect_seq_path, "image_02", "timestamps.txt")
        rect_timestamps03_path = os.path.join(rect_seq_path, "image_03", "timestamps.txt")

        # Load timestamps
        rect_timestamps02 = self._load_timestamps(rect_timestamps02_path)
        rect_timestamps03 = self._load_timestamps(rect_timestamps03_path)

        # Load labels
        labels_file = os.path.join(rect_seq_path, "labels.txt")
        labels = self._load_labels(labels_file) if os.path.exists(labels_file) else None

        # Get sorted image file lists
        rect_image_files02 = sorted(
            [
                f
                for f in os.listdir(rect_image_02_path)
                if os.path.isfile(os.path.join(rect_image_02_path, f))
                and f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
            ],
        )
        rect_image_files03 = sorted(
            [
                f
                for f in os.listdir(rect_image_03_path)
                if os.path.isfile(os.path.join(rect_image_03_path, f))
                and f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
            ],
        )

        # Log the number of image files found
        self.logger.debug(
            f"Found {len(rect_image_files02)} rectified left images and {len(rect_image_files03)} rectified right images.",
        )

        # Ensure that the number of images and timestamps match
        num_frames = min(
            len(rect_image_files02),
            len(rect_image_files03),
            len(rect_timestamps02),
            len(rect_timestamps03),
        )

        if num_frames == 0:
            self.logger.warning(f"No frames found in rectified sequence {rect_seq_path}.")
            return []

        self.logger.debug(f"Number of frames to load: {num_frames}")

        frames = []
        for i in range(num_frames):
            rect_img02 = rect_image_files02[i]
            rect_img03 = rect_image_files03[i]
            ts02 = rect_timestamps02[i]
            ts03 = rect_timestamps03[i]

            # Load rectified images
            img_left = cv2.imread(os.path.join(rect_image_02_path, rect_img02))
            img_right = cv2.imread(os.path.join(rect_image_03_path, rect_img03))

            if img_left is None or img_right is None:
                self.logger.error(
                    f"Failed to load rectified images: {rect_img02}, {rect_img03}. Skipping frame {i}.",
                )
                continue

            frame_number = self._extract_frame_number(rect_img02)
            frame_labels = self._get_labels_for_frame(labels, frame_number)

            frame = Frame(
                image_left=img_left,
                image_right=img_right,
                timestamp=(ts02, ts03),
                labels=frame_labels,
            )
            frames.append(frame)

        self.logger.debug(f"Loaded {len(frames)} rectified frames from sequence.")
        return frames

    def _load_sequence_raw(self, raw_seq_path: str, rect_seq_path: str) -> list[Frame]:
        """
        Load a single raw sequence of frames.

        Args:
            raw_seq_path (str): Path to the raw sequence.
            rect_seq_path (str): Path to the corresponding rectified sequence.

        Returns:
            List[Frame]: A list of Frame objects for the raw sequence.

        """
        self.logger.debug(f"Loading raw sequence from {raw_seq_path}")

        # Paths for raw images
        raw_image_02_path = os.path.join(raw_seq_path, "image_02", "data")
        raw_image_03_path = os.path.join(raw_seq_path, "image_03", "data")
        raw_timestamps02_path = os.path.join(raw_seq_path, "image_02", "timestamps.txt")
        raw_timestamps03_path = os.path.join(raw_seq_path, "image_03", "timestamps.txt")

        # Paths for rectified images (for labels and other metadata)
        rect_image_02_path = os.path.join(rect_seq_path, "image_02", "data")
        rect_image_03_path = os.path.join(rect_seq_path, "image_03", "data")
        rect_timestamps02_path = os.path.join(rect_seq_path, "image_02", "timestamps.txt")
        rect_timestamps03_path = os.path.join(rect_seq_path, "image_03", "timestamps.txt")

        # Load timestamps
        raw_timestamps02 = self._load_timestamps(raw_timestamps02_path)
        raw_timestamps03 = self._load_timestamps(raw_timestamps03_path)

        # Load labels
        labels_file = os.path.join(raw_seq_path, "labels.txt")
        labels = self._load_labels(labels_file) if os.path.exists(labels_file) else None

        # Get sorted image file lists
        raw_image_files02 = sorted(
            [
                f
                for f in os.listdir(raw_image_02_path)
                if os.path.isfile(os.path.join(raw_image_02_path, f))
                and f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
            ],
        )
        raw_image_files03 = sorted(
            [
                f
                for f in os.listdir(raw_image_03_path)
                if os.path.isfile(os.path.join(raw_image_03_path, f))
                and f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
            ],
        )

        # Log the number of image files found
        self.logger.debug(
            f"Found {len(raw_image_files02)} raw left images and {len(raw_image_files03)} raw right images.",
        )

        # Ensure that the number of images and timestamps match
        num_frames = min(
            len(raw_image_files02),
            len(raw_image_files03),
            len(raw_timestamps02),
            len(raw_timestamps03),
        )

        if num_frames == 0:
            self.logger.warning(f"No frames found in raw sequence {raw_seq_path}.")
            return []

        self.logger.debug(f"Number of frames to load: {num_frames}")

        frames = []
        for i in range(num_frames):
            raw_img02 = raw_image_files02[i]
            raw_img03 = raw_image_files03[i]
            ts02 = raw_timestamps02[i]
            ts03 = raw_timestamps03[i]

            # Load raw images
            img_left = cv2.imread(os.path.join(raw_image_02_path, raw_img02))
            img_right = cv2.imread(os.path.join(raw_image_03_path, raw_img03))

            if img_left is None or img_right is None:
                self.logger.error(
                    f"Failed to load raw images: {raw_img02}, {raw_img03}. Skipping frame {i}.",
                )
                continue

            frame_number = self._extract_frame_number(raw_img02)
            frame_labels = self._get_labels_for_frame(labels, frame_number)

            frame = Frame(
                image_left=img_left,
                image_right=img_right,
                timestamp=(ts02, ts03),
                labels=frame_labels,
            )
            frames.append(frame)

        self.logger.debug(f"Loaded {len(frames)} raw frames from sequence.")
        return frames

    def _load_timestamps(self, filepath: str) -> list[float]:
        """
        Load and parse timestamps from a file.

        Args:
            filepath (str): Path to the timestamps file.

        Returns:
            List[float]: A list of timestamps as float seconds since epoch.

        """
        timestamps = []
        try:
            with open(filepath) as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        self.logger.warning(f"Empty line at {filepath}:{line_num}. Skipping.")
                        continue
                    try:
                        # Truncate nanoseconds to microseconds
                        if "." in line:
                            dt_str, micro = line.split(".")
                            micro = micro[:6].ljust(6, "0")  # Ensure 6 digits
                            dt_str = f"{dt_str}.{micro}"
                        else:
                            dt_str = line
                        dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S.%f")
                        timestamp = dt.timestamp()
                        timestamps.append(timestamp)
                    except ValueError as e:
                        self.logger.error(
                            f"Error parsing timestamp '{line}' at {filepath}:{line_num}: {e}",
                        )
        except FileNotFoundError:
            self.logger.error(f"Timestamps file not found: {filepath}")
        except Exception as e:
            self.logger.error(f"Unexpected error while loading timestamps: {e}")

        self.logger.debug(f"Loaded {len(timestamps)} timestamps from {filepath}.")
        return timestamps

    def _load_labels(self, labels_file: str) -> list[dict]:
        """
        Load labels from a labels file.

        Args:
            labels_file (str): Path to the labels file.

        Returns:
            List[dict]: A list of label dictionaries.

        """
        labels = []
        try:
            with open(labels_file) as f:
                for line_num, line in enumerate(f, 1):
                    parts = line.strip().split()
                    if len(parts) < 17:
                        self.logger.warning(
                            f"Insufficient label data at {labels_file}:{line_num}. Skipping.",
                        )
                        continue
                    try:
                        label = {
                            "frame": int(parts[0]),
                            "track_id": int(parts[1]),
                            "type": parts[2],
                            "truncated": float(parts[3]),
                            "occluded": int(parts[4]),
                            "alpha": float(parts[5]),
                            "bbox": list(map(float, parts[6:10])),
                            "dimensions": list(map(float, parts[10:13])),
                            "location": list(map(float, parts[13:16])),
                            "rotation_y": float(parts[16]),
                            "score": float(parts[17]) if len(parts) > 17 else None,
                        }
                        labels.append(label)
                    except (ValueError, IndexError) as e:
                        self.logger.error(f"Error parsing label at {labels_file}:{line_num}: {e}")
        except FileNotFoundError:
            self.logger.error(f"Labels file not found: {labels_file}")
        except Exception as e:
            self.logger.error(f"Unexpected error while loading labels: {e}")

        self.logger.debug(f"Loaded {len(labels)} labels from {labels_file}.")
        return labels

    def _get_labels_for_frame(self, labels: list[dict] | None, frame_number: int) -> list[dict]:
        """
        Retrieve labels associated with a specific frame number.

        Args:
            labels (Optional[List[dict]]): All labels.
            frame_number (int): The frame number to filter labels for.

        Returns:
            List[dict]: A list of labels for the specified frame.

        """
        if not labels:
            return []
        frame_labels = [label for label in labels if label["frame"] == frame_number]
        self.logger.debug(f"Found {len(frame_labels)} labels for frame number {frame_number}.")
        return frame_labels

    def _extract_frame_number(self, filename: str) -> int:
        """
        Extract the frame number from the filename.

        Args:
            filename (str): Filename in the format '000001.png'.

        Returns:
            int: The frame number.

        """
        try:
            frame_number = int(os.path.splitext(filename)[0])
            return frame_number
        except ValueError:
            self.logger.error(f"Invalid filename format for extracting frame number: {filename}")
            return -1  # Indicates an invalid frame number
