import logging
import os
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


@dataclass
class TrackedObject:
    """Data class representing a tracked object with bounding box, class information, and 3D position."""

    track_id: int | None = None
    bbox: list[int] | None = None  # [left, top, right, bottom]
    confidence: float | None = None
    class_id: int | None = None
    label: str | None = None
    position_3d: list[float] | None = None
    dimensions: list[float] | None = None  # [height, width, length]
    # For tuner only
    frame_idx: int | None = None
    depth: float | None = None
    centroid: list[int] | None = None
    score: float | None = None
    cls: int | None = None
    sequence: str | None = None
    orig_img: np.ndarray | None = None


@dataclass
class Frame:
    """Data class representing a single frame with left and right images, timestamps, and labels."""

    image_left: np.ndarray
    image_right: np.ndarray
    timestamp: tuple[float, float]
    labels: list[dict] = field(default_factory=list)

    @property
    def images(self) -> tuple[np.ndarray, np.ndarray]:
        """Tuple of left and right images."""
        return self.image_left, self.image_right

    @property
    def timestamp_left(self) -> float:
        """Timestamp of the left image."""
        return self.timestamp[0]

    @property
    def timestamp_right(self) -> float:
        """Timestamp of the right image."""
        return self.timestamp[1]

    @property
    def num_labels(self) -> int:
        """Number of labels associated with the frame."""
        return len(self.labels)


class DataLoader:
    """Load and process raw and rectified sequences of frames."""

    def __init__(self, config: dict) -> None:
        """
        Initialize the DataLoader with data paths.

        Args:
            config (Dict): Data configuration dictionary.

        """
        self.raw_sequences_paths: list[str] = config.get("raw_sequences_path", [])
        self.rect_sequences_paths: list[str] = config.get("rect_sequences_path", [])
        self.calibration_file: str = config.get("calibration_file", "")
        self.rectified_images_path: str = config.get("rectified_images_path", "")
        self.logger = logging.getLogger("autonomous_perception.data_loader")
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Add cache directory
        self.cache_dir = config.get("cache_dir", "cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    def load_sequences_rect(self, calib_params: dict[str, np.ndarray]) -> dict[str, list[Frame]]:
        """
        Load all rectified sequences of frames based on rectified paths.

        Args:
            calib_params (Dict[str, np.ndarray]): Calibration parameters.

        Returns:
            Dict[str, List[Frame]]: Dictionary of rectified sequences with Frame objects.

        """
        self.logger.info("Loading rectified sequences")

        cache_file = os.path.join(self.cache_dir, "rect_sequences.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                all_frames = pickle.load(f)
            self.logger.info(f"Loaded rectified sequences from cache: {cache_file}")
            return all_frames

        all_frames: dict[str, list[Frame]] = {}
        for idx, rect_seq_path in enumerate(self.rect_sequences_paths, 1):
            self.logger.info(f"Loading rectified sequence {idx}: {rect_seq_path}")
            frames = self._load_sequence(rect_seq_path, calib_params, rectified=True)
            all_frames[str(idx)] = frames
            self.logger.info(f"Loaded {len(frames)} rectified frames from sequence {idx}")

        # Cache the loaded sequences
        with open(cache_file, "wb") as f:
            pickle.dump(all_frames, f)
        self.logger.info(f"Rectified sequences cached at: {cache_file}")

        self.logger.info(f"Total rectified sequences loaded: {len(all_frames)}")
        return all_frames

    def load_sequences_raw(self, calib_params: dict[str, np.ndarray]) -> dict[str, list[Frame]]:
        """
        Load all raw sequences of frames based on raw paths.

        Args:
            calib_params (Dict[str, np.ndarray]): Calibration parameters.

        Returns:
            Dict[str, List[Frame]]: Dictionary of raw sequences with Frame objects.

        """
        self.logger.info("Loading raw sequences")

        cache_file = os.path.join(self.cache_dir, "raw_sequences.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                all_frames = pickle.load(f)
            self.logger.info(f"Loaded raw sequences from cache: {cache_file}")
            return all_frames

        all_frames: dict[str, list[Frame]] = {}
        for idx, raw_seq_path in enumerate(self.raw_sequences_paths, 1):
            rect_seq_path = (
                self.rect_sequences_paths[idx - 1]
                if idx - 1 < len(self.rect_sequences_paths)
                else None
            )
            if not rect_seq_path:
                self.logger.warning(
                    f"No rectified sequence path for raw sequence {raw_seq_path}. Skipping.",
                )
                continue
            self.logger.info(f"Loading raw sequence {idx}: {raw_seq_path}")
            frames = self._load_sequence(raw_seq_path, calib_params, rectified=False)
            all_frames[str(idx)] = frames
            self.logger.info(f"Loaded {len(frames)} raw frames from sequence {idx}")

        # Cache the loaded sequences
        with open(cache_file, "wb") as f:
            pickle.dump(all_frames, f)
        self.logger.info(f"Raw sequences cached at: {cache_file}")

        self.logger.info(f"Total raw sequences loaded: {len(all_frames)}")
        return all_frames

    def _load_sequence(
        self,
        seq_path: str,
        calib_params: dict[str, np.ndarray],
        rectified: bool,
    ) -> list[Frame]:
        """
        Load a single sequence of frames.

        Args:
            seq_path (str): Path to the sequence.
            calib_params (Dict[str, np.ndarray]): Calibration parameters.
            rectified (bool): Whether the sequence is rectified.

        Returns:
            List[Frame]: List of Frame objects for the sequence.

        """
        images_left_path = Path(seq_path) / "image_02" / "data"
        images_right_path = Path(seq_path) / "image_03" / "data"
        timestamps_left_path = Path(seq_path) / "image_02" / "timestamps.txt"
        timestamps_right_path = Path(seq_path) / "image_03" / "timestamps.txt"
        labels_file = Path(seq_path) / "labels.txt"

        # Load timestamps
        timestamps_left = self._load_timestamps(timestamps_left_path.as_posix())
        timestamps_right = self._load_timestamps(timestamps_right_path.as_posix())

        # Load labels if available
        labels = self._load_labels(labels_file.as_posix()) if Path.exists(labels_file) else []

        # Get sorted image files
        image_files_left = self._get_sorted_image_files(images_left_path.as_posix())
        image_files_right = self._get_sorted_image_files(images_right_path.as_posix())

        # Determine number of frames to load
        num_frames = min(
            len(image_files_left),
            len(image_files_right),
            len(timestamps_left),
            len(timestamps_right),
        )
        if num_frames == 0:
            self.logger.warning(f"No frames found in sequence {seq_path}.")
            return []

        self.logger.debug(f"Number of frames to load from {seq_path}: {num_frames}")
        frames: list[Frame] = []
        for i in range(num_frames):
            img_left_file = image_files_left[i]
            img_right_file = image_files_right[i]
            ts_left = timestamps_left[i]
            ts_right = timestamps_right[i]

            img_left = cv2.imread(os.path.join(images_left_path, img_left_file))
            img_right = cv2.imread(os.path.join(images_right_path, img_right_file))
            if img_left is None or img_right is None:
                self.logger.error(
                    f"Failed to load images: {img_left_file}, {img_right_file}. Skipping frame {i + 1}.",
                )
                continue

            frame_number = self._extract_frame_number(img_left_file)
            if frame_number == -1:
                self.logger.warning(f"Invalid frame number for image {img_left_file}. Skipping.")
                continue

            frame_labels = self._get_labels_for_frame(labels, frame_number)

            frame = Frame(
                image_left=img_left,
                image_right=img_right,
                timestamp=(ts_left, ts_right),
                labels=frame_labels,
            )
            frames.append(frame)
            self.logger.debug(f"Loaded frame {i + 1}/{num_frames} from sequence.")
        return frames

    def _load_timestamps(self, filepath: str) -> list[float]:
        """
        Load and parse timestamps from a file.

        Args:
            filepath (str): Path to the timestamps file.

        Returns:
            List[float]: List of timestamps as float seconds since epoch.

        """
        timestamps: list[float] = []
        try:
            with open(filepath) as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        self.logger.warning(f"Empty line at {filepath}:{line_num}. Skipping.")
                        continue
                    try:
                        if "." in line:
                            dt_str, micro = line.split(".")
                            micro = micro[:6].ljust(6, "0")  # Ensure 6 digits
                            dt_str = f"{dt_str}.{micro}"
                        else:
                            dt_str = f"{line}.000000"
                        dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S.%f")
                        timestamp = dt.timestamp()
                        timestamps.append(timestamp)
                    except ValueError:
                        self.logger.exception(
                            f"Error parsing timestamp '{line}' at {filepath}:{line_num}",
                        )
        except FileNotFoundError:
            self.logger.exception(f"Timestamps file not found: {filepath}")
        except Exception:
            self.logger.exception(f"Unexpected error while loading timestamps from {filepath}")
        self.logger.debug(f"Loaded {len(timestamps)} timestamps from {filepath}.")
        return timestamps

    def _load_labels(self, labels_file: str) -> list[dict]:
        """
        Load labels from a labels file.

        Args:
            labels_file (str): Path to the labels file.

        Returns:
            List[Dict]: List of label dictionaries.

        """
        labels: list[dict] = []
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
                            "bbox": list(map(float, parts[6:10])),  # [left, top, right, bottom]
                            "dimensions": list(map(float, parts[10:13])),  # [height, width, length]
                            "location": list(map(float, parts[13:16])),  # [x, y, z]
                            "rotation_y": float(parts[16]),
                            "score": float(parts[17]) if len(parts) > 17 else None,
                        }
                        labels.append(label)
                    except (ValueError, IndexError):
                        self.logger.exception(f"Error parsing label at {labels_file}:{line_num}")
        except FileNotFoundError:
            self.logger.exception(f"Labels file not found: {labels_file}")
        except Exception:
            self.logger.exception(f"Unexpected error while loading labels from {labels_file}")
        self.logger.debug(f"Loaded {len(labels)} labels from {labels_file}.")
        return labels

    def _get_sorted_image_files(self, image_dir: str) -> list[str]:
        """
        Get sorted list of image files in a directory.

        Args:
            image_dir (str): Directory containing image files.

        Returns:
            List[str]: Sorted list of image filenames.

        """
        supported_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
        try:
            files = sorted(
                [
                    f
                    for f in os.listdir(image_dir)
                    if os.path.isfile(os.path.join(image_dir, f))
                    and f.lower().endswith(supported_extensions)
                ],
            )
            self.logger.debug(f"Found {len(files)} image files in {image_dir}.")
        except FileNotFoundError:
            self.logger.exception(f"Image directory not found: {image_dir}")
            return []
        except Exception:
            self.logger.exception(f"Error accessing image directory {image_dir}")
            return []
        else:
            return files

    def _get_labels_for_frame(self, labels: list[dict], frame_number: int) -> list[dict]:
        """
        Retrieve labels associated with a specific frame number.

        Args:
            labels (List[Dict]): All labels.
            frame_number (int): The frame number to filter labels for.

        Returns:
            List[Dict]: List of labels for the specified frame.

        """
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
            self.logger.exception(
                f"Invalid filename format for extracting frame number: {filename}",
            )
            return -1  # Indicates an invalid frame number
