import logging
import os

import cv2
import numpy as np


class Frame:
    def __init__(self, image_left: np.ndarray, image_right: np.ndarray, timestamp: tuple[float, float], labels: list[dict] | None = None):
        self.images = (image_left, image_right)
        self.timestamp = timestamp
        self.labels = labels

class DataLoader:
    def __init__(self, config):
        self.raw_sequences_path = config['raw_sequences_path']
        self.rect_sequences_path = config['rect_sequences_path']
        self.calibration_file = config['calibration_file']
        self.rectified_images_path = config['rectified_images_path']
        self.labels_path = config['labels_path']
        self.logger = logging.getLogger('autonomous_perception.data_loader')

    def load_sequences(self, calib_params):
        # Load raw and rectified sequences
        self.logger.info("Loading raw and rectified sequences")
        raw_sequences = sorted(os.listdir(self.raw_sequences_path))
        rect_sequences = sorted(os.listdir(self.rect_sequences_path))
        all_frames = []

        for raw_seq, rect_seq in zip(raw_sequences, rect_sequences, strict=False):
            self.logger.info(f"Loading sequence: {raw_seq}")
            raw_seq_path = os.path.join(self.raw_sequences_path, raw_seq)
            rect_seq_path = os.path.join(self.rect_sequences_path, rect_seq)
            frames = self._load_sequence(raw_seq_path, rect_seq_path)
            all_frames.extend(frames)

        return all_frames

    def _load_sequence(self, raw_seq_path: str, rect_seq_path: str) -> list[Frame]:
        # Paths for raw images
        raw_image02_path = os.path.join(raw_seq_path, 'image02', 'data')
        raw_image03_path = os.path.join(raw_seq_path, 'image03', 'data')
        raw_timestamps02 = self._load_timestamps(os.path.join(raw_seq_path, 'image02', 'timestamps.txt'))
        raw_timestamps03 = self._load_timestamps(os.path.join(raw_seq_path, 'image03', 'timestamps.txt'))

        # Paths for rectified images
        rect_image02_path = os.path.join(rect_seq_path, 'rectified_images', 'image02', 'data')
        rect_image03_path = os.path.join(rect_seq_path, 'rectified_images', 'image03', 'data')
        rect_timestamps02 = self._load_timestamps(os.path.join(rect_seq_path, 'rectified_images', 'image02', 'timestamps.txt'))
        rect_timestamps03 = self._load_timestamps(os.path.join(rect_seq_path, 'rectified_images', 'image03', 'timestamps.txt'))

        # Load labels
        labels_file = os.path.join(self.labels_path, f"labels_{raw_seq_path.split('_')[-1]}.txt")
        labels = self._load_labels(labels_file) if os.path.exists(labels_file) else None

        raw_image_files02 = sorted(os.listdir(raw_image02_path))
        raw_image_files03 = sorted(os.listdir(raw_image03_path))
        rect_image_files02 = sorted(os.listdir(rect_image02_path))
        rect_image_files03 = sorted(os.listdir(rect_image03_path))

        frames = []
        for raw_img02, raw_img03, rect_img02, rect_img03, ts02, ts03 in zip(
            raw_image_files02, raw_image_files03,
            rect_image_files02, rect_image_files03,
            raw_timestamps02, raw_timestamps03, strict=False,
        ):
            # Load raw images if needed
            # img_left_raw = cv2.imread(os.path.join(raw_image02_path, raw_img02))
            # img_right_raw = cv2.imread(os.path.join(raw_image03_path, raw_img03))

            # Load rectified images
            img_left = cv2.imread(os.path.join(rect_image02_path, rect_img02))
            img_right = cv2.imread(os.path.join(rect_image03_path, rect_img03))

            frame_labels = self._get_labels_for_frame(labels, frame_number=self._extract_frame_number(raw_img02))
            frame = Frame(
                image_left=img_left,
                image_right=img_right,
                timestamp=(ts02, ts03),
                labels=frame_labels,
            )
            frames.append(frame)

        return frames

    def _load_timestamps(self, filepath: str) -> list[float]:
        with open(filepath) as f:
            timestamps = [float(line.strip()) for line in f]
        return timestamps

    def _load_labels(self, labels_file: str) -> list[dict]:
        labels = []
        with open(labels_file) as f:
            for line in f:
                parts = line.strip().split()
                label = {
                    'frame': int(parts[0]),
                    'track_id': int(parts[1]),
                    'type': parts[2],
                    'truncated': float(parts[3]),
                    'occluded': int(parts[4]),
                    'alpha': float(parts[5]),
                    'bbox': list(map(float, parts[6:10])),
                    'dimensions': list(map(float, parts[10:13])),
                    'location': list(map(float, parts[13:16])),
                    'rotation_y': float(parts[16]),
                    'score': float(parts[17]) if len(parts) > 17 else None,
                }
                labels.append(label)
        return labels

    def _get_labels_for_frame(self, labels: list[dict], frame_number: int) -> list[dict]:
        if labels is None:
            return []
        return [label for label in labels if label['frame'] == frame_number]

    def _extract_frame_number(self, filename: str) -> int:
        # Extract frame number from filename e.g., "000001.png" -> 1
        return int(os.path.splitext(filename)[0])
