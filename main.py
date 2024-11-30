import argparse
import logging
import logging.config
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from ultralytics.utils.plotting import Annotator

from src.calibration.stereo_calibration import StereoCalibrator
from src.data_loader.dataset import DataLoader
from src.depth_estimation.depth_estimator import DepthEstimator
from src.models.feature_extractor.feature_extractor import FeatureExtractor
from src.models.model import load_model
from src.tracking.byte_tracker import BYTETracker


def plot(
    orig_img,
    tracks_active,
    tracks_lost,
    names,
    conf=True,
    line_width=10,
    font_size=20,
    font="Arial.ttf",
    pil=False,
    labels=True,
    show=False,
    save=False,
    filename=None,
    color_mode="class",
):
    """
    Plot tracking results on an input RGB image.

    Args:
        orig_img (np.ndarray): Original image as a numpy array.
        tracks_active (List[Track]): List of confirmed tracks.
        tracks_lost (List[Track]): List of lost tracks.
        names (Dict[int, str]): Dictionary mapping class IDs to class names.
        conf (bool): Whether to display confidence scores.
        line_width (int): Width of bounding box lines.
        font_size (float): Font size for class labels.
        font (str): Font type for class labels.
        pil (bool): Whether to use the Python Imaging Library (PIL).
        img (np.ndarray): Image as a numpy array.
        labels (bool): Whether to display class labels.
        show (bool): Whether to display the annotated image.
        save (bool): Whether to save the annotated image.
        filename (str): Name of the file to save the annotated image.
        color_mode (str): Color mode for

    Returns:
        (np.ndarray): Annotated image as a numpy array.

    """
    assert color_mode in {
        "instance",
        "class",
    }, f"Expected color_mode='instance' or 'class', not {color_mode}."
    annotator = Annotator(
        deepcopy(orig_img),
        line_width=line_width,
        font_size=font_size,
        font=font,
        pil=pil,
    )

    # Combine confirmed and lost tracks
    all_tracks = tracks_active + tracks_lost

    print(f"All tracks: {len(all_tracks)}")

    # Generate a color map for track IDs based on class
    # Red for pedestrians, green for cyclists, blue for vehicles
    # pedastrians: 0, cyclists: 1, vehicles: 2
    colours = defaultdict(lambda: (0, 0, 0))
    if color_mode == "class":
        colours = {
            0: (255, 0, 0),  # Red
            1: (0, 255, 0),  # Green
            2: (0, 0, 255),  # Blue
        }
    else:
        # color everything as red
        colours = defaultdict(lambda: (255, 0, 0))
    for track in all_tracks:
        # Get bounding box in xyzxy format
        bbox = track.xyzxy  # [x1, y1, z, x2, y2]

        # Check if depth is available
        depth = bbox[2] if len(bbox) > 2 else None

        bbox = [bbox[0], bbox[1], bbox[3], bbox[4]]  # [x1, y1, x2, y2]
        bbox = bbox.tolist() if isinstance(bbox, torch.Tensor) else bbox

        # Cap the bbox values to be within the image and not negative
        bbox[0] = min(max(0, bbox[0]), orig_img.shape[1])
        bbox[1] = min(max(0, bbox[1]), orig_img.shape[0])
        bbox[2] = max(min(orig_img.shape[1], bbox[2]), 0)
        bbox[3] = max(min(orig_img.shape[0], bbox[3]), 0)

        if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
            continue

        # Get class, score, and track ID
        cls_id = int(track.cls) if track.cls is not None else -1
        cls_name = names.get(cls_id, "Unknown") if cls_id != -1 else "Unknown"
        score = float(track.score) if track.score is not None else 0.0
        track_id = track.track_id

        # Define label
        if labels:
            if depth is not None and depth > 0:
                label = (
                    f"ID:{track_id} {cls_name} {score:.2f} D: {depth:.2f}"
                    if conf
                    else f"ID:{track_id} {cls_name} D: {depth:.2f}"
                )
            else:
                label = (
                    f"ID:{track_id} {cls_name} {score:.2f}" if conf else f"ID:{track_id} {cls_name}"
                )
        else:
            label = f"ID:{track_id}" if conf else f"ID:{track_id}"

        # Choose color
        color = colours[cls_id]

        # Draw bounding box
        annotator.box_label(bbox, label, color=color)

        # Optionally, draw track history
        # We use draw_centroid_and_tracks
        bboxes = track.history.values()
        print(f"Track {track_id} has {len(bboxes)} history entries.")
        points_to_draw = []
        for bbox in bboxes:
            # Get centroid of bbox
            x, y, z, w, h = bbox  # top-left x, top-left y, z, width, height
            x_center = x + w / 2
            y_center = y + h / 2

            # If the centroid is outside the image, skip
            if (
                x_center < 0
                or x_center >= orig_img.shape[1]
                or y_center < 0
                or y_center >= orig_img.shape[0]
            ):
                continue

            points_to_draw.append((x_center, y_center))

        # Remove all but the last 10 points
        points_to_draw = points_to_draw[-10:]
        if len(points_to_draw) > 0:
            annotator.draw_centroid_and_tracks(points_to_draw, color=color)

    # Save results
    if save:
        if filename is None:
            filename = "annotated_" + Path("annotated_image.jpg").name
        annotator.save(filename)

    return annotator.result()


def setup_logger(name: str, config_file: str | None = None) -> logging.Logger:
    """Set up logging based on the provided configuration."""
    if config_file:
        with open(config_file) as f:
            config = yaml.safe_load(f)
            logging.config.dictConfig(config)
    logger = logging.getLogger(name)
    return logger


@torch.no_grad()
def run(config_path: str) -> None:
    # Load configuration
    with open(config_path) as file:
        config = yaml.safe_load(file)

    # Setup logger
    logger = setup_logger("autonomous_perception", config.get("logging", {}).get("config_file", ""))
    logger.info(f"Configuration loaded from {config_path}")

    # Initialize calibrator
    use_rectified_data = config["processing"].get("use_rectified_data", True)
    calibrator = StereoCalibrator(config["calibration"])
    calibration_file_key = "calibration_file_rect" if use_rectified_data else "calibration_file_raw"
    calibration_file = config["calibration"].get(calibration_file_key)

    calib_params = calibrator.calibrate(calibration_file)

    depth_estimator = DepthEstimator(calib_params, config)

    # Initialize components
    data_loader = DataLoader(config["data"])
    # Initialize YOLO model with tracking
    yolo = load_model(config["models"]["classification"]["path"])
    # Load data
    data = (
        data_loader.load_sequences_rect(calib_params)
        if use_rectified_data
        else data_loader.load_sequences_raw(calib_params)
    )
    # Retrieve class names from the model
    names = yolo.names  # Ensure the model has a 'names' attribute

    # Initialize FeatureExtractor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feature_extractor = FeatureExtractor(device=device)

    # load bytetracker.yaml args
    with open("byte_tracker.yaml") as f:
        args_byte_tracker = yaml.safe_load(f)
    # convert from key-value to namespace
    args_byte_tracker = argparse.Namespace(**args_byte_tracker)
    tracker = BYTETracker(args_byte_tracker, frame_rate=10, feature_extractor=feature_extractor)

    # Make OpenCV window resizable
    cv2.namedWindow("Autonomous Perception", cv2.WINDOW_NORMAL)

    for seq_id, frames in data.items():
        logger.info(f"Processing sequence {seq_id} with {len(frames)} frames.")
        tracker.reset()

        for frame_idx, frame in enumerate(frames, 1):
            # Perform tracking
            img_left = frame.image_left
            img_right = frame.image_right

            # Compute disparity and depth maps
            disparity_map = depth_estimator.compute_disparity_map(img_left, img_right)
            depth_map = depth_estimator.compute_depth_map(disparity_map)

            # Perform object detection
            results = yolo(img_left)[0].boxes.cpu().numpy()
            print(f"Frame {frame_idx}: {len(results)} detections.")

            # Add zs to detection results
            # Assuming results have xywh format: [x_center, y_center, width, height]
            # and we need to insert z (depth)
            detections = []
            measurement_masks = []
            for i, result in enumerate(results):
                bbox = result.xywh[0].tolist()  # [x_center, y_center, width, height]
                depth, object_depth_range = depth_estimator.compute_object_depth(bbox, depth_map)
                if (
                    depth is not None and not np.isnan(depth) and 0 < depth < 6
                ):  # Example valid depth range
                    # Valid depth
                    detection = [bbox[0], bbox[1], depth, bbox[2], bbox[3]]
                    mask = [True, True, True, True, True]
                else:
                    # Missing or invalid depth
                    detection = [bbox[0], bbox[1], 0.0, bbox[2], bbox[3]]
                    mask = [True, True, False, True, True]
                detections.append(detection)
                measurement_masks.append(mask)

            detections = np.array(detections, dtype=np.float32)
            measurement_masks = np.array(measurement_masks, dtype=bool)

            # Create additional columns for scores and class labels if not already present
            # Assuming 'result' has 'conf' and 'cls' attributes
            if len(results) > 0 and hasattr(results[0], "conf") and hasattr(results[0], "cls"):
                scores = np.array([res.conf for res in results], dtype=np.float32)
                cls_labels = np.array([res.cls for res in results], dtype=np.int32)
                # Ensure that detections and measurement_masks are filtered based on valid_depth
                scores = scores[: len(detections)]
                cls_labels = cls_labels[: len(detections)]
            else:
                # Default scores and class labels
                scores = np.ones(len(detections), dtype=np.float32)
                cls_labels = np.zeros(len(detections), dtype=np.int32)

            # Reshape scores and cls_labels to 2D arrays
            scores = scores.reshape(-1, 1)  # Shape: (N, 1)
            cls_labels = cls_labels.reshape(-1, 1)  # Shape: (N, 1)

            # Concatenate detections with scores and class labels
            detections = np.hstack((detections, scores, cls_labels))

            # Update tracker with detections and measurement masks
            _ = tracker.update(detections, measurement_masks, img_left)

            tracks_lost = tracker.lost_stracks
            tracks_active = tracker.tracked_stracks

            print(
                f"Frame {frame_idx}: {len(tracks_active)} active tracks, {len(tracks_lost)} lost tracks.",
            )

            # Annotate frame with tracks
            annotated_frame = plot(
                orig_img=img_left,
                tracks_active=tracks_active,
                tracks_lost=tracks_lost,
                names=names,
                conf=True,
                line_width=2,
                font_size=15,
                labels=True,
                show=True,  # Set to True to display the annotated image
                save=False,  # Set to True to save the annotated image
                filename=None,  # Specify a filename if saving is enabled
                color_mode="class",
            )

            # Display the annotated frame
            cv2.imshow("Autonomous Perception", annotated_frame)

            # Handle exit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Exit key pressed. Closing visualization windows.")
                cv2.destroyAllWindows()
                return  # Exit the run function

    # Release resources and close windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced Autonomous Perception Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to the configuration YAML file",
    )
    args = parser.parse_args()
    run(args.config)
