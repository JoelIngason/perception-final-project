import logging

import numpy as np

from src.data_loader.dataset import TrackedObject
from src.evaluation.evaluator import Evaluator
from src.tracking.tracker import ObjectTracker
from src.utils.label_parser import GroundTruthObject


def test_evaluator():
    # Setup logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("test_evaluator")
    logger.setLevel(logging.DEBUG)

    # Example configuration
    config = {
        "tracking": {
            "max_age": 60,
            "min_hits": 1,  # Lowered for testing
            "dt": 1.0,
        },
    }

    # Example calibration parameters (dummy projection matrices)
    calib_params = {
        "P_rect_02": np.array([[700, 0, 600, 0], [0, 700, 300, 0], [0, 0, 1, 0]], dtype=np.float32),
        "P_rect_03": np.array(
            [[700, 0, 600, -700], [0, 700, 300, 0], [0, 0, 1, 0]],
            dtype=np.float32,
        ),
    }

    # Initialize tracker for the left camera
    tracker = ObjectTracker(config, calib_params, "left")

    # Initialize evaluator
    evaluator = Evaluator()

    # Create synthetic ground truth for frame 0
    gt_objects_frame0 = [
        GroundTruthObject(
            frame=0,
            track_id=1,
            obj_type="Car",
            truncated=0.0,
            occluded=0,
            alpha=0.0,
            bbox=[100, 150, 200, 250],  # [left, top, right, bottom]
            dimensions=[1.5, 1.8, 4.0],
            location=[10.0, 0.0, 20.0],
            rotation_y=0.0,
        ),
        GroundTruthObject(
            frame=0,
            track_id=2,
            obj_type="Pedestrian",
            truncated=0.0,
            occluded=0,
            alpha=0.0,
            bbox=[300, 400, 350, 500],
            dimensions=[1.7, 0.6, 0.6],
            location=[15.0, 0.0, 25.0],
            rotation_y=0.0,
        ),
    ]

    # Create synthetic tracked objects for frame 0
    tracked_objects_frame0 = [
        TrackedObject(
            track_id=0,
            bbox=[105, 155, 195, 245],  # Slightly shifted bounding box
            confidence=0.9,
            class_id=1,
            label="Car",
            position_3d=[10.5, 0.0, 20.5],
            dimensions=[1.6, 1.7, 4.1],
            mask=None,
        ),
        TrackedObject(
            track_id=1,
            bbox=[305, 405, 345, 495],
            confidence=0.85,
            class_id=2,
            label="Pedestrian",
            position_3d=[15.2, 0.0, 25.3],
            dimensions=[1.6, 0.6, 0.5],
            mask=None,
        ),
        TrackedObject(
            track_id=2,
            bbox=[400, 500, 450, 600],  # Extra detection (Cyclist)
            confidence=0.75,
            class_id=3,
            label="Cyclist",
            position_3d=[20.0, 0.0, 30.0],
            dimensions=[1.8, 0.5, 1.8],
            mask=None,
        ),
    ]

    # Evaluate frame 0
    evaluator.evaluate(tracked_objects_frame0, gt_objects_frame0)

    # Create synthetic ground truth for frame 1
    gt_objects_frame1 = [
        GroundTruthObject(
            frame=1,
            track_id=1,
            obj_type="Car",
            truncated=0.0,
            occluded=0,
            alpha=0.0,
            bbox=[110, 160, 210, 260],
            dimensions=[1.5, 1.8, 4.0],
            location=[10.5, 0.0, 20.5],
            rotation_y=0.0,
        ),
    ]

    # Create synthetic tracked objects for frame 1
    tracked_objects_frame1 = [
        TrackedObject(
            track_id=0,
            bbox=[115, 165, 205, 255],
            confidence=0.92,
            class_id=1,
            label="Car",
            position_3d=[11.0, 0.0, 21.0],
            dimensions=[1.6, 1.7, 4.1],
            mask=None,
        ),
    ]

    # Evaluate frame 1
    evaluator.evaluate(tracked_objects_frame1, gt_objects_frame1)

    # Report metrics
    evaluator.report()


if __name__ == "__main__":
    test_evaluator()
