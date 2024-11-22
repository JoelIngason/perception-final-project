import logging
from dataclasses import dataclass


@dataclass
class GroundTruthObject:
    """
    Represents a ground truth object in a frame.

    Attributes:
        frame (int): Frame number within the sequence where the object appears.
        track_id (int): Unique tracking ID of the object within the sequence.
        obj_type (str): Type of the object (e.g., 'Car', 'Pedestrian', 'Cyclist').
        truncated (float): Float from 0 (non-truncated) to 1 (truncated), indicating if the object is truncated in the image.
        occluded (int): Integer (0,1,2,3) indicating occlusion state:
                        0 = fully visible,
                        1 = partly occluded,
                        2 = largely occluded,
                        3 = unknown.
        alpha (float): Observation angle of the object, ranging [-π; π].
        bbox (List[float]): 2D bounding box in the rectified image [left, top, right, bottom].
        dimensions (List[float]): 3D object dimensions [height, width, length] in meters.
        location (List[float]): 3D object location [x, y, z] in camera coordinates in meters.
        rotation_y (float): Rotation around Y-axis in camera coordinates [-π; π].
        score (float): Confidence score (set to 0.0 for ground truth).

    """

    frame: int
    track_id: int
    obj_type: str
    truncated: float
    occluded: int
    alpha: float
    bbox: list[float]  # [left, top, right, bottom]
    dimensions: list[float]  # [height, width, length]
    location: list[float]  # [x, y, z]
    rotation_y: float
    score: float = 0.0  # Not used for ground truth


def parse_labels(label_file: str) -> dict[int, list[GroundTruthObject]]:
    """
    Parse the labels.txt file and organize ground truth objects by frame number.

    Args:
        label_file (str): Path to the labels.txt file.

    Returns:
        Dict[int, List[GroundTruthObject]]: Dictionary mapping frame numbers to lists of GroundTruthObject.

    """
    gt_dict: dict[int, list[GroundTruthObject]] = {}
    logger = logging.getLogger("autonomous_perception.label_parser")

    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    try:
        with open(label_file) as f:
            for line_num, line in enumerate(f, 1):
                parts = line.strip().split()
                if len(parts) < 17:
                    logger.warning(
                        f"Line {line_num}: Insufficient data ({len(parts)} columns). Expected at least 17. Skipping line.",
                    )
                    continue  # Skip invalid lines

                try:
                    frame = int(parts[0])
                    track_id = int(parts[1])
                    obj_type = parts[2]
                    truncated = float(parts[3])
                    occluded = int(parts[4])
                    alpha = float(parts[5])
                    bbox = list(map(float, parts[6:10]))  # [left, top, right, bottom]
                    dimensions = list(map(float, parts[10:13]))  # [height, width, length]
                    location = list(map(float, parts[13:16]))  # [x, y, z]
                    rotation_y = float(parts[16])

                    gt_obj = GroundTruthObject(
                        frame=frame,
                        track_id=track_id,
                        obj_type=obj_type,
                        truncated=truncated,
                        occluded=occluded,
                        alpha=alpha,
                        bbox=bbox,
                        dimensions=dimensions,
                        location=location,
                        rotation_y=rotation_y,
                        score=0.0,  # Ground truth does not have a score
                    )

                    if frame not in gt_dict:
                        gt_dict[frame] = []
                    gt_dict[frame].append(gt_obj)

                except ValueError as ve:
                    logger.error(f"Line {line_num}: Value conversion error: {ve}. Skipping line.")
                except IndexError as ie:
                    logger.error(f"Line {line_num}: Index error: {ie}. Skipping line.")

    except FileNotFoundError:
        logger.exception(f"Label file '{label_file}' not found.")
    except Exception as e:
        logger.exception(f"An unexpected error occurred while parsing labels: {e}")

    logger.info(f"Parsed {len(gt_dict)} frames from '{label_file}'.")
    return gt_dict
