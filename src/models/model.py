import logging
from pathlib import Path

from ultralytics import YOLO


def load_model(model_path: str = "yolov8/best.pt") -> YOLO:
    """
    Load a pretrained YOLO model.

    Args:
        model_path (str, optional): Path to the YOLO weights file. Defaults to "yolov8/best.pt".

    Returns:
        YOLO: The loaded YOLO model instance.

    Raises:
        FileNotFoundError: If the model file does not exist.

    """
    logger = logging.getLogger("autonomous_perception.models")
    logger.info(f"Loading YOLO model from {model_path}")
    return YOLO(model_path, verbose=False)


def get_model(model_path: str | None = None) -> YOLO:
    """
    Get the YOLO model instance.

    Args:
        model_path (Optional[str], optional): Path to the YOLO weights file. If None, uses default path. Defaults to None.

    Returns:
        YOLO: The YOLO model instance.

    """
    if model_path is None:
        model_path = str(Path(__file__).parent / "yolov8/best.pt")
    return load_model(str(model_path))
