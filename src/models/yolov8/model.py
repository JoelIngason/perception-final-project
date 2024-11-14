from pathlib import Path

from ultralytics import YOLO


def load_model(model_path: str = "yolov8n.pt") -> YOLO:
    """
    Load a pretrained YOLO model.

    Args:
        model_path (str, optional): Path to the YOLO weights file. Defaults to "yolov8n.pt".

    Returns:
        YOLO: The loaded YOLO model instance.

    """
    if not Path(model_path).exists():
        msg = f"YOLO model file not found at {model_path}"
        raise FileNotFoundError(msg)
    return YOLO(model_path, verbose=False)


def get_model():
    """
    Get the YOLO model instance.

    Returns:
        YOLO: The YOLO model instance.

    """
    path = Path(__file__).parent / "yolov8n.pt"
    return load_model(str(path))
