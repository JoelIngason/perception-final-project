# detection_result.py
class DetectionResult:
    """DetectionResult class to store detection information."""

    def __init__(self, bbox: list[float], confidence: float, class_id: int, label: str):
        """
        Initialize a DetectionResult object.

        Args:
            bbox (List[float]): Bounding box coordinates [x, y, w, h].
            confidence (float): Confidence score of the detection.
            class_id (int): Class ID of the detected object.
            label (str): Label of the detected object.

        """
        self.bbox = bbox  # [x, y, w, h]
        self.confidence = confidence
        self.class_id = class_id
        self.position_3d: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.label = label
        self.appearance_feature = None  # Add this line

    def add_3d_position(self, position_3d: tuple[float, float, float]) -> None:
        """
        Add 3D position to the detection result.

        Args:
            position_3d (Tuple[float, float, float]): 3D coordinates (X, Y, Z).

        """
        self.position_3d = position_3d
