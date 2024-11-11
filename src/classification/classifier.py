import logging

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class ClassificationResult:
    def __init__(self, class_id: int, confidence: float):
        self.class_id = class_id
        self.confidence = confidence

class Classifier:
    def __init__(self, config):
        self.model_path = config['model_path']
        self.input_size = config['input_size']
        self.logger = logging.getLogger('autonomous_perception.classification')
        self.model = self._load_model()
        self.transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def _load_model(self):
        self.logger.info("Loading classification model")
        model = torch.load(self.model_path, map_location=torch.device('cpu'))
        model.eval()
        self.logger.info("Classification model loaded successfully")
        return model

    def classify(self, tracked_objects: np.ndarray, images: tuple[np.ndarray, np.ndarray]) -> list[ClassificationResult]:
        classifications = []
        img_left, _ = images
        for obj in tracked_objects:
            bbox = obj[:4].astype(int)
            roi = self._get_roi(img_left, bbox)
            if roi is None:
                continue
            input_tensor = self.transform(roi).unsqueeze(0)
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, preds = torch.max(probabilities, 1)
                classifications.append(ClassificationResult(class_id=preds.item(), confidence=confidence.item()))
        self.logger.debug(f"Classification results: {len(classifications)}")
        return classifications

    def _get_roi(self, image: np.ndarray, bbox: list[int]) -> Image.Image | None:
        x1, y1, x2, y2 = bbox
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, image.shape[1] - 1), min(y2, image.shape[0] - 1)
        if x2 <= x1 or y2 <= y1:
            self.logger.warning(f"Invalid bounding box coordinates: {bbox}")
            return None
        roi = image[y1:y2, x1:x2]
        roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        return roi_pil
