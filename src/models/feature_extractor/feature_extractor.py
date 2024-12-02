import cv2
import numpy as np
import torch
from torch import nn
from torchvision import models, transforms


class FeatureExtractor(nn.Module):
    def __init__(self, device="cuda"):
        """
        Initialize the FeatureExtractor with a pretrained ResNet50 model.

        Args:
            device (str): The device to run the model on ('cuda' or 'cpu').

        """
        super().__init__()
        # Load the pretrained ResNet50 model
        resnet = models.resnet50(pretrained=True)
        # Remove the final classification layer
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = resnet.fc.in_features  # Typically 2048 for ResNet50
        self.device = device
        self.to(self.device)
        self.eval()  # Set to evaluation mode

        # Define the transformation pipeline, convert np.ndarray to torch.Tensor
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),  # ResNet50 expects 224x224 images
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                    std=[0.229, 0.224, 0.225],
                ),
            ],
        )

    def forward(self, images):
        """
        Extract feature embeddings from a batch of images.

        Args:
            images (list of np.ndarray): List of cropped images (H, W, C) in BGR format.

        Returns:
            np.ndarray: Array of feature embeddings with shape (N, D).

        """
        if len(images) == 0:
            return np.array([])

        # Convert BGR to RGB and apply transformations
        processed_imgs = [self.transform(img) for img in images]
        if not processed_imgs:
            return np.array([])

        # Stack into a batch tensor
        batch_tensor = torch.stack(processed_imgs).to(self.device)

        with torch.no_grad():
            features = self.feature_extractor(batch_tensor)
            features = features.view(features.size(0), -1)  # Flatten
            features = features.cpu().numpy()

        # Normalize the features
        norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-10
        normalized_features = features / norms

        return normalized_features
