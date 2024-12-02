import numpy as np
import torch


def crop_bbox(image, bbox, padding=0.2):
    """
    Crops a bounding box from the image with optional padding.

    Args:
        image (np.ndarray): The original image in BGR format.
        bbox (np.ndarray): Bounding box in [x1, y1, z, w, h] or [x1, y1, z, w, h, ...] format.
        padding (float): Padding ratio.

    Returns:
        np.ndarray: Cropped image.

    """
    x1, y1, z, w, h = bbox[:5]
    x_center = x1 + w / 2
    y_center = y1 + h / 2

    # Calculate padding
    pad_w = w * padding
    pad_h = h * padding

    x1_padded = max(int(x_center - (w / 2 + pad_w)), 0)
    y1_padded = max(int(y_center - (h / 2 + pad_h)), 0)
    x2_padded = min(int(x_center + (w / 2 + pad_w)), image.shape[1] - 1)
    y2_padded = min(int(y_center + (h / 2 + pad_h)), image.shape[0] - 1)

    cropped = image[y1_padded:y2_padded, x1_padded:x2_padded]
    return cropped


def bbox_ioa(box1, box2, iou=False, eps=1e-7):
    """
    Calculate the intersection over box2 area given box1 and box2. Boxes are in [x1, y1, z, x2, y2] format.

    Args:
        box1 (np.ndarray): A numpy array of shape (n, 5) representing n bounding boxes.
        box2 (np.ndarray): A numpy array of shape (m, 5) representing m bounding boxes.
        iou (bool): Calculate the standard IoU if True else return inter_area/box2_area.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (np.ndarray): A numpy array of shape (n, m) representing the intersection over box2 area.

    """
    if box1.shape[0] == 0 or box2.shape[0] == 0:
        return np.zeros((box1.shape[0], box2.shape[0]), dtype=np.float32)

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, _, b1_x2, b1_y2 = box1.T
    b2_x1, b2_y1, _, b2_x2, b2_y2 = box2.T

    # Intersection area
    inter_area = (np.minimum(b1_x2[:, None], b2_x2) - np.maximum(b1_x1[:, None], b2_x1)).clip(0) * (
        np.minimum(b1_y2[:, None], b2_y2) - np.maximum(b1_y1[:, None], b2_y1)
    ).clip(0)

    # Box2 area
    area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    if iou:
        box1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area = area + box1_area[:, None] - inter_area

    # Intersection over box2 area
    return inter_area / (area + eps)


def xyzwh2tlzwh(x):
    """
    Convert the bounding box format from [x, y, z, w, h]  to [x1, y1, z, w, h], where x1, y1 are the top-left coordinates, and z is the center depth.

    Args:
        x (np.ndarray | torch.Tensor):
            - Bounding boxes in [x, y, z, w, h] format.

    Returns:
        y (np.ndarray | torch.Tensor):
            - Bounding boxes in [x1, y1, z, w, h] format.

    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 3] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 4] / 2  # top left y
    return y
