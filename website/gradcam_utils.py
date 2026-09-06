"""
Grad-CAM heatmap and visualization utilities for brain tumor analysis.
"""

import base64
import io
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch

from website.dataset import val_tf


def preprocess_image(
    image_bytes: bytes, device: torch.device
) -> tuple[torch.Tensor, Image.Image]:
    """
    Load raw image bytes, convert to RGB, and apply inference transformations.
    """
    image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_tensor = val_tf(image_pil).unsqueeze(0).to(device)
    return image_tensor, image_pil


def image_to_base64(image_array: np.ndarray) -> str:
    """
    Encode a NumPy RGB image array to a base64 PNG string.
    """
    image_pil = Image.fromarray(image_array.astype("uint8"))
    buffer = io.BytesIO()
    image_pil.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def generate_gradcam(
    cam: GradCAM,
    image_tensor: torch.Tensor,
    image_pil: Image.Image,
) -> tuple[np.ndarray, dict[str, float] | None]:
    """
    Generate Grad-CAM activation overlay and compute activation bounding box.
    """
    grayscale_cam = cam(input_tensor=image_tensor, targets=None)[0, :]
    rgb_img = np.array(image_pil.resize((224, 224))) / 255.0
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # Focus on peak activation region (75% of maximum activation)
    threshold = grayscale_cam.max() * 0.75
    mask = grayscale_cam > threshold

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if rows.any() and cols.any():
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        height, width = grayscale_cam.shape
        padding_y = int((y_max - y_min) * 0.05)
        padding_x = int((x_max - x_min) * 0.05)

        y_min = max(0, y_min - padding_y)
        y_max = min(height - 1, y_max + padding_y)
        x_min = max(0, x_min - padding_x)
        x_max = min(width - 1, x_max + padding_x)

        # Minimum bounding box size (at least 10% of image dimensions)
        min_size = int(height * 0.1)
        if (y_max - y_min) < min_size:
            center_y = (y_min + y_max) // 2
            y_min = max(0, center_y - min_size // 2)
            y_max = min(height - 1, center_y + min_size // 2)
        if (x_max - x_min) < min_size:
            center_x = (x_min + x_max) // 2
            x_min = max(0, center_x - min_size // 2)
            x_max = min(width - 1, center_x + min_size // 2)

        bbox: dict[str, float] | None = {
            "x": float(x_min / width * 100.0),
            "y": float(y_min / height * 100.0),
            "width": float((x_max - x_min) / width * 100.0),
            "height": float((y_max - y_min) / height * 100.0),
            "confidence": float(grayscale_cam.max()),
        }
    else:
        bbox = None

    return visualization, bbox


def calc_iou(box1: dict[str, float], box2: dict[str, float]) -> float:
    """
    Calculate intersection-over-union between two bounding boxes.
    """
    x1 = max(box1["x"], box2["x"])
    y1 = max(box1["y"], box2["y"])
    x2 = min(box1["x"] + box1["width"], box2["x"] + box2["width"])
    y2 = min(box1["y"] + box1["height"], box2["y"] + box2["height"])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = box1["width"] * box1["height"]
    area2 = box2["width"] * box2["height"]
    union = area1 + area2 - intersection

    return float(intersection / union) if union > 0 else 0.0


def compute_attention_consistency(
    bboxes: list[dict[str, float]],
) -> tuple[str, float]:
    """
    Evaluate attention agreement across multiple models to detect Clever-Hans shortcuts.
    """
    if len(bboxes) < 2:
        return "Unknown", 0.0

    ious = [
        calc_iou(bboxes[i], bboxes[j])
        for i in range(len(bboxes))
        for j in range(i + 1, len(bboxes))
    ]

    avg_iou = float(np.mean(ious)) if ious else 0.0

    if avg_iou > 0.6:
        consistency = "High (models focus on same region)"
    elif avg_iou > 0.3:
        consistency = "Medium (some overlap)"
    else:
        consistency = "Low (potential Clever-Hans!)"

    return consistency, avg_iou
