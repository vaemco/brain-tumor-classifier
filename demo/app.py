"""
Brain Tumor Classifier - Educational Demo API
==============================================
Self-contained container with model, sample images, and explainability.
Designed for React frontend integration.

Run: python app.py
API: http://localhost:5000/api/...
"""

import io
import base64
import random
from pathlib import Path
from typing import Optional, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Optional: GradCAM
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    print("⚠️  pytorch-grad-cam not installed. GradCAM disabled.")

# ==============================================================================
# Configuration
# ==============================================================================
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model" / "brain_tumor_efficientnet_b0.pt"
SAMPLES_DIR = BASE_DIR / "samples"

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_INFO = {
    "glioma": {
        "name": "Glioma",
        "description": "A tumor that originates from glial cells in the brain or spine.",
        "severity": "high",
        "location": "Can occur anywhere in the brain or spinal cord",
        "characteristics": "Often irregular shape, infiltrative growth pattern"
    },
    "meningioma": {
        "name": "Meningioma",
        "description": "A tumor arising from the meninges, the membranes surrounding the brain.",
        "severity": "medium",
        "location": "Surface of brain, attached to dura mater",
        "characteristics": "Usually well-defined, round or oval shape"
    },
    "notumor": {
        "name": "No Tumor",
        "description": "Normal brain MRI scan with no visible tumor.",
        "severity": "none",
        "location": "N/A",
        "characteristics": "Normal brain tissue appearance"
    },
    "pituitary": {
        "name": "Pituitary Tumor",
        "description": "A tumor in the pituitary gland at the base of the brain.",
        "severity": "medium",
        "location": "Sella turcica (base of skull)",
        "characteristics": "Located centrally, often affects hormone production"
    }
}

IMG_SIZE = 224
DEVICE = torch.device("cpu")

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

# ==============================================================================
# Model Loading (Lazy - only loads when first needed)
# ==============================================================================
_model: Optional[nn.Module] = None
_gradcam = None


def get_model() -> nn.Module:
    """Lazy load model."""
    global _model
    if _model is None:
        print("Loading EfficientNet-B0...")
        _model = models.efficientnet_b0(weights=None)
        _model.classifier[1] = nn.Linear(_model.classifier[1].in_features, len(CLASS_NAMES))

        if MODEL_PATH.exists():
            state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
            _model.load_state_dict(state)
            print(f"✓ Loaded: {MODEL_PATH.name}")
        else:
            print(f"⚠️  Model not found: {MODEL_PATH}")

        _model.to(DEVICE)
        _model.eval()
    return _model


def get_gradcam():
    """Lazy load GradCAM."""
    global _gradcam
    if _gradcam is None and GRADCAM_AVAILABLE:
        model = get_model()
        _gradcam = GradCAM(model=model, target_layers=[model.features[-1]])
    return _gradcam


# ==============================================================================
# Preprocessing Pipeline (Educational)
# ==============================================================================
PREPROCESSING_STEPS = [
    {
        "id": "resize",
        "name": "Resize",
        "description": "Scale image to 256x256 pixels for uniform input size",
        "why": "Neural networks require fixed input dimensions. Larger images would use too much memory.",
        "params": {"size": [256, 256]}
    },
    {
        "id": "center_crop",
        "name": "Center Crop",
        "description": "Extract 224x224 center region",
        "why": "Focuses on the brain center where tumors typically appear. Anti-Clever-Hans technique to prevent learning edge artifacts.",
        "params": {"size": 224}
    },
    {
        "id": "grayscale",
        "name": "Grayscale Conversion",
        "description": "Convert to grayscale (3-channel for compatibility)",
        "why": "MRI scans are grayscale. Color would be noise. Prevents learning color artifacts.",
        "params": {"channels": 3}
    },
    {
        "id": "normalize",
        "name": "Normalize",
        "description": "Scale pixel values using ImageNet statistics",
        "why": "Normalization helps convergence. We use ImageNet stats because our model is pretrained on ImageNet.",
        "params": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    }
]

inference_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def preprocess_with_steps(image: Image.Image) -> tuple[dict, torch.Tensor]:
    """Preprocess with intermediate step images for education."""
    steps = {}
    img = image.convert("RGB")
    steps["original"] = image_to_base64(img)

    # Step 1: Resize
    img_resized = transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32))(img)
    steps["resize"] = image_to_base64(img_resized)

    # Step 2: Center Crop
    img_cropped = transforms.CenterCrop(IMG_SIZE)(img_resized)
    steps["center_crop"] = image_to_base64(img_cropped)

    # Step 3: Grayscale
    img_gray = transforms.Grayscale(num_output_channels=3)(img_cropped)
    steps["grayscale"] = image_to_base64(img_gray)

    # Step 4: Tensor + Normalize
    tensor = transforms.ToTensor()(img_gray)
    tensor_norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )(tensor)
    steps["normalize"] = tensor_to_base64(tensor_norm)

    return steps, tensor_norm.unsqueeze(0)


def image_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def tensor_to_base64(tensor: torch.Tensor) -> str:
    """Convert normalized tensor back to viewable base64 image."""
    arr = tensor.squeeze().permute(1, 2, 0).numpy()
    arr = (arr * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
    arr = arr.clip(0, 255).astype("uint8")
    img = Image.fromarray(arr)
    return image_to_base64(img)


# ==============================================================================
# API Routes
# ==============================================================================

@app.route("/")
def index():
    """API documentation."""
    return jsonify({
        "name": "Brain Tumor Classifier Demo API",
        "version": "1.0.0",
        "endpoints": {
            "GET /api/info": "Model and feature information",
            "GET /api/classes": "Tumor class details",
            "GET /api/preprocessing": "Preprocessing pipeline explanation",
            "GET /api/samples": "List available sample images",
            "GET /api/samples/<class>/<filename>": "Get a sample image",
            "GET /api/samples/random": "Get a random sample",
            "POST /api/predict": "Classify an uploaded image",
            "POST /api/predict/explain": "Classify with full explanation",
            "GET /health": "Health check"
        }
    })


@app.route("/api/info")
def model_info():
    """Model architecture and training information."""
    return jsonify({
        "model": {
            "name": "EfficientNet-B0",
            "parameters": "5.3M",
            "pretrained": "ImageNet",
            "finetuned": "Brain Tumor Dataset"
        },
        "training": {
            "epochs": 30,
            "optimizer": "AdamW",
            "techniques": [
                "Transfer Learning",
                "Anti-Clever-Hans Augmentation",
                "Label Smoothing (0.1)",
                "Gradient Clipping",
                "Early Stopping"
            ]
        },
        "anti_clever_hans": {
            "description": "Techniques to prevent model from learning artifacts instead of actual tumor features",
            "methods": [
                {"name": "Center-Biased Crop", "description": "Forces model to focus on brain center, not skull edges"},
                {"name": "Grayscale Conversion", "description": "Removes color artifacts from varied scanning equipment"},
                {"name": "Mixup Augmentation", "description": "Blends training images to prevent memorization"},
                {"name": "Random Erasing", "description": "Randomly masks image regions to prevent over-reliance"}
            ]
        },
        "input": {"size": f"{IMG_SIZE}x{IMG_SIZE}", "channels": 3, "format": "RGB (grayscale converted)"},
        "output": {"classes": CLASS_NAMES, "type": "softmax probabilities"},
        "features": {"gradcam": GRADCAM_AVAILABLE, "preprocessing_visualization": True, "confidence_calibration": True}
    })


@app.route("/api/classes")
def class_info():
    """Detailed information about each tumor class."""
    return jsonify({"classes": CLASS_INFO, "count": len(CLASS_NAMES)})


@app.route("/api/preprocessing")
def preprocessing_info():
    """Explain the preprocessing pipeline."""
    return jsonify({
        "description": "Images go through these steps before classification",
        "steps": PREPROCESSING_STEPS,
        "total_steps": len(PREPROCESSING_STEPS)
    })


@app.route("/api/samples")
def list_samples():
    """List available sample images by class."""
    samples = {}
    for cls in CLASS_NAMES:
        cls_dir = SAMPLES_DIR / cls
        if cls_dir.exists():
            files = [f.name for f in cls_dir.glob("*.jpg")][:5]
            samples[cls] = files
        else:
            samples[cls] = []
    return jsonify({"samples": samples, "base_url": "/api/samples"})


@app.route("/api/samples/<class_name>/<filename>")
def get_sample(class_name: str, filename: str):
    """Serve a sample image."""
    if class_name not in CLASS_NAMES:
        return jsonify({"error": "Invalid class"}), 400
    sample_dir = SAMPLES_DIR / class_name
    if not sample_dir.exists():
        return jsonify({"error": "No samples for this class"}), 404
    return send_from_directory(sample_dir, filename)


@app.route("/api/samples/random")
def random_sample():
    """Get a random sample image with its label."""
    available = []
    for cls in CLASS_NAMES:
        cls_dir = SAMPLES_DIR / cls
        if cls_dir.exists():
            for f in cls_dir.glob("*.jpg"):
                available.append((cls, f))
    if not available:
        return jsonify({"error": "No samples available"}), 404
    cls, filepath = random.choice(available)
    img = Image.open(filepath)
    return jsonify({
        "class": cls,
        "filename": filepath.name,
        "image": image_to_base64(img),
        "info": CLASS_INFO[cls]
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """Simple prediction endpoint."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        tensor = cast(torch.Tensor, inference_transform(img)).unsqueeze(0)
        model = get_model()
        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1)[0]
        pred_idx = int(probs.argmax().item())
        pred_class = CLASS_NAMES[pred_idx]
        return jsonify({
            "prediction": pred_class,
            "confidence": float(probs[pred_idx].item()),
            "probabilities": {name: float(probs[i].item()) for i, name in enumerate(CLASS_NAMES)},
            "class_info": CLASS_INFO[pred_class]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict/explain", methods=["POST"])
def predict_explain():
    """Full prediction with preprocessing steps, GradCAM, and explanation."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        preprocessing_images, tensor = preprocess_with_steps(img)

        model = get_model()
        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1)[0]

        pred_idx = int(probs.argmax().item())
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx].item())

        # Confidence calibration
        probs_list = probs.tolist()
        entropy = -sum(p * np.log(p + 1e-10) for p in probs_list)
        max_entropy = np.log(len(CLASS_NAMES))
        normalized_entropy = entropy / max_entropy

        # Confidence level
        if confidence > 0.9:
            conf_level, conf_expl = "high", "Model is very confident in this prediction."
        elif confidence > 0.7:
            conf_level, conf_expl = "medium", "Model has moderate confidence. Consider additional verification."
        else:
            conf_level, conf_expl = "low", "Model is uncertain. This case may need expert review."

        response = {
            "prediction": {
                "class": pred_class,
                "confidence": confidence,
                "confidence_level": conf_level,
                "confidence_explanation": conf_expl,
                "class_info": CLASS_INFO[pred_class]
            },
            "probabilities": {
                name: {"value": float(probs[i].item()), "rank": sorted(range(len(probs_list)), key=lambda x: probs_list[x], reverse=True).index(i) + 1}
                for i, name in enumerate(CLASS_NAMES)
            },
            "calibration": {
                "entropy": float(normalized_entropy),
                "interpretation": "Low entropy = confident, High entropy = uncertain",
                "recommendation": (
                    "Prediction appears reliable" if normalized_entropy < 0.3
                    else "Consider getting a second opinion" if normalized_entropy < 0.6
                    else "High uncertainty - expert review recommended"
                )
            },
            "preprocessing": {"steps": PREPROCESSING_STEPS, "images": preprocessing_images}
        }

        # GradCAM
        if GRADCAM_AVAILABLE:
            try:
                gradcam = get_gradcam()
                if gradcam is not None:
                    grayscale_cam = gradcam(input_tensor=tensor, targets=None)  # type: ignore[arg-type]
                    rgb_img = tensor.squeeze().permute(1, 2, 0).numpy()
                    rgb_img = rgb_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                    rgb_img = np.clip(rgb_img, 0, 1)
                    cam_image = show_cam_on_image(rgb_img, grayscale_cam[0], use_rgb=True)
                    cam_pil = Image.fromarray(cam_image)
                    response["gradcam"] = {
                        "image": image_to_base64(cam_pil),
                        "explanation": "GradCAM highlights regions the model focused on. Warm colors = high attention, cool colors = low attention.",
                        "interpretation": f"The model is looking at these regions to identify {CLASS_INFO[pred_class]['name']}."
                    }
            except Exception as e:
                response["gradcam"] = {"error": str(e)}
        else:
            response["gradcam"] = {"available": False}

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    """Health check."""
    return jsonify({
        "status": "ok",
        "model_loaded": _model is not None,
        "model_available": MODEL_PATH.exists(),
        "samples_available": SAMPLES_DIR.exists(),
        "gradcam_available": GRADCAM_AVAILABLE
    })


# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Brain Tumor Classifier - Educational Demo API")
    print("=" * 60)
    print(f"Model:   {MODEL_PATH}")
    print(f"Samples: {SAMPLES_DIR}")
    print(f"GradCAM: {'✓' if GRADCAM_AVAILABLE else '✗'}")
    print("=" * 60)
    print("\nEndpoints:")
    print("  GET  /api/info            Model info")
    print("  GET  /api/classes         Tumor classes")
    print("  GET  /api/preprocessing   Pipeline explanation")
    print("  GET  /api/samples         Sample images")
    print("  POST /api/predict         Quick classification")
    print("  POST /api/predict/explain Full explanation")
    print("=" * 60 + "\n")

    app.run(host="0.0.0.0", port=5000, debug=False)
