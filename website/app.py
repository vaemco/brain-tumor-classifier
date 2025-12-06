"""
Flask Backend for Brain Tumor Classifier
========================================

This file serves as the main backend entry point. It handles:
1.  Loading the PyTorch model (ResNet18).
2.  Processing uploaded images (preprocessing, normalization).
3.  Generating predictions and Grad-CAM heatmaps.
4.  Simulating advanced features (Multi-Model Consensus, Similar Cases, Active Learning).

How to Modify:
--------------
- Change Model: Update `MODEL_PATH` and the `load_model` function if you switch architectures (e.g., to EfficientNet).
- Add Classes: Update the `CLASSES` list if your model predicts different tumor types.
- API Endpoints: Add new `@app.route` functions to create new API capabilities.
- Simulations: The `consensus` and `similar_cases` logic in `/api/predict` is currently simulated.
  Replace these sections with real model inference or database lookups for production use.
"""

import base64
import io
import sys
import time
from pathlib import Path

import flask
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, jsonify, render_template, request
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import models
from werkzeug.utils import secure_filename

from website.dataset import val_tf

print("Flask app starting...", flush=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = Path("website/static/uploads")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max

# Ensure upload folder exists
app.config["UPLOAD_FOLDER"].mkdir(parents=True, exist_ok=True)

CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# Model Paths
MODEL_PATHS = {
    "resnet18": Path("models/brain_tumor_resnet18_v2.pt"),
    "efficientnet": Path("models/brain_tumor_efficientnet_b0.pt"),
    "densenet": Path("models/brain_tumor_densenet121.pt"),
}
MODEL_VERSION = "v3-multi-model"

# Ensure feedback directories exist
FEEDBACK_DIR = Path("data/feedback")
FEEDBACK_IMAGES_DIR = FEEDBACK_DIR / "images"
FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
FEEDBACK_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
FEEDBACK_FILE = FEEDBACK_DIR / "feedback_labels.csv"

# Ensure feedback CSV exists with header
if not FEEDBACK_FILE.exists():
    with open(FEEDBACK_FILE, "w") as f:
        f.write(
            "filename,predicted_label,true_label,confidence,timestamp,model_version\n"
        )

# Data directories for "Similar Cases" or testing
# (Adjust these paths to match your local structure)
BASE_DIR = Path(__file__).resolve().parent.parent
TEST_DIRS = [
    BASE_DIR / "data" / "Brain_Tumor_Dataset" / "external_dataset" / "testing",
    BASE_DIR / "data" / "Brain_Tumor_Dataset" / "Testing",
]

# Device setup - M2 optimized
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✓ Using MPS (Apple Silicon GPU)", flush=True)
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("✓ Using CUDA GPU", flush=True)
else:
    device = torch.device("cpu")
    print("⚠ Using CPU", flush=True)


# Global metrics tracking for monitoring dashboard
metrics = {
    "predictions": {
        "total": 0,
        "resnet18": 0,
        "efficientnet": 0,
        "densenet": 0,
    },
    "inference_times": {
        "resnet18": [],
        "efficientnet": [],
        "densenet": [],
    },
    "confidence_scores": [],
    "confidence_history": {
        "resnet18": [],
        "efficientnet": [],
        "densenet": [],
    },
    "consensus": {
        "full_agreement": 0,
        "majority": 0,
        "low_consensus": 0,
    },
    "class_distribution": {
        "glioma": 0,
        "meningioma": 0,
        "pituitary": 0,
        "no_tumor": 0,
    },
    "recent_predictions": [],  # Last 20 predictions
    "latest_images": {
        "original": None,
        "gradcam": None,
        "timestamp": None,
    },
    "start_time": time.time(),
}



# Load Models
def load_resnet18():
    """Load ResNet18 model"""
    print(f"Loading ResNet18 from {MODEL_PATHS['resnet18']}...", flush=True)
    num_classes = len(CLASSES)

    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3), nn.Linear(model.fc.in_features, num_classes)
    )

    state_dict = torch.load(MODEL_PATHS["resnet18"], map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_efficientnet():
    """Load EfficientNet-B0 model"""
    print(f"Loading EfficientNet-B0 from {MODEL_PATHS['efficientnet']}...", flush=True)
    num_classes = len(CLASSES)

    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3), nn.Linear(in_features, num_classes)
    )

    state_dict = torch.load(MODEL_PATHS["efficientnet"], map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_densenet():
    """Load DenseNet121 model"""
    print(f"Loading DenseNet121 from {MODEL_PATHS['densenet']}...", flush=True)
    num_classes = len(CLASSES)

    model = models.densenet121(weights=None)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3), nn.Linear(in_features, num_classes)
    )

    state_dict = torch.load(MODEL_PATHS["densenet"], map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# Initialize models
models_dict = {}
try:
    models_dict["resnet18"] = load_resnet18()
    print("✓ ResNet18 loaded successfully", flush=True)
except FileNotFoundError as e:
    print(f"⚠ Warning - ResNet18: {e}", flush=True)

try:
    models_dict["efficientnet"] = load_efficientnet()
    print("✓ EfficientNet-B0 loaded successfully", flush=True)
except FileNotFoundError as e:
    print(f"⚠ Warning - EfficientNet: {e}", flush=True)

try:
    models_dict["densenet"] = load_densenet()
    print("✓ DenseNet121 loaded successfully", flush=True)
except FileNotFoundError as e:
    print(f"⚠ Warning - DenseNet: {e}", flush=True)

if not models_dict:
    print("⚠ ERROR: No models could be loaded!", flush=True)
    model = None  # Fallback for compatibility
else:
    print(f"✓ {len(models_dict)} model(s) ready for inference", flush=True)
    # For backward compatibility with gradcam
    model = models_dict.get("resnet18") or list(models_dict.values())[0]


# Helper functions
def preprocess_image(image_bytes):
    """Preprocess image for model input"""
    image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_tensor = val_tf(image_pil).unsqueeze(0).to(device)
    return image_tensor, image_pil


def generate_gradcam(image_tensor, image_pil, model_for_cam=None):
    """Generate Grad-CAM heatmap and find region of interest"""
    if model_for_cam is None:
        model_for_cam = model

    # Determine target layer based on model type
    if hasattr(model_for_cam, "layer4"):  # ResNet
        target_layer = model_for_cam.layer4[-1]
    elif hasattr(model_for_cam, "features"):  # EfficientNet, DenseNet
        target_layer = model_for_cam.features[-1]
    else:
        raise ValueError("Unknown model architecture for GradCAM")
    cam = GradCAM(model=model, target_layers=[target_layer])

    # Generate CAM
    grayscale_cam = cam(input_tensor=image_tensor, targets=None)
    grayscale_cam = grayscale_cam[0, :]

    # Prepare RGB image
    rgb_img = np.array(image_pil.resize((224, 224))) / 255.0

    # Overlay CAM on image
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # Calculate precise bounding box from heatmap
    # Use HIGHER threshold to focus on peak activation (85% instead of 75%)
    threshold = grayscale_cam.max() * 0.85
    mask = grayscale_cam > threshold

    # Find bounding box coordinates
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if rows.any() and cols.any():
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        # Minimal padding (2%)
        height, width = grayscale_cam.shape
        padding_y = int((y_max - y_min) * 0.02)
        padding_x = int((x_max - x_min) * 0.02)

        y_min = max(0, y_min - padding_y)
        y_max = min(height - 1, y_max + padding_y)
        x_min = max(0, x_min - padding_x)
        x_max = min(width - 1, x_max + padding_x)

        # Calculate Box Area relative to image
        box_area = (x_max - x_min) * (y_max - y_min)
        img_area = width * height

        # Filter out if box is suspiciously large (>60%)
        if box_area > 0.6 * img_area:
             bbox = None
        else:
             bbox = {
                "x": float(x_min / width * 100),
                "y": float(y_min / height * 100),
                "width": float((x_max - x_min) / width * 100),
                "height": float((y_max - y_min) / height * 100),
                "confidence": float(grayscale_cam.max()),
            }
    else:
        # Fallback if no significant region found
        bbox = None

    return visualization, bbox


def image_to_base64(image_array):
    """Convert numpy array to base64 string"""
    image_pil = Image.fromarray(image_array.astype("uint8"))
    buffer = io.BytesIO()
    image_pil.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ... (Helper functions remain) ...

# Routes
@app.route("/")
def index():
    return render_template("monitor.html") # Redirect index to monitor as requested by user consolidation

@app.route("/monitor")
def monitor():
    return render_template("monitor.html")




@app.route("/api/random-test", methods=["GET"])
def random_test():
    """Get random test image and prediction"""
    try:
        # Collect all images from test directories
        test_images = []
        print(f"Searching for images in: {[str(d) for d in TEST_DIRS]}", flush=True)

        for d in TEST_DIRS:
            if d.exists():
                # Recursively find images (jpg, png, jpeg) - Case insensitive approach
                found_in_dir = []
                for ext in ["*.jpg", "*.JPG", "*.png", "*.PNG", "*.jpeg", "*.JPEG"]:
                    found_in_dir.extend(list(d.rglob(ext)))

                print(f"Found {len(found_in_dir)} images in {d}", flush=True)
                test_images.extend(found_in_dir)
            else:
                print(f"Directory not found: {d}", flush=True)

        if not test_images:
            print("ERROR: No test images found in any directory!", flush=True)
            return jsonify(
                {"error": "No test images found in configured directories"}
            ), 404

        import random

        image_path = random.choice(test_images)

        # Read and process image
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        image_tensor, image_pil = preprocess_image(image_bytes)

        # --- REAL Multi-Model Consensus ---
        # Run prediction with all available models
        model_predictions = []

        for model_name, loaded_model in models_dict.items():
            start_t = time.time()
            with torch.no_grad():
                output = loaded_model(image_tensor)
                probs = F.softmax(output, dim=1).cpu().numpy()[0]
            end_t = time.time()
            inf_time = (end_t - start_t) * 1000  # ms

            top_idx = probs.argmax()
            model_predictions.append(
                {
                    "name": model_name.upper().replace("_", "-"),
                    "prediction": CLASSES[top_idx],
                    "confidence": float(probs[top_idx]),
                    "probabilities": probs,
                    "inference_time": inf_time,
                }
            )

        # Soft Voting: Average the probabilities
        sum_probs = np.sum([p["probabilities"] for p in model_predictions], axis=0)
        avg_probs = sum_probs / len(models_dict)

        top_idx_ensemble = avg_probs.argmax()
        winner = CLASSES[top_idx_ensemble]
        avg_confidence = float(avg_probs[top_idx_ensemble])

        # Consensus Status
        votes = [pred["prediction"] for pred in model_predictions]
        vote_count = votes.count(winner)

        if vote_count == len(models_dict):
            status = "Unanimous"
        elif vote_count > len(models_dict) // 2:
            status = "Majority"
        else:
            status = "Ambiguous"

        consensus_data = {
            "models": [
                {
                    "name": p["name"],
                    "prediction": p["prediction"],
                    "confidence": p["confidence"],
                }
                for p in model_predictions
            ],
            "result": {
                "winner": winner,
                "score": f"{vote_count}/{len(models_dict)}",
                "status": status,
                "avg_confidence": avg_confidence,
            },
        }

        # Use the Ensemble result as the top prediction
        top_pred_class = winner
        top_pred_prob = avg_confidence

        # Get probabilities from ResNet18 (or first model) for backward compatibility
        probabilities = next(
            (p["probabilities"] for p in model_predictions if "RESNET18" in p["name"]),
            model_predictions[0]["probabilities"]
        )

        # Generate Grad-CAM with bounding box (explicitly pass the model)
        heatmap, bbox = generate_gradcam(image_tensor, image_pil, model_for_cam=model)
        heatmap_b64 = image_to_base64(heatmap)

        # Original image as base64
        original_resized = np.array(image_pil.resize((224, 224)))
        original_b64 = image_to_base64(original_resized)

        # --- Track Metrics for Monitoring Dashboard ---
        metrics["predictions"]["total"] += 1

        # Track per-model inference times and counts
        for pred in model_predictions:
            model_key = pred["name"].lower().replace("-", "")
            if model_key in metrics["predictions"]:
                metrics["predictions"][model_key] += 1

            # Track per-model confidence history
            if model_key in metrics["confidence_history"]:
                metrics["confidence_history"][model_key].append(float(pred["confidence"]))
                if len(metrics["confidence_history"][model_key]) > 100:
                    metrics["confidence_history"][model_key] = metrics["confidence_history"][model_key][-100:]

            # Track inference times
            if model_key in metrics["inference_times"]:
                metrics["inference_times"][model_key].append(pred["inference_time"])
                if len(metrics["inference_times"][model_key]) > 100:
                    metrics["inference_times"][model_key] = metrics["inference_times"][model_key][-100:]

        # Track class distribution
        class_key = top_pred_class.lower().replace(" ", "_")
        if class_key in metrics["class_distribution"]:
            metrics["class_distribution"][class_key] += 1

        # Track confidence
        metrics["confidence_scores"].append(float(avg_confidence))
        if len(metrics["confidence_scores"]) > 100:  # Keep last 100
            metrics["confidence_scores"] = metrics["confidence_scores"][-100:]

        # Track consensus
        if vote_count == len(models_dict):
            metrics["consensus"]["full_agreement"] += 1
        elif vote_count > len(models_dict) // 2:
            metrics["consensus"]["majority"] += 1
        else:
            metrics["consensus"]["low_consensus"] += 1

        # Track recent predictions
        metrics["recent_predictions"].append({
            "timestamp": time.time(),
            "prediction": top_pred_class,
            "confidence": float(avg_confidence),
            "consensus_status": status,
        })
        if len(metrics["recent_predictions"]) > 20:  # Keep last 20
            metrics["recent_predictions"] = metrics["recent_predictions"][-20:]

        # Store latest images for bidirectional sync
        metrics["latest_images"]["original"] = original_b64
        metrics["latest_images"]["gradcam"] = heatmap_b64
        metrics["latest_images"]["timestamp"] = time.time()

        # Prepare results
        results = {
            "filename": image_path.name,
            "predictions": [
                {"class": CLASSES[i], "probability": float(probabilities[i])}
                for i in range(len(CLASSES))
            ],
            "top_prediction": {
                "class": top_pred_class,
                "probability": top_pred_prob,
            },
            "gradcam": heatmap_b64,
            "original": original_b64,
            "bbox": bbox,  # Add bounding box data
            "model_version": MODEL_VERSION,
            "consensus": consensus_data,
        }

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
@app.route(
    "/api/upload", methods=["POST"]
)  # optional, falls dein Frontend /api/upload nutzt
def predict_upload():
    """Handle uploaded image, run model + Grad-CAM"""
    try:
        if model is None:
            return jsonify({"error": "Model is not loaded on the server"}), 500

        # Versuche sowohl 'file' als auch 'image' als Feldnamen
        uploaded_file = request.files.get("file") or request.files.get("image")
        if uploaded_file is None or uploaded_file.filename == "":
            return jsonify({"error": "No file uploaded"}), 400

        # Optional: Datei speichern (für Debug/Feedback)
        filename = secure_filename(uploaded_file.filename)
        save_path = app.config["UPLOAD_FOLDER"] / filename
        uploaded_file.seek(0)
        uploaded_file.save(save_path)

        # For the model we need the bytes
        uploaded_file.seek(0)
        image_bytes = uploaded_file.read()

        # Preprocessing
        image_tensor, image_pil = preprocess_image(image_bytes)

        # Prediction
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = F.softmax(output, dim=1).cpu().numpy()[0]

        # Grad-CAM erzeugen
        heatmap, bbox = generate_gradcam(image_tensor, image_pil)
        heatmap_b64 = image_to_base64(heatmap)

        # Originalbild (224x224) auch als Base64 zurückgeben
        original_resized = np.array(image_pil.resize((224, 224)))
        original_b64 = image_to_base64(original_resized)

        # --- REAL Multi-Model Consensus ---
        # Run prediction with all available models
        model_predictions = []

        for model_name, loaded_model in models_dict.items():
            start_t = time.time()
            with torch.no_grad():
                output = loaded_model(image_tensor)
                probs = F.softmax(output, dim=1).cpu().numpy()[0]
            end_t = time.time()
            inf_time = (end_t - start_t) * 1000  # ms

            top_idx = probs.argmax()
            model_predictions.append(
                {
                    "name": model_name.upper().replace("_", "-"),
                    "prediction": CLASSES[top_idx],
                    "confidence": float(probs[top_idx]),
                    "probabilities": probs,
                    "inference_time": inf_time,
                }
            )

        # Soft Voting: Average the probabilities
        sum_probs = np.sum([p["probabilities"] for p in model_predictions], axis=0)
        avg_probs = sum_probs / len(models_dict)

        top_idx_ensemble = avg_probs.argmax()
        winner = CLASSES[top_idx_ensemble]
        avg_confidence = float(avg_probs[top_idx_ensemble])

        # Consensus Status
        votes = [pred["prediction"] for pred in model_predictions]
        vote_count = votes.count(winner)

        if vote_count == len(models_dict):
            status = "Unanimous"
        elif vote_count > len(models_dict) // 2:
            status = "Majority"
        else:
            status = "Ambiguous"

        consensus_data = {
            "models": [
                {
                    "name": p["name"],
                    "prediction": p["prediction"],
                    "confidence": p["confidence"],
                }
                for p in model_predictions
            ],
            "result": {
                "winner": winner,
                "score": f"{vote_count}/{len(models_dict)}",
                "status": status,
                "avg_confidence": avg_confidence,
            },
        }

        # Use the Ensemble result as the top prediction
        top_pred_class = winner
        top_pred_prob = avg_confidence

        # --- Similar Cases Simulation ---
        # Simulate retrieving similar cases from database
        similar_cases = [
            {
                "id": "CASE-001",
                "label": top_pred_class,
                "similarity": 0.98,
                "image": "/static/img/placeholder_brain.png",
            },
            {
                "id": "CASE-042",
                "label": top_pred_class,
                "similarity": 0.95,
                "image": "/static/img/placeholder_brain.png",
            },
            {
                "id": "CASE-128",
                "label": top_pred_class,
                "similarity": 0.89,
                "image": "/static/img/placeholder_brain.png",
            },
        ]

        # --- Track Metrics for Monitoring Dashboard ---
        metrics["predictions"]["total"] += 1

        # Track per-model inference times and counts
        import time as time_module
        for pred in model_predictions:
            model_key = pred["name"].lower().replace("-", "")
            if model_key in metrics["predictions"]:
                metrics["predictions"][model_key] += 1

            # Track per-model confidence history
            if model_key in metrics["confidence_history"]:
                metrics["confidence_history"][model_key].append(float(pred["confidence"]))
                if len(metrics["confidence_history"][model_key]) > 100:
                    metrics["confidence_history"][model_key] = metrics["confidence_history"][model_key][-100:]

            # Track inference times
            if model_key in metrics["inference_times"]:
                metrics["inference_times"][model_key].append(pred["inference_time"])
                if len(metrics["inference_times"][model_key]) > 100:
                    metrics["inference_times"][model_key] = metrics["inference_times"][model_key][-100:]

        # Track class distribution
        class_key = top_pred_class.lower().replace(" ", "_")
        if class_key in metrics["class_distribution"]:
            metrics["class_distribution"][class_key] += 1

        # Track confidence
        metrics["confidence_scores"].append(float(avg_confidence))
        if len(metrics["confidence_scores"]) > 100:  # Keep last 100
            metrics["confidence_scores"] = metrics["confidence_scores"][-100:]

        # Track consensus
        if vote_count == len(models_dict):
            metrics["consensus"]["full_agreement"] += 1
        elif vote_count > len(models_dict) // 2:
            metrics["consensus"]["majority"] += 1
        else:
            metrics["consensus"]["low_consensus"] += 1

        # Track recent predictions
        metrics["recent_predictions"].append({
            "timestamp": time.time(),
            "prediction": top_pred_class,
            "confidence": float(avg_confidence),
            "consensus_status": status,
        })
        if len(metrics["recent_predictions"]) > 20:  # Keep last 20
            metrics["recent_predictions"] = metrics["recent_predictions"][-20:]

        # Store latest images for bidirectional sync
        metrics["latest_images"]["original"] = original_b64
        metrics["latest_images"]["gradcam"] = heatmap_b64
        metrics["latest_images"]["timestamp"] = time.time()

        results = {
            "filename": filename,
            "predictions": [
                {"class": CLASSES[i], "probability": float(probabilities[i])}
                for i in range(len(CLASSES))
            ],
            "top_prediction": {"class": top_pred_class, "probability": top_pred_prob},
            "gradcam": heatmap_b64,
            "original": original_b64,
            "bbox": bbox,
            "model_version": MODEL_VERSION,
            "consensus": consensus_data,
            "similar_cases": similar_cases,
        }

        return jsonify(results)

    except Exception as e:
        # Für Debug im Terminal:
        print("Error in /api/predict:", e, file=sys.stderr, flush=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/feedback", methods=["POST"])
def submit_feedback():
    """Save user feedback"""
    try:
        data = request.json

        filename = data.get("filename")
        predicted_label = data.get("predicted_label")
        true_label = data.get("true_label")
        confidence = data.get("confidence")
        timestamp = data.get("timestamp")
        model_version = data.get("model_version")

        # Save to CSV
        with open(FEEDBACK_FILE, "a") as f:
            f.write(
                f"{filename},{predicted_label},{true_label},{confidence},{timestamp},{model_version}\n"
            )

        # Try to find and copy the image from test directories
        for test_dir in TEST_DIRS:
            if test_dir.exists():
                for img_path in test_dir.rglob(filename):
                    import shutil

                    shutil.copy2(img_path, FEEDBACK_IMAGES_DIR / filename)
                    break

        return jsonify({"status": "success"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/train", methods=["GET"])
def train_stream():
    """Stream training progress (simulation)"""

    def generate():
        import json

        steps = [
            {"progress": 10, "message": "Initializing training..."},
            {"progress": 20, "message": "Loading feedback data..."},
            {"progress": 30, "message": "Epoch 1/5 - Loss: 0.45"},
            {"progress": 50, "message": "Epoch 2/5 - Loss: 0.32"},
            {"progress": 70, "message": "Epoch 3/5 - Loss: 0.21"},
            {"progress": 85, "message": "Epoch 4/5 - Loss: 0.15"},
            {"progress": 95, "message": "Epoch 5/5 - Loss: 0.11"},
            {"progress": 100, "message": "Training completed! Model saved."},
        ]

        for step in steps:
            time.sleep(0.8)  # Simulate work
            yield f"data: {json.dumps(step)}\n\n"

    return flask.Response(generate(), mimetype="text/event-stream")


@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    """Get current metrics for monitoring dashboard"""
    uptime = time.time() - metrics["start_time"]

    # Calculate average confidence
    avg_conf = (
        np.mean(metrics["confidence_scores"])
        if metrics["confidence_scores"]
        else 0.0
    )

    return jsonify(
        {
            "uptime_seconds": uptime,
            "predictions": metrics["predictions"],
            "confidence_history": metrics["confidence_history"],
            "inference_times": metrics["inference_times"],
            "class_distribution": metrics["class_distribution"],
            "average_confidence": float(avg_conf),
            "confidence_scores": metrics["confidence_scores"][-20:],  # Last 20
            "consensus": metrics["consensus"],
            "recent_predictions": metrics["recent_predictions"],
            "models_loaded": list(models_dict.keys()),
            "latest_images": metrics["latest_images"],
        }
    )


@app.route("/api/health", methods=["GET"])
def health():
    """Health check"""
    loaded_models = list(models_dict.keys())
    return jsonify(
        {
            "status": "ok",
            "model_version": MODEL_VERSION,
            "models_loaded": loaded_models,
            "model_count": len(loaded_models),
        }
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=3000)
