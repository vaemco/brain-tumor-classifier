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
    "resnet18": Path("models/brain_tumor_resnet18_v2_trained.pt"),
    "efficientnet": Path("models/brain_tumor_efficientnet_b0_trained.pt"),
    "densenet": Path("models/brain_tumor_densenet121_trained.pt"),
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


# Load Models
def load_resnet18():
    """Load ResNet18 model"""
    print(f"Loading ResNet18 from {MODEL_PATHS['resnet18']}...", flush=True)
    num_classes = len(CLASSES)

    model = models.resnet18(weights=None)
    # Match the trained checkpoint architecture:
    # Dropout -> Linear(512, 256) -> BatchNorm1d -> ReLU -> Dropout -> Linear(256, 4)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.fc.in_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
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
    # Match the trained checkpoint architecture (Simple Head):
    # Dropout(0.5) -> Linear(1280, 4)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes)
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
    # Match the trained checkpoint architecture:
    # Dropout -> Linear(1024, 256) -> BatchNorm1d -> ReLU -> Dropout -> Linear(256, 4)
    # Note: Training used p=0.5 for dropout
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(256, num_classes),
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
    cam = GradCAM(model=model_for_cam, target_layers=[target_layer])

    # Generate CAM
    grayscale_cam = cam(input_tensor=image_tensor, targets=None)
    grayscale_cam = grayscale_cam[0, :]

    # Prepare RGB image
    rgb_img = np.array(image_pil.resize((224, 224))) / 255.0

    # Overlay CAM on image
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # Calculate precise bounding box from heatmap
    # Use HIGHER threshold to focus on peak activation (75% instead of 50%)
    threshold = grayscale_cam.max() * 0.75
    mask = grayscale_cam > threshold

    # Find bounding box coordinates
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if rows.any() and cols.any():
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        # Add minimal padding (5% instead of 10%)
        height, width = grayscale_cam.shape
        padding_y = int((y_max - y_min) * 0.05)
        padding_x = int((x_max - x_min) * 0.05)

        y_min = max(0, y_min - padding_y)
        y_max = min(height - 1, y_max + padding_y)
        x_min = max(0, x_min - padding_x)
        x_max = min(width - 1, x_max + padding_x)

        # Ensure minimum box size (at least 10% of image)
        min_size = int(height * 0.1)
        if (y_max - y_min) < min_size:
            center_y = (y_min + y_max) // 2
            y_min = max(0, center_y - min_size // 2)
            y_max = min(height - 1, center_y + min_size // 2)
        if (x_max - x_min) < min_size:
            center_x = (x_min + x_max) // 2
            x_min = max(0, center_x - min_size // 2)
            x_max = min(width - 1, center_x + min_size // 2)

        # Convert to percentage for frontend (relative to 224x224)
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


# Routes
@app.route("/")
def index():
    """Serve the main page (new v2 UI)"""
    return render_template("index-v2.html")


@app.route("/legacy")
def index_legacy():
    """Serve the legacy UI (old version)"""
    return render_template("index.html")



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

        # Make prediction
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = F.softmax(output, dim=1).cpu().numpy()[0]

        # Generate Grad-CAM with bounding box (explicitly pass the model)
        heatmap, bbox = generate_gradcam(image_tensor, image_pil, model_for_cam=model)
        heatmap_b64 = image_to_base64(heatmap)

        # Original image as base64
        original_resized = np.array(image_pil.resize((224, 224)))
        original_b64 = image_to_base64(original_resized)

        # Prepare results
        results = {
            "filename": image_path.name,
            "predictions": [
                {"class": CLASSES[i], "probability": float(probabilities[i])}
                for i in range(len(CLASSES))
            ],
            "top_prediction": {
                "class": CLASSES[probabilities.argmax()],
                "probability": float(probabilities.max()),
            },
            "gradcam": heatmap_b64,
            "original": original_b64,
            "bbox": bbox,  # Add bounding box data
            "model_version": MODEL_VERSION,
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
            with torch.no_grad():
                output = loaded_model(image_tensor)
                probs = F.softmax(output, dim=1).cpu().numpy()[0]

            top_idx = probs.argmax()
            model_predictions.append(
                {
                    "name": model_name.upper().replace("_", "-"),
                    "prediction": CLASSES[top_idx],
                    "confidence": float(probs[top_idx]),
                    "probabilities": probs,
                }
            )

        # Determine Consensus via Majority Voting
        votes = [pred["prediction"] for pred in model_predictions]
        winner = max(set(votes), key=votes.count)
        vote_count = votes.count(winner)

        # Calculate average confidence only from models that voted for the winner
        winner_confidences = [
            p["confidence"] for p in model_predictions if p["prediction"] == winner
        ]
        avg_confidence = np.mean(winner_confidences) if winner_confidences else 0.0

        # Determine consensus status
        if vote_count == len(models_dict):
            status = "High Consensus" if vote_count >= 3 else "Full Agreement"
        elif vote_count > len(models_dict) // 2:
            status = "Medium Consensus"
        else:
            status = "Low Consensus"

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
                "avg_confidence": float(avg_confidence),
            },
        }

        # Use the consensus winner as the top prediction
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


@app.route("/api/analyze-detailed", methods=["POST"])
def analyze_detailed():
    """
    Detailed analysis endpoint for the new UI.
    Returns predictions from all 3 models with layer progress simulation,
    averaged predictions, and GradCAM from the most confident model.
    """
    try:
        if not models_dict:
            return jsonify({"error": "No models loaded on the server"}), 500

        # Get uploaded file
        uploaded_file = request.files.get("file") or request.files.get("image")
        if uploaded_file is None or uploaded_file.filename == "":
            return jsonify({"error": "No file uploaded"}), 400

        # Read image bytes
        uploaded_file.seek(0)
        image_bytes = uploaded_file.read()

        # Preprocess image
        image_tensor, image_pil = preprocess_image(image_bytes)

        # Model colors for frontend
        model_colors = {
            "resnet18": "#3B82F6",      # Blue
            "efficientnet": "#EF4444",  # Red
            "densenet": "#10B981",      # Green
        }

        # Layer names for simulation (approximate layer structure)
        layer_names = {
            "resnet18": ["conv1", "layer1", "layer2", "layer3", "layer4", "avgpool", "fc"],
            "efficientnet": ["stem", "blocks1-2", "blocks3-4", "blocks5-6", "blocks7", "head", "fc"],
            "densenet": ["conv0", "denseblock1", "denseblock2", "denseblock3", "denseblock4", "fc"],
        }

        # Run predictions on all models
        model_results = {}
        all_probs = []

        for model_name, loaded_model in models_dict.items():
            with torch.no_grad():
                output = loaded_model(image_tensor)
                probs = F.softmax(output, dim=1).cpu().numpy()[0]

            top_idx = int(probs.argmax())
            confidence = float(probs[top_idx])

            # Simulate layer progress (confidence building up through layers)
            num_layers = len(layer_names.get(model_name, ["fc"]))
            layer_progress = []
            for i in range(num_layers):
                # Simulate confidence building up: starts low, ends at final confidence
                progress = (i + 1) / num_layers
                # Use exponential curve for more realistic "thinking" effect
                simulated_conf = confidence * (1 - np.exp(-3 * progress)) / (1 - np.exp(-3))
                layer_progress.append(round(float(simulated_conf), 3))

            model_results[model_name] = {
                "color": model_colors.get(model_name, "#888888"),
                "predictions": [
                    {"class": CLASSES[i], "probability": float(probs[i])}
                    for i in range(len(CLASSES))
                ],
                "layer_progress": layer_progress,
                "layer_names": layer_names.get(model_name, ["fc"]),
                "top_class": CLASSES[top_idx],
                "confidence": confidence,
            }

            all_probs.append(probs)

        # Calculate averaged predictions across all models
        avg_probs = np.mean(all_probs, axis=0)
        averaged_predictions = [
            {"class": CLASSES[i], "probability": float(avg_probs[i])}
            for i in range(len(CLASSES))
        ]

        # Determine final result (class with highest average probability)
        final_idx = int(avg_probs.argmax())
        final_result = {
            "class": CLASSES[final_idx],
            "confidence": float(avg_probs[final_idx]),
        }

        # Find the most confident model for GradCAM
        best_model_name = max(model_results, key=lambda k: model_results[k]["confidence"])
        best_model = models_dict[best_model_name]

        # Generate GradCAM with the best model
        heatmap, bbox = generate_gradcam(image_tensor, image_pil, model_for_cam=best_model)
        heatmap_b64 = image_to_base64(heatmap)

        # Original image as base64
        original_resized = np.array(image_pil.resize((224, 224)))
        original_b64 = image_to_base64(original_resized)

        # Build response
        response = {
            "original_b64": original_b64,
            "heatmap_b64": heatmap_b64,
            "bbox": bbox,
            "gradcam_model": best_model_name,
            "models": model_results,
            "averaged_predictions": averaged_predictions,
            "final_result": final_result,
            "filename": secure_filename(uploaded_file.filename),
            "model_version": MODEL_VERSION,
        }

        return jsonify(response)

    except Exception as e:
        print("Error in /api/analyze-detailed:", e, file=sys.stderr, flush=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/random-test-detailed", methods=["GET"])
def random_test_detailed():
    """Get random test image with detailed analysis (for new UI)"""
    try:
        import random

        # Collect all images from test directories
        test_images = []
        for d in TEST_DIRS:
            if d.exists():
                for ext in ["*.jpg", "*.JPG", "*.png", "*.PNG", "*.jpeg", "*.JPEG"]:
                    test_images.extend(list(d.rglob(ext)))

        if not test_images:
            return jsonify({"error": "No test images found"}), 404

        image_path = random.choice(test_images)

        # Read image
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        image_tensor, image_pil = preprocess_image(image_bytes)

        # Model colors
        model_colors = {
            "resnet18": "#3B82F6",
            "efficientnet": "#EF4444",
            "densenet": "#10B981",
        }

        layer_names = {
            "resnet18": ["conv1", "layer1", "layer2", "layer3", "layer4", "avgpool", "fc"],
            "efficientnet": ["stem", "blocks1-2", "blocks3-4", "blocks5-6", "blocks7", "head", "fc"],
            "densenet": ["conv0", "denseblock1", "denseblock2", "denseblock3", "denseblock4", "fc"],
        }

        model_results = {}
        all_probs = []

        for model_name, loaded_model in models_dict.items():
            with torch.no_grad():
                output = loaded_model(image_tensor)
                probs = F.softmax(output, dim=1).cpu().numpy()[0]

            top_idx = int(probs.argmax())
            confidence = float(probs[top_idx])

            num_layers = len(layer_names.get(model_name, ["fc"]))
            layer_progress = []
            for i in range(num_layers):
                progress = (i + 1) / num_layers
                simulated_conf = confidence * (1 - np.exp(-3 * progress)) / (1 - np.exp(-3))
                layer_progress.append(round(float(simulated_conf), 3))

            model_results[model_name] = {
                "color": model_colors.get(model_name, "#888888"),
                "predictions": [
                    {"class": CLASSES[i], "probability": float(probs[i])}
                    for i in range(len(CLASSES))
                ],
                "layer_progress": layer_progress,
                "layer_names": layer_names.get(model_name, ["fc"]),
                "top_class": CLASSES[top_idx],
                "confidence": confidence,
            }

            all_probs.append(probs)

        avg_probs = np.mean(all_probs, axis=0)
        averaged_predictions = [
            {"class": CLASSES[i], "probability": float(avg_probs[i])}
            for i in range(len(CLASSES))
        ]

        final_idx = int(avg_probs.argmax())
        final_result = {
            "class": CLASSES[final_idx],
            "confidence": float(avg_probs[final_idx]),
        }

        best_model_name = max(model_results, key=lambda k: model_results[k]["confidence"])
        best_model = models_dict[best_model_name]

        heatmap, bbox = generate_gradcam(image_tensor, image_pil, model_for_cam=best_model)
        heatmap_b64 = image_to_base64(heatmap)

        original_resized = np.array(image_pil.resize((224, 224)))
        original_b64 = image_to_base64(original_resized)

        response = {
            "original_b64": original_b64,
            "heatmap_b64": heatmap_b64,
            "bbox": bbox,
            "gradcam_model": best_model_name,
            "models": model_results,
            "averaged_predictions": averaged_predictions,
            "final_result": final_result,
            "filename": image_path.name,
            "model_version": MODEL_VERSION,
        }

        return jsonify(response)

    except Exception as e:
        print("Error in /api/random-test-detailed:", e, file=sys.stderr, flush=True)
        return jsonify({"error": str(e)}), 500


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
