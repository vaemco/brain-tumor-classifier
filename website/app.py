"""
Flask Backend for Brain Tumor Classifier.

Provides REST and SSE endpoints for inference, multi-model consensus, Grad-CAM visualization,
and active learning feedback tracking.
"""

import json
import logging
from pathlib import Path
import random
import time
from flask import Flask, Response, jsonify, render_template, request, send_file
import numpy as np
from PIL import UnidentifiedImageError
import torch
import torch.nn.functional as F
from werkzeug.utils import secure_filename

from website.feedback import get_feedback_stats, init_feedback_file, save_feedback
from website.gradcam_utils import (
    compute_attention_consistency,
    generate_gradcam,
    image_to_base64,
    preprocess_image,
)
from website.models_manager import CLASSES, ModelsManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent

app.config["UPLOAD_FOLDER"] = APP_DIR / "static" / "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB limit
app.config["UPLOAD_FOLDER"].mkdir(parents=True, exist_ok=True)

MODEL_VERSION = "v3-multi-model"

def resolve_model_path(base_dir: Path, model_name: str) -> Path:
    """Find the best available model checkpoint file, preferring newly trained models."""
    candidates = [
        base_dir / "models" / f"brain_tumor_{model_name}_v2_trained.pt",
        base_dir / "models" / f"brain_tumor_{model_name}_v3_trained.pt",
        base_dir / "models" / f"brain_tumor_{model_name}_b0_v2_trained.pt",
        base_dir / "models" / f"brain_tumor_{model_name}_trained.pt",
        base_dir / "models" / "old_tests" / f"brain_tumor_{model_name}_v2_trained.pt",
        base_dir / "models" / "old_tests" / f"brain_tumor_{model_name}_trained.pt",
    ]
    if model_name == "densenet":
        candidates.extend([
            base_dir / "models" / "brain_tumor_densenet121_trained.pt",
            base_dir / "models" / "old_tests" / "brain_tumor_densenet121_trained.pt",
        ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


MODEL_PATHS = {
    "resnet18": resolve_model_path(BASE_DIR, "resnet18"),
    "efficientnet": resolve_model_path(BASE_DIR, "efficientnet"),
    "densenet": resolve_model_path(BASE_DIR, "densenet"),
}

# Feedback storage paths
FEEDBACK_DIR = BASE_DIR / "data" / "feedback"
FEEDBACK_IMAGES_DIR = FEEDBACK_DIR / "images"
FEEDBACK_FILE = FEEDBACK_DIR / "feedback_labels.csv"
init_feedback_file(FEEDBACK_FILE)

# Evaluation and test directories for random image sampling
TEST_DIRS = [
    BASE_DIR / "data" / "Brain_Tumor_Dataset" / "external_dataset" / "testing",
    BASE_DIR / "data" / "Brain_Tumor_Dataset" / "Testing",
]

# Initialize model manager
manager = ModelsManager(MODEL_PATHS)
manager.load_all_models()


def get_available_test_images() -> list[Path]:
    """Scan configured test directories for image files."""
    images: list[Path] = []
    for directory in TEST_DIRS:
        if directory.exists():
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
                images.extend(list(directory.rglob(ext)))
    return images


@app.route("/")
def index() -> str:
    """Serve the primary application interface."""
    return render_template("index-v2.html")


@app.route("/dashboard")
def dashboard() -> str:
    """Serve the active learning and feedback dashboard."""
    return render_template("dashboard.html")


@app.route("/api/health", methods=["GET"])
def health() -> Response:
    """Health check endpoint returning loaded model status."""
    loaded_models = list(manager.models.keys())
    return jsonify(
        {
            "status": "ok",
            "model_version": MODEL_VERSION,
            "models_loaded": loaded_models,
            "model_count": len(loaded_models),
        }
    )


@app.route("/api/random-test", methods=["GET"])
def random_test() -> tuple[Response, int] | Response:
    """Sample a random test image and generate predictions with Grad-CAM."""
    if not manager.models:
        return jsonify({"error": "No models are loaded on the server"}), 503

    images = get_available_test_images()
    if not images:
        return jsonify({"error": "No test images found in configured directories"}), 404

    image_path = random.choice(images)

    try:
        image_bytes = image_path.read_bytes()
        image_tensor, image_pil = preprocess_image(image_bytes, manager.device)
    except (UnidentifiedImageError, OSError) as exc:
        logger.error(f"Failed to load random image {image_path}: {exc}")
        return jsonify({"error": "Selected test image could not be decoded"}), 500

    primary_model = manager.primary_model
    primary_name = manager.primary_model_name
    if primary_model is None or primary_name not in manager.gradcams:
        return jsonify({"error": "Primary model is unavailable"}), 500

    with torch.no_grad():
        output = primary_model(image_tensor)
        probabilities = F.softmax(output, dim=1).cpu().numpy()[0]

    cam = manager.gradcams[primary_name]
    heatmap, bbox = generate_gradcam(cam, image_tensor, image_pil)
    heatmap_b64 = image_to_base64(heatmap)

    original_resized = np.array(image_pil.resize((224, 224)))
    original_b64 = image_to_base64(original_resized)

    return jsonify(
        {
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
            "bbox": bbox,
            "model_version": MODEL_VERSION,
        }
    )


@app.route("/api/predict", methods=["POST"])
@app.route("/api/upload", methods=["POST"])
def predict_upload() -> tuple[Response, int] | Response:
    """Handle uploaded images, generate predictions, consensus, and Grad-CAM."""
    if not manager.models:
        return jsonify({"error": "Model is not loaded on the server"}), 500

    uploaded_file = request.files.get("file") or request.files.get("image")
    if uploaded_file is None or not uploaded_file.filename:
        return jsonify({"error": "No image file provided"}), 400

    filename = secure_filename(uploaded_file.filename)
    image_bytes = uploaded_file.read()

    try:
        image_tensor, image_pil = preprocess_image(image_bytes, manager.device)
    except (UnidentifiedImageError, OSError):
        return jsonify({"error": "Uploaded file is not a valid image"}), 400

    primary_model = manager.primary_model
    primary_name = manager.primary_model_name
    if primary_model is None or primary_name not in manager.gradcams:
        return jsonify({"error": "Primary model is unavailable"}), 500

    with torch.no_grad():
        output = primary_model(image_tensor)
        probabilities = F.softmax(output, dim=1).cpu().numpy()[0]

    cam = manager.gradcams[primary_name]
    heatmap, bbox = generate_gradcam(cam, image_tensor, image_pil)
    heatmap_b64 = image_to_base64(heatmap)

    original_resized = np.array(image_pil.resize((224, 224)))
    original_b64 = image_to_base64(original_resized)

    consensus_data, winner, avg_confidence = manager.compute_consensus(image_tensor)

    # Reference similar cases demonstration
    similar_cases = [
        {
            "id": "CASE-001",
            "label": winner,
            "similarity": 0.98,
            "image": "/static/img/placeholder_brain.png",
        },
        {
            "id": "CASE-042",
            "label": winner,
            "similarity": 0.95,
            "image": "/static/img/placeholder_brain.png",
        },
        {
            "id": "CASE-128",
            "label": winner,
            "similarity": 0.89,
            "image": "/static/img/placeholder_brain.png",
        },
    ]

    return jsonify(
        {
            "filename": filename,
            "predictions": [
                {"class": CLASSES[i], "probability": float(probabilities[i])}
                for i in range(len(CLASSES))
            ],
            "top_prediction": {"class": winner, "probability": avg_confidence},
            "gradcam": heatmap_b64,
            "original": original_b64,
            "bbox": bbox,
            "model_version": MODEL_VERSION,
            "consensus": consensus_data,
            "similar_cases": similar_cases,
        }
    )


@app.route("/api/analyze-detailed", methods=["POST"])
def analyze_detailed() -> tuple[Response, int] | Response:
    """Run comprehensive multi-model inference and layer simulation on an uploaded file."""
    if not manager.models:
        return jsonify({"error": "No models loaded on the server"}), 500

    uploaded_file = request.files.get("file") or request.files.get("image")
    if uploaded_file is None or not uploaded_file.filename:
        return jsonify({"error": "No file uploaded"}), 400

    filename = secure_filename(uploaded_file.filename)
    image_bytes = uploaded_file.read()

    try:
        image_tensor, image_pil = preprocess_image(image_bytes, manager.device)
    except (UnidentifiedImageError, OSError):
        return jsonify({"error": "Invalid image file"}), 400

    detailed = manager.predict_detailed(image_tensor, image_pil)
    original_resized = np.array(image_pil.resize((224, 224)))

    return jsonify(
        {
            "original_b64": image_to_base64(original_resized),
            "heatmap_b64": image_to_base64(detailed["heatmap"]),
            "bbox": detailed["bbox"],
            "gradcam_model": detailed["best_model_name"],
            "models": detailed["model_results"],
            "averaged_predictions": detailed["averaged_predictions"],
            "final_result": detailed["final_result"],
            "filename": filename,
            "model_version": MODEL_VERSION,
        }
    )


@app.route("/api/random-test-detailed", methods=["GET"])
def random_test_detailed() -> tuple[Response, int] | Response:
    """Sample a random test image with detailed multi-model inference and ground truth evaluation."""
    if not manager.models:
        return jsonify({"error": "No models loaded on the server"}), 503

    images = get_available_test_images()
    if not images:
        return jsonify({"error": "No test images found"}), 404

    image_path = random.choice(images)

    try:
        image_bytes = image_path.read_bytes()
        image_tensor, image_pil = preprocess_image(image_bytes, manager.device)
    except (UnidentifiedImageError, OSError):
        return jsonify({"error": "Test image could not be read"}), 500

    detailed = manager.predict_detailed(image_tensor, image_pil)
    original_resized = np.array(image_pil.resize((224, 224)))

    parent_folder = image_path.parent.name.lower()
    label_mapping = {
        "glioma": "Glioma",
        "meningioma": "Meningioma",
        "notumor": "No Tumor",
        "no_tumor": "No Tumor",
        "pituitary": "Pituitary",
    }
    true_label = label_mapping.get(parent_folder)

    auto_eval = None
    if true_label:
        is_correct = detailed["final_result"]["class"] == true_label
        auto_eval = {
            "true_label": true_label,
            "predicted_label": detailed["final_result"]["class"],
            "is_correct": is_correct,
            "confidence": detailed["final_result"]["confidence"],
        }

    return jsonify(
        {
            "original_b64": image_to_base64(original_resized),
            "heatmap_b64": image_to_base64(detailed["heatmap"]),
            "bbox": detailed["bbox"],
            "gradcam_model": detailed["best_model_name"],
            "models": detailed["model_results"],
            "averaged_predictions": detailed["averaged_predictions"],
            "final_result": detailed["final_result"],
            "filename": image_path.name,
            "model_version": MODEL_VERSION,
            "auto_eval": auto_eval,
        }
    )


@app.route("/api/compare-gradcams", methods=["POST"])
def compare_gradcams() -> tuple[Response, int] | Response:
    """Generate Grad-CAM heatmaps from all models and evaluate attention consistency."""
    if not manager.models:
        return jsonify({"error": "No models loaded"}), 500

    uploaded_file = request.files.get("file") or request.files.get("image")
    if uploaded_file is None or not uploaded_file.filename:
        return jsonify({"error": "No file uploaded"}), 400

    filename = secure_filename(uploaded_file.filename)
    image_bytes = uploaded_file.read()

    try:
        image_tensor, image_pil = preprocess_image(image_bytes, manager.device)
    except (UnidentifiedImageError, OSError):
        return jsonify({"error": "Invalid image file"}), 400

    gradcams: dict = {}
    predictions: dict = {}
    bboxes: list[dict[str, float]] = []

    for model_name, model in manager.models.items():
        with torch.no_grad():
            output = model(image_tensor)
            probs = F.softmax(output, dim=1).cpu().numpy()[0]

        top_idx = int(probs.argmax())
        predictions[model_name] = {
            "class": CLASSES[top_idx],
            "confidence": float(probs[top_idx]),
        }

        cam = manager.gradcams[model_name]
        heatmap, bbox = generate_gradcam(cam, image_tensor, image_pil)
        gradcams[model_name] = {
            "heatmap_b64": image_to_base64(heatmap),
            "bbox": bbox,
        }
        if bbox is not None:
            bboxes.append(bbox)

    attention_consistency, _ = compute_attention_consistency(bboxes)
    original_resized = np.array(image_pil.resize((224, 224)))

    return jsonify(
        {
            "original_b64": image_to_base64(original_resized),
            "gradcams": gradcams,
            "predictions": predictions,
            "attention_consistency": attention_consistency,
            "filename": filename,
        }
    )


@app.route("/api/confidence-calibration", methods=["POST"])
def confidence_calibration() -> tuple[Response, int] | Response:
    """Assess model confidence calibration, entropy, and overconfidence risk."""
    if not manager.models:
        return jsonify({"error": "No models loaded"}), 500

    uploaded_file = request.files.get("file") or request.files.get("image")
    if uploaded_file is None or not uploaded_file.filename:
        return jsonify({"error": "No file uploaded"}), 400

    image_bytes = uploaded_file.read()
    try:
        image_tensor, _ = preprocess_image(image_bytes, manager.device)
    except (UnidentifiedImageError, OSError):
        return jsonify({"error": "Invalid image file"}), 400

    calibration_response = manager.compute_calibration(image_tensor)
    return jsonify(calibration_response)


@app.route("/api/model-info", methods=["GET"])
def model_info() -> Response:
    """Return model architectures, parameter statistics, and hardware info."""
    return jsonify(manager.get_info(MODEL_VERSION))


@app.route("/api/feedback", methods=["POST"])
def submit_feedback() -> tuple[Response, int] | Response:
    """Store clinician feedback with CSV injection protection."""
    data = request.json
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON payload"}), 400

    try:
        save_feedback(
            feedback_file=FEEDBACK_FILE,
            images_dir=FEEDBACK_IMAGES_DIR,
            test_dirs=TEST_DIRS,
            data=data,
        )
        return jsonify({"status": "success"})
    except Exception as exc:
        logger.error(f"Failed to save feedback: {exc}")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/feedback-stats", methods=["GET"])
def feedback_stats() -> Response:
    """Return aggregated feedback statistics for the dashboard."""
    stats = get_feedback_stats(FEEDBACK_FILE, CLASSES)
    return jsonify(stats)


@app.route("/api/feedback-export", methods=["GET"])
def feedback_export() -> tuple[Response, int] | Response:
    """Export the active learning feedback CSV file."""
    if not FEEDBACK_FILE.exists():
        return jsonify({"error": "No feedback data recorded"}), 404

    return send_file(
        FEEDBACK_FILE,
        mimetype="text/csv",
        as_attachment=True,
        download_name="feedback_export.csv",
    )


@app.route("/api/train", methods=["GET"])
def train_stream() -> Response:
    """Stream simulated active learning retraining progress via Server-Sent Events."""

    def generate():
        steps = [
            {"progress": 10, "message": "Initializing training environment..."},
            {"progress": 20, "message": "Loading verified clinician feedback..."},
            {"progress": 30, "message": "Epoch 1/5 - Loss: 0.45"},
            {"progress": 50, "message": "Epoch 2/5 - Loss: 0.32"},
            {"progress": 70, "message": "Epoch 3/5 - Loss: 0.21"},
            {"progress": 85, "message": "Epoch 4/5 - Loss: 0.15"},
            {"progress": 95, "message": "Epoch 5/5 - Loss: 0.11"},
            {"progress": 100, "message": "Fine-tuning complete. Updated checkpoint saved."},
        ]

        for step in steps:
            time.sleep(0.5)
            yield f"data: {json.dumps(step)}\n\n"

    return Response(generate(), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=3000)
