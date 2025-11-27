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
from flask import Flask, render_template, request, jsonify, send_from_directory
import flask
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from PIL import Image
import numpy as np
import io
import base64
import sys
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
from werkzeug.utils import secure_filename
from website.dataset import val_tf

print("Flask app starting...", flush=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = Path('website/static/uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Ensure upload folder exists
app.config['UPLOAD_FOLDER'].mkdir(parents=True, exist_ok=True)

CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
MODEL_PATH = Path('models/brain_tumor_resnet18_v2.pt')
MODEL_VERSION = "v2-feedback-trained"

# Ensure feedback directories exist
FEEDBACK_DIR = Path('data/feedback')
FEEDBACK_IMAGES_DIR = FEEDBACK_DIR / 'images'
FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
FEEDBACK_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
FEEDBACK_FILE = FEEDBACK_DIR / 'feedback_labels.csv'

# Ensure feedback CSV exists with header
if not FEEDBACK_FILE.exists():
    with open(FEEDBACK_FILE, 'w') as f:
        f.write("filename,predicted_label,true_label,confidence,timestamp,model_version\n")

# Data directories for "Similar Cases" or testing
# (Adjust these paths to match your local structure)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIRS = [
    BASE_DIR / 'data' / 'Brain_Tumor_Dataset' / 'external_dataset' / 'testing',
    BASE_DIR / 'data' / 'Brain_Tumor_Dataset' / 'Testing'
]

@app.route('/api/random-test', methods=['GET'])
def get_random_test_image():
    """Get a random image from the test set"""
    try:
        all_images = []
        for d in DATA_DIRS:
            if d.exists():
                all_images.extend(list(d.glob('**/*.jpg')))
                all_images.extend(list(d.glob('**/*.png')))
                all_images.extend(list(d.glob('**/*.jpeg')))
        
        if not all_images:
            return jsonify({'error': 'No test images found'}), 404
            
        selected = random.choice(all_images)
        
        # Copy to static/uploads to serve it
        dest_name = f"test_{selected.name}"
        dest_path = app.config['UPLOAD_FOLDER'] / dest_name
        import shutil
        shutil.copy(selected, dest_path)
        
        return jsonify({
            'url': f'/static/uploads/{dest_name}',
            'filename': dest_name,
            'true_label': selected.parent.name
        })
    except Exception as e:
        print(f"Error serving random test image: {e}")
        return jsonify({'error': str(e)}), 500

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

# Load model
def load_model():
    print(f"Loading model from {MODEL_PATH}...")
    num_classes = len(CLASSES)

    # ResNet18 anlegen – Pretrained ist egal, wird durch state_dict überschrieben
    model = models.resnet18(weights=None)

    # WICHTIG: gleicher Kopf wie im Notebook (Dropout + Linear)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.fc.in_features, num_classes)
    )

    # Gewichte laden
    state_dict = torch.load(MODEL_PATH, map_location=device)
    # falls deine torch-Version weights_only unterstützt und du es behalten willst:
    # state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

# Initialize model
try:
    model = load_model()
    print("✓ Model loaded successfully", flush=True)
except FileNotFoundError as e:
    print(f"⚠ Warning: {e}", flush=True)
    model = None

# Helper functions
def preprocess_image(image_bytes):
    """Preprocess image for model input"""
    image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_tensor = val_tf(image_pil).unsqueeze(0).to(device)
    return image_tensor, image_pil

def generate_gradcam(image_tensor, image_pil):
    """Generate Grad-CAM heatmap and find region of interest"""
    target_layer = model.layer4[-1]
    cam = GradCAM(model=model, target_layers=[target_layer])
    
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
            'x': float(x_min / width * 100),
            'y': float(y_min / height * 100),
            'width': float((x_max - x_min) / width * 100),
            'height': float((y_max - y_min) / height * 100),
            'confidence': float(grayscale_cam.max())
        }
    else:
        # Fallback if no significant region found
        bbox = None
    
    return visualization, bbox

def image_to_base64(image_array):
    """Convert numpy array to base64 string"""
    image_pil = Image.fromarray(image_array.astype('uint8'))
    buffer = io.BytesIO()
    image_pil.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

# Routes
@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/random-test', methods=['GET'])
def random_test():
    """Get random test image and prediction"""
    try:
        # Collect all images from test directories
        test_images = []
        for d in TEST_DIRS:
            if d.exists():
                # Recursively find images (jpg, png, jpeg)
                test_images.extend(list(d.rglob('*.jpg')))
                test_images.extend(list(d.rglob('*.png')))
                test_images.extend(list(d.rglob('*.jpeg')))
        
        if not test_images:
            return jsonify({'error': 'No test images found in configured directories'}), 404
        
        import random
        image_path = random.choice(test_images)
        
        # Read and process image
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
            
        image_tensor, image_pil = preprocess_image(image_bytes)
        
        # Make prediction
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = F.softmax(output, dim=1).cpu().numpy()[0]
        
        # Generate Grad-CAM with bounding box
        heatmap, bbox = generate_gradcam(image_tensor, image_pil)
        heatmap_b64 = image_to_base64(heatmap)
        
        # Original image as base64
        original_resized = np.array(image_pil.resize((224, 224)))
        original_b64 = image_to_base64(original_resized)
        
        # Prepare results
        results = {
            'filename': image_path.name,
            'predictions': [
                {'class': CLASSES[i], 'probability': float(probabilities[i])}
                for i in range(len(CLASSES))
            ],
            'top_prediction': {
                'class': CLASSES[probabilities.argmax()],
                'probability': float(probabilities.max())
            },
            'gradcam': heatmap_b64,
            'original': original_b64,
            'bbox': bbox,  # Add bounding box data
            'model_version': MODEL_VERSION
        }
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
@app.route('/api/upload', methods=['POST'])  # optional, falls dein Frontend /api/upload nutzt
def predict_upload():
    """Handle uploaded image, run model + Grad-CAM"""
    try:
        if model is None:
            return jsonify({'error': 'Model is not loaded on the server'}), 500

        # Versuche sowohl 'file' als auch 'image' als Feldnamen
        uploaded_file = request.files.get('file') or request.files.get('image')
        if uploaded_file is None or uploaded_file.filename == '':
            return jsonify({'error': 'No file uploaded'}), 400

        # Optional: Datei speichern (für Debug/Feedback)
        filename = secure_filename(uploaded_file.filename)
        save_path = app.config['UPLOAD_FOLDER'] / filename
        uploaded_file.seek(0)
        uploaded_file.save(save_path)

        # Für das Modell brauchen wir die Bytes
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

        results = {
            'filename': filename,
            'predictions': [
                {'class': CLASSES[i], 'probability': float(probabilities[i])}
                for i in range(len(CLASSES))
            ],
            'top_prediction': {
                'class': CLASSES[probabilities.argmax()],
                'probability': float(probabilities.max())
            },
            'gradcam': heatmap_b64,
            'original': original_b64,
            'bbox': bbox,
            'model_version': MODEL_VERSION
        }

        return jsonify(results)

    except Exception as e:
        # Für Debug im Terminal:
        print("Error in /api/predict:", e, file=sys.stderr, flush=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """Save user feedback"""
    try:
        data = request.json
        
        filename = data.get('filename')
        predicted_label = data.get('predicted_label')
        true_label = data.get('true_label')
        confidence = data.get('confidence')
        timestamp = data.get('timestamp')
        model_version = data.get('model_version')
        
        # Save to CSV
        with open(FEEDBACK_FILE, 'a') as f:
            f.write(f"{filename},{predicted_label},{true_label},{confidence},{timestamp},{model_version}\n")
            
        # Try to find and copy the image from test directories
        for test_dir in TEST_DIRS:
            if test_dir.exists():
                for img_path in test_dir.rglob(filename):
                    import shutil
                    shutil.copy2(img_path, FEEDBACK_IMAGES_DIR / filename)
                    break
            
        return jsonify({'status': 'success'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/train', methods=['GET'])
def train_stream():
    """Stream training progress (simulation)"""
    def generate():
        import time
        import json
        
        steps = [
            {"progress": 10, "message": "Initialisiere Training..."},
            {"progress": 20, "message": "Lade Feedback-Daten..."},
            {"progress": 30, "message": "Epoch 1/5 - Loss: 0.45"},
            {"progress": 50, "message": "Epoch 2/5 - Loss: 0.32"},
            {"progress": 70, "message": "Epoch 3/5 - Loss: 0.21"},
            {"progress": 85, "message": "Epoch 4/5 - Loss: 0.15"},
            {"progress": 95, "message": "Epoch 5/5 - Loss: 0.11"},
            {"progress": 100, "message": "Training abgeschlossen! Modell gespeichert."}
        ]
        
        for step in steps:
            time.sleep(0.8) # Simulate work
            yield f"data: {json.dumps(step)}\n\n"
            
    return flask.Response(generate(), mimetype='text/event-stream')


@app.route('/api/health', methods=['GET'])
def health():
    """Health check"""
    # --- Multi-Model Consensus Simulation ---
    # In a real scenario, we would load 3 different models.
    # Here we simulate it by perturbing the main model's probabilities.
    
    import random
    
    # Helper to simulate another model's prediction
    def simulate_model_prediction(base_probs, noise_level=0.1):
        # Add noise to probabilities
        new_probs = {}
        for cls, prob in base_probs.items():
            noise = random.uniform(-noise_level, noise_level)
            new_probs[cls] = max(0, min(1, prob + noise))
        
        # Normalize
        total = sum(new_probs.values())
        for cls in new_probs:
            new_probs[cls] /= total
            
        # Get top class
        top_class = max(new_probs, key=new_probs.get)
        return {"class": top_class, "probability": new_probs[top_class]}

    # Main Model (ResNet18)
    main_pred = top_prediction
    
    # Model 2 (EfficientNet-B0 Sim)
    model2_pred = simulate_model_prediction(probabilities, noise_level=0.15)
    
    # Model 3 (ViT-B/16 Sim)
    model3_pred = simulate_model_prediction(probabilities, noise_level=0.20)
    
    consensus_data = {
        "models": [
            {"name": "ResNet18", "prediction": main_pred['class'], "confidence": main_pred['probability']},
            {"name": "EfficientNet", "prediction": model2_pred['class'], "confidence": model2_pred['probability']},
            {"name": "VisionTransformer", "prediction": model3_pred['class'], "confidence": model3_pred['probability']}
        ]
    }
    
    # Determine Consensus
    votes = [m['prediction'] for m in consensus_data['models']]
    winner = max(set(votes), key=votes.count)
    vote_count = votes.count(winner)
    
    consensus_data['result'] = {
        "winner": winner,
        "score": f"{vote_count}/3",
        "status": "High Consensus" if vote_count == 3 else "Moderate Consensus"
    }

    # --- Similar Cases Simulation ---
    # In a real app, we would use embeddings (e.g. from the penultimate layer)
    # and find nearest neighbors in a vector database (FAISS/Chroma).
    # Here we select 3 random images from the same predicted class.
    
    similar_cases = []
    try:
        class_dir = Path(app.config['UPLOAD_FOLDER']).parent.parent / 'Training' / top_prediction['class']
        # If training dir doesn't exist (e.g. in this demo structure), mock it
        if not class_dir.exists():
             # Fallback: just return placeholders if no real data
             similar_cases = [
                 {"id": "CASE-001", "similarity": "98%", "label": top_prediction['class']},
                 {"id": "CASE-042", "similarity": "95%", "label": top_prediction['class']},
                 {"id": "CASE-113", "similarity": "89%", "label": top_prediction['class']}
             ]
        else:
            # Real logic if data exists
            all_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            if all_files:
                selected = random.sample(all_files, min(3, len(all_files)))
                # We would need to copy these to static to serve them
                # For this demo, we will use the fallback to avoid file copying complexity
                similar_cases = [
                     {"id": f"CASE-{random.randint(100,999)}", "similarity": f"{random.randint(85,99)}%", "label": top_prediction['class']} for _ in range(3)
                ]
    except Exception as e:
        print(f"Error finding similar cases: {e}")
        similar_cases = []

    return jsonify({
        'prediction': top_prediction,
        'probabilities': probabilities,
        'heatmap_url': f'/static/heatmaps/{heatmap_filename}',
        'bbox': bbox,
        'consensus': consensus_data,
        'similar_cases': similar_cases,
        'filename': file.filename
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)
