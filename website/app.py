"""
Brain Tumor Classification Web App
Flask Backend with Grad-CAM visualization
"""
from flask import Flask, render_template, request, jsonify, send_from_directory
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from PIL import Image
import numpy as np
import io
import base64
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2

from website.dataset import val_tf

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = Path('website/static/uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Ensure upload folder exists
app.config['UPLOAD_FOLDER'].mkdir(parents=True, exist_ok=True)

CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
MODEL_PATH = Path('models/brain_tumor_resnet18_final.pt')

# Device setup
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"✓ Using device: {device}")

# Load model
def load_model():
    """Load ResNet18 model"""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    return model

model = load_model()
print("✓ Model loaded")

# Grad-CAM setup
target_layer = model.layer4[-1]
cam = GradCAM(model=model, target_layers=[target_layer])


def preprocess_image(image_bytes):
    """Preprocess image for model"""
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    tensor = val_tf(image).unsqueeze(0).to(device)
    return tensor, image


def generate_gradcam(image_tensor, image_pil, target_class=None):
    """Generate Grad-CAM heatmap"""
    # Get prediction if no target specified
    if target_class is None:
        with torch.no_grad():
            output = model(image_tensor)
            target_class = output.argmax(dim=1).item()
    
    # Generate CAM
    grayscale_cam = cam(input_tensor=image_tensor, targets=None)[0, :]
    
    # Convert PIL image to numpy array
    img_np = np.array(image_pil.resize((224, 224))) / 255.0
    
    # Create heatmap
    heatmap = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    
    return heatmap


def image_to_base64(image_array):
    """Convert numpy array to base64 string"""
    img = Image.fromarray(image_array.astype('uint8'))
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        # Read image
        image_bytes = file.read()
        image_tensor, image_pil = preprocess_image(image_bytes)
        
        # Make prediction
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = F.softmax(output, dim=1).cpu().numpy()[0]
        
        # Generate Grad-CAM
        heatmap = generate_gradcam(image_tensor, image_pil)
        heatmap_b64 = image_to_base64(heatmap)
        
        # Original image as base64
        original_resized = np.array(image_pil.resize((224, 224)))
        original_b64 = image_to_base64(original_resized)
        
        # Prepare results
        results = {
            'predictions': [
                {'class': CLASSES[i], 'probability': float(probabilities[i])}
                for i in range(len(CLASSES))
            ],
            'top_prediction': {
                'class': CLASSES[probabilities.argmax()],
                'probability': float(probabilities.max())
            },
            'gradcam': heatmap_b64,
            'original': original_b64
        }
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'ok',
        'model': 'ResNet18',
        'device': str(device),
        'classes': CLASSES
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
