# Brain Tumor Classifier

A deep learning project and interactive web application for MRI brain tumor classification across four categories: Glioma, Meningioma, No Tumor, and Pituitary.

## Purpose

This project provides an end-to-end machine learning pipeline featuring:
- Reusable, anti-shortcut data augmentation transforms (Grayscale, CenterBiasedCrop, Gaussian noise, Mixup).
- Transfer learning with standardized architectures (ResNet18, EfficientNet-B0, DenseNet121).
- Stratified K-fold cross-validation with cosine annealing and early stopping.
- Explainable AI with Grad-CAM activation mapping and multi-model attention consistency checks.
- A modern web interface with ensemble voting, confidence calibration, and an active learning feedback dashboard.

## Project Structure

```
data_brain_tumor/
  configs/            - Training configuration YAML files
  data/               - Brain Tumor MRI dataset directory
  models/             - Trained PyTorch checkpoints (.pt)
  runs/               - Evaluation metrics and training history
  scripts/            - CLI scripts for preparation, training, and evaluation
    prepare_data.py   - Dataset train/test splitting
    train.py          - Single model fine-tuning with early stopping
    train_kfold.py    - Stratified K-fold cross-validation
    evaluate.py       - Test set evaluation and confusion matrix generation
  src/brain_tumor/    - Core Python package
    device.py         - Unified device detection (CUDA, MPS, CPU)
    model_factory.py  - Model architecture registry and Grad-CAM target layers
    paths.py          - Centralized path resolution
    transforms.py     - Augmentation and validation transforms
  website/            - Flask web application
    app.py            - Modular REST API and page routes
    dataset.py        - Inference transform definition
    feedback.py       - Clinician feedback storage and dashboard statistics
    gradcam_utils.py  - Grad-CAM heatmap generation and IoU calculation
    models_manager.py - Model loading, caching, and ensemble predictions
    static/           - CSS styles, JavaScript logic, and assets
    templates/        - HTML templates (index-v2.html, dashboard.html)
  tests/              - Unit tests
  pyproject.toml      - Package configuration
  environment.yml     - Conda environment specification
```

## Setup

### 1. Environment Setup

Using Conda:
```bash
conda env create -f environment.yml
conda activate data_brain
```

Install the package in editable mode:
```bash
pip install -e . --no-deps
```

### 2. Configuration

Set up environment variables or copy configuration files as needed:
```bash
# Example training config
copy configs\train.example.yaml configs\train.yaml
```

## Usage

### Training Models

Train ResNet18:
```bash
python scripts/train.py --model-name resnet18 --epochs 30 --patience 5
```

Train K-Fold Ensemble:
```bash
python scripts/train_kfold.py --model efficientnet --folds 5 --epochs 20
```

### Evaluation

Evaluate a trained checkpoint against test data:
```bash
python scripts/evaluate.py --model-name efficientnet --batch-size 32
```

Outputs will be saved in `runs/evaluation/` including classification reports and confusion matrices.

### Running the Web Application

Launch the Flask server:
```bash
python website/app.py
```

Then navigate to `http://localhost:3000` in your web browser. Access the dashboard at `http://localhost:3000/dashboard`.

## Model Architectures and Ensemble Cooperation

The project utilizes three distinct convolutional architectures, each selected for specific mathematical properties and diagnostic functions. Detailed technical documentation is available in [docs/models_and_ensemble.md](file:///c:/dev_project/04_ml_data/data_brain_tumor/docs/models_and_ensemble.md).

| Architecture | Parameters | Mathematical Mechanism | Diagnostic Role |
|---|---|---|---|
| **ResNet18** | ~11.2M | Residual identity shortcuts ($F(x) + x$) | **Structural Anchor**: Detects global brain geometry, hemisphere asymmetry, and large-scale midline shifts. |
| **EfficientNet-B0** | ~4.3M | Compound scaling with MBConv and Squeeze-and-Excitation | **Precision Classifier**: Emphasizes tumor core channels while suppressing skull background, providing boundary precision. |
| **DenseNet121** | ~7.2M | Dense connectivity ($[x_0, x_1, \dots, x_{l-1}]$) | **Texture Specialist**: Preserves low-level intensity gradients and micro-textures for subtle lesions (e.g. pituitary microadenomas). |

### The Cooperative Ensemble Pipeline

1. **Multi-Model Consensus**: All three models independently predict the class distribution. A majority voting mechanism determines the diagnostic winner, while softmax entropy measures uncertainty.
2. **Clever-Hans Shortcut Prevention**: Grad-CAM heatmaps are generated across all three models. The system computes the Intersection-over-Union (IoU) of peak activation bounding boxes. If all three models agree on the anatomical location (IoU > 0.6), the diagnosis is confirmed. If a model focuses on image borders or skull artifacts (IoU < 0.3), a warning is flagged.

## Testing

Run the test suite:
```bash
pytest tests/ -v
```
