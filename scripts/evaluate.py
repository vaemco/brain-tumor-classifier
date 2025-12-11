"""
Evaluation Script
=================

This script evaluates the trained model on an external test dataset.
It calculates accuracy, generates a confusion matrix, and saves misclassified images for analysis.

Key Features:
- Accuracy & Classification Report: Precision, Recall, F1-Score per class.
- Confusion Matrix: Visualizes where the model makes mistakes (e.g., confusing Glioma with Meningioma).
- Misclassified Analysis: Saves images that were predicted incorrectly into a `misclassified/` folder,
  organized by `TrueLabel_as_PredictedLabel`. This is crucial for debugging model errors.

How to Modify:
- Dataset: Change `DATA_DIR` to point to a different test set.
- Model: Change `MODEL_PATH` to evaluate a different checkpoint.
"""

import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, models

# Allow importing shared utilities from src/
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from brain_tumor.paths import get_data_dirs, get_models_dir  # noqa: E402
from brain_tumor.transforms import build_val_transforms  # noqa: E402

# Configuration
_, DATA_DIR = get_data_dirs(BASE_DIR)
MODEL_DIR = get_models_dir(BASE_DIR)
MODEL_PATH = MODEL_DIR / "brain_tumor_resnet18_v2_trained.pt"
if not MODEL_PATH.exists():
    print(f"Model {MODEL_PATH} not found, falling back to old naming")
    MODEL_PATH = MODEL_DIR / "brain_tumor_resnet18_v2.pt"

MISCLASSIFIED_DIR = BASE_DIR / "misclassified"
BATCH_SIZE = 32

# Device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# Transforms (Validation only)
val_tf = build_val_transforms()

# Load Data
print("Loading external dataset...")
try:
    dataset = datasets.ImageFolder(root=DATA_DIR, transform=val_tf)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    class_names = dataset.classes
    print(f"Classes: {class_names}")
    print(f"Samples: {len(dataset)}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Load Model
print(f"Loading model from {MODEL_PATH}...")
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
except:
    # Try loading with weights_only=True or False depending on pytorch version/warning
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

model = model.to(device)
model.eval()

# Evaluation
all_preds = []
all_labels = []
misclassified = []

print("Running evaluation...")
with torch.no_grad():
    for i, (x, y) in enumerate(loader):
        x = x.to(device)
        outputs = model(x)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.numpy())

        # Track misclassifications
        # We need original indices to get filenames.
        # ImageFolder preserves order if shuffle=False.
        start_idx = i * 32
        for j in range(len(preds)):
            if preds[j] != y[j]:
                idx = start_idx + j
                path, _ = dataset.samples[idx]
                misclassified.append(
                    {
                        "path": path,
                        "true_label": class_names[y[j]],
                        "pred_label": class_names[preds[j]],
                    }
                )

# Metrics
acc = accuracy_score(all_labels, all_preds)
print(f"\nAccuracy: {acc:.4f}")

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix_external.png")
print("Confusion matrix saved to confusion_matrix_external.png")

# Save Misclassified
print(f"\nSaving {len(misclassified)} misclassified images...")
if MISCLASSIFIED_DIR.exists():
    shutil.rmtree(MISCLASSIFIED_DIR)
MISCLASSIFIED_DIR.mkdir()

for item in misclassified:
    # Create subfolder: true_as_pred
    folder_name = f"{item['true_label']}_as_{item['pred_label']}"
    target_dir = MISCLASSIFIED_DIR / folder_name
    target_dir.mkdir(parents=True, exist_ok=True)

    # Copy file
    src = Path(item["path"])
    dst = target_dir / src.name
    shutil.copy2(src, dst)

print(f"Misclassified images saved to {MISCLASSIFIED_DIR}")
