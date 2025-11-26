
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shutil
from pathlib import Path

# Configuration
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "Brain_Tumor_Dataset" / "external_dataset" / "training"
MODEL_PATH = BASE_DIR / "models" / "brain_tumor_resnet18_v2.pt"
if not MODEL_PATH.exists():
    print(f"Model {MODEL_PATH} not found, falling back to v1")
    MODEL_PATH = BASE_DIR / "models" / "brain_tumor_resnet18_final.pt"

MISCLASSIFIED_DIR = Path("misclassified")
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
val_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

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
                misclassified.append({
                    "path": path,
                    "true_label": class_names[y[j]],
                    "pred_label": class_names[preds[j]]
                })

# Metrics
acc = accuracy_score(all_labels, all_preds)
print(f"\nAccuracy: {acc:.4f}")

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
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
    src = Path(item['path'])
    dst = target_dir / src.name
    shutil.copy2(src, dst)

print(f"Misclassified images saved to {MISCLASSIFIED_DIR}")
