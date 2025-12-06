"""
Enhanced ResNet18 Model with Strong Regularization for Overfitting Prevention

This script provides the improved model architecture to combat overfitting.
Copy this code into your notebook's Model Setup cell to replace the existing one.

Key improvements:
- Stronger Dropout (0.4 and 0.5 instead of 0.3)
- Batch Normalization in classifier head
- Increased L2 weight decay (5e-2 for classifier)
- More responsive LR scheduler (patience=2, factor=0.5)
- Intermediate hidden layer (512 → 256 → num_classes)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# Model Setup with Enhanced Regularization
num_classes = len(class_names)  # Assumes class_names is defined
base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Enhanced classifier head with stronger regularization:
# - Multiple Dropout layers (0.4 and 0.5) for progressive regularization
# - Batch Normalization for stable training and regularization
# - Intermediate hidden layer for better feature learning
in_features = base.fc.in_features
base.fc = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(in_features, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(256, num_classes)
)

# Unfreeze all layers for better fine-tuning (Differential Learning Rate will control updates)
for p in base.parameters():
    p.requires_grad = True

model = base.to(device)  # Assumes device is defined

# Optimizer: AdamW with Differential Learning Rates and INCREASED weight decay
# Lower LR for backbone (feature extractor), Higher LR for classifier (head)
# Increased weight_decay from 1e-2 to 5e-2 for stronger L2 regularization
params = [
    {"params": [p for n, p in model.named_parameters() if "fc" not in n], "lr": 1e-5, "weight_decay": 1e-2},
    {"params": model.fc.parameters(), "lr": 1e-3, "weight_decay": 5e-2},
]
optimizer = optim.AdamW(params)

# Scheduler: More aggressive - reduce LR faster when validation loss plateaus
# Changed: factor 0.1 → 0.5 (less aggressive reduction), patience 3 → 2 (quicker response)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2
)

# Loss with Label Smoothing to prevent overfitting
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

print("Model ready for training on", device)
