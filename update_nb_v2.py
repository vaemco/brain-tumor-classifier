import json
import os

nb_path = 'notebooks/03_train_efficientnet_b0.ipynb'

try:
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    updated = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if "base.classifier = nn.Sequential" in source and "nn.Dropout(p=0.5)" in source:
                print("Found classifier cell. Updating...")

                new_source = [
                    "# Match the deeper classifier architecture (same as ResNet18/DenseNet121)\n",
                    "in_features = base.classifier[1].in_features\n",
                    "base.classifier = nn.Sequential(\n",
                    "    nn.Dropout(p=0.3),\n",
                    "    nn.Linear(in_features, 256),\n",
                    "    nn.BatchNorm1d(256),\n",
                    "    nn.ReLU(),\n",
                    "    nn.Dropout(p=0.3),\n",
                    "    nn.Linear(256, num_classes),\n",
                    ")\n",
                    "\n",
                    "# Unfreeze all layers for better fine-tuning (Differential Learning Rate will control updates)\n",
                    "for p in base.parameters():\n",
                    "    p.requires_grad = True\n",
                    "\n",
                    "model = base.to(device)\n",
                    "\n",
                    "# Optimizer: AdamW with Differential Learning Rates\n",
                    "# Lower LR for backbone (feature extractor), Higher LR for classifier (head)\n",
                    "params = [\n",
                    "    {\n",
                    "        \"params\": [p for n, p in model.named_parameters() if \"classifier\" not in n],\n",
                    "        \"lr\": 1e-5,\n",
                    "    },\n",
                    "    {\"params\": model.classifier.parameters(), \"lr\": 1e-3},\n",
                    "]\n",
                    "optimizer = optim.AdamW(params, weight_decay=1e-2)\n",
                    "\n",
                    "# Scheduler: Reduce LR when validation loss plateaus\n",
                    "scheduler = optim.lr_scheduler.ReduceLROnPlateau(\n",
                    "    optimizer, mode=\"min\", factor=0.1, patience=3\n",
                    ")\n",
                    "\n",
                    "# Loss with Label Smoothing to prevent overfitting\n",
                    "criterion = nn.CrossEntropyLoss(label_smoothing=0.1)\n",
                    "\n",
                    "print(\"EfficientNet-B0 model ready for training on\", device)"
                ]
                cell['source'] = new_source
                updated = True
                break

    if updated:
        with open(nb_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print("Notebook updated successfully.")
    else:
        print("Could not find the target cell to update.")

except Exception as e:
    print(f"Error updating notebook: {e}")
