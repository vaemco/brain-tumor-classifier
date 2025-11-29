# Multi-Model Brain Tumor Classifier - Training Guide

## Overview

This project now includes three different deep learning models:

- **ResNet18** (already trained)
- **EfficientNet-B0** (new model)
- **DenseNet121** (new model)

The Flask app automatically loads all available models and performs a **real multi-model consensus analysis**.

---

## Training the New Models

### Prerequisites

Make sure your Conda environment is activated:

```bash
conda activate data_brain
```

### 1. Train EfficientNet-B0

```bash
cd /path/to/project/data_brain_tumor
jupyter notebook notebooks/train_efficientnet_b0.ipynb
```

**Important:** Adjust the data paths in the notebook (Cell 2):

- `data_dir`: Path to the main dataset
- `external_data_dir`: Path to the external dataset

**Expected training duration:** ~30-60 minutes on M2 MPS

The model will be automatically saved as:
`models/brain_tumor_efficientnet_b0_trained.pt`

---

### 2. Train DenseNet121

```bash
jupyter notebook notebooks/train_densenet121.ipynb
```

**Important:** Also adjust data paths here (Cell 2)

**Expected training duration:** ~30-60 minutes on M2 MPS

The model will be automatically saved as:
`models/brain_tumor_densenet121_trained.pt`

---

## Start Flask App

After at least one model has been trained:

```bash
cd /path/to/project/data_brain_tumor
python3 -m website.app
```

### Check Terminal Output

You should see the following:

```
Flask app starting...
✓ Using MPS (Apple Silicon GPU)
Loading ResNet18 from models/brain_tumor_resnet18_v2_trained.pt...
✓ ResNet18 loaded successfully
Loading EfficientNet-B0 from models/brain_tumor_efficientnet_b0_trained.pt...
✓ EfficientNet-B0 loaded successfully
Loading DenseNet121 from models/brain_tumor_densenet121_trained.pt...
✓ DenseNet121 loaded successfully
✓ 3 model(s) ready for inference
```

**If a model is missing:**

```
⚠ Warning - EfficientNet: [Errno 2] No such file or directory: 'models/brain_tumor_efficientnet_b0_trained.pt'
```

→ This is OK! The app works with fewer models as well.

---

## Test API Endpoints

### Health Check

```bash
curl http://localhost:3000/api/health
```

**Expected Response:**

```json
{
  "status": "ok",
  "model_version": "v3-multi-model",
  "models_loaded": ["resnet18", "efficientnet", "densenet"],
  "model_count": 3
}
```

---

## How Does Multi-Model Analysis Work?

1. **Inference**: Each loaded model makes a prediction
2. **Voting**: Majority vote determines the result
3. **Confidence**: Average of confidence from all models that voted for the winner
4. **Consensus Status**:
   - **High Consensus**: All models agree
   - **Medium Consensus**: Majority agrees
   - **Low Consensus**: No clear majority

---

## Recommended Workflow

**Option A: Sequential**

```bash
# 1. Train EfficientNet
jupyter notebook notebooks/train_efficientnet_b0.ipynb

# 2. Train DenseNet
jupyter notebook notebooks/train_densenet121.ipynb

# 3. Start app
python3 -m website.app
```

**Option B: Parallel (2 Terminals)**

Terminal 1:

```bash
jupyter notebook notebooks/train_efficientnet_b0.ipynb
```

Terminal 2:

```bash
jupyter notebook notebooks/train_densenet121.ipynb
```

Terminal 3 (after training):

```bash
python3 -m website.app
```

---

## Troubleshooting

**Problem**: `ModuleNotFoundError: No module named 'pytorch_grad_cam'`

```bash
conda activate data_brain
pip install grad-cam
```

**Problem**: Model doesn't load

- Check if `.pt` file exists: `ls -lh models/`
- Check terminal output when starting the app
- Make sure training completed successfully

**Problem**: Training too slow

- Use MPS (Apple Silicon): should be automatically detected in notebooks
- Reduce `batch_size` in notebooks (line with `DataLoader`)

---

## Model Comparison

| Model           | Parameters | Advantage          | Training Time (M2) |
| --------------- | ---------- | ------------------ | ------------------ |
| ResNet18        | ~11M       | Robust, proven     | ~45 min            |
| EfficientNet-B0 | ~5M        | Compact, efficient | ~35 min            |
| DenseNet121     | ~8M        | Dense Connections  | ~50 min            |

---

## Next Steps

After successful training:

1. ✅ Open the web app: http://localhost:3000
2. ✅ Upload a test image
3. ✅ Check the "Multi-Model Consensus" section
4. ✅ Compare predictions from individual models

For questions or problems, see `TRAINING_GUIDE.md` for architecture details.
