# Training Guide - Brain Tumor MRI Classifier

This guide describes how to train the models for the Brain Tumor MRI Classifier and what to pay attention to during training.

## Overview

The project uses three different architectures:
- **ResNet18** - Balanced performance and speed
- **EfficientNet-B0** - Optimized for mobile/edge devices
- **DenseNet121** - Dense connections for feature reuse

Each model is trained separately and can then be used for multi-model consensus inference.

## Prerequisites

### 1. Data Preparation

Ensure your data structure is correct:

```
data/Brain_Tumor_Dataset/
  Training/
    glioma/
    meningioma/
    notumor/
    pituitary/
  Testing/
    glioma/
    meningioma/
    notumor/
    pituitary/
  external_dataset/
    training/
      glioma/
      meningioma/
      notumor/
      pituitary/
    testing/
      glioma/
      meningioma/
      notumor/
      pituitary/
```

**Important:**
- Classes must be named exactly: `glioma`, `meningioma`, `notumor`, `pituitary`
- Pay attention to case sensitivity
- At least 100 images per class recommended for stable training

### 2. Hardware Requirements

- **Recommended:** GPU (CUDA) or Apple Silicon (MPS) for accelerated training
- **Minimum:** CPU (significantly slower, but possible)
- **RAM:** At least 8GB, recommended 16GB+
- **Storage:** ~2-5GB per model

### 3. Environment

Activate the Conda environment:

```bash
conda activate data_brain
```

## Training with Jupyter Notebooks

### Step 1: Select Notebook

Choose the appropriate notebook for the model you want to train:

- **ResNet18:** `notebooks/train_resnet18.ipynb`
- **EfficientNet-B0:** `notebooks/train_efficientnet_b0.ipynb`
- **DenseNet121:** `notebooks/train_densenet121.ipynb`

### Step 2: Open Notebook

```bash
cd notebooks
jupyter notebook train_resnet18.ipynb  # or corresponding notebook
```

### Step 3: Run Training

1. **Execute cells sequentially** - The notebooks are structured to be executed cell by cell
2. **Define transforms** - Ensure data augmentation transforms are correctly defined
3. **Start training** - The training loop runs automatically with early stopping

## Important Parameters and Settings

### Hyperparameters

The notebooks use optimized hyperparameters:

- **Optimizer:** AdamW with differential learning rates
  - Backbone (Feature Extractor): `1e-5`
  - Classifier Head: `1e-3`
- **Scheduler:** ReduceLROnPlateau
  - Factor: `0.1`
  - Patience: `3` epochs
- **Loss:** CrossEntropyLoss with Label Smoothing (`0.1`)
- **Batch Size:** `32` (adjustable based on available memory)
- **Epochs:** `30` (with early stopping at patience `7`)

### What to Pay Attention To

#### 1. **Data Augmentation**
- **Important:** Augmentation should be medically sensible
- Rotations up to 15° are generally safe
- No overly aggressive transformations that distort anatomical structures
- Normalization with ImageNet statistics (for transfer learning)

#### 2. **Class Imbalance**
- Check the distribution of classes
- For severe imbalance: Consider weighted loss or oversampling
- Currently: Label smoothing helps against overfitting

#### 3. **Validation Split**
- Currently: 20% of training data for validation
- **Important:** Validation set should be representative
- Separate test set should be used for final evaluation

#### 4. **Early Stopping**
- Patience of 7 epochs prevents overfitting
- Best model is saved based on validation loss
- **Don't stop too early** - Give the scheduler time to work

#### 5. **Device Detection**
- Notebooks automatically detect MPS/CUDA/CPU
- `pin_memory=True` only for CUDA, not for MPS
- Adjust `num_workers` based on CPU cores (recommended: 2-4)

#### 6. **Memory Management**
- For GPU memory errors: Reduce batch size
- Gradient accumulation as alternative
- Mixed precision training (FP16) for larger models

## Expected Results

### ResNet18
- **Validation Accuracy:** ~98-99%
- **Training Time:** ~2-4 hours (GPU), ~8-12 hours (CPU)
- **Model Size:** ~43MB

### EfficientNet-B0
- **Validation Accuracy:** ~97-98%
- **Training Time:** ~1.5-3 hours (GPU), ~6-10 hours (CPU)
- **Model Size:** ~16MB

### DenseNet121
- **Validation Accuracy:** ~98-99%
- **Training Time:** ~3-5 hours (GPU), ~12-16 hours (CPU)
- **Model Size:** ~27MB

## Troubleshooting

### Problem: "Out of Memory" Error

**Solution:**
- Reduce batch size (e.g., from 32 to 16)
- Reduce `num_workers`
- Enable mixed precision training

### Problem: Training Stagnates / Loss Doesn't Decrease

**Solution:**
- Check learning rate (possibly too low)
- Check data augmentation (possibly too aggressive)
- Check model initialization

### Problem: Overfitting (Train Acc >> Val Acc)

**Solution:**
- Increase dropout (currently 0.3)
- More data augmentation
- Increase weight decay
- Earlier early stopping

### Problem: Underfitting (Both Acc Low)

**Solution:**
- Train for more epochs
- Increase learning rate
- Increase model capacity
- Reduce data augmentation

### Problem: Dataset Not Found

**Solution:**
- Check path structure (see `FIX_DATA_PATHS.md`)
- Check BASE_DIR logic in notebook
- Set relative paths correctly

## Model Export

After training, models are automatically saved:

- **ResNet18:** `models/brain_tumor_resnet18_v2_trained.pt`
- **EfficientNet-B0:** `models/brain_tumor_efficientnet_b0_trained.pt`
- **DenseNet121:** `models/brain_tumor_densenet121_trained.pt`

**Important:** Model names contain `_trained` for unique identification.

## Evaluation

After training, you should evaluate the model on the separate test set:

```bash
python scripts/evaluate.py --model models/brain_tumor_resnet18_v2_trained.pt
```

This generates:
- Classification Report (Precision, Recall, F1-Score)
- Confusion Matrix
- Misclassified Samples

## Best Practices

1. **Versioning:** Document hyperparameters and results
2. **Backup:** Save checkpoints during long training sessions
3. **Monitoring:** Observe train/val loss curves
4. **Reproducibility:** Set random seeds
5. **Testing:** Always evaluate on separate test set, never on training data

## Next Steps

After successful training:

1. Save models in `models/` directory
2. Document metrics in `runs/` directory
3. Test web application with trained models
4. Test multi-model consensus (all 3 models together)

## Future Improvements

- **Hyperparameter Tuning:** Systematic search with Optuna/Hyperopt
- **Cross-Validation:** K-Fold for more robust evaluation
- **Ensemble Learning:** Weighted combination of models
- **Active Learning:** Feedback data for continuous improvement

---

**Note:** This guide is continuously updated. For questions or problems, please create issues in the repository.
