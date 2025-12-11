# Data Path Fix for Training Notebooks

## Problem

The notebooks may still contain placeholder paths and absolute directories.

## Solution (Relative and Anonymous)

Use the following block in the path cell so that paths work without usernames:

```python
from pathlib import Path

BASE_DIR = Path.cwd().resolve()
while not (BASE_DIR / "data").exists() and BASE_DIR.parent != BASE_DIR:
    BASE_DIR = BASE_DIR.parent  # go up until data directory is found

data_dir = BASE_DIR / "data" / "Brain_Tumor_Dataset" / "Training"
external_data_dir = BASE_DIR / "data" / "Brain_Tumor_Dataset" / "external_dataset" / "training"
```

## How to Fix the Notebooks

Set the block above in the path cell of:

- `train_densenet121.ipynb`
- `train_efficientnet_b0.ipynb`
- `train_resnet18.ipynb`

## After That You Can Continue

1. Save the notebook.
2. Execute cells (Run All).
3. Training should start and paths remain GitHub-compatible without personal directories.
