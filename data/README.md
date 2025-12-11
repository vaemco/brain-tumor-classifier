# Data Directory

Datasets are not committed. Place the Kaggle datasets here:

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

If your dataset is not split, run `python scripts/prepare_data.py` to create the structure above.
