import os
import random
import shutil

SOURCE_DIR = "/Users/valentinemser/dev_projects/03_data_projects/data_brain_tumor/data/Brain_Tumor_Dataset/external_data"
TARGET_DIR = "/Users/valentinemser/dev_projects/03_data_projects/data_brain_tumor/data/Brain_Tumor_Dataset/external_dataset"

TRAIN_RATIO = 0.8  # 80% train / 20% test
SEED = 42

random.seed(SEED)

classes = ["glioma", "meningioma", "notumor", "pituitary"]

for split in ["training", "testing"]:
    for cls in classes:
        os.makedirs(os.path.join(TARGET_DIR, split, cls), exist_ok=True)

for cls in classes:
    class_path = os.path.join(SOURCE_DIR, cls)
    images = [f for f in os.listdir(class_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    random.shuffle(images)

    split_index = int(len(images) * TRAIN_RATIO)
    train_imgs = images[:split_index]
    test_imgs = images[split_index:]

    for img in train_imgs:
        src = os.path.join(class_path, img)
        dst = os.path.join(TARGET_DIR, "training", cls, img)
        shutil.copy2(src, dst)

    for img in test_imgs:
        src = os.path.join(class_path, img)
        dst = os.path.join(TARGET_DIR, "testing", cls, img)
        shutil.copy2(src, dst)

    print(f"{cls}: {len(train_imgs)} train / {len(test_imgs)} test")
