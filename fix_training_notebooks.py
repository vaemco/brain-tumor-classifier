import json
import sys

# Define the missing data augmentation cell that needs to be added
DATA_AUGMENTATION_CELL = {
    "cell_type": "code",
    "execution_count": None,
    "id": "data_augmentation_cell",
    "metadata": {},
    "outputs": [],
    "source": [
        "# Data Augmentation and Preprocessing\n",
        "# IMPORTANT: PIL transforms first, then ToTensor, then tensor-based transforms\n",
        "train_tf = transforms.Compose([\n",
        "    transforms.Resize((224, 224)),\n",
        "    transforms.RandomHorizontalFlip(p=0.5),\n",
        "    transforms.RandomRotation(degrees=15),\n",
        "    transforms.ColorJitter(brightness=0.2, contrast=0.2),\n",
        "    transforms.ToTensor(),  # Convert PIL Image to Tensor\n",
        "    AddGaussianNoise(mean=0.0, std=0.05),  # Works on tensors only\n",
        "    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),\n",
        "])\n",
        "\n",
        "val_tf = transforms.Compose([\n",
        "    transforms.Resize((224, 224)),\n",
        "    transforms.ToTensor(),\n",
        "    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),\n",
        "])\n",
        "\n",
        "print(\"Data transformations defined.\")"
    ]
}

def fix_notebook(notebook_path):
    """Fix a training notebook by adding the missing data augmentation cell."""
    print(f"Fixing {notebook_path}...")

    # Read notebook
    with open(notebook_path, 'r') as f:
        nb = json.load(f)

    # Find the cell that tries to use train_tf (should be cell about loading datasets)
    dataset_cell_index = None
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell.get('source', []))
            if 'dataset1_train = datasets.ImageFolder' in source and 'train_tf' in source:
                dataset_cell_index = i
                break

    if dataset_cell_index is None:
        print(f"Could not find dataset loading cell in {notebook_path}")
        return False

    # Remove existing augmentation cells (old/incorrect ones)
    cells_to_remove = []
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell.get('source', []))
            if 'train_tf = transforms.Compose' in source:
                cells_to_remove.append(i)
                print(f"Found existing augmentation cell at position {i}, will remove it")

    # Remove cells in reverse order to maintain indices
    for i in sorted(cells_to_remove, reverse=True):
        nb['cells'].pop(i)
        # Adjust dataset_cell_index if needed
        if i < dataset_cell_index:
            dataset_cell_index -= 1

    # Insert the augmentation cell before the dataset loading cell
    nb['cells'].insert(dataset_cell_index, DATA_AUGMENTATION_CELL)
    print(f"Inserted corrected data augmentation cell at position {dataset_cell_index}")

    # Save fixed notebook
    with open(notebook_path, 'w') as f:
        json.dump(nb, f, indent=1)

    print(f"Fixed {notebook_path}")
    return True

if __name__ == "__main__":
    notebooks = [
        "/Users/valentinemser/dev_projects/03_data_projects/data_brain_tumor/notebooks/02_train_resnet18.ipynb",
        "/Users/valentinemser/dev_projects/03_data_projects/data_brain_tumor/notebooks/03_train_efficientnet_b0.ipynb",
        "/Users/valentinemser/dev_projects/03_data_projects/data_brain_tumor/notebooks/04_train_densenet121.ipynb"
    ]

    fixed_count = 0
    for nb_path in notebooks:
        if fix_notebook(nb_path):
            fixed_count += 1

    print(f"\nFixed {fixed_count} out of {len(notebooks)} notebooks")
