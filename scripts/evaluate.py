"""
Evaluation Script.

Evaluates a trained checkpoint on a test dataset, calculates classification metrics,
generates a confusion matrix, and exports misclassified samples for error analysis.
"""

import argparse
import logging
from pathlib import Path
import shutil
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets

# Allow importing shared utilities from src/
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from brain_tumor.device import get_device  # noqa: E402
from brain_tumor.model_factory import create_model  # noqa: E402
from brain_tumor.paths import get_data_dirs, get_models_dir, get_runs_dir, project_root  # noqa: E402
from brain_tumor.transforms import build_val_transforms  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def evaluate_model(
    model_path: Path,
    data_dir: Path,
    output_dir: Path,
    model_name: str = "resnet18",
    batch_size: int = 32,
) -> float:
    """
    Run evaluation on the specified dataset and checkpoint.

    Args:
        model_path: Path to the .pt model weights
        data_dir: Directory containing class subfolders of evaluation images
        output_dir: Destination for plots and misclassified images
        model_name: Model architecture family ('resnet18', 'efficientnet', 'densenet')
        batch_size: Evaluation batch size

    Returns:
        Accuracy score as float
    """
    device = get_device()
    logger.info(f"Using device: {device}")

    output_dir.mkdir(parents=True, exist_ok=True)
    misclassified_dir = output_dir / "misclassified"

    val_tf = build_val_transforms()

    logger.info(f"Loading evaluation dataset from {data_dir}...")
    try:
        dataset = datasets.ImageFolder(root=str(data_dir), transform=val_tf)
    except Exception as exc:
        logger.error(f"Failed to load dataset from {data_dir}: {exc}")
        raise

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    class_names = dataset.classes
    logger.info(f"Classes ({len(class_names)}): {class_names}")
    logger.info(f"Total evaluation samples: {len(dataset)}")

    logger.info(f"Loading {model_name} model from {model_path}...")
    model = create_model(model_name=model_name, num_classes=len(class_names), pretrained=False)

    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except Exception:
        state_dict = torch.load(model_path, map_location=device, weights_only=False)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    all_preds: list[int] = []
    all_labels: list[int] = []
    misclassified: list[dict[str, str]] = []

    logger.info("Running evaluation...")
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            preds_cpu = preds.cpu().tolist()
            labels_list = labels.tolist()

            all_preds.extend(preds_cpu)
            all_labels.extend(labels_list)

            start_idx = batch_idx * batch_size
            for offset, (pred_val, true_val) in enumerate(zip(preds_cpu, labels_list)):
                if pred_val != true_val:
                    sample_path, _ = dataset.samples[start_idx + offset]
                    misclassified.append(
                        {
                            "path": sample_path,
                            "true_label": class_names[true_val],
                            "pred_label": class_names[pred_val],
                        }
                    )

    acc = float(accuracy_score(all_labels, all_preds))
    logger.info(f"Accuracy: {acc:.4f}")
    logger.info("\n" + classification_report(all_labels, all_preds, target_names=class_names))

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
    plt.title(f"Confusion Matrix ({model_name})")

    cm_path = output_dir / f"confusion_matrix_{model_name}.png"
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix saved to {cm_path}")

    if misclassified_dir.exists():
        shutil.rmtree(misclassified_dir)
    misclassified_dir.mkdir(parents=True, exist_ok=True)

    for item in misclassified:
        folder_name = f"{item['true_label']}_as_{item['pred_label']}"
        target_subfolder = misclassified_dir / folder_name
        target_subfolder.mkdir(parents=True, exist_ok=True)

        src_file = Path(item["path"])
        shutil.copy2(src_file, target_subfolder / src_file.name)

    logger.info(f"Saved {len(misclassified)} misclassified images to {misclassified_dir}")
    return acc


def main() -> None:
    root = project_root()
    _, default_data_dir = get_data_dirs(root)
    default_models_dir = get_models_dir(root)
    default_runs_dir = get_runs_dir(root)

    parser = argparse.ArgumentParser(description="Evaluate brain tumor classification model.")
    parser.add_argument(
        "--model-name",
        type=str,
        default="resnet18",
        choices=["resnet18", "efficientnet", "densenet"],
        help="Model architecture family",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to checkpoint file (defaults to models/brain_tumor_<model_name>_v2_trained.pt)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir,
        help="Path to test dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_runs_dir / "evaluation",
        help="Directory to save evaluation plots and misclassified images",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    args = parser.parse_args()

    model_path = args.model_path
    if model_path is None:
        candidate_names = [
            f"brain_tumor_{args.model_name}_b0_v2_trained.pt",
            f"brain_tumor_{args.model_name}_v2_trained.pt",
            f"brain_tumor_{args.model_name}_v2.pt",
        ]
        for name in candidate_names:
            p = default_models_dir / name
            if p.exists():
                model_path = p
                break
        if model_path is None:
            model_path = default_models_dir / candidate_names[1]

    if not model_path.exists():
        logger.error(f"Checkpoint not found at: {model_path}")
        sys.exit(1)

    evaluate_model(
        model_path=model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
