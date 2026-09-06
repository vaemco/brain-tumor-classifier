"""
Feedback storage, CSV export, and metric aggregation for the active learning dashboard.
"""

from collections import Counter
import csv
import logging
from pathlib import Path
import shutil

logger = logging.getLogger(__name__)

FEEDBACK_HEADER = [
    "filename",
    "predicted_label",
    "true_label",
    "confidence",
    "timestamp",
    "model_version",
]


def init_feedback_file(feedback_file: Path) -> None:
    """Ensure feedback CSV exists with header row."""
    feedback_file.parent.mkdir(parents=True, exist_ok=True)
    if not feedback_file.exists():
        with open(feedback_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(FEEDBACK_HEADER)


def save_feedback(
    feedback_file: Path,
    images_dir: Path,
    test_dirs: list[Path],
    data: dict,
) -> None:
    """
    Save user correction feedback using csv.writer to prevent injection attacks.
    Optionally copy the referenced image from test directories into images_dir.
    """
    init_feedback_file(feedback_file)
    images_dir.mkdir(parents=True, exist_ok=True)

    filename = str(data.get("filename", "")).strip()
    predicted_label = str(data.get("predicted_label", "")).strip()
    true_label = str(data.get("true_label", "")).strip()
    confidence = float(data.get("confidence", 0.0))
    timestamp = str(data.get("timestamp", "")).strip()
    model_version = str(data.get("model_version", "")).strip()

    with open(feedback_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                filename,
                predicted_label,
                true_label,
                confidence,
                timestamp,
                model_version,
            ]
        )

    if filename:
        for search_dir in test_dirs:
            if search_dir.exists():
                matches = list(search_dir.rglob(filename))
                if matches:
                    shutil.copy2(matches[0], images_dir / filename)
                    break


def get_feedback_stats(feedback_file: Path, classes: list[str]) -> dict:
    """
    Compute aggregated feedback statistics for the dashboard.
    """
    stats: dict = {
        "total_feedback": 0,
        "correct_predictions": 0,
        "incorrect_predictions": 0,
        "accuracy": 0.0,
        "class_distribution": {cls: {"total": 0, "correct": 0} for cls in classes},
        "confusion_matrix": {cls: {c: 0 for c in classes} for cls in classes},
        "recent_feedback": [],
        "daily_counts": {},
        "model_accuracy_by_class": {cls: 0.0 for cls in classes},
    }

    if not feedback_file.exists():
        return stats

    try:
        with open(feedback_file, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        stats["total_feedback"] = len(rows)

        for row in rows:
            predicted = row.get("predicted_label", "")
            true_label = row.get("true_label", "")
            timestamp = row.get("timestamp", "")
            try:
                conf = float(row.get("confidence", 0.0))
            except (ValueError, TypeError):
                conf = 0.0

            is_correct = predicted == true_label
            if is_correct:
                stats["correct_predictions"] += 1
            else:
                stats["incorrect_predictions"] += 1

            if true_label in stats["class_distribution"]:
                stats["class_distribution"][true_label]["total"] += 1
                if is_correct:
                    stats["class_distribution"][true_label]["correct"] += 1

            if predicted in classes and true_label in classes:
                stats["confusion_matrix"][true_label][predicted] += 1

            if timestamp:
                date_str = timestamp[:10]
                stats["daily_counts"][date_str] = stats["daily_counts"].get(date_str, 0) + 1

        if stats["total_feedback"] > 0:
            stats["accuracy"] = stats["correct_predictions"] / stats["total_feedback"]

        for cls in classes:
            total = stats["class_distribution"][cls]["total"]
            correct = stats["class_distribution"][cls]["correct"]
            if total > 0:
                stats["model_accuracy_by_class"][cls] = correct / total

        recent = rows[-10:][::-1]
        stats["recent_feedback"] = [
            {
                "filename": r.get("filename", ""),
                "predicted": r.get("predicted_label", ""),
                "true_label": r.get("true_label", ""),
                "confidence": float(r.get("confidence", 0.0) or 0.0),
                "timestamp": r.get("timestamp", ""),
                "is_correct": r.get("predicted_label") == r.get("true_label"),
            }
            for r in recent
        ]

        return stats

    except Exception as exc:
        logger.error(f"Error reading feedback stats: {exc}")
        return stats
