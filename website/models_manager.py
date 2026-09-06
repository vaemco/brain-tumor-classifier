"""
Model lifecycle management, cached inference, ensemble voting, and Grad-CAM caching.
"""

import logging
from pathlib import Path
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
import torch
from torch import nn
import torch.nn.functional as F

from brain_tumor.device import get_device
from brain_tumor.model_factory import create_model, get_target_layer
from website.gradcam_utils import generate_gradcam

logger = logging.getLogger(__name__)

CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

MODEL_COLORS: dict[str, str] = {
    "resnet18": "#3B82F6",
    "efficientnet": "#EF4444",
    "densenet": "#10B981",
}

LAYER_NAMES: dict[str, list[str]] = {
    "resnet18": [
        "conv1",
        "layer1",
        "layer2",
        "layer3",
        "layer4",
        "avgpool",
        "fc",
    ],
    "efficientnet": [
        "stem",
        "blocks1-2",
        "blocks3-4",
        "blocks5-6",
        "blocks7",
        "head",
        "fc",
    ],
    "densenet": [
        "conv0",
        "denseblock1",
        "denseblock2",
        "denseblock3",
        "denseblock4",
        "fc",
    ],
}


class ModelsManager:
    """
    Manages loading, evaluation, ensembling, and cached GradCAM objects for models.
    """

    def __init__(self, model_paths: dict[str, Path]) -> None:
        self.model_paths = model_paths
        self.device = get_device()
        self.models: dict[str, nn.Module] = {}
        self.gradcams: dict[str, GradCAM] = {}
        self.classes = CLASSES

    def load_all_models(self) -> None:
        """Load available checkpoint files and initialize cached GradCAM instances."""
        logger.info(f"Using device: {self.device}")
        for model_name, path in self.model_paths.items():
            if not path.exists():
                logger.warning(f"Checkpoint not found for {model_name}: {path}")
                continue

            try:
                model = create_model(model_name=model_name, num_classes=len(self.classes))
                try:
                    state_dict = torch.load(path, map_location=self.device, weights_only=True)
                except Exception:
                    state_dict = torch.load(path, map_location=self.device, weights_only=False)

                model.load_state_dict(state_dict)
                model.to(self.device)
                model.eval()

                self.models[model_name] = model

                # Cache GradCAM instance for this model
                target_layers = get_target_layer(model, model_name)
                self.gradcams[model_name] = GradCAM(model=model, target_layers=target_layers)
                logger.info(f"Loaded {model_name} from {path}")

            except Exception as exc:
                logger.error(f"Failed to load {model_name} from {path}: {exc}")

    @property
    def primary_model(self) -> nn.Module | None:
        """Return the default primary model (prefers resnet18, falls back to first loaded)."""
        return self.models.get("resnet18") or (
            list(self.models.values())[0] if self.models else None
        )

    @property
    def primary_model_name(self) -> str:
        """Return the name of the primary model."""
        if "resnet18" in self.models:
            return "resnet18"
        if self.models:
            return list(self.models.keys())[0]
        return "none"

    def predict_detailed(
        self, image_tensor: torch.Tensor, image_pil: Image.Image
    ) -> dict:
        """
        Run multi-model prediction, layer progress estimation, and GradCAM on the top model.
        """
        model_results: dict = {}
        all_probs: list[np.ndarray] = []

        for model_name, loaded_model in self.models.items():
            with torch.no_grad():
                output = loaded_model(image_tensor)
                probs = F.softmax(output, dim=1).cpu().numpy()[0]

            top_idx = int(probs.argmax())
            confidence = float(probs[top_idx])

            layers = LAYER_NAMES.get(model_name, ["fc"])
            num_layers = len(layers)
            layer_progress: list[float] = []

            for i in range(num_layers):
                progress = (i + 1) / num_layers
                simulated_conf = confidence * (1.0 - np.exp(-3.0 * progress)) / (1.0 - np.exp(-3.0))
                layer_progress.append(round(float(simulated_conf), 3))

            model_results[model_name] = {
                "color": MODEL_COLORS.get(model_name, "#888888"),
                "predictions": [
                    {"class": self.classes[i], "probability": float(probs[i])}
                    for i in range(len(self.classes))
                ],
                "layer_progress": layer_progress,
                "layer_names": layers,
                "top_class": self.classes[top_idx],
                "confidence": confidence,
            }
            all_probs.append(probs)

        avg_probs = np.mean(all_probs, axis=0)
        averaged_predictions = [
            {"class": self.classes[i], "probability": float(avg_probs[i])}
            for i in range(len(self.classes))
        ]

        final_idx = int(avg_probs.argmax())
        final_result = {
            "class": self.classes[final_idx],
            "confidence": float(avg_probs[final_idx]),
        }

        best_model_name = max(
            model_results, key=lambda k: model_results[k]["confidence"]
        )

        best_cam = self.gradcams[best_model_name]
        heatmap, bbox = generate_gradcam(
            cam=best_cam,
            image_tensor=image_tensor,
            image_pil=image_pil,
        )

        return {
            "model_results": model_results,
            "averaged_predictions": averaged_predictions,
            "final_result": final_result,
            "best_model_name": best_model_name,
            "heatmap": heatmap,
            "bbox": bbox,
        }

    def compute_consensus(self, image_tensor: torch.Tensor) -> tuple[dict, str, float]:
        """
        Run all models and aggregate voting consensus.
        """
        model_predictions = []
        for model_name, loaded_model in self.models.items():
            with torch.no_grad():
                output = loaded_model(image_tensor)
                probs = F.softmax(output, dim=1).cpu().numpy()[0]

            top_idx = int(probs.argmax())
            model_predictions.append(
                {
                    "name": model_name.upper().replace("_", "-"),
                    "prediction": self.classes[top_idx],
                    "confidence": float(probs[top_idx]),
                }
            )

        votes = [pred["prediction"] for pred in model_predictions]
        winner = max(set(votes), key=votes.count) if votes else self.classes[0]
        vote_count = votes.count(winner)

        winner_confidences = [
            p["confidence"] for p in model_predictions if p["prediction"] == winner
        ]
        avg_confidence = float(np.mean(winner_confidences)) if winner_confidences else 0.0

        total_models = len(self.models)
        if vote_count == total_models:
            status = "High Consensus" if vote_count >= 3 else "Full Agreement"
        elif vote_count > total_models // 2:
            status = "Medium Consensus"
        else:
            status = "Low Consensus"

        consensus_data = {
            "models": model_predictions,
            "result": {
                "winner": winner,
                "score": f"{vote_count}/{total_models}",
                "status": status,
                "avg_confidence": avg_confidence,
            },
        }

        return consensus_data, winner, avg_confidence

    def compute_calibration(self, image_tensor: torch.Tensor) -> dict:
        """
        Analyze prediction confidence, entropy, and overconfidence risk.
        """
        calibration_data = {}
        for model_name, loaded_model in self.models.items():
            with torch.no_grad():
                output = loaded_model(image_tensor)
                probs = F.softmax(output, dim=1).cpu().numpy()[0]

            top_idx = int(probs.argmax())
            top_conf = float(probs[top_idx])

            entropy = float(-np.sum(probs * np.log(probs + 1e-10)))
            max_entropy = float(np.log(len(self.classes)))
            normalized_entropy = entropy / max_entropy

            overconfidence_risk = "Low"
            suggested_temp = 1.0

            if top_conf > 0.95 and normalized_entropy < 0.2:
                overconfidence_risk = "High"
                suggested_temp = 1.5
            elif top_conf > 0.85 and normalized_entropy < 0.3:
                overconfidence_risk = "Medium"
                suggested_temp = 1.2

            sorted_probs = np.sort(probs)[::-1]
            margin = float(sorted_probs[0] - sorted_probs[1])

            calibration_data[model_name] = {
                "top_class": self.classes[top_idx],
                "confidence": top_conf,
                "entropy": entropy,
                "normalized_entropy": normalized_entropy,
                "margin_top2": margin,
                "overconfidence_risk": overconfidence_risk,
                "suggested_temperature": suggested_temp,
                "all_probabilities": {
                    self.classes[i]: float(probs[i]) for i in range(len(self.classes))
                },
            }

        predictions = [calibration_data[m]["top_class"] for m in calibration_data]
        unique_predictions = list(set(predictions))
        disagreement = len(unique_predictions) > 1

        return {
            "models": calibration_data,
            "ensemble_disagreement": disagreement,
            "unique_predictions": unique_predictions,
            "recommendation": (
                "Models disagree - manual review recommended"
                if disagreement
                else "Models agree - prediction likely reliable"
            ),
        }

    def get_info(self, model_version: str) -> dict:
        """Return parameter count and architecture details for each model."""
        info = {}
        for name, loaded_model in self.models.items():
            total_params = sum(p.numel() for p in loaded_model.parameters())
            if "resnet" in name:
                arch = "ResNet18"
                layers = "4 residual blocks"
            elif "efficient" in name:
                arch = "EfficientNet-B0"
                layers = "7 MBConv blocks"
            elif "dense" in name:
                arch = "DenseNet121"
                layers = "4 dense blocks (121 layers)"
            else:
                arch = name.capitalize()
                layers = "Custom"

            info[name] = {
                "architecture": arch,
                "layers": layers,
                "total_parameters": total_params,
                "total_parameters_human": f"{total_params / 1e6:.2f}M",
                "model_path": str(self.model_paths.get(name, "Unknown")),
                "device": str(self.device),
            }

        return {
            "models": info,
            "version": model_version,
            "classes": self.classes,
            "num_classes": len(self.classes),
        }
