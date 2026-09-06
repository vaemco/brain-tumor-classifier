"""
Notebook-friendly performance helpers.
CUDA-only tweaks to speed up training without affecting MPS/CPU.

Includes training utilities like Mixup for better generalization.
"""

from __future__ import annotations

import random
from contextlib import nullcontext
import inspect
import os
import sys
from typing import Any, Callable, Dict, Optional, Tuple, cast

# On Windows with Conda, ensure cuDNN DLLs in Library/bin can be loaded
if sys.platform == "win32":
    _lib_bin = os.path.join(sys.prefix, "Library", "bin")
    if os.path.isdir(_lib_bin):
        if _lib_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = _lib_bin + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(_lib_bin)
            except Exception:
                pass

import torch
import torch.nn as nn


def is_cuda_device(device: torch.device) -> bool:
    return isinstance(device, torch.device) and device.type == "cuda"


def configure_cuda_perf(device: torch.device) -> None:
    """Enable CUDA-specific perf switches; no-op on other backends."""
    if not is_cuda_device(device):
        return

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def _recommended_num_workers(
    device: torch.device, *, max_cuda_workers: int, max_cpu_workers: int
) -> int:
    cpu_count = os.cpu_count() or 2
    if is_cuda_device(device):
        return min(max_cuda_workers, max(2, cpu_count // 2))
    return min(max_cpu_workers, max(0, cpu_count // 4))


def get_dataloader_kwargs(
    device: torch.device,
    *,
    num_workers: Optional[int] = None,
    max_cuda_workers: int = 8,
    max_cpu_workers: int = 2,
    prefetch_factor: int = 4,
) -> Dict[str, Any]:
    is_cuda = is_cuda_device(device)
    if num_workers is None:
        num_workers = _recommended_num_workers(
            device, max_cuda_workers=max_cuda_workers, max_cpu_workers=max_cpu_workers
        )
    kwargs: Dict[str, Any] = {"num_workers": num_workers, "pin_memory": bool(is_cuda)}
    if is_cuda and num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = prefetch_factor
    return kwargs


def autocast_context(device: torch.device):
    if is_cuda_device(device):
        if hasattr(torch, "autocast"):
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return torch.cuda.amp.autocast(dtype=torch.float16)
    return nullcontext()


def get_grad_scaler(device: torch.device) -> torch.amp.GradScaler:
    return torch.amp.GradScaler("cuda", enabled=is_cuda_device(device))


def move_batch(
    x: torch.Tensor, y: torch.Tensor, device: torch.device, use_cuda: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    if use_cuda:
        x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        return x, y
    return x.to(device), y.to(device)


def maybe_channels_last(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    if is_cuda_device(device):
        return model.to(device, memory_format=torch.channels_last)  # type: ignore[call-overload]
    return model


def maybe_compile(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    if not is_cuda_device(device):
        return model
    if not hasattr(torch, "compile"):
        return model
    # triton is required for torch.compile on CUDA but only available on Linux
    try:
        import triton  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        print("[INFO] triton not available - skipping torch.compile (Windows limitation)")
        return model
    try:
        compiled = torch.compile(model, mode="max-autotune")
        return cast(torch.nn.Module, compiled)
    except Exception as e:
        print(f"[WARNING] torch.compile failed: {e} - using eager mode")
        return model


def get_adamw_kwargs(device: torch.device) -> Dict[str, Any]:
    if not is_cuda_device(device):
        return {}
    try:
        if "fused" in inspect.signature(torch.optim.AdamW).parameters:
            return {"fused": True}
    except (TypeError, ValueError):
        pass
    return {}


# ============================================================================
# Anti-Clever-Hans Training Utilities
# ============================================================================


def mixup_data(
    x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Mixup augmentation: blends pairs of images and labels.
    Helps prevent memorization and improves generalization.

    Args:
        x: Input batch (B, C, H, W)
        y: Labels (B,)
        alpha: Mixup interpolation strength (0 = no mixup)

    Returns:
        mixed_x: Blended images
        y_a: Original labels
        y_b: Shuffled labels
        lam: Interpolation coefficient
    """
    if alpha > 0:
        lam = random.betavariate(alpha, alpha)
        # Ensure we use at least 50% of original image
        lam = max(lam, 1 - lam)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(
    criterion: Callable, pred: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float
) -> torch.Tensor:
    """Compute mixup loss as weighted combination of both labels."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy with label smoothing for better calibration.
    Reduces overconfidence and helps with Clever-Hans prevention.
    """

    def __init__(self, smoothing: float = 0.1, reduction: str = "mean"):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_classes = pred.size(-1)
        log_preds = torch.log_softmax(pred, dim=-1)

        # Smooth targets
        with torch.no_grad():
            smooth_target = torch.zeros_like(pred)
            smooth_target.fill_(self.smoothing / (n_classes - 1))
            smooth_target.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)

        loss = -smooth_target * log_preds
        loss = loss.sum(dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def train_step_with_mixup(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    criterion: Callable,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_cuda: bool,
    mixup_alpha: float = 0.2,
    gradient_clip: float = 1.0,
) -> Tuple[float, int, int]:
    """
    Single training step with Mixup augmentation and gradient clipping.

    Returns:
        loss: Scaled loss value
        correct: Number of correct predictions (based on dominant label)
        total: Batch size
    """
    x, y = move_batch(x, y, device, use_cuda)

    # Apply mixup
    if mixup_alpha > 0:
        x, y_a, y_b, lam = mixup_data(x, y, mixup_alpha)
    else:
        y_a, y_b, lam = y, y, 1.0

    optimizer.zero_grad(set_to_none=True)

    with autocast_context(device):
        out = model(x)
        if mixup_alpha > 0:
            loss = mixup_criterion(criterion, out, y_a, y_b, lam)
        else:
            loss = criterion(out, y)

    if scaler.is_enabled():
        scaler.scale(loss).backward()
        if gradient_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

    # For accuracy, use the dominant label (y_a with lam >= 0.5)
    preds = out.argmax(1)
    correct = (preds == y_a).sum().item()
    total = y.size(0)

    return loss.item() * total, correct, total
