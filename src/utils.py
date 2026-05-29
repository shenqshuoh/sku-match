"""Shared utility functions."""

import gc
from pathlib import Path

import numpy as np
import torch

# Ultralytics text model weights (mobileclip2_b.ts) are stored in models/
# so they don't need to be downloaded from GitHub at runtime (GFW-safe).
_MODELS_DIR = str(Path(__file__).resolve().parent.parent / "models")


def configure_ultralytics_weights() -> None:
    """Set ultralytics weights_dir to local models/ to avoid GitHub downloads."""
    from ultralytics import settings

    settings.update({"weights_dir": _MODELS_DIR})


def detect_device() -> str:
    """Auto-detect best available compute device."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def free_gpu_memory() -> None:
    """Release cached GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def embedding_to_list(emb) -> list:
    """Convert numpy array or array-like to a plain Python list for Chroma."""
    return emb.tolist() if isinstance(emb, np.ndarray) else list(emb)
