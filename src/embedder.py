import logging
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from src.features import Features, fused_embedding
from src.utils import detect_device

logger = logging.getLogger(__name__)

MODEL_DIMENSIONS: dict[str, int] = {
    "facebook/dinov2-small": 384,
    "facebook/dinov2-base": 768,
    "facebook/dinov2-large": 1024,
    "facebook/dinov2-giant": 1536,
    "facebook/dinov2-small-with-registers": 384,
    "facebook/dinov2-base-with-registers": 768,
    "facebook/dinov2-large-with-registers": 1024,
    "facebook/dinov2-giant-with-registers": 1536,
}

EmbedderVariant = Literal[
    "facebook/dinov2-small",
    "facebook/dinov2-base",
    "facebook/dinov2-large",
    "facebook/dinov2-giant",
    "facebook/dinov2-small-with-registers",
    "facebook/dinov2-base-with-registers",
    "facebook/dinov2-large-with-registers",
    "facebook/dinov2-giant-with-registers",
]

__all__ = ["Embedder", "EmbedderProtocol", "EmbedderVariant", "MODEL_DIMENSIONS"]

# Preprocessing config to match the original 518x518 input
_PROCESSOR_KWARGS = {
    "size": {"height": 518, "width": 518},
    "crop_size": {"height": 518, "width": 518},
}


@runtime_checkable
class EmbedderProtocol(Protocol):
    """Interface for SKU image embedders."""

    dim: int
    device: str

    def embed(self, image: Image.Image) -> np.ndarray: ...
    def embed_batch(self, images: list[Image.Image]) -> np.ndarray: ...
    def embed_path(self, path: Path) -> np.ndarray: ...
    def to(self, device: str) -> None: ...


def _resolve_model_path(model_name: str) -> str:
    """Resolve a HF Hub model ID to a local directory if one exists.

    Checks (in order):
      1. model_name itself (if it's already a path)
      2. models/<short_name>  (e.g. facebook/dinov2-small → models/dinov2-small)
    Falls back to the original model_name for HF Hub / cache resolution.
    """
    if Path(model_name).is_dir():
        return model_name
    local = Path("models") / model_name.split("/")[-1]
    if local.is_dir():
        logger.info("Using local model: %s → %s", model_name, local)
        return str(local)
    return model_name


class Embedder:
    """DINOv2 image embedder backed by HuggingFace transformers + Optimum ONNX."""

    def __init__(
        self,
        model_name: EmbedderVariant | str = "facebook/dinov2-base",
        device: str | None = None,
        use_onnx: bool = False,
        use_fused: bool = False,
        fuse_alpha: float = 0.5,
        gem_p: float = 3.0,
    ):
        if device is None:
            device = detect_device()

        if model_name not in MODEL_DIMENSIONS:
            raise ValueError(
                f"Unknown embedding model: {model_name}. "
                f"Supported: {sorted(MODEL_DIMENSIONS.keys())}"
            )

        self.model_name = model_name
        self.device = device
        self.dim = MODEL_DIMENSIONS[model_name]
        self._use_onnx = use_onnx
        self._use_fused = use_fused
        self._fuse_alpha = fuse_alpha
        self._gem_p = gem_p

        # Resolve local path before loading (avoids HF Hub network access)
        resolved = _resolve_model_path(model_name)

        logger.info(
            "Embedder initializing: model=%s, resolved=%s, device=%s, dim=%d, onnx=%s, fused=%s, alpha=%s",
            model_name, resolved, device, self.dim, use_onnx, use_fused, fuse_alpha,
        )

        # Shared image processor for both backends
        self._processor = AutoImageProcessor.from_pretrained(
            resolved, local_files_only=True, **_PROCESSOR_KWARGS,
        )

        if use_onnx:
            from optimum.onnxruntime import ORTModelForFeatureExtraction

            provider = (
                "CUDAExecutionProvider" if device == "cuda"
                else "CPUExecutionProvider"
            )
            self._model = ORTModelForFeatureExtraction.from_pretrained(
                resolved, export=True, provider=provider, local_files_only=True,
            )
        else:
            dtype = torch.float16 if device == "cuda" else torch.float32
            self._model = AutoModel.from_pretrained(
                resolved, dtype=dtype, local_files_only=True,
            )
            self._model.eval()
            if device != "cpu":
                self._model.to(device)

        logger.info("Embedder ready: %s", model_name)

    def to(self, device: str) -> None:
        """Move model to the specified device."""
        self.device = device
        self._model.to(device)

    def embed(self, image: Image.Image) -> np.ndarray:
        """Embed a single image. Returns shape (dim,)."""
        return self.embed_batch([image])[0]

    def embed_batch(self, images: list[Image.Image]) -> np.ndarray:
        """Embed a batch of images. Returns shape (batch, dim).

        Returns fused (CLS + GeM) or CLS-only vectors depending on configuration.
        """
        if not images:
            return np.empty((0, self.dim), dtype=np.float32)

        features_list = self.extract_features_batch(images)
        return np.stack([self.features_to_embedding(f) for f in features_list])

    def embed_path(self, path: Path) -> np.ndarray:
        """Embed an image from a file path."""
        image = Image.open(path).convert("RGB")
        return self.embed(image)

    def extract_features_batch(self, images: list[Image.Image]) -> list[Features]:
        """Extract CLS token and patch tokens for a batch of images.

        Returns a list of Features objects, one per image.  Both CLS and patches
        are L2-normalized.  Uses a single forward pass regardless of fusion config.

        Handles register tokens transparently: for models with registers
        (e.g. ``dinov2-base-with-registers``), register tokens at positions
        ``[1:1+K]`` are skipped and only patch tokens are returned.
        """
        if not images:
            return []

        inputs = self._processor(images=images, return_tensors="pt")

        if self._use_onnx:
            outputs = self._model(**inputs)
        else:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            if self.device == "cuda":
                inputs = {
                    k: (v.half() if v.is_floating_point() else v)
                    for k, v in inputs.items()
                }
            with torch.no_grad():
                outputs = self._model(**inputs)

        lhs = outputs.last_hidden_state  # (batch, 1+K+N, dim)

        # Number of register tokens (0 for standard models)
        num_registers = getattr(self._model.config, "num_register_tokens", 0)

        # CLS is always at position 0
        cls_tokens = lhs[:, 0, :]            # (batch, dim)
        # Patches start after CLS + registers
        patch_tokens = lhs[:, 1 + num_registers:, :]  # (batch, N, dim)

        # L2-normalize both
        cls_norm = cls_tokens / cls_tokens.norm(dim=-1, keepdim=True)
        patch_norm = patch_tokens / patch_tokens.norm(dim=-1, keepdim=True)

        # Convert to numpy
        cls_np = cls_norm.float().cpu().numpy()       # (batch, dim)
        patches_np = patch_norm.float().cpu().numpy()  # (batch, N, dim)

        return [
            Features(cls=cls_np[i], patches=patches_np[i])
            for i in range(len(images))
        ]

    def features_to_embedding(self, features: Features) -> np.ndarray:
        """Convert a Features object to the configured embedding vector.

        Returns fused embedding if ``use_fused=True``, otherwise CLS-only.
        """
        if self._use_fused:
            return fused_embedding(features.cls, features.patches, self._fuse_alpha, self._gem_p)
        return features.cls.copy()
