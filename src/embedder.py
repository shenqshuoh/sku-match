import json
import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from src.features import Features, fused_embedding
from src.utils import detect_device

logger = logging.getLogger(__name__)

DEFAULT_EMB_MODEL = "models/dinov2-base"

__all__ = ["Embedder", "EmbedderProtocol", "DEFAULT_EMB_MODEL"]

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


class Embedder:
    """DINOv2 image embedder backed by HuggingFace transformers + Optimum ONNX."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMB_MODEL,
        device: str | None = None,
        use_onnx: bool = False,
        use_fused: bool = False,
        fuse_alpha: float = 0.5,
        gem_p: float = 3.0,
    ):
        if device is None:
            device = detect_device()

        model_path = Path(model_name)
        if not model_path.is_dir():
            raise FileNotFoundError(
                f"Embedding model directory not found: {model_name}"
            )

        # Read embedding dimension from model config
        config_path = model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Model config not found: {config_path}"
            )
        self.dim = json.loads(config_path.read_text())["hidden_size"]

        self.model_name = model_name
        self.device = device
        self._use_onnx = use_onnx
        self._use_fused = use_fused
        self._fuse_alpha = fuse_alpha
        self._gem_p = gem_p

        logger.info(
            "Embedder initializing: model=%s, device=%s, dim=%d, onnx=%s, fused=%s, alpha=%s",
            model_name, device, self.dim, use_onnx, use_fused, fuse_alpha,
        )

        # Shared image processor for both backends
        self._processor = AutoImageProcessor.from_pretrained(
            model_name, local_files_only=True, **_PROCESSOR_KWARGS,
        )

        if use_onnx:
            from optimum.onnxruntime import ORTModelForFeatureExtraction

            provider = (
                "CUDAExecutionProvider" if device == "cuda"
                else "CPUExecutionProvider"
            )
            self._model = ORTModelForFeatureExtraction.from_pretrained(
                model_name, export=True, provider=provider, local_files_only=True,
            )
        else:
            dtype = torch.float16 if device == "cuda" else torch.float32
            self._model = AutoModel.from_pretrained(
                model_name, dtype=dtype, local_files_only=True,
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
