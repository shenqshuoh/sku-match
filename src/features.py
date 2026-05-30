"""DINOv2 feature representation and fusion functions.

Provides the Features dataclass for holding CLS + patch token outputs from
DINOv2, along with GeM pooling and weighted fusion utilities used by the
embedder and re-ranking pipeline.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class Features:
    """Raw DINOv2 output for a single image.

    Both cls and patches are L2-normalized along the last dimension.
    """

    __slots__ = ("cls", "patches", "dim")

    def __init__(self, cls: NDArray[np.floating], patches: NDArray[np.floating]) -> None:
        self.cls = cls          # (dim,) L2-normalized
        self.patches = patches  # (N, dim) L2-normalized
        self.dim = cls.shape[-1]

    def __repr__(self) -> str:
        n_patches = self.patches.shape[0] if self.patches.ndim == 2 else 0
        return f"Features(dim={self.dim}, patches={n_patches})"


def gem_pool(patches: NDArray[np.floating], p: float = 3.0) -> NDArray[np.floating]:
    """Generalized Mean (GeM) pooling over patch tokens.

    Args:
        patches: (N, dim) array of L2-normalized patch tokens.
        p: GeM exponent. Higher p emphasises larger activations.

    Returns:
        (dim,) L2-normalized pooled vector.
    """
    # Clamp to non-negative (ReLU equivalent) before power
    clamped = np.maximum(patches, 0.0)
    powered = clamped ** p
    mean_pooled = powered.mean(axis=0)
    gem = mean_pooled ** (1.0 / p)

    # L2 normalize
    norm = np.linalg.norm(gem)
    if norm > 0:
        gem = gem / norm
    return gem


def fused_embedding(
    cls: NDArray[np.floating],
    patches: NDArray[np.floating],
    alpha: float = 0.5,
    gem_p: float = 3.0,
) -> NDArray[np.floating]:
    """Weighted fusion of CLS token and GeM-pooled patch tokens.

    Args:
        cls: (dim,) L2-normalized CLS token.
        patches: (N, dim) L2-normalized patch tokens.
        alpha: Weight for CLS token. (1 - alpha) weights the GeM pool.
        gem_p: GeM exponent.

    Returns:
        (dim,) L2-normalized fused vector.
    """
    gem = gem_pool(patches, p=gem_p)
    fused = alpha * cls + (1.0 - alpha) * gem

    # L2 normalize
    norm = np.linalg.norm(fused)
    if norm > 0:
        fused = fused / norm
    return fused
