"""Patch token persistence for DINOv2 patch-level re-ranking.

Stores per-image patch token arrays as .npy files on disk, using the same
``{sku_id}__{media_id}`` naming convention as ChromaDB document IDs.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class PatchStore:
    """Manages patch token .npy files on disk.

    Each file is named ``{doc_id}.npy`` where ``doc_id`` matches the
    ChromaDB document ID (``{sku_id}__{media_id}``).  The stored array
    has shape ``(N, dim)`` — typically ``(1369, 384)`` or ``(1369, 768)``
    at 518 px input resolution.
    """

    def __init__(self, patch_dir: str | Path) -> None:
        self._dir = Path(patch_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        logger.info("PatchStore initialized at %s", self._dir)

    @property
    def patch_dir(self) -> Path:
        return self._dir

    # ── Write operations ───────────────────────────────────────────

    def save(self, doc_id: str, patches: NDArray[np.floating]) -> Path:
        """Save patch tokens to disk.

        Args:
            doc_id: Document identifier (``{sku_id}__{media_id}``).
            patches: (N, dim) array of L2-normalized patch tokens.

        Returns:
            Path to the saved .npy file.
        """
        path = self._dir / f"{doc_id}.npy"
        np.save(path, patches.astype(np.float32))
        return path

    # ── Read operations ────────────────────────────────────────────

    def load(self, doc_id: str) -> NDArray[np.float32]:
        """Load patch tokens for a single document.

        Args:
            doc_id: Document identifier.

        Returns:
            (N, dim) float32 array.

        Raises:
            FileNotFoundError: If the patch file does not exist.
        """
        path = self._dir / f"{doc_id}.npy"
        return np.load(path)

    def load_batch(self, doc_ids: list[str]) -> dict[str, NDArray[np.float32]]:
        """Load patch tokens for multiple documents.

        Missing files are silently skipped (logged as warning).

        Returns:
            Dict mapping doc_id → (N, dim) array for successfully loaded documents.
        """
        result: dict[str, NDArray[np.float32]] = {}
        for doc_id in doc_ids:
            path = self._dir / f"{doc_id}.npy"
            if path.exists():
                result[doc_id] = np.load(path)
            else:
                logger.warning("Patch file not found: %s", path)
        return result

    # ── Delete operations ──────────────────────────────────────────

    def delete(self, doc_id: str) -> None:
        """Delete a single patch file."""
        path = self._dir / f"{doc_id}.npy"
        if path.exists():
            path.unlink()
            logger.debug("Deleted patch file: %s", path)

    def delete_sku(self, sku_id: str) -> int:
        """Delete all patch files for a given SKU.

        Uses prefix matching on ``{sku_id}__``.

        Returns:
            Number of files deleted.
        """
        prefix = f"{sku_id}__"
        deleted = 0
        for path in self._dir.glob(f"{prefix}*.npy"):
            path.unlink()
            deleted += 1
        if deleted:
            logger.info("Deleted %d patch files for SKU %s", deleted, sku_id)
        return deleted

    # ── Utility ────────────────────────────────────────────────────

    def count(self) -> int:
        """Return the number of patch files in the store."""
        return sum(1 for _ in self._dir.glob("*.npy"))
