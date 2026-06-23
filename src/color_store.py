"""Color descriptor persistence for score-level SKU re-ranking.

Stores per-image color descriptors as .npy files on disk, using the same
``{sku_id}__{media_id}`` naming convention as ChromaDB document IDs. Each
descriptor is a 1-D float32 array (typically 49-dim at n_bins=16).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class ColorStore:
    """Manages color descriptor .npy files on disk.

    Each file is named ``{doc_id}.npy`` where ``doc_id`` matches the
    ChromaDB document ID (``{sku_id}__{media_id}``).  The stored array
    has shape ``(D,)`` — typically ``(49,)`` at ``n_bins=16``.
    """

    def __init__(self, color_dir: str | Path) -> None:
        self._dir = Path(color_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        logger.info("ColorStore initialized at %s", self._dir)

    @property
    def color_dir(self) -> Path:
        return self._dir

    # ── Write operations ───────────────────────────────────────────

    def save(self, doc_id: str, descriptor: NDArray[np.floating]) -> Path:
        """Save a color descriptor to disk.

        Args:
            doc_id: Document identifier (``{sku_id}__{media_id}``).
            descriptor: (D,) array of the L2-normalized color descriptor.

        Returns:
            Path to the saved .npy file.
        """
        path = self._dir / f"{doc_id}.npy"
        np.save(path, descriptor.astype(np.float32))
        return path

    # ── Read operations ────────────────────────────────────────────

    def load(self, doc_id: str) -> NDArray[np.float32]:
        """Load a color descriptor for a single document.

        Args:
            doc_id: Document identifier.

        Returns:
            (D,) float32 array.

        Raises:
            FileNotFoundError: If the color file does not exist.
        """
        path = self._dir / f"{doc_id}.npy"
        return np.load(path)

    def load_batch(self, doc_ids: list[str]) -> dict[str, NDArray[np.float32]]:
        """Load color descriptors for multiple documents.

        Missing files are silently skipped (logged as warning).

        Returns:
            Dict mapping doc_id → (D,) array for successfully loaded documents.
        """
        result: dict[str, NDArray[np.float32]] = {}
        for doc_id in doc_ids:
            path = self._dir / f"{doc_id}.npy"
            if path.exists():
                result[doc_id] = np.load(path)
            else:
                logger.warning("Color file not found: %s", path)
        return result

    # ── Delete operations ──────────────────────────────────────────

    def delete(self, doc_id: str) -> None:
        """Delete a single color file."""
        path = self._dir / f"{doc_id}.npy"
        if path.exists():
            path.unlink()
            logger.debug("Deleted color file: %s", path)

    def delete_sku(self, sku_id: str) -> int:
        """Delete all color files for a given SKU.

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
            logger.info("Deleted %d color files for SKU %s", deleted, sku_id)
        return deleted

    # ── Utility ────────────────────────────────────────────────────

    def count(self) -> int:
        """Return the number of color files in the store."""
        return sum(1 for _ in self._dir.glob("*.npy"))
