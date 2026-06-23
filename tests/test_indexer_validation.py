"""Tests for SKUIndexer.validate_color_dim — startup color-dimension guard."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chromadb
import numpy as np
import pytest

from src.indexer import SKUIndexer

_counter = 0


def _make_indexer() -> SKUIndexer:
    """Fresh in-memory indexer (isolated EphemeralClient) per call."""
    global _counter
    _counter += 1
    client = chromadb.EphemeralClient()
    col = client.create_collection(name=f"test_sku_embeddings_{_counter}")
    return SKUIndexer(collection=col)


def _add_vector(indexer: SKUIndexer, doc_id: str, color_dim=None) -> None:
    meta = {"sku_id": "s1", "sku_name": "S1", "enabled": True}
    if color_dim is not None:
        meta["color_dim"] = color_dim
    indexer.collection.add(
        ids=[doc_id],
        embeddings=[np.zeros(8, dtype=np.float32).tolist()],
        metadatas=[meta],
    )


def test_validate_color_dim_empty_passes():
    idx = _make_indexer()
    idx.validate_color_dim(49)  # empty collection — nothing to check


def test_validate_color_dim_match_passes():
    idx = _make_indexer()
    _add_vector(idx, "s1__0000", color_dim=49)
    idx.validate_color_dim(49)  # exact match — OK


def test_validate_color_dim_mismatch_raises():
    idx = _make_indexer()
    _add_vector(idx, "s1__0000", color_dim=25)  # built with COLOR_BINS=8
    with pytest.raises(RuntimeError, match="COLOR_BINS mismatch"):
        idx.validate_color_dim(49)


def test_validate_color_dim_absent_raises():
    """Index built without color descriptors — must fail fast at startup."""
    idx = _make_indexer()
    _add_vector(idx, "s1__0000")  # no color_dim metadata
    with pytest.raises(RuntimeError, match="no color_dim metadata"):
        idx.validate_color_dim(49)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("All indexer validation tests passed!")
