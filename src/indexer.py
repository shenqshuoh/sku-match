import json
from pathlib import Path

import numpy as np

from src.types import SKUReference


class SKUIndexer:
    def __init__(self) -> None:
        self.references: list[SKUReference] = []
        self.embeddings: np.ndarray | None = None
        self._embeddings_norm: np.ndarray | None = None
        self.sku_ids: list[str] = []
        self._unique_skus: list[str] = []
        self._sku_mask: np.ndarray | None = None
        self.model_name: str = ""

    def build(self, references: list[SKUReference]) -> None:
        self.references = references
        self.embeddings = np.vstack([r.embedding for r in references])
        self.sku_ids = [r.sku_id for r in references]
        self._unique_skus = list(dict.fromkeys(self.sku_ids))
        self._embeddings_norm = None
        self._sku_mask = None
        self._sku_indices = None
        self._refs_per_sku = None
        self._max_refs_per_sku = None
        self._compute_sku_groups()

    def _get_normalized_embeddings(self) -> np.ndarray:
        if self._embeddings_norm is None:
            embeddings = self.embeddings
            if embeddings is None:
                raise RuntimeError("Index not built")
            self._embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return self._embeddings_norm

    def _get_sku_mask(self) -> np.ndarray:
        if self._sku_mask is None:
            self._sku_mask = np.array(
                [[sku_id == u for u in self._unique_skus] for sku_id in self.sku_ids]
            )
        return self._sku_mask

    def _compute_sku_groups(self) -> None:
        sku_to_idx = {sku: i for i, sku in enumerate(self._unique_skus)}
        self._sku_indices = [[] for _ in range(len(self._unique_skus))]
        for ref_idx, sku_id in enumerate(self.sku_ids):
            self._sku_indices[sku_to_idx[sku_id]].append(ref_idx)
        self._sku_indices = [np.array(idxs, dtype=np.int64) for idxs in self._sku_indices]
        self._refs_per_sku = np.array([len(idxs) for idxs in self._sku_indices], dtype=np.int64)
        self._max_refs_per_sku = int(self._refs_per_sku.max())

    def search(self, query_embedding: np.ndarray) -> tuple[str, float]:
        results = self.search_batch(query_embedding.reshape(1, -1))
        return results[0]

    def search_batch(self, query_embeddings: np.ndarray) -> list[tuple[str, float]]:
        if self.embeddings is None:
            raise RuntimeError("Index not built")

        embeddings_norm = self._get_normalized_embeddings()
        similarities = embeddings_norm @ query_embeddings.T

        batch_size = query_embeddings.shape[0]
        num_skus = len(self._unique_skus)
        max_refs = self._max_refs_per_sku

        padded_sims = np.full((num_skus, max_refs, batch_size), -np.inf, dtype=similarities.dtype)
        for s, idxs in enumerate(self._sku_indices):
            padded_sims[s, : len(idxs)] = similarities[idxs]

        top2 = np.partition(padded_sims, -2, axis=1)[:, -2:, :]
        top2_sums = top2.sum(axis=1)

        if (self._refs_per_sku == 1).any():
            max_sims = np.max(padded_sims, axis=1)
            single_mask = self._refs_per_sku[:, None] == 1
            top2_sums = np.where(single_mask, max_sims, top2_sums)

        best_indices = np.argmax(top2_sums, axis=0)
        best_scores = top2_sums[best_indices, np.arange(batch_size)]

        if np.any(best_scores == -np.inf):
            raise RuntimeError("No matches found")

        return [
            (self._unique_skus[int(idx)], float(score))
            for idx, score in zip(best_indices, best_scores)
        ]

    def save(self, directory: Path, model_name: str) -> None:
        if self.embeddings is None:
            raise RuntimeError("Index not built")

        directory.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name

        np.save(directory / f"embeddings_{model_name}.npy", self.embeddings)

        metadata = {
            "model_name": model_name,
            "sku_ids": self.sku_ids,
            "image_paths": [str(r.image_path) for r in self.references],
        }
        with open(directory / f"metadata_{model_name}.json", "w") as f:
            json.dump(metadata, f)

    def load(self, directory: Path, model_name: str) -> None:
        self.model_name = model_name
        self.embeddings = np.load(directory / f"embeddings_{model_name}.npy")

        with open(directory / f"metadata_{model_name}.json") as f:
            metadata = json.load(f)

        self.sku_ids = metadata["sku_ids"]
        self.references = [
            SKUReference(
                sku_id=sku_id,
                image_path=Path(path),
                embedding=self.embeddings[i],
            )
            for i, (sku_id, path) in enumerate(zip(self.sku_ids, metadata["image_paths"]))
        ]
        self._unique_skus = list(dict.fromkeys(self.sku_ids))
        self._embeddings_norm = None
        self._sku_mask = None
        self._sku_indices = None
        self._refs_per_sku = None
        self._max_refs_per_sku = None
        self._compute_sku_groups()
