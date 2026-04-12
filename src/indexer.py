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

    def search(self, query_embedding: np.ndarray) -> tuple[str, float]:
        results = self.search_batch(query_embedding.reshape(1, -1))
        return results[0]

    def search_batch(self, query_embeddings: np.ndarray) -> list[tuple[str, float]]:
        if self.embeddings is None:
            raise RuntimeError("Index not built")

        embeddings_norm = self._get_normalized_embeddings()
        similarities = embeddings_norm @ query_embeddings.T

        sku_mask = self._get_sku_mask()
        batch_size = query_embeddings.shape[0]

        results = []
        for i in range(batch_size):
            sims = similarities[:, i]
            sim_matrix = sims[:, np.newaxis] * sku_mask
            sim_matrix[~sku_mask] = -np.inf

            top2_sums = np.partition(sim_matrix, -2, axis=0)[-2:].sum(axis=0)
            top2_sums = np.where((sku_mask.sum(axis=0) == 1), sim_matrix.max(axis=0), top2_sums)

            best_idx = int(np.argmax(top2_sums))
            best_sku = self._unique_skus[best_idx]
            best_score = float(top2_sums[best_idx])

            if best_score == -np.inf:
                raise RuntimeError("No matches found")

            results.append((best_sku, best_score))

        return results

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
