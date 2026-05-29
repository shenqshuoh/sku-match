"""Chroma-backed SKU index for DINOv2 embedding similarity search.

Replaces the previous file-based numpy index (.npy + .json) with a Chroma
PersistentClient using HNSW cosine distance. Supports incremental add/delete
of reference embeddings without full index rebuilds.
"""

import logging
from pathlib import Path

import chromadb
import numpy as np

from src.types import Detection, SKUMatch, SKUReference
from src.utils import embedding_to_list

EPSILON = 1e-8

logger = logging.getLogger(__name__)


class VectorMatch:
    """A single vector-level match result from similarity search."""

    __slots__ = ("sku_id", "sku_name", "similarity", "rank", "media_url")

    def __init__(
        self, sku_id: str, sku_name: str, similarity: float, rank: int, media_url: str
    ) -> None:
        self.sku_id = sku_id
        self.sku_name = sku_name
        self.similarity = similarity
        self.rank = rank
        self.media_url = media_url

COLLECTION_NAME = "sku_embeddings"

# Chroma cosine distance: 0 = identical, 2 = opposite. Similarity = 2.0 - distance.
COSINE_DISTANCE_TO_SIMILARITY = 2.0


def _softmax(scores: dict[str, float], temperature: float = 0.5) -> dict[str, float]:
    """Apply softmax to produce a probability distribution over SKUs."""
    keys = list(scores.keys())
    values = np.array([scores[k] for k in keys])
    scaled = values / temperature
    exp_values = np.exp(scaled - np.max(scaled))  # numerical stability
    probs = exp_values / exp_values.sum()
    return {k: float(p) for k, p in zip(keys, probs)}


def concentration_score(distribution: dict[str, float], top_k: int = 10) -> float:
    """Fraction of top-K probability mass held by the #1 SKU.

    Returns 0.0 when fewer than 2 SKUs compete or top-1 is zero.
    """
    top_scores = sorted(distribution.values(), reverse=True)[:top_k]
    if len(top_scores) < 2 or top_scores[0] == 0:
        return 0.0
    return top_scores[0] / sum(top_scores)


def score_matches(
    detections: list[Detection],
    search_results: list[tuple[dict[str, float], dict[str, list[int]], list[VectorMatch]]],
    indexer: "SKUIndexer",
    concentration_topk: int = 10,
) -> list[SKUMatch]:
    """Score detection crops against the SKU index.

    For each detection, takes the top-1 SKU from the softmax distribution
    and computes concentration score and rank positions.

    Args:
        detections: Parsed detection results (from parse_detections).
        search_results: Output from SKUIndexer.search_batch().
        indexer: SKU index for name lookups.
        concentration_topk: K for concentration score computation.

    Returns:
        List of SKUMatch with scored results. No threshold filtering is applied;
        callers should filter by match_score themselves.
    """
    matches = []
    for det, (distribution, rank_info, top_vectors) in zip(detections, search_results):
        ranked = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
        sku_id, confidence = ranked[0]
        sku_name = indexer.get_sku_name(sku_id) or sku_id
        conc = concentration_score(distribution, top_k=concentration_topk)

        positions = rank_info.get(sku_id, [])
        top2_ranks = (
            (positions[0], positions[1]) if len(positions) >= 2
            else (positions[0], 0) if len(positions) == 1
            else (0, 0)
        )

        matches.append(SKUMatch(
            detection=det,
            sku_id=sku_id,
            sku_name=sku_name,
            match_score=confidence,
            match_concentration=conc,
            top2_ranks=top2_ranks,
            sku_distribution=distribution,
            top_vectors=top_vectors,
        ))

    return matches


class SKUIndexer:
    """SKU reference index backed by a Chroma vector store.

    Each reference image is stored as a Chroma document with:
      - id: "{sku_id}__{media_id}"
      - embedding: DINOv2 vector (768-dim for vitb14)
      - metadata: sku_id, sku_name, enabled, class_name, media_url
    """

    def __init__(
        self,
        collection: chromadb.Collection | None = None,
        persist_dir: str | Path | None = None,
        search_multiplier: float = 0.5,
        temperature: float = 0.5,
    ) -> None:
        self._collection = collection
        self._persist_dir = Path(persist_dir) if persist_dir else None
        self._search_multiplier = search_multiplier
        self._temperature = temperature
        self._enabled_sku_count: int = 0
        self._sku_name_cache: dict[str, str] = {}
        self._cache_valid = False

    @property
    def collection(self) -> chromadb.Collection:
        if self._collection is None:
            raise RuntimeError("Index not initialised — call init_collection() or pass a collection")
        return self._collection

    def init_collection(self, persist_dir: str | Path | None = None) -> None:
        dir_path = Path(persist_dir) if persist_dir else self._persist_dir or Path("chroma_data")
        dir_path.mkdir(parents=True, exist_ok=True)

        client = chromadb.PersistentClient(
            path=str(dir_path),
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        self._collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            configuration={"hnsw": {"space": "cosine"}},
        )
        self._persist_dir = dir_path
        self._refresh_cache()
        logger.info(
            "Chroma collection '%s' opened at %s (%d vectors)",
            COLLECTION_NAME, dir_path, self._collection.count(),
        )

    def build(self, references: list[SKUReference]) -> None:
        """Populate the Chroma collection from a list of SKUReference objects. Clears existing data first."""
        if not references:
            return

        existing = self.collection.get(include=[])
        if existing["ids"]:
            self.collection.delete(ids=existing["ids"])

        ids: list[str] = []
        embeddings: list[list[float]] = []
        metadatas: list[dict] = []

        for i, ref in enumerate(references):
            doc_id = f"{ref.sku_id}__{i:04d}"
            ids.append(doc_id)
            embeddings.append(embedding_to_list(ref.embedding))
            metadatas.append({
                "sku_id": ref.sku_id,
                "sku_name": ref.sku_name,
                "enabled": True,
                "image_path": str(ref.image_path),
            })

        self.collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
        self._refresh_cache()
        logger.info("Built index: %d vectors, %d SKUs", len(ids), len({r.sku_id for r in references}))

    def add_reference(
        self,
        sku_id: str,
        sku_name: str,
        media_id: str,
        embedding: np.ndarray,
        metadata: dict | None = None,
    ) -> None:
        doc_id = f"{sku_id}__{media_id}"
        meta = {"sku_id": sku_id, "sku_name": sku_name, "enabled": True}
        if metadata:
            meta.update(metadata)

        emb_list = embedding.tolist() if isinstance(embedding, np.ndarray) else list(embedding)
        if isinstance(emb_list[0], list):
            emb_list = emb_list[0]

        self.collection.upsert(
            ids=[doc_id],
            embeddings=[emb_list],
            metadatas=[meta],
        )
        self._cache_valid = False
        logger.debug("Upserted reference %s", doc_id)

    def delete_sku(self, sku_id: str) -> None:
        self.collection.delete(where={"sku_id": sku_id})
        self._cache_valid = False
        logger.info("Deleted SKU %s from index", sku_id)

    def delete_media(self, sku_id: str, media_id: str) -> None:
        doc_id = f"{sku_id}__{media_id}"
        self.collection.delete(ids=[doc_id])
        self._cache_valid = False
        logger.debug("Deleted reference %s", doc_id)

    def set_enabled(self, sku_id: str, enabled: bool) -> None:
        results = self.collection.get(where={"sku_id": sku_id}, include=["metadatas"])
        if not results["ids"]:
            logger.warning("SKU %s not found in index", sku_id)
            return

        updated_metas = [{**m, "enabled": enabled} for m in results["metadatas"]]
        self.collection.update(ids=results["ids"], metadatas=updated_metas)
        self._cache_valid = False
        logger.info("Set SKU %s enabled=%s", sku_id, enabled)

    def update_sku_name(self, sku_id: str, new_name: str) -> None:
        """Update the sku_name metadata for all vectors of a given SKU."""
        results = self.collection.get(
            where={"sku_id": sku_id}, include=["metadatas"]
        )
        if not results["ids"]:
            return
        updated = [{**m, "sku_name": new_name} for m in results["metadatas"]]
        self.collection.update(ids=results["ids"], metadatas=updated)
        self._cache_valid = False
        logger.info("Updated sku_name to '%s' for SKU %s (%d vectors)", new_name, sku_id, len(results["ids"]))

    def search(self, query_embedding: np.ndarray) -> tuple[dict[str, float], dict[str, list[int]]]:
        results = self.search_batch(query_embedding.reshape(1, -1))
        return results[0]

    def search_batch(
        self, query_embeddings: np.ndarray
    ) -> list[tuple[dict[str, float], dict[str, list[int]], list[VectorMatch]]]:
        """Match detection crops using top-2-per-SKU scoring with softmax normalization.

        Algorithm: for each query, fetch top-N results from Chroma (cosine distance),
        convert to similarity (2.0 - distance), group by sku_id, sum the top-2
        similarities per SKU, then apply softmax to produce a probability distribution
        over all enabled SKUs. Unranked SKUs receive a small epsilon score.

        Returns:
            List of (distribution, rank_positions, top_vectors) tuples where:
              - distribution: dict mapping sku_id → softmax probability (sums to 1.0)
              - rank_positions: dict mapping sku_id → list of 1-indexed rank positions
                of the SKU's top-2 vectors in the overall Chroma result list
              - top_vectors: list of VectorMatch for each raw vector returned by Chroma,
                sorted by similarity descending
        """
        if self.collection.count() == 0:
            raise RuntimeError("Index is empty — no reference embeddings")

        if not self._cache_valid:
            self._refresh_cache()

        all_enabled_skus = set(self._sku_name_cache.keys())
        n_results = self._get_n_results_size()

        query_list = [embedding_to_list(emb) for emb in query_embeddings]

        results = self.collection.query(
            query_embeddings=query_list,
            n_results=n_results,
            where={"enabled": True},
            include=["metadatas", "distances"],
        )

        matches: list[tuple[dict[str, float], dict[str, list[int]], list[VectorMatch]]] = []

        for crop_idx in range(len(query_list)):
            sku_scores: dict[str, list[float]] = {}
            sku_positions: dict[str, list[int]] = {}
            metas = results["metadatas"][crop_idx]
            dists = results["distances"][crop_idx]

            # Build raw vector-level match list
            top_vectors: list[VectorMatch] = []
            for rank, (meta, distance) in enumerate(zip(metas, dists)):
                similarity = COSINE_DISTANCE_TO_SIMILARITY - distance
                sku_id = meta["sku_id"]
                sku_name = meta.get("sku_name", sku_id)
                media_url = meta.get("media_url", "")
                top_vectors.append(VectorMatch(sku_id, sku_name, similarity, rank + 1, media_url))
                sku_scores.setdefault(sku_id, []).append(similarity)
                sku_positions.setdefault(sku_id, []).append(rank + 1)  # 1-indexed

            # Already sorted by Chroma by distance ascending = similarity descending

            # Track top-2 rank positions per SKU
            rank_info: dict[str, list[int]] = {}
            for sku_id in all_enabled_skus:
                if sku_id in sku_scores:
                    # Sort by similarity desc, take top-2 positions
                    scored = list(zip(sku_scores[sku_id], sku_positions[sku_id]))
                    scored.sort(key=lambda x: x[0], reverse=True)
                    rank_info[sku_id] = [pos for _, pos in scored[:2]]
                else:
                    rank_info[sku_id] = []

            # Compute raw scores: top-2 sum for ranked SKUs, epsilon for unranked
            raw_scores: dict[str, float] = {}
            for sku_id in all_enabled_skus:
                if sku_id in sku_scores:
                    top2 = sorted(sku_scores[sku_id], reverse=True)[:2]
                    raw_scores[sku_id] = top2[0] if len(top2) == 1 else sum(top2)
                else:
                    raw_scores[sku_id] = EPSILON

            # Softmax normalization
            probs = _softmax(raw_scores, self._temperature)
            matches.append((probs, rank_info, top_vectors))

        return matches

    def get_sku_name(self, sku_id: str) -> str | None:
        if not self._cache_valid:
            self._refresh_cache()
        return self._sku_name_cache.get(sku_id)

    def _get_n_results_size(self) -> int:
        if not self._cache_valid:
            self._refresh_cache()
        return max(int(self._enabled_sku_count * self._search_multiplier), 20)

    def _refresh_cache(self) -> None:
        if self._collection is None:
            return

        results = self.collection.get(where={"enabled": True}, include=["metadatas"])

        sku_names: dict[str, str] = {}
        for meta in results["metadatas"]:
            sku_id = meta["sku_id"]
            if sku_id not in sku_names:
                sku_names[sku_id] = meta.get("sku_name", sku_id)

        self._sku_name_cache = sku_names
        self._enabled_sku_count = len(sku_names)
        self._cache_valid = True
