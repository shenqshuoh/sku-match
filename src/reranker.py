"""Patch-to-patch re-ranking for fine-grained SKU discrimination.

Re-ranks coarse ChromaDB retrieval results using bidirectional max-of-mean
patch similarity, then blends coarse and patch scores into a final ranking.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from src.indexer import VectorMatch
from src.patch_store import PatchStore

logger = logging.getLogger(__name__)


@dataclass
class RerankCandidate:
    """A single vector-level candidate after re-ranking."""

    doc_id: str
    sku_id: str
    sku_name: str
    media_url: str
    coarse_score: float  # Raw ChromaDB similarity
    patch_score: float   # Max-of-mean patch similarity


@dataclass
class SKURerankResult:
    """Aggregated re-ranking result for one SKU."""

    sku_id: str
    sku_name: str
    blended_score: float  # β × patch_score + (1−β) × norm_coarse
    coarse_score: float   # Max coarse score across candidates
    patch_score: float    # Max patch score across candidates
    num_candidates: int
    candidates: list[RerankCandidate] = field(default_factory=list, repr=False)


def max_of_mean_similarity(
    q_patches: NDArray[np.floating],
    db_patches: NDArray[np.floating],
) -> float:
    """Bidirectional max-of-mean patch similarity.

    For each query patch, find its best match in the DB (max), then average
    across all query patches (mean).  Repeat in both directions and average.

    Spatial-alignment agnostic — each region matches independently.

    Args:
        q_patches:  (N_q, D) — query patch tokens, pre-L2-normalized.
        db_patches: (N_db, D) — database patch tokens, pre-L2-normalized.

    Returns:
        Similarity score in [-1, 1].
    """
    sim_matrix = q_patches.astype(np.float32) @ db_patches.astype(np.float32).T
    sim_a2b = sim_matrix.max(axis=1).mean()
    sim_b2a = sim_matrix.max(axis=0).mean()
    return float((sim_a2b + sim_b2a) / 2.0)


def rerank(
    query_patches: NDArray[np.floating],
    patch_store: PatchStore,
    top_vectors: list[VectorMatch],
    blend_beta: float = 0.3,
    top_n: int = 5,
) -> list[SKURerankResult]:
    """Re-rank coarse retrieval candidates using patch-to-patch matching.

    Pipeline:
      1. Load patch files for all candidates via ``patch_store``.
      2. Compute max-of-mean similarity for each candidate vs query.
      3. Group by SKU, take max patch_score per SKU.
      4. Normalize coarse scores to [0, 1].
      5. Blend: ``blended[sku] = β × patch_score + (1−β) × norm_coarse``.
      6. Sort by blended_score descending, return top-N.

    Args:
        query_patches: (N, D) patch tokens from the query detection crop.
        patch_store: Patch file store for loading candidate patches.
        top_vectors: Stage-1 ChromaDB results (must have doc_id populated).
        blend_beta: Weight for patch score. 0.3 = 30% patch, 70% coarse.
        top_n: Number of SKU-level results to return.

    Returns:
        List of SKURerankResult sorted by blended_score descending.
    """
    if not top_vectors:
        return []

    # 1. Load patch files for all candidates
    doc_ids = [v.doc_id for v in top_vectors]
    patches_map = patch_store.load_batch(doc_ids)

    # 2. Compute patch similarity for each candidate
    candidates: list[RerankCandidate] = []
    for v in top_vectors:
        db_patches = patches_map.get(v.doc_id)
        if db_patches is None:
            logger.debug("Skipping candidate %s — no patch file", v.doc_id)
            continue

        patch_score = max_of_mean_similarity(query_patches, db_patches)
        candidates.append(RerankCandidate(
            doc_id=v.doc_id,
            sku_id=v.sku_id,
            sku_name=v.sku_name,
            media_url=v.media_url,
            coarse_score=v.similarity,
            patch_score=patch_score,
        ))

    if not candidates:
        return []

    # 3. Group by SKU, take max scores per SKU
    sku_candidates: dict[str, list[RerankCandidate]] = defaultdict(list)
    for c in candidates:
        sku_candidates[c.sku_id].append(c)

    sku_results: list[SKURerankResult] = []
    for sku_id, cands in sku_candidates.items():
        best_coarse = max(c.coarse_score for c in cands)
        best_patch = max(c.patch_score for c in cands)
        sku_name = cands[0].sku_name
        sku_results.append(SKURerankResult(
            sku_id=sku_id,
            sku_name=sku_name,
            blended_score=0.0,  # computed below
            coarse_score=best_coarse,
            patch_score=best_patch,
            num_candidates=len(cands),
            candidates=cands,
        ))

    # 4. Normalize coarse scores to [0, 1]
    max_coarse = max(r.coarse_score for r in sku_results)
    if max_coarse > 0:
        for r in sku_results:
            r.coarse_score = r.coarse_score / max_coarse

    # 5. Blend scores
    for r in sku_results:
        r.blended_score = blend_beta * r.patch_score + (1.0 - blend_beta) * r.coarse_score

    # 6. Sort by blended_score descending
    sku_results.sort(key=lambda r: r.blended_score, reverse=True)

    return sku_results[:top_n]
