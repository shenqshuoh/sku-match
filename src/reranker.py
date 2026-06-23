"""Patch-to-patch re-ranking for fine-grained SKU discrimination.

Re-ranks coarse ChromaDB retrieval results using bidirectional max-of-mean
patch similarity, then blends coarse and patch scores into a final ranking.

An optional color-histogram score-level blend can be enabled by passing a
``color_store``, ``query_color_descriptor``, and a positive ``color_gamma``
to :func:`rerank`. When any of these is absent (or ``color_gamma <= 0``) the
color stage is a no-op and the existing two-way (patch + coarse) blend is
used byte-for-byte unchanged.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from src.indexer import VectorMatch
from src.patch_store import PatchStore

if TYPE_CHECKING:
    from src.color_store import ColorStore

logger = logging.getLogger(__name__)


@dataclass
class RerankCandidate:
    """A single vector-level candidate after re-ranking."""

    doc_id: str
    sku_id: str
    sku_name: str
    media_url: str
    coarse_score: float       # Raw ChromaDB similarity
    patch_score: float        # Max-of-mean patch similarity
    color_score: float = 0.0  # Cosine sim of color descriptors (0 if unavailable)


@dataclass
class SKURerankResult:
    """Aggregated re-ranking result for one SKU."""

    sku_id: str
    sku_name: str
    blended_score: float  # β·patch + γ·color + (1−β−γ)·norm_coarse (γ=0 → 2-way)
    coarse_score: float   # Max coarse score across candidates
    patch_score: float    # Max patch score across candidates
    num_candidates: int
    color_score: float = 0.0  # Max color similarity across candidates (0 if color unused)
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
    color_store: "ColorStore | None" = None,
    query_color_descriptor: NDArray[np.floating] | None = None,
    color_gamma: float = 0.0,
) -> list[SKURerankResult]:
    """Re-rank coarse retrieval candidates using patch-to-patch matching.

    Pipeline:
      1. Load patch files for all candidates via ``patch_store``.
      2. Compute max-of-mean similarity for each candidate vs query.
      3. Group by SKU, take max patch_score per SKU.
      4. Normalize coarse scores to [0, 1].
      5. Blend: ``blended[sku] = β × patch_score + (1−β) × norm_coarse``.
      6. Sort by blended_score descending, return top-N.

    Optional color stage: when ``color_store``, ``query_color_descriptor`` and a
    positive ``color_gamma`` are all provided, a color cosine similarity is
    computed per candidate and the blend becomes a three-way simplex blend:

        ``blended[sku] = β·patch + γ·color + (1−β−γ)·norm_coarse``

    Constraint: ``blend_beta + color_gamma`` must be ``<= 1.0`` (validated at
    entry; raises ``ValueError`` otherwise). Color is a complete no-op when
    ``color_store``/``query_color_descriptor`` is ``None`` or ``color_gamma <= 0`` —
    in that case the existing two-way blend is used unchanged.

    Args:
        query_patches: (N, D) patch tokens from the query detection crop.
        patch_store: Patch file store for loading candidate patches.
        top_vectors: Stage-1 ChromaDB results (must have doc_id populated).
        blend_beta: Weight for patch score. 0.3 = 30% patch, 70% coarse.
        top_n: Number of SKU-level results to return.
        color_store: Optional color-descriptor store (duck-typed, only
            ``load_batch`` is used). ``None`` disables the color stage.
        query_color_descriptor: (D,) L2-normalized color descriptor for the
            query crop. ``None`` disables the color stage.
        color_gamma: Weight for the color score. ``<= 0`` disables the color
            stage.

    Returns:
        List of SKURerankResult sorted by blended_score descending.
    """
    if not top_vectors:
        return []

    if blend_beta + color_gamma > 1.0 + 1e-9:
        raise ValueError(
            f"blend_beta + color_gamma must be <= 1.0 (got beta={blend_beta}, gamma={color_gamma})"
        )

    use_color = (
        color_store is not None
        and query_color_descriptor is not None
        and color_gamma > 0.0
    )

    # 1. Load patch files for all candidates
    doc_ids = [v.doc_id for v in top_vectors]
    patches_map = patch_store.load_batch(doc_ids)
    if use_color and color_store is not None:
        color_map = color_store.load_batch(doc_ids)
    else:
        color_map = {}

    # 2. Compute patch similarity for each candidate
    candidates: list[RerankCandidate] = []
    for v in top_vectors:
        db_patches = patches_map.get(v.doc_id)
        if db_patches is None:
            logger.debug("Skipping candidate %s — no patch file", v.doc_id)
            continue

        patch_score = max_of_mean_similarity(query_patches, db_patches)

        if use_color and query_color_descriptor is not None:
            ref_color = color_map.get(v.doc_id)
            if ref_color is not None:
                color_score = float(
                    query_color_descriptor.astype(np.float32)
                    @ ref_color.astype(np.float32)
                )
            else:
                color_score = 0.0
        else:
            color_score = 0.0

        candidates.append(RerankCandidate(
            doc_id=v.doc_id,
            sku_id=v.sku_id,
            sku_name=v.sku_name,
            media_url=v.media_url,
            coarse_score=v.similarity,
            patch_score=patch_score,
            color_score=color_score,
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
        best_color = max(c.color_score for c in cands)
        sku_name = cands[0].sku_name
        sku_results.append(SKURerankResult(
            sku_id=sku_id,
            sku_name=sku_name,
            blended_score=0.0,  # computed below
            coarse_score=best_coarse,
            patch_score=best_patch,
            color_score=best_color,
            num_candidates=len(cands),
            candidates=cands,
        ))

    # 4. Normalize coarse scores to [0, 1]
    max_coarse = max(r.coarse_score for r in sku_results)
    if max_coarse > 0:
        for r in sku_results:
            r.coarse_score = r.coarse_score / max_coarse

    # 5. Blend scores
    if use_color:
        for r in sku_results:
            r.blended_score = (
                blend_beta * r.patch_score
                + color_gamma * r.color_score
                + (1.0 - blend_beta - color_gamma) * r.coarse_score
            )
    else:
        for r in sku_results:
            r.blended_score = blend_beta * r.patch_score + (1.0 - blend_beta) * r.coarse_score

    # 6. Sort by blended_score descending
    sku_results.sort(key=lambda r: r.blended_score, reverse=True)

    return sku_results[:top_n]
