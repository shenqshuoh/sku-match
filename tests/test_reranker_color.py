"""Tests for the optional color-histogram score-level blend in src/reranker.py.

These tests use lightweight duck-typed fakes (no real PatchStore/ColorStore/Chroma)
to verify:
  * weight validation (β + γ <= 1),
  * the color-absent path is byte-for-byte identical to the prior 2-way blend,
  * the three-way blend math,
  * graceful handling of missing color files.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.indexer import VectorMatch
from src.reranker import max_of_mean_similarity, rerank

# ── Fakes & helpers ──────────────────────────────────────────────────────────


class FakeStore:
    """Duck-typed store mirroring PatchStore/ColorStore.load_batch."""

    def __init__(self, data):
        self.data = data  # dict[str, NDArray]

    def load_batch(self, doc_ids):
        return {k: v for k, v in self.data.items() if k in doc_ids}


def vm(doc_id, sku_id, sim):
    """Build a VectorMatch (positional, matches __slots__ order)."""
    return VectorMatch(doc_id, sku_id, sku_id, sim, 0, "")


def l2norm(v):
    return np.asarray(v, dtype=np.float32) / (np.linalg.norm(v) + 1e-8)


def l2norm_rows(a):
    a = np.asarray(a, dtype=np.float32)
    norms = np.linalg.norm(a, axis=1, keepdims=True) + 1e-8
    return a / norms


# ── Tests ────────────────────────────────────────────────────────────────────


def test_validation_rejects_oversubscribed_weights():
    query_patches = l2norm_rows(np.random.default_rng(0).standard_normal((4, 8)))
    patch_store = FakeStore({"a1": l2norm_rows(np.random.default_rng(1).standard_normal((4, 8)))})
    top_vectors = [vm("a1", "A", 0.9)]
    try:
        rerank(
            query_patches,
            patch_store,
            top_vectors,
            blend_beta=0.7,
            color_gamma=0.5,
        )
    except ValueError:
        return
    raise AssertionError("Expected ValueError for blend_beta + color_gamma > 1.0")


def test_color_absent_matches_two_way():
    rng = np.random.default_rng(42)
    query_patches = l2norm_rows(rng.standard_normal((4, 8)))

    patch_store = FakeStore({
        "a1": l2norm_rows(rng.standard_normal((4, 8))),
        "a2": l2norm_rows(rng.standard_normal((4, 8))),
        "b1": l2norm_rows(rng.standard_normal((4, 8))),
    })
    top_vectors = [vm("a1", "A", 0.9), vm("a2", "A", 0.7), vm("b1", "B", 0.8)]

    # Plain (no color args at all)
    plain = rerank(query_patches, patch_store, top_vectors)

    # Color args present but color_gamma == 0.0 (color stage disabled)
    color_store = FakeStore({
        "a1": l2norm(rng.standard_normal(5)),
        "a2": l2norm(rng.standard_normal(5)),
        "b1": l2norm(rng.standard_normal(5)),
    })
    qcolor = l2norm(rng.standard_normal(5))
    gamma_zero = rerank(
        query_patches, patch_store, top_vectors,
        color_store=color_store, query_color_descriptor=qcolor, color_gamma=0.0,
    )

    # color_store=None explicitly
    none_store = rerank(
        query_patches, patch_store, top_vectors,
        color_store=None, query_color_descriptor=qcolor, color_gamma=0.4,
    )

    plain_scores = [r.blended_score for r in plain]
    gz_scores = [r.blended_score for r in gamma_zero]
    ns_scores = [r.blended_score for r in none_store]

    assert np.allclose(plain_scores, gz_scores)
    assert np.allclose(plain_scores, ns_scores)
    # Same ordering of sku_ids across all three
    assert [r.sku_id for r in plain] == [r.sku_id for r in gamma_zero] == [r.sku_id for r in none_store]
    # color_score is 0.0 everywhere when color stage disabled
    assert all(r.color_score == 0.0 for r in plain)
    assert all(r.color_score == 0.0 for r in gamma_zero)
    assert all(r.color_score == 0.0 for r in none_store)


def test_three_way_blend_math():
    # Deterministic, distinct fixtures so patch/color sims are non-trivial.
    qpatches = l2norm_rows(np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    ], dtype=np.float32))

    patches = {
        "a1": l2norm_rows(np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ], dtype=np.float32)),
        "a2": l2norm_rows(np.array([
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float32)),
        "b1": l2norm_rows(np.array([
            [0.7, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.7, 0.7, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.7, 0.7, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.7],
        ], dtype=np.float32)),
    }
    patch_store = FakeStore(patches)

    qcolor = l2norm(np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
    colors = {
        "a1": l2norm(np.array([0.9, 0.1, 0.0, 0.0, 0.0], dtype=np.float32)),  # ~0.994
        "a2": l2norm(np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)),  # ~0.0
        "b1": l2norm(np.array([0.3, 0.8, 0.0, 0.0, 0.0], dtype=np.float32)),  # ~0.351
    }
    color_store = FakeStore(colors)

    top_vectors = [vm("a1", "A", 0.9), vm("a2", "A", 0.7), vm("b1", "B", 0.8)]

    beta, gamma = 0.3, 0.4
    results = rerank(
        qpatches, patch_store, top_vectors,
        blend_beta=beta,
        color_store=color_store, query_color_descriptor=qcolor, color_gamma=gamma,
    )

    # Manually compute expected via the defined primitives.
    patch_a1 = max_of_mean_similarity(qpatches, patches["a1"])
    patch_a2 = max_of_mean_similarity(qpatches, patches["a2"])
    patch_b1 = max_of_mean_similarity(qpatches, patches["b1"])
    patch_A = max(patch_a1, patch_a2)
    patch_B = patch_b1

    color_a1 = float(qcolor @ colors["a1"])
    color_a2 = float(qcolor @ colors["a2"])
    color_b1 = float(qcolor @ colors["b1"])
    color_A = max(color_a1, color_a2)
    color_B = color_b1

    coarse_A = 0.9  # max(0.9, 0.7)
    coarse_B = 0.8
    max_coarse = max(coarse_A, coarse_B)
    coarse_norm_A = coarse_A / max_coarse
    coarse_norm_B = coarse_B / max_coarse

    expected_A = beta * patch_A + gamma * color_A + (1.0 - beta - gamma) * coarse_norm_A
    expected_B = beta * patch_B + gamma * color_B + (1.0 - beta - gamma) * coarse_norm_B

    by_sku = {r.sku_id: r.blended_score for r in results}
    assert np.allclose([by_sku["A"]], [expected_A], atol=1e-6)
    assert np.allclose([by_sku["B"]], [expected_B], atol=1e-6)

    # Sanity: aggregated max color scores per SKU are exposed too.
    color_by_sku = {r.sku_id: r.color_score for r in results}
    assert np.allclose([color_by_sku["A"]], [color_A], atol=1e-6)
    assert np.allclose([color_by_sku["B"]], [color_B], atol=1e-6)


def test_missing_color_file_defaults_zero():
    rng = np.random.default_rng(7)
    qpatches = l2norm_rows(rng.standard_normal((4, 8)))

    patches = {
        "a1": l2norm_rows(rng.standard_normal((4, 8))),
        "b1": l2norm_rows(rng.standard_normal((4, 8))),
    }
    patch_store = FakeStore(patches)

    qcolor = l2norm(np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
    # b1 has NO color file in the color store -> should default to 0.0, no crash.
    colors = {
        "a1": l2norm(np.array([0.9, 0.1, 0.0, 0.0, 0.0], dtype=np.float32)),
    }
    color_store = FakeStore(colors)

    top_vectors = [vm("a1", "A", 0.9), vm("b1", "B", 0.8)]

    beta, gamma = 0.3, 0.4
    results = rerank(
        qpatches, patch_store, top_vectors,
        blend_beta=beta,
        color_store=color_store, query_color_descriptor=qcolor, color_gamma=gamma,
    )
    by_sku = {r.sku_id: r for r in results}

    # b1 had no color file -> color_score must be exactly 0.0
    assert by_sku["B"].color_score == 0.0

    # Verify blended for B reflects color_score = 0.
    patch_b1 = max_of_mean_similarity(qpatches, patches["b1"])
    max_coarse = 0.9  # max(0.9, 0.8)
    coarse_norm_B = 0.8 / max_coarse
    expected_B = beta * patch_b1 + gamma * 0.0 + (1.0 - beta - gamma) * coarse_norm_B
    assert np.allclose([by_sku["B"].blended_score], [expected_B], atol=1e-6)


def test_color_absent_path_unchanged():
    """Focused regression: no-color rerank() must equal the current 2-way formula."""
    rng = np.random.default_rng(123)
    qpatches = l2norm_rows(rng.standard_normal((4, 8)))

    patches = {
        "a1": l2norm_rows(rng.standard_normal((4, 8))),
        "b1": l2norm_rows(rng.standard_normal((4, 8))),
        "b2": l2norm_rows(rng.standard_normal((4, 8))),
    }
    patch_store = FakeStore(patches)
    top_vectors = [vm("a1", "A", 0.6), vm("b1", "B", 0.9), vm("b2", "B", 0.3)]

    beta = 0.3
    results = rerank(qpatches, patch_store, top_vectors, blend_beta=beta)

    # Hand-compute the CURRENT 2-way formula.
    patch_a1 = max_of_mean_similarity(qpatches, patches["a1"])
    patch_b1 = max_of_mean_similarity(qpatches, patches["b1"])
    patch_b2 = max_of_mean_similarity(qpatches, patches["b2"])
    patch_A = patch_a1
    patch_B = max(patch_b1, patch_b2)

    coarse_A = 0.6
    coarse_B = 0.9  # max(0.9, 0.3)
    max_coarse = max(coarse_A, coarse_B)
    coarse_norm_A = coarse_A / max_coarse
    coarse_norm_B = coarse_B / max_coarse

    expected_A = beta * patch_A + (1.0 - beta) * coarse_norm_A
    expected_B = beta * patch_B + (1.0 - beta) * coarse_norm_B

    by_sku = {r.sku_id: r.blended_score for r in results}
    assert np.allclose([by_sku["A"]], [expected_A], atol=1e-6)
    assert np.allclose([by_sku["B"]], [expected_B], atol=1e-6)

    # Sort order must be descending by blended_score.
    scores = [r.blended_score for r in results]
    assert scores == sorted(scores, reverse=True)


if __name__ == "__main__":
    test_validation_rejects_oversubscribed_weights()
    test_color_absent_matches_two_way()
    test_three_way_blend_math()
    test_missing_color_file_defaults_zero()
    test_color_absent_path_unchanged()
    print("All reranker color tests passed!")
