# Fused Retrieval + Patch Re-ranking Pipeline

DINOv2 produces two complementary feature types: the **CLS token** (global image descriptor) and **patch tokens** (local per-region descriptors). This document describes the implemented two-stage pipeline that combines both for beverage SKU matching.

---

## Table of Contents

- [Why Patches Matter](#why-patches-matter)
- [How DINOv2 Produces CLS and Patch Tokens](#how-dinov2-produces-cls-and-patch-tokens)
- [Pipeline Architecture](#pipeline-architecture)
  - [Index Time](#index-time)
  - [Query Time](#query-time)
- [Configuration](#configuration)
- [Scoring Details](#scoring-details)
  - [Stage 1: Coarse Retrieval](#stage-1-coarse-retrieval)
  - [Stage 2: Patch Re-ranking](#stage-2-patch-re-ranking)
  - [Score Blending](#score-blending)
- [Storage Layout](#storage-layout)
- [Patch Similarity: Max-of-Mean](#patch-similarity-max-of-mean)
- [Top-K Selection for Coarse Retrieval](#top-k-selection-for-coarse-retrieval)
- [Per-SKU Aggregation: Max vs Mean](#per-sku-aggregation-max-vs-mean)
- [Key Numbers](#key-numbers)
- [Implementation Reference](#implementation-reference)

---

## Why Patches Matter

CLS tokens give you a single 768-dimensional vector that summarizes the entire image. This works well for coarse retrieval — distinguishing a Coke from a Sprite. But for fine-grained discrimination within the same brand, CLS falls short.

Consider Diet Coke vs Coke Zero:

```
CLS sees:  "Red can, white text, silver top, ~330ml"
           → Nearly identical embeddings (cosine similarity >0.95)

Patches see:
  Patch 47: "Diet" text, white on red
  Patch 48: "Coke" text, script font
  Patch 89: "330ml" text, small
  ...
           → Different text in specific regions → measurable patch-level divergence
```

Aggregating 1,369 patches into one vector (GeM/mean) dilutes the regional signal. The "Diet" text patch is 1 out of 1,369 — its contribution to a pooled descriptor is small. Patch-to-patch matching preserves these local differences.

### Benchmark Evidence

| Method | ΔR@1 vs CLS | Source |
|---|---|---|
| GeM-pooled patches | +5.13% | MarkoHaralovic, geo-image-retrieval (2025) |
| Max-pooled patches | +4.20% | Same |
| Mean-pooled patches | +3.13% | Same |
| CLS token | baseline | DINOv2 Table 9 |

GeM-pooled patches beat CLS by +5% R@1 zero-shot. But **patch-to-patch re-ranking** goes further — it enables region-level correspondence matching that no single-vector method can achieve.

---

## How DINOv2 Produces CLS and Patch Tokens

```
Input Image (518×518)
    ↓
Patch Embedding: Conv2d(3, D, kernel_size=14, stride=14)
    → 1,369 patch tokens, each representing a 14×14 pixel region
    ↓
Prepend CLS token (learnable parameter, initialized ~zero)
    → Sequence: [CLS, patch_1, patch_2, ..., patch_1369]
    ↓
Add positional embeddings
    ↓
12 Transformer blocks (multi-head self-attention + MLP)
    Every token attends to every other token at every layer.
    CLS has no spatial correspondence — it's a "virtual" summary token.
    ↓
Final LayerNorm (applied to all tokens equally)
    ↓
Output:
    CLS token      → (D,)       — single global vector
    Patch tokens   → (1369, D)  — one vector per 14×14 region
```

**Critical details:**

- **Neither output is L2-normalized.** The pipeline applies L2 normalization before computing cosine similarity.
- **Dual training objective** — DINO loss trains CLS for global representation; iBOT loss trains patches for local content. A DINOv2 author confirmed: *"at no point we expect the CLS and patch tokens to align."*
- **Register tokens** (e.g. `dinov2-with-registers-base`) absorb artifacts, keeping patches cleaner. This is the default model.
- **Resolution:** At 518px → 1,369 patches per image (37×37 grid).

---

## Pipeline Architecture

The implemented pipeline is a two-stage approach: **fused embedding for coarse retrieval**, then **patch-to-patch re-ranking** for fine-grained discrimination. Both stages share a single 518px forward pass through DINOv2.

### Index Time

```
Reference Image
    ↓
YOLOE Detection → Crop → (optional: mask background)
    ↓
DINOv2 Forward Pass (518×518)
    ↓
    ├── CLS token (D,) ──────────────┐
    │                                 │ Weighted fusion: α·CLS + (1-α)·GeM(patches)
    └── Patch tokens (1369, D) ──→ GeM pool ─┘
                                      ↓
                                  L2 normalize → Fused embedding → ChromaDB
                                      ↓
                                  Patch tokens → L2 normalize → .npy file on disk
```

Both the fused embedding and patch tokens are extracted from the **same forward pass** — no extra compute.

**Source:** `src/reference_processor.py` → `process_and_add()` / `build_from_directory()`

### Query Time

```
Detection Crop
    ↓
DINOv2 Forward Pass (518×518)
    ↓
    ├── Fused embedding → ChromaDB query → top-K coarse candidates
    │
    └── Patch tokens → load K .npy files → patch-to-patch similarity
                                      ↓
                        β × patch_score + (1−β) × norm_coarse
                                      ↓
                              Softmax → SKU distribution
```

**Source:** `api/services/recognition.py` → `recognize()` and `_score_with_reranking()`

---

## Configuration

All settings are in `api/config.py` (pydantic-settings, overridable via `.env`):

### Feature Extraction

| Setting | Default | Description |
|---|---|---|
| `USE_FUSED_FEATURES` | `True` | Use fused CLS+GeM embeddings (vs CLS-only) |
| `FUSE_ALPHA` | `0.5` | Weight for CLS token in fusion: `α·CLS + (1-α)·GeM` |
| `GEM_P` | `3.0` | GeM pooling exponent (higher = emphasize larger activations) |

### Patch Re-ranking

| Setting | Default | Description |
|---|---|---|
| `USE_RERANKING` | `True` | Enable stage 2 patch re-ranking |
| `RERANK_TOP_K` | `50` | How many coarse candidates to retrieve for re-ranking |
| `RERANK_BLEND_BETA` | `0.7` | Weight for patch score: `β × patch + (1−β) × coarse` |
| `PATCH_DIR` | `"data/patches"` | Directory for patch token `.npy` files |

**Feature type compatibility:** ChromaDB records the `feature_type` metadata (`"fused"` or `"cls"`) on each vector. Changing `USE_FUSED_FEATURES` after indexing raises a `RuntimeError` — rebuild the index to switch.

---

## Scoring Details

### Stage 1: Coarse Retrieval

1. Fused query embedding → ChromaDB cosine similarity search
2. Returns top-K candidates (controlled by `RERANK_TOP_K` when reranking is enabled, otherwise auto-computed)
3. Each candidate is a `VectorMatch` with `doc_id`, `sku_id`, `similarity`, and `rank`

**Source:** `src/indexer.py` → `search_batch()`

### Stage 2: Patch Re-ranking

1. Load `.npy` patch files for all K candidates via `PatchStore.load_batch()`
2. Compute bidirectional max-of-mean similarity between query and each candidate's patches
3. Group by SKU, take max patch score per SKU
4. Normalize coarse scores to [0, 1] (divide by max)
5. Blend: `blended = β × patch_score + (1−β) × norm_coarse`
6. Sort by blended score, return top-N SKUs
7. Apply softmax over blended scores to produce probability distribution

**Source:** `src/reranker.py` → `rerank()`

### Score Blending

The blend formula is:

```
blended_score = β × patch_score + (1 − β) × normalized_coarse_score
```

Where:
- **`β = 1`**: Pure patch re-ranking — ignores coarse retrieval order entirely
- **`β = 0`**: Pure coarse retrieval — equivalent to disabling re-ranking
- **`β = 0.7`** (default): 70% patch similarity, 30% coarse — strong emphasis on fine-grained matching

Coarse scores are normalized to [0, 1] before blending (best candidate = 1.0).

After blending, softmax with temperature 0.5 converts scores to a probability distribution over SKUs.

---

## Storage Layout

```
data/
├── patches/                              # Patch token numpy files
│   ├── t01__a1b2c3d4e5f6.npy            # {sku_id}__{media_id}.npy
│   ├── t01__f7e8d9c0b1a2.npy
│   ├── t02__3456789abcde.npy
│   └── ...                               # One file per reference image
│
├── references/                           # Original reference images
│   ├── t01/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── t02/
│       └── img1.jpg
│
chroma_data/                              # ChromaDB persistence
└── sku_embeddings/                       # Fused (or CLS) embeddings

sku_match.db                              # SQLite DB (SKU metadata, logs)
```

**Doc ID convention:** Both ChromaDB and patch files use `{sku_id}__{media_id}` as the document ID. This ensures consistent lookup across the two stores.

**Patch file contents:** `(1369, D)` float32 array of L2-normalized patch tokens, where D is the embedding dimension (768 for dinov2-base).

**Reinitialization:** `scripts/init_and_download.sh` wipes DB, Chroma, AND `data/patches/` to prevent stale doc_id mismatches after re-indexing.

---

## Patch Similarity: Max-of-Mean

The core similarity function is **bidirectional max-of-mean patch similarity**. For each patch in image A, find its best match in image B (max), then average across all query patches (mean). Repeat in both directions and average.

```
sim_matrix = query_patches @ db_patches.T    # (N_q, N_db) cosine similarities

sim_a→b = mean(max per query patch)           # each query patch finds its best match
sim_b→a = mean(max per db patch)              # each db patch finds its best match

score = (sim_a→b + sim_b→a) / 2
```

**Why max-of-mean (not mean-of-max or simple average):**

- **Max per query patch**: lets each region find its best correspondent, ignoring spatial misalignment
- **Mean across patches**: averages out noise from individual patch outliers
- **Bidirectional**: ensures both images are well-represented (prevents one image's unique patches from being ignored)

This is spatial-alignment agnostic — each 14×14 pixel region matches independently regardless of position.

**Source:** `src/reranker.py` → `max_of_mean_similarity()`

---

## Top-K Selection for Coarse Retrieval

`RERANK_TOP_K` controls how many candidates stage 1 returns for re-ranking. Too small → miss the correct SKU. Too large → wasted computation.

### Starting Point: K=50

For 200 SKUs × ~12 reference images (~2,400 total), K=50 retrieves ~4–5 SKUs' worth of candidates. This almost certainly includes 2+ images from the correct SKU.

### Factors That Push K Up or Down

| Factor | Direction | Reasoning |
|---|---|---|
| Many visually similar SKUs (5+ cola variants) | K ↑ (80–100) | Fused embedding scatters the correct SKU's images across a wider rank range |
| Small per-SKU reference set (2–3 images) | K ↑ | Fewer images cluster near the top |
| High angle variance in query images | K ↑ | Matching reference may not be the top-1 hit |
| SKUs are visually distinct | K ↓ (20–30) | Separates cleanly at top-5 |
| Tight latency budget | K ↓ | Each candidate costs ~0.1ms for patch computation |

### Latency Budget by K

| K | Patch files loaded | Disk I/O (SSD) | Patch computation | Total added latency |
|---|---|---|---|---|
| 20 | ~15 MB | ~1 ms | ~2 ms | ~3 ms |
| 50 | ~38 MB | ~2 ms | ~5 ms | ~7 ms |
| 100 | ~75 MB | ~4 ms | ~10 ms | ~14 ms |

At 518px resolution, each patch file is ~4 MB (1,369 × 768 × 4 bytes).

---

## Per-SKU Aggregation: Max vs Mean

With 4 reference angles per product (0°, 90°, 180°, 270°), a query image will be close to one angle and far from the other three. Using mean aggregation lets the three wrong-angle references drag down the score.

```
Query at ~15° (close to 0° reference):

  Candidate scores for SKU "diet-coke-330ml":
    0° ref:   patch_score = 0.87  ← close match
    90° ref:  patch_score = 0.62  ← wrong angle
    180° ref: patch_score = 0.58  ← wrong angle
    270° ref: patch_score = 0.64  ← wrong angle

  Mean: (0.87 + 0.62 + 0.58 + 0.64) / 4 = 0.678
  Max:  0.87
  Top-2 mean: (0.87 + 0.64) / 2 = 0.755

  → Max is the cleanest signal. The right-angle candidate dominates.
```

The implementation uses **max** across all retrieved candidates per SKU for both patch score and coarse score.

---

## Key Numbers

| Parameter | Value | Notes |
|---|---|---|
| Embedding dimension | 768 (dinov2-base) | Per-vector storage: 3 KB |
| Patches per image (518px) | 1,369 (37×37 grid) | Each 14×14 pixels |
| Patch file size (518px) | ~4 MB per image | float32, (1369, 768) |
| Total patch storage (2,400 imgs) | ~9 GB at 518px | On-disk, loaded on demand |
| Top-K coarse retrieval | 50 (default) | Tunable via `RERANK_TOP_K` |
| Per-query re-ranking latency | ~5 ms at K=50 | With SSD disk I/O |
| Forward pass resolution | 518×518 | Both CLS and patches extracted simultaneously |

---

## Implementation Reference

| Component | Source | Description |
|---|---|---|
| Feature extraction | `src/features.py` | `Features` dataclass, `gem_pool()`, `fused_embedding()` |
| Embedder | `src/embedder.py` | `extract_features_batch()` → CLS + patches in one pass; `features_to_embedding()` → fused or CLS |
| Patch storage | `src/patch_store.py` | `save()`, `load_batch()`, `delete()` — `.npy` files keyed by `{sku_id}__{media_id}` |
| Re-ranking | `src/reranker.py` | `rerank()` — max-of-mean similarity, per-SKU max aggregation, blended scoring |
| Coarse retrieval | `src/indexer.py` | `search_batch()` — ChromaDB cosine similarity, top-2-per-SKU scoring, softmax |
| Score assembly | `api/services/recognition.py` | `recognize()` → `_score_with_reranking()` or `score_matches()` fallback |
| Index building | `src/reference_processor.py` | `process_and_add()` / `build_from_directory()` — crop → features → ChromaDB + patches |
| API config | `api/config.py` | All tunable settings (fusion, re-ranking, paths) |
| Init script | `scripts/init_and_download.sh` | Wipes DB + Chroma + patches, rebuilds from reference images |
