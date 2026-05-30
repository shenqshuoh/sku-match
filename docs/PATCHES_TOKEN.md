# CLS + Patch Token Pipeline Approaches

DINOv2 produces two complementary feature types: the **CLS token** (global image descriptor) and **patch tokens** (local per-region descriptors). This document covers how to combine them into production pipelines for beverage SKU matching.

---

## Table of Contents

- [Why Patches Matter](#why-patches-matter)
- [How DINOv2 Produces CLS and Patch Tokens](#how-dinov2-produces-cls-and-patch-tokens)
- [Three Pipeline Approaches](#three-pipeline-approaches)
  - [Approach A: Fused Single Embedding](#approach-a-fused-single-embedding)
  - [Approach B: Two-Stage Pipeline](#approach-b-two-stage-pipeline)
  - [Approach C: Fused Retrieval + Patch Re-ranking](#approach-c-fused-retrieval--patch-re-ranking)
- [Top-K Selection for Coarse Retrieval](#top-k-selection-for-coarse-retrieval)
- [Head-to-Head Comparison](#head-to-head-comparison)
- [Decision Tree](#decision-tree)
- [Pre-Processing: Rotation Correction](#pre-processing-rotation-correction)
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

Aggregating 256 patches into one vector (GeM/mean) dilutes the regional signal. The "Diet" text patch is 1 out of 256 — its contribution to a pooled descriptor is small. Patch-to-patch matching preserves these local differences.

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
Input Image (224×224)
    ↓
Patch Embedding: Conv2d(3, D, kernel_size=14, stride=14)
    → 256 patch tokens, each representing a 14×14 pixel region
    ↓
Prepend CLS token (learnable parameter, initialized ~zero)
    → Sequence: [CLS, patch_1, patch_2, ..., patch_256]
    ↓
Add positional embeddings
    ↓
12–24 Transformer blocks (multi-head self-attention + MLP)
    Every token attends to every other token at every layer.
    CLS has no spatial correspondence — it's a "virtual" summary token.
    ↓
Final LayerNorm (applied to all tokens equally)
    ↓
Output:
    x_norm_clstoken    → (B, D)       — single global vector
    x_norm_patchtokens → (B, 256, D)  — one vector per 14×14 region
```

**Critical details:**

- **Neither output is L2-normalized.** You must apply `F.normalize(x, dim=-1, p=2)` yourself before computing cosine similarity.
- **Dual training objective** — DINO loss trains CLS for global representation; iBOT loss trains patches for local content. A DINOv2 author confirmed: *"at no point we expect the CLS and patch tokens to align."*
- **Register tokens** (`dinov2_vitl14_reg`) absorb artifacts, keeping patches cleaner. Recommended.
- **Resolution matters:** At 224px → 256 patches per image. At 518px → 1,369 patches per image.

---

## Three Pipeline Approaches

### Approach A: Fused Single Embedding

**One vector per image. Drop-in replacement for current CLS-only approach.**

```
Index time:
  Image → DINOv2 → CLS + patches → fuse → L2 normalize → single vector → ChromaDB

Query time:
  Same fusion → single ChromaDB query → final ranking → done
```

**Fusion methods:**

| Method | Formula | Dimension | Notes |
|---|---|---|---|
| **Weighted average** | `α·CLS + (1-α)·GeM(patches)` | 768 | Tunable balance, same dim as current CLS |
| **Simple average** | `(CLS + GeM(patches)) / 2` | 768 | Recommended by dino.txt (CVPR 2025) |
| **Concatenation** | `cat([CLS, GeM(patches)])` | 1536 | Preserves both signals independently, doubles storage |

**When to use:** SKUs are visually distinct enough that a single vector suffices. No near-duplicate brand variants.

**Pros:**
- Minimal implementation change — just swap the embedding function
- No additional storage (same 768-dim vectors in ChromaDB)
- No re-ranking stage, ~1ms per query
- +2–5% R@1 improvement over CLS alone

**Cons:**
- Cannot do region-level matching — fine-grained SKU variants still confused
- Averaging dilutes the strongest regional signals

```python
def extract_fused_embedding(model, img_tensor, alpha=0.5, gem_p=3.0):
    """
    Fused embedding: weighted combination of CLS + GeM(patches).
    Returns a single 768-dim vector. Drop-in for ChromaDB.
    """
    with torch.no_grad():
        features = model.forward_features(img_tensor.cuda())

    cls = features["x_norm_clstoken"]           # (1, 768)
    patches = features["x_norm_patchtokens"]     # (1, 256, 768)

    # L2 normalize individually
    cls_n = F.normalize(cls, dim=-1, p=2)
    patches_n = F.normalize(patches, dim=-1, p=2)

    # GeM pooling of patches
    gem = F.normalize(
        (F.relu(patches_n) ** gem_p).mean(dim=1) ** (1.0 / gem_p),
        dim=-1, p=2
    )  # (1, 768)

    # Weighted fusion
    fused = F.normalize(alpha * cls_n + (1 - alpha) * gem, dim=-1, p=2)

    return fused.cpu().numpy().flatten()
```

### Approach B: Two-Stage Pipeline

**CLS for coarse retrieval → patches for fine-grained re-ranking.**

```
Index time:
  Image → DINOv2 → CLS → L2 normalize → ChromaDB
                → patches → L2 normalize → numpy file on disk

Query time:
  Stage 1: CLS → ChromaDB top-K candidates
  Stage 2: Load K patch files → patch-to-patch re-rank → final ranking
```

**When to use:** You have near-duplicate SKU variants (e.g., Diet Coke vs Coke Zero, limited editions) that CLS alone cannot separate.

**Pros:**
- Best fine-grained discrimination — catches label text differences via region matching
- CLS stage is unchanged from current approach
- Re-ranking only runs on K candidates (not full database)
- Spatial-agnostic: max-of-mean similarity works across viewpoint changes

**Cons:**
- Additional storage: ~790 KB per image for patch files (~1.8 GB for 2,400 images at 224px)
- Additional latency: ~5ms per query for patch comparisons
- More code complexity

#### Stage 1: CLS Coarse Retrieval (Unchanged)

```python
def extract_cls_embedding(model, img_tensor):
    """Current approach — single CLS vector for ChromaDB."""
    with torch.no_grad():
        features = model.forward_features(img_tensor.cuda())

    cls = F.normalize(features["x_norm_clstoken"], dim=-1, p=2)
    return cls.cpu().numpy().flatten()


def extract_and_store_patches(model, img_tensor, image_id, patch_dir):
    """
    Extract patch tokens and save to disk.
    Returns CLS embedding for ChromaDB.
    """
    with torch.no_grad():
        features = model.forward_features(img_tensor.cuda())

    cls = F.normalize(features["x_norm_clstoken"], dim=-1, p=2).cpu().numpy().flatten()
    patches = F.normalize(features["x_norm_patchtokens"], dim=-1, p=2)

    # Save patches as numpy file
    patch_path = os.path.join(patch_dir, f"{image_id}.npy")
    np.save(patch_path, patches.cpu().numpy().squeeze(0))  # (256, 768)

    return cls
```

#### Stage 2: Patch Re-ranking

The core similarity function is **max-of-mean bidirectional similarity**. For each patch in image A, find its best match in image B (max), then average across all query patches (mean). Repeat in both directions and average.

```python
def max_of_mean_similarity(q_patches, db_patches):
    """
    Bidirectional max-of-mean patch similarity.

    For each query patch: find best match in DB (max).
    Average across all query patches (mean).
    Repeat A→B and B→A, then average both directions.

    This is spatial-alignment agnostic — works across viewpoint changes.
    Each 14×14 pixel region matches independently regardless of position.

    Args:
        q_patches:  (N_q, D) — query patch tokens, pre-L2-normalized
        db_patches: (N_db, D) — database patch tokens, pre-L2-normalized

    Returns:
        float: similarity score in [-1, 1]
    """
    # Pairwise cosine similarity matrix
    sim_matrix = q_patches @ db_patches.T  # (N_q, N_db)

    # A→B: for each query patch, best match in DB
    sim_a2b = sim_matrix.max(dim=1).values.mean().item()  # scalar

    # B→A: for each DB patch, best match in query
    sim_b2a = sim_matrix.max(dim=0).values.mean().item()  # scalar

    return (sim_a2b + sim_b2a) / 2.0
```

**Why max-of-mean (not mean-of-max or simple average):**
- **Max per query patch**: lets each region find its best correspondent, ignoring spatial misalignment
- **Mean across patches**: averages out noise from individual patch outliers
- **Bidirectional**: ensures both images are well-represented (prevents one image's unique patches from being ignored)

#### Full Query Pipeline

```python
def query_with_reranking(
    query_img,
    model,
    collection,      # ChromaDB collection
    patch_dir,       # directory with .npy patch files
    top_k_coarse=50, # Stage 1: how many candidates to retrieve
    top_n_final=5,   # Stage 2: how many results to return
):
    """
    Full 2-stage query pipeline.
    """
    # === Stage 1: CLS Coarse Retrieval ===
    with torch.no_grad():
        features = model.forward_features(query_img.cuda())

    query_cls = F.normalize(features["x_norm_clstoken"], dim=-1, p=2)
    query_patches = F.normalize(features["x_norm_patchtokens"], dim=-1, p=2)
    query_patches_np = query_patches.cpu().numpy().squeeze(0)  # (256, 768)

    results = collection.query(
        query_embeddings=[query_cls.cpu().numpy().flatten().tolist()],
        n_results=top_k_coarse,
        include=["metadatas", "distances", "documents"],
    )

    # === Stage 2: Patch Re-ranking ===
    candidates = []
    for i, metadata in enumerate(results["metadatas"][0]):
        image_id = metadata["image_id"]
        sku = metadata["sku"]
        cls_score = 1 - results["distances"][0][i]  # convert distance to similarity

        # Load candidate's patches
        patch_path = os.path.join(patch_dir, f"{image_id}.npy")
        db_patches = np.load(patch_path)  # (256, 768)

        # Compute patch similarity
        q_t = torch.from_numpy(query_patches_np)
        db_t = torch.from_numpy(db_patches)
        patch_score = max_of_mean_similarity(q_t, db_t)

        candidates.append({
            "image_id": image_id,
            "sku": sku,
            "cls_score": cls_score,
            "patch_score": patch_score,
        })

    # === Aggregate to SKU level ===
    # Group by SKU, take top-2 candidates per SKU
    from collections import defaultdict
    sku_candidates = defaultdict(list)
    for c in candidates:
        sku_candidates[c["sku"]].append(c)

    sku_scores = []
    for sku, cands in sku_candidates.items():
        # Sort by patch score, take top-2 per SKU
        cands.sort(key=lambda x: x["patch_score"], reverse=True)
        top2 = cands[:2]

        # Use MAX aggregation (not mean) for patch scores
        # Wrong-angle references drag down mean; max lets the right-angle candidate dominate
        best_patch = max(c["patch_score"] for c in top2)

        sku_scores.append({
            "sku": sku,
            "patch_score": best_patch,
            "cls_score": max(c["cls_score"] for c in top2),
            "num_candidates": len(cands),
        })

    # Sort by patch score (re-ranking result)
    sku_scores.sort(key=lambda x: x["patch_score"], reverse=True)

    return sku_scores[:top_n_final]
```

#### Per-SKU Aggregation: Max vs Mean

With 4 reference angles per product (0°, 90°, 180°, 270°), a query image will be close to one angle and far from the other three. Using mean aggregation across all retrieved candidates from the same SKU lets the three wrong-angle references drag down the score.

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
  Top-2 max:  0.87

  → Max is the cleanest signal. The right-angle candidate dominates.
```

**Recommendation: Use max across all retrieved candidates per SKU.** The top-2 strategy is an option if you want to also consider the adjacent angle (±90°), but max over all retrieved candidates is simpler and works equally well at K=50.

### Approach C: Fused Retrieval + Patch Re-ranking

**Best of both worlds: better coarse retrieval + fine-grained re-ranking.**

```
Index time:
  Image → DINOv2 → CLS + GeM(patches) → fuse → ChromaDB   (better coarse retrieval)
                → patches → numpy file                       (for re-ranking)

Query time:
  Stage 1: Fused embedding → ChromaDB top-K   (better candidate selection)
  Stage 2: Load K patch files → patch re-rank  (same as Approach B)
```

This combines A's stronger single-vector retrieval with B's region-level matching. The fused embedding produces better top-K candidates than CLS alone, reducing the risk that the correct SKU's images are missed in stage 1.

**When to use:** You need maximum accuracy and have the storage budget for patch files.

```python
def extract_fused_and_patches(model, img_tensor, image_id, collection, patch_dir,
                               alpha=0.5, gem_p=3.0):
    """
    Index-time extraction for Approach C.
    Stores fused embedding in ChromaDB + patches on disk.
    """
    with torch.no_grad():
        features = model.forward_features(img_tensor.cuda())

    cls = features["x_norm_clstoken"]           # (1, 768)
    patches = features["x_norm_patchtokens"]     # (1, 256, 768)

    cls_n = F.normalize(cls, dim=-1, p=2)
    patches_n = F.normalize(patches, dim=-1, p=2)

    # Fused embedding
    gem = F.normalize(
        (F.relu(patches_n) ** gem_p).mean(dim=1) ** (1.0 / gem_p),
        dim=-1, p=2
    )
    fused = F.normalize(alpha * cls_n + (1 - alpha) * gem, dim=-1, p=2)

    # Store fused in ChromaDB
    collection.add(
        ids=[image_id],
        embeddings=[fused.cpu().numpy().flatten().tolist()],
        metadatas=[{"image_id": image_id, "sku": "..."}],
    )

    # Store patches on disk
    np.save(
        os.path.join(patch_dir, f"{image_id}.npy"),
        patches_n.cpu().numpy().squeeze(0),  # (256, 768), pre-normalized
    )
```

---

## Top-K Selection for Coarse Retrieval

The `top_k_coarse` parameter controls how many candidates stage 1 returns for re-ranking. Too small → miss the correct SKU. Too large → wasted computation.

### Starting Point: K=50

For 200 SKUs × ~12 reference images (~2,400 total), K=50 retrieves ~4–5 SKUs' worth of candidates. This almost certainly includes 2+ images from the correct SKU.

### Factors That Push K Up or Down

| Factor | Direction | Reasoning |
|---|---|---|
| Many visually similar SKUs (5+ cola variants) | K ↑ (80–100) | CLS scatters the correct SKU's images across a wider rank range |
| Small per-SKU reference set (2–3 images) | K ↑ | Fewer images cluster near the top |
| High angle variance in query images | K ↑ | Matching reference may not be the top-1 CLS hit |
| SKUs are visually distinct | K ↓ (20–30) | CLS separates cleanly at top-5 |
| Tight latency budget | K ↓ | Each candidate costs ~0.1ms for patch computation |
| Untested CLS embedding quality | K ↑ | Be conservative until you have data |

### Tuning K Empirically

```python
def evaluate_k_sweep(test_queries, ground_truth_skus, collection, patch_dir,
                     k_range=range(10, 110, 10)):
    """
    For each K, measure: does the correct SKU have ≥2 candidates
    in the top-K retrieval results?
    """
    results = {}

    for k in k_range:
        hits = 0
        total = len(test_queries)

        for query_tensor, true_sku in zip(test_queries, ground_truth_skus):
            # Stage 1 retrieval
            query_cls = extract_cls_embedding(model, query_tensor)
            chroma_results = collection.query(
                query_embeddings=[query_cls.tolist()],
                n_results=k,
                include=["metadatas"],
            )

            correct_sku_count = sum(
                1 for m in chroma_results["metadatas"][0]
                if m["sku"] == true_sku
            )

            if correct_sku_count >= 2:
                hits += 1

        recall = hits / total
        results[k] = recall
        print(f"K={k:3d}: recall={recall:.3f} ({hits}/{total})")

    return results
```

**Expected output for this use case:**

```
K= 10: recall=0.82   ← too low for production
K= 20: recall=0.91   ← risky if wrong SKU is costly
K= 30: recall=0.95
K= 40: recall=0.97
K= 50: recall=0.99   ← sweet spot
K= 80: recall=1.00
K=100: recall=1.00   ← unnecessary cost
```

Pick the smallest K where recall ≥ 0.99.

### Latency Budget by K

| K | Patch files loaded | Disk I/O (SSD) | Patch computation | Total added latency |
|---|---|---|---|---|
| 20 | ~15 MB | ~1 ms | ~2 ms | ~3 ms |
| 40 | ~30 MB | ~2 ms | ~4 ms | ~6 ms |
| 50 | ~38 MB | ~2 ms | ~5 ms | ~7 ms |
| 80 | ~60 MB | ~3 ms | ~8 ms | ~11 ms |
| 100 | ~75 MB | ~4 ms | ~10 ms | ~14 ms |

If patches are loaded into memory at startup (recommended — only ~1.8 GB total), disk I/O drops to zero.

### Diagnosing K from Recall Sweep

| Recall pattern | Diagnosis | Action |
|---|---|---|
| 0.99 at K=30 | CLS is strong, SKUs well-separated | K=30–40 is safe |
| 0.99 at K=50 | Normal case, some similar SKUs | K=50 |
| 0.99 at K=80+ | CLS struggles with similar SKU pairs | Consider Approach C or fine-tuning |

---

## Head-to-Head Comparison

| | **A: Fused** | **B: 2-Stage** | **C: Fused + Patches** |
|---|---|---|---|
| **Coarse retrieval quality** | Better than CLS alone (+2–5% R@1 from GeM) | CLS only | Best of both |
| **Fine-grained discrimination** | Same ceiling as CLS — no region-level matching | **Much better** — catches label text differences via patch correspondence | Best |
| **Diet Coke vs Coke Zero** | ⚠️ Likely confuses — too similar globally | ✅ Catches "Diet" vs "Zero" in different patches | ✅ Best chance |
| **Per-query latency** | ~1 ms (single ChromaDB query) | ~6 ms (ChromaDB + K patch comparisons) | ~7 ms |
| **Storage per image** | 3 KB (one 768-d vector) | 3 KB + 790 KB (patches) | 3 KB + 790 KB |
| **Total storage (2,400 imgs)** | ~7 MB | ~1.8 GB | ~1.8 GB |
| **ChromaDB compatibility** | ✅ Drop-in for current CLS | ✅ Current CLS unchanged | ✅ Replace CLS with fused |
| **Implementation complexity** | Minimal — change extraction only | Moderate — add re-ranking stage | Moderate |

---

## Decision Tree

```
Current: CLS only → measure accuracy on labeled test set
         │
         ▼
   accuracy ≥ 95%? ──── YES ──→ Done. Ship CLS-only.
         │
         NO
         │
         ▼
   Switch to Approach A (fused CLS + GeM patches)
         │
         ▼
   accuracy ≥ 95%? ──── YES ──→ Done. Fused is your system.
         │                      Storage: ~7 MB. Latency: ~1 ms.
         NO
         │
         ▼
   Add patch re-ranking → Approach B or C
         │                (C if you already switched to fused; B if staying with CLS)
         ▼
   accuracy ≥ 95%? ──── YES ──→ Done. 2-stage is your system.
         │                      Storage: ~1.8 GB. Latency: ~6–7 ms.
         NO
         │
         ▼
   Investigate supplemental signals:
     - OCR on detected text (cheapest high-impact addition)
     - Fine-tuned per-SKU classifier (may outperform similarity search at 200 SKUs)
     - Background replacement instead of black masking
     See IMAGE_SIMILARITY_SEARCH_REPORT.md §8 for alternatives.
```

Each step is incremental. You don't throw away previous work — you add to it. At each step you have a measurable accuracy number to decide whether to continue or stop.

---

## Pre-Processing: Rotation Correction

Slightly tilted products in axis-aligned bounding boxes include extra background area. If the tilt is >10°, this can affect patch token alignment since patches are spatial.

### Quick Test: Does Rotation Matter?

```python
def test_rotation_impact(model, image_tensor, angles=[0, 5, 10, 15, 20, 30]):
    """
    Check how much rotation affects DINOv2 similarity.
    If similarity stays >0.95 at your worst-case tilt, skip correction.
    """
    cls_0 = extract_cls_embedding(model, image_tensor)

    for angle in angles:
        rotated = TF.rotate(image_tensor, angle)
        cls_r = extract_cls_embedding(model, rotated)
        sim = F.cosine_similarity(
            torch.from_numpy(cls_0).unsqueeze(0),
            torch.from_numpy(cls_r).unsqueeze(0),
        ).item()
        print(f"  {angle:2d}° rotation → similarity = {sim:.4f}")
```

**Expected results:**

| Rotation | CLS Similarity | Impact |
|---|---|---|
| 0° | 1.000 | Baseline |
| 5° | ~0.98 | Negligible |
| 10° | ~0.96 | Small |
| 15° | ~0.93 | Noticeable |
| 20° | ~0.88 | Significant |
| 30° | ~0.80 | Problematic |

**Break-even: ~10–15°.** Below that, correction is unnecessary. Above that, it helps.

### Post-Crop Straightening (No Retraining)

Since backgrounds are already masked to black, the mask itself provides the orientation signal:

```python
def straighten_crop(crop_masked):
    """
    Detect product orientation from the mask and rotate to align
    with crop edges. Background is already black (0,0,0).
    """
    gray = cv2.cvtColor(crop_masked, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return crop_masked

    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)  # ((cx, cy), (w, h), angle)

    (cx, cy), (w, h), angle = rect

    # Choose rotation that makes the longer side vertical (bottles/cans)
    rotation = angle if w < h else angle + 90

    # Skip negligible rotations
    if abs(rotation) < 3:
        return crop_masked

    # Rotate with expanded canvas to avoid cropping the product
    M = cv2.getRotationMatrix2D((cx, cy), rotation, 1.0)
    cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M[0, 2] += (new_w - crop_masked.shape[1]) / 2
    M[1, 2] += (new_h - crop_masked.shape[0]) / 2

    rotated = cv2.warpAffine(
        crop_masked, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),  # fill new areas with black
    )

    # Re-crop to tight bounding box of the non-black region
    gray_r = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    _, binary_r = cv2.threshold(gray_r, 10, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(binary_r)
    x, y, bw, bh = cv2.boundingRect(coords)

    # 5% padding to avoid cutting edges
    pad = int(max(bw, bh) * 0.05)
    x, y = max(0, x - pad), max(0, y - pad)
    bw = min(rotated.shape[1] - x, bw + 2 * pad)
    bh = min(rotated.shape[0] - y, bh + 2 * pad)

    return rotated[y:y+bh, x:x+bw]
```

---

## Implementation Reference

### Storage Layout

```
data/
├── chroma_db/                    # ChromaDB persistence directory
│   └── {collection_name}/       # CLS or fused embeddings
│       ├── id.bin
│       ├── index_metadata.bin
│       └── ...
│
├── patches/                      # Patch token numpy files
│   ├── sku001_angle0_img1.npy   # (256, 768) float32 = ~790 KB
│   ├── sku001_angle0_img2.npy
│   ├── sku001_angle90_img1.npy
│   └── ...                       # ~2,400 files = ~1.8 GB at 224px
│
└── reference_images/             # Original images (existing)
```

### Model Loading

```python
import torch
import torch.nn.functional as F
import numpy as np
import os
import cv2

# Use register tokens for cleaner patches
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
model.eval()
model.cuda()

D = 768  # embedding dimension for ViT-B/14 (use 1024 for ViT-L/14)
```

### Batch Indexing (Approach C)

```python
def index_reference_images(image_dir, collection, patch_dir, model,
                           alpha=0.5, gem_p=3.0):
    """Build the index from all reference images."""
    os.makedirs(patch_dir, exist_ok=True)

    for sku_dir in sorted(os.listdir(image_dir)):
        sku_path = os.path.join(image_dir, sku_dir)
        if not os.path.isdir(sku_path):
            continue

        for img_file in sorted(os.listdir(sku_path)):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            image_id = f"{sku_dir}_{os.path.splitext(img_file)[0]}"
            img = load_and_preprocess(os.path.join(sku_path, img_file))

            with torch.no_grad():
                features = model.forward_features(img.cuda())

            cls = features["x_norm_clstoken"]
            patches = features["x_norm_patchtokens"]

            cls_n = F.normalize(cls, dim=-1, p=2)
            patches_n = F.normalize(patches, dim=-1, p=2)

            # Fused embedding for ChromaDB
            gem = F.normalize(
                (F.relu(patches_n) ** gem_p).mean(dim=1) ** (1.0 / gem_p),
                dim=-1, p=2
            )
            fused = F.normalize(alpha * cls_n + (1 - alpha) * gem, dim=-1, p=2)

            # Add to ChromaDB
            collection.add(
                ids=[image_id],
                embeddings=[fused.cpu().numpy().flatten().tolist()],
                metadatas=[{"image_id": image_id, "sku": sku_dir}],
            )

            # Save patches
            np.save(
                os.path.join(patch_dir, f"{image_id}.npy"),
                patches_n.cpu().numpy().squeeze(0),
            )

    print(f"Indexed {collection.count()} images")
```

### Key Numbers

| Parameter | Value | Notes |
|---|---|---|
| Embedding dimension | 768 (ViT-B/14) or 1024 (ViT-L/14) | Per-vector storage: 3 KB or 4 KB |
| Patches per image (224px) | 256 (16×16 grid) | Each 14×14 pixels |
| Patches per image (518px) | 1,369 (37×37 grid) | Higher resolution matching |
| Patch file size (224px) | ~790 KB per image | float32, (256, 768) |
| Total patch storage (2,400 imgs) | ~1.8 GB at 224px, ~9 GB at 518px | Memory-mappable |
| Top-K coarse retrieval | Start at 50 | Tune via recall sweep |
| Per-query re-ranking latency | ~5 ms at K=50 | With patches in memory |
