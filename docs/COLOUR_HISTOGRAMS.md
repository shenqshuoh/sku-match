# Color Histograms for SKU Retrieval

DINOv2 produces color-invariant embeddings — a green Sprite can and a red Coke can receive nearly identical similarity scores because both are "cylindrical objects with text and a silver top." Color histograms add the missing color signal at score level with negligible cost.

---

## Table of Contents

- [The Problem: DINOv2 Is Color-Blind](#the-problem-dinov2-is-color-blind)
- [Why Score-Level Fusion for Color](#why-score-level-fusion-for-color)
- [Color Histogram Extraction](#color-histogram-extraction)
- [Parameter Conventions](#parameter-conventions)
- [Full Pipeline: CLS + Patches + Color](#full-pipeline-cls--patches--color)
- [Weight Tuning](#weight-tuning)
- [Storage and Latency](#storage-and-latency)
- [References](#references)

---

## The Problem: DINOv2 Is Color-Blind

DINOv2 is trained with aggressive color augmentation designed to **force the model to ignore color**:

| Augmentation | Parameter | Probability |
|---|---|---|
| ColorJitter brightness | ±40% | 80% |
| ColorJitter contrast | ±40% | 80% |
| ColorJitter saturation | ±20% | 80% |
| ColorJitter hue | ±36° | 80% |
| RandomGrayscale | — | 20% |

During training, the same red Coke can appears as green, gray, pink, or blue across different crops. The DINO loss forces all color variants to produce the **same embedding**. The model is literally penalized for treating color as meaningful.

### Quantitative Evidence

From Oh-A-DINO (Wagner et al., 2025, arXiv:2503.09867):

| Attribute | DINOv2 Retrieval Accuracy |
|---|---|
| Shape | **95.4%** |
| Size | **96.1%** |
| **Color** | **40.8%** |

Adding color as a required attribute for retrieval causes precision to **collapse from 72.8% → 13.0%**. This directly explains why our green Sprite and red Coke get very high similarity — DINOv2 sees both as structurally identical cans.

### What DINOv2 Sees

```
Sprite (green can):                Coke (red can):
  "Cylindrical 330ml can"            "Cylindrical 330ml can"
  "Text in center band"              "Text in center band"
  "Logo above text"                  "Logo above text"
  "Silver pull-tab top"              "Silver pull-tab top"

  cosine_similarity = 0.92           ← DINOv2: "basically the same thing"
```

Color — the **single most discriminative feature** between these products — is absent from the embedding.

---

## Why Score-Level Fusion for Color

We have three features in fundamentally different spaces:

| Feature | Dimensions | Space | Similarity range |
|---|---|---|---|
| CLS token | 768 | Learned DINOv2 | 0.70–0.95 (narrow, clustered high) |
| GeM(patches) | 768 | Learned DINOv2 | 0.65–0.95 (narrow, clustered high) |
| Color histogram | 49 | Hand-crafted HSV | 0.05–0.95 (wide, discriminative) |

CLS and patches share the same learned space — they were trained together, have comparable similarity distributions, and fuse naturally via weighted averaging (the `alpha` parameter).

Color histograms live in a completely different space. Naive concatenation into a single vector distorts the nearest-neighbor geometry because:

1. **Dimension imbalance:** 768 + 768 = 1,536 DINOv2 dims vs 49 color dims. Even with equal per-dimension contribution, DINOv2 dominates.
2. **Different similarity distributions:** DINOv2 similarities cluster high (0.7–0.95 for structurally similar cans). Color similarities are wider (0.1 for different colors, 0.9 for same). Mixing them into one vector muddies the ranking.
3. **No weight control without re-indexing:** Changing the color weight means rebuilding the entire ChromaDB index.

**Score-level fusion** avoids all three problems: each feature computes its own similarity independently, then we combine the calibrated scores with a tunable weight.

---

## Color Histogram Extraction

### HSV Histogram Descriptor

```python
import cv2
import numpy as np

def extract_color_descriptor(image_bgr, n_bins=16):
    """
    Compact color descriptor from HSV histograms.

    Uses the raw YOLO crop (not masked). The product's dominant color
    naturally dominates the histogram — background contributes noise
    but is outweighed by the product's area and saturation.

    Args:
        image_bgr: raw BGR crop from YOLO detection (numpy array)
        n_bins: histogram bins per channel (16 = good balance)

    Returns:
        (49,) L2-normalized vector:
          [hue_hist(16), saturation_hist(16), value_hist(16), dominant_hue(1)]
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # 1D histograms per channel
    h_hist = cv2.calcHist([hsv], [0], None, [n_bins], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [n_bins], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [n_bins], [0, 256]).flatten()

    # L2 normalize each histogram independently
    h_hist = h_hist / (np.linalg.norm(h_hist) + 1e-8)
    s_hist = s_hist / (np.linalg.norm(s_hist) + 1e-8)
    v_hist = v_hist / (np.linalg.norm(v_hist) + 1e-8)

    # Dominant hue: circular mean weighted by saturation
    # This captures the single most prominent color direction
    h_rad = hsv[:, :, 0].astype(np.float64) * (np.pi / 90.0)  # OpenCV 0-180 → radians
    s_weight = hsv[:, :, 1].astype(np.float64) / 255.0
    dominant_hue = np.arctan2(
        np.sum(np.sin(h_rad) * s_weight),
        np.sum(np.cos(h_rad) * s_weight)
    ) % (2 * np.pi)
    dominant_hue_norm = dominant_hue / (2 * np.pi)

    descriptor = np.concatenate([h_hist, s_hist, v_hist, [dominant_hue_norm]])
    return descriptor / (np.linalg.norm(descriptor) + 1e-8)
```

### Design Rationale

| Design choice | Why |
|---|---|
| HSV (not RGB) | Hue separates color from brightness. A dark red and bright red have the same hue but different RGB values. |
| Separate histograms per channel | Allows cosine similarity to independently match hue distribution, saturation level, and brightness. |
| 16 bins per channel | 16 hue bins = ~11° per bin. Enough to distinguish red (0-15°) from orange (15-26°) from yellow (26-45°). |
| Dominant hue scalar | Provides a single-anchor signal that's robust to histogram noise. Especially useful for solid-color products (red Coke, green Sprite). |
| L2 normalize each histogram | Prevents a single high-count bin from dominating the similarity score. Each histogram channel contributes equally. |
| Use raw (unmasked) crop | The product occupies most of the crop. Its color dominates the histogram. Background noise is small and consistent across images of the same scene type. |

---

## Parameter Conventions

The pipeline uses three parameters, one per fusion point:

```
alpha    — DINOv2 internal: CLS vs patches          (existing, vector-level)
beta     — Re-ranking: DINOv2 fused vs patch score   (existing, score-level)
gamma    — Color: existing pipeline vs color          (new, score-level)
```

### alpha — DINOv2 Fusion Weight

Fuses CLS token and GeM-pooled patches into a single DINOv2 embedding at index time.

```
dino_fused = normalize(alpha * CLS + (1 - alpha) * GeM(patches))
```

| Value | Effect |
|---|---|
| 1.0 | CLS only (original approach) |
| 0.5 | Equal CLS + patches (recommended) |
| 0.0 | Patches only (GeM) |

### beta — Re-Ranking Weight

Combines the DINOv2 fused similarity with the patch re-ranking score at query time.

```
rerank_score = beta * dino_fused_sim + (1 - beta) * patch_sim
```

| Value | Effect |
|---|---|
| 1.0 | DINOv2 only (no re-ranking) |
| 0.5 | Equal DINOv2 + patches |
| 0.0 | Patch score only |

### gamma — Color Fusion Weight

Combines the existing pipeline score with the color histogram similarity.

```
final_score = gamma * color_sim + (1 - gamma) * rerank_score
```

| Value | Effect |
|---|---|
| 1.0 | Color only (extreme — ignores shape/structure) |
| 0.5 | Equal color + existing pipeline |
| 0.4 | Color-heavy (recommended for beverage SKU matching) |
| 0.3 | Color-moderate |
| 0.0 | No color (existing pipeline unchanged) |

### Full Score Composition

```
final_score = gamma * color_sim
            + (1 - gamma) * [beta * dino_fused_sim + (1 - beta) * patch_sim]

where:  dino_fused_sim = cosine_sim(query_dino_fused, ref_dino_fused)
                          and dino_fused = normalize(alpha * CLS + (1-alpha) * GeM)
        patch_sim      = max_of_mean_bidirectional(query_patches, ref_patches)
        color_sim      = cosine_sim(query_color_hist, ref_color_hist)
```

---

## Full Pipeline: CLS + Patches + Color

### Architecture

```
INDEX TIME
─────────
Image ──┬──→ DINOv2 ──→ CLS ──────────────┐
        │         └──→ patches ──→ GeM ──→│──→ fuse(α) → ChromaDB (768-dim)
        │                                  │
        │         └──→ patches ──────────────→ numpy files (256×768)
        │
        └──→ HSV color histogram ──────────────→ numpy files (49-dim)

QUERY TIME
──────────
Query ──┬──→ DINOv2 fused ──→ ChromaDB top-K₁ ──────────────┐
        │                                                      │
        └──→ Color histogram ──→ brute-force top-K₂ ─────────┤
                                                                │
                                          Merge (union) ───────┤
                                                                │
                              Score: γ·color + (1-γ)·[β·dino + (1-β)·patch]
                                                                │
                                                          Final ranking
```

### Color Retrieval: Why Brute-Force

At 2,400 images × 49 dimensions, brute-force cosine similarity takes ~0.1ms. No index needed — just load all color vectors into a matrix and multiply.

```python
# Preload at startup (470 KB — fits in L2 cache)
color_matrix = np.stack([
    np.load(os.path.join(color_dir, f"{iid}.npy"))
    for iid in all_image_ids
])  # (2400, 49)

# At query time
query_color = extract_color_descriptor(raw_crop)         # (49,)
color_sims = color_matrix @ query_color                   # (2400,) — all similarities at once
```

### Complete Index-Time Code

```python
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os


def extract_all_features(model, img_tensor, raw_crop_bgr,
                         alpha=0.5, gem_p=3.0):
    """
    Extract all three features from one image.

    Args:
        model: DINOv2 model (with registers recommended)
        img_tensor: preprocessed tensor for DINOv2 (normalized, 224px)
        raw_crop_bgr: raw BGR crop from YOLO (for color extraction)
        alpha: CLS vs patches fusion weight
        gem_p: GeM pooling power

    Returns:
        dino_fused: (768,) L2-normalized
        color_desc: (49,) L2-normalized
        patches:    (256, 768) L2-normalized
    """
    with torch.no_grad():
        features = model.forward_features(img_tensor.cuda())

    cls = features["x_norm_clstoken"]           # (1, 768)
    patches = features["x_norm_patchtokens"]     # (1, 256, 768)

    cls_n = F.normalize(cls, dim=-1, p=2)
    patches_n = F.normalize(patches, dim=-1, p=2)

    # Fuse CLS + patches (same learned space → vector-level fusion)
    gem = F.normalize(
        (F.relu(patches_n) ** gem_p).mean(dim=1) ** (1.0 / gem_p),
        dim=-1, p=2
    )
    dino_fused = F.normalize(
        alpha * cls_n + (1 - alpha) * gem, dim=-1, p=2
    )

    # Color histogram (raw BGR crop)
    color_desc = extract_color_descriptor(raw_crop_bgr)

    return (
        dino_fused.cpu().numpy().flatten(),   # ChromaDB
        color_desc,                            # numpy
        patches_n.cpu().numpy().squeeze(0),    # numpy
    )


def build_index(reference_images, model, collection, patch_dir, color_dir,
                alpha=0.5, gem_p=3.0):
    """
    Build all three indexes.

    Args:
        reference_images: list of (img_tensor, raw_crop_bgr, sku, image_id)
    """
    os.makedirs(patch_dir, exist_ok=True)
    os.makedirs(color_dir, exist_ok=True)

    image_ids = []

    for img_tensor, raw_crop, sku, image_id in reference_images:
        dino_fused, color_desc, patches = extract_all_features(
            model, img_tensor, raw_crop, alpha=alpha, gem_p=gem_p
        )

        # DINOv2 fused → ChromaDB
        collection.add(
            ids=[image_id],
            embeddings=[dino_fused.tolist()],
            metadatas=[{"sku": sku, "image_id": image_id}],
        )

        # Patches → numpy
        np.save(os.path.join(patch_dir, f"{image_id}.npy"), patches)

        # Color → numpy (tiny)
        np.save(os.path.join(color_dir, f"{image_id}.npy"), color_desc)

        image_ids.append(image_id)

    print(f"Indexed {collection.count()} images")
    return image_ids


# Prerequisite: extract_color_descriptor() defined above
```

### Complete Query-Time Code

```python
def max_of_mean_similarity(q_patches, db_patches):
    """Bidirectional max-of-mean patch similarity."""
    sim_matrix = q_patches @ db_patches.T      # (N_q, N_db)
    sim_a2b = sim_matrix.max(dim=1).values.mean().item()
    sim_b2a = sim_matrix.max(dim=0).values.mean().item()
    return (sim_a2b + sim_b2a) / 2.0


def query(
    query_img_tensor,
    query_raw_crop_bgr,
    model,
    collection,
    patch_dir,
    color_matrix,      # (N, 49) preloaded, L2-normalized per row
    all_image_ids,     # list[str], aligned with color_matrix rows
    all_skus,          # list[str], aligned with color_matrix rows
    alpha=0.5,
    beta=0.3,
    gamma=0.4,
    top_k_dino=50,
    top_k_color=50,
    top_n_final=5,
    gem_p=3.0,
):
    """
    Full query pipeline: DINOv2 + color → merge → patch re-rank.

    Args:
        alpha: CLS vs patches weight in DINOv2 fusion
        beta:  DINOv2 vs patches weight in re-ranking
               rerank_score = beta * dino_sim + (1 - beta) * patch_sim
        gamma: Color vs pipeline weight in final score
               final_score = gamma * color_sim + (1 - gamma) * rerank_score
    """
    # === Extract query features ===
    dino_fused, color_desc, query_patches = extract_all_features(
        model, query_img_tensor, query_raw_crop_bgr,
        alpha=alpha, gem_p=gem_p
    )
    query_patches_t = torch.from_numpy(query_patches)

    # === Stage 1a: DINOv2 retrieval (ChromaDB) ===
    dino_results = collection.query(
        query_embeddings=[dino_fused.tolist()],
        n_results=top_k_dino,
        include=["metadatas", "distances"],
    )

    dino_map = {}  # image_id → {sku, dino_sim}
    for i, meta in enumerate(dino_results["metadatas"][0]):
        dino_map[meta["image_id"]] = {
            "sku": meta["sku"],
            "dino_sim": 1.0 - dino_results["distances"][0][i],
        }

    # === Stage 1b: Color retrieval (brute-force, ~0.1ms) ===
    color_sims = color_matrix @ color_desc          # (N,)
    top_color_indices = np.argsort(color_sims)[::-1][:top_k_color]
    color_map = {}  # image_id → color_sim
    for idx in top_color_indices:
        color_map[all_image_ids[idx]] = float(color_sims[idx])

    # Also store all color sims for candidates only in dino results
    color_sims_full = dict(zip(all_image_ids, color_sims.tolist()))

    # === Merge candidates (union) ===
    merged_ids = set(dino_map.keys()) | set(color_map.keys())

    candidates = []
    for cid in merged_ids:
        d_info = dino_map.get(cid, None)
        d_sim = d_info["dino_sim"] if d_info else 0.0
        sku = d_info["sku"] if d_info else all_skus[all_image_ids.index(cid)]
        c_sim = color_sims_full.get(cid, 0.0)

        candidates.append({
            "image_id": cid,
            "sku": sku,
            "dino_sim": d_sim,
            "color_sim": c_sim,
        })

    # === Stage 2: Patch re-ranking ===
    for c in candidates:
        patch_path = os.path.join(patch_dir, f"{c['image_id']}.npy")
        if os.path.exists(patch_path):
            ref_patches = np.load(patch_path)
            c["patch_sim"] = max_of_mean_similarity(
                query_patches_t, torch.from_numpy(ref_patches)
            )
        else:
            c["patch_sim"] = c["dino_sim"]  # fallback

    # === Compute final scores ===
    for c in candidates:
        rerank_score = beta * c["dino_sim"] + (1 - beta) * c["patch_sim"]
        c["rerank_score"] = rerank_score
        c["final_score"] = gamma * c["color_sim"] + (1 - gamma) * rerank_score

    # === Aggregate to SKU level (max per SKU) ===
    from collections import defaultdict
    sku_candidates = defaultdict(list)
    for c in candidates:
        sku_candidates[c["sku"]].append(c)

    sku_scores = []
    for sku, cands in sku_candidates.items():
        best = max(cands, key=lambda x: x["final_score"])
        sku_scores.append({
            "sku": sku,
            "final_score": best["final_score"],
            "dino_sim": best["dino_sim"],
            "patch_sim": best["patch_sim"],
            "color_sim": best["color_sim"],
        })

    sku_scores.sort(key=lambda x: x["final_score"], reverse=True)
    return sku_scores[:top_n_final]
```

### Score Composition in Practice

```
Query: Green Sprite can
alpha=0.5, beta=0.3, gamma=0.4

                    dino_sim  patch_sim  color_sim  rerank(β=0.3)  final(γ=0.4)
Sprite front        0.93      0.91       0.92       0.916           0.918  ← ✓
Sprite 90°          0.88      0.85       0.88       0.859           0.871
Coke front          0.91      0.89       0.08       0.896           0.570  ← separated
Coke 90°            0.85      0.82       0.10       0.829           0.537
Pepsi front         0.87      0.84       0.12       0.849           0.558
Fanta front         0.86      0.83       0.75       0.839           0.803  ← similar shape, different color
Diet Coke front     0.92      0.90       0.88       0.906           0.896  ← same color family
```

Without color (gamma=0.0):

```
                    dino_sim  patch_sim  rerank(β=0.3)
Coke front          0.91      0.89       0.896        ← ranks ABOVE some Sprite angles!
Sprite 90°          0.88      0.85       0.859
Pepsi front         0.87      0.84       0.849
```

With color (gamma=0.4), Sprite products separate clearly from Coke and Pepsi. Fanta (orange) and Diet Coke (red, same family as Coke) sort correctly by their color similarity.

---

## Weight Tuning

### Gamma Tuning Guide

The optimal `gamma` depends on how much color differentiates your SKUs:

| Catalog characteristics | Recommended gamma | Rationale |
|---|---|---|
| Products differ mainly by color (Coke red, Sprite green, Pepsi blue) | **0.5** | Color is the primary discriminant |
| Mix: some color variants, some brand variants (Coke, Coke Zero, Diet Coke — all red) | **0.4** | Color filters obvious mismatches; patches handle same-color fine distinctions |
| Very diverse catalog (cans, bottles, boxes, different shapes) | **0.2–0.3** | Shape/structure matters more; color is supplementary |
| Monochrome catalog (all products same color, e.g., all white bottles) | **0.0** | Color adds no information; disable it |

### Grid Search for Optimal gamma

```python
def tune_gamma(test_queries, ground_truth_skus,
               model, collection, patch_dir, color_matrix,
               all_image_ids, all_skus,
               alpha=0.5, beta=0.3,
               gamma_range=np.arange(0.0, 0.6, 0.05)):
    """
    Find the optimal gamma on labeled test data.
    Alpha and beta are held constant at their existing values.
    """
    results = {}

    for gamma in gamma_range:
        correct = 0
        total = len(test_queries)

        for query_tensor, query_crop, true_sku in test_queries:
            scores = query(
                query_tensor, query_crop, model,
                collection, patch_dir, color_matrix,
                all_image_ids, all_skus,
                alpha=alpha, beta=beta, gamma=gamma,
            )
            if scores[0]["sku"] == true_sku:
                correct += 1

        accuracy = correct / total
        results[gamma] = accuracy
        print(f"gamma={gamma:.2f}  →  top-1 accuracy = {accuracy:.3f} "
              f"({correct}/{total})")

    best_gamma = max(results, key=results.get)
    print(f"\nBest: gamma={best_gamma:.2f} → {results[best_gamma]:.3f}")
    return best_gamma, results
```

**Expected output:**

```
gamma=0.00  →  top-1 accuracy = 0.880  (44/50)    ← current system
gamma=0.10  →  top-1 accuracy = 0.900  (45/50)
gamma=0.20  →  top-1 accuracy = 0.920  (46/50)
gamma=0.30  →  top-1 accuracy = 0.940  (47/50)
gamma=0.40  →  top-1 accuracy = 0.960  (48/50)    ← sweet spot
gamma=0.50  →  top-1 accuracy = 0.940  (47/50)    ← over-weighted
gamma=0.55  →  top-1 accuracy = 0.920  (46/50)    ← too much color
```

The typical pattern: accuracy rises as gamma increases, peaks, then falls when color starts drowning out structural similarity. Pick the gamma at the peak.

### Cross-Validating alpha, beta, gamma Together

If you want to jointly optimize all three:

```python
def tune_all(test_queries, ground_truth_skus, model, collection,
             patch_dir, color_matrix, all_image_ids, all_skus):
    """
    Joint grid search over alpha, beta, gamma.
    Warning: slow — use coarse grid first, then refine.
    """
    best_acc = 0
    best_params = None

    for alpha in [0.3, 0.5, 0.7]:
        for beta in [0.1, 0.3, 0.5]:
            for gamma in [0.2, 0.3, 0.4, 0.5]:
                correct = sum(
                    1 for qt, qc, true_sku in test_queries
                    if query(qt, qc, model, collection, patch_dir,
                             color_matrix, all_image_ids, all_skus,
                             alpha=alpha, beta=beta, gamma=gamma)[0]["sku"] == true_sku
                )
                acc = correct / len(test_queries)
                print(f"α={alpha:.1f} β={beta:.1f} γ={gamma:.1f}  →  {acc:.3f}")
                if acc > best_acc:
                    best_acc = acc
                    best_params = (alpha, beta, gamma)

    print(f"\nBest: α={best_params[0]:.1f} β={best_params[1]:.1f} "
          f"γ={best_params[2]:.1f} → {best_acc:.3f}")
    return best_params
```

---

## Storage and Latency

### Storage

| Component | Per Image | Total (2,400 imgs) |
|---|---|---|
| DINOv2 fused (ChromaDB) | 768 floats = 3 KB | ~7.2 MB |
| Patch tokens (numpy) | 256 × 768 floats = 790 KB | ~1.8 GB |
| **Color histogram (numpy)** | **49 floats = 200 B** | **~470 KB** |

Color adds **0.026%** to total storage. The color matrix for all 2,400 images fits in L2 cache (~470 KB).

### Latency

| Step | Time | Notes |
|---|---|---|
| DINOv2 feature extraction | ~25 ms | GPU, single image |
| Color histogram extraction | ~0.5 ms | CPU, OpenCV |
| ChromaDB query (DINOv2) | ~1 ms | top-K=50 |
| Color brute-force search | ~0.1 ms | 2,400 × 49 dot product |
| Patch re-ranking (K=50) | ~5 ms | loading + computation |
| **Total query time** | **~32 ms** | dominated by DINOv2 forward pass |

The color search adds essentially zero latency — 0.1ms on top of the existing ~32ms pipeline.

### Preloading Color Matrix

```python
# At application startup — takes <50ms
color_matrix = np.zeros((len(all_image_ids), 49), dtype=np.float32)
for i, iid in enumerate(all_image_ids):
    color_matrix[i] = np.load(os.path.join(color_dir, f"{iid}.npy"))

# Ensure L2-normalized rows for cosine similarity
norms = np.linalg.norm(color_matrix, axis=1, keepdims=True)
color_matrix = color_matrix / (norms + 1e-8)
```

---

## References

1. **Wagner et al., "Oh-A-DINO: Enhancing Self-Supervised Representations for Multi-Object Instance Retrieval"** — arXiv:2503.09867 (2025). Proves DINOv2 color retrieval = 40.8% and shows score-level fusion restores it to 85.2%.
2. **Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations" (SimCLR)** — ICML 2020. Established color augmentation as anti-shortcut mechanism. Explains why SSL models are color-blind by design.
3. **Darcet et al., "Vision Transformers Need Registers"** — ICLR 2024 (Oral). DINOv2 augmentation parameters (ColorJitter 0.4/0.4/0.2/0.1, Grayscale p=0.2).
4. **Trendyol e-commerce DINOv2** — `Trendyol/trendyol-dino-v2-ecommerce-256d`. Fine-tuned DINOv2 with ArcFace on e-commerce products, achieving 0.890 cosine similarity on SKU matching.
