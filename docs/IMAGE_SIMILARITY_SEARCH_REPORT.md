# Image Similarity Search: Deep Research Report

## System Overview

**Goal:** Search top-N similar images from a reference database given YOLO-detected beverage crop inputs (background masked black). Return ranked results with confidence scores.

**Current Stack:** DINOv2 embeddings → cosine similarity → ChromaDB with FAISS

**Scale:** ~200 SKUs × ~12 reference images each ≈ **~2,400 reference images**. Upper bound of ~10K images. This is a **small, fixed-size database** — which fundamentally shapes infrastructure and architectural decisions.

---

## 1. Evaluation of Current Method: DINOv2 + Cosine Similarity

### 1.1 Why DINOv2 Is a Strong Choice for This Use Case

DINOv2 is a **self-supervised Vision Transformer** trained on 142M images without labels. It produces general-purpose visual features that capture **visual structure, texture, shape, and spatial composition** — precisely the properties needed for distinguishing beverage products.

**Strengths for beverage similarity search:**

| Property | Relevance |
|---|---|
| Purely visual model (no text dependency) | Ideal for image-to-image retrieval without needing captions |
| Sensitive to visual structure & texture | Critical for distinguishing subtle label/bottle differences |
| No fine-tuning required | Strong out-of-the-box frozen features |
| Competitive instance-level retrieval | 58.2 mAP on Oxford-Hard (ViT-g/14 with registers) |
| Robust cross-domain generalization | +29.6% on ImageNet-A, +22.1% on ImageNet-R vs prior SSL |

**Benchmark evidence:** On the DISC21 image similarity benchmark (150K gallery images), DINOv2 achieved **64% accuracy** vs CLIP's 28.45%. Both models extract features at ~70 images/second. DINOv2 consistently outperforms CLIP on pure image-to-image similarity tasks.

### 1.2 Known Limitations for Your Use Case

1. **Background bias in embeddings.** Research (ICMV 2022: "On Background Bias in Deep Metric Learning") demonstrates that DML models — including ViT-based ones — encode background information into embeddings. Even with black-masked backgrounds, the model may allocate embedding capacity to the mask boundary region rather than the product itself. The paper shows that replacing backgrounds during training with random images improves retrieval by forcing the model to focus on the foreground.

2. **CLS token vs patch tokens.** DINOv2 outputs `x_norm_clstoken` (global) and `x_norm_patchtokens` (local). For retrieval, the CLS token is standard. However, for fine-grained discrimination (e.g., distinguishing two beverage variants with similar packaging), patch-level features may provide richer signals. The DINOv2 authors recommend the CLS token for image retrieval.

3. **Normalization sensitivity.** DINOv2 requires specific preprocessing: `ImageNet_mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`. Using wrong normalization (e.g., `mean=0.5, std=0.5`) produces near-random similarity scores. Black-masked regions will be affected by normalization — pixel values of 0 become negative after normalization, which could influence the CLS token.

4. **No semantic understanding.** DINOv2 captures visual similarity but not semantic meaning. A red cola can and a red energy drink may have high visual similarity despite being different products. This is acceptable for pure visual matching but problematic for product disambiguation.

5. **Model size trade-offs.** The practical model choices are:
   - **ViT-B/14** (86M params, 768-dim): Good baseline, fast inference
   - **ViT-L/14** (300M params, 1024-dim): Best accuracy/speed balance
   - **ViT-g/14** (1.1B params, 1536-dim): Best accuracy, expensive inference

### 1.3 Verdict on Current Approach

**DINOv2 + cosine similarity is a solid, well-justified baseline** for this task. It is the right choice over CLIP for pure image-to-image visual similarity. The main areas for improvement are in (a) handling the masked black backgrounds, (b) potential fine-tuning, and (c) confidence score calibration.

---

## 2. Alternative Embedding Models

### 2.1 Model Comparison Matrix

| Model | Type | Embedding Dim | Strengths | Weaknesses | Best For |
|---|---|---|---|---|---|
| **DINOv2** (current) | Self-supervised ViT | 768-1536 | Visual structure, texture, no labels needed | No semantic understanding, background bias | Image-to-image visual matching |
| **SigLIP 2** | Contrastive (text+image) | 768-1152 | Best fine-grained retrieval, multilingual | Text dependency, less community tooling | Product search with text attributes |
| **OpenCLIP ViT-L/14** | Contrastive (text+image) | 768 | Massive ecosystem, fine-tuning recipes | Lower visual-only accuracy than DINOv2 | Cross-modal search |
| **EVA-02-CLIP** | Contrastive (enhanced) | 1024 | Strong classification (80.4% IN-1k) | Heavier model, less retrieval-focused | Classification-heavy pipelines |
| **DFN (Data Filtering Network)** | Contrastive | 768 | Strong zero-shot, filtered training data | Training data bias toward IN/COCO/Flickr | General-purpose zero-shot |
| **EfficientNet-B3** (supervised) | CNN | 1536 | Fast, good with fine-tuning | Requires supervised training, less generalizable | Domain-specific with labels |
| **DINOv2 + dino.txt** | DINOv2 + text alignment | 1024 | Visual features + text retrieval | Lower text-image retrieval than SigLIP | Visual-first with text capability |

### 2.2 SigLIP 2 — The Strongest Alternative

SigLIP 2 (Google, 2025) is the current SOTA in the CLIP family. Key advantages:

- **Sigmoid loss** treats each image-text pair independently (vs softmax in CLIP), enabling more scalable and robust fine-grained matching
- **Outperforms SigLIP, CLIP, OpenCLIP** across zero-shot classification and retrieval benchmarks
- **Best-in-class for e-commerce/product retrieval**: SigLIP achieved SOTA retrieval on 5 of 6 e-commerce datasets (leboncoin study, 2025)
- **Multi-objective training**: Adds image captioning loss, self-distillation, and masked prediction on top of contrastive loss

**However, for your use case (pure image-to-image similarity with masked crops):**
- SigLIP's text-image alignment capability is **not leveraged** since you only do image queries
- DINOv2's purely visual features are arguably **better suited** for distinguishing visual details on masked crops
- SigLIP may be worth testing as a secondary model or for a **dual-embedding approach**

### 2.3 Recommendations

1. **Stay with DINOv2 as primary model** — it is the correct choice for image-only visual similarity
2. **Test SigLIP 2 as a secondary/complementary model** — especially if you later add text-based search
3. **Consider dual embeddings** (DINOv2 + SigLIP 2) stored side-by-side in the vector DB for hybrid retrieval

---

## 3. Handling Masked/Black-Background Images

This is the **most critical optimization opportunity** for your pipeline.

### 3.1 The Problem

When YOLO crops a beverage and masks the background to black:
- The model must process an image that is partially real content, partially artificial (black = 0)
- After ImageNet normalization, black pixels (0,0,0) become (-2.118, -2.036, -1.804) — strong negative values
- These negative values can **dominate the CLS token** if the masked region is large relative to the product
- The model may learn features related to the **mask shape/boundary** rather than the product

### 3.2 Proven Solutions from Research

**A. Background Replacement During Training (BGAugment — ICMV 2022)**

Research from "On Background Bias in Deep Metric Learning" shows that replacing backgrounds with random images during training forces the model to focus on the foreground object. Key findings:
- Tested with 5 loss functions (contrastive, triplet, multi-similarity, ArcFace, normalized softmax) across 3 datasets
- Background replacement improved retrieval consistently, even with imperfect automatic masks
- The method requires **no additional labeling** and **no inference-time cost**

*Application to your case:* If you fine-tune DINOv2, replace the black backgrounds in your training data with random images/textures.

**B. Proper Preprocessing for Black-Masked Images**

Two approaches:
1. **Use the raw crop without masking** — let the model process the original bounding box content including background. DINOv2's robustness to background variation means it may perform better with natural backgrounds than with artificial black masks
2. **If masking is necessary**, ensure the black pixels are handled correctly:
   - Consider replacing black (0,0,0) with the ImageNet mean values (124, 116, 104) so after normalization they become ~0, reducing their influence
   - Or use a "neutral" gray value that normalizes close to zero

**C. Crop-Aware Embedding (Guided Cropping Approach)**

Research on "Zero-Shot Visual Classification with Guided Cropping" shows that guiding the model's attention to the object region improves classification:
- Use tight bounding boxes from YOLO
- Consider multiple crop scales (0% to 30% margin around the object) and average embeddings
- This "object-centric augmentation" consistently improves CLIP and similar models

**D. MaskInversion for Localized Embeddings**

For advanced use: MaskInversion (2024) optimizes a token representation so that its explainability map matches a given mask. This produces embeddings focused only on the masked region. However, this adds per-query optimization cost.

### 3.3 Practical Recommendation

For your pipeline, the priority order should be:

1. **Test without masking first** — use raw YOLO crops directly. DINOv2's robust features may handle background variation well
2. **If masking is needed**, replace black with ImageNet-mean gray rather than pure black
3. **For fine-tuning**, use BGAugment-style background replacement in training data
4. **Evaluate** whether tight crops vs. padded crops produce better similarity scores

---

## 4. Alternative Similarity/Matching Approaches

### 4.1 Beyond Cosine Similarity: Distance Metrics

| Metric | Properties | When to Use |
|---|---|---|
| **Cosine similarity** (current) | Measures angle between vectors; scale-invariant | Default choice for normalized embeddings |
| **Euclidean distance (L2)** | Measures absolute distance; sensitive to magnitude | When embedding magnitude carries information |
| **Dot product (inner product)** | Combines angle and magnitude | For unnormalized embeddings |
| **Mahalanobis distance** | Accounts for feature covariance | When features have different variances |

**Note:** With L2-normalized embeddings (standard for DINOv2), cosine similarity and dot product are **equivalent**. This is what FAISS uses internally when you normalize vectors and use `IndexFlatIP`.

### 4.2 Fine-Tuning with Metric Learning Losses

For your use case, if you have labeled pairs of "same product" / "different product," fine-tuning with metric learning can significantly improve discrimination:

| Loss Function | Mechanism | Best For | Fine-grained Performance |
|---|---|---|---|
| **Triplet Loss** (FaceNet) | Anchor-Positive-Negative ranking | Fine-grained retrieval | ★★★★★ Best for preserving intra-class variance |
| **Supervised Contrastive Loss (SCL)** | Multi-positive multi-negative | Balanced retrieval | ★★★★★ Similar to triplet, better variance |
| **ArcFace** | Angular margin on softmax | Face/product recognition | ★★★★ Excellent inter-class separation |
| **Multi-Similarity Loss** | Uses all informative pairs | General retrieval | ★★★★ Good mining strategy |
| **Contrastive Loss** (pairs) | Binary pair comparison | Simple binary similarity | ★★★ Quick but compacts too aggressively |
| **InfoNCE / NT-Xent** | Softmax over batch negatives | Self-supervised pretraining | ★★★ Great for pretraining, not fine-tuning |

**Key insight from research (arXiv 2025):** Triplet loss and SCL consistently outperform contrastive/InfoNCE losses on **fine-grained retrieval** tasks because they preserve greater intra-class variance while enforcing clear inter-class margins. This is directly relevant to beverage product matching where products within a brand family share many visual features.

### 4.3 Advanced Retrieval Strategies

**A. Multi-Scale Embedding**
- Compute embeddings at multiple crop scales and concatenate or average
- Captures both global product shape and local label/text details
- Research shows ~2-5% improvement in retrieval accuracy

**B. Patch-Level Features + Aggregation**
- Use DINOv2's patch tokens (`x_norm_patchtokens`) instead of CLS token
- Apply GeM pooling (Generalized Mean pooling) or VLAD-style aggregation
- DINOv2 SALAD achieves SOTA visual place recognition by aggregating patch tokens with optimal transport

**C. Re-Ranking Pipeline**
- Stage 1: Fast ANN search with global CLS embeddings → top-100 candidates
- Stage 2: Re-rank using patch-level features with geometric verification
- Significant improvement for fine-grained matching at the cost of latency

**D. Feature Fusion**
- Combine DINOv2 features with hand-crafted features (color histograms for beverage labels)
- Or combine multiple model embeddings (DINOv2 + SigLIP 2)
- Concatenation or learned fusion improves robustness

### 4.4 Confidence Score Calibration

Your requirement for confidence scores deserves special attention:

**Raw cosine similarity is NOT a calibrated probability.** Cosine similarity of 0.95 does not mean 95% confidence.

Calibration methods:
1. **Platt Scaling:** Fit a logistic regression on cosine similarity scores using a validation set with binary same/different labels
2. **Temperature Scaling:** Divide logits by a learned temperature parameter
3. **Isotonic Regression:** Non-parametric calibration, more flexible
4. **Score normalization per query:** Normalize the top-N similarity scores to sum to 1.0 for relative ranking

**Recommendation:** Use Platt scaling with a labeled validation set to convert cosine similarity to calibrated confidence scores.

### 4.5 SKU-Level Aggregation (Specific to Your Scale)

With only ~200 SKUs and ~12 reference images each, you have a unique advantage: **you can afford exhaustive comparison and aggregation strategies that are infeasible at larger scales.**

**Approach: Per-SKU Score Aggregation**

Rather than returning individual image matches, aggregate similarity scores to the SKU level:

1. **Max pooling:** For each SKU, take the maximum similarity score across its ~12 reference images. Best for "does this input look like ANY reference image of this SKU?"
2. **Mean pooling:** Average the similarity scores across all reference images of a SKU. More stable, captures overall visual similarity to the product.
3. **Top-K mean:** Average the top-K (e.g., top-3) similarity scores per SKU. Balances outlier robustness with selectivity.
4. **Weighted voting:** Weight reference images by quality, viewpoint diversity, or representative-ness.

**Recommendation:** Use **top-3 mean per SKU** — this is robust to a few bad reference images while still capturing the best matches. With 200 SKUs, this is computationally trivial (200 × 12 = 2,400 comparisons).

---

## 5. Vector Database & FAISS Optimization

### 5.1 Scale Assessment: ChromaDB Is Perfectly Fine

At ~2,400 vectors (max ~10K), your database is **tiny** by vector search standards. This has major implications:

- **Brute-force exact search** over 2,400 vectors takes **<0.1ms** on any modern CPU
- No approximate index (HNSW, IVF) is needed — the overhead of building and querying the index exceeds the cost of brute force
- **ChromaDB is a perfectly appropriate choice** at this scale. Its convenience (metadata, API, persistence) far outweighs its abstraction overhead when the underlying computation is negligible
- Memory usage is trivial: 2,400 × 768 × 4 bytes (float32) ≈ **7.4 MB**

**Bottom line: Keep ChromaDB.** There is zero reason to migrate to raw FAISS, Milvus, or Qdrant at this scale. The engineering cost of migration would provide no measurable performance benefit.

### 5.2 When to Reconsider

Only reconsider your vector DB if:
- You expand to **>100K SKUs** (unlikely given the business context)
- You need **complex metadata filtering** that ChromaDB can't support
- You need **multi-modal queries** (text + image) that a more advanced DB supports natively

### 5.3 FAISS Index Selection (Reference)

For future scaling reference, should the database grow:

| Scale | Recommended Index | Parameters | Recall@10 | Latency |
|---|---|---|---|---|
| **< 10K** ← *you are here* | `IndexFlatIP` (brute force) | Normalize vectors, use inner product | 100% | <0.1ms |
| **10K - 100K** | `IndexFlatIP` (still brute force) | Normalize vectors, use inner product | 100% | <2ms |
| **100K - 1M** | `IndexHNSWFlat` | M=32, efConstruction=200, efSearch=64 | >99% | <1ms |
| **1M+** | `IndexIVFFlat` / `IndexIVFPQ` | Tune nlist/nprobe per scale | >95% | <2ms |

### 5.4 Practical Optimization

Even at small scale, one optimization matters:

- **L2-normalize all embeddings** before storing and querying. This ensures cosine similarity = dot product, which is what `IndexFlatIP` computes. ChromaDB may handle this automatically, but verify.

---

## 6. Recommended Architecture

### 6.1 Optimized Pipeline (Near-Term — What to Build Now)

```
Input Image
    ↓
YOLO Detection → Crop
    ↓
[Optional] Replace black mask with ImageNet-mean gray
    ↓
DINOv2 ViT-L/14 (CLS token, L2-normalized)
    ↓
Cosine similarity against all ~2,400 ref embeddings (brute force)
    ↓
Aggregate to SKU level: top-3 mean similarity per SKU
    ↓
Platt-scaled confidence scores per SKU
    ↓
Ranked SKU list with confidence
```

**Why brute force is optimal here:** 2,400 dot products on a 768-dim vector takes <0.1ms. No index needed. ChromaDB handles this transparently.

### 6.2 Enhanced Pipeline (Medium-Term — If Accuracy Needs Improvement)

```
Input Image
    ↓
YOLO Detection → Crop (keep background, or use ImageNet-mean gray mask)
    ↓
Multi-scale embedding (average of 3 crop scales):
  - DINOv2 ViT-L/14 CLS token (global visual)
    ↓
Cosine similarity against all ref embeddings
    ↓
SKU aggregation (top-3 mean per SKU)
    ↓
Calibrated confidence scores → Ranked results
```

**Optional enhancement:** If certain SKU pairs are commonly confused, add patch-level features (DINOv2 `x_norm_patchtokens` + GeM pooling) as a secondary signal for disambiguation.

### 6.3 Advanced Pipeline (Long-Term — If Fine-Tuning Is Warranted)

```
Input Image
    ↓
YOLO Detection → Crop
    ↓
Fine-tuned DINOv2 (triplet loss or SCL, BGAugment training)
    ↓
Cosine similarity → SKU aggregation
    ↓
Optional: Re-rank confused SKU pairs with patch-level features
    ↓
Calibrated confidence → Ranked results
```

**When to consider this:** Only if the out-of-the-box DINOv2 pipeline has unacceptable confusion between specific SKU pairs after testing. Fine-tuning requires a labeled dataset of same-SKU / different-SKU pairs and careful evaluation.

---

## 7. Summary of Recommendations

### Quick Wins (Implement Now)

| # | Action | Expected Impact | Effort |
|---|---|---|---|
| 1 | **Test raw crops without black masking** | May improve accuracy by reducing mask artifacts | Low |
| 2 | **Ensure correct ImageNet normalization** | Critical — wrong normalization = random results | Low |
| 3 | **Use ViT-L/14 instead of ViT-B/14** if not already | +2-3% retrieval accuracy | Low |
| 4 | **Add SKU-level score aggregation** (top-3 mean per SKU) | More robust than single-image matching | Low |
| 5 | **Calibrate confidence scores** with Platt scaling | Meaningful confidence numbers | Medium |

### Medium-Term Improvements (If Accuracy Is Insufficient)

| # | Action | Expected Impact | Effort |
|---|---|---|---|
| 6 | **Multi-scale crop augmentation** during embedding | +2-5% retrieval accuracy | Medium |
| 7 | **Patch-level features + GeM pooling** for confused SKU pairs | Better discrimination of similar products | Medium |
| 8 | **Feature fusion** — add color histograms for beverage label colors | Additional signal for colorful packaging | Low |

### Long-Term (If Fine-Tuning Is Warranted)

| # | Action | Expected Impact | Effort |
|---|---|---|---|
| 9 | **Fine-tune DINOv2 with triplet loss** on your labeled beverage pairs | +5-15% for your domain | High |
| 10 | **Use BGAugment** (background replacement) during fine-tuning | Better foreground focus | Medium |
| 11 | **Dual-embedding system** (DINOv2 + SigLIP 2) | Best of visual + semantic matching | High |

### What NOT to Do

| # | Action | Why Skip |
|---|---|---|
| ❌ | **Migrate from ChromaDB to raw FAISS/Milvus/Qdrant** | No benefit at ~2,400 vectors. ChromaDB is perfectly adequate. |
| ❌ | **Build HNSW or IVF indexes** | Brute force is faster and 100% accurate at this scale. Indexes add complexity for no gain. |
| ❌ | **GPU-accelerated vector search** | CPU brute force over 2,400 vectors is already sub-millisecond. |
| ❌ | **Complex re-ranking pipelines** | With only 200 SKUs, the initial similarity search is already essentially exhaustive. |

---

## 8. Key References

1. **DINOv2:** Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision" (2023) — arXiv:2304.07193
2. **SigLIP 2:** Tschannen et al., "SigLIP 2: Multilingual Vision-Language Encoders" (2025) — arXiv:2502.14786
3. **Background Bias:** Beyer et al., "On Background Bias in Deep Metric Learning" (ICMV 2022) — arXiv:2210.01615
4. **Triplet Loss:** Schroff et al., "FaceNet: A Unified Embedding for Face Recognition" (CVPR 2015)
5. **Contrastive vs Triplet:** "Comparing Contrastive and Triplet Loss: Variance Analysis" (arXiv:2510.02161)
6. **FAISS:** Johnson et al., "The Faiss Library" (2024) — arXiv:2401.08281
7. **DINOv2 vs CLIP Benchmark:** Jeremy K, "CLIP vs DINOv2 in Image Similarity" (AI Monks, 2023)
8. **E-commerce Embeddings:** leboncoin tech, "How to Choose the Best Image Embeddings" (2025)
9. **Guided Cropping:** "Zero-Shot Visual Classification with Guided Cropping" (2023) — arXiv:2309.06581
10. **MaskInversion:** "MaskInversion: Localized Embeddings via Optimization" (2024) — arXiv:2407.20034
11. **DINOv2 SALAD:** "DINOv2 SALAD: Visual Place Recognition" (2024) — github.com/VictoireWood/UAV-DINOv2-fine-tune
12. **SigLIP 2 vs DINOv2:** Underfitted.dev, "Battle of the Embeddings Titans" (2026)
13. **Metric Loss Comparison:** "Variance Analysis and Optimization Behavior" (arXiv:2601.21450)
14. **Multimodal Benchmarking:** Marqo, "Benchmarking Models for Multi-modal Search" (2025)
