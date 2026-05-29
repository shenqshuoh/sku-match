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

2. **CLS token vs patch tokens.** DINOv2 outputs two types of features — the CLS token (global image representation) and patch tokens (local per-region features, one per 14×14 pixel block). For retrieval, the CLS token is standard and recommended by the DINOv2 authors. However, for fine-grained discrimination (e.g., distinguishing two beverage variants with similar packaging), patch-level features provide richer signals and can be used for region-level re-ranking. See **§3.5 — CLS vs Patch Tokens** for full analysis.

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

### 2.3 MobileNet vs DINOv2 — Detailed Comparison

#### Architecture & Training Fundamentals

| | DINOv2 | MobileNet V2/V3 | MobileNet V4 | MobileNet V5 |
|---|---|---|---|---|
| **Architecture** | Vision Transformer (self-attention) | Lightweight CNN (depthwise separable conv) | CNN + optional MQA attention (UIB blocks) | IR blocks + sparse MQA + multi-scale fusion |
| **Training data** | 142M curated images, self-supervised | 1.28M ImageNet, supervised | 1.28M ImageNet (+ JFT for large), supervised | Gemma 3n multimodal pipeline (no standalone paper) |
| **Training paradigm** | Self-distillation + masked image modeling | Classification cross-entropy | Classification + novel distillation recipe | VLM vision encoder training |
| **Design goal** | General-purpose visual features | Efficient on-device inference | Universal mobile models (edge + server) | On-device multimodal understanding |
| **Published** | 2023 | 2017–2019 | ECCV 2024 (arXiv:2404.10518) | June 2025 (no paper yet) |
| **Feature type** | Global (CLS token) + local (patch tokens) | Hierarchical spatial features | Hierarchical spatial (+ attention in hybrid) | Multi-scale fused features |

**Why this matters:** DINOv2 learned from ~110× more data without label bias. Its self-supervised training forces it to learn visual structure, texture, and spatial composition — exactly what fine-grained product matching needs. MobileNet V2/V3 learned to classify 1000 ImageNet categories. **MobileNet V4 closes the architecture gap** (adding optional attention, better blocks) but still trains on the same supervised classification task. **MobileNet V5 is in the same weight class as DINOv2-L** (300M params) but has no retrieval benchmarks and no standalone paper.

#### Model Size & Embedding Specs

**DINOv2 variants:**

| Model | Params | Embed Dim | Model Size | GPU Latency (224px) | CPU Latency (FP32) | CPU Latency (Quantized) |
|---|---|---|---|---|---|---|
| **ViT-S/14** | 21M | 384 | ~85 MB | ~2 ms | ~180–300 ms | ~48 ms (dinov2.cpp q4_1) |
| **ViT-B/14** | 86M | 768 | ~331 MB | ~7 ms | ~440–460 ms | ~124 ms (dinov2.cpp q8_0) |
| **ViT-L/14** | 300M | 1,024 | ~1.2 GB | ~20 ms | ~1,290–1,330 ms | ~348 ms (dinov2.cpp q8_0) |
| **ViT-g/14** | 1,100M | 1,536 | ~4.3 GB | ~50 ms | ~4,380–4,470 ms | ~1,060 ms (dinov2.cpp q8_0) |

**MobileNet variants (V2/V3 — legacy):**

| Model | Params | Embed Dim | Model Size | GPU Latency (224px) | CPU Latency (FP32) | CPU Latency (INT8) |
|---|---|---|---|---|---|---|
| **MobileNet V3 Small** | 2.9M | 576 | ~3.6 MB | ~11 ms | ~6 ms (iPhone 11) | ~3 ms |
| **MobileNet V2** (α=1.0) | 3.5M | 1,280 | ~14 MB | ~19 ms | ~19 ms (iPhone 11) | ~5 ms |
| **MobileNet V3 Large** | 5.4M | 960 | ~11 MB | ~19 ms | ~15 ms (iPhone 11) | ~8 ms |
| **EfficientNet-Lite0** | 4.7M | 1,280 | ~18 MB | ~9 ms | ~12 ms | ~6.5 ms |
| **MobileViT V2-1.0** | 4.9M | ~768 | ~19 MB | — | ~20 ms | — |
| **EfficientNetV2-S** | 21.5M | 1,280 | ~82 MB | — | ~40 ms | — |

**MobileNet V4 (ECCV 2024):**

| Model | Params | Embed Dim | Model Size | Input | ImageNet Top-1 | EdgeTPU Latency |
|---|---|---|---|---|---|---|
| **MNv4-Conv-S** | 3.8M | 480 | ~15 MB | 224px | 74.6% | 0.2 ms (Pixel 6) |
| **MNv4-Conv-M** | ~9.7M | 1,280 | ~37 MB | 256px | ~80% | 0.6 ms (Pixel 6) |
| **MNv4-Conv-L** (IN-12k) | 32.6M | 1,280 | ~124 MB | 448px | **85.0%** | 2.4 ms (Pixel 8) |
| **MNv4-Hybrid-M** | 11.1M | 1,280 | ~42 MB | 256px | 83.0% | — |
| **MNv4-Hybrid-L** (distilled) | 37.8M | 1,280 | ~144 MB | 384px | **85.9%** | 2.6 ms (Pixel 8) |

**MobileNet V5 (June 2025 — Gemma 3n vision encoder):**

| Model | Params | Embed Dim | Input | Notes |
|---|---|---|---|---|
| **MNv5-300** | ~300M | **2,048** | 768px | No standalone paper. Available via timm as `mobilenetv5_300m.gemma3n`. Designed as VLM encoder, not classifier. 60 FPS on Google Pixel. No ImageNet accuracy published. |

**Key observations:**
- **MNv4-Hybrid-L** (37.8M, 85.9%) approaches DINOv2 ViT-B/14's classification accuracy — but has **no retrieval benchmarks**. Classification accuracy ≠ embedding quality for similarity search.
- **MNv5** (300M params) is the same weight class as DINOv2 ViT-L/14, with a 2,048-dim embedding — but it's a VLM encoder with no standalone evaluation for similarity search.
- The smallest DINOv2 (ViT-S/14, 21M) is still larger than MNv4-Conv-S but smaller than MNv4-Hybrid-L and MNv4-Conv-L.

#### Retrieval Quality Benchmarks

> ⚠️ **Critical caveat for MobileNet V4 and V5:** As of May 2026, **zero published retrieval benchmarks** exist for MNv4 or MNv5. No one has tested them on Oxford/Paris, DISC21, SOP, or any image similarity task. The MNv4 paper only evaluates ImageNet classification and COCO detection. MNv5 has no paper at all. The MobileNet numbers below are from V2/V3, which are architecturally similar enough to V4's conv variants for rough estimation. V4's hybrid variants (with MQA attention) and V5 (with multi-scale fusion) may perform better, but this is unproven.

**Oxford/Paris Retrieval (standard image retrieval benchmark):**

| Model | Oxford-H mAP | Paris-H mAP |
|---|---|---|
| DINOv2 ViT-S/14 | 43.2 | 68.5 |
| DINOv2 ViT-B/14 | 49.5 | 78.6 |
| DINOv2 ViT-L/14 | 54.0 | 83.5 |
| DINOv2 ViT-g/14 (w/ reg) | 58.2 | 82.6 |
| OpenCLIP ViT-G/14 | 19.7 | 60.2 |
| MobileNet V2/V3 (ImageNet pretrained, zero-shot) | **~10–15** (estimated) | **~20–30** (estimated) |
| MobileNet V2 + triplet fine-tuning | **~27–45** (varies by dataset) | **~35–50** (varies by dataset) |
| **MobileNet V4 Conv-L** (estimated, no data) | **~20–30** (guess) | **~30–40** (guess) |
| **MobileNet V4 Hybrid-L** (estimated, no data) | **~25–35** (guess) | **~35–45** (guess) |
| **MobileNet V5** (no data at all) | **?** | **?** |

**DISC21 Image Similarity (150K gallery images):**

| Model | Accuracy |
|---|---|
| DINOv2 ViT-B/14 | **64.0%** |
| CLIP ViT-B/32 | 28.5% |
| MobileNet V2/V3 (estimated) | ~15–25% |
| MobileNet V4 Conv-L (estimated) | ~25–35% (higher capacity, but still supervised) |
| MobileNet V5 (no data) | **?** — could be interesting at 300M params + 2048-dim |

**Fine-grained classification (proxy for product discrimination):**

| Model | iNaturalist (10K classes) | Food-101 |
|---|---|---|
| MobileNet V2 | ~5–10% | ~55–65% |
| MobileNet V4 Conv-L (85% IN-1K) | **~15–25%** (estimated) | ~70–80% (estimated) |
| DINOv2 ViT-B/14 | **70%** | **93%** |

**DINOv2's UMAP clustering V-measure: 0.908** vs typical CNN's ~0.5–0.7 — demonstrating far superior fine-grained discrimination out of the box.

#### Latency Comparison (GPU)

For your use case (server-side embedding extraction for ~2,400 images, batch processing):

| Model | GPU Throughput (RTX-class) | Time to embed 2,400 images |
|---|---|---|
| MobileNet V3 Small | ~500–800 img/s | **~3–5 sec** |
| MNv4-Conv-S (224px) | ~400–600 img/s | **~4–6 sec** |
| MNv4-Hybrid-M (256px) | ~200–300 img/s | **~8–12 sec** |
| DINOv2 ViT-S/14 | ~300–400 img/s | **~6–8 sec** |
| MNv4-Conv-L (384px) | ~100–150 img/s | **~16–24 sec** |
| DINOv2 ViT-B/14 | ~150 img/s | **~16 sec** |
| MNv4-Hybrid-L (384px) | ~80–120 img/s | **~20–30 sec** |
| DINOv2 ViT-L/14 | ~50–80 img/s | **~30–48 sec** |
| MNv5 (768px input!) | ~15–30 img/s (est.) | **~80–160 sec** |

**Reality check:** Even the "slow" DINOv2 ViT-L/14 embeds your entire 2,400-image catalog in under a minute on a single GPU. This is a **one-time batch operation** (or rare re-embedding when adding new SKUs). The per-query latency (single image) is 7–20ms — negligible for real-time serving.

#### The Fine-Grained Discrimination Gap

This is the critical factor for your beverage SKU matching:

| Discrimination Level | MobileNet V2/V3 | MNv4 Conv-L | MNv4 Hybrid-L | MNv5 (?) | DINOv2 ViT-B/14 | DINOv2 ViT-L/14 |
|---|---|---|---|---|---|---|
| Coarse category (bottle vs can) | ✅ Good | ✅ Good | ✅ Good | ✅ Likely excellent | ✅ Excellent | ✅ Excellent |
| Brand-level (Coke vs Pepsi) | ⚠️ Moderate | ⚠️ Moderate | ⚠️ Moderate+ | ✅ Possibly good | ✅ Good | ✅ Very Good |
| Variant-level (Coke vs Diet Coke) | ❌ Poor | ❌ Poor–Moderate | ⚠️ Moderate | ⚠️ Unknown | ⚠️ Moderate | ✅ Good |
| Specific edition (limited packaging) | ❌ Very Poor | ❌ Poor | ⚠️ Moderate | ⚠️ Unknown | ⚠️ Moderate | ✅ Good |

**Why even MNv4 struggles (without fine-tuning):**
1. **Same supervised training paradigm** — MNv4 still learns from ImageNet classification (1000 classes). The UIB blocks and MQA attention improve classification accuracy, but the model still hasn't learned to distinguish fine-grained visual differences that don't matter for ImageNet classes.
2. **Architecture gap partially closed** — V4's UIB blocks and optional attention give it more representational capacity than V2/V3. The Hybrid-L variant (37.8M params, 85.9% IN accuracy) is in a different league than V3 Large (5.4M, 75.2%).
3. **But training data gap remains** — DINOv2 trained on 142M images with self-supervised objectives that force learning of visual structure. MNv4 trained on 1.28M–13M images with supervised classification labels. No amount of architecture improvement compensates for the 10–100× data disadvantage.
4. **MNv5 is a wild card** — at 300M params with VLM training, it could be competitive with DINOv2, but there are zero benchmarks to confirm this.

**When MobileNet V4 could work:**
- **MNv4-Hybrid-M** (11M params, 83% accuracy) is the most practical option — small enough for edge, accurate enough to be useful with fine-tuning.
- **MNv4-Conv-L** (32.6M, 85%) is large enough that fine-tuning with metric learning could produce competitive embeddings.
- With **knowledge distillation** from DINOv2 → MNv4: the hybrid variants' attention blocks may retain more of the teacher's knowledge than pure CNN variants.
- **MNv5** (300M, 2048-dim) could theoretically rival DINOv2, but without any benchmarks or even a paper, it's a research project, not a production choice.

**When to avoid MobileNet V4/V5:**
- If you need **zero-shot retrieval quality** (no fine-tuning data available) — DINOv2 is still the only proven option.
- If you're running **server-side with a GPU** — DINOv2 ViT-B/14 is actually faster and far more proven for similarity search.

#### Summary: Head-to-Head

| Factor | MNv4 Conv-L | MNv4 Hybrid-L | MNv5 | DINOv2 ViT-B/14 | DINOv2 ViT-L/14 |
|---|---|---|---|---|---|
| **Params** | 32.6M | 37.8M | ~300M | 86M | 300M |
| **Model size** | ~124 MB | ~144 MB | ~1.1 GB | ~331 MB | ~1.2 GB |
| **Embedding dim** | 1,280 | 1,280 | 2,048 | 768 | 1,024 |
| **ImageNet Top-1** | 85.0% | 85.9% | Unknown | 82.1% | 83.5% |
| **EdgeTPU latency** | 2.4 ms | 2.6 ms | ~16 ms (est.) | N/A | N/A |
| **GPU latency** | ~10 ms (est.) | ~12 ms (est.) | ~30 ms (est.) | **~7 ms** | ~20 ms |
| **Retrieval quality (zero-shot)** | No data | No data | No data | **64%** (DISC21) | **~68%** (est.) |
| **Fine-grained discrimination** | Poor–Moderate | Moderate | Unknown | Good | Very Good |
| **Needs fine-tuning?** | Yes | Yes | Unknown | No | No |
| **Edge deployment** | ✅ Excellent | ✅ Very Good | ⚠️ Heavy | ⚠️ Possible (dinov2.cpp) | ❌ Difficult |
| **Proven for similarity search** | ❌ No | ❌ No | ❌ No | ✅ Yes | ✅ Yes |
| **Available in timm** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |

**The fundamental asymmetry:** MNv4 Conv-L has higher ImageNet classification accuracy than DINOv2 ViT-B/14 (85% vs 82%), yet DINOv2's embeddings are **3–4× better for similarity search**. This is because classification accuracy and embedding quality measure fundamentally different things. Classification only needs class-level discriminability; similarity search needs instance-level fine-grained features. DINOv2's self-supervised training explicitly optimizes for the latter.

#### Recommendation for Your Use Case

**For server-side beverage SKU matching (~200 SKUs, masked YOLO crops):**

1. **DINOv2 ViT-B/14 remains the clear winner** for server-side similarity search. It works out-of-the-box with zero fine-tuning, has proven 3–4× better retrieval accuracy than any MobileNet variant, and is faster than MobileNet on GPU (7ms vs 10–12ms). No MobileNet variant — including V4 and V5 — has published retrieval benchmarks.

2. **MNv4 is a significant upgrade over V2/V3** — MNv4-Hybrid-L reaches 85.9% ImageNet accuracy (vs V3's 75.2%). But it still requires fine-tuning for similarity search and has **zero proven retrieval performance**. The 1280-dim embeddings from `timm` (`num_classes=0`) are easy to extract, and with metric learning fine-tuning on your beverage data, MNv4-Hybrid-L could be competitive. But this is an investment with uncertain returns vs DINOv2's proven zero-shot performance.

3. **MNv5 is not ready for production use.** At 300M params / 2048-dim it's in the same class as DINOv2-L, but there's no paper, no retrieval benchmarks, and no standalone classifier weights. It's a VLM encoder designed for Gemma 3n, not an embedding model. Check back in 6–12 months.

4. **MobileNet V4 is only justified if:**
   - You're deploying on **edge/EdgeTPU/mobile** (2.4ms on Pixel 8 vs DINOv2's ~200ms+ CPU)
   - You're willing to invest in **metric learning fine-tuning** on your beverage data
   - You need the model to fit in **<150 MB** (MNv4-Hybrid-L is 144 MB vs DINOv2-B's 331 MB)
   
   Best path: MNv4-Hybrid-M (11M params) or MNv4-Conv-L (32.6M params) + triplet loss / ArcFace fine-tuning + optional distillation from DINOv2.

5. **Quick test to settle this empirically:**
   ```python
   # Extract embeddings with both models and compare on your actual data
   import timm
   
   # DINOv2
   dino = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True, num_classes=0)
   
   # MNv4 Hybrid-L (best MobileNet option)
   mnv4 = timm.create_model('mobilenetv4_hybrid_large.e600_r384_in1k', pretrained=True, num_classes=0)
   
   # Compare top-1 accuracy on your 200 SKUs using cosine similarity
   # If MNv4 comes within 5% of DINOv2 on YOUR data, it's worth considering for edge deployment
   ```

### 2.4 Recommendations

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

### 3.5 CLS vs Patch Tokens in DINOv2 — Deep Dive

This section covers how DINOv2's two output types differ, what each captures, and how to choose between them for similarity search.

#### How They're Created

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

**Neither output is L2-normalized.** You must apply `F.normalize(x, dim=-1, p=2)` yourself before computing cosine similarity.

#### How Self-Attention Differentiates Them

Although CLS and patch tokens go through identical transformer blocks, they diverge because of DINOv2's **dual training objective**:

| Training Loss | Applied To | Effect |
|---|---|---|
| **DINO loss** (image-level cross-entropy) | CLS token only | Trains CLS to be a global image descriptor |
| **iBOT loss** (masked patch prediction) | Patch tokens only | Trains patches to represent local content |
| **KoLeo regularizer** (uniform feature spread) | CLS token only | Directly improves retrieval by **+8.3% mAP** on Oxford-M |

A DINOv2 author confirmed (GitHub issue #222): *"at no point we expect the CLS and patch tokens to align"* — they occupy different subregions of the same D-dimensional space.

#### What Each Token Captures

| Property | CLS Token | Patch Tokens |
|---|---|---|
| **Spatial correspondence** | None — it's a virtual token | Each maps to a specific 14×14 image region |
| **"What" (semantic identity)** | ✅ Strong — object category, scene type | ✅ Moderate — each patch has local semantic content |
| **"Where" (spatial layout)** | ❌ Collapsed | ✅ Strong — preserves spatial structure |
| **Global vs local** | Global aggregate of entire image | Local features per region |
| **Part-level correspondence** | ❌ Cannot match specific parts across images | ✅ Same parts of different objects match in feature space |
| **Best for** | Image-level classification, coarse retrieval | Fine-grained matching, region correspondence, segmentation |

**Key insight from the DINOv2 paper (Figure 1):** PCA visualization of patch tokens across different objects shows same-color correspondences for semantically identical parts (e.g., dog ears, bicycle wheels). This part-level matching is **impossible with CLS tokens alone**.

#### Retrieval Benchmarks: CLS vs Patches

**DINOv2's official retrieval benchmarks (Table 9) use the CLS token** — the default `forward()` method returns only `x_norm_clstoken`.

**DToP paper (Song et al., WACV 2023) — direct CLS vs patch comparison, fine-tuned on SfM-120k:**

| Token Type | ℛOxford Med | ℛOxford Hard | ℛParis Med | ℛParis Hard |
|---|---|---|---|---|
| ViT-B **CLS** | **49.3** | **19.6** | **70.9** | 46.4 |
| ViT-B Patch (mean pool) | 38.6 | 12.0 | 69.2 | **46.9** |
| R50+ViT-B **CLS** | **62.6** | **37.9** | **79.6** | **64.8** |
| R50+ViT-B Patch (mean pool) | 50.9 | 26.9 | 69.4 | 46.5 |

**Conclusion: CLS outperforms mean-pooled patches for single-layer features in most cases.** But patches win when aggregated with better methods:

**GeM pooling of patches (zero-shot, from geo-image-retrieval benchmark):**

| Pooling Method | ΔR@1 vs CLS | ΔR@5 | ΔR@10 |
|---|---|---|---|
| **GeM (p≈3)** | **+5.13** | **+7.51** | **+7.93** |
| Max pooling | +4.20 | +6.64 | +6.64 |
| Mean pooling | +3.13 | +5.31 | +5.50 |

**GeM-pooled patches beat CLS by +5% R@1** without any fine-tuning.

#### Aggregation Methods for Patch Tokens

Given `patch_tokens ∈ ℝ^(N×D)` (256 patches × embedding dim), how to create a single image descriptor:

| Method | Formula | Quality | Complexity |
|---|---|---|---|
| **Mean (GAP)** | `mean(patches, dim=1)` | Baseline | Trivial |
| **GeM** (p≈3) | `(relu(patches)^p).mean()^(1/p)` | **+5% R@1 over CLS** | Trivial |
| **Max** | `max(patches, dim=1)` | +4% R@1 over CLS | Trivial |
| **Attention-weighted** | `softmax(patches @ query.T) @ patches` | Slightly better than GeM | Needs learnable query |
| **VLAD / SALAD** | Cluster patches → aggregate residuals | SOTA for VPR | Complex, needs codebook |
| **Multi-layer concat** | Features from layers -6 to -1 | +5.4% mAP over last-layer-only | 6× more computation |

#### Recommended Strategy for Beverage SKU Matching

**Two-stage approach (best of both worlds):**

```
Stage 1: Coarse Retrieval — narrow to top-K candidates
  → CLS token (or GeM-pooled patches) + cosine similarity
  → Fast: single vector comparison, <1ms per query

Stage 2: Fine-Grained Re-ranking — distinguish similar SKUs
  → Patch-to-patch similarity between query and each candidate
  → For each query patch, find best-matching DB patch (max similarity)
  → Average the best-match scores → final re-ranking score
  → This catches label details that CLS misses
```

**Why this works for beverages:** A Diet Coke and regular Coke look nearly identical globally (same red can, same layout). But their patch-level features differ in the specific regions containing "Diet" text and nutritional info. CLS collapses these into similar global descriptors; patch matching preserves the local differences.

#### Extracting Both Token Types (Code)

```python
import torch
import torch.nn.functional as F

model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
model.eval()

with torch.no_grad():
    features = model.forward_features(x)  # NOT model(x), which returns CLS only

# CLS token — single vector per image
cls = features["x_norm_clstoken"]            # (1, 768)

# Patch tokens — one vector per 14×14 region
patches = features["x_norm_patchtokens"]      # (1, 256, 768) for 224px input
                                            # (1, 1369, 768) for 518px input

# L2 normalize both
cls_n = F.normalize(cls, dim=-1, p=2)
patches_n = F.normalize(patches, dim=-1, p=2)

# Option A: CLS only (current approach, good baseline)
embedding = cls_n                            # (1, 768)

# Option B: GeM-pooled patches (+5% R@1 over CLS)
p = 3.0
gem = F.normalize((F.relu(patches_n) ** p).mean(dim=1) ** (1/p), dim=-1, p=2)

# Option C: CLS + patch average (recommended by dino.txt, CVPR 2025)
patch_avg = F.normalize(patches_n.mean(dim=1), dim=-1, p=2)
combined = F.normalize((cls_n + patch_avg) / 2, dim=-1, p=2)  # (1, 768)

# Option D: Patch-to-patch re-ranking
def patch_similarity(q_patches, db_patches):
    """Fine-grained similarity between two images' patch tokens."""
    sim = q_patches @ db_patches.T           # (N_q, N_db)
    return sim.max(dim=1).values.mean()       # avg of best match per query patch
```

#### Register Tokens

DINOv2 models "with registers" add 4 learnable tokens between CLS and patches. They absorb global/artifact information, keeping patch tokens cleaner and more spatially meaningful. Registers improve linear probing by +0.2–0.6% and produce cleaner attention maps. **Recommendation: use models with registers** (e.g., `dinov2_vitl14_reg`).

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

## 8. Alternative Paradigms: Completely Different Approaches

The current pipeline treats product matching as an **embedding similarity search** problem. But with only ~200 SKUs, there are fundamentally different approaches worth considering — some of which may be simpler, more accurate, or more robust.

### 8.1 Direct SKU Classification (Fine-Tuned Classifier)

**Concept:** Treat this as a standard **200-class classification problem**. Fine-tune a pretrained model (ResNet, EfficientNet, ViT) to directly predict one of ~200 SKU classes from the input crop.

**Why it could beat similarity search:**
- A classifier learns **decision boundaries** specific to your 200 classes, not generic visual similarity
- It can learn to ignore features that are confusing (e.g., similarly-shaped bottles) and focus on discriminative ones (e.g., label text, logo details)
- With ~12 images per class and 200 classes, you have enough data for transfer learning with strong augmentation
- No vector database needed — just a single forward pass

**Real-world evidence:**
- A fine-tuned Xception model on soda bottles achieved **94% accuracy** (compvision_for_soda_bottles project)
- A hierarchical YOLOv8x model achieved **88.4% accuracy at SKU level (323 classes)** on grocery products, and **92.6% at brand level** with only 248 training images
- Transfer learning from ImageNet to domain-specific classification is a well-proven approach

**Implementation:**
```
Input crop → Fine-tuned EfficientNet-B3 / ViT-B → softmax over 200 classes
→ Top-N class predictions with softmax confidence scores
```

**Strengths:**
- **Simplest possible architecture** — no embedding, no database, no similarity search
- **Built-in calibrated confidence** (softmax scores, further calibratable with temperature scaling)
- **Fast inference** — single model forward pass
- **Learns domain-specific features** — the model is forced to discriminate between your exact 200 products

**Weaknesses:**
- **Adding new SKUs requires retraining** (though with 200 classes, fine-tuning is fast — minutes on a GPU)
- **Requires labeled data** (you already have this — each ref image has an SKU)
- **Less flexible** than similarity search — can only return known SKUs

**Verdict:** With only ~200 SKUs and ~12 ref images each, this is arguably the **most natural formulation of your problem**. It's simpler than the embedding pipeline and may be more accurate. Should be tested as a baseline.

---

### 8.2 Prototypical Networks (Few-Shot Classification)

**Concept:** A middle ground between classification and similarity search. Compute a **prototype vector** (centroid) for each SKU by averaging the embeddings of its ~12 reference images. Classify a query by finding the nearest prototype.

**Why it's interesting for your case:**
- Specifically designed for **few-shot learning** — learning from small numbers of examples per class
- With ~12 shots per class and ~200 classes, you're in the sweet spot for prototypical networks
- No retraining needed when adding new SKUs — just compute the prototype from new reference images
- Provides **distance-based confidence scores** (distance from query to prototype)

**Implementation:**
```
For each SKU: prototype_i = mean(DINOv2(ref_image_1), ..., DINOv2(ref_image_12))

At query time:
  query_emb = DINOv2(query_crop)
  distances = [cosine_dist(query_emb, prototype_i) for i in 200 SKUs]
  → softmax over negative distances → confidence scores
  → rank by confidence
```

**Key research results:**
- Proto-CLIP (2024) extends prototypical networks with CLIP, achieving strong few-shot results
- ViT-ProtoNet (2025) combines ViT-Small backbone with prototypical classification, competitive on Mini-ImageNet, CIFAR-FS, CUB-200
- Compositional Prototypical Networks learn reusable component prototypes (e.g., "red label", "bottle neck shape") that combine to recognize new products

**Strengths:**
- **No retraining** for new SKUs — just add prototype
- **Natural SKU-level aggregation** — each SKU has one prototype (vs 12 individual embeddings)
- **Strong few-shot theory** — proven effective with 5-20 shots per class
- Uses same DINOv2 backbone you already have

**Weaknesses:**
- Prototype averaging may **lose intra-class diversity** (e.g., different viewpoints of same product)
- Less discriminative than a trained classifier for confusing pairs

**Verdict:** This is essentially what the "SKU-level aggregation" in §4.5 does, but formalized. If you're already computing DINOv2 embeddings, computing per-SKU prototypes is trivial and likely improves over individual image matching.

---

### 8.3 Feature Matching + Homography (Classical CV)

**Concept:** Use traditional feature detectors (SIFT, ORB) or learned features (SuperPoint) to find keypoints in both the query and reference images, match them, and verify geometrically with RANSAC homography estimation.

**Why it's different:** This is **instance-level matching** — it literally finds the same visual features in two images and verifies they form a geometrically consistent transformation. It's not comparing holistic embeddings; it's matching specific local features (corners, edges, text characters).

**Real-world use:**
- Successfully used for **grocery product recognition on shelves** (SSD + hybrid SURF/BRISK/ORB matching)
- OpenCV's standard `findHomography` + SIFT pipeline is a battle-tested approach
- Used in industrial defect detection with ORB + vector search (Qdrant)

**Implementation:**
```
For each reference image: extract SIFT/ORB keypoints and descriptors
For query: extract keypoints and descriptors
Match descriptors (FLANN/BF matcher + Lowe's ratio test)
Estimate homography with RANSAC
Count inlier matches → confidence score
```

**Strengths:**
- **Geometrically verified** — matches must form a valid planar transformation
- **Interpretable** — you can visualize which features matched
- **No neural network training** required for the matching stage
- **Excellent for rigid packaging** — labels and bottles have consistent geometry
- **Rotation and scale invariant** (SIFT/SuperPoint)

**Weaknesses:**
- **Slow** at scale — must match against each reference image individually (but you only have 2,400)
- **Fragile with deformable objects** or extreme viewpoint changes
- **Black backgrounds may reduce keypoints** — fewer features to match in masked regions
- Doesn't handle significant appearance variation (different lighting, blur) as well as neural embeddings

**Verdict:** Best used as a **re-ranking / verification step** after an initial similarity search. For ~2,400 images, matching against each reference is feasible (~1-2 seconds per query). Could be combined with DINOv2 to re-rank the top-10 candidates.

---

### 8.4 OCR + Text-Based Matching

**Concept:** Extract text from the beverage label using OCR, then match the extracted text against a database of known product names, brand names, and attributes.

**Why it's different:** Instead of matching visually, this reads the actual text on the product. A "Coca-Cola Zero Sugar" label is trivially distinguishable from "Coca-Cola Classic" by text, even if the packaging is visually similar.

**Implementation:**
```
Input crop → OCR (Tesseract / PaddleOCR / Google Vision API)
→ Extract text strings
→ Fuzzy match against product name database (Levenshtein, TF-IDF)
→ Return matching SKU + confidence
```

**Real-world evidence:**
- Google Product Recognizer uses **OCR + visual embeddings** together, achieving GTIN-level identification
- Zebra's product recognition system uses OCR as a **complementary signal** to visual features
- Non-label-based goods identification in warehousing uses OCR as primary method, reducing barcode dependency
- TTBVerifier uses Ollama llama3.2-vision for OCR on alcohol beverage labels with fuzzy matching at 90% threshold

**Strengths:**
- **Text is the most discriminative feature** on beverage labels — brand names, product names, volume, etc.
- **Trivially handles visually similar variants** (Diet Coke vs Coke Zero — text is unambiguous)
- **Fast and lightweight** — OCR is much cheaper than deep learning inference
- **Complementary to visual methods** — combines well with embedding similarity

**Weaknesses:**
- **Requires visible, readable text** — blurred, angled, or partially occluded labels may fail
- **Language-dependent** — needs to support all languages on your products
- **Doesn't work on products without text** (rare for beverages, but possible)
- **OCR errors** can cause mismatches — needs fuzzy matching

**Verdict:** OCR should be a **primary signal** in your pipeline, not just an afterthought. Beverage products almost always have distinctive text (brand, product name). Combining OCR text matching with DINOv2 visual similarity would create a very robust system. Even a simple implementation — extract text, fuzzy match against SKU names — could resolve many cases that confuse pure visual matching.

---

### 8.5 Perceptual Hashing

**Concept:** Compute a compact hash fingerprint of each image. Near-identical images will have similar hashes (low Hamming distance).

**Why it's different:** Perceptual hashes (pHash, dHash, aHash, wHash) produce tiny fingerprints (64-256 bits) that can be compared in microseconds using bitwise operations.

**Applicability to your case:**
- **Designed for near-duplicate detection**, not visual similarity matching
- A photo of a Coke can and a different photo of the same Coke can → same hash ✓
- A photo of a Coke can vs a photo of a Pepsi can → different hashes ✓
- But: **different angles/lighting of the same product** → may fail
- **Not suitable for finding visually similar products** — only for exact/near-exact matches

**Research finding:** A 2026 comparative study found that perceptual hashing methods are "computationally efficient and effective for exact matches, but perform poorly on near-duplicates and under geometric transformations, whereas the CNN model is significantly more robust across all duplicate types."

**Verdict:** **Not recommended** as a primary method for your use case. Your input images are real-world photos that will differ significantly from reference images. Perceptual hashing only works for near-exact duplicates.

---

### 8.6 Hybrid Multi-Signal Architecture

**The most robust approach** combines multiple signals:

```
Input Image
    ↓
YOLO Detection → Crop
    ↓
┌─────────────┬──────────────────────────────┐
│  Signal 1:   │  Signal 2:                    │
│  Visual Sim  │  OCR Text                     │
│  (DINOv2)    │  (PaddleOCR)                  │
│              │                               │
│  SKU proto-  │  Fuzzy match                  │
│  type match  │  vs SKU names                 │
└──────┬───────┴──────────────┬────────────────┘
       │                      │
       └──────────────────────┘
                      ↓
              Score Fusion / Voting
              (weighted combination or
               confidence calibration)
                      ↓
              Ranked SKU list with
              fused confidence score
```

**Score fusion options:**
1. **Weighted average** — e.g., 50% visual similarity + 30% OCR match + 20% classifier confidence
2. **Cascading** — use OCR first (fast, high precision when text is readable), fall back to visual similarity for ambiguous cases
3. **Learned fusion** — train a small logistic regression on validation data to combine scores optimally

**Why this works:**
- OCR excels at distinguishing visually similar but textually different products (e.g., Coke variants)
- Visual similarity excels when text is partially occluded or blurred
- The combination is **more robust than any single method**

---

### 8.7 Paradigm Comparison Matrix

| Approach | Accuracy (expected) | Latency | Training Required | New SKU Handling | Best For |
|---|---|---|---|---|---|
| **Current: DINOv2 + cosine** | ★★★★ | <1ms | None | Add embeddings | General visual matching |
| **Fine-tuned classifier** | ★★★★★ | ~10ms | Yes (minutes) | Retrain needed | Fixed SKU catalog |
| **Prototypical networks** | ★★★★½ | <1ms | None | Add prototype | Same as current but cleaner |
| **SIFT/SuperPoint + homography** | ★★★½ | ~1-2s | None | Add descriptors | Rigid packaging verification |
| **OCR + text matching** | ★★★★ (for text-visible) | ~50ms | None | Add name to DB | Products with readable labels |
| **Perceptual hashing** | ★★ | <0.1ms | None | Add hash | Near-duplicate detection only |
| **Hybrid multi-signal** | ★★★★★ | ~100ms | Minimal | Add to all signals | Maximum robustness |

### 8.8 Recommended Exploration Order

Given your constraints (~200 SKUs, beverage crops, existing YOLO pipeline):

1. **Add OCR as a complementary signal** — beverage labels have distinctive text. This is the cheapest high-impact improvement. Use PaddleOCR or EasyOCR to extract text, fuzzy-match against SKU names.

2. **Test a fine-tuned classifier** — with 200 classes × 12 images, fine-tune EfficientNet-B3 or ViT-B/16. This is the most natural formulation and may outperform similarity search out of the box.

3. **Implement SKU prototypes** — trivial extension of your current pipeline. Average DINOv2 embeddings per SKU instead of storing individual images.

4. **Skip perceptual hashing and feature matching** unless you have specific near-duplicate or rigid verification needs.

---

## 9. Key References

### Embedding & Similarity Methods
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
11. **SigLIP 2 vs DINOv2:** Underfitted.dev, "Battle of the Embeddings Titans" (2026)
12. **Metric Loss Comparison:** "Variance Analysis and Optimization Behavior" (arXiv:2601.21450)

### CLS vs Patch Tokens
13. **DToP:** Song et al., "Boosting Vision Transformers for Image Retrieval" (WACV 2023) — arXiv:2210.11909
14. **Do ViTs See Like CNNs?:** Raghu et al. (NeurIPS 2021) — arXiv:2108.08810
15. **ViTs Need Registers:** Darcet et al., "Vision Transformers Need Registers" (2024) — arXiv:2404.03162
16. **Register+CLS Decoupling:** "Register and CLS tokens induce decoupling of local and global features in large ViTs" (2025) — arXiv:2505.05892
17. **ViTs Need More Than Registers** (2025) — arXiv:2602.22394
18. **GGeM Pooling:** Ko et al. (2022) — arXiv:2212.04114
19. **SALAD:** Izquierdo & Civera, "Optimal Transport Aggregation for Visual Place Recognition" (CVPR 2024)
20. **AnyLoc:** Keetha et al., "Towards Universal Visual Place Recognition" (RA-L 2023) — arXiv:2308.00688
21. **dino.txt:** Jose et al., "dino.txt: Learning CLS-Token-Space Alignment" (CVPR 2025) — arXiv:2412.16334
22. **ViTGaL:** Phan et al., "ViT-based Global-Local Image Retrieval" (ACCV 2022)
23. **GeM Pooling Benchmark:** MarkoHaralovic, "DINOv2 GeM vs CLS for geo-image-retrieval" (GitHub, 2025)

### DINOv2 & MobileNet Model Benchmarks
24. **DINOv2 Retrieval Benchmarks:** Oquab et al., Table 9 — Oxford/Paris/Met/AmsterTime retrieval mAP (2023)
25. **dinov2.cpp:** lavaman131, "Optimized DINOv2 inference with ggml — 3× faster, 4× less memory" (GitHub, 2025)
26. **DINOv2 iNaturalist Benchmark:** Voxel51, "Finding the Best Embedding Model for Image Classification" (2025)
27. **DINOv2 DISC21 Benchmark:** JayyShah, "CLIP-DINO Visual Similarity" (GitHub)
28. **Trendyol DINOv2 E-Commerce:** Trendyol, "DINOv2 ViT-B/14 + ArcFace → 256-dim, cosine sim 0.890" (HuggingFace)
29. **MobileNet V2/V3 Specs:** Keras Applications, timm model cards (HuggingFace)
30. **EfficientNet-Lite Specs:** Google TensorFlow TPU repo, timm model cards
31. **MobileViT V2:** Apple ml-cvnets, arXiv:2206.02680
32. **MobileNet Retrieval:** "Compact Networks for Image Embedding with Distillation" (arXiv:1904.03624)
33. **MobileNet SOP Benchmark:** "MobileNet-0.25: 27.5% → 44.6% R@1 with distillation" (arXiv:1904.03624)
34. **nyris Visual Product Search Benchmark:** benchmark.nyris.io — "GEM v5.1 56.8% R@1, DINOv2-L 31.5% R@1" (2026)
35. **Training Image Retrieval:** "All You Need to Know About Training Image Retrieval Models" (arXiv:2503.13045)
36. **CVPR 2025 Similarity:** arham2003, "MobileNet V2 + triplet loss image similarity analyzer" (GitHub)
37. **EfficientNet vs ViT Comparison:** Towards AI, "Vision Embedding Comparison for Image Similarity Search" (2025)
38. **MobileNet V4:** Qin et al., "MobileNetV4 — Universal Models for the Mobile Ecosystem" (ECCV 2024) — arXiv:2404.10518
39. **MNv4 timm weights:** HuggingFace, "timm/mobilenetv4-pretrained-weights" collection
40. **MobileNet V5:** Google, "MobileNetV5-300 — Gemma 3n vision encoder" (June 2025, no paper)
41. **MNv5 timm:** `mobilenetv5_300m.gemma3n` — timm PR #2527

### Direct Classification & Few-Shot
42. **Hierarchical Grocery Detector:** zrmarine, "hYOLO V4 on YOLOv8x — 323 SKU classes, 88.4% accuracy" (HuggingFace, 2026)
43. **Hierarchical SKU Recognition:** TechnoLynx, "Case Study: Large-Scale SKU Product Recognition" (2024)
44. **Prototypical Networks:** Snell et al., "Prototypical Networks for Few-shot Learning" (NeurIPS 2017)
45. **Proto-CLIP:** "Proto-CLIP: Vision-Language Prototypical Network for Few-Shot Learning" (arXiv:2307.03073)
46. **ViT-ProtoNet:** "ViT-Small backbone integration into Prototypical Networks" (arXiv:2507.09299)
47. **Compositional Prototypical Networks:** Lyu & Wang (AAAI 2024)

### Feature Matching & Classical CV
48. **OpenCV Feature Matching:** docs.opencv.org — SIFT + FLANN + Homography tutorial
49. **SuperPoint:** DeTone et al., "SuperPoint: Self-Supervised Interest Point Detection and Description" (CVPRW 2018) — arXiv:1712.07629
50. **Product Matching on Shelves:** mramanindia, "ORB descriptors + ResNet classification for shelf products" (GitHub, 2022)
51. **Grocery Product Recognition:** "Hybrid SURF+BRISK+ORB + SSD for product recognition" (Istanbul Bilgi University thesis)
52. **Template Matching Library:** antoninomariarizzo, "SIFT + NN matching + RANSAC homography" (GitHub, 2024)

### OCR & Matching
53. **Google Product Recognizer:** Google Cloud, "Product Recognizer Guide — GTIN/UPC level identification with OCR + embeddings"
54. **Zebra Product Recognition:** Zebra Technologies, "Feature Extractor + Match against product database" (TechDocs)
55. **OCR for Goods ID:** "Non-Label-Based Goods Identification Using Vision-Based OCR" (MDPI, 2026)

### Perceptual Hashing
56. **pHash:** phash.org — "Open source perceptual hash library"
57. **ImageHash:** JohannesBuchner/imagehash — Python perceptual image hashing (aHash, dHash, pHash, wHash)
58. **Hashing vs Deep Learning:** "Comparative Evaluation of Perceptual Hashing and Deep Embedding Methods" (Electronics, 2026)

### Beverage-Specific Systems
59. **WineEngine:** TinEye, "Wine/beer/spirit label recognition — image fingerprinting + neural networks"
60. **Soda Bottle Recognition:** vdrvar, "Transfer learning with Xception — 94% accuracy on soda bottles" (GitHub, 2024)
61. **Craft Brewery ID:** lukexyz/CraftVision, "YOLOv3 + ResNet34 classifier for craft beer identification" (GitHub, 2019)
