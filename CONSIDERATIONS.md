# Considered Improvements & Modules

Evaluated additions to the beverage-cashier pipeline. Updated as new options are researched.

---

## Detection Improvements

### SAHI (Slicing Aided Hyper Inference)

**What**: Splits large images into overlapping tiles, runs detection on each, then merges results with NMS. Improves small object recall.

**Status**: ❌ Deferred

| Pros | Cons |
|------|------|
| Native YOLOE support (`model_type='yoloe'`) via SAHI | ~3-5x slower per image (multiple inferences + merge) |
| Works with `set_classes()` and segmentation masks | Bottles/cans are medium-large objects in tabletop photos — not the use case SAHI targets |
| Drop-in via `get_sliced_prediction()` | Adds `sahi` dependency (~50MB) |
| Proven for aerial/drone/satellite imagery | Increasing `imgsz` or lowering `conf` is cheaper for our scale |

**When to reconsider**: Store shelf photos from far away, densely packed items, 4K+ images where objects appear small.

**Reference**: `sahi/models/yoloe.py`, SAHI docs at https://obss.github.io/sahi/

---

## Matching & Retrieval

### FAISS (instead of Chroma)

**What**: Facebook's high-performance vector similarity search library.

**Status**: ❌ Evaluated, not chosen (Chroma selected instead)

| Pros | Cons |
|------|------|
| Fastest search (2ms for 1M vectors on GPU) | No built-in persistence — must build storage layer manually |
| Most memory-efficient (300MB for 100K vectors) | No metadata filtering — must implement manually |
| GPU acceleration available | No incremental updates — full rebuild required on SKU changes |
| Battle-tested at billion scale | Overkill for 100-2000 vectors |

**When to reconsider**: Scale exceeds 100K+ reference vectors, sub-millisecond latency required.

---

### Qdrant (instead of Chroma)

**What**: Rust-based vector database with advanced filtering.

**Status**: ⏳ Migration target if Chroma has issues

| Pros | Cons |
|------|------|
| 2-3x faster queries than Chroma | Requires Docker or separate server process (not embedded) |
| Best filtered search performance (55ms vs 520ms at 10M vectors) | More ops complexity for single-server deployment |
| Reliable persistence, no memory leaks | Heavier setup than `pip install chromadb` |
| Single-binary deployment | |

**When to migrate**: Chroma memory leaks become problematic, scale exceeds 10K vectors, need 99.9%+ uptime without restarts.

---

## Training & Model Improvements

### Single-SKU Fine-Tuning

**What**: Fine-tune DINOv2 embedding model on new angles/variants of a specific SKU.

**Status**: ❌ Deferred (API.md requirement, future scope)

| Pros | Cons |
|------|------|
| Better matching for visually similar SKUs | Requires GPU training pipeline |
| Handles packaging variations (e.g., limited editions) | Risk of catastrophic forgetting without careful setup |
| Improves with more data organically | Adds complexity to SKU creation workflow |

**When to implement**: When false-positive rate between similar SKUs becomes a business problem.

---

### YOLOE Detection Fine-Tuning

**What**: Fine-tune the detection model on beverage-specific data for better bbox/mask quality.

**Status**: ❌ Deferred

| Pros | Cons |
|------|------|
| Better detection of partially occluded bottles | Requires labeled training data (bbox + masks) |
| Reduces false negatives on tricky angles | Training requires GPU + time |
| Can add custom classes (e.g., "thermos", "growler") | May lose generalization on other container types |

**When to implement**: Detection recall drops below acceptable threshold on real-world photos.

---

### Continuous Training from Corrections

**What**: Feed `/fix` endpoint corrections back into model weights automatically.

**Status**: ❌ Deferred (API.md requirement, future scope)

| Pros | Cons |
|------|------|
| Self-improving system | Complex pipeline: corrections → dataset → training → deployment |
| Reduces manual tuning over time | Risk of feedback loops (bad corrections → worse model) |
| Builds proprietary training data | Needs monitoring and guardrails |

**When to implement**: After API server is stable and correction volume justifies automation.

---

## API & Infrastructure

### Video Recognition (`mode: VIDEO`)

**What**: Process video files for frame-by-frame detection + temporal deduplication.

**Status**: ❌ Deferred (API.md mentions it)

| Pros | Cons |
|------|------|
| Real-time counting for conveyor belt scenarios | Frame extraction + deduplication adds complexity |
| Temporal context reduces false positives | GPU memory for video decoding + inference |
| Natural extension of image pipeline | Storage for video frames grows fast |

**When to implement**: Video use case is specified by the business.

---

### Multi-Photo Merge (同一场景多角度)

**What**: Combine detections from multiple photos of the same scene (different angles) into a unified count.

**Status**: ❌ Deferred (API.md mentions it)

| Pros | Cons |
|------|------|
| Higher confidence via multi-view consensus | Needs scene grouping logic (how to know photos are same scene?) |
| Reduces occlusion-based misses | Cross-image deduplication of duplicate detections |
| More accurate total counts | API design needs thought (batch of images per request?) |

**When to implement**: Business requires multi-angle counting accuracy.

---

### OSS Result Image Hosting

**What**: Upload annotated result images to object storage (Alibaba OSS) instead of serving locally.

**Status**: ⏳ Design decision needed

| Pros | Cons |
|------|------|
| Scales horizontally (multiple API instances) | External dependency (OSS availability) |
| No local disk management | Adds latency for upload |
| CDN-cacheable URLs for client | Configuration complexity (credentials, buckets) |

**When to implement**: Multi-instance deployment or disk space becomes an issue.

---

## Currently Implemented

| Module | Status | Notes |
|--------|--------|-------|
| Chroma vector store | ✅ Done | Replaced numpy-based index. Section 10 of PLAN.md |
| Device auto-detect (cuda→mps→cpu) | ✅ Done | All modules default to auto-detect |
| `sku_name` in data model | ✅ Done | Added to SKUReference, SKUMatch, indexer metadata |
| SSL fix (macOS guard) | ✅ Done | embedder.py |
| `clip` dep removal | ✅ Done | Dead dependency |
| `coremltools` moved to optional | ✅ Done | `[export]` group |

---

*Last updated: 2026-04-27*
