# sku-match Roadmap & Future Plans

> **Updated**: 2026-05-29
> **Status**: API server fully implemented. All audit items resolved. This document tracks future work.

---

## Current State

The sku-match API server is **fully operational** with the following capabilities:

- **Detection**: YOLOE with 7 beverage container classes, FP16 inference on CUDA
- **Matching**: DINOv2 embeddings + top-2-per-SKU cosine similarity against Chroma vector store
- **API**: 10 REST endpoints (recognition, SKU CRUD, logs, system), FastAPI + SQLAlchemy async
- **Security**: API key auth + per-IP rate limiting (disabled by default)
- **Infrastructure**: Qiniu CDN upload for annotated images, async task runner for embedding jobs, background cleanup of old results
- **Deployment**: systemd service, FP16 + ONNX backend options, auto device detection (cuda → mps → cpu)

---

## Near-Term: Security & Hardening

Production readiness items from the improvement plan. These should be implemented before any public-facing deployment.

### SSRF Protection

- **File**: `api/services/image_storage.py`
- **What**: Block internal/private IPs in `download_image()` URL resolver
- **Blocked ranges**: `127.0.0.0/8`, `169.254.0.0/16`, `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`, `[::1]`, `0.0.0.0`
- **Also**: Reject `file://` scheme explicitly

### Path Traversal Prevention

- **File**: `api/services/image_storage.py`
- **What**: Restrict local file paths to whitelisted directories (`data/`, `results/`)
- **Block**: Absolute paths and `..` traversal sequences

### Error Information Leak

- **Files**: All API route handlers
- **What**: Return generic error messages to API clients, log full `str(e)` server-side
- **Replace**: `msg=str(e)` → `msg="Internal error"` + `logger.error(...)` pattern

### File Type Validation

- **File**: `api/services/image_storage.py`
- **What**: Validate downloaded content is actually an image
- **Methods**: Check `Content-Type` header, verify magic bytes, reject non-image extensions

### Unclosed PIL Images

- **Files**: `src/embedder.py`, `src/matcher.py`, `src/reference_processor.py`, `api/services/recognition.py`
- **What**: Use `with Image.open(...) as img:` pattern or explicit `img.close()` after use
- **Impact**: Prevents gradual memory leak in long-running API processes

### Engine Disposal on Shutdown

- **File**: `api/app.py` (lifespan shutdown)
- **What**: Add `await engine.dispose()` to cleanly close SQLAlchemy connection pool on server stop

### Graceful Shutdown

- **File**: `api/app.py` (lifespan shutdown)
- **What**: Change `shutdown(wait=False)` → `shutdown(wait=True, cancel_futures=True)` with timeout
- **Impact**: In-flight GPU inference completes or is cancelled cleanly, preventing corrupted state

---

## Medium-Term: API Improvements

Response model and API contract improvements for better client integration.

### HTTP Status Codes

- **Files**: All route files
- **What**: Use proper HTTP status codes instead of always returning 200 with `code=0/1`
  - `200`: Success
  - `400`: Validation error (invalid input)
  - `404`: Not found (skuId, taskId, trainJobId)
  - `409`: Conflict (duplicate skuId, taskId)
  - `500`: Internal server error

### Typed Response Models

- **Files**: `api/schemas.py`, `api/services/recognition.py`
- **What**: Use `DetectionItem` and `DetectData` as `response_model` on detect endpoint
- **Also**: Recognition service should return typed data instead of raw dict

### Typed ApiResponse.data

- **File**: `api/schemas.py`
- **What**: Replace `data: Any` with typed variants — either `data: dict | list | None` or generic `ApiResponse[T]`

---

## Feature Roadmap

### Detection Improvements

| Feature | Description | Trigger to Implement |
|---------|-------------|---------------------|
| **SAHI** (Slicing Aided Hyper Inference) | Split large images into tiles, detect per-tile, merge with NMS. Native YOLOE support. ~3-5x slower per image. | Store shelf photos from far away, densely packed items, 4K+ images where objects appear small |
| **YOLOE Detection Fine-Tuning** | Fine-tune detection model on beverage-specific data for better bbox/mask quality. Can add custom classes (thermos, growler). | Detection recall drops below acceptable threshold on real-world photos |

### Matching & Retrieval

| Feature | Description | Trigger to Implement |
|---------|-------------|---------------------|
| **FAISS / Qdrant migration** | Replace Chroma with higher-performance vector DB if scale exceeds current capacity. Qdrant: 2-3x faster queries, better persistence. FAISS: fastest brute-force at massive scale. | Chroma memory leaks become problematic, scale exceeds 10K+ vectors, need 99.9%+ uptime without restarts |
| **Multi-Scale Embedding** | Average embeddings from 3 crop scales (0% to 30% margin) for richer feature capture. +2-5% retrieval accuracy expected. | Single-scale matching insufficient for similar SKUs |
| **Patch-Level Features + GeM Pooling** | Use DINOv2 patch tokens with GeM pooling instead of CLS token for fine-grained discrimination of similar packaging. | Confused SKU pairs that CLS token cannot disambiguate |
| **Calibrated Confidence (Platt Scaling)** | Fit logistic regression on cosine similarity scores with labeled same/different pairs. Convert raw similarity to meaningful confidence. | Business requires interpretable confidence percentages |
| **Test Raw Crops Without Masking** | Evaluate whether raw YOLO crops (with natural background) perform better than black-masked crops. DINOv2 may handle background variation well. | Low effort, high potential gain — should test before any fine-tuning |
| **BGAugment (Background Replacement)** | Replace black backgrounds with random images during fine-tuning to force model focus on foreground. | If fine-tuning DINOv2 is implemented |

### Training & Model Improvements

| Feature | Description | Trigger to Implement |
|---------|-------------|---------------------|
| **Single-SKU Fine-Tuning** | Fine-tune DINOv2 with ArcFace or triplet loss on actual SKU image pairs. Projects to more discriminative embedding space. | False-positive rate between visually similar SKUs becomes a business problem |
| **Continuous Training from Corrections** | Feed `/fix` endpoint corrections back into model weights automatically. Self-improving system with guardrails against feedback loops. | API server is stable and correction volume justifies automation |
| **Fine-Tune DINOv2 with Triplet Loss** | Domain-specific fine-tuning with labeled beverage pairs. +5-15% expected gain. BGAugment recommended during training. | Out-of-box DINOv2 has unacceptable confusion between specific SKU pairs |

### New Features

| Feature | Description | Trigger to Implement |
|---------|-------------|---------------------|
| **Video Recognition** (`mode: VIDEO`) | Frame-by-frame detection + temporal deduplication for conveyor belt scenarios. Natural extension of image pipeline. | Video use case specified by business |
| **Multi-Photo Merge** | Combine detections from multiple photos of the same scene (different angles) into unified count. Requires scene grouping logic and cross-image deduplication. | Business requires multi-angle counting accuracy |

---

## When to Revisit Deferred Items

Items from the audit that were explicitly deferred. Each has a specific trigger condition.

| Item | Description | Severity | Trigger to Revisit | Risk if Ignored |
|------|-------------|----------|-------------------|-----------------|
| **#14** — Qiniu upload reads entire file into memory | `file_path.read_bytes()` loads full image (~5-20MB for 4K). Combined with concurrent ops, potential OOM. | Medium | Memory-constrained deployment or OOM events | Expand server memory; stream upload instead |
| **#15** — SSL verification disabled globally | `ssl._create_default_https_context` monkey-patched. Affects all HTTPS connections including Qiniu token/upload. | Medium | Production deployment with sensitive credentials | Implement per-request SSL context; use certifi bundle |
| **#23** — Mask normalization logic fragile | `normalize_mask` checks `if mask.max() <= 1` — works but relies on implicit assumptions about value ranges. | Low | YOLOE segmentation masks enabled in production (`retina_masks=True`) | Rewrite with explicit value range handling |
| **#25** — No DB migration strategy | `create_all()` only creates new tables, doesn't alter existing ones. Schema changes require manual migration. | Low | Any ORM model change (adding/removing/modifying columns) | Add Alembic migration support before any schema change |

---

## References

- **[AUDIT.md](AUDIT.md)** — Full codebase audit findings (25 issues, all resolved or deferred)
- **[CONSIDERATIONS.md](CONSIDERATIONS.md)** — Evaluated improvements (SAHI, FAISS, Qdrant, fine-tuning, video recognition)
- **[IMAGE_SIMILARITY_SEARCH_REPORT.md](IMAGE_SIMILARITY_SEARCH_REPORT.md)** — Deep research on embedding models, matching strategies, and optimization
- **[IMPLAN.md.bak](IMPLAN.md.bak)** — Archived improvement plan (security, API consistency, resource management)
- **[PERF_PLAN.md](PERF_PLAN.md)** — GPU performance optimization tracking
