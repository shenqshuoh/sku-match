# Performance Improvement Plan

**Created:** 2026-05-07
**Updated:** 2026-05-29
**GPU:** NVIDIA L20-2Q, 2048 MiB VRAM

## Benchmarks

| Mode | Image 1 (14 det) | Image 2 (15 det) | Total |
|------|---------|---------|-------|
| ONNX CPU (broken, both models loaded) | 8.65s | 8.69s | 20.85s |
| PyTorch FP32 | 2.44s | 2.50s | 8.71s |
| PyTorch FP16 | 1.40s | 1.07s | 5.79s |

FP16 is ~1.8x faster than FP32. Default mode is PyTorch FP16.

## Items

### 1. FP16 Inference — HIGH ✅
- **Status:** DONE
- **Files:** `src/embedder.py`, `src/matcher.py`, `api/app.py`
- **Change:** `model.half()` at load time (after `.to(device)`) when `device=="cuda"`. Both YOLOE and DINOv2. Input tensors also converted to half before forward pass.
- **Note:** `half=True` in `predict()` was tried but causes dtype mismatch in ultralytics segmentation mask processing. Using `model.half()` at setup time works correctly.
- **Result:** ~1.8x speedup over FP32 (5.79s vs 8.71s for 2 images)

### 2. Fix `retina_masks=True` → `False` everywhere — HIGH ✅
- **Status:** DONE
- **Files:** `api/services/recognition.py`, `src/core.py`, `src/reference_processor.py`
- **Change:** `retina_masks=False` in all `predict()` calls (API recognition, CLI detect, reference crop)
- **Reason:** `retina_masks=True` allocates tensor at original image resolution (e.g., 4000×3000 = 672MB) → OOM on 2GB GPU

### 3. Dedicated GPU thread pool (max_workers=1) for API — MEDIUM ✅
- **Status:** DONE
- **Files:** `api/routes/recognition.py`, `api/app.py`
- **Change:** `ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu_inference")` stored on `app.state.inference_executor`. Route uses `loop.run_in_executor()` instead of `asyncio.to_thread()`. Executor shutdown in lifespan cleanup.
- **Reason:** Default thread pool allows concurrent GPU access → OOM under load
- **How it works:** FastAPI runs request handlers as async coroutines on the event loop. GPU inference (`recognize()`) is CPU/CUDA-bound and would block the event loop if called directly. `asyncio.to_thread()` offloads it to the default `ThreadPoolExecutor`, but that pool is shared across all async tasks and has `max_workers=min(32, os.cpu_count()+4)`. With `max_workers=1`, only one inference job runs at a time — requests queue in the executor and are processed sequentially. This prevents concurrent CUDA memory allocations that would exceed 2GB VRAM. The async event loop remains free to handle HTTP connections while waiting for GPU results.

### 4. Pass numpy array to YOLOE (avoid double image I/O) — MEDIUM ✅
- **Status:** DONE
- **Files:** `api/services/recognition.py`, `src/reference_processor.py`
- **Change:** `source=image_np` instead of `source=str(image_path)` — image already loaded as numpy. Applied to both recognition service and reference processor (audit #20).

### 5. Pre-compute union mask O(D²)→O(D) — MEDIUM ✅
- **Status:** DONE
- **Files:** `api/services/recognition.py`, `src/matcher.py`
- **Change:** Pre-compute union of all masks once with `cv2.bitwise_or`, then per-detection exclusion = `cv2.bitwise_and(union_mask, cv2.bitwise_not(own_mask))`

### 6. YOLOE warm-up inference on startup — MEDIUM ✅
- **Status:** DONE
- **Files:** `api/app.py`
- **Change:** Dummy `predict()` on 640×640 zeros after `set_classes()` — eliminates 1-3s cold start

### 7. ONNX Runtime for DINOv2 — MEDIUM ⚠️
- **Status:** DONE (code exists, behind `--onnx` flag / `USE_ONNX` env, NOT viable on 2GB GPU)
- **Files:** `src/embedder.py`, `pyproject.toml`, `scripts/export_onnx.py` (new)
- **Change:** Dual-mode embedder. `use_onnx: bool = False` param. ONNX only activated when `use_onnx=True`. ONNX deps moved to optional `[onnx]` group in pyproject.toml.
- **How it works:** DINOv2 runs in two modes: (1) **PyTorch** — loads the `.pth` weights via torchvision, runs `model.forward()` on GPU with FP16 tensors. (2) **ONNX** — loads a pre-exported `.onnx` graph via `onnxruntime.InferenceSession` with CUDA execution provider. Both modes accept the same input (batch of normalized image tensors) and return the same output (L2-normalized embedding vectors). The ONNX path skips PyTorch's Python dispatch overhead and can fuse ops, but ONNX Runtime lacks flash attention — it materializes the full N×N attention matrix in VRAM (687MB for batch=16 at dim=384), which causes OOM on 2GB. With ≥4GB VRAM, the ONNX path is expected to be faster.
- **Verdict:** Not viable on current 2GB GPU. See "Future: GPU Upgrade Path" section for plan.

### 8. JPEG quality on annotated output — LOW ✅
- **Status:** DONE
- **Files:** `src/image_utils.py`
- **Change:** `cv2.imwrite(...)` with `[cv2.IMWRITE_JPEG_QUALITY, 85]`

## Not Implemented (Deferred)

| Item | Reason |
|------|--------|
| ~~Reduce imgsz from 1280 → 640~~ | **Rejected.** 1280 resolution is necessary for detection accuracy — small/cans and distant bottles are missed at 640. |
| HTTP client connection reuse | Deferred. httpx.AsyncClient created per-download in ImageStorage; reusing a single client avoids TCP handshake overhead |
| Async image download during GPU inference | Deferred. Download next image while current one is being processed (pipeline overlap) |
| Embedding cache for repeated queries | Deferred. Same image sent multiple times re-embeds from scratch; could cache by image hash |

## Potential: Chroma HNSW Parameter Tuning

Chroma uses **HNSW (Hierarchical Navigable Small World)** for approximate nearest-neighbor search. HNSW builds a multi-layer graph where each vector is a node connected to its nearest neighbors. At query time, it starts from a random entry point and greedily traverses the graph toward the query vector, finding approximate top-K results in sub-linear time.

Key parameters in the current configuration (`src/indexer.py`):

- **`space: "cosine"`** — Distance metric. Cosine similarity is correct for DINOv2 embeddings (L2-normalized vectors). No change needed.
- **`ef_construct`** (default: 128 in Chroma) — Controls graph quality during index build. Higher = more accurate graph but slower inserts and more memory. Our inserts happen at SKU creation time (infrequent), so accuracy during build matters more than speed. Current default is likely fine, but increasing to 256 could improve recall for large catalogs.
- **`ef_search`** (default: 10 in Chroma) — Controls search quality at query time. Higher = more accurate results but slower search. With ~10-50 SKUs currently, search is already fast (~1ms). If the catalog grows to 1000+, tuning this to 64-128 would maintain recall.
- **`M`** (default: 16 in Chroma) — Max connections per node per layer. Higher = better recall, more memory. Default is sufficient for current scale.

**When to tune:** If the SKU catalog exceeds 500+ items and match accuracy degrades, increasing `ef_search` is the first knob to turn. For current scale (< 50 SKUs), defaults are optimal.

## Future: GPU Upgrade Path

The current GPU (NVIDIA L20-2Q, 2GB VRAM) is the main performance bottleneck. Model size (~100MB combined for YOLOE + DINOv2) is negligible — both fit comfortably in VRAM with room to spare. The constraint is *runtime memory* (attention tensors, intermediate feature maps), not model weights.

**With a GPU ≥4GB VRAM**, the recommended setup is:
- **YOLOE**: Continue using PyTorch FP16 (fast, stable, well-tested)
- **DINOv2**: Switch to ONNX Runtime with CUDA execution provider. ONNX Runtime avoids PyTorch's Python overhead and can fuse operations, but it lacks flash/memory-efficient attention — the full attention matrix is materialized (687MB for batch=16), which causes OOM on 2GB VRAM. With ≥4GB, this becomes viable and should provide meaningful speedup.

Enable by setting `USE_ONNX=true` (env) or `--onnx` (CLI). No code changes needed — the dual-mode embedder (`src/embedder.py`) already supports it.

> Note: Exporting YOLOE to ONNX is **not planned**. Ultralytics' segmentation models have complex post-processing that doesn't export cleanly. The performance gain would be marginal compared to the DINOv2 ONNX switch.

## Changelog

| Date | Items | Status |
|------|-------|--------|
| 2026-05-07 | Items 1-8 implemented | All done |
| 2026-05-08 | Updated: #1 changed to model.half() approach, #7 marked not viable on 2GB GPU, benchmarks added | — |
| 2026-05-09 | Updated: #2 expanded to all predict() calls, #7 noted optional dep group, added deferred items, GPU upgrade path | Current |
| 2026-05-29 | Updated: #4 expanded to reference_processor.py, updated date | Current |
