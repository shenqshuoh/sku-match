# Cleanup Plan — sku-match v0.2.0

Generated: 2026-05-07

## Progress Legend
- [ ] Not started
- [~] In progress
- [x] Done
- [!] Skipped / won't fix (with reason)

---

## 1. Bugs

### 1.1 [x] matcher.py class-name lookup uses wrong class list
**Priority:** HIGH  
**File:** `src/matcher.py`  
**Problem:** YOLOE is initialized with `set_classes(BEVERAGE_CONTAINER_CLASSES)` (7 items), so `box.cls` indices are 0–6 (bottle, canned, etc.). But matcher.py looks up class names via `OBJECTS365_CLASSES[int(box.cls[0])]`, which is the full 365-class list — wrong mapping.  
**Fix:** Changed to use `BEVERAGE_CONTAINER_CLASSES[int(box.cls[0])]` with bounds check. Also removed unused `OBJECTS365_CLASSES` import and unused `import cv2`.  
**Completed:** 2026-05-07

### 1.2 [x] RecognitionService.recognize() blocks async event loop
**Priority:** HIGH  
**File:** `api/services/recognition.py`, `api/routes/recognition.py`  
**Problem:** `recognize()` was `async def` but contained sync CPU-bound + GPU-bound code, blocking the event loop.  
**Fix:** Changed `recognize()` to regular `def` (sync). Route handler now calls it via `asyncio.to_thread()`.  
**Completed:** 2026-05-07

---

## 2. Code Deduplication

### 2.1 [x] Extract shared utilities to `src/utils.py`
**Priority:** MEDIUM  
**Fix:** Created `src/utils.py` with `detect_device()` and `free_gpu_memory()`. Updated all consumers:
- `src/indexer.py` — removed `_detect_device()`, now imports from utils
- `src/core.py` — removed `_free_gpu_memory()`, `_detect_device` import, `import gc/torch`; now imports from utils
- `src/matcher.py` — removed `_free_gpu_memory()`, `_detect_device` import, `import gc/torch`; now imports from utils
- `scripts/crop_reference.py` — now imports `detect_device` from utils
- `src/yoloe2o365.py` — now imports `detect_device` from utils  
**Completed:** 2026-05-07

---

## 3. Dead Code & Stale Files

### 3.1 [x] Remove `src/classes/lvis_targets.py`
**Priority:** MEDIUM  
**Reason:** Not imported anywhere.  
**Completed:** 2026-05-07

### 3.2 [x] Remove stale files
**Priority:** MEDIUM  
**Deleted:**
- `TODO.md` — old pipeline candidate notes
- `tree_data_ref_raw.txt` — unknown purpose
- `index/embeddings_dinov2_vits14.npy`, `embeddings_dinov2_vitb14.npy`, `embeddings.npy` — stale numpy index
- `index/metadata_dinov2_vits14.json`, `metadata_dinov2_vitb14.json`, `metadata.json` — stale metadata  
**Completed:** 2026-05-07

### 3.3 [x] ~~Remove~~ Relocate mobileclip model files
**Priority:** MEDIUM  
**Investigation:** `mobileclip2_b.ts` is a TorchScript MobileCLIP text encoder needed by ultralytics YOLOE's `set_classes()`. The model's YAML config specifies `text_model: mobileclip2:b`, which triggers `MobileCLIPTS` to load the file. If not found locally, ultralytics downloads it from `github.com/ultralytics/assets` — blocked by GFW.  
**Fix:** Moved `mobileclip2_b.ts` to `models/` directory. Added `configure_ultralytics_weights()` to `src/utils.py` that sets `ultralytics.settings['weights_dir']` to the absolute path of `models/`. Called at startup in both `main.py` and `api/app.py`. No GitHub download needed at runtime.  
**Note:** `mobileclip_blt.ts` (572MB, older blt variant) was NOT needed — only `mobileclip2_b.ts` (the "b" variant) is used.  
**Deleted:** `mobileclip_blt.ts` (572MB, unused variant)  
**Completed:** 2026-05-07

---

## 4. Misleading Code

### 4.1 [x] Fix async defs in IndexManager
**Priority:** LOW  
**Fix:** Removed `async` from all 5 methods in `IndexManager`. Updated callers in `api/routes/goods.py` and `api/tasks.py` to use `asyncio.to_thread()`.  
**Completed:** 2026-05-07

---

## 5. Dependency Cleanup

### 5.1 [x] ~~Remove~~ Vendor `clip` as local dependency
**Priority:** MEDIUM  
**Initial action:** Removed `clip` thinking it was unused (no direct imports in our code).  
**Reverted:** `clip` IS required — ultralytics YOLOE's `set_classes()` → `get_text_pe()` → `ultralytics/nn/text_model.py` imports `clip` internally.  
**GFW issue:** `clip` is only available via `git+https://github.com/ultralytics/CLIP.git` — unreachable from China's GFW. `ghfast.top` mirror failed (not a git proxy, partial transfer errors).  
**Final fix:** Vendored `clip` as a local package at `vendor/clip_package/` with its own `pyproject.toml`. `pyproject.toml` uses `[tool.uv.sources]` to override: `clip = { path = "./vendor/clip_package" }`. No network access needed at install time.  
**Completed:** 2026-05-07

---

## 6. Minor Fixes

### 6.1 [x] Fix `src/yoloe2o365.py` hardcoded device
**Priority:** LOW  
**Fix:** Changed `device="mps"` to `device=detect_device()` using new shared utility.  
**Completed:** 2026-05-07

### 6.2 [x] Update `src/__init__.py` exports
**Priority:** LOW  
**Fix:** Added exports: `DINOv2Embedder`, `DINOv2Variant`, `SKUIndexer`, `SKUMatcher`, `detect_device`, `free_gpu_memory`.  
**Completed:** 2026-05-07

---

## 7. Documentation Updates

### 7.1 [x] Update AGENTS.md
**Priority:** MEDIUM  
**Fix:** Regenerated to reflect current architecture: API server, Chroma indexer, device auto-detection, correct file paths, code map, API endpoints table.  
**Completed:** 2026-05-07

### 7.2 [x] Update PLAN.md
**Priority:** MEDIUM  
**Fix:** Updated status header to "IMPLEMENTED", updated current state section, added status column to fix table marking all items done.  
**Completed:** 2026-05-07

---

## 8. Post-Cleanup Issues Found

### 8.1 [x] Fix `_detect_device` import in `api/app.py` and `scripts/build_index.py`
**Priority:** HIGH  
**Problem:** Fixers removed `_detect_device` from `src/indexer.py` but `api/app.py` and `scripts/build_index.py` still imported it from there.  
**Fix:** Changed imports to `from src.utils import detect_device as _detect_device`.  
**Completed:** 2026-05-07

---

## 9. Feature Enhancements

### 9.1 [x] Create ReferenceProcessor module
**Priority:** HIGH  
**Files:** `src/reference_processor.py` (new), `api/services/index_manager.py`, `api/app.py`  
**Change:** New `ReferenceProcessor` class unifies crop→embed→index pipeline:
- `crop_reference()`: YOLOE detect → select best detection (center-preferring) → mask-isolated crop
- `process_and_add()`: crop → embed → add to Chroma, with full-image fallback
- `build_from_directory()`: batch build from `data/references/{sku_id}/*.jpg`
- Delegates: delete_sku_references, delete_media_reference, set_sku_enabled, update_sku_name
- `IndexManager` now a thin adapter wrapping `ReferenceProcessor`
- `main.py` auto-builds index if collection is empty  
**Completed:** 2026-05-07

### 9.2 [x] Implement softmax probability distribution
**Priority:** HIGH  
**Files:** `src/indexer.py`, `src/matcher.py`, `api/services/recognition.py`, `api/schemas.py`, `api/config.py`, `main.py`  
**Change:** `search_batch()` now returns softmax-normalized probability distribution over all enabled SKUs:
- Top-2-per-SKU cosine similarity sum → softmax with temperature=0.5
- Unranked SKUs get epsilon score (1e-8)
- `match_conf` threshold now operates on probability (0–1 range), default 0.5
- `search_multiplier` param (default 1) replaces hardcoded multiplier  
**Completed:** 2026-05-07

### 9.3 [x] Add concentration score and detection confidence to matching
**Priority:** HIGH  
**Files:** `src/types.py`, `src/matcher.py`, `src/indexer.py`, `main.py`, `api/schemas.py`, `api/services/recognition.py`, `api/config.py`, `api/app.py`  
**Change:**
- `SKUMatch.match_concentration`: top-1 share of top-K probability mass (replaces `match_ratio`)
- `concentration_score()` function in `src/indexer.py`: `top_scores[0] / sum(top_scores[:top_k])`
- `SKUMatcher.concentration_threshold`: gate for "unk" decision (default 0.0, disabled)
- `SKUMatcher.concentration_topk`: K for concentration calculation (default 10)
- `SKUMatch.top2_ranks`: 1-indexed rank positions of matched SKU's top-2 vectors in Chroma results
- `SKUMatcher.det_conf`: detection confidence threshold passed to YOLOE `predict()`
- `--conf` flag now works in matching mode (was only wired to detection-only mode)
- `--match-concentration` CLI flag (default 0.0), `--match-concentration-topk` (default 10)
- API: `CONCENTRATION_TOPK: int = 10` setting
- Verbose output: `detection n: conf=X.XX score=X.XXX conc=X.XX, top 2 at (a, b)`  
**Completed:** 2026-05-08

### 9.4 [x] Embedding model research
**Priority:** MEDIUM  
**Findings:** Evaluated 20+ visual embedding models across self-supervised (DINOv2), CLIP-family (SigLIP, OpenCLIP, EVA02, DFN), lightweight (MobileCLIP), and domain-specific (Marqo-Ecommerce, Trendyol DINOv2-Ecom) categories.  
**Key finding:** Current DINOv2 ViT-B/14 is already the best general-purpose choice. Highest-ROI upgrade is domain adaptation: freeze DINOv2 backbone + train ArcFace projection head (768→256 dim) on actual SKU data. The ViT forward pass (bottleneck) remains unchanged — speed gain is minimal, only from smaller vector dimensions in search.  
**Noted in:** PLAN.md Section 11  
**Completed:** 2026-05-07

---

## 10. Performance Optimizations

### 10.1 [x] FP16 inference for YOLOE + DINOv2
**Priority:** HIGH  
**Files:** `src/embedder.py`, `src/matcher.py`, `api/app.py`  
**Change:** `model.half()` at load time when `device=="cuda"`. Input tensors also half. ~1.8x speedup (5.79s vs 8.71s for 2 images).  
**Completed:** 2026-05-08

### 10.2 [x] Fix retina_masks OOM in API
**Priority:** HIGH  
**Files:** `api/services/recognition.py`  
**Change:** `retina_masks=True` → `retina_masks=False`. Prevents 672MB allocation for full-res masks.  
**Completed:** 2026-05-08

### 10.3 [x] Dedicated GPU thread pool
**Priority:** MEDIUM  
**Files:** `api/routes/recognition.py`, `api/app.py`  
**Change:** `ThreadPoolExecutor(max_workers=1)` for GPU inference. Prevents concurrent GPU access under load.  
**Completed:** 2026-05-08

### 10.4 [x] Pass numpy array to YOLOE
**Priority:** MEDIUM  
**Files:** `api/services/recognition.py`  
**Change:** `source=image_np` instead of re-opening file. ~100ms saved per request.  
**Completed:** 2026-05-08

### 10.5 [x] Union mask O(D²)→O(D)
**Priority:** MEDIUM  
**Files:** `api/services/recognition.py`, `src/matcher.py`  
**Change:** Pre-compute union of all masks once, per-detection exclusion = union - own_mask.  
**Completed:** 2026-05-08

### 10.6 [x] YOLOE warm-up on startup
**Priority:** MEDIUM  
**Files:** `api/app.py`  
**Change:** Dummy predict on 640×640 zeros after set_classes(). Eliminates cold start.  
**Completed:** 2026-05-08

### 10.7 [x] ONNX Runtime (behind --onnx flag)
**Priority:** MEDIUM  
**Files:** `src/embedder.py`, `pyproject.toml`, `scripts/export_onnx.py` (new)  
**Change:** Dual-mode embedder with `use_onnx: bool = False` param. Export script for DINOv2. NOT viable on 2GB GPU (ORT lacks flash attention → 687MB for batch=16). Future option for ≥4GB GPUs.  
**Completed:** 2026-05-08

### 10.8 [x] JPEG quality on annotated output
**Priority:** LOW  
**Files:** `src/image_utils.py`  
**Change:** `cv2.imwrite()` with `[cv2.IMWRITE_JPEG_QUALITY, 85]`.  
**Completed:** 2026-05-08

### 10.9 [x] Default embedding model changed to vits14
**Priority:** MEDIUM  
**Files:** `api/config.py`, `main.py`  
**Change:** Default `dinov2_vits14` (384-dim, ~85MB) instead of `dinov2_vitb14` (768-dim, ~330MB). Better fit for 2GB GPU alongside YOLOE.  
**Completed:** 2026-05-08

### 10.10 [x] DINOv2 vendoring for GFW
**Priority:** HIGH  
**Files:** `src/embedder.py`, `vendor/dinov2/` (new), `models/dinov2_*_pretrain.pth` (new)  
**Change:** Full dinov2 source vendored at `vendor/dinov2/`. Weights at `models/dinov2_vits14_pretrain.pth` (84MB) and `models/dinov2_vitb14_pretrain.pth` (330MB). Embedder loads from local paths, falls back to internet if weight missing. No GitHub/fbpublicfiles download needed.  
**Completed:** 2026-05-08

---

## Changelog

| Date | Action |
|------|--------|
| 2026-05-07 | Plan created |
| 2026-05-07 | Completed items 1.1, 1.2, 2.1, 3.1–3.3, 4.1, 5.1, 6.1–6.2 |
| 2026-05-07 | Completed items 7.1, 7.2, 8.1 — ALL DONE |
| 2026-05-07 | **Reverted** 5.1: restored `clip` dep — ultralytics requires it internally for YOLOE `set_classes()` |
| 2026-05-07 | **Vendored** `clip` to `vendor/clip_package/` — eliminates git dependency, GFW-safe |
| 2026-05-07 | **Relocated** `mobileclip2_b.ts` to `models/`, added `configure_ultralytics_weights()` — avoids GitHub download at runtime |
| 2026-05-07 | Completed items 9.1–9.4: ReferenceProcessor, softmax scoring, match ratio, det_conf, embedding model research |
| 2026-05-08 | Completed items 10.1–10.10: FP16, retina_masks fix, GPU thread pool, numpy array, union mask, warm-up, ONNX (behind flag), JPEG quality, vits14 default, DINOv2 vendoring |
| 2026-05-08 | Replaced match_ratio with concentration_score (item 9.3) |
| 2026-05-29 | Plan complete — all items done. Further improvements tracked in `docs/PLAN.md` (roadmap) and `docs/AUDIT.md` (issues). |
