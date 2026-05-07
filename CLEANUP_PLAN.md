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

### 9.3 [x] Add match ratio and detection confidence to matching
**Priority:** HIGH  
**Files:** `src/types.py`, `src/matcher.py`, `main.py`  
**Change:**
- `SKUMatch.match_ratio`: top-1/top-2 probability ratio (disabled by default, threshold 0.0)
- `SKUMatch.top2_ranks`: 1-indexed rank positions of matched SKU's top-2 vectors in Chroma results
- `SKUMatcher.det_conf`: detection confidence threshold passed to YOLOE `predict()`
- `--conf` flag now works in matching mode (was only wired to detection-only mode)
- `--match-ratio` CLI flag (default 0.0)
- Verbose output: `detection n: conf=X.XX score=X.XXX ratio=X.XX, top 2 at (a, b)`  
**Completed:** 2026-05-07

### 9.4 [x] Embedding model research
**Priority:** MEDIUM  
**Findings:** Evaluated 20+ visual embedding models across self-supervised (DINOv2), CLIP-family (SigLIP, OpenCLIP, EVA02, DFN), lightweight (MobileCLIP), and domain-specific (Marqo-Ecommerce, Trendyol DINOv2-Ecom) categories.  
**Key finding:** Current DINOv2 ViT-B/14 is already the best general-purpose choice. Highest-ROI upgrade is domain adaptation: freeze DINOv2 backbone + train ArcFace projection head (768→256 dim) on actual SKU data. The ViT forward pass (bottleneck) remains unchanged — speed gain is minimal, only from smaller vector dimensions in search.  
**Noted in:** PLAN.md Section 11  
**Completed:** 2026-05-07

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
