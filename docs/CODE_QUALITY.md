# Code Quality Review — sku-match v0.2.0

## Executive Summary

The codebase is functional and well-organized at a surface level, with clean separation between `src/` (core ML) and `api/` (FastAPI server). A deep review previously revealed code duplication, module boundary violations, dead code, and best-practice issues. Most have been resolved.

---

## Fix Status

### ✅ Completed

| Issue | Section | Fix |
|-------|---------|-----|
| Detection result parsing duplicated 3x | §1.1 | Extracted `parse_detections()` in `src/core.py` |
| SKU matching pipeline duplicated | §1.2 | Extracted `score_matches()` in `src/indexer.py` |
| Embedding + upload + cleanup duplicated | §1.3 | Extracted `process_single_media()` in `api/tasks.py` |
| ReferenceProcessor CRUD delegation | §2.1 | Removed 4 pass-through methods; routes call indexer directly |
| SKUIndexer.load() dead alias | §2.2 / §4.6 | Removed; callers use `init_collection()` directly |
| src/\_\_init\_\_.py re-exports | §2.4 | Cleared — file is now empty |
| hasattr guards in goods.py | §2.5 | Removed — processor always set in lifespan |
| Routes access infrastructure directly | §3.1 | Routes use `Depends()` from `api/dependencies.py` |
| DIY service locator via app.state | §3.3 | Typed DI providers in `api/dependencies.py` |
| src/classes/objects365_classes.py dead code | §4.1 | Deleted |
| src/yoloe2o365.py dead code | §4.2 | Deleted |
| SKUMatch.crop_path never set | §4.3 | Removed from dataclass |
| SKUListItem never instantiated | §4.5 | Fixed — goods.py constructs proper Pydantic objects |
| Global SSL monkey-patch | §5.1 | Removed `disable_ssl_verification()` + download fallback. Hard `FileNotFoundError` on missing weights. |
| tempfile.mktemp() deprecated | §5.2 | Replaced with `NamedTemporaryFile(delete=False)` |
| Module-level side effects | §5.3 | `configure_ultralytics_weights()` moved into `main()` and `lifespan()` |
| asyncio.get_event_loop() deprecated | §5.4 | Replaced with `get_running_loop()` in all 3 sites |
| No response_model on routes | §5.5 | Added `response_model=ApiResponse` to all 10 route decorators |
| embedding_to_list() inline import | §5.7 | Moved `import numpy` to module level |
| Mask extraction logic scattered | — | Created `src/masking.py` with configurable background color |
| train_status incorrect on partial failure | — | `SUCCESS` on zero failures, `FAILED: a/b` on partial failure, `FAILED` on exception |

### ⬜ Not Yet Addressed

| Issue | Section | Notes |
|-------|---------|-------|
| "No Detections" empty result copy-pasted | §1.4 | Low priority — only 2 sites |
| save_crop() one-liner wrapper | §2.3 | Low priority — harmless convenience |
| Detection.mask field never set | §4.4 | Kept for potential future use |

---

## Original Findings (reference)

### 1. Redundancy & Duplicated Code

#### 1.1 Detection Result Parsing — ✅ Fixed
Three separate locations parsed YOLOE `result.boxes` into equivalent structures. Extracted `parse_detections(result) -> list[Detection]` into `src/core.py`. Both CLI and API now use this shared function.

#### 1.2 SKU Matching Pipeline — ✅ Fixed
`src/matcher.py:match_images()` (CLI) and `api/services/recognition.py:recognize()` (API) had parallel scoring logic. Extracted `score_matches()` into `src/indexer.py`. Both callers use it, then format results for their context.

#### 1.3 Embedding + Upload + Cleanup Pattern — ✅ Fixed
Two locations implemented the same download → embed → upload → cleanup flow. Extracted `process_single_media()` in `api/tasks.py`, used by both `start_embed_task()` and `manage_media()` route.

#### 1.4 "No Detections" Empty Result — ⬜ Open
`api/services/recognition.py` returns the same empty result dict in two places. Could extract `_empty_result(task_id, image_storage) -> dict`.

#### 1.5 Chroma Metadata Update Pattern — ✅ Fixed (different approach)
`update_sku_name()` added directly to `SKUIndexer` instead of a generic `_update_metadata()`.

#### 1.6 Inconsistent SKU Lookup in Routes — ✅ Fixed
All routes now use `_get_sku_or_error()` consistently.

### 2. Unnecessary Abstractions & Wrapper Indirection

#### 2.1 ReferenceProcessor Pass-Through Methods — ✅ Fixed
Removed 4 CRUD delegation methods (`delete_sku_references`, `delete_media_reference`, `set_sku_enabled`, `update_sku_name`). Routes call indexer directly.

#### 2.2 SKUIndexer.load() — ✅ Removed
Was an alias for `init_collection()` with unused `model_name` parameter. Callers updated.

#### 2.3 save_crop() — ⬜ Open
One-liner wrapper. Harmless convenience, low priority.

#### 2.4 src/\_\_init\_\_.py Re-exports — ✅ Fixed
File is now empty — no re-exports.

#### 2.5 hasattr Guards — ✅ Removed
Routes no longer check `hasattr(req.app.state, ...)`. DI providers handle this.

### 3. Module Boundary Violations

#### 3.1 Routes Reach Into Infrastructure — ✅ Fixed
Routes use `Depends()` from `api/dependencies.py` for typed service access. No direct `app.state` access in route handlers.

#### 3.2 SKUMatcher God Object — ⬜ Open
Still a convenience facade for CLI mode. Acceptable for now since API path uses `RecognitionService` directly.

#### 3.3 DIY Service Locator via app.state — ✅ Fixed
`app.state` still holds objects (for lifespan management), but routes never access it directly. All access through typed DI providers.

### 4. Stale / Dead Code

#### 4.1 objects365_classes.py — ✅ Deleted
#### 4.2 yoloe2o365.py — ✅ Deleted
#### 4.3 SKUMatch.crop_path — ✅ Removed
#### 4.4 Detection.mask — ⬜ Open
Field exists but is never populated. Kept for potential future masking use.
#### 4.5 SKUListItem — ✅ Fixed
#### 4.6 SKUIndexer.load() — ✅ Removed

### 5. Best Practice Issues

#### 5.1 Global SSL Monkey-Patch — ✅ Fixed
Removed `disable_ssl_verification()` and download fallback. Hard `FileNotFoundError` when weights are missing.

#### 5.2 tempfile.mktemp() — ✅ Fixed
Replaced with `NamedTemporaryFile(delete=False)` in `reference_processor.py`.

#### 5.3 Module-Level Side Effects — ✅ Fixed
`configure_ultralytics_weights()` called inside `main()` and `lifespan()`, not at module scope.

#### 5.4 asyncio.get_event_loop() — ✅ Fixed
Replaced with `get_running_loop()` in all 3 sites.

#### 5.5 No response_model on Routes — ✅ Fixed
All 10 route decorators have `response_model=ApiResponse`.

#### 5.6 ThreadPoolExecutor in Routes — ✅ Fixed (via DI)
Executor accessed via `Depends(get_inference_executor)` — type-safe DI provider.

#### 5.7 embedding_to_list() Inline Import — ✅ Fixed
`import numpy` moved to module level.

---

## 6. Architecture After Fixes

```
src/
├── core.py                # detect() + parse_detections() (shared)
├── embedder.py            # DINOv2Embedder (unchanged)
├── indexer.py             # SKUIndexer + score_matches() + update_sku_name()
├── matcher.py             # SKUMatcher (CLI convenience facade)
├── masking.py             # NEW — extract_binary_masks, mask_background
├── reference_processor.py # crop→embed pipeline (no CRUD delegation)
├── image_utils.py         # save_crop, draw_annotations
├── types.py               # Detection, SKUReference, SKUMatch (crop_path removed)
├── utils.py               # detect_device, free_gpu_memory, embedding_to_list
└── classes/
    └── beverage_cls.py    # Only file — objects365 deleted

api/
├── app.py                 # Lifespan + wiring
├── dependencies.py        # NEW — FastAPI DI providers (typed)
├── routes/
│   ├── goods.py           # Uses Depends() — no app.state access
│   ├── recognition.py     # Uses Depends() — no app.state access
│   ├── logs.py            # Unchanged
│   └── system.py          # Unchanged
├── services/
│   ├── recognition.py     # Uses shared score_matches()
│   └── image_storage.py   # Unchanged
└── tasks.py               # process_single_media() shared helper
```
