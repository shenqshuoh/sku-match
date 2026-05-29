# Code Quality Review — sku-match v0.2.0

## Executive Summary

The codebase is functional and well-organized at a surface level, with clean separation between `src/` (core ML) and `api/` (FastAPI server). However, a deeper review reveals **significant code duplication** (detection parsing in 3 places, matching pipeline in 2 parallel implementations, embedding+upload flow in 2 places), **module boundary violations** (routes bypass the service layer to access infrastructure directly), **dead code** (~400 lines), and several **best-practice issues** including a global SSL monkey-patch and use of deprecated APIs.

This report is structured into six sections: redundancy, unnecessary abstractions, module boundaries, stale code, best practices, and a proposed cleaner module map.

---

## 1. Redundancy & Duplicated Code

### 1.1 Detection Result Parsing — 3 Independent Implementations (HIGH)

Three separate locations parse YOLOE `result.boxes` into equivalent structures:

| Site | File | Lines | Output |
|------|------|-------|--------|
| CLI matcher | `src/matcher.py` | 155–165 | `Detection` dataclass |
| API recognition | `api/services/recognition.py` | 75–85 | Plain `dict` |
| CLI detection-only | `src/core.py` | 73–76 | Raw list for JSON |

All three perform the same operations: `box.xyxy[0].tolist()`, `int(box.cls[0])`, `float(box.conf[0])`, class name lookup via `BEVERAGE_CONTAINER_CLASSES[cls_id]`. The API version builds a dict; the CLI version builds a `Detection` dataclass. There is no shared parser.

**Recommendation**: Extract `parse_detections(result) -> list[Detection]` into `src/core.py` or a new utility. The API path converts `Detection` → dict at the boundary.

### 1.2 SKU Matching Pipeline — 2 Parallel Implementations (HIGH)

`src/matcher.py:match_images()` (CLI) and `api/services/recognition.py:recognize()` (API) implement nearly identical scoring logic:

```
embed crops → search_batch → sorted(distribution.items()) → ranked[0] →
concentration_score → threshold check → build results
```

Side-by-side comparison:
- `matcher.py` lines 99–106 vs `recognition.py` lines 117–136 — identical ranking + distribution + concentration logic
- The API version adds `match_conf` threshold filtering inline; the CLI version defers to output formatting in `main.py:135`
- The API version truncates distribution to top-5; the CLI version keeps the full distribution

**Recommendation**: Extract a shared `score_matches(detections, search_results, indexer, conf_threshold, concentration_topk) -> list[MatchResult]` function. Both CLI and API call this, then format results for their context.

### 1.3 Embedding + Upload + Cleanup Pattern — 2 Sites (HIGH)

Two locations implement the same download → embed → upload → cleanup flow:

- `api/routes/goods.py:manage_media()` lines 204–245
- `api/tasks.py:start_embed_task()` lines 39–62

Both do: `download_image()` → `process_and_add()` → `upload_to_qiniu(crop_path)` → `cleanup_download(crop_path)` → `cleanup_download(local_path)`, with the same error handling pattern.

**Recommendation**: Extract into a `MediaService.process_and_upload(sku_id, sku_name, media_id, url)` method.

### 1.4 "No Detections" Empty Result — Copy-Pasted (MEDIUM)

`api/services/recognition.py` returns the same empty result dict at **lines 67–72** (no boxes at all) and **lines 96–103** (boxes exist but all filtered by roiRect). The dict is identical:

```python
{"counts": {}, "detections": [], "matched_image": ..., "taskId": task_id}
```

**Recommendation**: Extract `_empty_result(task_id, image_storage) -> dict`.

### 1.5 Chroma Metadata Update Pattern — 2 Sites (LOW)

`reference_processor.py:update_sku_name()` (lines 232–241) and `indexer.py:set_enabled()` (lines 180–189) both follow the same `collection.get(where=...) → update metadatas → invalidate cache` pattern.

**Recommendation**: Add a generic `_update_metadata(sku_id, field, value)` method to `SKUIndexer`.

### 1.6 Inconsistent SKU Lookup in Routes (LOW)

`api/routes/goods.py` defines `_get_sku_or_error()` (line 31) used by `update_sku`, `enable_sku`, and `manage_media`. But `delete_sku` at lines 110–112 does its own inline lookup instead of using the helper.

**Recommendation**: Use `_get_sku_or_error` consistently. It returns the ORM object, which `delete_sku` needs for `db.delete(sku)`.

---

## 2. Unnecessary Abstractions & Wrapper Indirection

### 2.1 `ReferenceProcessor` — Half Its Methods Are Pass-Through Delegation (MEDIUM)

`src/reference_processor.py` lines 220–241 contain four methods that are pure delegation + logging:

```python
def delete_sku_references(self, sku_id):     # → self.indexer.delete_sku(sku_id) + log
def delete_media_reference(self, sku_id, media_id):  # → self.indexer.delete_media(...) + log
def set_sku_enabled(self, sku_id, enabled):  # → self.indexer.set_enabled(...) + log
def update_sku_name(self, sku_id, new_name): # → slightly more logic, but mirrors indexer pattern
```

The indexer already logs these operations internally (`indexer.py:172, 178, 189`). These wrapper methods produce **double-logging** for every operation.

**Assessment**: `ReferenceProcessor`'s real value is `crop_reference()` and `process_and_add()` — the detect→crop→embed pipeline. The CRUD delegation methods add no value. Either remove them (routes call indexer directly) or fold the reference processing logic into a merged indexer module.

### 2.2 `SKUIndexer.load()` — Dead Parameter (LOW)

`src/indexer.py` lines 283–284:
```python
def load(self, directory: Path, model_name: str) -> None:
    self.init_collection(persist_dir=directory)
```

`model_name` is accepted but completely ignored. The caller (`matcher.py:206`) passes it as `indexer.load(index_dir, emb_model)` — pointless. The method itself is just an alias for `init_collection()`.

### 2.3 `save_crop()` — One-Liner Wrapper (LOW)

`src/image_utils.py` lines 59–71 wraps `crop.save(output_path / name)`. It adds no validation, transformation, or error handling. Called from two places. Not harmful, but not earning its complexity budget.

### 2.4 `src/__init__.py` Re-exports — Trigger Heavy Imports (LOW)

`src/__init__.py` re-exports 7 symbols from submodules. **Zero files** use `from src import ...` — all imports go directly to submodules (`from src.core import detect`, `from src.indexer import SKUIndexer`, etc.). This `__init__.py` means any `import src.*` triggers torch, ultralytics, chromadb to load unnecessarily.

**Recommendation**: Make `src/__init__.py` empty or remove the re-exports.

### 2.5 `hasattr(req.app.state, "processor")` Guards — Unnecessary Defensive Code (LOW)

Three routes in `goods.py` (lines 101, 117, 137) check `hasattr(req.app.state, "processor")` before calling it. The processor is **always** set in `lifespan()` at `app.py:121`. If lifespan fails to set it, the entire server is broken. These guards are dead branches that add noise.

---

## 3. Module Boundary Violations

### 3.1 Routes Reach Directly Into Infrastructure (HIGH — Most Significant Issue)

This is the single most impactful architectural problem. Routes bypass the service layer and directly access infrastructure:

**`api/routes/goods.py`** accesses:
- `req.app.state.processor` — a `ReferenceProcessor` from `src/` (lines 77, 102, 118, 138, 215, 255)
- `req.app.state.image_storage` (lines 211, 233, 237, 245)
- `req.app.state.inference_executor` (lines 202, 213)

**`api/routes/recognition.py`** accesses:
- `req.app.state.inference_executor` (line 42)
- `req.app.state.recognition_service` (line 43)
- `req.app.state.image_storage` (lines 55, 61, 64, 85)
- `req.app.state.device` (line 47)

This means routes are performing service orchestration — downloading images, scheduling GPU work, uploading to Qiniu, cleaning up temp files. The `manage_media` route handler (`goods.py:188–261`) is 73 lines of business logic that should live in a service layer.

### 3.2 `SKUMatcher` — God Object (MEDIUM)

`src/matcher.py` imports from 7 out of ~10 `src/` modules: `embedder`, `indexer`, `image_utils`, `classes`, `types`, `utils`, plus `ultralytics`, `torch`, `numpy`, `PIL`. It owns detection, cropping, embedding, search, scoring, crop saving, and verbose printing — 6 responsibilities in one class.

It works as a CLI convenience facade, but any change to any part of the pipeline requires modifying this file, and the API service (`recognition.py`) cannot reuse any of its logic because it needs different orchestration.

### 3.3 DIY Service Locator via `app.state` (MEDIUM)

`api/app.py:lifespan()` manually wires 9 objects onto `app.state`:

```python
app.state.inference_executor = inference_executor
app.state.device = device
app.state.detector = detector
app.state.embedder = embedder
app.state.chroma_client = chroma_client
app.state.indexer = indexer
app.state.image_storage = image_storage
app.state.processor = processor
app.state.recognition_service = recognition_service
```

Every route then accesses these via `req.app.state.X`. Problems:
- **No type safety** — `req.app.state.processor` is `Any` to mypy
- **Routes know too much** about infrastructure internals
- **Easy to get wrong** — typo in attribute name = runtime crash
- **Hard to test** — requires mocking `app.state` attributes

**Recommendation**: Use FastAPI's dependency injection. Define provider functions that yield typed services. Routes declare them as `Depends()` parameters.

---

## 4. Stale / Dead Code

### 4.1 `src/classes/objects365_classes.py` — 372 Lines, Zero Imports

The 365-class Objects365 list is **never imported** by any Python file in the project. The project uses `BEVERAGE_CONTAINER_CLASSES` (7 items) exclusively. This file is dead weight.

**Action**: Delete the file.

### 4.2 `src/yoloe2o365.py` — Never Imported by Any Module

CoreML export utility. Has a `__main__` block for standalone execution but is never imported. Belongs in `scripts/`, not `src/`.

**Action**: Move to `scripts/export_coreml.py` or delete.

### 4.3 `SKUMatch.crop_path` Field — Never Set

`src/types.py:33` defines `crop_path: Path | None = None`. The only place `SKUMatch` is constructed (`matcher.py:107–115`) never sets this field. It is always `None`. (Note: `ProcessResult.crop_path` in `reference_processor.py` is a different, actively-used field.)

**Action**: Remove the field from `SKUMatch`.

### 4.4 `Detection.mask` Field — Never Set

`src/types.py:13` defines `mask: np.ndarray | None = None`. The only place `Detection` objects are constructed (`matcher.py:159–164`) never passes `mask`. It is always `None`. Masks are used in `core.py:_save_result()` but through raw YOLOE result objects, not the `Detection` dataclass.

**Action**: Remove the field from `Detection`, or actually populate it in `_detect_all()`.

### 4.5 `SKUListItem` Schema — Defined But Never Instantiated

`api/schemas.py:147` defines `SKUListItem(BaseModel)` and `SKUListData.list` is typed as `list[SKUListItem]`. But `goods.py:list_skus()` at lines 168–178 builds plain dicts inside `SKUListData(list=[...])` instead of constructing `SKUListItem` objects. This bypasses the schema validation entirely, defeating its purpose.

**Action**: Either construct `SKUListItem` objects properly or remove the unused model.

### 4.6 `SKUIndexer.load()` — Unnecessary Alias

`indexer.py:283–284` is an alias for `init_collection()` with an unused `model_name` parameter.

**Action**: Replace callers with `init_collection()` directly and remove the method.

---

## 5. Best Practice Issues

### 5.1 Global SSL Monkey-Patch (HIGH — Security)

`src/utils.py:39`:
```python
ssl._create_default_https_context = ssl._create_unverified_context
```

This disables SSL verification **globally** for the entire process. Called from `indexer.py:96` on macOS. Any code in the same process — including `httpx` calls to Qiniu token service and upload endpoints — inherits this weakened security posture.

**Recommendation**: Scope the SSL override. Use a custom `ssl.SSLContext` for the specific chromadb/weight-download HTTP calls that need it, rather than patching the global default.

### 5.2 `tempfile.mktemp()` — Deprecated, Race Condition (MEDIUM — Security)

`src/reference_processor.py:141`:
```python
crop_path = Path(tempfile.mktemp(suffix=".jpg", prefix=f"crop_{sku_id}_{media_id}_"))
```

`mktemp()` has been deprecated since Python 2.3 due to TOCTOU race conditions. Another process could create a file at the returned path before `crop.save()` writes to it.

**Recommendation**: Use `tempfile.NamedTemporaryFile(suffix=".jpg", prefix=..., delete=False)` instead.

### 5.3 Module-Level Side Effects (MEDIUM)

`main.py:17` and `api/app.py:28` both call `configure_ultralytics_weights()` at module level. This triggers when the module is imported, not just when the program runs. It modifies ultralytics global settings, affecting tests and any code that imports these modules.

**Recommendation**: Call inside `main()` / `lifespan()`, not at module scope.

### 5.4 `asyncio.get_event_loop()` — Deprecated (MEDIUM)

Three call sites use the deprecated `asyncio.get_event_loop()`:
- `api/routes/recognition.py:39`
- `api/routes/goods.py:201`
- `api/tasks.py:37`

This emits deprecation warnings in Python 3.12+ and may break in future versions.

**Recommendation**: Replace with `asyncio.get_running_loop()`.

### 5.5 No `response_model` on Route Decorators (MEDIUM)

No route decorator declares a `response_model`. All routes return `ApiResponse` (a Pydantic model), but without `response_model=ApiResponse`:
- FastAPI doesn't validate/filter response fields
- OpenAPI docs show generic response schemas
- No compile-time return type checking

**Recommendation**: Add `response_model=ApiResponse` to route decorators and `-> ApiResponse` return annotations.

### 5.6 `ThreadPoolExecutor` Leaks Into Routes (MEDIUM)

`inference_executor` is stored on `app.state` and accessed directly by routes (`goods.py:202`, `recognition.py:42`). Routes should not know about thread pools — this is an implementation detail of GPU scheduling. Routes call services; services manage concurrency internally.

**Recommendation**: Encapsulate the executor inside service classes. Routes call `service.recognize()`; the service handles `loop.run_in_executor()` internally.

### 5.7 `embedding_to_list()` — Inline Import (LOW)

`src/utils.py:42–45` does `import numpy as np` inside the function body. While numpy is virtually always loaded in this project, the inline import adds overhead per call. The function is also called in hot loops (`indexer.py:222`).

**Recommendation**: Move the import to module level.

---

## 6. Proposed Cleaner Module Map

### Deep Module Principles

A "deep module" has a simple interface that hides significant complexity behind it. The current `src/` modules are reasonably deep individually, but the API layer has shallow boundaries — routes dig through to infrastructure instead of going through a service interface.

### Current Dependency Graph

```
main.py ──→ SKUMatcher ──→ SKUIndexer, DINOv2Embedder, image_utils
                     ──→ ReferenceProcessor ──→ SKUIndexer, DINOv2Embedder, YOLOE

api/routes/* ──→ app.state.{processor, recognition_service, image_storage, inference_executor}
api/services/recognition.py ──→ SKUIndexer, DINOv2Embedder, YOLOE, image_utils
api/tasks.py ──→ processor, image_storage, inference_executor (passed as args)
```

Problems: routes know about infrastructure (executor, processor internals). Two parallel matching pipelines. `ReferenceProcessor` is a pass-through for half its methods.

### Proposed Structure

```
src/
├── detection.py          # YOLOE wrapper + parse_detections()
│                           Exports: Detector, parse_detections()
│
├── embedding.py          # DINOv2Embedder (largely unchanged)
│
├── index.py              # Chroma index + metadata CRUD + search scoring
│                           Absorbs: SKUIndexer + ReferenceProcessor's CRUD methods
│                           Exports: SKUIndex, search_and_score()
│
├── matching.py           # Thin CLI orchestrator (current SKUMatcher, slimmed)
│                           Uses: detection, embedding, index
│
├── reference.py          # Reference image processing (crop + embed pipeline only)
│                           Uses: detection, embedding, index
│                           No CRUD delegation methods
│
├── image_utils.py        # Unchanged
├── types.py              # Remove dead fields (crop_path, mask)
├── utils.py              # Unchanged
└── classes/
    └── beverage_cls.py   # Only this file — delete objects365_classes.py

api/
├── app.py                # Lifespan + wiring (consider DI providers)
├── dependencies.py       # FastAPI dependency providers (typed)
│                           def get_recognition_service() -> RecognitionService
│                           def get_sku_service() -> SKUService
│                           def get_image_storage() -> ImageStorage
│
├── services/
│   ├── recognition.py    # Full pipeline (detect → match → annotate)
│   │                       Uses shared scoring from src.index.search_and_score()
│   ├── sku.py            # NEW — SKU management (embed + upload + cleanup)
│   │                       Absorbs duplicated pattern from goods.py + tasks.py
│   └── image_storage.py  # Unchanged
│
├── routes/
│   ├── recognition.py    # Thin — parse request → call service → return
│   ├── goods.py          # Thin — parse request → call service → return
│   ├── logs.py           # Unchanged
│   └── system.py         # Unchanged
```

### Key Design Decisions

| Question | Decision | Rationale |
|----------|----------|-----------|
| Should CLI and API share matching pipeline? | **Yes** — extract `score_matches()` | Eliminates §1.2 duplication; both callers format results differently |
| Should `ReferenceProcessor` CRUD methods fold into `SKUIndexer`? | **Yes** — they're logging wrappers over methods that already log | Removes §2.1 indirection; routes call indexer directly |
| Should `app.state` be replaced with DI? | **Yes** — define typed provider functions | Type safety, testability, idiomatic FastAPI |
| Should `src/` split differently? | **Marginally** — extract `detection.py` from `core.py` | Main issue is API routes, not src module boundaries |

---

## 7. Priority Summary

| Priority | Issue | Effort | Section |
|----------|-------|--------|---------|
| **P0** | `tempfile.mktemp()` race condition | 5 min | §5.2 |
| **P0** | Global SSL monkey-patch | 30 min | §5.1 |
| **P1** | Extract shared detection parsing | 1 hr | §1.1 |
| **P1** | Extract shared scoring logic | 2 hr | §1.2 |
| **P1** | Move route logic to services | 3 hr | §3.1 |
| **P2** | Extract embedding+upload service | 1 hr | §1.3 |
| **P2** | Replace `app.state` with DI | 2 hr | §3.3 |
| **P2** | Remove dead code (~400 lines) | 30 min | §4 |
| **P2** | Fix deprecated `get_event_loop()` | 10 min | §5.4 |
| **P3** | Remove `SKUIndexer.load()` alias | 5 min | §4.6 |
| **P3** | Clean `src/__init__.py` re-exports | 5 min | §2.4 |
| **P3** | Add `response_model` to routes | 1 hr | §5.5 |

---

## 8. What's Already Good

- **`src/embedder.py`** — Clean, well-structured. Good device handling, FP16 logic, ONNX fallback path
- **`api/config.py`** — Proper `pydantic-settings` usage with sensible defaults
- **`api/auth.py`** — Clean optional auth pattern, correctly disabled when not configured
- **`src/indexer.py:search_batch()`** — Complex algorithm, but well-documented with clear invariants and return types
- **`api/database.py`** — Minimal, correct async SQLAlchemy setup
- **`api/services/image_storage.py`** — Good separation of concerns, handles local and remote files cleanly
- **`concentration_score()` and `_softmax()`** — Pure functions, easy to test, well-named
- **`api/app.py:lifespan()`** — Despite the service locator issue, the startup/shutdown lifecycle is well-managed with proper cleanup
