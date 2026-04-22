# beverage-cashier API Server Implementation Plan

> **Status**: Planning phase — no implementation yet.  
> **Target**: Linux server deployment.  
> **Branch**: `api-server-plan`

---

## 1. Context

This project currently exists as a **local CLI tool** for bottle/can detection + DINOv2 SKU matching. The API.md spec defines a **production REST API** that wraps this pipeline into a multi-user web service with SKU management, recognition logging, and training lifecycle.

### Current State (CLI)

- **Detection**: YOLOE with open-vocabulary detection (bottle, canned, carton, etc.)
- **Matching**: DINOv2 embeddings + top-2 cosine similarity against reference index
- **Index**: File-based (`embeddings_{model}.npy` + `metadata_{model}.json`)
- **Input**: Hardcoded `data/images/` directory
- **Device**: Hardcoded to `mps` (Apple Silicon)

### Target State (API)

- **10 REST endpoints** across 4 domains: recognition, SKU CRUD, logs, system
- **URL-based image input** from OSS
- **ROI filtering** (waiter-drawn rectangle)
- **Async task tracking** for index rebuilds / training
- **Per-request item IDs** and annotated result images
- **Concurrent request handling** on Linux servers

---

## 2. Fixes & Improvements to Existing Code

These must happen before or alongside API work.

| # | Issue | Fix | Priority |
|---|-------|-----|----------|
| F1 | `ssl._create_unverified_context` in `embedder.py:36` | Remove or guard with `sys.platform == "darwin"` | Critical |
| F2 | Default device = `mps` everywhere | Auto-detect: `cuda → mps → cpu` | Critical |
| F3 | `sku_name` missing from data model | Add to `SKUReference`, `SKUMatch`, indexer metadata | Critical |
| F4 | No `itemId` per detection | Add sequential counter per request | High |
| F5 | No ROI filtering in matcher | Add `roi` parameter to filter detections by bbox overlap | High |
| F6 | No URL image loading | Add `ImageLoader` utility (local path + HTTP URL support) | High |
| F7 | No annotated image generation | Add bbox + label drawing utility for `matched_image` | High |
| F8 | `clip` dependency in `pyproject.toml` | Dead dependency — not imported anywhere. Remove. | Medium |
| F9 | `coremltools` in main deps | Only needed for iOS export. Move to optional `[export]` group. | Medium |
| F10 | `print()` statements everywhere | Replace with Python `logging` module | Medium |
| F11 | `sys.path.insert(0, ...)` in scripts | Anti-pattern. Use proper package config. | Low |

---

## 3. Implementation Phases

### Phase 0: Foundation Fixes (Day 1)

Apply the fixes above to existing `src/` code **without changing the CLI behavior**.

1. **Embedder**: Remove SSL hack, add `_detect_device()` helper.
2. **Types**: Add `sku_name` to `SKUReference` and `SKUMatch`.
3. **Indexer**: Store/retrieve `sku_names` in metadata JSON.
4. **Matcher**: Add `roi` filtering, `itemId` generation. Keep CLI path working.
5. **Image Utils**: Add `draw_annotations()` for bbox + label overlay.
6. **pyproject.toml**: Remove `clip`, move `coremltools` to optional, add `logging` config.

### Phase 1: Web Framework + Database (Day 2)

Build the API skeleton.

```
api/
├── app.py              # FastAPI app + lifespan (model loading)
├── config.py           # pydantic-settings (env vars, .env)
├── database.py         # SQLAlchemy async engine + session
├── models.py           # ORM tables
├── schemas.py          # Pydantic request/response schemas
├── routes/
│   ├── recognition.py  # POST /detect, POST /fix
│   ├── goods.py        # SKU CRUD + media management
│   ├── logs.py         # GET /logs/recognition/get
│   └── system.py       # GET /train-status, GET /health
├── services/
│   ├── recognition.py  # Orchestrate: download → detect → match → annotate
│   ├── index_manager.py# Build/reload index from DB, atomic swap
│   └── image_storage.py# Download URLs, save files, generate result URLs
└── tasks.py            # Background task definitions
```

#### Database Schema (SQLite)

| Table | Columns |
|-------|---------|
| `sku` | id, sku_id (unique), sku_name, enabled, train_status, created_at |
| `sku_media` | id, sku_id (FK), media_url, media_type (IMAGE/VIDEO), created_at |
| `recognition_log` | id, task_id (unique), request_json, ai_result_json, user_correction_json, visual_image_path, created_at |
| `train_job` | id, train_job_id (unique), sku_id (FK), status (pending/training/completed/failed), progress (0-100), estimated_time, created_at, updated_at |

#### Model Lifecycle

- Load YOLOE + DINOv2 + SKUIndexer **once** at FastAPI lifespan startup.
- Reuse across requests (expensive to reload).
- Use `asyncio.Lock` when swapping index (atomic rename).

### Phase 2: Core API Endpoints (Days 3-5)

#### 2.1 `POST /api/v1/recognition/detect`

**Request**:
```json
{
  "taskId": "uuid",
  "mode": "IMAGE",
  "files": "https://oss.example.com/image.jpg",
  "roiRect": [x1, y1, x2, y2]
}
```

**Flow**:
1. Validate request → log to `recognition_log` (status: processing)
2. Download image via `ImageLoader` (URL → temp file)
3. Run YOLOE detection on image
4. Filter detections by `roiRect` (IoU or center-point overlap)
5. Embed each crop with DINOv2
6. Match against SKU index (enabled SKUs only)
7. Draw bboxes + SKU labels on original image → save as `matched_image`
8. Store result, update log status → return JSON

**Response**:
```json
{
  "code": 1,
  "data": {
    "counts": {"sku_id": N, ...},
    "detections": [{"itemId", "bbox", "class_id", "class_name", "detection_conf", "sku_id", "sku_name", "match_score"}],
    "matched_image": "https://.../results/taskId_annotated.jpg",
    "taskId": "..."
  },
  "msg": "成功"
}
```

#### 2.2 `POST /api/v1/recognition/fix`

**Request**:
```json
{
  "taskId": "...",
  "fixItems": [
    {"fixType": "CHANGE_SKU", "itemId": 3, "skuId": "..."},
    {"fixType": "LOST", "roiRect": [...], "skuId": "..."}
  ]
}
```

**Flow**:
1. Look up original `recognition_log` by `taskId`
2. Apply corrections: update SKU or add missing detection
3. Store corrected result in `user_correction` column
4. Return success

> **Note**: Actual model retraining from corrections is deferred (see Section 6).

#### 2.3 `GET /api/v1/goods/sku/list`

- Paginated with `page`, `size`, `keyword` (matches `sku_name` or `sku_id`)
- Returns list with embedded `medias` array

#### 2.4 `POST /api/v1/goods/sku/new`

- Insert `sku` row with `trainStatus: PENDING`
- Insert `sku_media` rows
- Download reference images to local storage
- Queue async index rebuild → returns `trainJobId`

#### 2.5 `POST /api/v1/goods/sku/update`

- Update `sku_name` only

#### 2.6 `POST /api/v1/goods/sku/delete`

- Delete `sku` + `sku_media` + local reference files
- Queue async index rebuild

#### 2.7 `POST /api/v1/goods/sku/enable`

- Toggle `enabled` flag
- If disabling: queue async index rebuild (exclude disabled SKUs)

#### 2.8 `POST /api/v1/goods/sku/media`

- `action: add` → download new images, insert rows, queue rebuild
- `action: delete` → delete files + rows, queue rebuild

#### 2.9 `GET /api/v1/logs/recognition/get`

- Query by `taskId`
- Return `ai_result`, `user_correction`, `visual_image_url`

#### 2.10 `GET /api/v1/system/train-status/get`

- Query `train_job` by `trainJobId`
- Return `{status, progress, estimated_time}`

### Phase 3: Index Management (Day 5)

#### 3.1 Incremental Index Rebuild

- Trigger: SKU/media CRUD operations
- Process: Read all enabled SKUs + their media from DB → embed → build new index
- Output: Save to temp file → atomic rename to active index path

#### 3.2 Hot-Reload

```python
# Pseudocode
async with index_lock:
    new_index = build_index_from_db()
    new_index.save(tmp_path)
    os.replace(tmp_path, active_path)  # atomic on Linux
    matcher.reload_index(active_path)
```

#### 3.3 Background Task Runner

- For single-server Linux: `asyncio.create_task` with a task tracker dictionary
- Store task state in `train_job` table
- Poll `GET /api/v1/system/train-status/get` for progress

### Phase 4: Deployment Readiness (Day 6)

| Item | Details |
|------|---------|
| Environment config | All paths, device, model paths, DB URL via `pydantic-settings` + `.env` |
| Dockerfile | Python 3.11 + CUDA runtime base (or CPU fallback) |
| `pyproject.toml` | Add `[project.scripts]` for `beverage-cashier-api` entry point |
| Health check | `GET /health` — verify models loaded, DB accessible, disk space |
| Logging | Structured JSON logs, log rotation |
| Static file serving | Serve result images via FastAPI `StaticFiles` or reverse proxy |

---

## 4. Data Flow (Single Detection Request)

```
Client → POST /recognition/detect
  → Download image from OSS URL
  → YOLOE detect (bottle/can/carton/etc.)
  → Filter by roiRect
  → DINOv2 embed each crop
  → SKUIndexer.match_batch (top-2 similarity)
  → Draw annotated image
  → Save to storage
  → Log to recognition_log
  → Return JSON result
```

---

## 5. Updated Directory Structure

```
beverage-cashier/
├── api/                          # NEW - FastAPI application
│   ├── app.py
│   ├── config.py
│   ├── database.py
│   ├── models.py
│   ├── schemas.py
│   ├── routes/
│   │   ├── recognition.py
│   │   ├── goods.py
│   │   ├── logs.py
│   │   └── system.py
│   ├── services/
│   │   ├── recognition.py
│   │   ├── index_manager.py
│   │   └── image_storage.py
│   └── tasks.py
├── src/                          # EXISTING - ML pipeline
│   ├── core.py
│   ├── embedder.py               # Fix SSL, device auto-detect
│   ├── indexer.py                # Add sku_name support
│   ├── matcher.py                # Add ROI, itemId
│   ├── types.py                  # Add sku_name
│   ├── image_utils.py            # Add annotation drawing
│   └── ...
├── scripts/                      # EXISTING
├── main.py                       # EXISTING - CLI entry
├── Dockerfile                    # NEW
├── .env.example                  # NEW
├── .env                          # NEW (gitignored)
└── pyproject.toml                # Updated deps
```

---

## 6. Deferred (Training Scope)

Per requirements, these are noted but **not in this plan**:

- **Single-item training**: Fine-tuning DINOv2 or YOLOE on new SKU angles. The current plan only rebuilds the embedding index (retrieval), not the underlying model weights.
- **Continuous training from corrections**: The `/fix` endpoint logs corrections but does not feed them back into model weights. The `train_job` table is scaffolding for this.
- **YOLOE detection fine-tuning**: Out of scope.
- **Video recognition**: `mode: VIDEO` in the detect request is not yet implemented.
- **Multi-photo merge**: "同一场景多角度多张照片合并识别统计" mentioned in requirements is not yet designed.

---

## 7. Dependency Changes

### Additions
- `fastapi`
- `uvicorn[standard]`
- `sqlalchemy[asyncio]`
- `aiosqlite`
- `httpx`
- `pydantic-settings`
- `python-multipart`
- `python-jose` (if auth needed later)

### Removals
- `clip` (dead dependency)

### Moves
- `coremltools` → `[project.optional-dependencies] export`

---

## 8. Risk Notes

1. **Model loading time**: YOLOE + DINOv2 takes ~30-60s to load. Must happen at app startup, not per-request.
2. **GPU memory**: DINOv2 vitl14 + YOLOE on a single GPU may OOM with concurrent requests. Batch processing within a single request is fine; multiple concurrent requests may need request queuing.
3. **Index rebuild time**: For 100+ SKUs with multiple images, embedding all references takes time. Background task with status polling is the right pattern.
4. **Image download failures**: OSS URLs may be slow/unavailable. Add timeout and retry logic.

---

## 9. Open Questions

1. **Result image hosting**: Should the API serve result images directly (`/static/...`) or upload them to OSS?
2. **Authentication**: No auth in current spec. Is this API public or behind an auth gateway?
3. **Concurrent detection limits**: Should we cap concurrent `/detect` requests to prevent GPU OOM?
4. **Index persistence**: Should index files live on disk or in object storage for multi-instance deployment?

---

*Plan written on 2026-04-22. Branch: `api-server-plan`.*
