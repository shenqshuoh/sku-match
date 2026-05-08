# sku-match API Server Implementation Plan

> **Status**: ✅ **IMPLEMENTED** — All phases (0–4) complete. API server is fully functional.  
> **Target**: Linux server deployment.  
> **Branch**: `api-server-plan`

---

## 1. Context

This project currently exists as a **local CLI tool** for bottle/can detection + DINOv2 SKU matching. The API.md spec defines a **production REST API** that wraps this pipeline into a multi-user web service with SKU management, recognition logging, and training lifecycle.

### Current State (CLI + API)

- **Detection**: YOLOE with BEVERAGE_CONTAINER_CLASSES (7 items: Bottle, Canned, etc.)
- **Matching**: DINOv2 embeddings + top-2 cosine similarity against Chroma vector store
- **Index**: Chroma `PersistentClient` with HNSW cosine index — incremental add/delete, metadata filtering, persistent storage
- **Input**: CLI uses `data/images/` directory; API accepts URL-based images from OSS
- **Device**: Auto-detect via `detect_device()` (cuda → mps → cpu)
- **API**: FastAPI server with 10 endpoints, SQLAlchemy async DB, async task runner

### Target State (API)

- **10 REST endpoints** across 4 domains: recognition, SKU CRUD, logs, system
- **URL-based image input** from OSS
- **ROI filtering** (waiter-drawn rectangle)
- **Chroma vector store** replacing file-based numpy index — enables incremental add/delete, metadata filtering, and persistent storage
- **Async task tracking** for index rebuilds / training
- **Per-request item IDs** and annotated result images
- **Concurrent request handling** on Linux servers

---

## 2. Fixes & Improvements to Existing Code

These must happen before or alongside API work.

| # | Issue | Fix | Priority | Status |
|---|-------|-----|----------|--------|
| F1 | `ssl._create_unverified_context` in `embedder.py:36` | Remove or guard with `sys.platform == "darwin"` | Critical | Kept (macOS compat) |
| F2 | Default device = `mps` everywhere | Auto-detect: `cuda → mps → cpu` via `detect_device()` | Critical | ✅ Done |
| F3 | `sku_name` missing from data model | Add to `SKUReference`, `SKUMatch`, indexer metadata | Critical | ✅ Done |
| F4 | No `itemId` per detection | Add sequential counter per request | High | ✅ Done |
| F5 | No ROI filtering in matcher | Add `roi` parameter to filter detections by bbox overlap | High | ✅ Done |
| F6 | No URL image loading | Add `ImageLoader` utility (local path + HTTP URL support) | High | ✅ Done |
| F7 | No annotated image generation | Add bbox + label drawing utility for `matched_image` | High | ✅ Done |
| F8 | `clip` dependency in `pyproject.toml` | Dead dependency — not imported anywhere. Remove. | Medium | ✅ Done |
| F9 | `coremltools` in main deps | Only needed for iOS export. Move to optional `[export]` group. | Medium | ✅ Done |
| F10 | `print()` statements everywhere | Replace with Python `logging` module | Medium | ✅ Done |
| F11 | `sys.path.insert(0, ...)` in scripts | Anti-pattern. Use proper package config. | Low | Open |
| F12 | File-based numpy index (`indexer.py`) | Replace with Chroma vector store (see Section 10) | Critical | ✅ Done |

---

## 3. Implementation Phases

### Phase 0: Foundation Fixes (Day 1)

Apply the fixes above to existing `src/` code **without changing the CLI behavior**.

1. **Embedder**: Remove SSL hack, add `_detect_device()` helper.
2. **Types**: Add `sku_name` to `SKUReference` and `SKUMatch`.
3. **Indexer → Chroma**: Replace numpy-based `SKUIndexer` with Chroma-backed implementation (see Section 10 for detailed design). CLI continues to work.
4. **Matcher**: Add `roi` filtering, `itemId` generation. Keep CLI path working.
5. **Image Utils**: Add `draw_annotations()` for bbox + label overlay.
6. **pyproject.toml**: Remove `clip`, move `coremltools` to optional, add `chromadb` + `logging` config.

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

- Load YOLOE + DINOv2 **once** at FastAPI lifespan startup.
- Initialize Chroma `PersistentClient` + `sku_embeddings` collection at startup.
- All three are reused across requests (expensive to reload).
- SKU enable/disable handled via Chroma `where` filter — no index rebuild needed.
- Adding/deleting reference embeddings is incremental via `collection.upsert()` / `collection.delete()`.

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
- Queue async embedding task: download reference images → embed with DINOv2 → `collection.upsert()` to Chroma
- Returns `skuId` + `trainJobId`

#### 2.5 `POST /api/v1/goods/sku/update`

- Update `sku_name` only
- Also update Chroma metadata: `collection.update(ids, metadatas=[{"sku_name": new_name}])`

#### 2.6 `POST /api/v1/goods/sku/delete`

- Delete `sku` + `sku_media` + local reference files
- Delete from Chroma: `collection.delete(where={"sku_id": "..."})` — **immediate, no rebuild**

#### 2.7 `POST /api/v1/goods/sku/enable`

- Toggle `enabled` flag in DB
- Update Chroma metadata: `collection.update(ids, metadatas=[{"enabled": true/false}])` — **immediate, filtered at query time**

#### 2.8 `POST /api/v1/goods/sku/media`

- `action: add` → download images → embed → `collection.upsert()` — **immediate**
- `action: delete` → delete files + DB rows → `collection.delete(ids=[...])` — **immediate**

#### 2.9 `GET /api/v1/logs/recognition/get`

- Query by `taskId`
- Return `ai_result`, `user_correction`, `visual_image_url`

#### 2.10 `GET /api/v1/system/train-status/get`

- Query `train_job` by `trainJobId`
- Return `{status, progress, estimated_time}`

### Phase 3: Index Management with Chroma (Day 5)

> Chroma replaces the previous plan's full-index-rebuild approach.  
> **Key benefit**: Incremental add/delete of embeddings — no full rebuild when SKUs change.

#### 3.1 Chroma Collection Design

```python
import chromadb

client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
collection = client.get_or_create_collection(
    name="sku_embeddings",
    configuration={
        "hnsw": {
            "space": "cosine",       # DINOv2 uses cosine similarity
            "ef_construction": 200,   # Higher = better recall
            "max_neighbors": 32,      # Graph density
        }
    }
)
```

**Each reference image = one Chroma document:**
- `id`: `"{sku_id}__{media_id}"` (unique per reference image)
- `embedding`: DINOv2 768-dim vector (pre-computed, passed directly)
- `metadata`: `{"sku_id": "...", "sku_name": "...", "enabled": True, "class_name": "Bottle"}`

#### 3.2 Incremental Operations (No Full Rebuild)

| Operation | Chroma API | Impact |
|-----------|-----------|--------|
| Add new SKU | `collection.upsert(ids, embeddings, metadatas)` | Immediate — HNSW supports dynamic inserts |
| Add media to SKU | `collection.upsert(ids, embeddings, metadatas)` | Immediate — adds new reference vectors |
| Delete media | `collection.delete(ids=[...])` | Immediate — removes specific reference vectors |
| Delete SKU | `collection.delete(where={"sku_id": "..."})` | Immediate — removes all vectors for that SKU |
| Disable SKU | `collection.update(ids, metadatas=[{"enabled": False}])` | Immediate — filtered at query time via `where` |
| Enable SKU | `collection.update(ids, metadatas=[{"enabled": True}])` | Immediate |

> **This eliminates the "queue async index rebuild" step** from the original plan for most SKU operations. The only time a full embed + rebuild is needed is initial index population from an empty collection.

#### 3.3 Matching: Top-2 Per SKU (Post-Processing)

Chroma's native query returns top-N results across all documents. The project's "sum of top-2 cosine similarities per SKU" matching requires post-processing:

```python
# 1. Query Chroma for top-N results (N large enough to get ≥2 per SKU)
results = collection.query(
    query_embeddings=crop_embeddings,  # batch: all detection crops
    n_results=num_skus * 2,            # enough to get 2+ per SKU
    where={"enabled": True},           # filter disabled SKUs
    include=["metadatas", "distances"]
)

# 2. Post-process: group by sku_id, sum top-2 similarities
# Chroma returns cosine DISTANCE (0 = identical, 2 = opposite)
# Convert: similarity = 2.0 - distance  (for cosine space)
for crop_idx in range(len(results["ids"])):
    sku_scores: dict[str, list[float]] = {}
    for sku_id, distance in zip(
        [m["sku_id"] for m in results["metadatas"][crop_idx]],
        results["distances"][crop_idx]
    ):
        similarity = 2.0 - distance
        sku_scores.setdefault(sku_id, []).append(similarity)
    
    # Sum top-2 per SKU → softmax probability distribution (current algorithm)
    best_sku, best_score = None, -1.0
    for sku_id, sims in sku_scores.items():
        top2 = sorted(sims, reverse=True)[:2]
        score = top2[0] if len(top2) == 1 else sum(top2)
        if score > best_score:
            best_sku, best_score = sku_id, score
```

> **Note**: This post-processing is lightweight (pure Python dict ops on ~50-200 results per crop). No measurable performance impact.

> **Update (2026-05-08)**: The current implementation uses **softmax normalization** to produce a probability distribution (0–1) over all enabled SKUs. The `match_conf` threshold operates on this probability (default 0.5). **Concentration score** (top-1 share of top-K probability mass, default K=10) replaced the earlier match_ratio as the confidence signal. Detection rank positions (`top2_ranks`) of the matched SKU's nearest vectors are tracked for diagnostics. FP16 inference (`model.half()`) is enabled automatically when `device=="cuda"` for both YOLOE and DINOv2, giving ~1.8x speedup. Default embedding model changed to `dinov2_vits14` (384-dim) for better GPU memory fit.

#### 3.4 Background Task Runner

Still needed for **initial index population** and **re-embedding after media changes**:

```python
async def rebuild_index_task(job_id: str, sku_ids: list[str] | None = None):
    """Embed reference images and upsert to Chroma.
    
    If sku_ids is None: full rebuild from DB.
    If sku_ids provided: only re-embed those SKUs.
    """
    # 1. Fetch SKU + media from DB
    # 2. Download reference images
    # 3. Embed with DINOv2
    # 4. collection.upsert(ids, embeddings, metadatas)
    # 5. Update train_job status
```

For single-server Linux: `asyncio.create_task` with task tracker dict. Store state in `train_job` table.

### Phase 4: Deployment Readiness (Day 6)

| Item | Details |
|------|---------|
| Environment config | All paths, device, model paths, DB URL, Chroma dir via `pydantic-settings` + `.env` |
| Dockerfile | Python 3.11 + CUDA runtime base (or CPU fallback) |
| `pyproject.toml` | Add `[project.scripts]` for `beverage-cashier-api` entry point |
| Health check | `GET /health` — verify models loaded, DB accessible, Chroma heartbeat |
| Logging | Structured JSON logs, log rotation |
| Static file serving | Serve result images via FastAPI `StaticFiles` or reverse proxy |
| `.gitignore` | Add `chroma_data/` to ignore Chroma persistence directory |

---

## 4. Data Flow (Single Detection Request)

```
Client → POST /recognition/detect
  → Download image from OSS URL
  → YOLOE detect (bottle/can/carton/etc.)
  → Filter by roiRect
  → DINOv2 embed each crop
  → Chroma collection.query(query_embeddings=crops, where={enabled: True})
  → Post-process: group by sku_id, sum top-2 cosine similarities
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
│   │   ├── index_manager.py    # Chroma collection management + top-2 matching
│   │   └── image_storage.py
│   └── tasks.py
├── src/                          # EXISTING - ML pipeline
│   ├── core.py
│   ├── embedder.py               # Fix SSL, device auto-detect
│   ├── indexer.py                # REWRITE → Chroma-backed SKU index
│   ├── matcher.py                # Add ROI, itemId
│   ├── types.py                  # Add sku_name
│   ├── image_utils.py            # Add annotation drawing
│   └── ...
├── scripts/                      # EXISTING
├── chroma_data/                  # NEW (gitignored) - Chroma persistence dir
│   └── chroma.sqlite3            # + UUID collection dirs with HNSW indexes
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
- `chromadb>=1.5.0` — vector store with HNSW index, cosine similarity, metadata filtering, persistent storage (SQLite + Parquet)
- `python-jose` (if auth needed later)

### Removals
- `clip` (dead dependency)
- `scikit-learn` (no longer needed — Chroma handles similarity search)

### Moves
- `coremltools` → `[project.optional-dependencies] export`

---

## 8. Risk Notes

1. **Model loading time**: YOLOE + DINOv2 takes ~30-60s to load. Must happen at app startup, not per-request.
2. **GPU memory**: DINOv2 vitl14 + YOLOE on a single GPU may OOM with concurrent requests. Batch processing within a single request is fine; multiple concurrent requests may need request queuing.
3. **ONNX Runtime**: Not viable on 2GB GPU — lacks flash attention, materializes full attention matrix (687MB for batch=16). Available behind `--onnx` flag for future use with ≥4GB GPUs.
4. **Chroma memory**: HNSW index resides in RAM. For 1000 SKUs × 5 images × 768-dim = ~15MB — negligible. Scales linearly; 10K vectors ≈ 30MB. Not a concern at projected scale.
5. **Chroma persistence**: Uses SQLite under the hood. Must store on local SSD, **not network storage** (EFS/NFS causes SQLite corruption). For Docker: mount `/data` volume correctly.
6. **Image download failures**: OSS URLs may be slow/unavailable. Add timeout and retry logic.
7. **Chroma single-writer**: SQLite locks under concurrent writes. Acceptable for our use case (writes only happen during SKU CRUD, not during detection queries). Detection queries are read-only and don't contend.
8. **Top-2 post-processing overhead**: Minimal — pure Python dict operations on ~50-200 results per crop. No measurable latency impact.

---

## 9. Open Questions

1. **Result image hosting**: Should the API serve result images directly (`/static/...`) or upload them to OSS?
2. **Authentication**: No auth in current spec. Is this API public or behind an auth gateway?
3. **Concurrent detection limits**: Should we cap concurrent `/detect` requests to prevent GPU OOM?
4. **Chroma deployment mode**: `PersistentClient` (embedded, in-process) vs `HttpClient` (separate Docker container). Embedded is simpler and recommended for single-server. HttpClient enables independent scaling later.
5. **CLI backward compatibility**: Should `build_index.py` continue to produce `.npy` files for CLI mode, or migrate CLI to also use Chroma?

---

## 10. Chroma Vector Store — Detailed Design

### 10.1 Why Chroma

| Factor | Assessment |
|--------|------------|
| **Scale** | 100-2000 reference vectors (well within Chroma's comfort zone of millions) |
| **Incremental updates** | HNSW supports dynamic inserts/deletes — no full rebuild when SKUs change |
| **Metadata filtering** | `where={"enabled": True}` — filter disabled SKUs at query time |
| **Persistence** | SQLite + HNSW index files — zero-config, auto-persists |
| **Custom embeddings** | Pass DINOv2 vectors directly via `embeddings=` parameter, no embedding function needed |
| **Batch queries** | Native support — all detection crops in one API call |
| **Single-server** | `PersistentClient` runs in-process, no separate service to manage |

### 10.2 Collection Schema

**Collection**: `sku_embeddings`
**Distance**: `cosine` (configured at creation, immutable)

| Field | Chroma Role | Content |
|-------|-------------|---------|
| `id` | Document ID | `"{sku_id}__{media_id}"` — unique per reference image |
| `embedding` | Vector | DINOv2 768-dim (vits14=384, vitb14=768, vitl14=1024) |
| `metadata.sku_id` | Metadata | SKU identifier for grouping |
| `metadata.sku_name` | Metadata | Display name (e.g., "可口可乐330ml无糖听装") |
| `metadata.enabled` | Metadata | `True`/`False` — filtered via `where` clause |
| `metadata.class_name` | Metadata | "Bottle" / "Canned" / "Carton" etc. |
| `metadata.media_url` | Metadata | Original OSS URL of the reference image |

### 10.3 Indexer Rewrite (`src/indexer.py`)

The current `SKUIndexer` class does:
- Store embeddings in numpy array
- Manual cosine similarity via matrix multiply
- Custom top-2-per-SKU grouping with numpy partitioning
- Save/load to `.npy` + `.json` files

**New `SKUIndexer` responsibilities**:
- Wrap a Chroma `Collection`
- `add_references(sku_id, sku_name, media_id, embedding, metadata)` → `collection.upsert()`
- `delete_sku(sku_id)` → `collection.delete(where={"sku_id": sku_id})`
- `delete_media(sku_id, media_id)` → `collection.delete(ids=[f"{sku_id}__{media_id}"])`
- `set_enabled(sku_id, enabled)` → `collection.update(ids=..., metadatas=...)`
- `search_batch(query_embeddings)` → `collection.query()` + post-process top-2-per-SKU
- CLI backward compat: `build_from_directory()` reads local images and populates Chroma
- `save()` / `load()` become no-ops (Chroma handles persistence)

### 10.4 Top-2 Per SKU Matching Algorithm

The current matching logic computes "sum of top-2 cosine similarities per SKU" to score matches. This must be preserved with Chroma:

```python
def search_batch(self, query_embeddings: list[list[float]]) -> list[tuple[str, float]]:
    """Match detection crops against SKU references using top-2 sum."""
    num_skus = self._count_enabled_skus()
    n_results = max(num_skus * 2, 20)  # Enough to get 2+ hits per SKU
    
    results = self.collection.query(
        query_embeddings=query_embeddings,
        n_results=n_results,
        where={"enabled": True},
        include=["metadatas", "distances"],
    )
    
    matches = []
    for crop_idx in range(len(query_embeddings)):
        sku_scores: dict[str, list[float]] = {}
        for meta, distance in zip(
            results["metadatas"][crop_idx],
            results["distances"][crop_idx],
        ):
            similarity = 2.0 - distance  # Cosine distance → similarity
            sku_scores.setdefault(meta["sku_id"], []).append(similarity)
        
        best_sku, best_score = "", -1.0
        for sku_id, sims in sku_scores.items():
            top2 = sorted(sims, reverse=True)[:2]
            score = top2[0] if len(top2) == 1 else sum(top2)
            if score > best_score:
                best_sku, best_score = sku_id, score
        
        matches.append((best_sku, best_score))
    
    return matches
```

### 10.5 Chroma Client Lifecycle in FastAPI

```python
from contextlib import asynccontextmanager
import chromadb

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Init Chroma (PersistentClient = embedded, in-process)
    app.state.chroma_client = chromadb.PersistentClient(
        path=settings.CHROMA_PERSIST_DIR,
        settings=chromadb.Settings(anonymized_telemetry=False),
    )
    app.state.sku_collection = app.state.chroma_client.get_or_create_collection(
        name="sku_embeddings",
        configuration={"hnsw": {"space": "cosine"}},
    )
    
    # Startup: Load ML models
    app.state.embedder = DINOv2Embedder(model=settings.EMB_MODEL, device=settings.DEVICE)
    app.state.detector = YOLOE(settings.DET_MODEL)
    app.state.detector.set_classes(BEVERAGE_CONTAINER_CLASSES)
    
    # Startup: Init indexer with Chroma collection
    app.state.indexer = SKUIndexer(collection=app.state.sku_collection)
    
    yield
    
    # Shutdown: Clean close (Chroma v1.5.2+)
    app.state.chroma_client.close()
```

### 10.6 Chroma Persistence Storage

```
chroma_data/                          # Configured via CHROMA_PERSIST_DIR
├── chroma.sqlite3                    # Metadata, WAL, migrations
└── {collection_uuid}/                # One per collection
    ├── header.bin                    # HNSW index metadata
    ├── data_level0.bin               # Base layer vectors
    ├── link_lists.bin                # Graph adjacency lists
    └── ...
```

- **Auto-persists**: No explicit `save()` calls needed
- **Gitignored**: Add `chroma_data/` to `.gitignore`
- **Backup**: Copy entire directory before upgrades
- **Docker**: Mount as volume (`-v ./chroma_data:/data`)

### 10.7 Migration Path (Current → Chroma)

1. **Phase 0**: Rewrite `SKUIndexer` to use Chroma internally. CLI (`main.py`, `build_index.py`) continues to work — now writes to Chroma instead of `.npy` files.
2. **Phase 1**: API server uses the same Chroma-backed `SKUIndexer`. Both CLI and API share the same Chroma persistence directory.
3. **Data migration**: One-time script reads existing `index/embeddings_*.npy` + `metadata_*.json` → populates Chroma collection.

---

## 11. Embedding Model Research & Upgrade Paths

> Researched 2026-05-07. Default model changed to DINOv2 ViT-S/14 (384-dim) for 2GB GPU deployment. ViT-B/14 (768-dim) still supported.

### 11.1 Current Model Assessment

**DINOv2 ViT-B/14** is well-suited for SKU matching:
- Self-supervised (no text bias) — learns purely visual features, ideal for distinguishing visually-similar packaging
- Best speed/quality balance in the DINOv2 family
- Oxford-H retrieval mAP: 49.5 (peaks at ViT-L/14 with 54.0)
- iNaturalist fine-grained: 76.3%

### 11.2 Alternative Models (Ranked for This Use Case)

#### Tier 1: Drop-in Replacements (No Fine-Tuning)

| Model | Emb. Dim | Params | Size (MB) | Pros | Cons |
|-------|----------|--------|-----------|------|------|
| **DINOv2 ViT-L/14** | 1024 | 300M | 1,189 | Best retrieval mAP (54.0) | 4× larger; slower; needs index rebuild |
| **SigLIP ViT-SO400M/14** | 1152 | 428M | ~1,700 | Best CLIP accuracy/speed (82% zero-shot) | Text bias may group similar SKUs; 1152-dim |
| **Marqo-Ecommerce-B** | 768 | 203M | ~800 | Pre-tuned on e-commerce; same 768-dim | Not beverage-specific |

#### Tier 2: Domain Adaptation (Requires Fine-Tuning on SKU Data)

| Approach | How | Effort | Expected Gain |
|----------|-----|--------|---------------|
| **DINOv2-B + ArcFace projection** (recommended) | Freeze DINOv2 backbone, train linear + ArcFace head on SKU pairs. Projects 768→256 dim. Pattern validated by Trendyol's DINOv2-Ecom model. | Medium — need labeled SKU image pairs | Better discrimination at smaller dim; marginal speed gain only from smaller vectors (ViT forward pass unchanged) |
| **Fine-tune on beverage images** | LoRA or full fine-tune of DINOv2 on beverage container images | High | Significant for unusual packaging |

**Note on ArcFace speed**: The ArcFace projection head (768→256 linear layer) adds negligible inference overhead. The ViT backbone forward pass — the actual bottleneck — remains identical. The speed improvement comes only from smaller embedding dimensions in vector search, which is negligible at current catalog scales (<10K vectors).

#### Tier 3: Multi-Stage Pipeline (Large Catalogs)

| Stage 1 (Coarse) | Stage 2 (Fine) | Use When |
|-------------------|----------------|----------|
| MobileCLIP-S2 (36M, 3.6ms) → top-50 | DINOv2-B/14 → re-rank | 1000+ SKUs |
| SigLIP SO400M → top-30 | DINOv2-B/14 → re-rank | 500+ SKUs |

### 11.3 Recommendation

For current scale (<200 SKUs) with 2GB GPU: **DINOv2 ViT-S/14** (384-dim, ~85MB) is the default for GPU memory constraints. ViT-B/14 remains the best choice when VRAM allows.

When accuracy on visually-similar SKUs becomes a bottleneck: implement **DINOv2-B + ArcFace projection** (Tier 2). This is the highest-ROI upgrade — uses the existing backbone, requires only labeled SKU pairs (which the API collects via `/fix` corrections), and produces more discriminative embeddings.

Model swapping (Tier 1) is lower effort but likely yields smaller improvements than domain adaptation.

---

*Plan updated on 2026-05-08. FP16 inference, concentration score, vits14 default, ONNX status updated.*
