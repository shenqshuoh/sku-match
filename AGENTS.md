# PROJECT KNOWLEDGE BASE

**Generated:** 2026-05-29
**Project:** sku-match v0.2.0

## OVERVIEW

YOLOE-based bottle & can detection with DINOv2 SKU matching, served via FastAPI REST API. Supports CLI and API server modes. Chroma vector store for SKU embeddings.

## STRUCTURE

```
sku-match/
├── main.py                   # CLI entry point (matching/detection modes)
├── pyproject.toml            # v0.2.0, deps: ultralytics, chromadb, fastapi, sqlalchemy, etc.
├── AGENTS.md                 # This file — project knowledge base
├── uv.lock                   # uv lockfile
├── .env                      # Environment variables
├── src/
│   ├── __init__.py           # Exports: detect, DINOv2Embedder, SKUIndexer, SKUMatcher, etc.
│   ├── core.py               # CLI detection: detect(), _save_result()
│   ├── embedder.py           # DINOv2Embedder: auto device detect, FP16 on CUDA, optional ONNX backend
│   ├── indexer.py            # Chroma-backed SKUIndexer: build, add/delete/enable/search
│   ├── matcher.py            # SKUMatcher: detection + SKU matching pipeline
│   ├── types.py              # Dataclasses: Detection, SKUReference, SKUMatch, VectorMatch
│   ├── utils.py              # Shared: detect_device(), free_gpu_memory(), disable_ssl_verification(), embedding_to_list(), configure_ultralytics_weights()
│   ├── reference_processor.py # ReferenceProcessor: crop→embed→index pipeline
│   ├── image_utils.py        # isolate_object, save_crop, process_detection_crops, draw_annotations, CJK font rendering via PIL (Noto Sans CJK)
│   ├── yoloe2o365.py         # CoreML export utility
│   └── classes/
│       ├── beverage_cls.py       # BEVERAGE_CONTAINER_CLASSES (7 items)
│       └── objects365_classes.py  # 365-class list (legacy, used for name lookups)
├── api/
│   ├── __init__.py
│   ├── app.py                # FastAPI + lifespan: loads detector, embedder, Chroma, services
│   ├── auth.py               # API key auth + per-IP rate limiting (disabled by default)
│   ├── config.py             # pydantic-settings: DET_MODEL, EMB_MODEL, DEVICE, MATCH_CONF, etc.
│   ├── database.py           # SQLAlchemy async engine + get_db + init_db
│   ├── models.py             # ORM: SKU, SKUMedia, RecognitionLog, TrainJob
│   ├── schemas.py            # Pydantic request/response models
│   ├── tasks.py              # asyncio task runner for embedding jobs
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── goods.py          # SKU CRUD (new/update/delete/enable/list/media)
│   │   ├── recognition.py    # POST /detect, POST /fix
│   │   ├── logs.py           # GET /recognition/get
│   │   └── system.py         # GET /train-status/get
│   └── services/
│       ├── __init__.py
│       ├── recognition.py    # RecognitionService: download → detect → embed → match → annotate
│       └── image_storage.py  # ImageStorage: download (local/URL), get result path/URL, upload_to_qiniu, cleanup_download
├── scripts/
│   ├── build_index.py        # Build Chroma index from reference images
│   ├── crop_reference.py     # Crop raw refs with YOLOE
│   ├── export_onnx.py        # Export DINOv2 to ONNX format
│   ├── init_and_download.sh  # First-time setup: install deps, download weights
│   ├── sync_local.sh         # Sync from remote to local
│   ├── sync_remote.sh        # Sync to remote server
│   └── test_detection.py     # Detection test script
├── docs/
│   ├── API.md.bak            # Chinese API requirements spec (archived)
│   ├── API_GUIDE.md          # API usage guide
│   ├── AUDIT.md              # Codebase audit findings
│   ├── CLEANUP_PLAN.md       # Cleanup progress tracker
│   ├── CONSIDERATIONS.md     # Evaluated improvements (SAHI, FAISS, Qdrant, etc.)
│   ├── IMAGE_SIMILARITY_SEARCH_REPORT.md
│   ├── PERF_PLAN.md          # Performance improvement plan
│   ├── PLAN.md               # API server implementation plan
│   ├── QINIU.md              # Qiniu CDN integration docs
│   └── SETUP.md              # Setup instructions
├── test/                     # Test images
├── data/
│   ├── images/               # Input images (CLI mode)
│   ├── references/           # SKU reference images (sku_id/*.jpg)
│   └── references_raw/       # Original photos to be cropped
├── models/                   # YOLOE weights, DINOv2 .pth weights, mobileclip2_b.ts
├── vendor/
│   ├── dinov2/               # Full DINOv2 source (vendored)
│   └── clip_package/         # Vendored clip package (GFW-safe)
├── chroma_data/              # Chroma persistence (gitignored)
├── results/                  # API result images (annotated + downloads)
├── runs/                     # CLI output directory
└── tests/
    └── test_detection.py     # Smoke test
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| CLI entry | main.py | Default: SKU matching. Use --detection-only for detection only |
| API entry | api/app.py | FastAPI + lifespan, uvicorn server |
| API config | api/config.py | pydantic-settings, .env file |
| API auth | api/auth.py | API key verification + per-IP rate limiting (disabled by default) |
| Detection logic | src/core.py | detect() function (CLI) |
| DINOv2 embedder | src/embedder.py | 3 variants: vits14(384-dim), vitb14(768-dim), vitl14(1024-dim). Optional ONNX backend |
| SKU indexer | src/indexer.py | Chroma-backed: search_batch with top-2-per-SKU scoring |
| Detection + matching | src/matcher.py | SKUMatcher class with crop saving |
| Reference processing | src/reference_processor.py | crop→embed→index pipeline, build_from_directory() |
| Recognition service | api/services/recognition.py | Full pipeline: detect → embed → match → annotate |
| Image storage | api/services/image_storage.py | Download, upload to Qiniu, cleanup, old file deletion |
| Shared utilities | src/utils.py | detect_device(), free_gpu_memory(), disable_ssl_verification(), embedding_to_list(), configure_ultralytics_weights() |
| Image utilities | src/image_utils.py | isolate_object, save_crop, draw_annotations, CJK font rendering via PIL (Noto Sans CJK) |
| Build ref index | scripts/build_index.py | --model arg for embedding variant selection |
| Crop raw refs | scripts/crop_reference.py | Pre-process raw photos with YOLOE detection |
| API routes | api/routes/ | goods.py (SKU CRUD), recognition.py (detect/fix), logs.py, system.py |
| ORM models | api/models.py | SKU, SKUMedia, RecognitionLog, TrainJob |
| Pydantic schemas | api/schemas.py | Request/response models |
| Embedding task runner | api/tasks.py | Async embedding jobs with progress tracking |
| Database setup | api/database.py | SQLAlchemy async engine, pool config |
| Performance plan | docs/PERF_PLAN.md | GPU optimization tracking |

## CODE MAP

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| main() | function | main.py:163 | CLI entry (argparse) — delegates to run_matching() or run_detection() |
| run_detection() | function | main.py:30 | Detection-only mode (--detection-only) |
| run_matching() | function | main.py:44 | SKU matching mode (default) |
| get_next_match_dir() | function | main.py:20 | Auto-increment match output directory |
| _save_image_results() | function | main.py:124 | Save match results to JSON + copy original |
| detect() | function | src/core.py:11 | Core YOLOE inference (CLI) — batch prediction with crop saving |
| _save_result() | function | src/core.py:65 | Save detection JSON + masked crops |
| DINOv2Embedder | class | src/embedder.py:28 | DINOv2 embedding wrapper (FP16 on CUDA, optional ONNX backend) |
| DINOv2Variant | type | src/embedder.py:16 | Literal type: "dinov2_vits14"/"dinov2_vitb14"/"dinov2_vitl14" |
| DIMENSIONS | dict | src/embedder.py:17 | Model name → embedding dimension mapping |
| embed() | method | src/embedder.py:110 | Embed single PIL Image → np.ndarray |
| embed_batch() | method | src/embedder.py:113 | Embed batch of PIL Images → np.ndarray |
| embed_path() | method | src/embedder.py:137 | Embed image from file path |
| VectorMatch | class | src/indexer.py:23 | Single vector-level match result (sku_id, similarity, rank, media_url) |
| SKUIndexer | class | src/indexer.py:64 | Chroma-backed SKU index with top-2 scoring |
| _softmax() | function | src/indexer.py:43 | Softmax probability distribution over SKU scores |
| concentration_score() | function | src/indexer.py:53 | Top-1 share of top-K probability mass |
| init_collection() | method | src/indexer.py:94 | Initialize Chroma collection from persist dir |
| build() | method | src/indexer.py:116 | Populate collection from SKUReference list |
| add_reference() | method | src/indexer.py:144 | Upsert single reference embedding |
| delete_sku() | method | src/indexer.py:169 | Delete all vectors for a SKU |
| delete_media() | method | src/indexer.py:174 | Delete single media vector |
| set_enabled() | method | src/indexer.py:180 | Toggle SKU enabled flag in Chroma metadata |
| search() | method | src/indexer.py:191 | Single-query similarity search |
| search_batch() | method | src/indexer.py:195 | Batch similarity search with top-2-per-SKU scoring |
| get_sku_name() | method | src/indexer.py:278 | Get SKU name from cache |
| load() | method | src/indexer.py:283 | Load collection from directory |
| SKUMatcher | class | src/matcher.py:20 | Detection + SKU matching with crop saving |
| match_images() | method | src/matcher.py:42 | Match SKUs across images (detect → embed → search) |
| _detect_all() | method | src/matcher.py:133 | Run YOLOE detection on all images in batches |
| from_index_dir() | classmethod | src/matcher.py:186 | Factory: load models + index from directory |
| Detection | dataclass | src/types.py:8 | Detection result: bbox, confidence, class_name, class_id, mask |
| SKUReference | dataclass | src/types.py:17 | Reference entry: sku_id, sku_name, image_path, embedding |
| SKUMatch | dataclass | src/types.py:25 | Match result: detection, sku_id, match_score, concentration, distribution |
| detect_device() | function | src/utils.py:20 | Auto-detect: cuda → mps → cpu |
| free_gpu_memory() | function | src/utils.py:29 | Release cached GPU memory (gc + cuda.empty_cache) |
| disable_ssl_verification() | function | src/utils.py:36 | Disable SSL cert verification (macOS compat) |
| embedding_to_list() | function | src/utils.py:42 | Convert numpy/torch embedding to plain list for Chroma |
| configure_ultralytics_weights() | function | src/utils.py:13 | Sets ultralytics weights_dir to local models/ |
| ProcessResult | dataclass | src/reference_processor.py:21 | Result from process_and_add: success + crop_path |
| select_best_detection() | function | src/reference_processor.py:27 | Select smallest center-covering detection box |
| ReferenceProcessor | class | src/reference_processor.py:50 | Processes raw reference images: detect → crop → embed → index |
| crop_reference() | method | src/reference_processor.py:69 | Detect primary object and return cropped PIL Image |
| process_and_add() | method | src/reference_processor.py:115 | Full pipeline: crop → embed → index → save crop to temp |
| build_from_directory() | method | src/reference_processor.py:146 | Build index from directory of reference images |
| delete_sku_references() | method | src/reference_processor.py:220 | Delete all vectors for a SKU |
| delete_media_reference() | method | src/reference_processor.py:224 | Delete single media vector |
| set_sku_enabled() | method | src/reference_processor.py:228 | Toggle SKU enabled flag |
| update_sku_name() | method | src/reference_processor.py:232 | Update SKU name in Chroma metadata |
| normalize_mask() | function | src/image_utils.py:11 | Normalize mask to uint8 (0 or 255) |
| isolate_object() | function | src/image_utils.py:25 | Isolate object using binary mask with optional exclusion |
| save_crop() | function | src/image_utils.py:59 | Save a cropped PIL Image |
| process_detection_crops() | function | src/image_utils.py:74 | Process all detections and save masked crops |
| extract_binary_masks() | function | src/image_utils.py:122 | Extract binary masks from YOLOE result |
| _load_cjk_font() | function | src/image_utils.py:155 | Load Noto Sans CJK font for Chinese text rendering |
| draw_annotations() | function | src/image_utils.py:173 | Draw bounding boxes and SKU labels on image |
| BEVERAGE_CONTAINER_CLASSES | list | src/classes/beverage_cls.py:10 | 7 beverage container class names for YOLOE |
| Settings | class | api/config.py:6 | pydantic-settings configuration |
| settings | instance | api/config.py:42 | Global settings singleton |
| verify_api_key() | function | api/auth.py:17 | Verify X-API-Key header (disabled if API_KEY empty) |
| check_rate_limit() | function | api/auth.py:32 | Per-IP rate limiting (disabled if RATE_LIMIT=0) |
| engine | instance | api/database.py:8 | SQLAlchemy async engine |
| async_session | instance | api/database.py:16 | Async session factory |
| get_db() | function | api/database.py:19 | FastAPI dependency: yield async DB session |
| init_db() | function | api/database.py:24 | Create all tables |
| Base | class | api/models.py:7 | SQLAlchemy declarative base |
| SKU | class | api/models.py:11 | ORM: sku_id, sku_name, enabled, train_status |
| SKUMedia | class | api/models.py:27 | ORM: media_id, sku_id, media_url, media_type |
| RecognitionLog | class | api/models.py:40 | ORM: task_id, request_json, ai_result_json, user_correction_json, visual_image_path |
| TrainJob | class | api/models.py:52 | ORM: train_job_id, sku_id, status, progress, embedding_failed |
| DetectRequest | class | api/schemas.py:6 | Pydantic: taskId, mode, files, roiRect |
| FixRequest | class | api/schemas.py:41 | Pydantic: taskId, fixItems |
| FixItem | class | api/schemas.py:34 | Pydantic: fixType, itemId, roiRect, skuId |
| SKUNewRequest | class | api/schemas.py:53 | Pydantic: skuId, skuName, files, trainJobId |
| SKUUpdateRequest | class | api/schemas.py:75 | Pydantic: skuId, skuName |
| SKUDeleteRequest | class | api/schemas.py:87 | Pydantic: skuId |
| SKUEnableRequest | class | api/schemas.py:98 | Pydantic: skuId, enabled |
| SKUMediaRequest | class | api/schemas.py:108 | Pydantic: skuId, action (add/delete), media list |
| ApiResponse | class | api/schemas.py:135 | Pydantic: code (1=ok, 0=error), data, msg |
| start_embed_task() | function | api/tasks.py:15 | Start async embedding task for new SKU |
| get_task_status() | function | api/tasks.py:122 | Check running task status |
| RecognitionService | class | api/services/recognition.py:22 | Full recognition pipeline (sync) |
| recognize() | method | api/services/recognition.py:44 | Single-image recognition: detect → embed → match → annotate |
| ImageStorage | class | api/services/image_storage.py:13 | Download images, manage result paths, Qiniu upload |
| download_image() | method | api/services/image_storage.py:54 | Download image (local path or URL) |
| cleanup_download() | method | api/services/image_storage.py:89 | Remove downloaded temp file |
| cleanup_old_results() | method | api/services/image_storage.py:36 | Delete annotated images older than max_age_hours |
| get_result_path() | method | api/services/image_storage.py:98 | Get annotated image path for task_id |
| get_result_url() | method | api/services/image_storage.py:103 | Get annotated image relative URL |
| upload_to_qiniu() | method | api/services/image_storage.py:130 | Upload file to Qiniu and return CDN URL |
| _fetch_qiniu_token() | method | api/services/image_storage.py:108 | Fetch fresh upload token from internal service |
| UnicodeJSONResponse | class | api/app.py:170 | JSONResponse with ensure_ascii=False |
| main() | function | api/app.py:203 | Entry point for sku-match-api console script |
| _get_sku_or_error() | function | api/routes/goods.py:31 | Helper: fetch SKU or return ApiResponse error |
| _to_full_url() | function | api/routes/goods.py:143 | Convert relative path to full URL |

## API ENDPOINTS

| Method | Path | Handler | Notes |
|--------|------|---------|-------|
| POST | /api/v1/recognition/detect | recognition.detect | Detect + match SKUs in image. Deduplicates by taskId. |
| POST | /api/v1/recognition/fix | recognition.fix | Apply corrections to detection result. Rejects unknown taskIds. |
| POST | /api/v1/goods/sku/new | goods.new_sku | Create SKU + start embedding task |
| POST | /api/v1/goods/sku/update | goods.update_sku | Update SKU name (DB + Chroma metadata) |
| POST | /api/v1/goods/sku/delete | goods.delete_sku | Delete SKU + cascade media + Chroma vectors |
| POST | /api/v1/goods/sku/enable | goods.enable_sku | Toggle SKU enabled flag (DB + Chroma) |
| GET  | /api/v1/goods/sku/list | goods.list_skus | Paginated SKU list with media, keyword search |
| POST | /api/v1/goods/sku/media | goods.manage_media | Add/delete SKU media (add embeds + uploads crop) |
| GET  | /api/v1/logs/recognition/get | logs.get_log | Query recognition log by taskId |
| GET  | /api/v1/system/train-status/get | system.train_status | Query train job status (includes embedding_failed) |
| GET  | /results/{path} | app.static | Static file serving for result images |
| GET  | /health | app.health | Health check |

> **Note:** All `/api/v1/*` endpoints have `Depends(verify_api_key)` and `Depends(check_rate_limit)` (disabled by default via config — empty `API_KEY` and `RATE_LIMIT=0`).

## ENTRY POINTS

| Command | Script | Purpose |
|---------|--------|---------|
| `uv run sku-match` | main.py | SKU matching mode (default) |
| `uv run sku-match --detection-only` | main.py | Detection only mode |
| `uv run sku-match --det-model X --emb-model Y` | main.py | Custom models |
| `uv run sku-match-api` | api/app.py | Start API server |
| `uv run python scripts/build_index.py -r data/references/ -o chroma_data/ -m dinov2_vitb14` | build_index.py | Build Chroma index |
| `uv run python scripts/crop_reference.py -r data/references_raw/ -o data/references/` | crop_reference.py | Crop raw reference photos |
| `uv run python tests/test_detection.py` | test_detection.py | Smoke tests |

## CONVENTIONS

- **Type hints**: Full on function signatures (Python 3.10+ union: `str | None`)
- **Imports**: Absolute imports (`from src.core import detect`)
- **Linting**: Ruff configured (line-length: 100, py310 target)
- **Device**: Auto-detect via `detect_device()` (cuda → mps → cpu)
- **Detection classes**: `BEVERAGE_CONTAINER_CLASSES` (7 items: bottle, canned, carton, empty paper box, full paper box, strawed drink, keg)
- **Index**: Chroma vector store with cosine similarity, top-2-per-SKU scoring. Unified path: `chroma_data/`
- **API prefix**: `/api/v1/`
- **API response**: All endpoints return `ApiResponse(code=1/0, data=..., msg="success")`
- **Async pattern**: CPU/GPU-bound work runs via `asyncio.to_thread()` or `loop.run_in_executor()` to avoid blocking event loop
- **GPU concurrency**: API uses dedicated `ThreadPoolExecutor(max_workers=1)` for GPU inference
- **Warm-up**: YOLOE warm-up prediction runs at API startup (dummy 640x640 image)
- **FP16**: Both YOLOE and DINOv2 use `model.half()` at load time when device=cuda
- **CJK rendering**: Annotations use PIL + Noto Sans CJK font for Chinese character support
- **Qiniu CDN**: Annotated images uploaded to Qiniu, `matched_image` returns full CDN URL. Falls back to local path with `qiniu_upload_failed=True` flag on failure.
- **DB pool**: SQLAlchemy pool_size=5, max_overflow=10, pool_recycle=3600, pool_pre_ping=True
- **Temp cleanup**: Downloaded images cleaned up after processing in both recognition routes and embedding tasks
- **match_conf threshold**: Filters low-confidence matches — below threshold sets sku_id="", sku_name="", match_score=0.0 (marks as unmatched)
- **sku_distribution**: Returns only top 5 entries by probability (sorted descending)
- **Annotated image cleanup**: Background task runs hourly, deletes files older than `RESULTS_MAX_AGE_HOURS` (default 24)
- **Rate limiter**: Uses `asyncio.Lock` for atomicity, prunes empty IP entries when >1000 IPs tracked
- **_running_tasks**: Auto-cleaned via `add_done_callback` — tasks remove themselves when done
- **Fix endpoint**: Rejects unknown taskIds with `ApiResponse(code=0, msg="taskId '...' not found")`
- **Duplicate detection**: Detect endpoint rejects duplicate taskIds; new_sku rejects duplicate skuIds and trainJobIds
- **Crop naming**: `{image_name}_{index}_{sku_id}.jpg`
- **Index operations**: All Chroma writes go through `ReferenceProcessor` (sync), called via `asyncio.to_thread()` in routes
- **Unicode JSON**: API uses `UnicodeJSONResponse` (ensure_ascii=False) as default response class
- **uv required**: All commands use `uv run` — never `python` or `pip` directly. No manual venv activation. `uv sync` for install, `uv run` for execution.

## ANTI-PATTERNS (THIS PROJECT)

- **sys.path hack**: tests/test_detection.py, scripts/, and scripts/export_onnx.py manipulate sys.path
- **No docstrings**: Minimal documentation on most functions
- **SSL override**: `ssl._create_default_https_context` disabled in utils.py (macOS compat, fallback path only)
- **GFW vendor hacks**: clip vendored locally, mobileclip2_b.ts tracked in models/, aliyun PyPI mirror configured

## COMMANDS

```bash
# Setup
uv sync                           # Install all deps (default CPU torch)
uv sync --extra cu128             # Install with CUDA 12.8 torch
uv sync --extra cu126             # Install with CUDA 12.6 torch
uv sync --extra dev               # Install dev tools (pytest, ruff)

# CLI: SKU matching mode (default)
uv run sku-match

# CLI: SKU matching with custom thresholds
uv run sku-match --conf 0.3 --match-conf 0.5 --match-concentration 0.0 --match-verbose

# CLI: Custom models
uv run sku-match --det-model models/yoloe-26l-seg.pt --emb-model dinov2_vitb14

# CLI: FP16 + ONNX mode (GPU only)
uv run sku-match --swap --onnx

# CLI: Detection only mode
uv run sku-match --detection-only

# API: Start server
uv run sku-match-api

# API: With debug/reload
DEBUG=true uv run sku-match-api

# API: With Qiniu config
QINIU_ACCESS_KEY=xxx QINIU_SECRET_KEY=xxx QINIU_BUCKET=xxx uv run sku-match-api

# Build Chroma index from reference images
uv run python scripts/build_index.py -r data/references/ -o chroma_data/ -m dinov2_vitb14

# Build index with different embedding model
uv run python scripts/build_index.py -r data/references/ -o chroma_data/ -m dinov2_vits14

# Crop raw reference photos
uv run python scripts/crop_reference.py -r data/references_raw/ -o data/references/

# Export DINOv2 to ONNX
uv run python scripts/export_onnx.py --model dinov2_vits14

# Run smoke test
uv run python tests/test_detection.py

# Lint
uv run ruff check .

# Deploy: Sync to remote and restart
./scripts/sync_remote.sh --restart

# Linux: Install CJK font (required for Chinese text in annotations)
apt-get install fonts-noto-cjk
```

## NOTES

- **Default mode**: SKU matching (not detection-only)
- **Embedding models**: dinov2_vits14 (384-dim, default), dinov2_vitb14 (768-dim), dinov2_vitl14 (1024-dim)
- **Chroma index**: PersistentClient with HNSW cosine similarity. Incremental add/delete — no full rebuild needed. Unified path: `chroma_data/` across CLI, API, and build scripts.
- **Match scoring**: Top-2-per-SKU cosine similarity sum → softmax probability distribution (0–1 range). Threshold default 0.5. Concentration score (top-1 share of top-K probability mass) replaces match_ratio. Detection rank positions (top2_ranks) tracked per match.
- **Device auto-detect**: `detect_device()` returns cuda → mps → cpu
- **SSL fix**: Disabled certificate verification for macOS compatibility (fallback path only when local weights unavailable)
- **First run**: Downloads YOLOE model (~400MB). DINOv2 weights vendored locally at models/*.pth (no download needed).
- **API state**: Models loaded once at FastAPI lifespan startup, reused across requests
- **ReferenceProcessor**: Unified crop→embed→index pipeline. Crops via YOLOE detection + mask isolation, falls back to full-image embedding. Used directly by API routes.
- **Detection confidence**: `--conf` flag (CLI) and `DET_CONF` setting (API) control YOLOE detection threshold. Passed to `predict(conf=)`. retina_masks=False to prevent OOM on high-res images.
- **GFW compatibility**: `clip` vendored locally at `vendor/clip_package/`. `mobileclip2_b.ts` in `models/` avoids GitHub download. PyPI via aliyun mirror.
- **Index operations**: All Chroma writes go through `ReferenceProcessor` (sync), called via `asyncio.to_thread()` in routes.
- **FP16 inference**: Both YOLOE and DINOv2 use model.half() at load time when device=cuda. ~1.8x speedup over FP32 on L20 GPU.
- **ONNX backend**: Optional via --onnx flag / USE_ONNX setting. Export with scripts/export_onnx.py. Currently not viable on 2GB GPU (ONNX Runtime lacks flash attention).
- **DINOv2 vendoring**: Full dinov2 source at vendor/dinov2/ + weights at models/dinov2_*_pretrain.pth. No internet needed at runtime.
- **Performance**: See docs/PERF_PLAN.md for optimization details. FP16 ~1.8x faster. GPU thread pool prevents OOM under concurrent load.
- **Qiniu integration**: Annotated result images uploaded to Qiniu cloud storage. Token cached until 60s before deadline. Key format: `sku-match/{YYYY-MM/DD}/{filename}`. Region: South China (z2). CDN URL rewrite for faster domestic downloads (iovip origin pull).
- **API auth**: Optional API key authentication via `X-API-Key` header. Rate limiting per-IP (requests/minute). Both disabled by default (empty API_KEY, RATE_LIMIT=0).
- **Input validation**: Pydantic Field constraints (max_length, min_length), Literal types for mode/action, non-empty string validators on all request models.
- **Console scripts**: `sku-match` (CLI) and `sku-match-api` (API server) defined in pyproject.toml [project.scripts].
