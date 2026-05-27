# PROJECT KNOWLEDGE BASE

**Generated:** 2026-05-09
**Project:** sku-match v0.2.0

## OVERVIEW

YOLOE-based bottle & can detection with DINOv2 SKU matching, served via FastAPI REST API. Supports CLI and API server modes. Chroma vector store for SKU embeddings.

## STRUCTURE

```
sku-match/
├── main.py                   # CLI entry point (matching/detection modes)
├── pyproject.toml            # v0.2.0, deps: ultralytics, chromadb, fastapi, sqlalchemy, etc.
├── AGENTS.md                 # This file — project knowledge base
├── API.md                    # Chinese requirements spec (source of truth for API design)
├── PLAN.md                   # API server implementation plan (COMPLETED)
├── CONSIDERATIONS.md         # Evaluated improvements (SAHI, FAISS, Qdrant, etc.)
├── CLEANUP_PLAN.md           # Cleanup progress tracker
├── PERF_PLAN.md              # Performance improvement plan
├── src/
│   ├── __init__.py           # Exports: detect, DINOv2Embedder, SKUIndexer, SKUMatcher, etc.
│   ├── core.py               # CLI detection: detect(), _save_result()
│   ├── embedder.py           # DINOv2Embedder: auto device detect, FP16 on CUDA, optional ONNX backend
│   ├── indexer.py            # Chroma-backed SKUIndexer: build, add/delete/enable/search
│   ├── matcher.py            # SKUMatcher: detection + SKU matching pipeline
│   ├── types.py              # Dataclasses: Detection, SKUReference, SKUMatch
│   ├── utils.py              # Shared: detect_device(), free_gpu_memory(), disable_ssl_verification(), embedding_to_list(), configure_ultralytics_weights()
│   ├── reference_processor.py # ReferenceProcessor: crop→embed→index pipeline
│   ├── image_utils.py        # isolate_object, save_crop, process_detection_crops, draw_annotations, CJK font rendering via PIL (Noto Sans CJK)
│   ├── yoloe2o365.py         # CoreML export utility
│   └── classes/
│       ├── beverage_cls.py       # BEVERAGE_CONTAINER_CLASSES (7 items)
│       └── objects365_classes.py  # 365-class list (legacy, used for name lookups)
├── api/
│   ├── app.py                # FastAPI + lifespan: loads detector, embedder, Chroma, services
│   ├── auth.py               # API key auth + per-IP rate limiting (disabled by default)
│   ├── config.py             # pydantic-settings: DET_MODEL, EMB_MODEL, DEVICE, MATCH_CONF, etc.
│   ├── database.py           # SQLAlchemy async engine + get_db + init_db
│   ├── models.py             # ORM: SKU, SKUMedia, RecognitionLog, TrainJob
│   ├── schemas.py            # Pydantic request/response models
│   ├── tasks.py              # asyncio task runner for embedding jobs
│   ├── routes/
│   │   ├── goods.py          # SKU CRUD (new/update/delete/enable/list/media)
│   │   ├── recognition.py    # POST /detect, POST /fix
│   │   ├── logs.py           # GET /recognition/get
│   │   └── system.py         # GET /train-status/get
│   └── services/
│       ├── recognition.py    # RecognitionService: download → detect → embed → match → annotate
│       └── image_storage.py  # ImageStorage: download (local/URL), get result path/URL, upload_to_qiniu, cleanup_download
├── scripts/
│   ├── build_index.py        # Build Chroma index from reference images
│   ├── crop_reference.py     # Crop raw refs with YOLOE
│   └── export_onnx.py        # Export DINOv2 to ONNX format
├── data/
│   ├── images/               # Input images (CLI mode)
│   ├── references/           # SKU reference images (sku_id/*.jpg)
│   └── references_raw/       # Original photos to be cropped
├── models/                   # YOLOE weights
├── chroma_data/              # Chroma persistence (gitignored)
├── results/                  # API result images (annotated)
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
| Recognition service | api/services/recognition.py | Full pipeline: download → detect → embed → match → annotate |
| Shared utilities | src/utils.py | detect_device(), free_gpu_memory(), disable_ssl_verification(), embedding_to_list(), configure_ultralytics_weights() |
| Image utilities | src/image_utils.py | isolate_object, save_crop, draw_annotations, CJK font rendering via PIL (Noto Sans CJK) |
| Build ref index | scripts/build_index.py | --model arg for embedding variant selection |
| Crop raw refs | scripts/crop_reference.py | Pre-process raw photos with YOLOE detection |
| API routes | api/routes/ | goods.py (SKU CRUD), recognition.py (detect/fix), logs.py, system.py |
| ORM models | api/models.py | SKU, SKUMedia, RecognitionLog, TrainJob |
| Pydantic schemas | api/schemas.py | Request/response models |
| Performance plan | PERF_PLAN.md | GPU optimization tracking |

## CODE MAP

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| main() | function | main.py | CLI entry (default: matching mode) |
| run_detection() | function | main.py | Detection-only mode (--detection-only) |
| run_matching() | function | main.py | SKU matching mode (default) |
| detect() | function | src/core.py | Core YOLO inference (CLI) |
| DINOv2Embedder | class | src/embedder.py | DINOv2 embedding wrapper (FP16 on CUDA, optional ONNX backend) |
| DINOv2Variant | type | src/embedder.py | Literal type: vits14/vitb14/vitl14 |
| SKUIndexer | class | src/indexer.py | Chroma-backed SKU index with top-2 scoring |
| concentration_score | function | src/indexer.py | Top-1 share of top-K probability mass |
| SKUMatcher | class | src/matcher.py | Detection + SKU matching with crop saving |
| detect_device() | function | src/utils.py | Auto-detect: cuda → mps → cpu |
| free_gpu_memory() | function | src/utils.py | Release cached GPU memory |
| disable_ssl_verification() | function | src/utils.py | Disable SSL cert verification (macOS compat) |
| embedding_to_list() | function | src/utils.py | Convert numpy/torch embedding to plain list |
| configure_ultralytics_weights | function | src/utils.py | Sets ultralytics weights_dir to local models/ |
| _load_cjk_font() | function | src/image_utils.py | Load Noto Sans CJK font for Chinese text rendering |
| ReferenceProcessor | class | src/reference_processor.py | Crop→embed→index pipeline with mask isolation |
| RecognitionService | class | api/services/recognition.py | Full recognition pipeline (sync) |
| ImageStorage | class | api/services/image_storage.py | Download images, manage result paths |
| upload_to_qiniu() | method | api/services/image_storage.py | Upload annotated image to Qiniu cloud storage |
| cleanup_download() | method | api/services/image_storage.py | Remove downloaded temp image after processing |
| verify_api_key() | function | api/auth.py | Verify X-API-Key header (disabled if API_KEY empty) |
| check_rate_limit() | function | api/auth.py | Per-IP rate limiting (disabled if RATE_LIMIT=0) |
| Settings | class | api/config.py | pydantic-settings configuration |
| export_yoloe_to_lvis() | function | src/yoloe2o365.py | CoreML export |

## API ENDPOINTS

| Method | Path | Handler | Notes |
|--------|------|---------|-------|
| POST | /api/v1/recognition/detect | recognition.detect | Detect + match SKUs in image |
| POST | /api/v1/recognition/fix | recognition.fix | Apply corrections to detection result |
| POST | /api/v1/goods/sku/new | goods.new_sku | Create SKU + start embedding task |
| POST | /api/v1/goods/sku/update | goods.update_sku | Update SKU name |
| POST | /api/v1/goods/sku/delete | goods.delete_sku | Delete SKU + Chroma vectors |
| POST | /api/v1/goods/sku/enable | goods.enable_sku | Toggle SKU enabled flag |
| GET  | /api/v1/goods/sku/list | goods.list_skus | Paginated SKU list with media |
| POST | /api/v1/goods/sku/media | goods.manage_media | Add/delete SKU media |
| GET  | /api/v1/logs/recognition/get | logs.get_log | Query recognition log by taskId |
| GET  | /api/v1/system/train-status/get | system.train_status | Query train job status |
| GET  | /results/{path} | app.static | Static file serving for result images |
| GET  | /health | app.health | Health check |

> **Note:** All `/api/v1/*` endpoints have `Depends(verify_api_key)` and `Depends(check_rate_limit)` (disabled by default via config — empty `API_KEY` and `RATE_LIMIT=0`).

## ENTRY POINTS

| Command | Script | Purpose |
|---------|--------|---------|
| `python main.py` | main.py | SKU matching mode (default) |
| `python main.py --detection-only` | main.py | Detection only mode |
| `python main.py --det-model X --emb-model Y` | main.py | Custom models |
| `python -m api.app` | api/app.py | Start API server |
| `python scripts/build_index.py -r data/references/ -o chroma_data/ -m dinov2_vitb14` | build_index.py | Build Chroma index |
| `python scripts/crop_reference.py -r data/references_raw/ -o data/references/` | crop_reference.py | Crop raw reference photos |
| `python tests/test_detection.py` | test_detection.py | Smoke tests |

## CONVENTIONS

- **Type hints**: Full on function signatures (Python 3.10+ union: `str | None`)
- **Imports**: Absolute imports (`from src.core import detect`)
- **Linting**: Ruff configured (line-length: 100, py310 target)
- **Device**: Auto-detect via `detect_device()` (cuda → mps → cpu)
- **Detection classes**: `BEVERAGE_CONTAINER_CLASSES` (7 items: Bottle, Canned, etc.)
- **Index**: Chroma vector store with cosine similarity, top-2-per-SKU scoring. Unified path: `chroma_data/`
- **API prefix**: `/api/v1/`
- **API response**: All endpoints return `ApiResponse(code=1/0, data=..., msg="success")`
- **Async pattern**: CPU-bound work runs via `asyncio.to_thread()` to avoid blocking event loop
- **Crop naming**: `{image_name}_{index}_{sku_id}.jpg`
- **FP16**: Both YOLOE and DINOv2 use `model.half()` at load time when device=cuda
- **GPU concurrency**: API uses dedicated ThreadPoolExecutor(max_workers=1) for GPU inference
- **Warm-up**: YOLOE warm-up prediction runs at API startup
- **CJK rendering**: Annotations use PIL + Noto Sans CJK font for Chinese character support
- **Qiniu CDN**: Annotated images uploaded to Qiniu, `matched_image` returns full CDN URL. Falls back to local path if upload fails.
- **DB pool**: SQLAlchemy pool_size=5, max_overflow=10, pool_recycle=3600, pool_pre_ping=True
- **Temp cleanup**: Downloaded images cleaned up after processing in both recognition routes and embedding tasks
- **Index operations**: All Chroma writes go through `ReferenceProcessor` (sync), called via `asyncio.to_thread()` in routes. IndexManager removed.

## ANTI-PATTERNS (THIS PROJECT)

- **sys.path hack**: tests/test_detection.py, scripts/, and scripts/export_onnx.py manipulate sys.path
- **No docstrings**: Minimal documentation
- **No [project.scripts]**: pyproject.toml lacks console_scripts entry
- **SSL override**: `ssl._create_default_https_context` disabled in embedder.py (macOS compat, fallback path only)
- **GFW vendor hacks**: clip vendored locally, mobileclip2_b.ts tracked in models/, aliyun PyPI mirror configured

## COMMANDS

```bash
# Setup (uv venv — no activate script, use .venv/bin/python directly)
.venv/bin/python main.py

# CLI: SKU matching mode (default)
.venv/bin/python main.py

# CLI: SKU matching with custom thresholds
.venv/bin/python main.py --conf 0.3 --match-conf 0.5 --match-concentration 0.0 --match-verbose

# CLI: Custom models
.venv/bin/python main.py --det-model models/yoloe-26l-seg.pt --emb-model dinov2_vitb14

# CLI: FP16 + ONNX mode (GPU only)
.venv/bin/python main.py --swap --onnx

# CLI: Detection only mode
.venv/bin/python main.py --detection-only

# API: Start server
.venv/bin/python -m api.app

# API: With debug/reload
DEBUG=true .venv/bin/python -m api.app

# API: With Qiniu config
QINIU_ACCESS_KEY=xxx QINIU_SECRET_KEY=xxx QINIU_BUCKET=xxx .venv/bin/python -m api.app

# Build Chroma index from reference images
.venv/bin/python scripts/build_index.py -r data/references/ -o chroma_data/ -m dinov2_vitb14

# Build index with different embedding model
.venv/bin/python scripts/build_index.py -r data/references/ -o chroma_data/ -m dinov2_vits14

# Crop raw reference photos
.venv/bin/python scripts/crop_reference.py -r data/references_raw/ -o data/references/

# Export DINOv2 to ONNX
.venv/bin/python scripts/export_onnx.py --model dinov2_vits14

# Run smoke test
.venv/bin/python tests/test_detection.py

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
- **ReferenceProcessor**: Unified crop→embed→index pipeline. Crops via YOLOE detection + mask isolation, falls back to full-image embedding. Used directly by API routes (IndexManager adapter removed).
- **Detection confidence**: `--conf` flag (CLI) and `DET_CONF` setting (API) control YOLOE detection threshold. Passed to `predict(conf=)`. retina_masks=False to prevent OOM on high-res images.
- **GFW compatibility**: `clip` vendored locally at `vendor/clip_package/`. `mobileclip2_b.ts` in `models/` avoids GitHub download. PyPI via aliyun mirror.
- **Index operations**: All Chroma writes go through `ReferenceProcessor` (sync), called via `asyncio.to_thread()` in routes. IndexManager removed.
- **FP16 inference**: Both YOLOE and DINOv2 use model.half() at load time when device=cuda. ~1.8x speedup over FP32 on L20 GPU.
- **ONNX backend**: Optional via --onnx flag / USE_ONNX setting. Export with scripts/export_onnx.py. Currently not viable on 2GB GPU (ONNX Runtime lacks flash attention).
- **DINOv2 vendoring**: Full dinov2 source at vendor/dinov2/ + weights at models/dinov2_*_pretrain.pth. No internet needed at runtime.
- **Performance**: See PERF_PLAN.md for optimization details. FP16 ~1.8x faster. GPU thread pool prevents OOM under concurrent load.
- **Qiniu integration**: Annotated result images uploaded to Qiniu cloud storage. Token cached until 60s before deadline. Key format: `sku-match/{YYYY-MM/DD}/{filename}`. Region: South China (z2).
- **API auth**: Optional API key authentication via `X-API-Key` header. Rate limiting per-IP (requests/minute). Both disabled by default (empty API_KEY, RATE_LIMIT=0).
- **Input validation**: Pydantic Field constraints (max_length, min_length), Literal types for mode/action, non-empty string validators on all request models.
