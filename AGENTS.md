# PROJECT KNOWLEDGE BASE

**Generated:** 2026-05-30
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
│   ├── __init__.py           # Empty (no re-exports)
│   ├── core.py               # CLI detection: detect(), parse_detections(), _save_result()
│   ├── features.py           # Features dataclass, GeM pooling, fused CLS+GeM embedding
│   ├── embedder.py           # Embedder: HuggingFace transformers + Optimum ONNX, fused features, auto device detect, FP16 on CUDA
│   ├── indexer.py            # Chroma-backed SKUIndexer + score_matches(): build, add/delete/enable/search
│   ├── matcher.py            # SKUMatcher: detection + SKU matching pipeline
│   ├── types.py              # Dataclasses: Detection, SKUReference, SKUMatch
│   ├── utils.py              # Shared: detect_device(), free_gpu_memory(), embedding_to_list(), configure_ultralytics_weights()
│   ├── masking.py            # Mask extraction + configurable background masking (ImageNet mean default)
│   ├── patch_store.py        # PatchStore: per-image patch token .npy persistence for re-ranking
│   ├── reranker.py           # Patch-to-patch re-ranking: max-of-mean similarity + blended scoring
│   ├── reference_processor.py # ReferenceProcessor: crop→embed→index pipeline, on-demand CROP_MODEL, patch saving
│   ├── image_utils.py        # save_crop, process_detection_crops, draw_annotations, CJK font rendering via PIL
│   └── classes/
│       └── beverage_cls.py       # BEVERAGE_CONTAINER_CLASSES (7 items)
├── api/
│   ├── __init__.py
│   ├── app.py                # FastAPI + lifespan: loads detector, embedder, Chroma, services
│   ├── auth.py               # API key auth + per-IP rate limiting (disabled by default)
│   ├── config.py             # pydantic-settings: DET_MODEL, CROP_MODEL, EMB_MODEL, DEVICE, MATCH_CONF, TEMPERATURE, USE_TOP2_SUM, RERANK_BLEND_BETA, etc.
│   ├── database.py           # SQLAlchemy async engine + get_db + init_db
│   ├── models.py             # ORM: SKU, SKUMedia, RecognitionLog, TrainJob
│   ├── schemas.py            # Pydantic request/response models
│   ├── tasks.py              # asyncio task runner for embedding jobs
│   ├── dependencies.py       # FastAPI DI providers replacing app.state access
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
│   ├── build_index.py        # [DEPRECATED] Legacy direct-Chroma builder (no crop/patch/DB); use init_and_download.sh
│   ├── crop_reference.py     # Crop raw refs with YOLOE
│   ├── init_and_download.sh  # Reinit DB + Chroma + patches, add SKUs from SKUDB/, download crops
│   ├── sync_local.sh         # Sync from remote to local
│   ├── sync_remote.sh        # Sync to remote server
│   └── test_detection.py     # Detection test script
├── docs/
│   ├── API_GUIDE.md          # API usage guide
│   ├── AUDIT.md              # Codebase audit findings
│   ├── CLEANUP_PLAN.md       # Cleanup progress tracker
│   ├── CODE_QUALITY.md       # Code quality review + fix tracking
│   ├── CONSIDERATIONS.md     # Evaluated improvements (SAHI, FAISS, Qdrant, etc.)
│   ├── IMAGE_SIMILARITY_SEARCH_REPORT.md
│   ├── PATCH_TOKENS.md       # Fused retrieval + patch re-ranking pipeline documentation
│   ├── PERF_PLAN.md          # Performance improvement plan
│   ├── PLAN.md               # API server implementation plan
│   ├── QINIU.md              # Qiniu CDN integration docs
│   └── SETUP.md              # Setup instructions
├── test/                     # Test images
├── data/
│   ├── images/               # Input images (CLI mode)
│   ├── references/           # SKU reference images (sku_id/*.jpg)
│   ├── references_raw/       # Original photos to be cropped
│   └── patches/              # Patch token .npy files for re-ranking (gitignored)
├── models/                   # YOLOE weights, mobileclip2_b.ts
├── vendor/
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
| API DI | api/dependencies.py | Typed FastAPI dependency providers |
| Detection logic | src/core.py | detect() function (CLI), parse_detections() shared |
| Image embedder | src/embedder.py | HuggingFace transformers DINOv2, fused CLS+GeM features, Optimum ONNX backend |
| Feature fusion | src/features.py | Features dataclass, GeM pooling, fused CLS+GeM embedding |
| SKU indexer | src/indexer.py | Chroma-backed: search_batch, score_matches with top-2-per-SKU scoring |
| Patch storage | src/patch_store.py | Per-image patch token .npy files for re-ranking |
| Patch re-ranking | src/reranker.py | Max-of-mean patch similarity + blended scoring (β×patch + (1−β)×coarse) |
| Detection + matching | src/matcher.py | SKUMatcher class with crop saving |
| Masking | src/masking.py | extract_binary_masks, mask_background with configurable background color |
| Reference processing | src/reference_processor.py | crop→embed→index pipeline, on-demand CROP_MODEL, build_from_directory() |
| Recognition service | api/services/recognition.py | Full pipeline: detect → embed → match → rerank → annotate |
| Image storage | api/services/image_storage.py | Download, upload to Qiniu, cleanup, old file deletion |
| Shared utilities | src/utils.py | detect_device(), free_gpu_memory(), embedding_to_list(), configure_ultralytics_weights() |
| Image utilities | src/image_utils.py | save_crop, process_detection_crops, draw_annotations, CJK font rendering via PIL |
| Populate SKU index | scripts/init_and_download.sh | Server: reinit DB + Chroma, bulk-add SKUs (CLI auto-builds via main.py) |
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
| main() | function | main.py:160 | CLI entry (argparse) — delegates to run_matching() or run_detection() |
| run_detection() | function | main.py:27 | Detection-only mode (--detection-only) |
| run_matching() | function | main.py:41 | SKU matching mode (default) |
| get_next_match_dir() | function | main.py:17 | Auto-increment match output directory |
| parse_detections() | function | src/core.py:13 | Shared YOLOE box→Detection parsing |
| detect() | function | src/core.py:45 | Core YOLOE inference (CLI) — batch prediction with crop saving |
| _save_result() | function | src/core.py:99 | Save detection JSON + masked crops |
| Embedder | class | src/embedder.py:40 | DINOv2 embedder via HuggingFace transformers + Optimum ONNX, fused features |
| EmbedderProtocol | protocol | src/embedder.py:28 | Interface for SKU image embedders |
| DEFAULT_EMB_MODEL | constant | src/embedder.py:16 | Default local model path ("models/dinov2-base") |
| Embedder.embed() | method | src/embedder.py:112 | Embed single PIL Image → np.ndarray |
| Embedder.embed_batch() | method | src/embedder.py:116 | Embed batch of PIL Images → np.ndarray |
| Embedder.embed_path() | method | src/embedder.py:127 | Embed image from file path |
| Embedder.extract_features_batch() | method | src/embedder.py:132 | Extract CLS + patch tokens (single forward pass) |
| Embedder.features_to_embedding() | method | src/embedder.py:182 | Convert Features → fused or CLS embedding |
| Embedder.to() | method | src/embedder.py:107 | Move model to specified device |
| Features | class | src/features.py:14 | Raw DINOv2 output: cls + patches, both L2-normalized |
| gem_pool() | function | src/features.py:32 | Generalized Mean pooling over patch tokens |
| fused_embedding() | function | src/features.py:55 | Weighted fusion of CLS + GeM(patches) → L2-normalized |
| PatchStore | class | src/patch_store.py:18 | Manages patch token .npy files on disk |
| PatchStore.save() | method | src/patch_store.py:38 | Save patch tokens for a doc_id |
| PatchStore.load_batch() | method | src/patch_store.py:69 | Load patch tokens for multiple doc_ids (silent skip missing) |
| PatchStore.delete_sku() | method | src/patch_store.py:95 | Delete all patch files for a SKU |
| rerank() | function | src/reranker.py:71 | Re-rank coarse candidates using patch-to-patch matching |
| max_of_mean_similarity() | function | src/reranker.py:47 | Bidirectional max-of-mean patch similarity |
| RerankCandidate | dataclass | src/reranker.py:23 | Single vector-level candidate after re-ranking |
| SKURerankResult | dataclass | src/reranker.py:35 | Aggregated re-ranking result for one SKU |
| VectorMatch | class | src/indexer.py:22 | Single vector-level match result (sku_id, similarity, rank, media_url) |
| SKUIndexer | class | src/indexer.py:112 | Chroma-backed SKU index with top-2 scoring |
| EPSILON | constant | src/indexer.py:17 | Small score for unranked SKUs in softmax (1e-8) |
| COLLECTION_NAME | constant | src/indexer.py:36 | Chroma collection name ("sku_embeddings") |
| COSINE_DISTANCE_TO_SIMILARITY | constant | src/indexer.py:39 | Conversion factor (2.0) |
| _softmax() | function | src/indexer.py:42 | Softmax probability distribution over SKU scores |
| concentration_score() | function | src/indexer.py:52 | Top-1 share of top-K probability mass |
| score_matches() | function | src/indexer.py:63 | Shared scoring: detections + search_results → list[SKUMatch] |
| SKUIndexer.init_collection() | method | src/indexer.py:142 | Initialize Chroma collection from persist dir |
| SKUIndexer.build() | method | src/indexer.py:161 | Populate collection from SKUReference list |
| SKUIndexer.add_reference() | method | src/indexer.py:189 | Upsert single reference embedding |
| SKUIndexer.delete_sku() | method | src/indexer.py:214 | Delete all vectors for a SKU |
| SKUIndexer.delete_media() | method | src/indexer.py:219 | Delete single media vector |
| SKUIndexer.set_enabled() | method | src/indexer.py:225 | Toggle SKU enabled flag in Chroma metadata |
| SKUIndexer.update_sku_name() | method | src/indexer.py:236 | Update SKU name in Chroma metadata |
| SKUIndexer.search() | method | src/indexer.py:248 | Single-query similarity search |
| SKUIndexer.search_batch() | method | src/indexer.py:252 | Batch similarity search with top-2-per-SKU scoring |
| SKUIndexer.get_sku_name() | method | src/indexer.py:335 | Get SKU name from cache |
| SKUMatcher | class | src/matcher.py:21 | Detection + SKU matching with crop saving |
| SKUMatcher.match_images() | method | src/matcher.py:43 | Match SKUs across images (detect → embed → search) |
| SKUMatcher.from_index_dir() | classmethod | src/matcher.py:154 | Factory: load models + index from directory |
| Detection | dataclass | src/types.py:8 | Detection result: bbox, confidence, class_name, class_id |
| SKUReference | dataclass | src/types.py:17 | Reference entry: sku_id, sku_name, image_path, embedding |
| SKUMatch | dataclass | src/types.py:25 | Match result: detection, sku_id, match_score, concentration, distribution, top_vectors |
| detect_device() | function | src/utils.py:21 | Auto-detect: cuda → mps → cpu |
| free_gpu_memory() | function | src/utils.py:30 | Release cached GPU memory (gc + cuda.empty_cache) |
| embedding_to_list() | function | src/utils.py:37 | Convert numpy/torch embedding to plain list for Chroma |
| configure_ultralytics_weights() | function | src/utils.py:14 | Sets ultralytics weights_dir to local models/ |
| IMAGENET_MEAN_RGB | constant | src/masking.py:15 | Default background color for masking (123.5, 116.5, 103.5) |
| normalize_mask() | function | src/masking.py:18 | Normalize mask to uint8 (0 or 255) |
| extract_binary_masks() | function | src/masking.py:32 | Extract binary masks from YOLOE result |
| mask_background() | function | src/masking.py:62 | Apply configurable background color behind mask |
| ProcessResult | dataclass | src/reference_processor.py:21 | Result from process_and_add: success + crop_path |
| select_best_detection() | function | src/reference_processor.py:27 | Select smallest center-covering detection box |
| ReferenceProcessor | class | src/reference_processor.py:50 | Processes raw reference images: detect → crop → embed → index |
| ReferenceProcessor._crop_detect() | method | src/reference_processor.py:79 | On-demand crop model or reuse startup detector |
| ReferenceProcessor.crop_reference() | method | src/reference_processor.py:110 | Detect primary object and return cropped PIL Image |
| ReferenceProcessor.process_and_add() | method | src/reference_processor.py:151 | Full pipeline: crop → embed → index → save crop to temp |
| ReferenceProcessor.build_from_directory() | method | src/reference_processor.py:191 | Build index from directory of reference images |
| save_crop() | function | src/image_utils.py:12 | Save a cropped PIL Image |
| process_detection_crops() | function | src/image_utils.py:27 | Process all detections and save masked crops |
| draw_annotations() | function | src/image_utils.py:93 | Draw bounding boxes and SKU labels on image |
| BEVERAGE_CONTAINER_CLASSES | list | src/classes/beverage_cls.py:10 | 7 beverage container class names for YOLOE |
| Settings | class | api/config.py:4 | pydantic-settings configuration |
| settings | instance | api/config.py:55 | Global settings singleton |
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
| process_single_media() | function | api/tasks.py:15 | Shared download→embed→upload→cleanup for single media item |
| start_embed_task() | function | api/tasks.py:55 | Start async embedding task for new SKU |
| get_task_status() | function | api/tasks.py:136 | Check running task status |
| RecognitionService | class | api/services/recognition.py:24 | Full recognition pipeline (sync) |
| RecognitionService.recognize() | method | api/services/recognition.py:118 | Single-image recognition: detect → embed → match → annotate |
| ImageStorage | class | api/services/image_storage.py:13 | Download images, manage result paths, Qiniu upload |
| ImageStorage.download_image() | method | api/services/image_storage.py:54 | Download image (local path or URL) |
| ImageStorage.cleanup_download() | method | api/services/image_storage.py:89 | Remove downloaded temp file |
| ImageStorage.cleanup_old_results() | method | api/services/image_storage.py:36 | Delete annotated images older than max_age_hours |
| ImageStorage.get_result_path() | method | api/services/image_storage.py:98 | Get annotated image path for task_id |
| ImageStorage.get_result_url() | method | api/services/image_storage.py:103 | Get annotated image relative URL |
| ImageStorage.upload_to_qiniu() | method | api/services/image_storage.py:130 | Upload file to Qiniu and return CDN URL |
| UnicodeJSONResponse | class | api/app.py:170 | JSONResponse with ensure_ascii=False |
| main() | function | api/app.py:203 | Entry point for sku-match-api console script |
| get_indexer() | function | api/dependencies.py:16 | DI provider → SKUIndexer |
| get_processor() | function | api/dependencies.py:20 | DI provider → ReferenceProcessor |
| get_recognition_service() | function | api/dependencies.py:24 | DI provider → RecognitionService |
| get_image_storage() | function | api/dependencies.py:28 | DI provider → ImageStorage |
| get_inference_executor() | function | api/dependencies.py:32 | DI provider → ThreadPoolExecutor |
| get_device() | function | api/dependencies.py:36 | DI provider → device string || _get_sku_or_error() | function | api/routes/goods.py:35 | Helper: fetch SKU or return ApiResponse error |
| _to_full_url() | function | api/routes/goods.py:161 | Convert relative path to full URL |

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

> **Note:** All `/api/v1/*` endpoints have `Depends(verify_api_key)` and `Depends(check_rate_limit)` (disabled by default via config — empty `API_KEY` and `RATE_LIMIT=0`). All routes use `response_model=ApiResponse`.

## ENTRY POINTS

| Command | Script | Purpose |
|---------|--------|---------|
| `uv run sku-match` | main.py | SKU matching mode (default) |
| `uv run sku-match --detection-only` | main.py | Detection only mode |
| `uv run sku-match --det-model X --emb-model Y` | main.py | Custom models |
| `uv run sku-match-api` | api/app.py | Start API server |
| `bash scripts/init_and_download.sh` | init_and_download.sh | Server: reinit DB + Chroma, bulk-add SKUs from SKUDB/ |
| `uv run python scripts/crop_reference.py -r data/references_raw/ -o data/references/` | crop_reference.py | Crop raw reference photos |
| `uv run python tests/test_detection.py` | test_detection.py | Smoke tests |

## CONVENTIONS

- **Type hints**: Full on function signatures (Python 3.10+ union: `str | None`)
- **Imports**: Absolute imports (`from src.core import detect`)
- **Linting**: Ruff configured (line-length: 100, py310 target)
- **Device**: Auto-detect via `detect_device()` (cuda → mps → cpu)
- **Detection classes**: `BEVERAGE_CONTAINER_CLASSES` (7 items: bottle, canned, carton, empty paper box, full paper box, strawed drink, keg)
- **Index**: Chroma vector store with cosine similarity, top-2-per-SKU scoring. Unified path: `chroma_data/`
- **Feature fusion**: `USE_FUSED_FEATURES` (default true) controls CLS+GeM vs CLS-only embeddings. `feature_type` recorded in ChromaDB metadata, hard error on mismatch.
- **Patch re-ranking**: Two independent toggles: `USE_FUSED_FEATURES` (stage 1) and `USE_RERANKING` (stage 2, default true). Blend: `β × patch_score + (1−β) × norm_coarse` (β=0.7 default, `RERANK_BLEND_BETA`). β=1 = pure patch, β=0 = pure coarse.
- **CROP_MODEL**: Separate YOLOE model for cropping reference images. Empty string reuses DET_MODEL (backward compatible). Loaded on-demand, unloaded immediately after crop. Runs in FP32.
- **API prefix**: `/api/v1/`
- **API response**: All endpoints return `ApiResponse(code=1/0, data=..., msg="success")` with `response_model=ApiResponse`
- **DI pattern**: Routes use `Depends()` from `api/dependencies.py` — no direct `app.state` access in routes
- **Async pattern**: CPU/GPU-bound work runs via `asyncio.to_thread()` or `loop.run_in_executor()` to avoid blocking event loop
- **GPU concurrency**: API uses dedicated `ThreadPoolExecutor(max_workers=1)` for GPU inference
- **Warm-up**: YOLOE warm-up prediction runs at API startup (dummy 640x640 image)
- **FP16**: Startup detector uses `model.half()` before first `predict()` (no `half=True` kwarg). On-demand CROP_MODEL runs in FP32 (avoids fuse_conv_and_bn crash on models with many BatchNorm layers). Embedder loads with `torch_dtype=torch.float16` when device=cuda
- **CJK rendering**: Annotations use PIL + Noto Sans CJK font for Chinese character support
- **Qiniu CDN**: Annotated images uploaded to Qiniu, `matched_image` returns full CDN URL. Falls back to local path with `qiniu_upload_failed=True` flag on failure.
- **DB pool**: SQLAlchemy pool_size=5, max_overflow=10, pool_recycle=3600, pool_pre_ping=True
- **Temp cleanup**: Downloaded images cleaned up after processing in both recognition routes and embedding tasks
- **match_conf threshold**: Configurable via `MATCH_CONF` env var (default 0 = disabled). Filters low-confidence matches — below threshold sets sku_id="", sku_name="", match_score=0.0
- **train_status**: `SUCCESS` if all images embed OK. `FAILED: a/b` if a out of b images failed. `FAILED` on task-level exception.
- **sku_distribution**: Returns only top 5 entries by probability (sorted descending)
- **Annotated image cleanup**: Background task runs hourly, deletes files older than `RESULTS_MAX_AGE_HOURS` (default 24)
- **Rate limiter**: Uses `asyncio.Lock` for atomicity, prunes empty IP entries when >1000 IPs tracked
- **_running_tasks**: Auto-cleaned via `add_done_callback` — tasks remove themselves when done
- **Fix endpoint**: Rejects unknown taskIds with `ApiResponse(code=0, msg="taskId '...' not found")`
- **Duplicate detection**: Detect endpoint rejects duplicate taskIds; new_sku rejects duplicate skuIds and trainJobIds
- **Crop naming**: `{image_name}_{index}_{sku_id}.jpg`
- **Index operations**: Chroma writes go through `ReferenceProcessor` (sync) for crop+embed pipeline; direct metadata CRUD on `SKUIndexer` (enable, delete, rename). Called via `asyncio.to_thread()` in routes. Temperature and top-2/top-1 scoring configurable via `SKUIndexer` constructor.
- **Unicode JSON**: API uses `UnicodeJSONResponse` (ensure_ascii=False) as default response class
- **uv required**: All commands use `uv run` — never `python` or `pip` directly. No manual venv activation. `uv sync` for install, `uv run` for execution.

## ANTI-PATTERNS (THIS PROJECT)

- **sys.path hack**: tests/test_detection.py and scripts/ manipulate sys.path
- **No docstrings**: Minimal documentation on most functions
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
uv run sku-match --det-model models/yoloe-26l-seg.pt --emb-model facebook/dinov2-base

# CLI: FP16 + ONNX mode (GPU only)
uv run sku-match --swap --onnx

# CLI: Detection only mode
uv run sku-match --detection-only

# API: Start server
uv run sku-match-api

# API: With debug/reload
DEBUG=true uv run sku-match-api

# API: With custom match confidence
MATCH_CONF=0.5 uv run sku-match-api

# API: With Qiniu config
QINIU_ACCESS_KEY=xxx QINIU_SECRET_KEY=xxx QINIU_BUCKET=xxx uv run sku-match-api

# CLI: index auto-builds from data/references/ on first run (embed model via --emb-model)
uv run sku-match

# API/server: reinitialize DB + Chroma and bulk-add SKUs from SKUDB/
bash scripts/init_and_download.sh

# Crop raw reference photos
uv run python scripts/crop_reference.py -r data/references_raw/ -o data/references/

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
- **Embedding models**: facebook/dinov2-small (384-dim), facebook/dinov2-base (768-dim), facebook/dinov2-large (1024-dim), facebook/dinov2-giant (1536-dim). Also available with registers: *-with-registers variants. Default API model: local `models/dinov2-with-registers-base`. Default CLI model: local `models/dinov2-base`.
- **Chroma index**: PersistentClient with HNSW cosine similarity. Incremental add/delete — no full rebuild needed. Unified path: `chroma_data/` across CLI, API, and build scripts.
- **Match scoring**: Top-2-per-SKU cosine similarity sum (or top-1 if `USE_TOP2_SUM=False`) → softmax probability distribution (0–1 range). `TEMPERATURE` controls softmax sharpness (default 0.2, lower = more decisive). `MATCH_CONF` default 0 (disabled). Concentration score (top-1 share of top-K probability mass). Detection rank positions (top2_ranks) tracked per match. When reranking enabled: `β × patch_score + (1−β) × norm_coarse` → softmax over blended scores.
- **Device auto-detect**: `detect_device()` returns cuda → mps → cpu
- **First run**: Downloads YOLOE model (~400MB). DINOv2 weights from local `models/` directory (HuggingFace format).
- **API state**: Models loaded once at FastAPI lifespan startup, reused across requests. Services accessed via typed DI providers in `api/dependencies.py`.
- **ReferenceProcessor**: Unified crop→embed→index pipeline. Crops via YOLOE detection + mask isolation, falls back to skipping if no detection. Used by API routes for new SKU and media add. On-demand CROP_MODEL loaded/unloaded per crop. Saves patch tokens for re-ranking when PatchStore configured.
- **Masking module**: `src/masking.py` — configurable background color (default ImageNet mean RGB). Used by CLI detection and reference processor.
- **Detection confidence**: `--conf` flag (CLI) and `DET_CONF` setting (API) control YOLOE detection threshold. Passed to `predict(conf=)`. retina_masks=False to prevent OOM on high-res images.
- **GFW compatibility**: `clip` vendored locally at `vendor/clip_package/`. `mobileclip2_b.ts` in `models/` avoids GitHub download. PyPI via aliyun mirror.
- **FP16 inference**: Startup detector uses `model.half()` before first `predict()` (no `half=True` kwarg). ~1.8x speedup over FP32 on L20 GPU. On-demand CROP_MODEL runs in FP32 (avoids fuse_conv_and_bn crash on models with many BatchNorm layers). Embedder loads with `torch_dtype=torch.float16` when device=cuda.
- **ONNX backend**: Optional via --onnx flag / USE_ONNX setting. Uses Optimum ORTModelForFeatureExtraction with on-the-fly ONNX export. CUDA provider on GPU, CPU provider otherwise.
- **Embedder**: HuggingFace transformers AutoModel + AutoImageProcessor. 8 variants (small/base/large/giant ± registers). Weights from local `models/` directory (HuggingFace format). Single forward pass extracts both CLS and patch tokens.
- **Performance**: See docs/PERF_PLAN.md for optimization details. FP16 ~1.8x faster. GPU thread pool prevents OOM under concurrent load.
- **Qiniu integration**: Annotated result images uploaded to Qiniu cloud storage. Token cached until 60s before deadline. Key format: `sku-match/{YYYY-MM/DD}/{filename}`. Region: South China (z2). CDN URL rewrite for faster domestic downloads (iovip origin pull).
- **API auth**: Optional API key authentication via `X-API-Key` header. Rate limiting per-IP (requests/minute). Both disabled by default (empty API_KEY, RATE_LIMIT=0).
- **Input validation**: Pydantic Field constraints (max_length, min_length), Literal types for mode/action, non-empty string validators on all request models.
- **Console scripts**: `sku-match` (CLI) and `sku-match-api` (API server) defined in pyproject.toml [project.scripts].
- **Shared pipeline**: `parse_detections()` (src/core.py) and `score_matches()` (src/indexer.py) used by both CLI matcher and API recognition service.
- **process_single_media()**: Shared download→embed→upload→cleanup helper in api/tasks.py, used by both start_embed_task and manage_media route.
- **Masking module**: `src/masking.py` — configurable background color (default ImageNet mean RGB). Used by CLI detection and reference processor.
- **Init script**: `scripts/init_and_download.sh` wipes DB + Chroma + `data/patches/` to prevent stale doc_id mismatches after re-indexing. Rebuilds via API `/goods/sku/new` endpoints.
