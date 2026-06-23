import asyncio
import concurrent.futures
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import chromadb
from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLOE

from api.auth import check_rate_limit, verify_api_key
from api.config import settings
from api.database import init_db
from api.routes import goods, logs, recognition, system
from api.services.image_storage import ImageStorage
from api.services.recognition import RecognitionService
from src.classes.beverage_cls import BEVERAGE_CONTAINER_CLASSES
from src.embedder import Embedder
from src.indexer import SKUIndexer
from src.patch_store import PatchStore
from src.reference_processor import ReferenceProcessor
from src.utils import configure_ultralytics_weights
from src.utils import detect_device as _detect_device

logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _get_device() -> str:
    return settings.DEVICE if settings.DEVICE else _detect_device()


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = _get_device()
    logger.info("Starting up — device=%s", device)

    await init_db()

    # Ensure ultralytics finds local model weights (mobileclip2_b.ts) without GitHub download
    configure_ultralytics_weights()

    # Load ML models
    logger.info("Loading YOLOE detector: %s", settings.DET_MODEL)
    detector = YOLOE(settings.DET_MODEL)
    detector.set_classes(BEVERAGE_CONTAINER_CLASSES)

    # Convert detector to FP16 for faster inference on CUDA
    if device == "cuda":
        import torch
        detector.model.half()
        logger.info("Detector converted to FP16")

    # Warm up detector: first CUDA call is slow due to lazy kernel initialization
    logger.info("Warming up detector...")
    import numpy as np
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    detector.predict(source=dummy, retina_masks=False, verbose=False)
    logger.info("Detector warm-up complete")

    feature_type = "fused" if settings.USE_FUSED_FEATURES else "cls"
    logger.info(
        "Loading embedder: %s (onnx=%s, fused=%s, alpha=%s)",
        settings.EMB_MODEL, settings.USE_ONNX, settings.USE_FUSED_FEATURES, settings.FUSE_ALPHA,
    )
    embedder = Embedder(
        model_name=settings.EMB_MODEL,
        device=device,
        use_onnx=settings.USE_ONNX,
        use_fused=settings.USE_FUSED_FEATURES,
        fuse_alpha=settings.FUSE_ALPHA,
        gem_p=settings.GEM_P,
    )

    # Init Chroma
    logger.info("Opening Chroma index: %s", settings.CHROMA_PERSIST_DIR)
    chroma_client = chromadb.PersistentClient(
        path=settings.CHROMA_PERSIST_DIR,
        settings=chromadb.Settings(anonymized_telemetry=False),
    )
    collection = chroma_client.get_or_create_collection(
        name="sku_embeddings",
        configuration={"hnsw": {"space": "cosine"}},
    )

    indexer = SKUIndexer(
        collection=collection,
        temperature=settings.TEMPERATURE,
        use_top2_sum=settings.USE_TOP2_SUM,
    )
    logger.info("Indexer config: temperature=%s, top2_sum=%s", settings.TEMPERATURE, settings.USE_TOP2_SUM)

    # Validate feature type compatibility
    indexer.validate_feature_type(feature_type)
    indexer.validate_emb_model(settings.EMB_MODEL)

    # Initialize patch store for re-ranking
    patch_store: PatchStore | None = None
    if settings.USE_RERANKING:
        patch_store = PatchStore(patch_dir=settings.PATCH_DIR)
        logger.info("Patch re-ranking enabled: patch_dir=%s, top_k=%d, blend_beta=%s",
                     settings.PATCH_DIR, settings.RERANK_TOP_K, settings.RERANK_BLEND_BETA)
    else:
        logger.info("Patch re-ranking disabled")

    image_storage = ImageStorage(
        results_dir=settings.RESULTS_DIR,
        download_timeout=settings.DOWNLOAD_TIMEOUT,
        qiniu_token_url=settings.QINIU_TOKEN_URL,
        qiniu_upload_url=settings.QINIU_UPLOAD_URL,
        qiniu_domain=settings.QINIU_DOMAIN,
        qiniu_iovip_url=settings.QINIU_IOVIP_URL,
        qiniu_key_prefix=settings.QINIU_KEY_PREFIX,
    )
    # Resolve crop model: empty string means reuse DET_MODEL (no separate loading)
    crop_model_path = settings.CROP_MODEL or None
    if crop_model_path:
        logger.info("Crop model: %s (on-demand)", crop_model_path)
    else:
        logger.info("Crop model: using DET_MODEL")

    processor = ReferenceProcessor(
        detector=detector,
        embedder=embedder,
        indexer=indexer,
        device=device,
        patch_store=patch_store,
        feature_type=feature_type,
        crop_model_path=crop_model_path,
    )
    recognition_service = RecognitionService(
        detector=detector,
        embedder=embedder,
        indexer=indexer,
        image_storage=image_storage,
        det_conf=settings.DET_CONF,
        imgsz=settings.IMGSZ,
        match_conf=settings.MATCH_CONF,
        concentration_topk=settings.CONCENTRATION_TOPK,
        patch_store=patch_store,
        use_reranking=settings.USE_RERANKING,
        rerank_top_k=settings.RERANK_TOP_K,
        rerank_blend_beta=settings.RERANK_BLEND_BETA,
    )

    # Dedicated GPU inference executor — prevents concurrent GPU access under load
    inference_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="gpu_inference"
    )
    app.state.inference_executor = inference_executor

    app.state.device = device
    app.state.detector = detector
    app.state.embedder = embedder
    app.state.chroma_client = chroma_client
    app.state.indexer = indexer
    app.state.image_storage = image_storage
    app.state.processor = processor
    app.state.recognition_service = recognition_service
    app.state.patch_store = patch_store
    app.state.feature_type = feature_type

    logger.info("Startup complete")

    # Background task: periodically clean up old annotated images
    cleanup_stop = asyncio.Event()
    app.state._cleanup_stop = cleanup_stop

    async def _results_cleanup_loop():
        while not cleanup_stop.is_set():
            try:
                await asyncio.wait_for(cleanup_stop.wait(), timeout=3600)
            except asyncio.TimeoutError:
                pass  # timeout means it's time to run cleanup
            if cleanup_stop.is_set():
                break
            try:
                await asyncio.to_thread(
                    image_storage.cleanup_old_results,
                    settings.RESULTS_MAX_AGE_HOURS,
                )
            except Exception:
                logger.exception("Failed to clean up old annotated images")

    cleanup_task = asyncio.create_task(_results_cleanup_loop())
    app.state._cleanup_task = cleanup_task

    yield

    # Stop cleanup task
    cleanup_stop.set()
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

    logger.info("Shutting down")
    try:
        inference_executor.shutdown(wait=False)
    except Exception:
        pass
    try:
        chroma_client.close()
    except Exception:
        pass


class UnicodeJSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        return json.dumps(content, ensure_ascii=False, allow_nan=False).encode("utf-8")


app = FastAPI(
    title="SKU Match API",
    version="0.2.0",
    lifespan=lifespan,
    default_response_class=UnicodeJSONResponse,
)

# Register routers
app.include_router(recognition.router, prefix="/api/v1/recognition", tags=["recognition"],
                   dependencies=[Depends(verify_api_key), Depends(check_rate_limit)])
app.include_router(goods.router, prefix="/api/v1/goods", tags=["goods"],
                   dependencies=[Depends(verify_api_key), Depends(check_rate_limit)])
app.include_router(logs.router, prefix="/api/v1/logs", tags=["logs"],
                   dependencies=[Depends(verify_api_key), Depends(check_rate_limit)])
app.include_router(system.router, prefix="/api/v1/system", tags=["system"],
                   dependencies=[Depends(verify_api_key), Depends(check_rate_limit)])


@app.get("/health")
async def health():
    return {"status": "ok"}

# Serve annotated result images (public)
results_path = Path(settings.RESULTS_DIR)
results_path.mkdir(parents=True, exist_ok=True)
app.mount("/results", StaticFiles(directory=str(results_path)), name="results")


def _setup_file_logging() -> None:
    """Add file handler to root logger. Called once from main() before uvicorn."""
    _log_file = Path(settings.LOG_FILE)
    _log_file.parent.mkdir(parents=True, exist_ok=True)
    _fh = logging.FileHandler(_log_file, mode="a", encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(_fh)


def main():
    """Entry point for `sku-match-api` console script."""
    _setup_file_logging()
    import uvicorn
    uvicorn.run(
        "api.app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )


if __name__ == "__main__":
    main()
