import concurrent.futures
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import chromadb
from fastapi import Depends, FastAPI
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLOE

from api.auth import check_rate_limit, verify_api_key
from api.config import settings
from api.database import init_db
from api.routes import goods, logs, recognition, system
from api.services.image_storage import ImageStorage
from api.services.recognition import RecognitionService
from src.classes.beverage_cls import BEVERAGE_CONTAINER_CLASSES
from src.embedder import DINOv2Embedder
from src.indexer import SKUIndexer
from src.reference_processor import ReferenceProcessor
from src.utils import configure_ultralytics_weights
from src.utils import detect_device as _detect_device

# Ensure ultralytics finds local model weights (mobileclip2_b.ts) without GitHub download
configure_ultralytics_weights()

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

    # Load ML models
    logger.info("Loading YOLOE detector: %s", settings.DET_MODEL)
    detector = YOLOE(settings.DET_MODEL)
    detector.set_classes(BEVERAGE_CONTAINER_CLASSES)
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

    logger.info("Loading DINOv2 embedder: %s (onnx=%s)", settings.EMB_MODEL, settings.USE_ONNX)
    embedder = DINOv2Embedder(
        model_name=settings.EMB_MODEL,
        device=device,
        use_onnx=settings.USE_ONNX,
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

    indexer = SKUIndexer(collection=collection)
    image_storage = ImageStorage(results_dir=settings.RESULTS_DIR, download_timeout=settings.DOWNLOAD_TIMEOUT)
    processor = ReferenceProcessor(
        detector=detector,
        embedder=embedder,
        indexer=indexer,
        device=device,
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

    logger.info("Startup complete")

    yield

    logger.info("Shutting down")
    try:
        inference_executor.shutdown(wait=False)
    except Exception:
        pass
    try:
        chroma_client.close()
    except Exception:
        pass


app = FastAPI(
    title="SKU Match API",
    version="0.2.0",
    lifespan=lifespan,
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

# Serve annotated result images
results_path = Path(settings.RESULTS_DIR)
results_path.mkdir(parents=True, exist_ok=True)
app.mount("/results", StaticFiles(directory=str(results_path)), name="results")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
