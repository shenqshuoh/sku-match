import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import get_db
from api.dependencies import (
    get_device,
    get_image_storage,
    get_inference_executor,
    get_recognition_service,
)
from api.models import RecognitionLog
from api.schemas import ApiResponse, DetectRequest, FixRequest
from api.services.image_storage import ImageStorage
from api.services.recognition import RecognitionService

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/detect", response_model=ApiResponse)
async def detect(
    request: DetectRequest,
    recognition_service: RecognitionService = Depends(get_recognition_service),
    image_storage: ImageStorage = Depends(get_image_storage),
    inference_executor: ThreadPoolExecutor = Depends(get_inference_executor),
    device: str = Depends(get_device),
    db: AsyncSession = Depends(get_db),
):
    downloaded_path = None
    log = None
    try:
        # 1. Check for duplicate taskId
        existing = await db.execute(
            select(RecognitionLog).where(RecognitionLog.task_id == request.taskId)
        )
        if existing.scalar_one_or_none() is not None:
            return ApiResponse(code=0, msg=f"taskId '{request.taskId}' already exists")

        # 2. Log request to recognition_log table
        log = RecognitionLog(
            task_id=request.taskId,
            request_json=request.model_dump_json(),
        )
        db.add(log)
        await db.commit()

        # 3. Download image via image storage
        downloaded_path = await image_storage.download_image(request.files)

        # 4. Run recognition (CPU/GPU-bound, offload to dedicated inference thread)
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                inference_executor,
                recognition_service.recognize,
                downloaded_path,
                request.taskId,
                request.roiRect,
                device,
            )
        except RuntimeError as e:
            if "Index is empty" in str(e):
                return ApiResponse(code=0, msg="No SKUs indexed yet — add reference images first")
            raise

        # 4b. Cleanup downloaded temp file
        image_storage.cleanup_download(downloaded_path)

        # 5. Upload annotated image to Qiniu
        matched_image = result.get("matched_image", "")
        qiniu_upload_failed = False
        if matched_image:
            annotated_path = image_storage.get_result_path(request.taskId)
            if annotated_path.exists():
                try:
                    cdn_url = await image_storage.upload_to_qiniu(annotated_path)
                    if cdn_url:
                        result["matched_image"] = cdn_url
                        matched_image = cdn_url
                except Exception as e:
                    logger.warning("Qiniu upload failed, keeping local path: %s", e)
                    qiniu_upload_failed = True
        if qiniu_upload_failed:
            result["qiniu_upload_failed"] = True

        # 6. Update log with result and visual image path
        log.ai_result_json = json.dumps(result)
        log.visual_image_path = matched_image
        await db.commit()

        # 7. Return response
        return ApiResponse(data=result)
    except Exception as e:
        logger.exception("Recognition detect failed: %s", e)
        if downloaded_path is not None:
            try:
                image_storage.cleanup_download(downloaded_path)
            except Exception:
                pass
        # Update log with error if log record was created
        if log is not None:
            try:
                log.ai_result_json = json.dumps({"error": str(e)})
                await db.commit()
            except Exception:
                logger.exception("Failed to update log on error for taskId=%s", request.taskId)
        return ApiResponse(code=0, msg=str(e))


@router.post("/fix", response_model=ApiResponse)
async def fix(request: FixRequest, db: AsyncSession = Depends(get_db)):
    try:
        # 1. Look up RecognitionLog by task_id
        result = await db.execute(select(RecognitionLog).where(RecognitionLog.task_id == request.taskId))
        log = result.scalar_one_or_none()
        if log is None:
            return ApiResponse(code=0, msg=f"taskId '{request.taskId}' not found")
        # 2. Store user corrections as JSON text
        log.user_correction_json = json.dumps([fi.model_dump() for fi in request.fixItems])
        await db.commit()
        return ApiResponse(data={})
    except Exception as e:
        logger.exception("Recognition fix failed: %s", e)
        return ApiResponse(code=0, msg=str(e))
