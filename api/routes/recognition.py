import asyncio
import json
import logging
from fastapi import APIRouter, Depends, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from api.database import get_db
from api.models import RecognitionLog
from api.schemas import DetectRequest, FixRequest, ApiResponse, StatusResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/detect")
async def detect(request: DetectRequest, req: Request, db: AsyncSession = Depends(get_db)):
    try:
        # 1. Log request to recognition_log table
        log = RecognitionLog(
            task_id=request.taskId,
            request_json=request.model_dump_json(),
        )
        db.add(log)
        await db.commit()

        # 2. Download image via image storage
        downloaded_path = await req.app.state.image_storage.download_image(request.files)

        # 3. Run recognition (CPU/GPU-bound, offload to thread)
        result = await asyncio.to_thread(
            req.app.state.recognition_service.recognize,
            image_path=downloaded_path,
            task_id=request.taskId,
            roi_rect=request.roiRect,
            device=req.app.state.device,
        )

        # 4. Update log with result and visual image path
        log.ai_result_json = json.dumps(result)
        log.visual_image_path = result.get("matched_image") if isinstance(result, dict) else None
        await db.commit()

        # 5. Return response
        return ApiResponse(data=result)
    except Exception as e:
        logger.exception("Recognition detect failed: %s", e)
        return ApiResponse(code=0, msg=str(e))


@router.post("/fix")
async def fix(request: FixRequest, db: AsyncSession = Depends(get_db)):
    try:
        # 1. Look up RecognitionLog by task_id
        result = await db.execute(select(RecognitionLog).where(RecognitionLog.task_id == request.taskId))
        log = result.scalar_one_or_none()
        if log is None:
            log = RecognitionLog(task_id=request.taskId)
            db.add(log)
        # 2. Store user corrections as JSON text
        log.user_correction_json = json.dumps([fi.model_dump() for fi in request.fixItems])
        await db.commit()
        return StatusResponse(status="success")
    except Exception as e:
        logger.exception("Recognition fix failed: %s", e)
        return ApiResponse(code=0, msg=str(e))
