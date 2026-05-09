import json
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from api.database import get_db
from api.models import RecognitionLog
from api.schemas import ApiResponse
from sqlalchemy import select

router = APIRouter()


@router.get("/recognition/get")
async def get_log(taskId: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(RecognitionLog).where(RecognitionLog.task_id == taskId))
    log = result.scalar_one_or_none()
    ai_result = json.loads(log.ai_result_json) if log and log.ai_result_json else None
    user_correction = json.loads(log.user_correction_json) if log and log.user_correction_json else None
    visual_image_url = log.visual_image_path if log else None
    return ApiResponse(data={
        "ai_result": ai_result,
        "user_correction": user_correction,
        "visual_image_url": visual_image_url,
    })
