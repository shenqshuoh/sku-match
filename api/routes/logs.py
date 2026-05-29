import json
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from api.database import get_db
from api.models import RecognitionLog
from api.schemas import ApiResponse
from sqlalchemy import select

router = APIRouter()


@router.get("/recognition/get", response_model=ApiResponse)
async def get_log(taskId: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(RecognitionLog).where(RecognitionLog.task_id == taskId))
    log = result.scalar_one_or_none()
    if log is None:
        return ApiResponse(code=0, msg=f"taskId '{taskId}' not found", data={"ai_result": None, "user_correction": None, "visual_image_url": None})
    ai_result = json.loads(log.ai_result_json) if log.ai_result_json else None
    user_correction = json.loads(log.user_correction_json) if log.user_correction_json else None
    visual_image_url = log.visual_image_path
    return ApiResponse(data={
        "ai_result": ai_result,
        "user_correction": user_correction,
        "visual_image_url": visual_image_url,
    })
