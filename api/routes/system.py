from fastapi import APIRouter, Depends

import json
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import get_db
from api.models import TrainJob
from api.schemas import ApiResponse

router = APIRouter()


@router.get("/train-status/get", response_model=ApiResponse)
async def train_status(trainJobId: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(TrainJob).where(TrainJob.train_job_id == trainJobId))
    tj = result.scalar_one_or_none()
    if tj is None:
        return ApiResponse(code=0, msg="train_job not found")
    response = {
        "status": tj.status,
        "progress": tj.progress,
        "estimated_time": tj.estimated_time,
    }
    if tj.embedding_failed:
        try:
            response["embedding_failed"] = json.loads(tj.embedding_failed)
        except (json.JSONDecodeError, TypeError):
            response["embedding_failed"] = []
    return ApiResponse(data=response)
