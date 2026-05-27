from fastapi import APIRouter, Depends

import json
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import get_db
from api.models import TrainJob

router = APIRouter()


@router.get("/train-status/get")
async def train_status(trainJobId: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(TrainJob).where(TrainJob.train_job_id == trainJobId))
    tj = result.scalar_one_or_none()
    if tj is None:
        return {"status": "fail", "msg": "train_job not found"}
    response = {
        "status": tj.status,
        "progress": tj.progress,
        "estimated_time": tj.estimated_time,
    }
    if tj.skipped_images:
        try:
            response["skipped_images"] = json.loads(tj.skipped_images)
        except (json.JSONDecodeError, TypeError):
            response["skipped_images"] = []
    return response
