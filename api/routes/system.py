import json

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import get_db
from api.models import TrainJob
from api.responses import error_response
from api.schemas import ApiResponse

router = APIRouter()


@router.get("/train-status/get", response_model=ApiResponse)
async def train_status(trainJobId: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(TrainJob).where(TrainJob.train_job_id == trainJobId))
    tj = result.scalar_one_or_none()
    if tj is None:
        return error_response(f"trainJobId '{trainJobId}' not found", status_code=404)
    response = {
        "status": tj.status,
        "progress": tj.progress,
        "estimatedTime": tj.estimated_time,
        "failedCount": tj.failed_count,
        "totalCount": tj.total_count,
    }
    if tj.embedding_failed:
        try:
            response["embeddingFailed"] = json.loads(tj.embedding_failed)
        except (json.JSONDecodeError, TypeError):
            response["embeddingFailed"] = []
    return ApiResponse(data=response)
