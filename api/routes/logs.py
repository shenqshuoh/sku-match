import json
import logging
from datetime import datetime

from fastapi import APIRouter, Depends, Request
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import get_db
from api.dependencies import get_image_storage
from api.models import RecognitionLog
from api.responses import error_response, to_full_url
from api.schemas import ApiResponse, LogDeleteRequest, LogListData, LogListItem, LogStatusRequest
from api.services.image_storage import ImageStorage

logger = logging.getLogger(__name__)
router = APIRouter()

_VALID_STATUSES = {"pending", "corrected", "reviewed"}


def _parse_time_param(value: str, name: str) -> datetime:
    """Parse an ISO datetime query param. Aware values are converted to server-local naive."""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone().replace(tzinfo=None)
    return parsed


@router.get("/recognition/get", response_model=ApiResponse)
async def get_log(
    request: Request,
    taskId: str,
    includeOriginal: bool = False,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(RecognitionLog).where(RecognitionLog.task_id == taskId))
    log = result.scalar_one_or_none()
    if log is None:
        return error_response(f"taskId '{taskId}' not found", status_code=404)
    final_result = json.loads(log.final_result_json) if log.final_result_json else None
    original_result = json.loads(log.ai_result_json) if log.ai_result_json else None
    user_correction = json.loads(log.user_correction_json) if log.user_correction_json else None
    # `aiResult` is the effective result: corrected view when a fix exists, original otherwise
    ai_result = final_result if final_result is not None else original_result
    # Expand any relative image paths (local fallback) to absolute URLs for external access
    if isinstance(ai_result, dict) and str(ai_result.get("matchedImage", "")).startswith("/"):
        ai_result = {**ai_result, "matchedImage": to_full_url(request, ai_result["matchedImage"])}
    data = {
        "aiResult": ai_result,
        "correctionStatus": log.correction_status,
        "correctionCount": log.correction_count,
        "correctedAt": log.corrected_at.isoformat() if log.corrected_at else None,
        "detectionDiff": log.detection_diff,
        "skuMismatchCount": log.sku_mismatch_count,
        "inputImageUrl": to_full_url(request, log.input_image_path),
        "userCorrection": user_correction,
        "visualImageUrl": to_full_url(request, log.visual_image_path),
    }
    if includeOriginal:
        # originalResult is the verbatim immutable AI output — served as stored
        data["originalResult"] = original_result
        data["originalImageUrl"] = to_full_url(request, log.original_visual_image_path)
    return ApiResponse(data=data)


@router.get("/recognition/list", response_model=ApiResponse)
async def list_logs(
    request: Request,
    page: int = 1,
    size: int = 20,
    startTime: str | None = None,
    endTime: str | None = None,
    correctionStatus: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    # Validate + parse filters
    start_dt = end_dt = None
    if startTime is not None:
        try:
            start_dt = _parse_time_param(startTime, "startTime")
        except ValueError:
            return error_response(f"startTime '{startTime}' is not a valid ISO datetime", status_code=400)
    if endTime is not None:
        try:
            end_dt = _parse_time_param(endTime, "endTime")
        except ValueError:
            return error_response(f"endTime '{endTime}' is not a valid ISO datetime", status_code=400)
    if correctionStatus is not None and correctionStatus not in _VALID_STATUSES:
        return error_response(
            f"correctionStatus must be one of {sorted(_VALID_STATUSES)}, got '{correctionStatus}'",
            status_code=400,
        )

    conditions = []
    if start_dt is not None:
        conditions.append(RecognitionLog.created_at >= start_dt)
    if end_dt is not None:
        conditions.append(RecognitionLog.created_at <= end_dt)
    if correctionStatus is not None:
        conditions.append(RecognitionLog.correction_status == correctionStatus)

    total = await db.scalar(select(func.count()).select_from(RecognitionLog).where(*conditions)) or 0

    offset = (page - 1) * size
    query = (
        select(RecognitionLog)
        .where(*conditions)
        .order_by(RecognitionLog.created_at.desc(), RecognitionLog.id.desc())
        .offset(offset)
        .limit(size)
    )
    result = await db.execute(query)
    logs = result.scalars().all()

    data = LogListData(
        list=[
            LogListItem(
                taskId=row.task_id,
                createdAt=row.created_at,
                correctionStatus=row.correction_status,
                correctedAt=row.corrected_at,
                correctionCount=row.correction_count,
                detectionCount=row.detection_count,
                detectionDiff=row.detection_diff,
                skuMismatchCount=row.sku_mismatch_count,
                inputImageUrl=to_full_url(request, row.input_image_path),
                visualImageUrl=to_full_url(request, row.visual_image_path),
            )
            for row in logs
        ],
        page=page,
        pageSize=size,
        total=total,
    )
    return ApiResponse(data=data)


@router.post("/recognition/status", response_model=ApiResponse)
async def set_log_status(request: LogStatusRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(RecognitionLog).where(RecognitionLog.task_id == request.taskId))
    log = result.scalar_one_or_none()
    if log is None:
        return error_response(f"taskId '{request.taskId}' not found", status_code=404)
    # Workflow state only: corrected_at / correction_count track actual fix submissions
    # and are intentionally left untouched by manual status changes.
    log.correction_status = request.correctionStatus
    await db.commit()
    return ApiResponse(data={})


@router.post("/recognition/delete", response_model=ApiResponse)
async def delete_logs(
    request: LogDeleteRequest,
    image_storage: ImageStorage = Depends(get_image_storage),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(RecognitionLog).where(RecognitionLog.task_id.in_(request.taskIds))
    )
    logs = result.scalars().all()
    deleted_ids = [row.task_id for row in logs]

    # Best-effort local file cleanup (annotated, inputs, annotated_original).
    # NOTE: CDN objects are intentionally left orphaned — no Qiniu delete credentials
    # configured yet. When credentials become available, add deletion here via
    # image_storage.cdn_key_from_url() on visual_image_path / original_visual_image_path.
    for row in logs:
        for path in (
            image_storage.get_result_path(row.task_id),
            image_storage.get_input_path(row.task_id),
            image_storage.get_annotated_original_path(row.task_id),
        ):
            try:
                path.unlink(missing_ok=True)
            except Exception:
                logger.warning("Failed to delete file %s", path, exc_info=True)

    for row in logs:
        await db.delete(row)
    await db.commit()

    not_found = [tid for tid in request.taskIds if tid not in deleted_ids]
    return ApiResponse(data={"deleted": len(deleted_ids), "notFound": not_found})
