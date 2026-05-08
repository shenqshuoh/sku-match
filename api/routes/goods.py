import asyncio
import json
import uuid
import logging
from typing import List

from fastapi import APIRouter, Depends, Request
from sqlalchemy import select, update, delete, func, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from api.database import get_db
from api.models import SKU, SKUMedia, TrainJob
from api.schemas import (
    SKUNewRequest,
    SKUUpdateRequest,
    SKUDeleteRequest,
    SKUEnableRequest,
    SKUMediaRequest,
    MediaItem,
    ApiResponse,
    SKUListData,
    MediaResponse,
    StatusResponse,
)
from api.tasks import start_embed_task

logger = logging.getLogger(__name__)
router = APIRouter()


async def _get_sku_or_error(db: AsyncSession, sku_id: str):
    """Fetch SKU by sku_id, returning ApiResponse error if not found."""
    sku = await db.scalar(select(SKU).where(SKU.sku_id == sku_id))
    if sku is None:
        return None, ApiResponse(code=0, msg=f"skuId '{sku_id}' not found")
    return sku, None


@router.post("/sku/new")
async def new_sku(request: SKUNewRequest, req: Request, db: AsyncSession = Depends(get_db)):
    # 1. Check for duplicate skuId and trainJobId
    existing_sku = await db.scalar(select(SKU).where(SKU.sku_id == request.skuId))
    if existing_sku is not None:
        return ApiResponse(code=0, msg=f"skuId '{request.skuId}' already exists")
    existing_job = await db.scalar(select(TrainJob).where(TrainJob.train_job_id == request.trainJobId))
    if existing_job is not None:
        return ApiResponse(code=0, msg=f"trainJobId '{request.trainJobId}' already exists")

    # 2. Insert SKU
    sku = SKU(sku_id=request.skuId, sku_name=request.skuName, enabled=True, train_status="PENDING")
    db.add(sku)
    await db.commit()

    # 2. Insert SKUMedia rows
    media_urls: List[str] = []
    media_ids: List[str] = []
    for url in request.files:
        media_id = uuid.uuid4().hex
        media = SKUMedia(media_id=media_id, sku_id=request.skuId, media_url=url, media_type="IMAGE")
        db.add(media)
        media_urls.append(url)
        media_ids.append(media_id)
    await db.commit()

    # 3. Create TrainJob
    train_job = TrainJob(train_job_id=request.trainJobId, sku_id=request.skuId, status="pending", progress=0)
    db.add(train_job)
    await db.commit()

    # 4. Start embedding task
    sku_name = request.skuName
    await start_embed_task(
        train_job_id=request.trainJobId,
        sku_id=request.skuId,
        media_urls=media_urls,
        media_ids=media_ids,
        processor=req.app.state.processor,
        image_storage=req.app.state.image_storage,
        sku_name=sku_name,
    )

    # 5. Return identifiers
    return ApiResponse(data={"skuId": request.skuId, "trainJobId": request.trainJobId})


@router.post("/sku/update")
async def update_sku(request: SKUUpdateRequest, req: Request, db: AsyncSession = Depends(get_db)):
    # 1. Check SKU exists
    sku, err = await _get_sku_or_error(db, request.skuId)
    if err:
        return err

    # 2. Update SKU name
    await db.execute(
        update(SKU).where(SKU.sku_id == request.skuId).values(sku_name=request.skuName)
    )
    await db.commit()

    # 3. Update Chroma metadata via index manager
    if hasattr(req.app.state, "processor"):
        await asyncio.to_thread(req.app.state.processor.update_sku_name, request.skuId, request.skuName)

    return StatusResponse(status="success")


@router.post("/sku/delete")
async def delete_sku(request: SKUDeleteRequest, req: Request, db: AsyncSession = Depends(get_db)):
    # 1. Delete SKU row (cascade deletes media)
    sku = await db.scalar(select(SKU).where(SKU.sku_id == request.skuId))
    if sku is not None:
        await db.execute(delete(SKU).where(SKU.sku_id == request.skuId))
        await db.commit()

    # 2. Remove references from Chroma
    if hasattr(req.app.state, "processor"):
        await asyncio.to_thread(req.app.state.processor.delete_sku_references, request.skuId)

    return StatusResponse(status="success")


@router.post("/sku/enable")
async def enable_sku(request: SKUEnableRequest, req: Request, db: AsyncSession = Depends(get_db)):
    # 1. Check SKU exists
    sku, err = await _get_sku_or_error(db, request.skuId)
    if err:
        return err

    # 2. Update enabled flag
    await db.execute(
        update(SKU).where(SKU.sku_id == request.skuId).values(enabled=request.enabled)
    )
    await db.commit()

    # 3. Update Chroma
    if hasattr(req.app.state, "processor"):
        await asyncio.to_thread(req.app.state.processor.set_sku_enabled, request.skuId, request.enabled)

    return StatusResponse(status="success")


@router.get("/sku/list")
async def list_skus(page: int = 1, size: int = 20, keyword: str | None = None, db: AsyncSession = Depends(get_db)):
    offset = (page - 1) * size
    kw = f"%{keyword}%" if keyword else None
    base = select(SKU).options(selectinload(SKU.medias))
    if kw:
        base = base.where(or_(SKU.sku_id.ilike(kw), SKU.sku_name.ilike(kw)))
    total_q = select(func.count(SKU.id))
    if kw:
        total_q = total_q.where(or_(SKU.sku_id.ilike(kw), SKU.sku_name.ilike(kw)))
    total_res = await db.execute(total_q)
    total = total_res.scalar_one()
    result = await db.execute(base.offset(offset).limit(size))
    skus = result.scalars().all()

    data = SKUListData(
        list=[
            {
                "id": s.id,
                "skuId": s.sku_id,
                "skuName": s.sku_name,
                "trainStatus": s.train_status,
                "medias": [
                    {"mediaId": m.media_id, "mediaType": m.media_type, "mediaUrl": m.media_url}
                    for m in s.medias
                ],
            }
            for s in skus
        ],
        page=page,
        pageSize=size,
        total=total,
    )

    return ApiResponse(data=data)


@router.post("/sku/media")
async def manage_media(request: SKUMediaRequest, req: Request, db: AsyncSession = Depends(get_db)):
    if request.action == "add":
        # Validate: mediaUrl is required for add action
        for item in request.media:
            if not item.mediaUrl:
                return ApiResponse(code=0, msg="mediaUrl is required for add action")
        # 2. Check SKU exists
        sku, err = await _get_sku_or_error(db, request.skuId)
        if err:
            return err
        sku_name = sku.sku_name
        for item in request.media:
            if item.mediaId:
                continue
            media_id = uuid.uuid4().hex
            media = SKUMedia(media_id=media_id, sku_id=request.skuId, media_url=item.mediaUrl, media_type="IMAGE")
            db.add(media)
            await db.commit()
            downloaded_path = await req.app.state.image_storage.download_image(item.mediaUrl)
            await asyncio.to_thread(
                req.app.state.processor.process_and_add,
                request.skuId,
                sku_name,
                media_id,
                downloaded_path,
            )
        return StatusResponse(status="success")
    elif request.action == "delete":
        for item in request.media:
            if not item.mediaId:
                continue
            await db.execute(delete(SKUMedia).where(SKUMedia.media_id == item.mediaId))
            await db.commit()
            await asyncio.to_thread(
                req.app.state.processor.delete_media_reference,
                request.skuId, item.mediaId,
            )
        return StatusResponse(status="success")
    else:
        return ApiResponse(code=0, msg="Unsupported action")
