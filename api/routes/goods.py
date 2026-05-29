import asyncio
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import List

from fastapi import APIRouter, Depends, Request
from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from api.database import get_db
from api.dependencies import get_image_storage, get_indexer, get_inference_executor, get_processor
from api.models import SKU, SKUMedia, TrainJob
from api.schemas import (
    ApiResponse,
    MediaResponse,
    SKUDeleteRequest,
    SKUEnableRequest,
    SKUListData,
    SKUListItem,
    SKUMediaRequest,
    SKUNewRequest,
    SKUUpdateRequest,
)
from api.services.image_storage import ImageStorage
from api.tasks import process_single_media, start_embed_task
from src.indexer import SKUIndexer
from src.reference_processor import ReferenceProcessor

logger = logging.getLogger(__name__)
router = APIRouter()


async def _get_sku_or_error(db: AsyncSession, sku_id: str):
    """Fetch SKU by sku_id, returning ApiResponse error if not found."""
    sku = await db.scalar(select(SKU).where(SKU.sku_id == sku_id))
    if sku is None:
        return None, ApiResponse(code=0, msg=f"skuId '{sku_id}' not found")
    return sku, None


@router.post("/sku/new", response_model=ApiResponse)
async def new_sku(
    request: SKUNewRequest,
    processor: ReferenceProcessor = Depends(get_processor),
    image_storage: ImageStorage = Depends(get_image_storage),
    inference_executor: ThreadPoolExecutor = Depends(get_inference_executor),
    db: AsyncSession = Depends(get_db),
):
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

    # 3. Insert SKUMedia rows
    media_urls: List[str] = []
    media_ids: List[str] = []
    for url in request.files:
        media_id = uuid.uuid4().hex
        media = SKUMedia(media_id=media_id, sku_id=request.skuId, media_url=url, media_type="IMAGE")
        db.add(media)
        media_urls.append(url)
        media_ids.append(media_id)
    await db.commit()

    # 4. Create TrainJob
    train_job = TrainJob(train_job_id=request.trainJobId, sku_id=request.skuId, status="pending", progress=0)
    db.add(train_job)
    await db.commit()

    # 5. Start embedding task
    await start_embed_task(
        train_job_id=request.trainJobId,
        sku_id=request.skuId,
        media_urls=media_urls,
        media_ids=media_ids,
        processor=processor,
        image_storage=image_storage,
        sku_name=request.skuName,
        inference_executor=inference_executor,
    )

    # 6. Return identifiers
    return ApiResponse(data={"skuId": request.skuId, "trainJobId": request.trainJobId})


@router.post("/sku/update", response_model=ApiResponse)
async def update_sku(
    request: SKUUpdateRequest,
    indexer: SKUIndexer = Depends(get_indexer),
    db: AsyncSession = Depends(get_db),
):
    # 1. Check SKU exists
    sku, err = await _get_sku_or_error(db, request.skuId)
    if err:
        return err

    # 2. Update SKU name
    await db.execute(
        update(SKU).where(SKU.sku_id == request.skuId).values(sku_name=request.skuName)
    )
    await db.commit()

    # 3. Update Chroma metadata via indexer
    await asyncio.to_thread(indexer.update_sku_name, request.skuId, request.skuName)

    return ApiResponse(data={})


@router.post("/sku/delete", response_model=ApiResponse)
async def delete_sku(
    request: SKUDeleteRequest,
    indexer: SKUIndexer = Depends(get_indexer),
    db: AsyncSession = Depends(get_db),
):
    # 1. Delete SKU row (cascade deletes media)
    sku, err = await _get_sku_or_error(db, request.skuId)
    if err:
        return err
    await db.delete(sku)  # ORM delete — triggers cascade="all, delete-orphan"
    await db.commit()

    # 2. Remove references from Chroma
    await asyncio.to_thread(indexer.delete_sku, request.skuId)

    return ApiResponse(data={})


@router.post("/sku/enable", response_model=ApiResponse)
async def enable_sku(
    request: SKUEnableRequest,
    indexer: SKUIndexer = Depends(get_indexer),
    db: AsyncSession = Depends(get_db),
):
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
    await asyncio.to_thread(indexer.set_enabled, request.skuId, request.enabled)

    return ApiResponse(data={})


def _to_full_url(request: Request, path: str) -> str:
    """Convert a relative path to a full URL using the request's base URL."""
    if path.startswith(("http://", "https://")):
        return path
    base = str(request.base_url).rstrip("/")
    return f"{base}{path}"


@router.get("/sku/list", response_model=ApiResponse)
async def list_skus(request: Request, page: int = 1, size: int = 20, keyword: str | None = None, db: AsyncSession = Depends(get_db)):
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
            SKUListItem(
                id=s.id,
                skuId=s.sku_id,
                skuName=s.sku_name,
                trainStatus=s.train_status,
                medias=[
                    MediaResponse(
                        mediaId=m.media_id,
                        mediaType=m.media_type,
                        mediaUrl=_to_full_url(request, m.media_url),
                    )
                    for m in s.medias
                ],
            )
            for s in skus
        ],
        page=page,
        pageSize=size,
        total=total,
    )

    return ApiResponse(data=data)


@router.post("/sku/media", response_model=ApiResponse)
async def manage_media(
    request: SKUMediaRequest,
    processor: ReferenceProcessor = Depends(get_processor),
    indexer: SKUIndexer = Depends(get_indexer),
    image_storage: ImageStorage = Depends(get_image_storage),
    inference_executor: ThreadPoolExecutor = Depends(get_inference_executor),
    db: AsyncSession = Depends(get_db),
):
    if request.action == "add":
        # Validate: mediaUrl is required for add action
        for item in request.media:
            if not item.mediaUrl:
                return ApiResponse(code=0, msg="mediaUrl is required for add action")
        # Check SKU exists
        sku, err = await _get_sku_or_error(db, request.skuId)
        if err:
            return err
        sku_name = sku.sku_name
        embedding_failed: list[str] = []

        for item in request.media:
            if item.mediaId:
                continue
            if not item.mediaUrl:
                continue
            media_id = uuid.uuid4().hex
            success = await process_single_media(
                request.skuId, sku_name, media_id, item.mediaUrl,
                processor, image_storage, inference_executor,
            )
            if success:
                media = SKUMedia(
                    media_id=media_id, sku_id=request.skuId,
                    media_url=item.mediaUrl, media_type="IMAGE",
                )
                db.add(media)
                await db.commit()
            else:
                embedding_failed.append(item.mediaUrl or "")
        if embedding_failed:
            return ApiResponse(code=0, msg="Some images failed to embed", data={"embedding_failed": embedding_failed})
        return ApiResponse(data={})
    elif request.action == "delete":
        for item in request.media:
            if not item.mediaId:
                continue
            await db.execute(delete(SKUMedia).where(SKUMedia.media_id == item.mediaId))
            await db.commit()
            await asyncio.to_thread(
                indexer.delete_media,
                request.skuId, item.mediaId,
            )
        return ApiResponse(data={})
    else:
        return ApiResponse(code=0, msg="Unsupported action")
