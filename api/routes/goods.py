import asyncio
import logging
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import List

from fastapi import APIRouter, Depends, Request
from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from api.database import get_db
from api.dependencies import (
    get_color_store,
    get_image_storage,
    get_indexer,
    get_inference_executor,
    get_patch_store,
    get_processor,
)
from api.models import SKU, SKUMedia, TrainJob
from api.responses import error_response, to_full_url
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
from api.tasks import recompute_train_status, start_embed_task
from src.color_store import ColorStore
from src.indexer import SKUIndexer
from src.patch_store import PatchStore
from src.reference_processor import ReferenceProcessor

logger = logging.getLogger(__name__)
router = APIRouter()


async def _get_sku_or_error(db: AsyncSession, sku_id: str):
    """Fetch SKU by sku_id, returning ApiResponse error if not found."""
    sku = await db.scalar(select(SKU).where(SKU.sku_id == sku_id))
    if sku is None:
        return None, error_response(f"skuId '{sku_id}' not found", status_code=404)
    return sku, None


_NUM_PREFIX_RE = re.compile(r"^(\d+)_")


def _sku_id_suffix(sku_id: str) -> str:
    """The bare part of a stored sku_id (everything after a leading numeric prefix)."""
    return _NUM_PREFIX_RE.sub("", sku_id, count=1)


def _next_sku_number(all_ids: list[str]) -> int:
    """Max leading numeric prefix + 1 (starts at 100001 on an empty table)."""
    numbers = [int(m.group(1)) for sid in all_ids if (m := _NUM_PREFIX_RE.match(sid))]
    return max(numbers, default=100000) + 1


@router.post("/sku/new", response_model=ApiResponse)
async def new_sku(
    request: SKUNewRequest,
    processor: ReferenceProcessor = Depends(get_processor),
    image_storage: ImageStorage = Depends(get_image_storage),
    inference_executor: ThreadPoolExecutor = Depends(get_inference_executor),
    db: AsyncSession = Depends(get_db),
):
    # 1. Auto-number: the submitted skuId is the verbatim bare suffix; the server
    #    always prepends the next sequence number (never mutates the bare part).
    bare = request.skuId
    all_skus = (await db.execute(select(SKU.sku_id, SKU.sku_name))).fetchall()

    # Duplicate gate: same bare suffix AND same name -> probable double submission.
    # Same suffix with a different name is allowed (numbering proceeds).
    for sid, sname in all_skus:
        if _sku_id_suffix(sid) == bare and sname == request.skuName:
            return error_response(
                f"sku '{bare}' with name '{request.skuName}' already exists as '{sid}'",
                status_code=409,
            )

    final_id = f"{_next_sku_number([sid for sid, _ in all_skus])}_{bare}"

    existing_sku = await db.scalar(select(SKU).where(SKU.sku_id == final_id))
    if existing_sku is not None:
        return error_response(f"skuId '{final_id}' already exists", status_code=409)
    existing_job = await db.scalar(select(TrainJob).where(TrainJob.train_job_id == request.trainJobId))
    if existing_job is not None:
        return error_response(f"trainJobId '{request.trainJobId}' already exists", status_code=409)

    # 2. Insert SKU (unique-constraint backstop for concurrent next-number races)
    sku = SKU(sku_id=final_id, sku_name=request.skuName, enabled=True, train_status="pending")
    db.add(sku)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        return error_response(
            f"skuId '{final_id}' already exists (concurrent add) — retry", status_code=409
        )

    # 3. Insert SKUMedia rows
    media_urls: List[str] = []
    media_ids: List[str] = []
    for url in request.files:
        media_id = uuid.uuid4().hex
        media = SKUMedia(media_id=media_id, sku_id=final_id, media_url=url, media_type="IMAGE")
        db.add(media)
        media_urls.append(url)
        media_ids.append(media_id)
    await db.commit()

    # 4. Create TrainJob
    train_job = TrainJob(train_job_id=request.trainJobId, sku_id=final_id, status="pending", progress=0)
    db.add(train_job)
    await db.commit()

    # 5. Start embedding task
    await start_embed_task(
        train_job_id=request.trainJobId,
        sku_id=final_id,
        media_urls=media_urls,
        media_ids=media_ids,
        processor=processor,
        image_storage=image_storage,
        sku_name=request.skuName,
        inference_executor=inference_executor,
    )

    # 6. Return identifiers (final auto-numbered skuId — clients must read this)
    return ApiResponse(data={"skuId": final_id, "trainJobId": request.trainJobId})


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
    patch_store: PatchStore | None = Depends(get_patch_store),
    color_store: ColorStore | None = Depends(get_color_store),
    db: AsyncSession = Depends(get_db),
):
    # 1. Check SKU exists
    sku, err = await _get_sku_or_error(db, request.skuId)
    if err:
        return err
    await db.delete(sku)  # ORM delete — triggers cascade="all, delete-orphan"
    await db.commit()

    # 2. Remove references from Chroma + stored artifacts (patches, color descriptors)
    await asyncio.to_thread(indexer.delete_sku, request.skuId)
    if patch_store is not None:
        await asyncio.to_thread(patch_store.delete_sku, request.skuId)
    if color_store is not None:
        await asyncio.to_thread(color_store.delete_sku, request.skuId)

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


_VALID_TRAIN_STATUSES = {"pending", "indexing", "completed", "failed"}


@router.get("/sku/list", response_model=ApiResponse)
async def list_skus(
    request: Request,
    page: int = 1,
    size: int = 20,
    keyword: str | None = None,
    trainStatus: str | None = None,
    enabled: bool | None = None,
    db: AsyncSession = Depends(get_db),
):
    # Validate trainStatus (enabled is typed by FastAPI — bad values become 422 automatically)
    if trainStatus is not None and trainStatus not in _VALID_TRAIN_STATUSES:
        return error_response(
            f"trainStatus must be one of {sorted(_VALID_TRAIN_STATUSES)}, got '{trainStatus}'",
            status_code=400,
        )

    offset = (page - 1) * size
    kw = f"%{keyword}%" if keyword else None

    def apply_filters(q):
        if kw:
            q = q.where(or_(SKU.sku_id.ilike(kw), SKU.sku_name.ilike(kw)))
        if trainStatus is not None:
            q = q.where(SKU.train_status == trainStatus)
        if enabled is not None:
            q = q.where(SKU.enabled == enabled)
        return q

    base = apply_filters(select(SKU).options(selectinload(SKU.medias)))
    total_q = apply_filters(select(func.count(SKU.id)))
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
                        mediaUrl=to_full_url(request, m.media_url) or "",
                        failed=m.failed,
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
    patch_store: PatchStore | None = Depends(get_patch_store),
    color_store: ColorStore | None = Depends(get_color_store),
    db: AsyncSession = Depends(get_db),
):
    if request.action == "add":
        # Validate: mediaUrl is required for add action
        for item in request.media:
            if not item.mediaUrl:
                return error_response("mediaUrl is required for add action", status_code=400)
        # Check SKU exists
        sku, err = await _get_sku_or_error(db, request.skuId)
        if err:
            return err
        sku_name = sku.sku_name

        # Resolve trainJobId: panel-supplied (dedupe 409) or server-generated
        train_job_id = (request.trainJobId or "").strip() or None
        if train_job_id:
            existing_job = await db.scalar(
                select(TrainJob).where(TrainJob.train_job_id == train_job_id)
            )
            if existing_job is not None:
                return error_response(
                    f"trainJobId '{train_job_id}' already exists", status_code=409
                )

        # Items carrying a mediaId are skipped (no re-embed of existing media)
        embeddable = [item for item in request.media if not item.mediaId]
        if not embeddable:
            return ApiResponse(data={})

        # Insert media rows upfront (failed=False); embedding runs async
        media_ids = [uuid.uuid4().hex for _ in embeddable]
        for item, media_id in zip(embeddable, media_ids):
            db.add(
                SKUMedia(
                    media_id=media_id, sku_id=request.skuId,
                    media_url=item.mediaUrl, media_type="IMAGE",
                )
            )
        if train_job_id is None:
            train_job_id = f"job_{request.skuId}_{int(time.time())}"
        db.add(TrainJob(train_job_id=train_job_id, sku_id=request.skuId, status="pending"))
        await db.commit()

        await start_embed_task(
            train_job_id=train_job_id,
            sku_id=request.skuId,
            media_urls=[item.mediaUrl or "" for item in embeddable],
            media_ids=media_ids,
            processor=processor,
            image_storage=image_storage,
            sku_name=sku_name,
            inference_executor=inference_executor,
            pre_cropped_flags=[bool(item.preCropped) for item in embeddable],
        )
        # Partial failures are reported via /system/train-status/get polling,
        # not in this response (failed media rows are flagged in the SKU list)
        return ApiResponse(data={"trainJobId": train_job_id, "mediaIds": media_ids})
    elif request.action == "delete":
        deleted = False
        for item in request.media:
            if not item.mediaId:
                continue
            await db.execute(delete(SKUMedia).where(SKUMedia.media_id == item.mediaId))
            await db.commit()
            doc_id = f"{request.skuId}__{item.mediaId}"
            await asyncio.to_thread(indexer.delete_media, request.skuId, item.mediaId)
            if patch_store is not None:
                await asyncio.to_thread(patch_store.delete, doc_id)
            if color_store is not None:
                await asyncio.to_thread(color_store.delete, doc_id)
            deleted = True
        if deleted:
            # Deleting the last failed image flips a failed SKU back to
            # completed; deleting everything leaves it pending
            await recompute_train_status(request.skuId)
        return ApiResponse(data={})
    else:
        return error_response(f"Unsupported action: '{request.action}'", status_code=400)
