import asyncio
import json
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
from fastapi import APIRouter, Depends, Request
from PIL import Image, ImageOps
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import get_db
from api.dependencies import (
    get_device,
    get_image_storage,
    get_indexer,
    get_inference_executor,
    get_recognition_service,
)
from api.models import RecognitionLog
from api.responses import error_response, to_full_url
from api.schemas import ApiResponse, DetectData, DetectRequest, FixRequest
from api.services.image_storage import ImageStorage
from api.services.recognition import RecognitionService
from src.image_utils import draw_annotations
from src.indexer import SKUIndexer

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/detect", response_model=ApiResponse[DetectData])
async def detect(
    request: DetectRequest,
    raw_request: Request,
    recognition_service: RecognitionService = Depends(get_recognition_service),
    image_storage: ImageStorage = Depends(get_image_storage),
    inference_executor: ThreadPoolExecutor = Depends(get_inference_executor),
    device: str = Depends(get_device),
    db: AsyncSession = Depends(get_db),
):
    downloaded_path = None
    log = None
    try:
        # 1. Check for duplicate taskId
        existing = await db.execute(
            select(RecognitionLog).where(RecognitionLog.task_id == request.taskId)
        )
        if existing.scalar_one_or_none() is not None:
            return error_response(
                f"taskId '{request.taskId}' already exists", status_code=409
            )

        # 2. Log request to recognition_log table
        log = RecognitionLog(
            task_id=request.taskId,
            request_json=request.model_dump_json(),
        )
        db.add(log)
        await db.commit()

        # 3. Download image via image storage
        downloaded_path = await image_storage.download_image(request.files)

        # 4. Run recognition (CPU/GPU-bound, offload to dedicated inference thread)
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                inference_executor,
                recognition_service.recognize,
                downloaded_path,
                request.taskId,
                request.roiRect,
                device,
            )
        except RuntimeError as e:
            if "Index is empty" in str(e):
                return error_response(
                    "No SKUs indexed yet — add reference images first",
                    status_code=503,
                )
            raise

        # 4b. Persist the source image for fix-time re-annotation (replaces temp cleanup)
        image_storage.persist_input(downloaded_path, request.taskId)
        downloaded_path = None  # moved — skip cleanup in success/error paths

        # 5. Upload annotated image to Qiniu (versioned key: detect = v1; each fix
        #    writes v+1 — insert-only tokens reject same-key overwrites)
        matched_image = result.get("matchedImage", "")
        qiniu_upload_failed = False
        if matched_image:
            annotated_path = image_storage.get_result_path(request.taskId)
            if annotated_path.exists():
                try:
                    cdn_url = await image_storage.upload_to_qiniu(
                        annotated_path, key=image_storage.annotated_key(request.taskId, 1)
                    )
                    if cdn_url:
                        result["matchedImage"] = cdn_url
                        matched_image = cdn_url
                except Exception as e:
                    logger.warning("Qiniu upload failed, keeping local path: %s", e)
                    qiniu_upload_failed = True
        if qiniu_upload_failed:
            result["qiniuUploadFailed"] = True

        # 5b. Upload the unannotated input image to Qiniu (permanent URL for logs /
        #     fine-tuning data). Fallback: ephemeral local static URL (72h retention).
        input_image_path = image_storage.get_input_path(request.taskId)
        if input_image_path.exists():
            try:
                cdn_url = await image_storage.upload_to_qiniu(input_image_path)
                input_image_url = cdn_url if cdn_url else image_storage.get_input_url(request.taskId)
            except Exception as e:
                logger.warning("Qiniu input upload failed, using local path: %s", e)
                input_image_url = image_storage.get_input_url(request.taskId)
        else:
            input_image_url = None

        # 6. Update log with result, visual image path, and detection count
        log.ai_result_json = json.dumps(result)
        log.visual_image_path = matched_image
        log.input_image_path = input_image_url
        log.detection_count = len(result.get("detections", []))
        await db.commit()

        # 7. Return response (validate through DetectData); expand relative image
        #    paths (local fallback) to absolute URLs for external access
        result["matchedImage"] = to_full_url(raw_request, result.get("matchedImage")) or ""
        return ApiResponse(data=DetectData(**result))
    except Exception as e:
        logger.exception("Recognition detect failed: %s", e)
        if downloaded_path is not None:
            try:
                image_storage.cleanup_download(downloaded_path)
            except Exception:
                pass
        # Update log with error if log record was created
        if log is not None:
            try:
                log.ai_result_json = json.dumps({"error": str(e)})
                await db.commit()
            except Exception:
                logger.exception("Failed to update log on error for taskId=%s", request.taskId)
        return error_response(str(e), status_code=500)


async def _load_input_image(task_id: str, log: RecognitionLog, image_storage: ImageStorage) -> np.ndarray | None:
    """Resolve the detect-time source image as an RGB numpy array.

    Resolution order: persisted inputs/ store -> re-download from the request source.
    Returns None (and logs) when both fail.
    """
    input_path = image_storage.get_input_path(task_id)

    if not input_path.exists():
        source_url = None
        try:
            req = json.loads(log.request_json) if log.request_json else None
            source_url = req.get("files") if isinstance(req, dict) else None
        except json.JSONDecodeError:
            pass
        if not source_url:
            logger.warning("Cannot obtain input image for %s: no persisted input and no source URL", task_id)
            return None
        try:
            downloaded = await image_storage.download_image(source_url)
            input_path = image_storage.persist_input(downloaded, task_id)
        except Exception:
            logger.warning("Cannot obtain input image for %s: source unavailable", task_id, exc_info=True)
            return None

    try:
        # EXIF rotation must match the detect pipeline, or boxes misalign on phone photos
        image = ImageOps.exif_transpose(Image.open(input_path)).convert("RGB")
        return np.array(image)
    except Exception:
        logger.warning("Failed to load input image for %s", task_id, exc_info=True)
        return None


async def _preserve_original_annotation(log: RecognitionLog, image_storage: ImageStorage) -> None:
    """Best-effort: snapshot the pre-fix annotation image on the FIRST fix only.

    Resolution order for the snapshot: existing local annotated file -> redraw from
    the input image + the ORIGINAL ai_result_json boxes -> give up (leave path null).
    Uploads to a distinct Qiniu key so the CDN copy outlives local retention.
    """
    task_id = log.task_id
    snapshot_path = image_storage.get_annotated_original_path(task_id)
    annotated_path = image_storage.get_result_path(task_id)

    if snapshot_path.exists() and log.original_visual_image_path:
        # Already preserved — nothing to do (idempotent on repeated first-fix calls)
        return

    if not snapshot_path.exists():
        if annotated_path.exists():
            shutil.copyfile(annotated_path, snapshot_path)
        else:
            # Local annotated file purged — redraw the pre-fix annotation from
            # the immutable original result
            image_np = await _load_input_image(task_id, log, image_storage)
            if image_np is None:
                return
            try:
                original = json.loads(log.ai_result_json) if log.ai_result_json else {}
                await asyncio.to_thread(
                    draw_annotations, image_np, original.get("detections", []), snapshot_path,
                )
            except Exception:
                logger.warning("Failed to redraw pre-fix annotation for %s", task_id, exc_info=True)
                return

    # Persist URL: overwrite same-key on CDN (idempotent across repeated first-fixes)
    url = log.visual_image_path or ""
    cdn_key = image_storage.cdn_key_from_url(url)
    if cdn_key:
        snapshot_key = cdn_key.replace("_annotated", "_annotated_original")
        try:
            await image_storage.upload_to_qiniu(snapshot_path, key=snapshot_key)
            log.original_visual_image_path = f"{image_storage.qiniu_domain}{snapshot_key}"
        except Exception:
            logger.warning("Qiniu upload of pre-fix snapshot failed for %s", task_id, exc_info=True)
            log.original_visual_image_path = str(snapshot_path)
    else:
        # No CDN (upload failed at detect / local-only deployment) — serve local static
        log.original_visual_image_path = f"/results/annotated_original/{task_id}_annotated.jpg"


async def _regenerate_annotated_image(
    log: RecognitionLog, final: dict, image_storage: ImageStorage
) -> str | None:
    """Best-effort: re-draw annotations for the corrected result onto the input image.

    Returns the URL of the regenerated image: a fresh versioned CDN URL when the
    upload succeeds, or the local static URL when it doesn't (the local file IS
    regenerated either way). None when redrawing itself failed — the JSON result
    stays authoritative.
    """
    task_id = log.task_id

    image_np = await _load_input_image(task_id, log, image_storage)
    if image_np is None:
        return None

    try:
        annotated_path = image_storage.get_result_path(task_id)
        await asyncio.to_thread(draw_annotations, image_np, final.get("detections", []), annotated_path)
    except Exception:
        logger.warning("Failed to redraw annotations for %s", task_id, exc_info=True)
        return None

    # Versioned key (correction_count already incremented for this fix): insert-only
    # tokens reject same-key overwrites, so each fix uploads to a fresh key and the
    # stored URL changes per fix.
    version = (log.correction_count or 0) + 1
    try:
        cdn_url = await image_storage.upload_to_qiniu(
            annotated_path, key=image_storage.annotated_key(task_id, version)
        )
        if cdn_url:
            return cdn_url
    except Exception:
        logger.warning("Qiniu re-upload failed for %s (local file regenerated)", task_id, exc_info=True)
    return image_storage.get_result_url(task_id)


@router.post("/fix", response_model=ApiResponse)
async def fix(
    request: FixRequest,
    raw_request: Request,
    indexer: SKUIndexer = Depends(get_indexer),
    image_storage: ImageStorage = Depends(get_image_storage),
    db: AsyncSession = Depends(get_db),
):
    try:
        # 1. Look up RecognitionLog by task_id
        result = await db.execute(select(RecognitionLog).where(RecognitionLog.task_id == request.taskId))
        log = result.scalar_one_or_none()
        if log is None:
            return error_response(f"taskId '{request.taskId}' not found", status_code=404)

        # 2. Load the immutable original AI result (reject failed/error payloads)
        if not log.ai_result_json:
            return error_response(f"taskId '{request.taskId}' has no AI result to correct", status_code=400)
        original = json.loads(log.ai_result_json)
        detections = original.get("detections") if isinstance(original, dict) else None
        if not isinstance(detections, list):
            return error_response(
                f"taskId '{request.taskId}' has no valid AI result to correct (detection failed?)",
                status_code=400,
            )

        # 2b. Base for this fix: the current corrected result when present (fixes are
        #     incremental — prior fixes are kept), otherwise the original result
        final = None
        if log.final_result_json:
            try:
                loaded = json.loads(log.final_result_json)
                if isinstance(loaded, dict) and isinstance(loaded.get("detections"), list):
                    final = loaded
            except json.JSONDecodeError:
                logger.warning(
                    "final_result_json corrupt for %s, rebasing on original", request.taskId
                )
        if final is None:
            final = json.loads(log.ai_result_json)  # fresh copy — original stays immutable

        # 3. Validate itemIds against the current result (items removed by an
        #    earlier fix can no longer be referenced). fixType "add" creates a
        #    new entry and carries no itemId.
        final_dets = final["detections"]
        known_ids = {d.get("itemId") for d in final_dets}
        for fi in request.fixItems:
            if fi.fixType != "add" and fi.itemId not in known_ids:
                return error_response(
                    f"itemId {fi.itemId} not found in taskId '{request.taskId}'",
                    status_code=400,
                )

        # 4. Apply fixes on top of the current result (incremental — earlier fixes
        #    remain in effect). ai_result_json stays immutable.
        next_item_id = max((d.get("itemId") or 0) for d in final_dets)
        for fi in request.fixItems:
            if fi.fixType == "add":
                # Manually added entry: model-output fields don't exist — nulls +
                # source marker keep the data honest (no fake confidence values;
                # fine-tuning pipelines can filter on source)
                new_sku_id = fi.skuId or ""
                sku_name = await asyncio.to_thread(indexer.get_sku_name, new_sku_id) or new_sku_id
                next_item_id += 1
                final_dets.append({
                    "itemId": next_item_id,
                    "bbox": list(fi.roiRect or []),
                    "classId": None,
                    "className": None,
                    "detectionConf": None,
                    "skuId": new_sku_id,
                    "skuName": sku_name,
                    "matchScore": None,
                    "skuDistribution": None,
                    "matchedVectorTags": None,
                    "source": "manual",
                })
                continue
            det = next((d for d in final_dets if d.get("itemId") == fi.itemId), None)
            if det is None:
                return error_response(
                    f"itemId {fi.itemId} no longer present after an earlier fix in the same request",
                    status_code=400,
                )
            if fi.fixType == "remove":
                final_dets.remove(det)
            elif fi.fixType == "reassign":
                # No validation against the live SKU table: corrections may reference
                # deleted SKUs (historical data). Fallback to the raw id when unknown.
                new_sku_id = fi.skuId or ""
                sku_name = await asyncio.to_thread(indexer.get_sku_name, new_sku_id) or new_sku_id
                det["skuId"] = new_sku_id
                det["skuName"] = sku_name
            elif fi.fixType == "adjust-roi":
                det["bbox"] = list(fi.roiRect or [])

        # 5. Recompute counts from the remaining detections
        counts: dict[str, int] = {}
        for d in final_dets:
            sid = d.get("skuId", "")
            if sid:
                counts[sid] = counts.get(sid, 0) + 1
        final["counts"] = counts

        # 5b. Diff metrics: current final vs original (cumulative across all fixes).
        #     Added = fix-added entries still present (source="manual" — every add
        #     is stamped at creation and history was backfilled); removed = original
        #     entries no longer present. itemId-reuse after removal (add reuses a
        #     freed max id) is handled: the reused id counts as added via its
        #     manual source, and the original entry it shadows counts as removed.
        original_by_id = {d.get("itemId"): d for d in detections}
        detections_added = sum(1 for d in final_dets if d.get("source") == "manual")
        survivors = len(final_dets) - detections_added
        detections_removed = max(0, len(detections) - survivors)
        sku_mismatch_count = sum(
            1
            for d in final_dets
            if d.get("itemId") in original_by_id
            and original_by_id[d["itemId"]].get("skuId") != d.get("skuId")
        )

        # 6. Persist — fixes accumulate: userCorrection keeps every submitted item
        #    in submission order (full audit trail across fixes)
        try:
            corrections = json.loads(log.user_correction_json) if log.user_correction_json else []
        except json.JSONDecodeError:
            corrections = []
        if not isinstance(corrections, list):
            corrections = []
        corrections.extend(fi.model_dump() for fi in request.fixItems)
        log.user_correction_json = json.dumps(corrections)
        log.final_result_json = json.dumps(final)
        log.correction_status = "corrected"
        log.correction_count = (log.correction_count or 0) + 1
        log.detections_added = detections_added
        log.detections_removed = detections_removed
        log.sku_mismatch_count = sku_mismatch_count
        log.corrected_at = datetime.now()

        # 7. Best-effort image updates — snapshot pre-fix annotation on the FIRST fix,
        #    then regenerate the final annotation (fresh versioned CDN URL per fix)
        if log.correction_count == 1:
            await _preserve_original_annotation(log, image_storage)
        new_image_url = await _regenerate_annotated_image(log, final, image_storage)
        if new_image_url:
            log.visual_image_path = new_image_url
            final["matchedImage"] = new_image_url
            log.final_result_json = json.dumps(final)
        await db.commit()  # persist original_visual_image_path / new image URL
        return ApiResponse(
            data={"matchedImage": to_full_url(raw_request, new_image_url)} if new_image_url else {}
        )
    except Exception as e:
        logger.exception("Recognition fix failed: %s", e)
        return error_response(str(e), status_code=500)
