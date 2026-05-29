import asyncio
import json
import logging

from sqlalchemy import select, update

from api.database import async_session
from api.models import SKU, TrainJob

logger = logging.getLogger(__name__)

_running_tasks: dict[str, asyncio.Task] = {}


async def process_single_media(
    sku_id: str,
    sku_name: str,
    media_id: str,
    media_url: str,
    processor,
    image_storage,
    inference_executor,
) -> bool:
    """Download, embed, upload masked crop, and cleanup for a single media item.

    Returns True on success, False on failure.
    """
    local_path = await image_storage.download_image(media_url)
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            inference_executor,
            processor.process_and_add,
            sku_id,
            sku_name,
            media_id,
            local_path,
            {"media_url": media_url},
        )
        if result.success and result.crop_path:
            try:
                await image_storage.upload_to_qiniu(result.crop_path)
            except Exception:
                logger.exception("Failed to upload crop for %s/%s", sku_id, media_id)
            finally:
                image_storage.cleanup_download(result.crop_path)
        return result.success
    except Exception:
        logger.exception("Failed to process media %s", media_url)
        return False
    finally:
        image_storage.cleanup_download(local_path)


async def start_embed_task(
    train_job_id: str,
    sku_id: str,
    media_urls: list[str],
    media_ids: list[str],
    processor,
    image_storage,
    sku_name: str,
    inference_executor,
) -> None:
    async def _run():
        try:
            async with async_session() as session:
                result = await session.execute(
                    select(TrainJob).where(TrainJob.train_job_id == train_job_id)
                )
                tj = result.scalar_one_or_none()
                if tj is not None:
                    tj.status = "indexing"
                    await session.commit()

            embedding_failed: list[str] = []

            for idx, (url, media_id) in enumerate(zip(media_urls, media_ids), start=1):
                success = await process_single_media(
                    sku_id, sku_name, media_id, url,
                    processor, image_storage, inference_executor,
                )
                if not success:
                    embedding_failed.append(url)

                progress = int((idx / max(len(media_urls), 1)) * 100)
                async with async_session() as session:
                    await session.execute(
                        update(TrainJob)
                        .where(TrainJob.train_job_id == train_job_id)
                        .values(progress=progress)
                    )
                    await session.commit()

            failed_json = json.dumps(embedding_failed) if embedding_failed else None

            if embedding_failed:
                logger.warning(
                    "Indexing task completed with %d/%d failed images: train_job_id=%s",
                    len(embedding_failed), len(media_urls), train_job_id,
                )
                async with async_session() as session:
                    await session.execute(
                        update(TrainJob)
                        .where(TrainJob.train_job_id == train_job_id)
                        .values(status="completed", progress=100, embedding_failed=failed_json)
                    )
                    await session.execute(
                        update(SKU).where(SKU.sku_id == sku_id).values(train_status="SUCCESS")
                    )
                    await session.commit()
            else:
                async with async_session() as session:
                    await session.execute(
                        update(TrainJob)
                        .where(TrainJob.train_job_id == train_job_id)
                        .values(status="completed", progress=100)
                    )
                    await session.execute(
                        update(SKU).where(SKU.sku_id == sku_id).values(train_status="SUCCESS")
                    )
                    await session.commit()

                logger.info("Indexing task completed: train_job_id=%s", train_job_id)

        except Exception:
            logger.exception("Indexing task failed: train_job_id=%s", train_job_id)
            async with async_session() as session:
                await session.execute(
                    update(TrainJob)
                    .where(TrainJob.train_job_id == train_job_id)
                    .values(status="failed")
                )
                await session.execute(
                    update(SKU).where(SKU.sku_id == sku_id).values(train_status="FAILED")
                )
                await session.commit()

    task = asyncio.create_task(_run())
    _running_tasks[train_job_id] = task
    task.add_done_callback(lambda t: _running_tasks.pop(train_job_id, None))


def get_task_status(train_job_id: str) -> str | None:
    task = _running_tasks.get(train_job_id)
    if task is None:
        return None
    return "running" if not task.done() else "done"
