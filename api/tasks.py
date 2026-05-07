import asyncio
import logging

from sqlalchemy import select, update

from api.database import async_session
from api.models import SKU, TrainJob

logger = logging.getLogger(__name__)

_running_tasks: dict[str, asyncio.Task] = {}


async def start_embed_task(
    train_job_id: str,
    sku_id: str,
    media_urls: list[str],
    media_ids: list[str],
    index_manager,
    image_storage,
    sku_name: str,
) -> None:
    async def _run():
        try:
            async with async_session() as session:
                result = await session.execute(
                    select(TrainJob).where(TrainJob.train_job_id == train_job_id)
                )
                tj = result.scalar_one_or_none()
                if tj is not None:
                    tj.status = "training"
                    await session.commit()

            for idx, (url, media_id) in enumerate(zip(media_urls, media_ids), start=1):
                local_path = await image_storage.download_image(url)
                await asyncio.to_thread(
                    index_manager.embed_and_add_reference,
                    sku_id,
                    sku_name,
                    media_id,
                    local_path,
                )
                progress = int((idx / max(len(media_urls), 1)) * 100)
                async with async_session() as session:
                    await session.execute(
                        update(TrainJob)
                        .where(TrainJob.train_job_id == train_job_id)
                        .values(progress=progress)
                    )
                    await session.commit()

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

            logger.info("Embedding task completed: train_job_id=%s", train_job_id)

        except Exception:
            logger.exception("Embedding task failed: train_job_id=%s", train_job_id)
            async with async_session() as session:
                await session.execute(
                    update(TrainJob)
                    .where(TrainJob.train_job_id == train_job_id)
                    .values(status="failed")
                )
                await session.commit()

    task = asyncio.create_task(_run())
    _running_tasks[train_job_id] = task


def get_task_status(train_job_id: str) -> str | None:
    task = _running_tasks.get(train_job_id)
    if task is None:
        return None
    return "running" if not task.done() else "done"
