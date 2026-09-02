from collections.abc import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from api.config import settings
from api.models import Base

engine = create_async_engine(
    settings.DATABASE_URL,
    future=True,
    pool_size=5,
    max_overflow=10,
    pool_recycle=3600,
    pool_pre_ping=True,
)
async_session = async_sessionmaker(bind=engine, expire_on_commit=False, class_=AsyncSession)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as session:
        yield session


# Columns added after initial release. create_all() only creates missing tables,
# not missing columns, so these are applied via idempotent ALTER TABLE.
_COLUMN_MIGRATIONS: dict[str, dict[str, str]] = {
    "recognition_log": {
        "correction_status": "TEXT NOT NULL DEFAULT 'pending'",
        "correction_count": "INTEGER NOT NULL DEFAULT 0",
        "corrected_at": "DATETIME",
        "detection_count": "INTEGER NOT NULL DEFAULT 0",
        "final_result_json": "TEXT",
        "original_visual_image_path": "VARCHAR",
        "input_image_path": "VARCHAR",
        "detection_diff": "INTEGER NOT NULL DEFAULT 0",
        "sku_mismatch_count": "INTEGER NOT NULL DEFAULT 0",
    },
    "sku_media": {
        "failed": "BOOLEAN NOT NULL DEFAULT 0",
    },
}


async def _migrate_columns(conn: AsyncConnection) -> None:
    """Add missing columns per table (SQLite, idempotent)."""
    detection_count_added = False
    for table, new_cols in _COLUMN_MIGRATIONS.items():
        result = await conn.execute(text(f"PRAGMA table_info({table})"))  # noqa: S608
        columns = {row[1] for row in result.fetchall()}
        for col, ddl in new_cols.items():
            if col not in columns:
                await conn.execute(
                    text(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}")  # noqa: S608
                )
                if table == "recognition_log" and col == "detection_count":
                    detection_count_added = True

    # One-time backfill: detection counts for pre-existing logs (runs only when the
    # column was just added — subsequent startups see it present and skip)
    if detection_count_added:
        await conn.execute(text(
            "UPDATE recognition_log SET detection_count = "
            "COALESCE(json_array_length(json_extract(ai_result_json, '$.detections')), 0) "
            "WHERE ai_result_json IS NOT NULL"
        ))


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _migrate_columns(conn)
