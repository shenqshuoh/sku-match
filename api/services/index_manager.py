import logging
from pathlib import Path

from src.reference_processor import ReferenceProcessor

logger = logging.getLogger(__name__)


class IndexManager:
    """Adapter wrapping ReferenceProcessor for use via asyncio.to_thread() in API routes."""

    def __init__(self, processor: ReferenceProcessor):
        self.processor = processor

    def embed_and_add_reference(
        self,
        sku_id: str,
        sku_name: str,
        media_id: str,
        image_path: Path,
        metadata: dict | None = None,
    ) -> None:
        self.processor.process_and_add(sku_id, sku_name, media_id, image_path, metadata)

    def delete_sku_references(self, sku_id: str) -> None:
        self.processor.delete_sku_references(sku_id)

    def delete_media_reference(self, sku_id: str, media_id: str) -> None:
        self.processor.delete_media_reference(sku_id, media_id)

    def set_sku_enabled(self, sku_id: str, enabled: bool) -> None:
        self.processor.set_sku_enabled(sku_id, enabled)

    def update_sku_name(self, sku_id: str, new_name: str) -> None:
        self.processor.update_sku_name(sku_id, new_name)
