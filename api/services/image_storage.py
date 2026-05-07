import logging
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

import httpx

logger = logging.getLogger(__name__)


class ImageStorage:

    def __init__(self, results_dir: str | Path = "results", download_timeout: int = 30):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.download_timeout = download_timeout

    async def download_image(self, url: str) -> Path:
        downloads_dir = self.results_dir / "downloads"
        downloads_dir.mkdir(parents=True, exist_ok=True)

        source = Path(url)
        if source.exists():
            ext = source.suffix or ".jpg"
            local_path = downloads_dir / f"{uuid4()}{ext}"
            local_path.write_bytes(source.read_bytes())
            logger.info("Copied local file %s -> %s", url, local_path)
            return local_path

        parsed = urlparse(url)
        ext = Path(parsed.path).suffix or ".jpg"
        local_path = downloads_dir / f"{uuid4()}{ext}"

        try:
            async with httpx.AsyncClient(timeout=self.download_timeout) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                local_path.write_bytes(resp.content)
            logger.info("Downloaded %s -> %s", url, local_path)
            return local_path
        except Exception as e:
            logger.error("Failed to download %s: %s", url, e)
            raise ValueError(f"Could not download image from {url}: {e}") from e

    def get_result_path(self, task_id: str) -> Path:
        annotated_dir = self.results_dir / "annotated"
        annotated_dir.mkdir(parents=True, exist_ok=True)
        return annotated_dir / f"{task_id}_annotated.jpg"

    def get_result_url(self, task_id: str) -> str:
        return f"/results/annotated/{task_id}_annotated.jpg"
