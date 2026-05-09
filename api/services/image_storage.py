import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

import httpx

logger = logging.getLogger(__name__)


class ImageStorage:

    def __init__(
        self,
        results_dir: str | Path = "results",
        download_timeout: int = 30,
        qiniu_token_url: str = "",
        qiniu_upload_url: str = "https://upload.qiniup.com",
        qiniu_domain: str = "",
        qiniu_key_prefix: str = "",
    ):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.download_timeout = download_timeout
        self.qiniu_token_url = qiniu_token_url
        self.qiniu_upload_url = qiniu_upload_url
        self.qiniu_domain = qiniu_domain.rstrip("/") + "/" if qiniu_domain else ""
        self.qiniu_key_prefix = qiniu_key_prefix
        self._qiniu_token: str = ""
        self._qiniu_token_deadline: float = 0

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

    def cleanup_download(self, path: Path) -> None:
        """Remove a downloaded temp file. Silently ignore if already deleted."""
        try:
            if path.exists() and path.is_file():
                path.unlink()
                logger.debug("Cleaned up temp file: %s", path)
        except Exception:
            logger.warning("Failed to cleanup temp file: %s", path, exc_info=True)

    def get_result_path(self, task_id: str) -> Path:
        annotated_dir = self.results_dir / "annotated"
        annotated_dir.mkdir(parents=True, exist_ok=True)
        return annotated_dir / f"{task_id}_annotated.jpg"

    def get_result_url(self, task_id: str) -> str:
        return f"/results/annotated/{task_id}_annotated.jpg"

    # ── Qiniu upload ──

    async def _fetch_qiniu_token(self) -> None:
        """Fetch a fresh upload token from the internal token service."""
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(self.qiniu_token_url)
            resp.raise_for_status()
            data = resp.json()
        if data.get("code") != 1:
            raise ValueError(f"Qiniu token service error: {data.get('msg', 'unknown')}")
        token_data = data["data"]
        self._qiniu_token = token_data["upToken"]
        self._qiniu_token_deadline = float(token_data["deadline"])

    def _is_qiniu_token_valid(self) -> bool:
        """Check if cached token is still valid (with 60s safety margin)."""
        return bool(self._qiniu_token) and time.time() < self._qiniu_token_deadline - 60

    def _build_qiniu_key(self, filename: str) -> str:
        """Build a storage key with date-based path: {prefix}{YYYY-MM/DD}/{filename}."""
        now = datetime.now(tz=timezone.utc)
        date_path = now.strftime("%Y-%m/%d")
        return f"{self.qiniu_key_prefix}{date_path}/{filename}"

    async def upload_to_qiniu(self, file_path: Path, key: str | None = None) -> str:
        """Upload a file to Qiniu and return the full CDN URL.

        Args:
            file_path: Local file to upload.
            key: Optional storage key. Auto-generated if not provided.

        Returns:
            Full CDN URL (e.g. https://vr.jihaihotpot.com/sku-match/2026-05/09/xxx.jpg)
        """
        if not self.qiniu_token_url:
            logger.warning("Qiniu not configured, returning local URL")
            return ""

        if not self._is_qiniu_token_valid():
            await self._fetch_qiniu_token()

        if key is None:
            key = self._build_qiniu_key(file_path.name)

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                self.qiniu_upload_url,
                data={"token": self._qiniu_token, "key": key},
                files={"file": (file_path.name, file_path.read_bytes(), "image/jpeg")},
            )
            resp.raise_for_status()

        cdn_url = f"{self.qiniu_domain}{key}"
        logger.info("Uploaded %s -> %s", file_path.name, cdn_url)
        return cdn_url
