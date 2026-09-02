import logging
import shutil
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
        qiniu_iovip_url: str = "",
        qiniu_key_prefix: str = "",
    ):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.download_timeout = download_timeout
        self.qiniu_token_url = qiniu_token_url
        self.qiniu_upload_url = qiniu_upload_url
        self.qiniu_domain = qiniu_domain.rstrip("/") + "/" if qiniu_domain else ""
        self.qiniu_iovip_url = qiniu_iovip_url.rstrip("/") if qiniu_iovip_url else ""
        self.qiniu_key_prefix = qiniu_key_prefix
        self._qiniu_token: str = ""
        self._qiniu_token_deadline: float = 0

    def cleanup_old_results(self, max_age_hours: int = 72) -> int:
        """Delete expired files across all results subdirectories. Returns count of deleted files.

        Covers annotated/ (final annotation images), inputs/ (persisted source images),
        and annotated_original/ (pre-fix annotation snapshots).
        """
        cutoff = time.time() - max_age_hours * 3600
        deleted = 0
        for sub in ("annotated", "inputs", "annotated_original"):
            sub_dir = self.results_dir / sub
            if not sub_dir.exists():
                continue
            for f in sub_dir.iterdir():
                if f.is_file() and f.stat().st_mtime < cutoff:
                    try:
                        f.unlink()
                        deleted += 1
                    except Exception:
                        logger.warning("Failed to delete old file: %s", f, exc_info=True)
        if deleted:
            logger.info("Cleaned up %d expired files older than %dh", deleted, max_age_hours)
        return deleted

    def get_input_path(self, task_id: str) -> Path:
        inputs_dir = self.results_dir / "inputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        return inputs_dir / f"{task_id}_original.jpg"

    def get_input_url(self, task_id: str) -> str:
        """Servable local static URL for the persisted input image."""
        return f"/results/inputs/{task_id}_original.jpg"

    def persist_input(self, downloaded_path: Path, task_id: str) -> Path:
        """Move a downloaded source image into the inputs store for fix-time re-annotation."""
        dest = self.get_input_path(task_id)
        shutil.move(str(downloaded_path), str(dest))
        logger.debug("Persisted input for %s -> %s", task_id, dest)
        return dest

    def get_annotated_original_path(self, task_id: str) -> Path:
        """Path of the pre-fix annotation snapshot (frozen at first fix)."""
        orig_dir = self.results_dir / "annotated_original"
        orig_dir.mkdir(parents=True, exist_ok=True)
        return orig_dir / f"{task_id}_annotated.jpg"

    def cdn_key_from_url(self, url: str) -> str | None:
        """Extract the Qiniu storage key from a CDN URL (inverse of upload_to_qiniu)."""
        if self.qiniu_domain and url.startswith(self.qiniu_domain):
            return url[len(self.qiniu_domain):]
        return None

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
            # Rewrite Qiniu CDN URLs to origin storage domain for faster domestic access
            download_url = url
            headers: dict[str, str] = {}
            if self.qiniu_iovip_url and self.qiniu_domain and url.startswith(self.qiniu_domain):
                key = url[len(self.qiniu_domain):]
                download_url = f"{self.qiniu_iovip_url}/{key}"
                headers["Host"] = urlparse(self.qiniu_domain).netloc

            async with httpx.AsyncClient(timeout=self.download_timeout) as client:
                resp = await client.get(download_url, headers=headers or None)
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

    def annotated_key(self, task_id: str, version: int) -> str:
        """Versioned key for the final annotation image.

        Insert-only upload tokens reject same-key overwrites (Qiniu 614), so every
        write gets a fresh version: detect = v1, each fix = v+1. Callers must store
        the returned URL (visual_image_path / matchedImage) — it changes per version.
        """
        return self._build_qiniu_key(f"{task_id}_annotated_v{version}.jpg")

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
