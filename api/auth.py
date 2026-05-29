"""API authentication and rate limiting dependencies."""

import asyncio
import time
import logging
from collections import defaultdict

from fastapi import Header, HTTPException, Request

from api.config import settings

logger = logging.getLogger(__name__)

# --- API Key ---


async def verify_api_key(x_api_key: str | None = Header(None, alias="X-API-Key")) -> None:
    """Verify API key if configured. Skip when API_KEY is empty (auth disabled)."""
    if not settings.API_KEY:
        return
    if x_api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# --- Rate Limiter ---

_window_secs: int = 60
_client_requests: dict[str, list[float]] = defaultdict(list)
_rate_limit_lock = asyncio.Lock()


async def check_rate_limit(request: Request) -> None:
    """In-memory per-IP rate limiter. Disabled when RATE_LIMIT <= 0."""
    limit = settings.RATE_LIMIT
    if limit <= 0:
        return
    client_ip = request.client.host if request.client else "0.0.0.0"
    now = time.time()
    async with _rate_limit_lock:
        # Prune expired entries
        timestamps = [t for t in _client_requests[client_ip] if now - t < _window_secs]
        if len(timestamps) >= limit:
            _client_requests[client_ip] = timestamps
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        timestamps.append(now)
        _client_requests[client_ip] = timestamps
        # Prune IPs with no recent requests to prevent unbounded memory growth
        if len(_client_requests) > 1000:
            empty_ips = [ip for ip, ts in _client_requests.items() if not ts]
            for ip in empty_ips:
                del _client_requests[ip]
