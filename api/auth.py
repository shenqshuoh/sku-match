"""API authentication and rate limiting dependencies."""

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


async def check_rate_limit(request: Request) -> None:
    """In-memory per-IP rate limiter. Disabled when RATE_LIMIT <= 0."""
    limit = settings.RATE_LIMIT
    if limit <= 0:
        return
    client_ip = request.client.host if request.client else "0.0.0.0"
    now = time.time()
    timestamps = _client_requests[client_ip]
    # Prune expired entries
    _client_requests[client_ip] = [t for t in timestamps if now - t < _window_secs]
    if len(_client_requests[client_ip]) >= limit:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    _client_requests[client_ip].append(now)
