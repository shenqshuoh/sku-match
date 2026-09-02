"""Shared API response utilities."""

import json
from typing import Any
from urllib.parse import quote

from fastapi import Request
from fastapi.responses import JSONResponse


class UnicodeJSONResponse(JSONResponse):
    """JSONResponse with ensure_ascii=False for proper Unicode output."""

    def render(self, content) -> bytes:
        return json.dumps(content, ensure_ascii=False, allow_nan=False).encode("utf-8")


def error_response(msg: str, status_code: int = 400, data: Any = None) -> UnicodeJSONResponse:
    """Return an ApiResponse-formatted error with a proper HTTP status code.

    The body envelope ``{"code": 0, "data": ..., "msg": ...}`` is preserved for
    backwards compatibility — only the HTTP status code changes.
    """
    return UnicodeJSONResponse(
        status_code=status_code,
        content={"code": 0, "data": data, "msg": msg},
    )


def to_full_url(request: Request, path: str | None) -> str | None:
    """Expand a stored relative path (e.g. /results/...) to an absolute URL.

    Absolute URLs (CDN) pass through unchanged; None passes through as None.
    Non-ASCII characters are percent-encoded per RFC 3986.
    """
    if path is None or path.startswith(("http://", "https://")):
        return path
    base = str(request.base_url).rstrip("/")
    return f"{base}{quote(path, safe='/')}"
