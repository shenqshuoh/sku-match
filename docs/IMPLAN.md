# Improvement Plan: Security, API Consistency & Resource Management

> Tracked from comprehensive code quality review (2026-05-09).
> P3 (dedup/type safety) and P4 (hygiene) are implemented — see commit `bed2f0f`.
> P0 (partial), P1, and P2 are implemented — see commit `1ef6deb`.

---

## P0 — Security (before any public exposure)

### S0: API Key Authentication — ✅ DONE
- **File**: `api/auth.py` (new), `api/config.py`, `api/app.py`
- **Implementation**: `verify_api_key()` dependency checks `X-API-Key` header. Disabled by default (empty `API_KEY`). Applied to all `/api/v1/*` routers via `dependencies=[Depends(verify_api_key)]`.

### S1: Input Validation — ✅ DONE
- **File**: `api/schemas.py`
- **Implementation**: `Field(max_length=...)` on all IDs/names, `Literal["IMAGE","VIDEO"]` for mode, `Literal["add","delete"]` for action, `field_validator` for non-empty strings.

### S2: Rate Limiting — ✅ DONE
- **File**: `api/auth.py`, `api/config.py`
- **Implementation**: `check_rate_limit()` in-memory per-IP rate limiter. Disabled by default (`RATE_LIMIT=0`). Applied alongside API key dependency.

### S3: SSRF Protection
- **File**: `api/services/image_storage.py`
- **Fix**: Block internal IPs in `download_image()` URL handler
- Block: `127.0.0.0/8`, `169.254.0.0/16`, `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`, `[::1]`, `0.0.0.0`
- Reject `file://` scheme explicitly

### S4: Path Traversal Prevention
- **File**: `api/services/image_storage.py`
- **Fix**: Restrict local file paths to whitelisted directories (`data/`, `results/`)
- Reject absolute paths and `..` traversal

### S5: Error Information Leak
- **File**: `api/routes/recognition.py`
- **Fix**: Return generic error message to client, log full `str(e)` server-side
- Replace `msg=str(e)` with `msg="Internal error"` + `logger.error()`

### S6: File Type Validation
- **File**: `api/services/image_storage.py`
- **Fix**: Validate downloaded content is an image (check Content-Type header or magic bytes)
- Reject non-image extensions

---

## P1 — API Consistency

### A1: Standardize Response Format — ✅ DONE
- **Files**: All route files
- **Fix**: All endpoints return `ApiResponse(code=1/0, data=..., msg="success")`. Removed `StatusResponse`, `LogResponse`, `TrainStatusResponse`.

### A2: Literal Types for Schema Fields — ✅ DONE
- **File**: `api/schemas.py`
- **Fix**: `mode: Literal["IMAGE", "VIDEO"]`, `action: Literal["add", "delete"]`, `fixType` still uses `str`

### A3: HTTP Status Codes
- **Files**: All route files
- **Fix**: Use proper HTTP status codes:
  - 200: Success
  - 400: Validation error (invalid input)
  - 404: Not found (skuId, taskId, trainJobId)
  - 409: Conflict (duplicate skuId, taskId)
  - 500: Internal server error
- Requires adding `status_code=` to `ApiResponse` or using `JSONResponse`

### A4: Use Response Models
- **Files**: `api/schemas.py`, `api/services/recognition.py`
- **Fix**: Use `DetectionItem` and `DetectData` as `response_model` on detect endpoint, or remove them
- Recognition service should return typed data instead of raw dict

### A5: Typed ApiResponse.data
- **File**: `api/schemas.py`
- **Fix**: Replace `data: Any` with `data: dict | list | None` or use Generic `ApiResponse[T]`

---

## P2 — Resource Leaks & Robustness

### R1: Downloaded File Cleanup — ✅ DONE
- **Files**: `api/routes/recognition.py`, `api/tasks.py`, `api/services/image_storage.py`
- **Fix**: `cleanup_download()` method added to ImageStorage. Called after processing in both recognition routes and embedding tasks. Cleanup on error paths too.

### R2: Connection Pool Tuning — ✅ DONE
- **File**: `api/database.py`
- **Fix**: Added `pool_size=5, max_overflow=10, pool_recycle=3600, pool_pre_ping=True` to engine.

### R3: Unclosed PIL Images
- **Files**: `src/embedder.py:142`, `src/matcher.py:76`, `src/reference_processor.py:85,132,193`, `api/services/recognition.py:52`
- **Fix**: Use `with Image.open(...) as img:` or explicit `img.close()` after use

### R4: Task Dict Cleanup
- **File**: `api/tasks.py`
- **Fix**: Remove completed tasks from `_running_tasks` after they finish
- Add task cancellation on shutdown

### R5: Engine Disposal on Shutdown
- **File**: `api/app.py`
- **Fix**: Add `from api.database import engine; await engine.dispose()` in lifespan shutdown

### R6: Graceful Shutdown
- **File**: `api/app.py`
- **Fix**: Change `shutdown(wait=False)` to `shutdown(wait=True, cancel_futures=True)` with a timeout
