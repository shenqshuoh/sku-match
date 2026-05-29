# Codebase Audit Report — Surprise Behaviors

> **Date**: 2026-05-27
> **Scope**: Full API + ML pipeline codebase audit for silent failures, race conditions, data integrity issues, and unexpected behaviors.
> **Trigger**: Discovery of silent fallback in `ReferenceProcessor.process_and_add()` — failed reference image detection silently fell back to full-image embedding, polluting the vector index with noisy vectors.

---

## Summary

| Severity | Count | Fixed | Remaining | Key Themes |
|----------|-------|-------|-----------|-----------|
| **Critical** | 3 | 3 | 0 | GPU race conditions (goods/tasks bypass `inference_executor`), inconsistent API response format |
| **High** | 6 | 6 | 0 | DB/Chroma desync, silent failures, response bloat, resource leaks, API contract violations |
| **Medium** | 8 | 5 | 3 | Race conditions, memory leaks, security (SSL), unhandled edge cases |
| **Low** | 8 | 6 | 2 | Minor inconsistencies, dead code, edge cases |
| **Security & Infra** | 8 | 0 | 8 | SSRF, path traversal, error leaks, file validation, resource management, HTTP status codes |

> **Fix history**: `902c17e` (audit #3, #12), `85711b3` (chores), `await` fix (tasks.py sync method), cascade delete fix (ORM delete), Qiniu CDN workaround, audit #4/#5/#6 fixes, #7 (top-5 distribution), #8 (annotated image cleanup), #10 (rate limiter), #13 (empty index), #16 (_running_tasks), #17 (phantom logs), #18/#22 (ApiResponse standardization), #19 (match_conf threshold), #20 (double-open), #21 (ultralytics weights), #24 (Qiniu fallback flag).

---

## High Severity Issues

*(All resolved — see "Resolved & Ignored Issues" section.)*

---

## Medium Severity Issues

### Issue #10 — Rate limiter not thread-safe and leaks memory

- **Severity**: Medium
- **Location**: `api/auth.py:27,37-41`
- **Category**: Race condition / resource leak

**Description**: `_client_requests` is a `defaultdict(list)` shared across all async requests. The pruning at line 39 creates a new list, but the check at line 40 and append at line 41 are not atomic — two concurrent requests from the same IP could both pass the rate limit check before either appends. Also, IP entries are never fully removed; even pruned entries leave an empty list in the dict, causing unbounded memory growth over time.

**Trigger**: High concurrent traffic from the same IP.

**Impact**: Rate limit can be exceeded by ~2x under concurrent requests. Memory grows linearly with unique client IPs.

---

### Issue #11 — SKU cache in indexer not refreshed atomically with Chroma writes *(Ignored — serialized by executor)*

- **Severity**: Medium
- **Location**: `src/indexer.py:69-71,151,156,162,173,199-200,267-280`
- **Category**: Race condition / data drift

**Description**: After every Chroma write (add, delete, upsert, update), `_cache_valid` is set to `False`. The next `search_batch` call refreshes the cache via `_refresh_cache()`. But between the write and the next search, the cache is stale. Since Chroma operations from goods routes run on different threads than recognition searches, there's a window where a search uses the old cache.

**Trigger**: Concurrent SKU add/delete and recognition.

**Impact**: Stale SKU name cache, incorrect `n_results` calculation, potentially missing newly added SKUs from search results.

> **Decision**: Both writes (goods routes) and reads (recognition) are serialized through the single-threaded `inference_executor`. No actual concurrency gap.

---

### Issue #13 — search_batch with 0 enabled SKUs raises RuntimeError, not handled cleanly *(Fixed)*

- **Severity**: Medium
- **Location**: `src/indexer.py:196-197` and `api/services/recognition.py:134`
- **Category**: Unhandled edge case

**Description**: `search_batch` raises `RuntimeError("Index is empty — no reference embeddings")` when the collection has 0 vectors. If a recognition call arrives before any SKUs have been indexed (or after all are deleted), this exception propagates up through the inference executor, gets caught by the broad `except Exception`, and returns a generic error message. The error message is unhelpful.

**Trigger**: Recognition call with empty Chroma index.

**Impact**: Cryptic error message returned to API consumer instead of a clear "no SKUs indexed" message.

**Fix applied**: Detect route catches `RuntimeError` with "Index is empty" message and returns `ApiResponse(code=0, msg="No SKUs indexed yet — add reference images first")`.

---

### Issue #14 — Qiniu upload reads entire file into memory *(Ignored)*

- **Severity**: Medium
- **Location**: `api/services/image_storage.py:126`
- **Category**: Resource exhaustion

**Description**: `file_path.read_bytes()` loads the entire annotated image into memory as a single allocation. For very large images (e.g., 4K resolution JPEGs), this can be 5-20MB. Combined with other concurrent operations (image download, YOLOE inference), this can push memory usage high.

**Trigger**: Recognition of a large image with Qiniu upload enabled.

**Impact**: Potential OOM on memory-constrained environments.

> **Decision**: Ignored for now. We'll expand memory if insufficient.

---

### Issue #15 — SSL verification disabled globally for all HTTPS connections *(Ignored)*

- **Severity**: Medium
- **Location**: `src/utils.py:39-42` and `src/indexer.py:80-81`
- **Category**: Security

**Description**: `disable_ssl_verification()` monkey-patches `ssl._create_default_https_context` globally. It's called in `indexer.py:init_collection()` (when platform is darwin) and in `embedder.py` (fallback path). This affects **all** subsequent HTTPS connections made by the process, including httpx calls to Qiniu token/upload URLs and image downloads from external URLs.

**Trigger**: Running on macOS, or using the embedder fallback path.

**Impact**: Man-in-the-middle attacks on any HTTPS connection made by the server. Qiniu token endpoint credentials could be intercepted.

> **Decision**: Ignored for now.

---

### Issue #16 — _running_tasks dict grows unboundedly in tasks.py *(Fixed)*

- **Severity**: Medium
- **Location**: `api/tasks.py:12,102`
- **Category**: Resource leak

**Description**: Every `start_embed_task` call adds an entry to `_running_tasks[train_job_id] = task`. Completed tasks are never removed from this dict. Over time, with many SKU creation calls, this dict grows without bound.

**Trigger**: Repeated `POST /api/v1/goods/sku/new` calls.

**Impact**: Slow memory leak in long-running server processes.

**Fix applied**: Added `task.add_done_callback(lambda t: _running_tasks.pop(train_job_id, None))` to auto-remove entries when tasks complete.

---

### Issue #17 — Fix endpoint creates phantom RecognitionLog if taskId doesn't exist *(Fixed)*

- **Severity**: Medium
- **Location**: `api/routes/recognition.py:87-89`
- **Category**: Silent failure / data integrity

**Description**: If a fix request comes in for a `taskId` that doesn't exist in `RecognitionLog`, the code **creates a new log entry** with only `task_id` and `user_correction_json` — no `request_json` or `ai_result_json`. This means a fix can be submitted for a detection that never happened, creating a phantom log entry.

**Trigger**: `POST /api/v1/recognition/fix` with a taskId that was never used for detection.

**Impact**: Phantom log entries with no corresponding detection. Could mask bugs in client logic or be used to pollute the logs.

**Fix applied**: Fix endpoint now returns `{"status": "fail", "msg": "taskId '...' not found"}` when the log entry doesn't exist, instead of creating a phantom entry.

---

## Low Severity Issues

### Issue #18 — Logs endpoint returns null for non-existent taskId without error *(Fixed)*

- **Severity**: Low
- **Location**: `api/routes/logs.py:12-22`
- **Category**: API contract violation

**Description**: When `taskId` is not found, all fields return `null`. The caller cannot distinguish "not found" from "found but no results yet".

**Trigger**: `GET /api/v1/logs/recognition/get?taskId=nonexistent`.

**Impact**: Ambiguous response — client can't tell if the task exists.

**Fix applied**: Logs endpoint now returns `{code: 0, msg: "taskId '...' not found"}` for missing IDs, and `{code: 1, msg: "success", ...}` for found entries.

---

### Issue #19 — match_conf threshold is dead code in API path *(Fixed)*

- **Severity**: Low
- **Location**: `api/services/recognition.py:33,42,138-155`
- **Category**: Logic inconsistency

**Description**: `RecognitionService` accepts `match_conf` in its constructor and stores it, but **never uses it**. Every detection is included in the response regardless of match confidence. The caller must implement their own threshold filtering.

**Trigger**: Recognition call where the top match has score below the configured threshold.

**Impact**: API returns matches with very low confidence. Threshold setting has no effect.

**Fix applied**: Detections with top match score below `match_conf` are now returned with empty `sku_id`, `sku_name`, zero `match_score`, and empty distribution — effectively marking them as unmatched.

---

### Issue #20 — Image opened twice in crop_reference *(Fixed)*

- **Severity**: Low
- **Location**: `src/reference_processor.py:65-79`
- **Category**: Performance / resource waste

**Description**: `crop_reference` runs YOLOE detection (which opens the image internally) then opens the same image again with PIL at line 78. The image is decoded from disk twice.

**Trigger**: Every reference processing call.

**Impact**: Unnecessary I/O and CPU overhead.

**Fix applied**: Open image once with PIL, pass numpy array to YOLOE `predict`. Reuse the same PIL Image for cropping.

---

### Issue #21 — configure_ultralytics_weights silently ignores all errors *(Fixed)*

- **Severity**: Low
- **Location**: `src/utils.py:15-20`
- **Category**: Silent failure

**Description**: The bare `except Exception: pass` means if ultralytics settings can't be updated, the function silently fails and the server will try to download weights from GitHub, which may fail silently or hang in GFW environments.

**Trigger**: Ultralytics settings module changes API.

**Impact**: Silent failure leading to confusing download errors later.

**Fix applied**: Removed try/except wrapper. Function now raises on failure, causing fast fail at startup.

---

### Issue #22 — list_skus uses ApiResponse but other goods endpoints don't *(Fixed)*

- **Severity**: Low
- **Location**: `api/routes/goods.py:183` vs all other handlers in the same file
- **Category**: API contract violation / inconsistency

**Description**: `list_skus` returns `ApiResponse(data=data)` with the standard envelope. Every other handler in `goods.py` returns a bare dict. Inconsistent even within the same router.

**Trigger**: Comparing responses from `/sku/list` vs `/sku/new`.

**Impact**: Client code must handle two different response formats for the same router.

**Fix applied**: All goods endpoints now use `ApiResponse(code=1, data=..., msg="success")` for success and `ApiResponse(code=0, msg=...)` for errors. Also standardized recognition/fix, logs, and system endpoints.

---

### Issue #23 — Mask normalization logic is fragile *(Ignored — masks not in use)*

- **Severity**: Low
- **Location**: `src/image_utils.py:20` and `api/services/recognition.py:107,117`
- **Category**: Edge case

**Description**: `normalize_mask` checks `if mask.max() <= 1` to decide whether to scale. The logic works correctly for current use but is fragile and confusing — relies on implicit assumptions about mask value ranges.

**Trigger**: Unlikely edge case with specific mask values.

**Impact**: No functional impact currently, but could break with mask format changes.

> **Decision**: Ignored — YOLOE segmentation masks are not currently used in production (retina_masks=False).

---

### Issue #24 — Qiniu upload failure silently falls back to local path *(Fixed)*

- **Severity**: Low
- **Location**: `api/routes/recognition.py:56-62`
- **Category**: Silent fallback

**Description**: If Qiniu upload fails, the local path `/results/annotated/{taskId}_annotated.jpg` is used as `matched_image`. This path is accessible via static mount if the client can reach the API server directly. But if the API is behind a reverse proxy that doesn't proxy `/results/`, the URL will be broken.

**Trigger**: Qiniu service outage.

**Impact**: `matched_image` URL returns 404 from the client's perspective.

**Fix applied**: When Qiniu upload fails, response now includes `"qiniu_upload_failed": true` flag so the client knows the `matched_image` URL is a local fallback path.

---

### Issue #25 — No database migration strategy *(Ignored — defer until schema changes)*

- **Severity**: Low
- **Location**: `api/database.py:24-26`
- **Category**: Data integrity

**Description**: `init_db` uses `Base.metadata.create_all()` which only creates tables that don't exist. It does not apply schema changes to existing tables. If the ORM models change (e.g., adding a column), existing databases will not be updated automatically.

**Trigger**: Deploying a version with schema changes.

**Impact**: Runtime errors when new columns are accessed but don't exist in the DB.

> **Decision**: Ignored for now. Current schema is stable. Should add Alembic migration support before any schema changes are planned.

---

## Security & Infrastructure (from IMPLAN.md)

> Items migrated from archived `docs/IMPLAN.md.bak`. S0–S2 (auth, validation, rate limiting) already implemented. A1–A2 (ApiResponse, Literal types) already implemented. R1–R2 (cleanup, pool tuning) already implemented. R4 (task dict cleanup) = Issue #16 (fixed). Remaining open items:

### Issue #26 — No SSRF protection on image download

- **Severity**: Medium
- **Location**: `api/services/image_storage.py:36-69`
- **Category**: Security

**Description**: `download_image()` accepts arbitrary URLs from API requests. No validation prevents requests to internal network addresses (127.0.0.1, 10.x, 172.16.x, 192.168.x, 169.254.x, [::1]). The `file://` scheme is not explicitly rejected. An attacker could use the image download to scan internal services.

**Trigger**: Malicious `files` parameter in detect/new_sku/media requests.

**Impact**: Server-side request forgery — internal network scanning, cloud metadata access (169.254.169.254).

---

### Issue #27 — No path traversal prevention on local file downloads

- **Severity**: Medium
- **Location**: `api/services/image_storage.py:40-46`
- **Category**: Security

**Description**: `download_image()` checks if the URL is a local file path via `Path(url).exists()`. No validation restricts the path to whitelisted directories. Absolute paths and `..` traversal are not rejected.

**Trigger**: `files` parameter set to `/etc/passwd`, `../../.env`, or any absolute path on the server.

**Impact**: Arbitrary file read from the server filesystem.

---

### Issue #28 — Error messages expose internal details to API consumers

- **Severity**: Low
- **Location**: `api/routes/recognition.py:92`
- **Category**: Information leak

**Description**: The catch-all `except Exception as e` returns `msg=str(e)` which may include file paths, Python tracebacks, or internal service details.

**Trigger**: Any unhandled exception during recognition.

**Impact**: Information disclosure — helps attackers understand internal architecture.

---

### Issue #29 — No file type validation on downloaded images

- **Severity**: Low
- **Location**: `api/services/image_storage.py:36-69`
- **Category**: Security / data integrity

**Description**: `download_image()` saves whatever content comes back from the URL without checking Content-Type or file magic bytes. A non-image file could be saved and passed to PIL/YOLOE.

**Trigger**: URL pointing to a non-image file (HTML, executable, etc.).

**Impact**: Unexpected errors in image processing, potential for crafted-input exploits.

---

### Issue #30 — PIL Images never explicitly closed

- **Severity**: Low
- **Location**: `src/embedder.py`, `src/reference_processor.py`, `api/services/recognition.py`
- **Category**: Resource leak

**Description**: `Image.open()` is called without `with` statements or explicit `.close()`. PIL lazy-loads image data — file handles remain open until garbage collection.

**Trigger**: Every image processing call.

**Impact**: File handle leak under sustained load.

---

### Issue #31 — SQLAlchemy engine not disposed on shutdown

- **Severity**: Low
- **Location**: `api/app.py` lifespan shutdown
- **Category**: Resource management

**Description**: The lifespan shutdown closes Chroma and the inference executor but does not call `await engine.dispose()` to close the SQLAlchemy connection pool.

**Trigger**: Server shutdown.

**Impact**: Connections may not be cleanly closed, especially with connection pooling enabled.

---

### Issue #32 — Inference executor shutdown doesn't wait or cancel futures

- **Severity**: Low
- **Location**: `api/app.py:159-162`
- **Category**: Resource management

**Description**: `inference_executor.shutdown(wait=False)` abandons in-flight GPU tasks. Should use `shutdown(wait=True, cancel_futures=True)` with a timeout.

**Trigger**: Server restart during active recognition.

**Impact**: GPU tasks may be interrupted mid-inference, potentially corrupting GPU state.

---

### Issue #33 — No HTTP status codes (all responses return 200)

- **Severity**: Low
- **Location**: All route files
- **Category**: API contract

**Description**: All endpoints return HTTP 200 with `code=0` for errors. Proper REST practice uses HTTP status codes: 400 for validation errors, 404 for not found, 409 for conflicts, 500 for internal errors.

**Trigger**: Any error response.

**Impact**: API consumers cannot use standard HTTP error handling. Monitoring/alerting based on status codes won't work.

## Priority Recommendations

### Completed

1. ~~**Set SKU.train_status="FAILED" in the generic exception handler** (Issue #4)~~ — Fixed.
2. ~~**Remove or make sku_distribution opt-in** (Issue #7)~~ — Fixed (top 5 only).
3. ~~**Clean up annotated images periodically** (Issue #8)~~ — Fixed (background task, 24h).

### Short-term (when needed)

4. ~~Add proper error responses for non-existent resources (Issues #6, #18)~~ — Fixed.
5. ~~Prevent phantom log creation on fix (Issue #17)~~ — Fixed.
6. ~~Clean up `_running_tasks` dict (Issue #16)~~ — Fixed.
7. ~~Standardize API response format (Issue #22)~~ — Fixed.
8. Add Alembic migration support (Issue #25) — when schema changes are planned.

### Deferred

- Issue #14 (Qiniu file memory) — expand memory if insufficient.
- Issue #15 (SSL verification) — macOS compat, acceptable risk.
- Issue #23 (mask normalization) — masks not in production use.

### Security (before public exposure)

- Issue #26 (SSRF protection) — block internal IPs in download_image().
- Issue #27 (path traversal) — restrict local file paths to whitelisted dirs.
- Issue #28 (error info leak) — return generic error messages to client.
- Issue #29 (file type validation) — validate downloaded content is an image.

### Infrastructure

- Issue #30 (PIL Image close) — use `with Image.open(...)` pattern.
- Issue #31 (engine disposal) — add `await engine.dispose()` on shutdown.
- Issue #32 (executor shutdown) — use `shutdown(wait=True, cancel_futures=True)`.
- Issue #33 (HTTP status codes) — use proper status codes instead of always 200.

> Full roadmap with prioritization and trigger conditions: see `docs/PLAN.md`.

---

---

## Resolved & Ignored Issues

> Issues that have been fixed or explicitly marked as intentional. Kept for historical reference.

### Issue #4 — new_sku commits SKU row before embedding task — DB/Chroma desync on failure *(Fixed)*

- **Severity**: High
- **Location**: `api/routes/goods.py:49-68` and `api/tasks.py:91-99`
- **Category**: Data integrity / DB-Chroma drift

**Description**: `new_sku` commits the SKU row, SKUMedia rows, and TrainJob to the DB (lines 52, 63, 68), then fires off a background `asyncio.Task`. If the embedding task fails (e.g., all images fail detection, download error, GPU OOM), the task catches the exception (line 91), sets `status="failed"` on the TrainJob, but **does not** update `SKU.train_status` — only the `skipped_images` branch updates it to "FAILED". The generic exception handler updates the TrainJob but **never sets SKU.train_status to FAILED**, leaving it as "PENDING" forever.

**Trigger**: Network failure during image download, GPU OOM, corrupted image file.

**Impact**: SKU remains in `train_status="PENDING"` permanently. Subsequent recognition calls may try to match against an SKU with no embeddings in Chroma. TrainJob status says "failed" but SKU says "PENDING" — inconsistent state.

**Fix applied**: Added `update(SKU).where(SKU.sku_id == sku_id).values(train_status="FAILED")` in the generic `except Exception` handler of `start_embed_task` (tasks.py). SKU.train_status is now always set to "FAILED" when the embedding task fails.

---

### Issue #5 — Detect log record committed before recognition — not updated on failure *(Fixed)*

- **Severity**: High
- **Location**: `api/routes/recognition.py:27-32` and `api/routes/recognition.py:65-67`
- **Category**: Data integrity

**Description**: The detect endpoint inserts a `RecognitionLog` row with just `request_json` at line 31-32, then runs recognition. If the recognition call fails mid-way, the catch block at line 71-78 returns an error response but **never updates the log record** with the error. The log row remains with `ai_result_json=NULL` and `visual_image_path=NULL`, indistinguishable from an in-progress state.

**Trigger**: Any exception during recognition after the log insert.

**Impact**: Silent data loss — recognition errors are not persisted in the log. The log entry exists but with no result, making debugging and auditing difficult.

**Fix applied**: Added error logging to the `except` block in detect endpoint — `log.ai_result_json` is now set to `{"error": str(e)}` when recognition fails, making errors queryable in the log.

---

### Issue #6 — sku/delete returns success even when SKU doesn't exist *(Fixed)*

- **Severity**: High
- **Location**: `api/routes/goods.py:107-118`
- **Category**: Silent failure / API contract violation

**Description**: `delete_sku` checks if the SKU exists (line 109) and only deletes if found. But if the SKU is `None`, it **skips the DB delete entirely** and still falls through to delete from Chroma (line 115-116) and returns `{"status": "success"}`. The caller thinks the delete worked, but if the SKU never existed, no error is signaled.

**Trigger**: `POST /api/v1/goods/sku/delete` with a non-existent `skuId`.

**Impact**: Caller cannot distinguish "deleted" from "never existed". Could mask bugs in the caller's logic.

**Fix applied**: Delete endpoint now returns `{"status": "fail", "msg": "skuId '...' not found"}` when SKU doesn't exist. Also fixed earlier: ORM cascade delete (24 orphans → 0).

---

### Issue #8 — Annotated images never cleaned up from disk *(Fixed)*

- **Severity**: High
- **Location**: `api/routes/recognition.py:49-62`, `api/services/image_storage.py`, `api/app.py`
- **Category**: Resource leak

**Description**: The detect endpoint generates annotated images and uploads to Qiniu, but the local copies in `results/annotated/` are never cleaned up. Over time, disk space is exhausted.

**Trigger**: Every successful recognition call.

**Impact**: Disk space exhaustion on long-running servers.

**Fix applied**: Added `ImageStorage.cleanup_old_results(max_age_hours)` method. Background asyncio task runs every hour, deleting annotated images older than `RESULTS_MAX_AGE_HOURS` (default 24h). Configurable via env var.

---

### Issue #7 — sku_distribution leaks full probability distribution to API consumers *(Fixed)*

- **Severity**: High
- **Location**: `api/services/recognition.py:136`
- **Category**: Information leak / performance

**Description**: Each detection item in the response includes `"sku_distribution": distribution` which is the **full softmax probability distribution over all enabled SKUs**. If there are 10,000 SKUs, each detection item carries a 10,000-entry dict. For an image with 20 detections, that's 200,000 key-value pairs in the JSON response. This also leaks information about all SKUs in the system (their IDs and relative scores).

**Trigger**: Any recognition call with multiple SKUs in the index.

**Impact**: Bloated response sizes, potential information disclosure of all SKU IDs, performance degradation.

**Fix applied**: `sku_distribution` now returns only the top 5 entries by probability, instead of the full distribution.

---

### Issue #1 — Inconsistent API response format across endpoints *(Ignored — intentional)*

- **Severity**: Critical
- **Location**: `api/routes/goods.py` (all handlers), `api/routes/recognition.py:24,78,93,96`, `api/routes/system.py:19-28`, `api/routes/logs.py:18-22`
- **Category**: API contract violation

**Description**: The project defines `ApiResponse(code=1, data=..., msg="success")` as the standard response envelope. However, almost no endpoints actually use it consistently:

- `goods.py` returns bare dicts like `{"status": "fail", "msg": ...}` — **no `code` field, no `data` wrapper**.
- `recognition.py:detect` returns `ApiResponse(code=0, msg=...)` for errors but `ApiResponse(data=result)` for success — inconsistent `code` (0 vs 1).
- `recognition.py:fix` returns `{"status": "success"}` / `{"status": "fail"}` — completely different format from `detect`.
- `system.py` returns raw fields without any envelope.
- `logs.py` returns raw fields without any envelope.
- Only `list_skus` wraps in `ApiResponse(data=...)` and `detect` success wraps in `ApiResponse(data=...)`.

**Trigger**: Any API call.

**Impact**: Clients cannot parse responses uniformly. Some responses lack `code` entirely, making error detection impossible without parsing differently per endpoint.

---

### Issue #2 — GPU operations in goods routes bypass inference_executor *(Fixed)*

- **Severity**: Critical
- **Category**: Race condition / GPU concurrency
- **Location**: `api/routes/goods.py:101,116,136,207,225` vs `api/routes/recognition.py:39`

**Description**: Recognition correctly uses the single-threaded `inference_executor` for GPU work. But `goods.py` calls `asyncio.to_thread()` for `processor.update_sku_name`, `processor.delete_sku_references`, `processor.set_sku_enabled`, `processor.process_and_add`, and `processor.delete_media_reference` — these all run on the **default** `ThreadPoolExecutor`, not the dedicated `inference_executor`.

This means a `/sku/media` add (which calls `process_and_add` → YOLOE detection + DINOv2 embedding) can run **concurrently** with a `/detect` call that also uses YOLOE + DINOv2 on the GPU. Concurrent GPU access without serialization.

**Trigger**: Concurrent `POST /api/v1/goods/sku/media` (add action) and `POST /api/v1/recognition/detect` requests.

**Impact**: GPU memory corruption, CUDA errors, OOM crashes, or silent wrong inference results under concurrent load.

**Fix applied**: `manage_media add` path now uses `loop.run_in_executor(executor, processor.process_and_add, ...)` where `executor = req.app.state.inference_executor` (goods.py:201-212). Remaining `asyncio.to_thread` calls are for Chroma metadata ops only — no GPU involvement.

---

### Issue #3 — Embedding task runs process_and_add outside inference_executor *(Fixed)*

- **Severity**: Critical
- **Category**: Race condition / GPU concurrency
- **Location**: `api/tasks.py:39-45`

**Description**: `start_embed_task` creates an `asyncio.Task` that calls `asyncio.to_thread(processor.process_and_add, ...)` which runs on the default executor's threads, not the dedicated `inference_executor`. During a batch `sku/new` with multiple images, each `process_and_add` call runs YOLOE detection + DINOv2 embedding. If a `/detect` request comes in simultaneously, both compete for the GPU.

**Trigger**: `POST /api/v1/goods/sku/new` followed quickly by `POST /api/v1/recognition/detect`.

**Impact**: Same as #2 — GPU race condition, potential crashes or corrupted results.

**Fix applied**: `api/tasks.py` now accepts `inference_executor` param (line 23), uses `loop.run_in_executor(inference_executor, processor.process_and_add, ...)` (line 41-42). Also fixed `await` on sync `cleanup_download()` → sync call (line 49).

---

### Issue #9 — manage_media add never cleans up downloaded temp files *(Fixed)*

- **Severity**: High
- **Location**: `api/services/image_storage.py:38-44` and `api/routes/goods.py:203-213`
- **Category**: Resource leak

**Description**: For each media item in the add loop, the code downloads the image, calls `process_and_add`, but never calls `cleanup_download`. The downloaded temp file in `results/downloads/` is left behind for every media add operation — both on success and failure.

**Trigger**: `POST /api/v1/goods/sku/media` with action=add.

**Impact**: Temp files accumulate in `results/downloads/` for every media add operation.

**Fix applied**: Added `finally` block (goods.py:233-235) with `cleanup_download(downloaded_path)` — runs on both success and failure paths.

---

### Issue #12 — manage_media add commits SKUMedia row before embedding — orphan row on failure *(Fixed)*

- **Severity**: Medium
- **Location**: `api/routes/goods.py:203-213`
- **Category**: Data integrity / DB-Chroma drift

**Description**: For each media item in the add loop, the code: (1) inserts SKUMedia row and commits, (2) downloads the image, (3) calls `process_and_add`. If step 2 or 3 fails, the SKUMedia row exists in the DB but has no corresponding embedding in Chroma. The function does NOT delete the orphan SKUMedia row on failure. If ALL images are skipped, it returns `{"status": "fail", "skipped_images": [...]}` but the SKUMedia rows are already committed.

**Trigger**: Image download failure or detection failure for all media in an add request.

**Impact**: DB has media records pointing to images that were never embedded. Listing SKU media shows images that don't contribute to matching.

**Fix applied**: SKUMedia row is now only committed after successful embedding (goods.py:220-227). Embedding runs via `inference_executor` (goods.py:212-219). On failure, no row is created — no orphan possible.
