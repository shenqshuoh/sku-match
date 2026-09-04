# SKU Match — API Guide

> [中文](API_GUIDE_zh.md) | **English**

## 1. Authentication & Rate Limiting

### API Key

All `/api/v1/*` endpoints require the `X-API-Key` header when `API_KEY` is set:

```bash
curl -H "X-API-Key: your-secret-api-key-here" \
     http://localhost:8000/api/v1/goods/sku/list
```

Missing or wrong key returns `401 Unauthorized`.

When `API_KEY` is empty (the default), authentication is disabled and all endpoints are accessible without a key.

### Rate Limiting

When `RATE_LIMIT` is set to a positive integer (requests per minute per IP), clients exceeding the limit receive `429 Too Many Requests`. Disabled by default (`RATE_LIMIT=0`).

### Public Endpoints

These are never behind auth or rate limiting:

- `GET /health` — health check
- `GET /results/*` — static annotated result images
- `GET /opt/SKUDB/*` — static reference images (when `REFERENCE_DIR` exists)

### CORS

All origins are allowed (`Access-Control-Allow-Origin: *`). Allowed headers: `X-API-Key`, `Content-Type`. Allowed methods: `GET`, `POST`, `OPTIONS`. Preflight `OPTIONS` requests are handled automatically by the middleware.

## 2. Response Format

All endpoints return the `ApiResponse` envelope:

**Success (HTTP 200):**

```json
{"code": 1, "data": { ... }, "msg": "success"}
```

**Error (HTTP 4xx/5xx):**

```json
{"code": 0, "data": null, "msg": "error description"}
```

The HTTP status code indicates the error type:

| HTTP Status | Meaning | Example |
|---|---|---|
| 400 | Bad request (invalid input) | `Unsupported action: '...'` |
| 404 | Resource not found | `skuId '...' not found` |
| 409 | Conflict (duplicate resource) | `taskId '...' already exists` |
| 422 | Validation error (Pydantic) | Malformed request body |
| 500 | Internal server error | Unexpected exception |
| 503 | Service unavailable | No SKUs indexed yet |

All response field names use **camelCase** (e.g. `matchedImage`, `skuId`, `classId`).

## 3. Quick Start

### List SKUs

```bash
curl "http://localhost:8000/api/v1/goods/sku/list?page=1&size=20"
# {"code":1,"data":{"list":[],"page":1,"pageSize":20,"total":0},"msg":"success"}
```

Optional `keyword` query param searches across `skuId` and `skuName` (case-insensitive):

```bash
curl "http://localhost:8000/api/v1/goods/sku/list?page=1&size=20&keyword=cola"
```

### Add a SKU

```bash
curl -X POST http://localhost:8000/api/v1/goods/sku/new \
  -H "Content-Type: application/json" \
  -d '{
    "skuId": "test-cola-001",
    "skuName": "Coca-Cola 330ml Can",
    "files": ["https://example.com/cola-front.jpg", "https://example.com/cola-side.jpg"],
    "trainJobId": "job-001"
  }'
```

Embedding runs asynchronously. Poll `/system/train-status/get` for completion.

### Detect SKUs in an image

```bash
curl -X POST http://localhost:8000/api/v1/recognition/detect \
  -H "Content-Type: application/json" \
  -d '{
    "taskId": "task-001",
    "mode": "IMAGE",
    "files": "https://example.com/fridge-photo.jpg"
  }'
```

## 4. Endpoint Summary

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check (public) |
| `POST` | `/api/v1/recognition/detect` | Detect & match SKUs in an image |
| `POST` | `/api/v1/recognition/fix` | Submit correction for a detection (reassign / remove / adjust-roi / add) |
| `POST` | `/api/v1/goods/sku/new` | Create SKU + start embedding |
| `POST` | `/api/v1/goods/sku/update` | Update SKU name |
| `POST` | `/api/v1/goods/sku/delete` | Delete SKU + index data |
| `POST` | `/api/v1/goods/sku/enable` | Enable/disable SKU |
| `GET` | `/api/v1/goods/sku/list` | Paginated SKU list with optional keyword search |
| `POST` | `/api/v1/goods/sku/media` | Add/delete SKU media |
| `GET` | `/api/v1/logs/recognition/get` | Query recognition log (effective + optional original result) |
| `GET` | `/api/v1/logs/recognition/list` | Paginated recognition log list (time / correction-status filters) |
| `POST` | `/api/v1/logs/recognition/status` | Manually set correction status |
| `POST` | `/api/v1/logs/recognition/delete` | Batch-delete recognition logs |
| `GET` | `/api/v1/system/train-status/get` | Query embedding job status |

---

## 5. Recognition

### POST `/api/v1/recognition/detect`

Detect beverage containers in an image and match each against the indexed SKU catalogue.

**Request:**

| Field | Type | Required | Default | Notes |
|-------|------|----------|---------|-------|
| `taskId` | string | yes | — | Unique per request. Duplicate taskIds are rejected (409). |
| `mode` | `"IMAGE"` \| `"VIDEO"` | no | `"IMAGE"` | `VIDEO` is accepted by the schema but currently processed the same as `IMAGE`. |
| `files` | string | yes | — | A single image URL or local path. |
| `roiRect` | `number[4]` | no | `null` | Region of interest `[x1, y1, x2, y2]`. Detections whose center falls outside are filtered out. Must be exactly 4 elements if provided. |

```json
{
  "taskId": "task-001",
  "mode": "IMAGE",
  "files": "https://example.com/fridge-photo.jpg",
  "roiRect": [0, 0, 1920, 1080]
}
```

**Response `data` (`DetectData`):**

| Field | Type | Description |
|-------|------|-------------|
| `taskId` | string | Echoed back task ID. |
| `matchedImage` | string | Annotated image — a Qiniu CDN URL on success, or a local `/results/` path if upload failed. When upload fails, `qiniuUploadFailed: true` is also present. |
| `counts` | `object` | Map of `skuId → detection count` (only includes matched SKUs above threshold). |
| `detections` | `array` | One entry per detected object (`DetectionItem[]`). See below. |

**Each detection item (`DetectionItem`):**

| Field | Type | Description |
|-------|------|-------------|
| `itemId` | int | Sequential index (1-based). |
| `bbox` | `number[4]` | Bounding box `[x1, y1, x2, y2]` in image pixel coordinates. |
| `classId` | int | YOLOE class ID. |
| `className` | string | Class label (e.g. `"bottle"`, `"canned"`). |
| `detectionConf` | float | YOLOE detection confidence (0–1). |
| `skuId` | string | Matched SKU ID. Empty string if below `MATCH_CONF` threshold. |
| `skuName` | string | Matched SKU name. Empty if below threshold. |
| `matchScore` | float | Match confidence (softmax probability, 0–1). `0.0` if below threshold. |
| `matchConcentration` | float | **Deprecated** — redundant with `matchScore`. Still returned for backwards compat; will be removed in a future version. |
| `skuDistribution` | object \| null | Top N candidates as `{skuId: {skuName, score}}`, sorted descending. N is configurable via `DISTRIBUTION_TOP_K` (default 10). Empty `{}` if below threshold. The actual number returned may be fewer than N: when patch re-ranking is enabled, the distribution is built from the `RERANK_TOP_K` candidate pool (default 50 vectors), so the number of unique SKUs depends on vectors-per-SKU (e.g., 50 vectors ÷ ~10 images/SKU ≈ 5 SKUs). Increase `RERANK_TOP_K` to surface more candidates. |
| `matchedVectorTags` | array \| null | Up to 20 reference vectors: `[{skuId, score, mediaUrl}]`. Empty `[]` if below threshold. |
| `source` | string | Entry origin: `"model"` (auto-detected) or `"manual"` (added via fix `add`). Manual entries carry `null` for all model-output fields above. Entries in logs created before this flag existed were backfilled to `"model"`. |

**Example response:**

```json
{
  "code": 1,
  "data": {
    "taskId": "task-001",
    "matchedImage": "https://vr.jihaihotpot.com/sku-match/2026-08/12/task-001.jpg",
    "counts": {"100001_1664": 2},
    "detections": [
      {
        "itemId": 1,
        "bbox": [120.0, 80.0, 340.0, 520.0],
        "classId": 0,
        "className": "bottle",
        "detectionConf": 0.92,
        "skuId": "100001_1664",
        "skuName": "1664",
        "matchScore": 0.87,
        "matchConcentration": 0.73,
        "skuDistribution": {
          "100001_1664": {"skuName": "1664", "score": 0.87},
          "100003_ws": {"skuName": "乌苏", "score": 0.08},
          "100010_ywcm": {"skuName": "怡泉柠檬味", "score": 0.03}
        },
        "matchedVectorTags": [
          {"skuId": "100001_1664", "score": 0.891234, "mediaUrl": "https://..."}
        ]
      }
    ]
  },
  "msg": "success"
}
```

**Error responses:**

| HTTP | Condition |
|------|-----------|
| 409 | Duplicate `taskId` |
| 503 | Index empty (no SKUs indexed) |
| 500 | Any other exception |

---

### POST `/api/v1/recognition/fix`

Submit human corrections for a previous detection. Corrections are applied to a **corrected copy** of the result (`finalResult`); the original AI output (`aiResult`) is immutable and preserved for fine-tuning. It does not re-run detection.

Submitting a fix sets the log's `correctionStatus` to `corrected`, increments `correctionCount`, and stamps `correctedAt`. Fixes are **incremental**: each submission is applied on top of the current `finalResult` — previous fixes are kept. `userCorrection` accumulates every submitted fix item in order (full audit trail).

**Request:**

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `taskId` | string | yes | Must reference an existing detection task. |
| `fixItems` | array (min 1) | yes | List of corrections. |

**Each fix item:**

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `fixType` | `"reassign"` \| `"remove"` \| `"adjust-roi"` \| `"add"` | yes | **Breaking change**: free-form values (e.g. `"misidentification"`) are now rejected (422). Use `"reassign"`. |
| `itemId` | int \| null | yes (except `add`) | Which detection item this correction applies to. Must exist in the **current** result (400 otherwise) — items removed by an earlier fix can no longer be referenced. Ignored for `add`. |
| `roiRect` | `number[4]` \| null | required for `adjust-roi`, `add` | Corrected bounding box `[x1, y1, x2, y2]` (exactly 4 elements). For `add`: the manually drawn box of the new detection. |
| `skuId` | string \| null | required for `reassign`, `add` | Corrected SKU ID. Not validated against the live SKU table (historical corrections may reference deleted SKUs). For `add`: the SKU assigned to the new detection. |

**Fix semantics:**

- `remove` — drops the detection from `finalResult` (reduces counts). The removed item remains traceable via `userCorrection` and is still present in `originalResult`.
- `reassign` — replaces `skuId`/`skuName` in `finalResult`; counts are recomputed.
- `adjust-roi` — replaces `bbox` in `finalResult`.
- `add` — creates a **new, manually added detection** (model miss): gets the next free `itemId`, uses `roiRect` as its `bbox`, resolves `skuName` from the index. Model-output fields are `null` (`detectionConf`, `classId`, `className`, `matchScore`, `skuDistribution`, `matchedVectorTags`) and the entry carries `"source": "manual"` (model entries carry `"source": "model"`) — clients should null-check these fields; fine-tuning pipelines can filter on `source`.
- `counts` in `finalResult` is always recomputed from the remaining detections.

```json
{
  "taskId": "task-001",
  "fixItems": [
    {"fixType": "remove", "itemId": 2},
    {"fixType": "reassign", "itemId": 1, "skuId": "100003_ws"},
    {"fixType": "adjust-roi", "itemId": 3, "roiRect": [10, 20, 30, 40]},
    {"fixType": "add", "roiRect": [50, 60, 120, 300], "skuId": "100001_1664"}
  ]
}
```

**Annotated image regeneration & pre-fix snapshot:** after each fix, the annotated image is regenerated from the corrected result (best-effort — the JSON result stays authoritative). On the **first** fix, the pre-fix annotation is also snapshotted and uploaded to a distinct Qiniu key; `originalImageUrl` (returned with `includeOriginal=true`) always shows the pre-fix image. The regenerated image is uploaded to a **fresh versioned key per fix** (`{taskId}_annotated_v2.jpg`, `_v3`, ... — the upload tokens are insert-only and reject overwrites), so **`matchedImage` / `visualImageUrl` change with every fix** — always read the latest URL from the fix response, `GET /logs/recognition/get`, or the list endpoint; do not cache it client-side. The source image is persisted locally at detect time for regeneration (`inputs/`, 72h retention); if purged, the server re-downloads from the request source, and if that fails the previous image remains.

**Response:** `data = {"matchedImage": "<url>"}` — the URL of the regenerated annotated image (fresh versioned CDN URL, or an absolute local URL when the upload failed). `data` is `{}` if regeneration was skipped (e.g. input image unavailable).

**Error responses:**

| HTTP | Condition |
|------|-----------|
| 400 | `itemId` not found in the current result (e.g. already removed by an earlier fix), or the log has no valid AI result (failed detection) |
| 404 | Unknown `taskId` |
| 500 | Internal error |

---

## 6. Goods (SKU Management)

### POST `/api/v1/goods/sku/new`

Create a SKU record, insert media rows, and start an asynchronous embedding task.

**Request:**

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `skuId` | string | yes | **Bare** SKU identifier (max 128) — no numeric prefix. The server always prepends an auto-incremented number (`100001_`, `100002_`, ...): submitting `kkkl` when the current max is `100106_x` creates `100107_kkkl`. The submitted string is used verbatim as the suffix (never mutated). **Read the final id from the response** — don't assume what you submitted. Duplicate guard: same suffix **and** same name already exist → 409; same suffix with a different name is allowed. |
| `skuName` | string | yes | Display name (max 256). |
| `files` | string[] | yes | Reference image URLs/paths (1–50 items, non-empty strings). |
| `trainJobId` | string | yes | Unique job ID for tracking (max 128). |

**Response:** `data = {"skuId": "...", "trainJobId": "..."}`

**Error responses:**

| HTTP | Condition |
|------|-----------|
| 409 | Duplicate `skuId` or `trainJobId` |

---

### POST `/api/v1/goods/sku/update`

Update a SKU's display name in both the database and the Chroma index metadata.

**Request:**

| Field | Type | Required |
|-------|------|----------|
| `skuId` | string | yes |
| `skuName` | string | yes |

**Response:** `data` is an empty object `{}`.

**Error responses:** 404 — unknown `skuId`.

---

### POST `/api/v1/goods/sku/delete`

Delete a SKU and cascade-clean all related data: DB media rows, Chroma vectors, patch files, and color descriptor files.

**Request:** `{"skuId": "..."}`

**Response:** `data` is an empty object `{}`.

**Error responses:** 404 — unknown `skuId`.

---

### POST `/api/v1/goods/sku/enable`

Enable or disable a SKU. Disabled SKUs are excluded from matching queries.

**Request:**

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `skuId` | string | yes | |
| `enabled` | boolean | yes | `true` to enable, `false` to disable. |

**Response:** `data` is an empty object `{}`.

**Error responses:** 404 — unknown `skuId`.

---

### GET `/api/v1/goods/sku/list`

Paginated SKU listing with optional keyword search.

**Query parameters:**

| Param | Type | Default | Notes |
|-------|------|---------|-------|
| `page` | int | 1 | Page number (1-based). |
| `size` | int | 20 | Page size. |
| `keyword` | string | — | Searches `skuId` and `skuName` (ILIKE, case-insensitive). |
| `trainStatus` | string | — | Filter: `"pending"` / `"indexing"` / `"completed"` / `"failed"`. Invalid values → 400. |
| `enabled` | bool | — | Filter: `true` / `false`. Omitted → all SKUs. |

**Response `data`:**

| Field | Type | Description |
|-------|------|-------------|
| `list` | array | SKU items (see below). |
| `page` | int | Current page. |
| `pageSize` | int | Page size. |
| `total` | int | Total matching SKU count. |

**Each SKU list item:**

| Field | Type | Description |
|-------|------|-------------|
| `id` | int | Database row ID. |
| `skuId` | string | SKU identifier. |
| `skuName` | string | Display name. |
| `trainStatus` | string | Embedding status (see values below). |
| `medias` | array | `[{mediaId, mediaType, mediaUrl, failed}]` — `mediaUrl` is expanded to a full URL; `failed: true` marks a reference image that did not produce a crop/embedding (manual intervention needed — visible in the list, deletable, and safe to re-add). |

**`trainStatus` values:** `"pending"` → `"indexing"` → `"completed"` (or `"failed"` on error).

---

### POST `/api/v1/goods/sku/media`

Add or delete media for an existing SKU.

**Request:**

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `skuId` | string | yes | |
| `action` | `"add"` \| `"delete"` | yes | |
| `media` | array | yes | List of media items (see below). |
| `trainJobId` | string \| null | no | Add action only. Panel-supplied job id for polling; collides with an existing job → 409. Absent → server generates `job_{skuId}_{unix_ts}`. |

**Media item:**

| Field | Type | Required (add) | Required (delete) | Notes |
|-------|------|----------------|-------------------|-------|
| `mediaId` | string \| null | no | yes | Which media to delete. Items carrying a `mediaId` in an add request are skipped. |
| `mediaUrl` | string \| null | yes | no | Image URL/path to embed (add action). |
| `preCropped` | bool | no | — | Default `false`: server runs YOLOE crop + background masking. `true`: image is already a cropped reference — used as-is (EXIF-transposed), no detection step, so it cannot fail on "no detection". |

**Add behavior (async):** Media rows are inserted immediately and embedding runs in a background job (same pipeline as `/goods/sku/new`: download → crop unless `preCropped` → embed → Chroma index → patch/color caching → Qiniu crop upload). The response returns right away with a `trainJobId` — poll `GET /api/v1/system/train-status/get` for progress; images that fail to embed are listed there (`embeddingFailed`) and flagged as `failed` in the SKU media list. The SKU's `trainStatus` is recomputed from all its media when the job ends. If no item is embeddable (all carry a `mediaId`), no job is created and `data` is `{}`.

**Delete behavior:** Removes the DB media row, Chroma vector, patch file, and color descriptor file for each `mediaId`, then recomputes the SKU's `trainStatus` (deleting the last failed image flips `failed` → `completed`; deleting all media leaves `pending`).

**Response (add):** `data = {"trainJobId": "...", "mediaIds": ["...", ...]}` — the assigned media ids in request order. Delete: `data = {}`.

**Error responses:**

| HTTP | Condition |
|------|-----------|
| 400 | Missing `mediaUrl` (add), or unsupported `action` |
| 404 | Unknown `skuId` (add) |
| 409 | Supplied `trainJobId` already exists |

---

## 7. Logs

### GET `/api/v1/logs/recognition/get`

Retrieve the AI result (corrected view if available), correction state, and human corrections for a detection task.

**Query parameters:**

| Param | Type | Default | Notes |
|-------|------|---------|-------|
| `taskId` | string | — (required) | |
| `includeOriginal` | bool | `false` | Also return `originalResult` (the immutable AI output — fine-tuning data). |

**Response `data`:**

| Field | Type | Description |
|-------|------|-------------|
| `aiResult` | object \| null | **The effective result**: `finalResult` (corrected view) when a fix has been submitted, otherwise the original AI output. Same field name as before — no client change needed. |
| `originalResult` | object \| null | Immutable original AI output. Present only when `includeOriginal=true`. |
| `originalImageUrl` | string \| null | Pre-fix annotation image (CDN URL or local path). Present only when `includeOriginal=true`; null when it could not be preserved (e.g. first fix after local retention expired and source unavailable). |
| `correctionStatus` | string | `"pending"` / `"corrected"` / `"reviewed"`. |
| `correctionCount` | int | Number of fix submissions. |
| `correctedAt` | string \| null | Timestamp of the latest fix (ISO 8601). |
| `detectionsAdded` | int | Fix-added detections still present in the current result vs the original. |
| `detectionsRemoved` | int | Original detections no longer present in the current result. |
| `skuMismatchCount` | int | Detections reassigned by the latest fix (vs original). |
| `inputImageUrl` | string \| null | Unannotated input image URL (absolute). |
| `userCorrection` | array \| null | All correction items submitted via `/recognition/fix`, accumulated in submission order — including `remove` entries, so removed detections stay traceable. |
| `visualImageUrl` | string \| null | Final annotated image URL. **Always absolute** — CDN URL when uploaded, otherwise `http://<host>:<port>/results/annotated/...` derived from the request's base URL. |

**Error responses:** 404 — unknown `taskId`.

---

### GET `/api/v1/logs/recognition/list`

Paginated recognition-log listing, filterable by time range and correction status. Ordered by `createdAt` descending.

**Query parameters:**

| Param | Type | Default | Notes |
|-------|------|---------|-------|
| `page` | int | 1 | Page number (1-based). |
| `size` | int | 20 | Page size. |
| `startTime` | string | — | ISO 8601; filters `createdAt >= startTime`. Aware timestamps are converted to server-local time. |
| `endTime` | string | — | ISO 8601; filters `createdAt <= endTime`. |
| `correctionStatus` | string | — | `"pending"` / `"corrected"` / `"reviewed"`. |

**Response `data`:**

| Field | Type | Description |
|-------|------|-------------|
| `list` | array | Log items (see below). |
| `page` / `pageSize` / `total` | int | Pagination envelope. |

**Each log item:**

| Field | Type | Description |
|-------|------|-------------|
| `taskId` | string | |
| `createdAt` | string \| null | ISO 8601. |
| `correctionStatus` | string | `"pending"` / `"corrected"` / `"reviewed"`. |
| `correctedAt` | string \| null | Timestamp of the latest fix. |
| `correctionCount` | int | Fix submissions so far. |
| `detectionCount` | int | Detections in the original result (0 for pre-migration / failed logs). |
| `detectionsAdded` | int | Fix-added detections still present in the current result vs the original. Cumulative across all fixes. |
| `detectionsRemoved` | int | Original detections no longer present in the current result. Cumulative across all fixes. |
| `skuMismatchCount` | int | Detections whose `skuId` differs between the latest fix and the original (reassignments). |
| `inputImageUrl` | string \| null | Unannotated input image URL (absolute — CDN, or local `/results/inputs/...` expanded with the request host when the upload failed). |
| `visualImageUrl` | string \| null | Final annotated image URL (absolute). |

**Error responses:**

| HTTP | Condition |
|------|-----------|
| 400 | Invalid `startTime`/`endTime` format or invalid `correctionStatus` |

---

### POST `/api/v1/logs/recognition/status`

Manually change a log's correction status — e.g. mark as `"reviewed"` (checked, approved as-is) or reset to `"pending"` to re-open it. Only the workflow state changes: `correctionCount`, `correctedAt`, and `finalResult` (which follow actual fix submissions) are untouched.

**Request:**

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `taskId` | string | yes | |
| `correctionStatus` | `"pending"` \| `"corrected"` \| `"reviewed"` | yes | |

**Response:** `data` is an empty object `{}`.

**Error responses:** 404 — unknown `taskId`.

---

### POST `/api/v1/logs/recognition/delete`

Batch-delete recognition logs. Removes DB rows and local files (annotated, inputs, pre-fix snapshots — best-effort). CDN objects are currently left orphaned (no Qiniu delete credentials configured); deletion will be added when credentials are available.

**Request:**

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `taskIds` | string[] (1–100) | yes | Task IDs to delete. |

**Response:** `data = {"deleted": <int>, "notFound": ["taskId", ...]}`

```json
{"code": 1, "data": {"deleted": 2, "notFound": ["gone_id"]}, "msg": "success"}
```

---

## 8. System

### GET `/api/v1/system/train-status/get`

Check the status of an asynchronous embedding job.

**Query parameter:** `trainJobId` (string, required)

**Response `data`:**

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Job lifecycle: `"pending"` → `"indexing"` → `"completed"` (or `"failed"` on exception). |
| `progress` | int | 0–100 percentage. |
| `estimatedTime` | string \| null | Reserved; always `null` currently. |
| `failedCount` | int | Number of images that failed to embed. |
| `totalCount` | int | Total number of images in the job. |
| `embeddingFailed` | array \| null | Present only when some images failed: list of failed media URLs. Applies to jobs from both `/goods/sku/new` and `/goods/sku/media` (add). |

**Example:**

```json
{
  "code": 1,
  "data": {
    "status": "completed",
    "progress": 100,
    "estimatedTime": null,
    "failedCount": 0,
    "totalCount": 5
  },
  "msg": "success"
}
```

**Error responses:** 404 — unknown `trainJobId`.
