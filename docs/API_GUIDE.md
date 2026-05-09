# SKU Match — API Guide

## 1. API Authentication

### When `API_KEY` is set

All `/api/v1/*` endpoints require the `X-API-Key` header:

```bash
curl -H "X-API-Key: your-secret-api-key-here" \
     http://localhost:8000/api/v1/goods/sku/list
```

Missing or wrong key returns `401 Unauthorized`.

### When `API_KEY` is empty (default)

Authentication is disabled. All endpoints are accessible without a key.

### Unauthenticated endpoints

- `GET /health` — always public
- `GET /results/*` — static file serving (annotated images), always public

## 2. Quick Test

### List SKUs (empty by default)

```bash
curl "http://localhost:8000/api/v1/goods/sku/list?page=1&size=20"
# {"code":1,"data":{"list":[],"page":1,"pageSize":20,"total":0},"msg":"success"}
```

### Add a new SKU with reference images

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

### Check training status

```bash
curl "http://localhost:8000/api/v1/system/train-status/get?trainJobId=job-001"
# {"code":1,"data":{"status":"completed","progress":100,"estimated_time":null},"msg":"success"}
```

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

## 3. API Endpoint Summary

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check (public) |
| `POST` | `/api/v1/recognition/detect` | Detect & match SKUs in image |
| `POST` | `/api/v1/recognition/fix` | Submit correction for a detection |
| `POST` | `/api/v1/goods/sku/new` | Create SKU + start embedding |
| `POST` | `/api/v1/goods/sku/update` | Update SKU name |
| `POST` | `/api/v1/goods/sku/delete` | Delete SKU + index data |
| `POST` | `/api/v1/goods/sku/enable` | Enable/disable SKU |
| `GET` | `/api/v1/goods/sku/list` | Paginated SKU list |
| `POST` | `/api/v1/goods/sku/media` | Add/delete SKU media |
| `GET` | `/api/v1/logs/recognition/get` | Query recognition log |
| `GET` | `/api/v1/system/train-status/get` | Query training job status |

## 4. Response Format

All endpoints return `ApiResponse` format:

**Success:**
```json
{"code": 1, "data": { ... }, "msg": "success"}
```

**Error:**
```json
{"code": 0, "data": null, "msg": "error description"}
```
