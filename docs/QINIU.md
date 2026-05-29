# Qiniu CDN Routing Workaround

## Problem

CDN domain `vr.jihaihotpot.com` resolves to overseas CDN nodes (`all.ov.qnyglobal.com` → `154.41.93.240/241`), giving ~8 KB/s download speed from the China-based API server.

## Workaround

`QINIU_IOVIP_URL` in `api/config.py` routes downloads through the Qiniu origin storage domain (`iovip-z2.qiniuio.com`) instead of the CDN domain. This resolves to domestic China IPs (125.94.x / 183.60.x) and gives ~16 MB/s.

The rewrite happens in `ImageStorage.download_image()`:
- URLs starting with `QINIU_DOMAIN` are rewritten to `QINIU_IOVIP_URL/<key>` with a `Host` header
- All other URLs download as-is

## Revert This When...

The CDN domain routing is fixed to resolve to domestic China nodes. Steps:

1. In Qiniu Console → CDN → Domain Management → `vr.jihaihotpot.com`, change coverage area to **中国大陆**
2. Verify: `dig vr.jihaihotpot.com` should resolve to domestic IPs (119.x / 183.x / 14.x), NOT `154.41.x.x`
3. Remove `QINIU_IOVIP_URL` from `api/config.py`
4. Remove the `qiniu_iovip_url` parameter from `ImageStorage.__init__()` in `api/services/image_storage.py`
5. Remove the URL rewrite logic in `download_image()` — revert to simple `client.get(url)`
6. Remove `qiniu_iovip_url` from the `ImageStorage(...)` constructor call in `api/app.py`

## Config

| Setting | Value | Purpose |
|---------|-------|---------|
| `QINIU_IOVIP_URL` | `http://iovip-z2.qiniuio.com` | Origin storage domain for z2 (华南) region |
| `QINIU_DOMAIN` | `https://vr.jihaihotpot.com/` | CDN domain (used for upload results and Host header) |

## Fallback Behavior

When Qiniu upload fails, the API returns the local static path (`/results/annotated/{taskId}_annotated.jpg`) as `matched_image` with a `qiniu_upload_failed: true` flag in the response. Clients can check this flag to know the URL is a local fallback that may not be accessible from outside the server network.

## Reference

- Bucket: `visual-recognition` (z2 / 华南-广东)
- Upload endpoint: `upload-z2.qiniup.com`
- Origin endpoint: `iovip-z2.qiniuio.com`
- Token service: `http://172.16.88.119:12001/api/qiniu/token/vr`
- Server: Alibaba Cloud ECS (110.76.42.124) — no Qiniu internal network access
