# API Change 0904

## 1. SKU media 添加改为异步训练任务

`POST /api/v1/goods/sku/media`（action=add）由同步 embedding 改为异步 job，与 `/goods/sku/new` 行为一致：

- 请求立即返回 `data = {"trainJobId": "...", "mediaIds": [...]}`，mediaIds 与请求顺序一致。
- 可选字段 `trainJobId`：panel 提供的 job id，用于轮询；与已有 job 重复返回 409；缺省时 server 生成 `job_{skuId}_{unix_ts}`。
- 进度与失败信息改由 `GET /api/v1/system/train-status/get` 轮询获取：失败的图片在 `embeddingFailed` 中列出，并在 SKU media 列表中标记 `failed`（原 HTTP 200 + code=0 + data.embeddingFailed 的同步失败响应移除）。
- 所有 item 均携带 mediaId（无可 embed 项）时不创建 job，`data` 为 `{}`。

## 2. trainStatus 全量重算（add 与 delete 后）

`SKU.trainStatus` 改为基于该 SKU 全部 media 行重算：任一 media failed → `failed`；无任何 media → `pending`；否则 `completed`。

- media add 的 job 结束时重算（不再只看本批次结果）。
- media delete 后重算：删除最后一张失败图片时 `failed` → `completed`；全部删空后为 `pending`。
- 历史 stale 数据由 `scripts/recompute_train_status.py` 一次性修正（幂等，支持 --dry-run）。
