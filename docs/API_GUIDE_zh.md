# API 接口文档

## 1. Authentication & Rate Limiting

### API Key

```bash
curl -H "X-API-Key: your-secret-api-key-here" \
     http://localhost:8000/api/v1/goods/sku/list
```

当 `API_KEY` 为空时无需 key 即可访问。

## 2. Response 格式

**成功（HTTP 200）：**

```json
{"code": 1, "data": { ... }, "msg": "success"}
```

**错误（HTTP 4xx/5xx）：**

```json
{"code": 0, "data": null, "msg": "skuId '...' not found"}
```

HTTP status code 表示错误类型：

| HTTP Status | 含义 | msg 示例 |
|---|---|---|
| 400 | Request 无效 | `Unsupported action: '...'` |
| 404 | Resource 不存在 | `skuId '...' not found` |
| 409 | 冲突（resource 重复） | `taskId '...' already exists` |
| 422 | Validation error（Pydantic） | Request body 格式错误 |
| 500 | 服务器内部错误 | 未预期的异常 |
| 503 | Service unavailable | `No SKUs indexed yet — add reference images first` |

> 例外：`/goods/sku/media` 的部分失败返回 HTTP 200 + `code=0`， `data.embeddingFailed` 列出失败的 URL（部分图片可能已成功处理）。需要人工检查图片考虑是否重新拍摄。

## 3. 查询/新增/检测

### 查询 SKU 列表

```bash
curl "http://localhost:8000/api/v1/goods/sku/list?page=1&size=20"
# {"code":1,"data":{"list":[],"page":1,"pageSize":20,"total":0},"msg":"success"}
```

可选 `keyword` query parameter，按 `skuId` 和 `skuName` 模糊搜索（不区分大小写）：

```bash
curl "http://localhost:8000/api/v1/goods/sku/list?page=1&size=20&keyword=cola"
```

### 新增 SKU

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

上传后触发 Embedding，异步执行。`/system/train-status/get` 查询完成状态。

### 图片检测识别

```bash
curl -X POST http://localhost:8000/api/v1/recognition/detect \
  -H "Content-Type: application/json" \
  -d '{
    "taskId": "task-001",
    "mode": "IMAGE",
    "files": "https://example.com/fridge-photo.jpg"
  }'
```

## 4. Endpoint

| Method | Path | 说明 |
|--------|------|------|
| `GET` | `/health` | |
| `POST` | `/api/v1/recognition/detect` | 检测图片中的饮料容器并匹配 SKU |
| `POST` | `/api/v1/recognition/fix` | 提交检测结果的人工 correction（reassign / remove / adjust-roi） |
| `POST` | `/api/v1/goods/sku/new` | 创建 SKU 并启动 embedding 任务 |
| `POST` | `/api/v1/goods/sku/update` | 更新 SKU 名称 |
| `POST` | `/api/v1/goods/sku/delete` | 删除 SKU 及 index 数据 |
| `POST` | `/api/v1/goods/sku/enable` | 启用/禁用 SKU |
| `GET` | `/api/v1/goods/sku/list` | 分页查询 SKU 列表（支持关键词搜索） |
| `POST` | `/api/v1/goods/sku/media` | 添加/删除 SKU media |
| `GET` | `/api/v1/logs/recognition/get` | 查询 recognition log（有效结果 + 可选原始结果） |
| `GET` | `/api/v1/logs/recognition/list` | 分页查询 recognition log（时间 / correction status 过滤） |
| `POST` | `/api/v1/logs/recognition/status` | 手动设置 correction status |
| `POST` | `/api/v1/logs/recognition/delete` | 批量删除 recognition log |
| `GET` | `/api/v1/system/train-status/get` | 查询 embedding 任务状态 |

---

## 5. Recognition

### POST `/api/v1/recognition/detect`

检测图片中的饮料与 SKU 库进行匹配。

**Request：**

| Field | Type | Required | Default | 说明 |
|-------|------|----------|---------|------|
| `taskId` | string | yes | — | 拒绝重复的 taskId（409） |
| `mode` | `"IMAGE"` \| `"VIDEO"` | no | `"IMAGE"` | `VIDEO` 当前不可用 |
| `files` | string | yes | — | 单张图片的 URL 或本地路径。 |
| `roiRect` | `number[4]` | no | `null` | Region of interest `[x1, y1, x2, y2]`。忽略中心点落在区域外的检测目标 |

```json
{
  "taskId": "task-001",
  "mode": "IMAGE",
  "files": "https://example.com/fridge-photo.jpg",
  "roiRect": [0, 0, 1920, 1080]
}
```

**Response `data`（`DetectData`）：**

| Field | Type | 说明 |
|-------|------|------|
| `taskId` | string | |
| `matchedImage` | string | 检测结果图片链接。七牛云上传失败时为本地 `/results/` 路径，附带 `qiniuUploadFailed: true`。 |
| `counts` | `object` | skuId → 数量 |
| `detections` | `array` | 检测目标详情 |

**Detection item（`DetectionItem`）field：**

| Field | Type | 说明 |
|-------|------|------|
| `itemId` | int | 从 1 开始 |
| `bbox` | `number[4]` | Bounding box `[x1, y1, x2, y2]`，饮料像素坐标。 |
| `classId` | int | YOLOE class ID。 |
| `className` | string | YOLOE 内部 Class label（bottle/canned/...），可无视 |
| `detectionConf` | float | YOLOE detection
confidence（0–1），系统判定物品为饮料的置信度 |
| `skuId` | string | 匹配的 SKU ID。低于 `MATCH_CONF` threshold 时为空 |
| `skuName` | string | 匹配的 SKU 名称。低于 threshold 时为空 |
| `matchScore` | float | Match confidence（softmax 概率，0–1）。低于 threshold 时为 `0.0`。 |
| `skuDistribution` | object \| null | 匹配SKU至多前N名，格式为 `{skuId: {skuName, score}}`，按概率降序排列。N 通过 `DISTRIBUTION_TOP_K` 配置（默认 10）。具体数量受搜索策略和各SKU参考图片数量影响，当前配置下通常显示前5。实测中如果不够需要更改测试更大的搜索策略 |
| `matchedVectorTags` | array \| null | 最多 20 条 reference vector：`[{skuId, score, mediaUrl}]`，调试用，可无视 |

**Response**

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

**Error response：**

| HTTP | 触发条件 |
|------|----------|
| 409 | `taskId` 重复 |
| 503 | Index 为空（尚无 SKU） |
| 500 | 其他异常 |

---

### POST `/api/v1/recognition/fix`

对检测结果人工纠错。Correction 会应用到一份**修正副本**（`finalResult`）上；原始 AI 输出（`originalResult`）不可变，永久保留（可用于后续 fine-tuning）。不会重新执行 detection。

提交 fix 后，log 的 `correctionStatus` 自动置为 `corrected`，`correctionCount` 加一，`correctedAt` 记录时间戳。重复提交为 **latest-wins**：`finalResult` 总是从原始结果重新计算。

**Request：**

| Field | Type | Required | 说明 |
|-------|------|----------|------|
| `taskId` | string | yes | 已存在的检测任务 |
| `fixItems` | array（min 1） | yes | 纠错列表。 |

**`FixItem`：**

| Field | Type | Required | 说明 |
|-------|------|----------|------|
| `fixType` | `"reassign"` \| `"remove"` \| `"adjust-roi"` | yes | **Breaking change**：不再接受自定义值（如 `"misidentification"`，会被 422 拒绝），请改用 `"reassign"`。 |
| `itemId` | int | yes | 针对的 detection item 序号，必须存在于原始结果中（否则 400）。 |
| `roiRect` | `number[4]` \| null | `adjust-roi` 时必填 | 纠错后的 bounding box `[x1, y1, x2, y2]`（恰好 4 个元素）。 |
| `skuId` | string \| null | `reassign` 时必填 | 纠错后的 SKU ID。不校验是否存在于当前 SKU 表（历史纠错可能引用已删除的 SKU）。 |

**Fix 语义：**

- `remove` — 从 `finalResult` 中移除该 detection（counts 相应减少）。被移除的 item 仍可通过 `userCorrection` 追溯，且完整保留在 `originalResult` 中。
- `reassign` — 替换 `finalResult` 中的 `skuId`/`skuName`，counts 重新计算。
- `adjust-roi` — 替换 `finalResult` 中的 `bbox`。
- `finalResult` 的 `counts` 总是基于剩余 detection 重新计算。

```json
{
  "taskId": "task-001",
  "fixItems": [
    {"fixType": "remove", "itemId": 2},
    {"fixType": "reassign", "itemId": 1, "skuId": "100003_ws"},
    {"fixType": "adjust-roi", "itemId": 3, "roiRect": [10, 20, 30, 40]}
  ]
}
```

**Annotated image 重新生成与 pre-fix 快照：** 每次 fix 后会基于修正结果重新生成 annotated image（best-effort —— JSON 结果始终是权威数据）。**首次** fix 时，同时会把 fix 前的标注图快照上传到独立的七牛 key；`originalImageUrl`（随 `includeOriginal=true` 返回）始终展示 fix 前的图。重新生成的图**每次 fix 上传到新的版本化 key**（`{taskId}_annotated_v2.jpg`、`_v3`…… —— upload token 为 insert-only，不允许覆盖同 key），因此 **`matchedImage` / `visualImageUrl` 每次 fix 都会变化** —— 请始终从 fix 响应、`GET /logs/recognition/get` 或 list 接口读取最新 URL，客户端不要缓存。detect 时会把源图持久化到本地（`inputs/`，保留 72 小时）用于重新生成；若已被清理则回退重新下载请求源图，再失败则保留旧图。

**Response：** `data = {"matchedImage": "<url>"}` —— 重新生成的 annotated image URL（新的版本化 CDN URL；上传失败时为含 host 的完整本地 URL）。若跳过重新生成（如输入图不可用），`data` 为 `{}`。

**Error response：**

| HTTP | 触发条件 |
|------|----------|
| 400 | `itemId` 在原始结果中不存在，或该 log 无有效 AI 结果（detection 失败） |
| 404 | `taskId` 不存在 |
| 500 | 内部错误 |

---

## 6. Goods（SKU Management）

### POST `/api/v1/goods/sku/new`

创建 SKU 记录、插入 media 行，并启动 async embedding 任务。

**Request：**

| Field | Type | Required | 说明 |
|-------|------|----------|------|
| `skuId` | string | yes | **纯 bare** 标识（max 128）—— 不带数字前缀。服务器会自动追加递增编号（`100001_`、`100002_`……）：当前最大为 `100106_x` 时提交 `kkkl` 会创建 `100107_kkkl`。提交的字符串原样作为后缀（不做任何改动）。**请从 response 中读取最终 skuId** —— 不要假设与你提交的一致。重复保护：suffix 相同**且**名称也相同 → 409；suffix 相同但名称不同则允许创建。 |
| `skuName` | string | yes | 显示名称（max 256）。 |
| `files` | string[] | yes | Reference 图片 URL/路径列表（1–50 条，非空字符串）。 |
| `trainJobId` | string | yes | 唯一 job ID，用于 tracking（max 128）。 |

**Response：** `data = {"skuId": "...", "trainJobId": "..."}`

**Error response：**

| HTTP | 触发条件 |
|------|----------|
| 409 | `skuId` 或 `trainJobId` 重复 |

---

### POST `/api/v1/goods/sku/update`

更新 SKU 显示名称（同步更新 database 和 Chroma index metadata）。

**Request：**

| Field | Type | Required |
|-------|------|----------|
| `skuId` | string | yes |
| `skuName` | string | yes |

**Response：** `data` 为空对象 `{}`。

**Error response：** 404 — `skuId` 不存在。

---

### POST `/api/v1/goods/sku/delete`

删除 SKU 并 cascade 清理所有关联数据：database media 行、Chroma vector、patch 文件、color descriptor 文件。

**Request：** `{"skuId": "..."}`

**Response：** `data` 为空对象 `{}`。

**Error response：** 404 — `skuId` 不存在。

---

### POST `/api/v1/goods/sku/enable`

启用或禁用 SKU。被禁用的 SKU 不参与 matching query。

**Request：**

| Field | Type | Required | 说明 |
|-------|------|----------|------|
| `skuId` | string | yes | |
| `enabled` | boolean | yes | `true` 启用，`false` 禁用。 |

**Response：** `data` 为空对象 `{}`。

**Error response：** 404 — `skuId` 不存在。

---

### GET `/api/v1/goods/sku/list`

分页查询 SKU 列表，支持关键词搜索。

**Query parameter：**

| Param | Type | Default | 说明 |
|------|------|--------|------|
| `page` | int | 1 | 页码（从 1 开始）。 |
| `size` | int | 20 | 每页条数。 |
| `keyword` | string | — | 按 `skuId` 和 `skuName` 模糊搜索（ILIKE，不区分大小写）。 |
| `trainStatus` | string | — | 过滤：`"pending"` / `"indexing"` / `"completed"` / `"failed"`。非法值返回 400。 |
| `enabled` | bool | — | 过滤：`true` / `false`。不传则返回全部。 |

**Response `data`：**

| Field | Type | 说明 |
|-------|------|------|
| `list` | array | SKU 列表（详见下文）。 |
| `page` | int | 当前页码。 |
| `pageSize` | int | 每页条数。 |
| `total` | int | 符合条件的 SKU 总数。 |

**每条 SKU list item：**

| Field | Type | 说明 |
|-------|------|------|
| `id` | int | Database 行 ID。 |
| `skuId` | string | SKU 标识。 |
| `skuName` | string | 显示名称。 |
| `trainStatus` | string | Embedding 状态（见下方取值）。 |
| `medias` | array | `[{mediaId, mediaType, mediaUrl, failed}]` — `mediaUrl` 会展开为完整 URL；`failed: true` 表示该 reference 图未产出 crop/embedding（需人工干预 —— 在列表中可见、可删除、可重新添加）。 |

**`trainStatus`:** `"pending"` → `"indexing"` → `"completed"`（异常时为 `"failed"`）。

---

### POST `/api/v1/goods/sku/media`

为已有 SKU 添加或删除 media。

**Request：**

| Field | Type | Required | 说明 |
|-------|------|----------|------|
| `skuId` | string | yes | |
| `action` | `"add"` \| `"delete"` | yes | |
| `media` | array | yes | Media item 列表（详见下文）。 |

**Media item（`MediaItem`）：**

| Field | Type | add 时必填 | delete 时必填 | 说明 |
|-------|------|------------|------------|------|
| `mediaId` | string \| null | no | yes | 要删除的 media ID。 |
| `mediaUrl` | string \| null | yes | no | 要 embed 的图片 URL/路径（add 操作）。 |

**Add 行为：** 每张图片依次 download → YOLOE crop → embedding → 写入 Chroma index → 上传到七牛。成功后插入 media 行

**Delete 行为：** 删除 database media 行、Chroma vector、patch 文件、color descriptor 文件。

**Response：** 成功时 `data` 为空对象 `{}`。

**Error response：**

| HTTP | 触发条件 |
|------|----------|
| 400 | 缺少 `mediaUrl`（add），或 `action` 不支持 |
| 404 | `skuId` 不存在（add） |
| 200 + `code=0` | 部分图片 embedding 失败 — `data.embeddingFailed` 列出失败的 URL |

---

## 7. Log

### GET `/api/v1/logs/recognition/get`

查询某次 detection task 的 AI 结果（有纠错时为修正视图）、correction 状态和人工纠错内容。

**Query parameter：**

| Param | Type | Default | 说明 |
|-------|------|---------|------|
| `taskId` | string | —（必填） | |
| `includeOriginal` | bool | `false` | 同时返回 `originalResult`（不可变的原始 AI 输出，fine-tuning 数据）。 |

**Response `data`：**

| Field | Type | 说明 |
|-------|------|------|
| `aiResult` | object \| null | **有效结果**：提交过 fix 时为 `finalResult`（修正视图），否则为原始 AI 输出。字段名与原来一致，客户端无需修改。 |
| `originalResult` | object \| null | 不可变的原始 AI 输出。仅 `includeOriginal=true` 时返回。 |
| `originalImageUrl` | string \| null | fix 前的标注图（CDN URL 或本地路径）。仅 `includeOriginal=true` 时返回；无法保留时为 null（如首次 fix 时本地文件已过期且源图不可用）。 |
| `correctionStatus` | string | `"pending"` / `"corrected"` / `"reviewed"`。 |
| `correctionCount` | int | fix 提交次数。 |
| `correctedAt` | string \| null | 最近一次 fix 的时间戳（ISO 8601）。 |
| `detectionDiff` | int | 最新 fix 相对原始结果的 detection 数量变化（有符号，`-2` 表示移除了 2 个）。每次 fix 重新计算（latest-wins）。 |
| `skuMismatchCount` | int | 最新 fix 与原始结果之间 `skuId` 不同的 detection 数量（即 reassign 的数量）。 |
| `inputImageUrl` | string \| null | 未标注的输入原图（完整绝对 URL；上传失败时为基于请求 host 展开的本地 `/results/inputs/...` URL）。 |
| `userCorrection` | array \| null | 最近一次通过 `/recognition/fix` 提交的 correction（含 `remove` 记录，被移除的 detection 可追溯）。 |
| `visualImageUrl` | string \| null | 最终 annotated image URL（**始终为绝对 URL** —— 已上传时为 CDN URL，否则为 `http://<host>:<port>/results/annotated/...`）。 |

**Error response：** 404 — `taskId` 不存在。

---

### GET `/api/v1/logs/recognition/list`

分页查询 recognition log，支持按时间范围和 correction status 过滤。按 `createdAt` 降序排列。

**Query parameter：**

| Param | Type | Default | 说明 |
|-------|------|---------|------|
| `page` | int | 1 | 页码（从 1 开始）。 |
| `size` | int | 20 | 每页条数。 |
| `startTime` | string | — | ISO 8601；过滤 `createdAt >= startTime`。带时区的时间会转换为服务器本地时间。 |
| `endTime` | string | — | ISO 8601；过滤 `createdAt <= endTime`。 |
| `correctionStatus` | string | — | `"pending"` / `"corrected"` / `"reviewed"`。 |

**Response `data`：**

| Field | Type | 说明 |
|-------|------|------|
| `list` | array | Log 列表（见下文）。 |
| `page` / `pageSize` / `total` | int | 分页信息。 |

**每条 log item：**

| Field | Type | 说明 |
|-------|------|------|
| `taskId` | string | |
| `createdAt` | string \| null | ISO 8601。 |
| `correctionStatus` | string | `"pending"` / `"corrected"` / `"reviewed"`。 |
| `correctedAt` | string \| null | 最近一次 fix 的时间戳。 |
| `correctionCount` | int | fix 提交次数。 |
| `detectionCount` | int | 原始结果中的 detection 数量（迁移前 / 失败的 log 为 0）。 |
| `detectionDiff` | int | 最新 fix 相对原始结果的 detection 数量变化（有符号）。 |
| `skuMismatchCount` | int | 最新 fix 中被 reassign 的 detection 数量。 |
| `inputImageUrl` | string \| null | 未标注的输入原图 URL（绝对 URL）。 |
| `visualImageUrl` | string \| null | 最终 annotated image URL（绝对 URL）。 |

**Error response：**

| HTTP | 触发条件 |
|------|----------|
| 400 | `startTime`/`endTime` 格式无效，或 `correctionStatus` 取值非法 |

---

### POST `/api/v1/logs/recognition/status`

手动修改 log 的 correction status —— 例如标记为 `"reviewed"`（已检查、无需修改）或重置为 `"pending"` 重新打开。只改 workflow 状态：`correctionCount`、`correctedAt` 和 `finalResult` 跟随实际 fix 提交，不受影响。

**Request：**

| Field | Type | Required | 说明 |
|-------|------|----------|------|
| `taskId` | string | yes | |
| `correctionStatus` | `"pending"` \| `"corrected"` \| `"reviewed"` | yes | |

**Response：** `data` 为空 `{}`。

**Error response：** 404 — `taskId` 不存在。

---

### POST `/api/v1/logs/recognition/delete`

批量删除 recognition log。删除数据库记录和本地文件（annotated、inputs、pre-fix 快照 —— best-effort）。CDN 对象目前会成为 orphan（尚未配置七牛删除凭证）；凭证可用后会补充删除逻辑。

**Request：**

| Field | Type | Required | 说明 |
|-------|------|----------|------|
| `taskIds` | string[]（1–100） | yes | 要删除的 taskId 列表。 |

**Response：** `data = {"deleted": <int>, "notFound": ["taskId", ...]}`

```json
{"code": 1, "data": {"deleted": 2, "notFound": ["gone_id"]}, "msg": "success"}
```

---

## 8. System

### GET `/api/v1/system/train-status/get`

查询 async embedding 任务的状态。

**Query parameter：** `trainJobId`（string，必填）

**Response `data`：**

| Field | Type | 说明 |
|-------|------|------|
| `status` | string | 任务 lifecycle：`"pending"` → `"indexing"` → `"completed"`（异常时为 `"failed"`）。 |
| `progress` | int | 进度百分比（0–100）。 |
| `estimatedTime` | string \| null | 保留 field；当前始终为 `null`。 |
| `failedCount` | int | Embedding 失败的图片数量。 |
| `totalCount` | int | 任务中的图片总数。 |
| `embeddingFailed` | array \| null | 仅在有图片失败时出现：失败的 media URL 列表。 |

**示例：**

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

**Error response：** 404 — `trainJobId` 不存在。
