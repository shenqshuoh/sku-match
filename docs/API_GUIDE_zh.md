# SKU Match API 指南

Base URL http://<host>:8000，所有 endpoint 前缀 /api/v1。

## 1. Authentication

api/config.py: API_KEY 非空时，所有 /api/v1/* request 需带 header（为空时免认证，当前部署默认禁用）：

```bash
curl -H "X-API-Key: <key>" http://localhost:8000/api/v1/goods/sku/list
```

## 2. Response 格式

统一 envelope：

```json
{"code": 1, "data": {...}, "msg": "success"}
{"code": 0, "data": null, "msg": "..."}
```

错误时 HTTP status code 与 code=0 同时返回：

| HTTP | 场景                | msg 示例                                   |
|----|-------------------|------------------------------------------|
| 400  | 无效 request        | Unsupported action: '...'                  |
| 404  | resource 不存在     | skuId '...' not found                      |
| 409  | resource 重复       | taskId '...' already exists                |
| 422  | validation error    | Pydantic 报错                              |
| 500  | 内部错误            | Unexpected exception                       |
| 503  | service unavailable | No SKUs indexed yet — add references first |

## 3. Endpoint 一览

| Method | Path                            | 说明                                                 |
|------|-------------------------------|----------------------------------------------------|
| GET    | /health                         | Health check                                         |
| POST   | /api/v1/recognition/detect      | 识别饮料容器并匹配 SKU                               |
| POST   | /api/v1/recognition/fix         | 提交人工纠错（reassign / remove / adjust-roi / add） |
| POST   | /api/v1/goods/sku/new           | 创建 SKU 并启动 embedding 任务                       |
| POST   | /api/v1/goods/sku/update        | 更新 skuName                                         |
| POST   | /api/v1/goods/sku/delete        | 删除 SKU                                             |
| POST   | /api/v1/goods/sku/enable        | 启用 / 禁用 SKU                                      |
| GET    | /api/v1/goods/sku/list          | 分页查询 SKU 列表                                    |
| POST   | /api/v1/goods/sku/media         | 添加 / 删除 SKU media                                |
| GET    | /api/v1/logs/recognition/get    | 查询单条 recognition log                             |
| GET    | /api/v1/logs/recognition/list   | 分页查询 recognition log                             |
| POST   | /api/v1/logs/recognition/status | 设置纠错状态                                         |
| POST   | /api/v1/logs/recognition/delete | 批量删除 recognition log                             |
| GET    | /api/v1/system/train-status/get | 查询 embedding 任务状态                              |

## 4. Recognition

### POST /api/v1/recognition/detect

识别图中饮料容器并与 SKU index 匹配。

**Request：**

| Field   | Type               | Required | Default | 说明                                             |
|-------|------------------|--------|-------|------------------------------------------------|
| taskId  | string             | yes      | —       | 重复提交返回 409                                 |
| mode    | "IMAGE" \| "VIDEO" | no       | "IMAGE" | VIDEO 当前不可用                                 |
| files   | string             | yes      | —       | 单张图片 URL 或本地路径                          |
| roiRect | number[4] \| null  | no       | null    | [x1, y1, x2, y2]，忽略中心点落在区域外的检测目标 |

```json
{
  "taskId": "task-001",
  "files": "https://example.com/fridge-photo.jpg",
  "roiRect": [0, 0, 1920, 1080]
}
```

**Response data：**

| Field        | Type   | 说明                                                                       |
|------------|------|--------------------------------------------------------------------------|
| taskId       | string |                                                                            |
| matchedImage | string | Annotated image URL；Qiniu 上传失败时为本地 URL 且 qiniuUploadFailed: true |
| counts       | object | skuId → 数量                                                               |
| detections   | array  | 检测目标列表（见下）                                                       |

**检测目标字段：**

| Field             | Type           | 说明                                                                                                                                                         |
|-----------------|--------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| itemId            | int            | 序号，从 1 开始                                                                                                                                              |
| bbox              | number[4]      | [x1, y1, x2, y2] 像素坐标                                                                                                                                    |
| classId           | int            | YOLOE class ID                                                                                                                                               |
| className         | string         | YOLOE class label（bottle / canned / …），业务可忽略                                                                                                         |
| detectionConf     | float          | YOLOE detection confidence（0–1）                                                                                                                            |
| skuId             | string         | 匹配结果；低于 MATCH_CONF threshold 时为空                                                                                                                   |
| skuName           | string         | 同上                                                                                                                                                         |
| matchScore        | float          | Match confidence（softmax，0–1）；低于 threshold 时为 0.0                                                                                                    |
| skuDistribution   | object \| null | Top-N {skuId: {skuName, score}}，按 score 降序；N 由 DISTRIBUTION_TOP_K（默认 10）控制，实际条数受 RERANK_TOP_K 搜索池限制（当前约 5）                       |
| matchedVectorTags | array \| null  | 调试用的 reference vector 列表，可忽略                                                                                                                       |
| source            | string \| null | 检测目标来源："model"（模型自动检测）或 "manual"（经 fix add 人工新增）。人工条目的上述模型输出字段均为 null。该字段上线前创建的历史日志已统一回填为 "model" |

**Response 示例：**

```json
{
  "code": 1,
  "data": {
    "taskId": "task-001",
    "matchedImage": "http://110.76.42.124:8000/results/annotated/task-001_annotated.jpg",
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
        "skuDistribution": {
          "100001_1664": {"skuName": "1664", "score": 0.87},
          "100003_ws": {"skuName": "乌苏", "score": 0.08}
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

**错误：** 409 taskId 重复；503 index 为空；500 其他异常。

### POST /api/v1/recognition/fix

对识别结果人工纠错，不重新执行识别。纠错应用在修正副本 finalResult 上；原始输出 originalResult 不可变（fine-tuning 数据）。fix 为**增量累积**：每次提交应用在当前 finalResult 上，之前的纠错保留；userCorrection 按提交顺序累积全部纠错记录。提交成功后 correctionStatus 为 corrected，correctionCount 加一。

userCorrection 为操作日志，每条记录由服务端按纠错发生时的视图补全明细：reassign 记录 itemId、roiRect（纠错时的 bbox）、skuId（新值）与 skuIdOld（该次 fix 前视图中的 skuId，即操作者当时看到的值）；remove 记录被移除检测目标当时的 roiRect 与 skuId；adjust-roi 记录 roiRect（新值）、roiRectOld（fix 前 bbox）与该目标的 skuId；add 记录服务端分配的 itemId（请求不携带）、roiRect 与 skuId。add 的 itemId 单调递增且不复用：下一个 add id 取原始结果、当前视图与历史 add 记录中出现过的最大 itemId 加一 —— 已添加后又被移除的目标仍占用其 id（记录保留在 userCorrection 中），后续 add 跳过该 id。

**Request：**

| Field    | Type           | Required | 说明             |
|--------|--------------|--------|----------------|
| taskId   | string         | yes      | 已存在的识别任务 |
| fixItems | array（min 1） | yes      | 纠错列表（见下） |

**FixItem：**

| Field   | Type                                            | Required              | 说明                                                                                 |
|-------|-----------------------------------------------|---------------------|------------------------------------------------------------------------------------|
| fixType | "reassign" \| "remove" \| "adjust-roi" \| "add" | yes                   | **Breaking**：不再接受自定义值（如 "misidentification"，422 拒绝），用 reassign 代替 |
| itemId  | int \| null                                     | 除 add 外必填         | 必须存在于当前结果中（否则 400；已被 remove 的 item 不可再引用）；add 忽略该字段     |
| roiRect | number[4] \| null                               | adjust-roi / add 必填 | 修正后的 bbox；add 时为手动框选的新检测目标 bbox                                     |
| skuId   | string \| null                                  | reassign / add 必填   | 修正后的 skuId（允许引用已删除的 SKU）；add 时为新检测目标指定的 SKU                 |

remove 移除该检测目标（可经 userCorrection 追溯）；reassign 替换 skuId/skuName；adjust-roi 替换 bbox；add 新增人工检测目标（补漏）：分配下一个空闲 itemId（单调递增、不复用，见上文 userCorrection 明细），模型输出字段（detectionConf / classId / className / matchScore / skuDistribution / matchedVectorTags）均为 null，并携带 source: "manual" 标记（模型条目为 source: "model"）—— 客户端需对这些字段做 null 判断，fine-tuning 可按 source 过滤。

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

**Annotated image 重新生成：** 每次 fix 后基于 finalResult 重新生成并上传到新的 versioned key（{taskId}_annotated_v2.jpg、_v3…，token 为 insert-only 不可覆盖），matchedImage / visualImageUrl 每次 fix 都变化 —— 始终从 fix response 或 log 接口读取最新 URL，客户端不要缓存。首次 fix 时同时保留 originalImageUrl。源图在 detect 时持久化到 inputs/（72 小时），过期则回退重新下载源图。

**Response：** data = {"matchedImage": "<url>"}（绝对 URL）；跳过重新生成时为 {}。

**错误：** 404 taskId 不存在；400 itemId 不存在于当前结果（如已被之前的 fix 移除）或 log 无有效 AI 结果；500 内部错误。

## 5. Goods（SKU Management）

### POST /api/v1/goods/sku/new

创建 SKU 并启动 async embedding 任务（状态经 /system/train-status/get 查询）。

**Request：**

| Field      | Type     | Required | 说明                                 |
|----------|--------|--------|------------------------------------|
| skuId      | string   | yes      | Bare id（见下）                      |
| skuName    | string   | yes      | 显示名称                             |
| files      | string[] | yes      | Reference 图片 URL / 路径（1–50 条） |
| trainJobId | string   | yes      | 唯一 job ID                          |

**skuId 自动编号：**

- 提交 bare id（拼音缩写，不含数字前缀），自动加递增前缀：当前最大 100106_x 时提交 kkkl → 100107_kkkl。提交串原样作为 suffix，不做清洗。
- 重复判定：缩写相同**且** skuName 相同 → 409；skuName 不同 → 允许创建。

**Response：** data = {"skuId": "...", "trainJobId": "..."}

**错误：** 409 skuId 或 trainJobId 重复。

### POST /api/v1/goods/sku/update

更新 skuName（同步 database 与 Chroma metadata）。

**Request：** {"skuId": "...", "skuName": "..."} → data 为 {}。**错误：** 404 skuId 不存在。

### POST /api/v1/goods/sku/delete

删除 SKU 并清理 media 行、Chroma vector、patch / color 文件。

**Request：** {"skuId": "..."} → data 为 {}。**错误：** 404 skuId 不存在。

### POST /api/v1/goods/sku/enable

禁用的 SKU 不参与匹配。

**Request：** {"skuId": "...", "enabled": true} → data 为 {}。**错误：** 404 skuId 不存在。

### GET /api/v1/goods/sku/list

**Query parameter：**

| Param       | Type   | Default | 说明                                                |
|-----------|------|-------|---------------------------------------------------|
| page        | int    | 1       | 页码（从 1 开始）                                   |
| size        | int    | 20      | 每页条数                                            |
| keyword     | string | —       | 按 skuId / skuName 模糊搜索（不区分大小写）         |
| trainStatus | string | —       | pending / indexing / completed / failed，非法值 400 |
| enabled     | bool   | —       | true / false，不传返回全部                          |

**Response data：** {list, page, pageSize, total}，其中 list item：

| Field       | Type   | 说明                                                                                        |
|-----------|------|-------------------------------------------------------------------------------------------|
| skuId       | string | 含数字前缀的完整 id                                                                         |
| skuName     | string |                                                                                             |
| trainStatus | string | pending → indexing → completed（异常 failed）                                               |
| medias      | array  | [{mediaId, mediaType, mediaUrl, failed}]；failed: true 表示该图未产出 embedding，需人工处理 |

### POST /api/v1/goods/sku/media

**Request：**

| Field        | Type              | Required | 说明                                                                                                     |
|------------|-----------------|--------|--------------------------------------------------------------------------------------------------------|
| skuId        | string            | yes      |                                                                                                          |
| action       | "add" \| "delete" | yes      |                                                                                                          |
| media        | array             | yes      | Media item 列表                                                                                          |
| trainJobId   | string            | no       | 仅 add：panel 提供的 job id，用于轮询；与已有 job 重复返回 409；缺省时 server 生成 job_{skuId}_{unix_ts} |

**Media item：** add 需 mediaUrl（要 embed 的图片），delete 需 mediaId；add 时携带 mediaId 的 item 会被跳过；另有可选字段 preCropped（默认 false）：

| Field      | Type | 说明                                                                                                                                            |
|----------|----|-----------------------------------------------------------------------------------------------------------------------------------------------|
| preCropped | bool | 默认 false：server 执行 YOLOE crop + 背景遮罩。true：图片已是裁好的 reference，原样使用（仅 EXIF 校正），跳过检测环节，不会因“未检测到目标”失败 |

**行为（add 为异步）：** 先写入 media 行，随后在后台 job 中 embedding（与 /goods/sku/new 相同管线：download → crop（preCropped: true 跳过）→ embedding → 写入 Chroma → patch / color 缓存 → 上传 Qiniu）。请求立即返回 trainJobId —— 通过 GET /api/v1/system/train-status/get 轮询进度，失败的图片在轮询结果的 embeddingFailed 中列出，并在 SKU media 列表中标记 failed；job 结束时基于该 SKU 全部 media 重算 trainStatus。若所有 item 均携带 mediaId（无可 embed 项），不创建 job，data 为 {}。

**行为（delete）：** 清理 media 行、vector、patch / color 文件，随后重算该 SKU 的 trainStatus（删除最后一张失败图片时 failed → completed；全部删空后为 pending）。

**Response：** add 返回 data = {"trainJobId": "...", "mediaIds": ["...", ...]}（mediaIds 与请求顺序一致）；delete 为 {}。

**错误：** 400 参数缺失或 action 非法；404 skuId 不存在；409 提供的 trainJobId 已存在。

## 6. Log

### GET /api/v1/logs/recognition/get

**Query parameter：** taskId（必填）；includeOriginal=true 时额外返回 originalResult / originalImageUrl。

**Response data：**

| Field             | Type           | 说明                                                                                                                                                                                               |
|-----------------|--------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| aiResult          | object \| null | **有效结果**：有 fix 时为 finalResult（修正视图），否则为原始 AI 输出                                                                                                                              |
| originalResult    | object \| null | 不可变原始输出，仅 includeOriginal=true 时返回                                                                                                                                                     |
| originalImageUrl  | string \| null | Pre-fix annotated image URL，仅 includeOriginal=true 时返回                                                                                                                                        |
| correctionStatus  | string         | pending / corrected / reviewed                                                                                                                                                                     |
| correctionCount   | int            | fix 提交次数                                                                                                                                                                                       |
| correctedAt       | string \| null | 最近一次 fix 时间（ISO 8601）                                                                                                                                                                      |
| detectionsAdded   | int            | 当前结果相对原始结果新增的检测目标数量（fix 手工添加且仍存在）                                                                                                                                     |
| detectionsRemoved | int            | 原始结果中已被移除、不再存在于当前结果的检测目标数量                                                                                                                                               |
| skuMismatchCount  | int            | 最新 fix 中被 reassign 的数量                                                                                                                                                                      |
| inputImageUrl     | string \| null | 未标注输入原图 URL（绝对 URL）                                                                                                                                                                     |
| userCorrection    | array \| null  | 历次提交的 fixItems 按顺序累积（操作日志）。每条记录含服务端补全的明细：全部类型记录 roiRect 与 skuId，reassign 另有 skuIdOld（fix 前值），adjust-roi 另有 roiRectOld，add 记录服务端分配的 itemId |
| visualImageUrl    | string \| null | 最终 annotated image URL（始终为绝对 URL，本地 fallback 时含 host）                                                                                                                                |

**错误：** 404 taskId 不存在。

### GET /api/v1/logs/recognition/list

按 createdAt 降序分页。

**Query parameter：**

| Param            | Type   | Default | 说明                             |
|----------------|------|-------|--------------------------------|
| page / size      | int    | 1 / 20  | 分页                             |
| startTime        | string | —       | ISO 8601，createdAt >= startTime |
| endTime          | string | —       | ISO 8601，createdAt <= endTime   |
| correctionStatus | string | —       | pending / corrected / reviewed   |

**Response data：** {list, page, pageSize, total}，list item 为 taskId、createdAt、correctionStatus、correctedAt、correctionCount、detectionCount、detectionsAdded、detectionsRemoved、skuMismatchCount、inputImageUrl、visualImageUrl（含义同上）。

**错误：** 400 时间格式无效或 correctionStatus 非法。

### POST /api/v1/logs/recognition/status

手动设置纠错状态，例如标记 reviewed 或重置 pending。只改 workflow 状态，不影响 finalResult / correctionCount。

**Request：** {"taskId": "...", "correctionStatus": "reviewed"} → data 为 {}。**错误：** 404 taskId 不存在。

### POST /api/v1/logs/recognition/delete

批量删除 log 及本地文件（annotated / inputs / pre-fix snapshot，best-effort）。CDN 对象暂为 orphan（未配置 Qiniu 删除凭证）。

**Request：** {"taskIds": ["...", "..."]}（1–100 条）

**Response：** data = {"deleted": 2, "notFound": ["gone_id"]}

## 7. System

### GET /api/v1/system/train-status/get

**Query parameter：** trainJobId（必填）

**Response data：**

| Field           | Type           | 说明                                                                                          |
|---------------|--------------|---------------------------------------------------------------------------------------------|
| status          | string         | pending → indexing → completed / failed                                                       |
| progress        | int            | 0–100                                                                                         |
| estimatedTime   | string \| null | 保留字段，当前恒为 null                                                                       |
| failedCount     | int            | Embedding 失败图片数                                                                          |
| totalCount      | int            | 图片总数                                                                                      |
| embeddingFailed | array \| null  | 失败的 media URL 列表（仅在失败时出现；/goods/sku/new 与 /goods/sku/media add 的 job 均适用） |

**错误：** 404 trainJobId 不存在。
