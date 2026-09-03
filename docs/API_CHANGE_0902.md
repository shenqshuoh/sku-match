# API 变更说明（2026-09-02）

本次接口优化内容的变更明细，面向接入方（panel / 客户端）。完整字段说明见 API_GUIDE_zh.md。

## 1. 修复操作后重新生成识别图片

fix 提交成功后，server 基于修正结果重新生成 annotated image 并上传新的 versioned key（{taskId}_annotated_v2.jpg、_v3…，upload token 为 insert-only 不可覆盖）。

fix 语义为**增量累积**：每次提交的 fixItems 应用在当前修正结果上，之前的纠错保留（不是整体覆盖）。userCorrection 字段按提交顺序累积全部纠错记录。注意：已被 remove 的检测目标不可在后续 fix 中再引用（返回 400）。

fixType 新增 **add**：新增人工检测目标（补漏），需提供 roiRect（手动框选 bbox）与 skuId。新条目分配下一个空闲 itemId，模型输出字段（detectionConf / classId / className / matchScore / skuDistribution / matchedVectorTags）为 null，并携带 source: "manual" 标记 —— 客户端渲染时需 null 判断，fine-tuning 数据可按 source 过滤。

- fix response 新增返回：data = {"matchedImage": "<url>"}（绝对 URL）；输入图不可用等跳过重新生成的场景为 {}
- matchedImage / visualImageUrl 每次 fix 后都会变化 —— 客户端始终从 fix response 或 log 接口读取最新 URL，不要缓存
- 首次 fix 时同时保留 fix 前的图片快照，经 originalImageUrl 返回（见下节）
- 识别源图在 detect 时持久化到本地 inputs/（保留 72 小时）供重新生成使用；过期则回退重新下载源图
- JSON 结果（finalResult）始终是权威数据，图片重新生成为 best-effort

## 2. 日志列表返回原始图与识别结果图片

GET /api/v1/logs/recognition/list 的 list item 新增两个图片字段，GET /api/v1/logs/recognition/get 同步返回：

| Field          | Type          | 说明                                                 |
|----------------|---------------|------------------------------------------------------|
| inputImageUrl  | string \| null | 未标注的输入原始图 URL                               |
| visualImageUrl | string \| null | 识别结果 annotated image URL（有纠错时为修正后版本） |

两个 URL 均为**绝对 URL**（此前为相对路径）：CDN 上传成功时为 CDN URL，失败时为含 host 的本地 http://<host>:<port>/results/... URL。历史 log 的 inputImageUrl 可能为 null（该功能上线前的任务无持久化源图）。

GET /api/v1/logs/recognition/get 新增 query parameter includeOriginal=true：额外返回 originalResult（不可变的原始识别输出，fine-tuning 数据）与 originalImageUrl（fix 前的 annotated image URL，无法保留时为 null）。

## 3. 识别日志删除接口

新增 POST /api/v1/logs/recognition/delete，批量删除识别 log。

**Request：**

| Field   | Type              | Required | 说明                 |
|---------|-------------------|----------|----------------------|
| taskIds | string[]（1–100） | yes      | 要删除的 taskId 列表 |

**Response：** data = {"deleted": 2, "notFound": ["gone_id"]}（不存在的 taskId 列入 notFound，不报错）

同时清理本地文件（annotated / inputs / pre-fix 快照，best-effort）；CDN 对象暂为 orphan（未配置 Qiniu 删除凭证），后续补充。

## 4. SKU 媒体列表增加识别失败标识

GET /api/v1/goods/sku/list 的 medias item 新增 failed 标志：

| Field  | Type    | 说明                                                            |
|--------|---------|-----------------------------------------------------------------|
| failed | boolean | true 表示该 reference 图**识别失败**（未产出 crop / embedding） |

- 识别失败的 media 行照常入库并在列表可见，可经 /goods/sku/media delete 删除后重新添加
- POST /api/v1/goods/sku/media（add）行为同步调整：部分图片 embedding 失败不再整体报错，返回 HTTP 200 + code=0，data.embeddingFailed 列出失败 URL（其余可能已成功），需人工检查后决定重拍或重新添加
- media item 新增可选字段 preCropped（默认 false）：true 表示图片已是裁好的 reference，server 跳过 YOLOE crop / 背景遮罩直接 embed —— 适用于 panel 端已裁剪或外部产出的成品图
