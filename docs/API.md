# 寄海火锅酒水饮料照片AI识别统计后端功能需求

## 一、酒水识别系统业务需求

### 1.1 业务场景

在顾客结账时，服务员要统计顾客消费的各种酒水数量并追加到订单中，目前还是人工手动统计。因此需要实现自动化统计以提升工作效率。

1. 服务员将顾客消费的全部各种酒水瓶子宽松摆放在桌台或空地上，然后使用手持买单机（安卓智能终端）拍照，圈出酒水瓶子范围（防止将邻桌的酒水误拍误统计），然后上传到AI酒水识别系统，酒水识别系统识别后返回识别的各酒水所在位置bbox、酒水sku和名称、匹配度等，服务员确认，发现匹配错误的点击选择正确酒水后提交，追回到订单中。
3. 考虑是否通过同一场景多角度多张照片合并识别统计

### 1.2 训练功能

1. 单品训练：上传一个酒水SKU单品上下左右前后多个角度、多种氛围灯光甚至损毁的照片进行训练
2. 持续训练：识别结果纠正提交后持续训练识别错误的酒水
3. 变更训练：新增或删除SKU对应图片时再次进行持续训练

### 1.3 商品库管理

1. 翻页列表已经训练的商品SKU
2. 新增商品SKU，上传图片甚至短视频，后台自动训练
3. 删除或禁用SKU，禁止识别此SKU
4. 查看SKU原始训练图片库，可以添加、删除其中的照片/短视频，后台自动重新训练，或修改sku基础信息

### 1.4 识别日志

1. 记录每次识别日志与识别错误修正日志
2. 提供识别日志翻页查看与详情查看



# 二、API 接口设计

## 2.1 酒水识别核心接口

### 1. 提交图片进行 AI 识别

- **接口名称**：`POST /api/v1/recognition/detect`

- **功能描述**：上传一张照片，返回识别到的酒水 SKU、数量及坐标。识别结果图片应体现SKU对应的名称。

- **请求参数 (application/json)**：

  ```json
  {
      "taskId": "22d6d14065a94c48873284ae0390570a",
      "mode": "IMAGE",
      "files": "https://wx3.sinaimg.cn/mw690/008hlF7Tgy1hrg8akgyc0j30m80m8sd2.jp",
      "roiRect": [
          1332.196533203125,
          1512.436767578125,
          1594.2884521484375,
          1954.576171875
      ]
  }
  ```
  
  **字段说明**
  
  - `file`: String(必填) - 照片或视频文件存放OSS的url地址。
- `taskId`: String (必填) - 任务请求ID。
  - `roiRect`: Object (可选) - 服务员圈选的范围坐标 `{x1, y1, x2, y2}`。

- **返回参数 (JSON)**：

  ```json
  {
    "code": 1,
    "data": {
      "counts": {
        "t03": 2,
        "t04": 1
      },
      "detections": [
        {
          "bbox": [
            1332.196533203125,
            1512.436767578125,
            1594.2884521484375,
            1954.576171875
          ],
          "class_id": 1,
          "class_name": "Canned",
          "detection_conf": 0.8952532410621643,
          "itemId": 1,
          "match_score": 1.5100553035736084,
          "sku_id": "t03",
          "sku_name": "可口可乐330ml无糖听装"
        },
        {
          "bbox": [
            43.39860916137695,
            1369.1656494140625,
            367.02008056640625,
            1879.909423828125
          ],
          "class_id": 1,
          "class_name": "Canned",
          "detection_conf": 0.8890917301177979,
          "itemId": 2,
          "match_score": 1.01047945022583,
          "sku_id": "t03",
          "sku_name": "可口可乐330ml无糖听装"
        },
        {
          "bbox": [
            2074.9052734375,
            1362.5091552734375,
            2323.474853515625,
            1980.7591552734375
          ],
          "class_id": 0,
          "class_name": "Bottle",
          "detection_conf": 0.8829686045646667,
          "itemId": 3,
          "match_score": 1.576993465423584,
          "sku_id": "t04",
          "sku_name": "东方树叶-茉莉花茶500ml"
        }
      ],
      "matched_image": "https://jihai.qiniu.cn/mw690/008hlF7Tgy1hrg8akgyc0j30m80m8abm.jp",
      "taskId": "93b6fb09db0e432684bcf847f889b818"
    },
    "msg": "成功"
  }
  ```
  

### 2. 纠正识别结果

- **接口名称**：`POST /api/v1/recognition/fix`

- **功能描述**：服务员人工核对后，提交最终准确数据，用于追加订单及后续持续训练。

- **请求参数 (JSON)**：
  
  ```json
  {
    "fixItems": [
      {
        "fixType": "CHANGE_SKU",
        "itemId": 3,
        "skuId": "9f801c3884664728b4340debefc6b74f"
      },
      {
        "fixType": "LOST",
        "roiRect": [
          1332.196533203125,
          1512.436767578125,
          1594.2884521484375,
          1954.576171875
        ],
        "skuId": "7a88824b00024780884547a7aa05a40a"
      }
    ],
    "taskId": "22d6d14065a94c48873284ae0390570a"
  }
  ```
  
  
  
  - `taskId`: String (必填) - 对应识别请求 ID。
  - `fixItems`: Array - 修正后的最终列表。
    - `itemId`: Int 对应修正的ItemId
    - fixType: String 纠正方式： CHANGE_SKU - 更换sku，LOST - 补充未识别
    - `skuId`: String 正确的skuId
  
- **返回参数 (JSON)**：
  - `status`: String (success/fail)。

------

## 2.2 商品库与训练管理接口

### 1. 新增商品 SKU 并发起训练

- **接口名称**：`POST /api/v1/goods/sku/new`
- **功能描述**：创建新商品，上传多角度素材并自动触发增量训练。
- **请求参数 (JSON)**：
  
  ```json
  {
    "files": [
      "https://wx3.sinaimg.cn/mw690/008hlF7Tgy1hrg8akgyc0j30m80m8abm.jp",
      "https://wx3.sinaimg.cn/mw690/008hlF7Tgy1hrg8akgyc0j30m80m8sd2.jp",
      "https://wx3.sinaimg.cn/mw690/008hlF7Tgy1hrg8akgyc0j30m80m8cvx.jp"
    ],
    "skuId": "e6704c6b456c4382a3eb3b8829fe94f4",
    "skuName": "哇米诺豆奶",
    "trainJobId": "22d6d14065a94c48873284ae0390570a"
  }
  ```
  
  
  
  - `trainJobId`: String(必填) - 训练任务ID
  - `skuName`: String (必填) - 商品名称。
  - `skuId`: String - SKU编号。
  - `files`: String[] - 多角度照片或视频存放在OSS的url地址数组。
- **返回参数 (JSON)**：
  - `skuId`: String。
  - `trainJobId`: String - 训练任务 ID。

### 2. 修改sku名称

- **接口名称**：`POST /api/v1/goods/sku/update`

- **功能描述**：修改sku的商品名称。

- **请求参数 (JSON)**：

  ```json
  {
    "skuId": "e6704c6b456c4382a3eb3b8829fe94f4",
    "skuName": "哇米诺豆奶"
  }
  ```

  

  - `skuName`: String (必填) - 商品名称。
  - `skuId`: String - SKU编号。

- **返回参数 (JSON)**：

  - `status`: String (success/fail)。

### 3. 删除sku

- **接口名称**：`POST /api/v1/goods/sku/delete`

- **功能描述**：删除SKU，同时删除对应的训练数据。

- **请求参数 (JSON)**：

  ```json
  {
    "skuId": "e6704c6b456c4382a3eb3b8829fe94f4"
  }
  ```

  

  - skuId`: String - SKU编号。

- **返回参数 (JSON)**：

  - `status`: String (success/fail)。

### 4. 禁用/启用sku 

- **接口名称**：`POST /api/v1/goods/sku/enable`

- **功能描述**：启用或禁用sku。禁用sku时不识别此sku，但并不删除此sku数据

- **请求参数 (JSON)**：

  ```json
  {
    "skuId": "e6704c6b456c4382a3eb3b8829fe94f4",
    "enabled": true
  }
  ```

  

  - `enabled`: Bool (必填) - true/false。
  - `skuId`: String - SKU编号。

- **返回参数 (JSON)**：

  - `status`: String (success/fail)。

### 5. 获取商品列表（翻页）

- **接口名称**：`GET /api/v1/goods/sku/list`

- **请求参数**：`page`, `size`, `keyword` (可选，匹配商品名称与skuId)。

- **返回参数 (JSON)**：

  ```json
  {
    "code": 1,
    "data": {
      "list": [
        {
          "id": 1,
          "medias": [
            {
              "mediaId": "14154708460d4b3e9dd0dbec56935073",
              "mediaType": "IMAGE",
              "mediaUrl": "https://wx3.sinaimg.cn/mw690/008hlF7Tgy1hrg8akgyc0j30m80m8abm.jpg"
            },
            {
              "mediaId": "df23662194eb48b4b5140ddb91fa8eb6",
              "mediaType": "IMAGE",
              "mediaUrl": "https://wx3.sinaimg.cn/mw690/008hlF7Tgy1hrg8akgyc0j30m80m8abk.jpg"
            },
            {
              "mediaId": "0b077a3f48624bfe9199d33098fd10c4",
              "mediaType": "IMAGE",
              "mediaUrl": "https://wx3.sinaimg.cn/mw690/008hlF7Tgy1hrg8akgyc0j30m80m8awa.jpg"
            },
            {
              "mediaId": "ced21a625b5e4052841f7903c59f6a34",
              "mediaType": "IMAGE",
              "mediaUrl": "https://wx3.sinaimg.cn/mw690/008hlF7Tgy1hrg8akgyc0j30m80m8azv.jpg"
            },
            {
              "mediaId": "e3ae3f4beab54649978226ead631e908",
              "mediaType": "IMAGE",
              "mediaUrl": "https://wx3.sinaimg.cn/mw690/008hlF7Tgy1hrg8akgyc0j30m80m8abq.jpg"
            }
          ],
          "skuId": "d1f99b0a04354b02ae7439b6b57f15d3",
          "skuName": "东方树叶茉莉花茶500ml",
          "trainStatus": "SUCCESS"
        }
      ],
      "page": 1,
      "pageSize": 20,
      "total": 1
    },
    "msg": "success"
  }
  ```

  

### 6. 修改/删除 SKU 素材

- **接口名称**：`POST /api/v1/goods/sku/media`

- **功能描述**：添加或删除特定 SKU 的训练照片/视频，操作后自动触发模型更新。

- **请求参数 (JSON)**：
  
  ```json
  {
    "action": "add",
    "media": [
      {
        "mediaId": "",
        "mediaUrl": "https://wx3.sinaimg.cn/mw690/008hlF7Tgy1hrg8akgyc0j30m80m8abm.jp"
      },
      {
        "mediaUrl": "https://wx3.sinaimg.cn/mw690/008hlF7Tgy1hrg8akgyc0j30m80m8sd2.jp"
      }
    ],
    "skuId": "e6704c6b456c4382a3eb3b8829fe94f4"
  }
  ```
  
  
  
  - `skuId`: String
  - `action`: Enum (add/delete)。
  - `media`: Array 
    - `mediaId`: 删除时必须有，添加时为空
    - `mediaUrl`: 新增的图片或视频的oss的url
  
- **返回参数 (JSON)**：
  - `status`: String。

------

## 2.3 识别日志接口

### 1. 获取识别日志详情

- **接口名称**：`GET /api/v1/logs/recognition/get`
- **功能描述**：查看某次识别的具体图片、AI 标框结果以及人工修正对比。
- **请求参数**：`taskId` 识别任务id
- **返回参数 (JSON)**：
  - `ai_result`: Object (原始识别数据)。
  - `user_correction`: Object (人工修正数据)。
  - `visual_image_url`: String (带有 bbox 渲染的预览图)。

------

## 2.4 系统与状态接口

### 8. 训练任务状态查询

- **接口名称**：`GET /api/v1/system/train-status/get`
- **功能描述**：查询增量训练或变更训练的进度。
- **请求参数**：`trainJobId` 查询指定训练任务的进度状态
- **返回参数 (JSON)**：
  - `progress`: Integer (0-100)。
  - `status`: Enum (training, completed, failed)。
  - `estimated_time`: String - 预计剩余时间。

------

### 设计说明：

1. **异步处理**：由于视频识别和模型训练属于耗时操作，接口采用任务 ID (`trainJobId` / `taskId`) 模式，前端可通过轮询或 WebSocket 获取结果。
2. **闭环训练**：通过 `2.1.2 确认纠正` 接口，系统能够获取“标注数据”，直接喂给训练功能实现持续进化。
3. **容错性**：支持多图上传，后端可采用多帧融合算法提升识别精度。