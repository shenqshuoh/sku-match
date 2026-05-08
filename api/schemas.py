from typing import Any

from pydantic import BaseModel, field_validator


class DetectRequest(BaseModel):
    taskId: str
    mode: str = "IMAGE"
    files: str
    roiRect: list[float] | None = None

    @field_validator("roiRect")
    @classmethod
    def validate_roi_rect(cls, v):
        if v is not None and len(v) != 4:
            raise ValueError("roiRect must have exactly 4 elements [x1, y1, x2, y2]")
        return v


class FixItem(BaseModel):
    fixType: str
    itemId: int | None = None
    roiRect: list[float] | None = None
    skuId: str


class FixRequest(BaseModel):
    taskId: str
    fixItems: list[FixItem]


class SKUNewRequest(BaseModel):
    skuId: str
    skuName: str
    files: list[str]
    trainJobId: str


class SKUUpdateRequest(BaseModel):
    skuId: str
    skuName: str


class SKUDeleteRequest(BaseModel):
    skuId: str


class SKUEnableRequest(BaseModel):
    skuId: str
    enabled: bool


class MediaItem(BaseModel):
    mediaId: str | None = None
    mediaUrl: str | None = None


class SKUMediaRequest(BaseModel):
    skuId: str
    action: str
    media: list[MediaItem]


class DetectionItem(BaseModel):
    itemId: int
    bbox: list[float]
    class_id: int
    class_name: str
    detection_conf: float
    sku_id: str
    sku_name: str
    match_score: float
    match_concentration: float = 0.0
    sku_distribution: dict[str, float] | None = None


class DetectData(BaseModel):
    counts: dict[str, int]
    detections: list[DetectionItem]
    matched_image: str
    taskId: str


class ApiResponse(BaseModel):
    code: int = 1
    data: Any | None = None
    msg: str = "\u6210\u529f"


class MediaResponse(BaseModel):
    mediaId: str
    mediaType: str
    mediaUrl: str


class SKUListItem(BaseModel):
    id: int
    skuId: str
    skuName: str
    trainStatus: str
    medias: list[MediaResponse]


class SKUListData(BaseModel):
    list: list[SKUListItem]
    page: int
    pageSize: int
    total: int


class StatusResponse(BaseModel):
    status: str


class TrainStatusResponse(BaseModel):
    status: str
    progress: int
    estimated_time: str | None


class LogResponse(BaseModel):
    ai_result: Any | None
    user_correction: Any | None
    visual_image_url: str | None
