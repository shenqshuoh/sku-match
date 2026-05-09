from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class DetectRequest(BaseModel):
    taskId: str = Field(max_length=128)
    mode: Literal["IMAGE", "VIDEO"] = "IMAGE"
    files: str = Field(min_length=1, max_length=2048)
    roiRect: list[float] | None = None

    @field_validator("taskId")
    @classmethod
    def validate_task_id(cls, v):
        if not v.strip():
            raise ValueError("taskId must not be empty")
        return v

    @field_validator("files")
    @classmethod
    def validate_files(cls, v):
        if not v.strip():
            raise ValueError("files must not be empty")
        return v

    @field_validator("roiRect")
    @classmethod
    def validate_roi_rect(cls, v):
        if v is not None and len(v) != 4:
            raise ValueError("roiRect must have exactly 4 elements [x1, y1, x2, y2]")
        return v


class FixItem(BaseModel):
    fixType: str = Field(max_length=64)
    itemId: int | None = None
    roiRect: list[float] | None = None
    skuId: str = Field(max_length=128)


class FixRequest(BaseModel):
    taskId: str = Field(max_length=128)
    fixItems: list[FixItem] = Field(min_length=1)

    @field_validator("taskId")
    @classmethod
    def validate_not_empty(cls, v):
        if not v.strip():
            raise ValueError("taskId must not be empty")
        return v


class SKUNewRequest(BaseModel):
    skuId: str = Field(max_length=128)
    skuName: str = Field(max_length=256)
    files: list[str] = Field(min_length=1, max_length=50)
    trainJobId: str = Field(max_length=128)

    @field_validator("skuId", "skuName", "trainJobId")
    @classmethod
    def validate_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Field must not be empty")
        return v

    @field_validator("files")
    @classmethod
    def validate_files_list(cls, v):
        for url in v:
            if not url.strip():
                raise ValueError("File URLs must not be empty")
        return v


class SKUUpdateRequest(BaseModel):
    skuId: str = Field(max_length=128)
    skuName: str = Field(max_length=256)

    @field_validator("skuId", "skuName")
    @classmethod
    def validate_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Field must not be empty")
        return v


class SKUDeleteRequest(BaseModel):
    skuId: str = Field(max_length=128)

    @field_validator("skuId")
    @classmethod
    def validate_not_empty(cls, v):
        if not v.strip():
            raise ValueError("skuId must not be empty")
        return v


class SKUEnableRequest(BaseModel):
    skuId: str = Field(max_length=128)
    enabled: bool


class MediaItem(BaseModel):
    mediaId: str | None = None
    mediaUrl: str | None = None


class SKUMediaRequest(BaseModel):
    skuId: str = Field(max_length=128)
    action: Literal["add", "delete"]
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
