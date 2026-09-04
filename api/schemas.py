from datetime import datetime
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, Field, field_validator, model_validator

T = TypeVar("T")


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
    fixType: Literal["reassign", "remove", "adjust-roi", "add"]
    itemId: int | None = None  # not used by fixType "add" (new entry gets the next free id)
    roiRect: list[float] | None = None
    skuId: str | None = Field(default=None, max_length=128)

    @model_validator(mode="after")
    def validate_conditional_fields(self):
        if self.fixType == "reassign" and not (self.skuId or "").strip():
            raise ValueError("skuId is required when fixType is 'reassign'")
        if self.fixType in ("adjust-roi", "add"):
            if self.roiRect is None:
                raise ValueError(f"roiRect is required when fixType is '{self.fixType}'")
            if len(self.roiRect) != 4:
                raise ValueError("roiRect must have exactly 4 elements [x1, y1, x2, y2]")
        if self.fixType == "add" and not (self.skuId or "").strip():
            raise ValueError("skuId is required when fixType is 'add'")
        return self


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
    preCropped: bool = False  # add: image is already cropped — skip YOLOE crop/mask


class SKUMediaRequest(BaseModel):
    skuId: str = Field(max_length=128)
    action: Literal["add", "delete"]
    media: list[MediaItem]
    # Optional panel-supplied job id (add action only). Absent → the server
    # generates one; supplied ids are deduplicated (409 on collision).
    trainJobId: str | None = Field(default=None, max_length=128)


class SkuDistributionEntry(BaseModel):
    skuName: str
    score: float


class DetectionItem(BaseModel):
    itemId: int
    bbox: list[float]
    classId: int
    className: str
    detectionConf: float
    skuId: str
    skuName: str
    matchScore: float
    matchConcentration: float = Field(
        default=0.0,
        deprecated=True,
        description="Deprecated: redundant with matchScore. Will be removed in a future version.",
    )
    skuDistribution: dict[str, SkuDistributionEntry] | None = None
    matchedVectorTags: list[dict[str, Any]] | None = None
    source: str | None = None  # "model" (auto-detected) | "manual" (added via fix)


class DetectData(BaseModel):
    counts: dict[str, int]
    detections: list[DetectionItem]
    matchedImage: str
    taskId: str
    qiniuUploadFailed: bool | None = None


class ApiResponse(BaseModel, Generic[T]):
    code: int = 1
    data: T | None = None
    msg: str = "success"


class MediaResponse(BaseModel):
    mediaId: str
    mediaType: str
    mediaUrl: str
    failed: bool = False


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


class LogStatusRequest(BaseModel):
    taskId: str = Field(max_length=128)
    correctionStatus: Literal["pending", "corrected", "reviewed"]

    @field_validator("taskId")
    @classmethod
    def validate_not_empty(cls, v):
        if not v.strip():
            raise ValueError("taskId must not be empty")
        return v


class LogDeleteRequest(BaseModel):
    taskIds: list[str] = Field(min_length=1, max_length=100)

    @field_validator("taskIds")
    @classmethod
    def validate_not_empty(cls, v):
        for tid in v:
            if not tid.strip():
                raise ValueError("taskIds must not contain empty strings")
        return v


class LogListItem(BaseModel):
    taskId: str
    createdAt: datetime | None = None
    correctionStatus: str
    correctedAt: datetime | None = None
    correctionCount: int = 0
    detectionCount: int = 0
    detectionsAdded: int = 0
    detectionsRemoved: int = 0
    skuMismatchCount: int = 0
    inputImageUrl: str | None = None
    visualImageUrl: str | None = None


class LogListData(BaseModel):
    list: list[LogListItem]
    page: int
    pageSize: int
    total: int
