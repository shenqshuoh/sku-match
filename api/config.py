from pydantic_settings import BaseSettings, SettingsConfigDict

from src.embedder import DINOv2Variant


class Settings(BaseSettings):
    DATA_DIR: str = "data"
    MODELS_DIR: str = "models"
    CHROMA_PERSIST_DIR: str = "chroma_data"
    RESULTS_DIR: str = "results"
    DATABASE_URL: str = "sqlite+aiosqlite:///./sku_match.db"

    DET_MODEL: str = "models/yoloe-26l-seg.pt"
    EMB_MODEL: DINOv2Variant = "dinov2_vits14"
    DEVICE: str | None = None

    DET_CONF: float = 0.25
    IMGSZ: int = 1280
    MATCH_CONF: float = 0.5
    CONCENTRATION_TOPK: int = 10
    USE_ONNX: bool = False

    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False

    DOWNLOAD_TIMEOUT: int = 30
    RESULTS_MAX_AGE_HOURS: int = 24  # Annotated images older than this are cleaned up

    API_KEY: str = ""  # Empty = auth disabled
    RATE_LIMIT: int = 0  # Requests per minute per IP; 0 = disabled

    QINIU_TOKEN_URL: str = "http://172.16.88.119:12001/api/qiniu/token/vr"
    QINIU_UPLOAD_URL: str = "https://upload-z2.qiniup.com"
    QINIU_DOMAIN: str = "https://vr.jihaihotpot.com/"
    QINIU_IOVIP_URL: str = "http://iovip-z2.qiniuio.com"
    QINIU_KEY_PREFIX: str = "sku-match/"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
