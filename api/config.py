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

    API_KEY: str = ""  # Empty = auth disabled
    RATE_LIMIT: int = 0  # Requests per minute per IP; 0 = disabled

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
