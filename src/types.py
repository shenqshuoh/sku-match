from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Detection:
    bbox: tuple[float, float, float, float]
    confidence: float
    class_name: str
    class_id: int
    mask: np.ndarray | None = None


@dataclass(frozen=True)
class SKUReference:
    sku_id: str
    image_path: Path
    embedding: np.ndarray


@dataclass(frozen=True)
class SKUMatch:
    detection: Detection
    sku_id: str
    match_score: float
    crop_path: Path | None = None
