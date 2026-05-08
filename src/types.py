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
    sku_name: str
    image_path: Path
    embedding: np.ndarray


@dataclass(frozen=True)
class SKUMatch:
    detection: Detection
    sku_id: str
    sku_name: str
    match_score: float
    match_concentration: float = 0.0
    top2_ranks: tuple[int, int] = (0, 0)
    sku_distribution: dict[str, float] | None = None
    crop_path: Path | None = None
