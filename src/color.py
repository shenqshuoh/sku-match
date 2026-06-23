"""HSV color-histogram descriptor for score-level SKU re-ranking.

DINOv2 is trained with aggressive color augmentation and is effectively
color-blind. This descriptor recovers the color signal as a compact,
L2-normalized vector used for cosine similarity at re-rank time.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def extract_color_descriptor(
    image: NDArray[np.uint8],
    n_bins: int = 16,
) -> NDArray[np.float32]:
    """Compact HSV color descriptor from a raw crop.

    Args:
        image: RGB uint8 ndarray, shape (H, W, 3). Codebase convention is RGB
            (PIL / numpy). Conversion to HSV is performed internally; do NOT
            pass BGR.
        n_bins: histogram bins per HSV channel. 16 → descriptor length
            3*16 + 1 = 49.

    Returns:
        (3*n_bins + 1,) float32, L2-normalized vector:
        [hue_hist(n_bins), sat_hist(n_bins), val_hist(n_bins), dominant_hue(1)].

    Raises:
        ValueError: if image is not a 3-channel uint8 array, or n_bins < 1.
    """
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("image must be a 3-channel uint8 array of shape (H, W, 3)")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")

    # Convert RGB → HSV. OpenCV HSV ranges: H∈[0,180], S∈[0,256], V∈[0,256]
    # (identical to the BGR2HSV output ranges).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # 1D histograms per channel, independently L2-normalized.
    h_hist = cv2.calcHist([hsv], [0], None, [n_bins], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [n_bins], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [n_bins], [0, 256]).flatten()

    h_hist = h_hist / (np.linalg.norm(h_hist) + 1e-8)
    s_hist = s_hist / (np.linalg.norm(s_hist) + 1e-8)
    v_hist = v_hist / (np.linalg.norm(v_hist) + 1e-8)

    # Dominant hue: saturation-weighted circular mean over the hue wheel.
    # OpenCV stores hue as 0-180; scale maps the full wheel to [0, 2π).
    h_rad = hsv[:, :, 0].astype(np.float64) * (np.pi / 90.0)
    s_weight = hsv[:, :, 1].astype(np.float64) / 255.0
    dominant_hue = np.arctan2(
        np.sum(np.sin(h_rad) * s_weight),
        np.sum(np.cos(h_rad) * s_weight),
    ) % (2 * np.pi)
    dominant_hue_norm = dominant_hue / (2 * np.pi)

    descriptor = np.concatenate([h_hist, s_hist, v_hist, [dominant_hue_norm]])
    return (descriptor / (np.linalg.norm(descriptor) + 1e-8)).astype(np.float32)


def color_descriptor_dim(n_bins: int) -> int:
    """Length of the color descriptor produced for a given bin count.

    The descriptor is ``[hue(n_bins), sat(n_bins), val(n_bins), dominant_hue(1)]``,
    i.e. ``3 * n_bins + 1``. Centralising the formula here keeps the indexer
    (which records the dimension) and the re-rank path (which validates it) in
    sync with ``extract_color_descriptor``.

    Raises:
        ValueError: if n_bins < 1.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    return 3 * n_bins + 1
