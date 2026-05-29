"""Configurable background masking for detected objects.

Masks background pixels with a neutral color (default: ImageNet mean)
to improve embedding quality by reducing background noise.

Used at two stages:
  1. Between YOLOE detection and DINOv2 SKU matching (embedding crops)
  2. When adding new SKU reference images (reference preprocessing)
"""

import cv2
import numpy as np

# ImageNet mean values — neutral grey that minimizes embedding artefacts
IMAGENET_MEAN_RGB = (124, 116, 104)


def normalize_mask(mask: np.ndarray) -> np.ndarray:
    """Normalize mask to uint8 binary (0 or 255).

    Args:
        mask: Binary mask (H, W) with values in [0, 1] or [0, 255]

    Returns:
        Normalized uint8 mask (0 or 255)
    """
    if mask.max() <= 1:
        return (mask * 255).astype(np.uint8)
    return mask.astype(np.uint8)


def extract_binary_masks(result) -> list[np.ndarray | None]:
    """Extract binary masks from YOLOE prediction result.

    Uses masks.data for full binary mask instead of masks.xy (simplified contour).
    This ensures the complete instance is masked, not a smaller polygon approximation.

    Args:
        result: YOLOE prediction result

    Returns:
        List of binary masks (H, W) or None for each detection
    """
    if result.masks is None:
        return [None] * len(result.boxes)

    masks = []
    img_h, img_w = result.orig_img.shape[:2]

    for mask_tensor in result.masks.data:
        mask = mask_tensor.cpu().numpy()

        if mask.shape != (img_h, img_w):
            mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_LINEAR)

        mask = (mask > 0.5).astype(np.uint8)
        masks.append(mask)

    return masks


def mask_background(
    image: np.ndarray,
    mask: np.ndarray | None,
    background_color: tuple[int, int, int] = IMAGENET_MEAN_RGB,
    exclude_masks: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Replace background pixels with a neutral color using binary mask.

    Args:
        image: RGB image (H, W, 3)
        mask: Binary mask (H, W), nonzero = object, zero = background. None = no masking.
        background_color: RGB color for background pixels (default: ImageNet mean).
        exclude_masks: Other object masks to also treat as background.

    Returns:
        Image with background replaced by background_color.
    """
    if mask is None:
        return image

    mask = normalize_mask(mask)

    # Exclude overlapping regions from other detections
    if exclude_masks:
        other_mask = np.zeros_like(mask)
        for om in exclude_masks:
            other_mask = cv2.bitwise_or(other_mask, normalize_mask(om))
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(other_mask))

    # 3-channel float mask: 1.0 = object (keep), 0.0 = background (replace)
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0

    # Blend: object pixels stay, background pixels become background_color
    background = np.full_like(image, background_color, dtype=np.uint8)
    result = image.astype(np.float32) * mask_3ch + background.astype(np.float32) * (1.0 - mask_3ch)
    return result.astype(np.uint8)
