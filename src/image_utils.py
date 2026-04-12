"""Image processing utilities for masking and crop operations."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def isolate_object(
    img: np.ndarray,
    mask: np.ndarray | None,
    other_masks: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Isolate an object from the image using its mask.

    Args:
        img: Input image (H, W, 3)
        mask: Object mask contour (N, 1, 2) or None
        other_masks: List of other object masks to exclude

    Returns:
        Isolated object with black background
    """
    if mask is None:
        return img

    b_mask = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(b_mask, [mask], -1, 255, cv2.FILLED)

    if other_masks:
        other_mask = np.zeros(img.shape[:2], np.uint8)
        for om in other_masks:
            cv2.drawContours(other_mask, [om], -1, 255, cv2.FILLED)
        b_mask = cv2.bitwise_and(b_mask, cv2.bitwise_not(other_mask))

    mask3ch = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR)
    isolated = cv2.bitwise_and(mask3ch, img)

    return isolated


def save_crop(
    crop: Image.Image,
    name: str,
    output_path: Path,
) -> None:
    """Save a cropped image.

    Args:
        crop: PIL Image to save
        name: Filename for the crop
        output_path: Full path to save to
    """
    crop.save(output_path / name)


def process_detection_crops(
    img: np.ndarray,
    boxes: list,
    masks: list[np.ndarray | None],
    class_names: list[str],
    output_dir: Path,
    image_name: str,
    exclude_other_masks: bool = True,
) -> list[Path]:
    """Process all detections and save masked crops.

    Args:
        img: Original image (BGR from YOLOE)
        boxes: List of bounding boxes [(x1, y1, x2, y2), ...]
        masks: List of mask contours or None
        class_names: List of class names for each detection
        output_dir: Directory to save crops
        image_name: Base name of the image
        exclude_other_masks: Whether to exclude overlapping masks

    Returns:
        List of saved crop paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    saved_paths = []

    for i, (box, mask, class_name) in enumerate(zip(boxes, masks, class_names)):
        x1, y1, x2, y2 = map(int, box)

        if exclude_other_masks and mask is not None:
            other_masks = [m for j, m in enumerate(masks) if j != i and m is not None]
        else:
            other_masks = None

        isolated = isolate_object(img_rgb, mask, other_masks)
        crop = isolated[y1:y2, x1:x2]

        if crop.size > 0:
            crop_pil = Image.fromarray(crop)
            crop_name = f"{image_name}_{i:03d}_{class_name}.jpg"
            crop_path = output_dir / crop_name
            save_crop(crop_pil, crop_name, output_dir)
            saved_paths.append(crop_path)

    return saved_paths
