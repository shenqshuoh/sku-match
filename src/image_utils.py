"""Image processing utilities for masking and crop operations."""

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


def normalize_mask(mask: np.ndarray) -> np.ndarray:
    """Normalize mask to uint8 with values 0 or 255.

    Args:
        mask: Binary mask (H, W) with values in [0, 1] or [0, 255]

    Returns:
        Normalized uint8 mask
    """
    if mask.max() <= 1:
        return (mask * 255).astype(np.uint8)
    return mask.astype(np.uint8)


def isolate_object(
    img: np.ndarray,
    mask: np.ndarray | None,
    other_masks: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Isolate an object from the image using its binary mask.

    Args:
        img: Input image (H, W, 3)
        mask: Binary mask (H, W) with values 0 or 1/255, or None
        other_masks: List of other binary masks to exclude

    Returns:
        Isolated object with black background
    """
    if mask is None:
        return img

    # Ensure mask is binary (0 or 255)
    mask = normalize_mask(mask)

    # Exclude other masks if provided
    if other_masks:
        other_mask = np.zeros_like(mask)
        for om in other_masks:
            other_mask = cv2.bitwise_or(other_mask, normalize_mask(om))
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(other_mask))

    mask3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
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
        masks: List of binary masks (H, W) or None
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


def extract_binary_masks(result) -> list[np.ndarray | None]:
    """Extract binary masks from YOLOE result.

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
        # Convert tensor to numpy binary mask
        mask = mask_tensor.cpu().numpy()

        # Resize to original image dimensions if needed
        if mask.shape != (img_h, img_w):
            mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_LINEAR)

        # Ensure binary (0 or 1)
        mask = (mask > 0.5).astype(np.uint8)
        masks.append(mask)

    return masks


def draw_annotations(
    image: np.ndarray,
    detections: list[dict],
    output_path: Path,
) -> Path:
    """Draw bounding boxes and SKU labels on image and save.

    Args:
        image: Original image as numpy array (RGB, HxWx3)
        detections: List of dicts with keys: bbox (list[float]), sku_name (str), match_score (float)
        output_path: Where to save the annotated image

    Returns:
        Path to the saved annotated image
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    annotated = image.copy()

    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        sku_name = det.get("sku_name", "unknown")
        score = det.get("match_score", 0.0)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"{sku_name} ({score:.2f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        cv2.rectangle(
            annotated,
            (x1, y1 - text_h - baseline - 6),
            (x1 + text_w, y1),
            (0, 255, 0),
            -1,
        )

        cv2.putText(
            annotated,
            label,
            (x1, y1 - baseline - 3),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
        )

    # Save as JPEG with quality 85 (convert RGB→BGR for OpenCV imwrite)
    cv2.imwrite(str(output_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return output_path
