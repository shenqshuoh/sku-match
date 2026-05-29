"""Image processing utilities for crop and annotation operations."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.masking import mask_background


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

        masked = mask_background(img_rgb, mask, exclude_masks=other_masks)
        crop = masked[y1:y2, x1:x2]

        if crop.size > 0:
            crop_pil = Image.fromarray(crop)
            crop_name = f"{image_name}_{i:03d}_{class_name}.jpg"
            crop_path = output_dir / crop_name
            save_crop(crop_pil, crop_name, output_dir)
            saved_paths.append(crop_path)

    return saved_paths


def _load_cjk_font(size: int = 24) -> ImageFont.FreeTypeFont:
    """Load a CJK-capable TrueType font, falling back to PIL default."""
    import glob

    candidates = glob.glob("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    candidates += glob.glob("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc")
    candidates += glob.glob("/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc")

    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue

    # Fallback: PIL default (won't render CJK, but won't crash)
    return ImageFont.load_default()


def draw_annotations(
    image: np.ndarray,
    detections: list[dict],
    output_path: Path,
) -> Path:
    """Draw bounding boxes and SKU labels on image and save.

    Uses PIL for text rendering to support CJK characters.

    Args:
        image: Original image as numpy array (RGB, HxWx3)
        detections: List of dicts with keys: bbox (list[float]), sku_name (str), match_score (float)
        output_path: Where to save the annotated image

    Returns:
        Path to the saved annotated image
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    font = _load_cjk_font(size=24)

    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        sku_name = det.get("sku_name", "unknown")
        score = det.get("match_score", 0.0)

        # Bounding box (green)
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)

        # Label background
        label = f"{sku_name} ({score:.2f})"
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        text_h = text_bbox[3] - text_bbox[1]
        draw.rectangle(
            [x1, y1 - text_h - 8, text_bbox[2], y1],
            fill=(0, 255, 0),
        )

        # Label text
        draw.text((x1, y1 - text_h - 5), label, fill=(0, 0, 0), font=font)

    # Save as JPEG with quality 85
    pil_image.save(str(output_path), "JPEG", quality=85)
    return output_path
