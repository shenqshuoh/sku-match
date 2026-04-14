#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLOE

from src.classes.beverage_cls import BEVERAGE_CONTAINER_CLASSES
from src.classes.objects365_classes import OBJECTS365_CLASSES


def select_center_detection(boxes: list, img_w: int, img_h: int) -> int | None:
    if not boxes:
        return None

    cx, cy = img_w / 2, img_h / 2
    candidates = []

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            area = (x2 - x1) * (y2 - y1)
            candidates.append((i, area))

    if candidates:
        return min(candidates, key=lambda x: x[1])[0]

    min_dist = float("inf")
    min_idx = 0
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        bx, by = (x1 + x2) / 2, (y1 + y2) / 2
        dist_sq = (bx - cx) ** 2 + (by - cy) ** 2
        if dist_sq < min_dist:
            min_dist = dist_sq
            min_idx = i

    return min_idx


def apply_mask(
    img: np.ndarray,
    contour: np.ndarray | None,
    other_contours: list[np.ndarray] | None = None,
) -> np.ndarray:
    if contour is None:
        return img

    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)

    if other_contours:
        other_mask = np.zeros(img.shape[:2], np.uint8)
        for oc in other_contours:
            cv2.drawContours(other_mask, [oc], -1, 255, cv2.FILLED)
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(other_mask))

    return cv2.bitwise_and(img, img, mask=mask)


def process_raw_references(
    raw_dir: Path,
    output_dir: Path,
    model_path: str = "models/yoloe-26l-seg.pt",
    device: str = "mps",
    conf: float = 0.25,
    use_mask: bool = False,
    verbose: bool = False,
) -> None:
    model = YOLOE(model_path)
    model.set_classes(BEVERAGE_CONTAINER_CLASSES)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for sku_dir in sorted(Path(raw_dir).iterdir()):
        if not sku_dir.is_dir():
            continue

        sku_out = output_dir / sku_dir.name
        sku_out.mkdir(parents=True, exist_ok=True)

        for f in sku_out.glob("*.jpg"):
            f.unlink()

        images = (
            list(sku_dir.glob("*.jpg")) + list(sku_dir.glob("*.jpeg")) + list(sku_dir.glob("*.png"))
        )

        if verbose:
            print(f"Processing {sku_dir.name}: {len(images)} images")

        sorted_images = sorted(images)
        if sorted_images:
            results = model.predict(
                source=[str(p) for p in sorted_images],
                device=device,
                conf=conf,
                retina_masks=True,
                verbose=False,
            )
        else:
            results = []

        for idx, (img_path, result) in enumerate(zip(sorted_images, results), 1):
            if verbose:
                print(f"  {img_path.name}...", end=" ")

            if not result.boxes:
                if verbose:
                    print("no detections")
                continue

            img = cv2.cvtColor(result.orig_img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]

            boxes = [tuple(map(float, b.xyxy[0])) for b in result.boxes]
            masks = [
                result.masks.xy[i].astype(np.int32).reshape(-1, 1, 2) if result.masks else None
                for i in range(len(result.boxes))
            ]

            sel_idx = select_center_detection(boxes, w, h)
            if sel_idx is None:
                if verbose:
                    print("no center detection")
                continue

            x1, y1, x2, y2 = map(int, boxes[sel_idx])

            if use_mask and masks[sel_idx] is not None:
                other_masks = [m for i, m in enumerate(masks) if i != sel_idx and m is not None]
                crop = apply_mask(img, masks[sel_idx], other_masks)[y1:y2, x1:x2]
                if verbose:
                    print(f"masked crop ({len(boxes)} detections)")
            else:
                crop = img[y1:y2, x1:x2]
                if verbose:
                    print(f"crop ({len(boxes)} detections)")

            cv2.imwrite(str(sku_out / f"{idx:02d}.jpg"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))


def main():
    parser = argparse.ArgumentParser(description="Crop raw reference photos using YOLOE detection")
    parser.add_argument("-r", "--raw-dir", type=Path, default="data/references_raw/")
    parser.add_argument("-o", "--output-dir", type=Path, default="data/references/")
    parser.add_argument("-m", "--model", type=str, default="models/yoloe-26l-seg.pt")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--mask", action="store_true", help="Apply segmentation masking")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    if not args.raw_dir.exists():
        print(f"Error: Raw directory not found: {args.raw_dir}", file=sys.stderr)
        sys.exit(1)

    process_raw_references(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        model_path=args.model,
        device=args.device,
        conf=args.conf,
        use_mask=args.mask,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
