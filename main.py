#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

from shutil import copy2

from src.core import detect
from src.matcher import SKUMatcher
from src.types import SKUMatch


def get_next_match_dir(base_dir: Path = Path("runs/match")) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)

    n = 1
    while (base_dir / f"match{n:02d}").exists():
        n += 1

    return base_dir / f"match{n:02d}"


def run_detection(args):
    try:
        detect(
            model_path=args.det_model,
            device=args.device,
            conf=args.conf,
            imgsz=args.imgsz,
            batch=args.batch,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def run_matching(args):
    if not args.index.exists():
        print(f"Error: Index directory not found: {args.index}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading index from: {args.index}")
    print(f"Detection model: {args.det_model}")
    print(f"Embedding model: {args.emb_model}")
    print(f"Segmentation batch: {args.seg_batch}")
    print(f"Embedding batch: {args.emb_batch}")

    matcher = SKUMatcher.from_index_dir(
        index_dir=args.index,
        det_model=args.det_model,
        emb_model=args.emb_model,
        device=args.device,
        confidence_threshold=args.match_conf,
    )

    input_path = Path("data/images")
    if not input_path.exists():
        print("Error: data/images/ directory not found", file=sys.stderr)
        sys.exit(1)

    image_paths = sorted(input_path.glob("*.jpg"))
    if not image_paths:
        print("No images found in data/images/", file=sys.stderr)
        sys.exit(1)

    output_dir = get_next_match_dir()
    output_dirs = [output_dir / p.stem / "crops" for p in image_paths]

    for d in output_dirs:
        d.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing {len(image_paths)} images...")

    all_results = matcher.match_images(
        image_paths=image_paths,
        output_dirs=output_dirs,
        seg_batch=args.seg_batch,
        emb_batch=args.emb_batch,
    )

    for img_path, img_results in zip(image_paths, all_results):
        img_dir = output_dir / img_path.stem
        _save_image_results(img_path, img_dir, img_results, args.match_conf)
        print(f"{img_path.name}: {len(img_results)} detections")

    print(f"\nAll results saved to: {output_dir}")


def _save_image_results(
    img_path: Path,
    img_dir: Path,
    img_results: list[SKUMatch],
    match_conf: float,
) -> None:
    save_results = []
    counts: dict[str, int] = {}

    for match in img_results:
        save_sku_id = "unk" if match.match_score < match_conf else match.sku_id
        save_results.append(
            {
                "bbox": match.detection.bbox,
                "class_name": match.detection.class_name,
                "class_id": match.detection.class_id,
                "detection_conf": match.detection.confidence,
                "sku_id": save_sku_id,
                "match_score": match.match_score,
            }
        )
        counts[save_sku_id] = counts.get(save_sku_id, 0) + 1

    output_data = {
        "detections": save_results,
        "counts": counts,
    }

    results_path = img_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(output_data, f, indent=2)

    copy2(img_path, img_dir / "original.jpg")


def main():
    parser = argparse.ArgumentParser(description="YOLOE Detection with DINOv2 SKU Matching")
    parser.add_argument(
        "--detection-only",
        action="store_true",
        help="Run detection only (no SKU matching)",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default="index/",
        help="Index directory for SKU matching (default: index/)",
    )
    parser.add_argument(
        "--det-model",
        type=str,
        default="models/yoloe-26l-seg.pt",
        help="YOLOE detection model path (default: models/yoloe-26l-seg.pt)",
    )
    parser.add_argument(
        "--emb-model",
        type=str,
        choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14"],
        default="dinov2_vitb14",
        help="DINOv2 embedding model variant (default: dinov2_vitb14)",
    )
    parser.add_argument(
        "--device",
        default="mps",
        help="Device: mps, cuda, cpu",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold",
    )
    parser.add_argument(
        "--match-conf",
        type=float,
        default=1.2,
        help="Minimum match score for SKU assignment (default: 1.2)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1080,
        help="Batch size",
    )
    parser.add_argument(
        "--seg-batch",
        type=int,
        default=8,
        help="YOLOE segmentation batch size (default: 16)",
    )
    parser.add_argument(
        "--emb-batch",
        type=int,
        default=8,
        help="DINOv2 embedding batch size (default: 16)",
    )

    args = parser.parse_args()

    if args.detection_only:
        run_detection(args)
    else:
        run_matching(args)


if __name__ == "__main__":
    main()
