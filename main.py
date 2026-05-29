#!/usr/bin/env python3
import argparse
import json
import sys
import time
from pathlib import Path

from shutil import copy2

from src.core import detect
from src.matcher import SKUMatcher
from src.reference_processor import ReferenceProcessor
from src.types import SKUMatch
from src.utils import configure_ultralytics_weights


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
            batch_size=args.batch,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def run_matching(args):
    args.index.mkdir(parents=True, exist_ok=True)

    print(f"Loading index from: {args.index}")
    print(f"Detection model: {args.det_model}")
    print(f"Embedding model: {args.emb_model}")
    print(f"Batch size: {args.batch}")

    matcher = SKUMatcher.from_index_dir(
        index_dir=args.index,
        det_model=args.det_model,
        emb_model=args.emb_model,
        device=args.device,
        confidence_threshold=args.match_conf,
        concentration_threshold=args.match_concentration,
        concentration_topk=args.match_concentration_topk,
        det_conf=args.conf,
        swap_models=args.swap,
        use_onnx=args.onnx,
    )

    # Auto-build index if empty (crop + embed from data/references/)
    if matcher.indexer.collection.count() == 0:
        ref_dir = Path("data/references")
        if not ref_dir.exists() or not any(ref_dir.iterdir()):
            print("Error: Index is empty and no reference images found in data/references/", file=sys.stderr)
            sys.exit(1)

        print(f"\nIndex is empty — auto-building from {ref_dir} (crop + embed)...")
        from src.utils import detect_device
        device = args.device or detect_device()
        processor = ReferenceProcessor(
            detector=matcher.detector,
            embedder=matcher.embedder,
            indexer=matcher.indexer,
            device=device,
        )
        count = processor.build_from_directory(ref_dir, batch_size=args.batch)
        if count == 0:
            print("Error: Failed to build index — no reference images processed", file=sys.stderr)
            sys.exit(1)
        print(f"Index built: {count} reference images\n")

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

    total_start = time.perf_counter()

    all_results = matcher.match_images(
        image_paths=image_paths,
        output_dirs=output_dirs,
        batch_size=args.batch,
        verbose=args.match_verbose,
    )

    for img_path, (img_results, elapsed) in zip(image_paths, all_results):
        img_dir = output_dir / img_path.stem
        _save_image_results(img_path, img_dir, img_results, args.match_conf, args.match_concentration)
        print(f"{img_path.name}: {len(img_results)} detections ({elapsed:.2f}s)")

    total_elapsed = time.perf_counter() - total_start
    print(f"\nAll results saved to: {output_dir}")
    print(f"Total time: {total_elapsed:.2f}s")


def _save_image_results(
    img_path: Path,
    img_dir: Path,
    img_results: list[SKUMatch],
    match_conf: float,
    match_concentration: float,
) -> None:
    save_results = []
    counts: dict[str, int] = {}

    for match in img_results:
        is_match = match.match_score >= match_conf and match.match_concentration >= match_concentration
        save_sku_id = match.sku_id if is_match else "unk"
        save_results.append(
            {
                "bbox": match.detection.bbox,
                "class_name": match.detection.class_name,
                "class_id": match.detection.class_id,
                "detection_conf": match.detection.confidence,
                "sku_id": save_sku_id,
                "sku_name": match.sku_name,
                "match_score": match.match_score,
                "match_concentration": match.match_concentration,
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
        default="chroma_data/",
        help="Index directory for SKU matching (default: chroma_data/)",
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
        default="dinov2_vits14",
        help="DINOv2 embedding model variant (default: dinov2_vits14)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device: cuda, mps, cpu (default: auto-detect)",
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
        default=0.5,
        help="Minimum match confidence (softmax probability) for SKU assignment (default: 0.5)",
    )
    parser.add_argument(
        "--match-concentration",
        type=float,
        default=0.0,
        help="Minimum concentration score (top-1 share of top-K mass) for confident match (default: 0.0, disabled)",
    )
    parser.add_argument(
        "--match-concentration-topk",
        type=int,
        default=10,
        help="Number of top SKUs to consider for concentration score (default: 10)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Input image size",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Batch size for detection and embedding (default: 1)",
    )
    parser.add_argument(
        "--match-verbose",
        action="store_true",
        help="Show per-image timing and match score distribution",
    )
    parser.add_argument(
        "--swap",
        action="store_true",
        help="Enable model swapping: unload detector from GPU after detection, "
        "then load embedder to GPU. Saves VRAM on low-memory GPUs (~2GB).",
    )
    parser.add_argument(
        "--onnx",
        action="store_true",
        help="Use ONNX Runtime for DINOv2 embedding instead of PyTorch (requires .onnx model file).",
    )

    args = parser.parse_args()

    # Ensure ultralytics finds local model weights (mobileclip2_b.ts) without GitHub download
    configure_ultralytics_weights()

    if args.detection_only:
        run_detection(args)
    else:
        run_matching(args)


if __name__ == "__main__":
    main()
