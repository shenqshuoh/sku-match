#!/usr/bin/env python3
import argparse
import sys

from src.core import detect


def main():
    parser = argparse.ArgumentParser(description="YOLOv12 Detection")
    parser.add_argument(
        "--model",
        default="yolo12l",
        help="Model name (e.g., yolo12n, yolo12s, yolo12m, yolo12l, yolo12x)",
    )
    parser.add_argument("--device", default="mps", help="Device: mps, cuda, cpu")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size for inference")

    args = parser.parse_args()

    model_path = f"models/{args.model}.pt"

    try:
        detect(
            model_path=model_path,
            device=args.device,
            conf=args.conf,
            imgsz=args.imgsz,
            batch=args.batch,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
