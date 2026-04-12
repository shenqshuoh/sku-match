"""Utility for converting YOLOE models to LVIS beverage detection models.

Based on GitHub issues:
- #22630: YOLO11 Objects365 pretrained model request
- #22744: YOLOE seg→det conversion pattern

YOLOE (YOLO Enterprise) models support text prompts for open-vocabulary detection.
This utility converts a YOLOE segmentation model to a standard detection model
predicting LVIS drink, beverage, and fluid container categories.
"""

import argparse

from ultralytics import YOLOE

from src.classes.beverage_cls import BEVERAGE_CONTAINER_CLASSES


def export_yoloe_to_lvis(model_path: str):
    """Export a YOLOE segmentation model to LVIS beverage detection model.

    Args:
        model: path to model file
    """

    model = YOLOE(model_path)
    model.set_classes(BEVERAGE_CONTAINER_CLASSES)
    model.export(format="coreml", imgsz=1280, dynamic=True, batch=8, device="mps")

    return


def main():
    parser = argparse.ArgumentParser(description="Export YOLOE models for LVIS beverage detection")
    parser.add_argument("model", help="path to model file")

    args = parser.parse_args()

    model_path = f"models/{args.model}"
    export_yoloe_to_lvis(model_path)


if __name__ == "__main__":
    main()
