import json
from pathlib import Path

from ultralytics import YOLOE

from src.classes.beverage_cls import BEVERAGE_CONTAINER_CLASSES
from src.image_utils import process_detection_crops
from src.masking import extract_binary_masks
from src.types import Detection
from src.utils import detect_device, free_gpu_memory


def parse_detections(result) -> list[Detection]:
    """Parse YOLOE prediction result into Detection dataclasses.

    Args:
        result: YOLOE prediction result (single image)

    Returns:
        List of Detection with bbox, confidence, class info.
    """
    if result.boxes is None or len(result.boxes) == 0:
        return []

    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_id = int(box.cls[0])
        class_name = (
            BEVERAGE_CONTAINER_CLASSES[cls_id]
            if cls_id < len(BEVERAGE_CONTAINER_CLASSES)
            else str(cls_id)
        )
        detections.append(
            Detection(
                bbox=(x1, y1, x2, y2),
                confidence=float(box.conf[0]),
                class_name=class_name,
                class_id=cls_id,
            )
        )
    return detections


def detect(
    model_path: str = "models/yoloe-26l-seg.pt",
    device: str | None = None,
    conf: float = 0.25,
    imgsz: int = 1280,
    batch_size: int = 1,
) -> None:
    if device is None:
        device = detect_device()

    input_path = Path("data/images")
    if not input_path.exists():
        raise FileNotFoundError("data/images/ directory not found")

    model = YOLOE(model_path)
    model.set_classes(BEVERAGE_CONTAINER_CLASSES)

    image_paths = sorted(input_path.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError("No images found in data/images/")

    first_result = model.predict(
        source=str(image_paths[0]),
        device=device,
        conf=conf,
        imgsz=imgsz,
        augment=True,
        save=True,
        save_crop=False,
        retina_masks=False,
    )
    output_dir = Path(first_result[0].save_dir)
    crops_dir = output_dir / "crops"

    _save_result(first_result[0], output_dir, crops_dir)
    free_gpu_memory()

    for img_path in image_paths[1:]:
        results = model.predict(
            source=str(img_path),
            device=device,
            conf=conf,
            imgsz=imgsz,
            augment=True,
            save=False,
            save_crop=False,
            retina_masks=False,
        )
        _save_result(results[0], output_dir, crops_dir)
        free_gpu_memory()

    print(f"Predictions saved to: {output_dir}")


def _save_result(result, output_dir: Path, crops_dir: Path) -> None:
    img_name = Path(result.path).stem
    json_path = output_dir / f"{img_name}.json"
    preds = json.loads(result.to_json())

    with open(json_path, "w") as f:
        json.dump(preds, f, indent=2)

    if result.masks is not None:
        detections = parse_detections(result)
        masks = extract_binary_masks(result)

        process_detection_crops(
            img=result.orig_img,
            boxes=[det.bbox for det in detections],
            masks=masks,
            class_names=[det.class_name for det in detections],
            output_dir=crops_dir,
            image_name=img_name,
            exclude_other_masks=True,
        )
