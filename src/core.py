import json
from pathlib import Path

from ultralytics import YOLOE

from src.classes.beverage_cls import BEVERAGE_CONTAINER_CLASSES
from src.classes.objects365_classes import OBJECTS365_CLASSES
from src.image_utils import process_detection_crops, extract_binary_masks


def detect(
    model_path: str = "models/yoloe-26l-seg.pt",
    device: str = "mps",
    conf: float = 0.25,
    imgsz: int = 1280,
    batch: int = 1,
) -> None:
    input_path = Path("data/images")
    if not input_path.exists():
        raise FileNotFoundError("data/images/ directory not found")

    model = YOLOE(model_path)
    predict_kwargs = {
        "source": str(input_path),
        "device": device,
        "conf": conf,
        "imgsz": imgsz,
        "batch": batch,
        "augment": True,
        "save": True,
        "save_crop": False,
        "retina_masks": True,
    }
    model.set_classes(BEVERAGE_CONTAINER_CLASSES)
    results = model.predict(**predict_kwargs)

    output_dir = Path(results[0].save_dir)
    crops_dir = output_dir / "crops"

    for result in results:
        img_name = Path(result.path).stem
        json_path = output_dir / f"{img_name}.json"
        preds = json.loads(result.to_json())

        with open(json_path, "w") as f:
            json.dump(preds, f, indent=2)

        if result.masks is not None:
            boxes = [box.xyxy[0].tolist() for box in result.boxes]
            masks = extract_binary_masks(result)
            class_names = [BEVERAGE_CONTAINER_CLASSES[int(box.cls[0])] for box in result.boxes]

            process_detection_crops(
                img=result.orig_img,
                boxes=boxes,
                masks=masks,
                class_names=class_names,
                output_dir=crops_dir,
                image_name=img_name,
                exclude_other_masks=True,
            )

    print(f"Predictions saved to: {output_dir}")
