import json
from pathlib import Path

from ultralytics import YOLO


def detect(
    model_path: str = "models/yolo12m.pt",
    device: str | None = None,
    conf: float = 0.25,
    imgsz: int = 640,
    batch: int = 1,
) -> None:
    input_path = Path("data/images")
    if not input_path.exists():
        raise FileNotFoundError("data/images/ directory not found")

    model = YOLO(model_path)
    results = model.predict(
        source=str(input_path),
        device=device,
        conf=conf,
        imgsz=imgsz,
        batch=batch,
        classes=[39],
        augment=True,
        save=True,
        save_crop=True,
    )

    output_dir = Path(results[0].save_dir)

    for result in results:
        img_name = Path(result.path).stem
        json_path = output_dir / f"{img_name}.json"
        preds = json.loads(result.to_json())

        with open(json_path, "w") as f:
            json.dump(preds, f, indent=2)

    print(f"Predictions saved to: {output_dir}")
