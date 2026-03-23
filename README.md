# beverage-cashier

YOLOv12-based bottle detection on Apple Silicon.

## Overview

- **Model**: YOLOv12 (configurable: n, s, m, l, x)
- **Target**: Bottle class only (COCO class 39)
- **Features**: Test Time Augmentation (TTA) enabled
- **Device**: Optimized for MPS (Apple Silicon)

## Setup

```bash
cd beverage-cashier

uv venv
source .venv/bin/activate
uv pip install -e .
```

Models auto-download on first run to `models/`.

## Usage

```bash
# Default detection
python main.py

# Custom settings
python main.py --model yolo12x --conf 0.5 --batch 4
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | yolo12m | Model: yolo12n, yolo12s, yolo12m, yolo12l, yolo12x |
| `--device` | mps | Device: mps (Apple Silicon), cuda, cpu |
| `--conf` | 0.25 | Confidence threshold |
| `--imgsz` | 640 | Input image size |
| `--batch` | 1 | Batch size |

## Output

```
runs/detect/predict/
├── *.jpg                 # Annotated images
├── *.json                # Detection data (one per image)
└── crops/                # Cropped detections
```

### Detection JSON (000000000183.json)

```json
[
  {
    "name": "bottle",
    "class": 39,
    "confidence": 0.891,
    "box": {"x1": 75.8, "y1": 16.4, "x2": 143.7, "y2": 162.7}
  }
]
```

## Configuration

Edit `src/core.py` to modify:
- **Input path**: Change `data/images`
- **Target class**: Change `classes=[39]`
- **Augmentation**: Set `augment=False`
- **Crops**: Set `save_crop=False`

## Testing

```bash
python tests/test_detection.py
```
