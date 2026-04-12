# beverage-cashier

YOLOE-based bottle & can detection with DINOv2 SKU matching for Apple Silicon (MPS).

## Overview

- **Detection**: YOLOE (You Only Look Once Enterprise) with open-vocabulary detection
- **SKU Matching**: DINOv2 embeddings + top-2 similarity search for product identification
- **Target**: Bottle & Canned (Objects365 classes 0, 1), or all 365 classes with `--all-cls`
- **Device**: Optimized for MPS (Apple Silicon GPU acceleration)

## Setup

Requires Python 3.10+ and [uv](https://github.com/astral-sh/uv):

```bash
# Create virtual environment and install dependencies
uv sync
source .venv/bin/activate
```

Models auto-download on first run to `models/` and `~/.cache/torch/hub/`.

## Usage

### SKU Matching (Default Mode)

Match detected objects against a reference gallery of SKU images:

```bash
# 1. Prepare reference images in data/references/{sku_id}/*.jpg
# Example: data/references/coke_001/01.jpg, data/references/coke_001/02.jpg

# 2. Build the reference index (one-time setup)
uv run scripts/build_index.py -r data/references/ -o index/ -m dinov2_vitb14

# 3. Run detection with SKU matching (default mode)
uv run main.py

# With custom models
uv run main.py --det-model models/yoloe-26l-seg.pt --emb-model dinov2_vitb14 --match-conf 1.4
```

### Detection Only Mode

Run detection without SKU matching:

```bash
uv run main.py --detection-only --det-model models/yoloe-26l-seg.pt --conf 0.5
```

### Preparing Reference Photos from Raw Images

If you have raw reference photos with background clutter, use `crop_reference.py` to automatically extract bottle/can crops:

```bash
# 1. Place raw photos in data/references_raw/{sku_id}/*.jpg
# Example: data/references_raw/coke_330ml/raw_001.jpg

# 2. Auto-crop using YOLOE detection (saves largest bottle/can crop)
uv run scripts/crop_reference.py -r data/references_raw/ -o data/references/

# 3. Build index and run matching
uv run scripts/build_index.py -r data/references/ -o index/
uv run main.py
```

### Build Index with Different Embedding Models

```bash
# Small model (faster, lower accuracy): 384-dim embeddings
uv run scripts/build_index.py -r data/references/ -o index/ -m dinov2_vits14

# Base model (balanced): 768-dim embeddings
uv run scripts/build_index.py -r data/references/ -o index/ -m dinov2_vitb14

# Large model (slower, higher accuracy): 1024-dim embeddings
uv run scripts/build_index.py -r data/references/ -o index/ -m dinov2_vitl14
```

### CLI Options

**main.py:**

| Option | Default | Description |
|--------|---------|-------------|
| `--detection-only` | - | Run detection without SKU matching |
| `--index` | index/ | Index directory for SKU matching |
| `--det-model` | models/yoloe-26l-seg.pt | YOLOE detection model path |
| `--emb-model` | dinov2_vitb14 | DINOv2 variant: dinov2_vits14/vitb14/vitl14 |
| `--device` | mps | Device: mps, cuda, cpu |
| `--conf` | 0.25 | Detection confidence threshold |
| `--match-conf` | 1.4 | Minimum match score (sum of top 2 similarities) |
| `--imgsz` | 640 | Input image size |
| `--batch` | 16 | Batch size |
| `--all-cls` | - | Detect all 365 Objects365 classes |

**build_index.py:**

| Option | Default | Description |
|--------|---------|-------------|
| `-r, --reference-dir` | data/references/ | Reference images directory |
| `-o, --output-dir` | index/ | Output directory for index |
| `-m, --model` | dinov2_vitb14 | Embedding model variant |
| `--device` | mps | Device for inference |

**crop_reference.py:**

| Option | Default | Description |
|--------|---------|-------------|
| `-r, --raw-dir` | data/references_raw/ | Raw reference photos directory |
| `-o, --output-dir` | data/references/ | Output directory for cropped refs |
| `-m, --model` | models/yoloe-26l-seg.pt | YOLOE model for detection |
| `--device` | mps | Device for inference |
| `--conf` | 0.25 | Detection confidence threshold |

## Project Structure

```
beverage-cashier/
├── main.py                   # CLI entry point
├── pyproject.toml            # Project config
├── src/
│   ├── __init__.py           # Package init
│   ├── core.py               # YOLOE detection logic
│   ├── embedder.py           # DINOv2 embedding wrapper
│   ├── indexer.py            # SKU index management
│   ├── matcher.py            # Detection + SKU matching pipeline
│   ├── types.py              # Dataclasses
│   ├── objects365_classes.py # 365 class labels
│   └── yoloe2o365.py         # CoreML export utility
├── scripts/
│   ├── build_index.py        # Build SKU reference index
│   └── crop_reference.py     # Crop raw reference photos
├── tests/
│   └── test_detection.py     # Smoke test
├── data/
│   ├── images/               # Input images for detection
│   ├── references/           # SKU reference images (sku_id/*.jpg)
│   └── references_raw/       # Raw reference photos for cropping
├── models/                   # YOLOE weights (.pt, .mlpackage)
├── index/                    # Generated SKU index
│   ├── embeddings_{model}.npy
│   └── metadata_{model}.json
└── runs/                     # Output directory
    ├── match/                # SKU matching results
    │   ├── match01/          # First run results
    │   │   ├── crops/        # All segment crops with SKU names
    │   │   └── match_results.json
    │   ├── match02/          # Second run results
    │   └── ...
    └── detect/predict*/      # Detection-only results
```

## Reference Image Organization

### Cropped References (for matching)

Organize cropped reference photos by SKU:

```
data/references/
├── coke_330ml/
│   ├── 01.jpg
│   ├── 02.jpg
│   └── 03.jpg
├── pepsi_330ml/
│   ├── 01.jpg
│   └── 02.jpg
└── sprite_500ml/
    ├── 01.jpg
    ├── 02.jpg
    └── 03.jpg
```

- Each SKU gets its own subdirectory
- Multiple angles per SKU improve matching accuracy
- Supported formats: .jpg, .jpeg, .png

### Raw References (for pre-processing)

Raw photos with background clutter:

```
data/references_raw/
├── coke_330ml/
│   ├── raw_photo_01.jpg
│   └── raw_photo_02.jpg
└── pepsi_330ml/
    ├── raw_photo_01.jpg
    └── raw_photo_02.jpg
```

Process with `crop_reference.py` to extract bottle/can crops automatically.

## SKU Matching Algorithm

1. **Detection**: YOLOE detects bottles/cans in the input image
2. **Embedding**: DINOv2 extracts visual features from each crop (384/768/1024-dim)
3. **Matching**: Top-2 similarity search against reference embeddings
   - For each SKU, find the 2 most similar reference embeddings
   - Score = sum of top 2 cosine similarities
   - Best matching SKU wins

## Output

### SKU Matching Mode

Results saved to incrementing directories (`runs/match/match01/`, `runs/match/match02/`, etc.):

```
runs/match/match01/
├── match_results.json      # JSON results for all images
└── crops/                  # All segment crops with SKU names
    ├── image_001_001_coke_330ml.jpg
    ├── image_001_002_pepsi_330ml.jpg
    └── image_002_001_coke_330ml.jpg
```

Each run creates a new `matchXX` directory, preserving previous results.

**match_results.json format:**
```json
{
  "scene_001.jpg": [
    {
      "bbox": [100.5, 200.3, 150.8, 380.2],
      "class_name": "Bottle",
      "class_id": 0,
      "detection_conf": 0.89,
      "sku_id": "coke_330ml",
      "match_score": 1.82
    }
  ]
}
```

### Detection Only Mode

Results saved to `runs/detect/predict*/`:

```
runs/detect/predict/
├── *.jpg                 # Annotated images
├── *.json                # Detection data
└── crops/                # Cropped detections
    ├── Bottle/
    └── Canned/
```

## Export to CoreML

Convert YOLOE models to CoreML format for iOS deployment:

```bash
uv run python src/yoloe2o365.py yoloe-26l-seg.pt
```

## Testing

```bash
# Run smoke tests
uv run tests/test_detection.py
```

## Notes

- **First run**: Downloads YOLOE and DINOv2 models (~400MB total)
- **MPS compilation**: First inference is slow due to model warmup
- **SSL**: Includes SSL context override for macOS compatibility
- **Index files**: Embeddings and metadata are model-specific (e.g., `embeddings_dinov2_vitb14.npy`)
- **Rebuilding index**: Run `build_index.py` whenever reference photos change or when switching embedding models
- **Match confidence**: Default threshold is 1.4 (sum of top 2 similarities, range 0-2)
