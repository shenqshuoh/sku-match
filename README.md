# sku-match

YOLOE-based bottle & can detection with DINOv2 SKU matching. Supports both CLI and FastAPI REST API modes.

## Overview

- **Detection**: YOLOE open-vocabulary detection for beverage containers (7 classes)
- **SKU Matching**: DINOv2 visual embeddings with softmax probability scoring
- **Dual Mode**: CLI for batch processing, FastAPI REST API for integration
- **Vector Store**: Chroma-backed incremental index — no full rebuild on SKU changes
- **Cloud Storage**: Qiniu CDN upload for annotated result images

## Features

- Open-vocabulary beverage detection (7 container classes: Bottle, Canned, etc.)
- DINOv2 visual similarity matching with softmax probability scoring (0–1 range)
- Chroma-backed incremental vector index (add/delete SKUs without full rebuild)
- FastAPI REST API with SKU CRUD, recognition, and training lifecycle
- CJK text rendering in annotated images
- Qiniu CDN upload for result images
- FP16 inference on CUDA (~1.8x speedup)
- Auto device detection (cuda → mps → cpu)

## Quick Start

See [docs/SETUP.md](docs/SETUP.md) for detailed setup instructions.

```bash
# 1. Clone and set up environment
git clone <repo-url> && cd sku-match
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 2. Place model weights in models/
#    - YOLOE: models/yoloe-26l-seg.pt
#    - DINOv2: models/dinov2_vits14_pretrain.pth (vendored)

# 3. Start API server
python -m api.app

# 4. Health check
curl localhost:8000/health
```

See [docs/API_GUIDE.md](docs/API_GUIDE.md) for API usage and endpoint reference.

## CLI Usage

### SKU Matching (Default Mode)

```bash
# 1. Prepare reference images in data/references/{sku_id}/*.jpg

# 2. Build the reference index (one-time setup)
python scripts/build_index.py -r data/references/ -o chroma_data/ -m dinov2_vits14

# 3. Run detection with SKU matching
python main.py

# With custom options
python main.py --conf 0.3 --match-conf 0.5 --match-concentration 0.0 --match-verbose

# With custom models
python main.py --det-model models/yoloe-26l-seg.pt --emb-model dinov2_vitb14
```

### Detection Only Mode

```bash
python main.py --detection-only --conf 0.5
```

### Preparing Reference Photos from Raw Images

```bash
# 1. Place raw photos in data/references_raw/{sku_id}/*.jpg

# 2. Auto-crop using YOLOE detection
python scripts/crop_reference.py -r data/references_raw/ -o data/references/

# 3. Build index and run matching
python scripts/build_index.py -r data/references/ -o chroma_data/
python main.py
```

### Build Index with Different Embedding Models

```bash
# Small model (faster): 384-dim embeddings
python scripts/build_index.py -r data/references/ -o chroma_data/ -m dinov2_vits14

# Base model (balanced): 768-dim embeddings
python scripts/build_index.py -r data/references/ -o chroma_data/ -m dinov2_vitb14

# Large model (higher accuracy): 1024-dim embeddings
python scripts/build_index.py -r data/references/ -o chroma_data/ -m dinov2_vitl14
```

### CLI Options

**main.py:**

| Option | Default | Description |
|--------|---------|-------------|
| `--detection-only` | - | Run detection without SKU matching |
| `--index` | chroma_data/ | Chroma index directory |
| `--det-model` | models/yoloe-26l-seg.pt | YOLOE detection model path |
| `--emb-model` | dinov2_vits14 | DINOv2 variant: dinov2_vits14/vitb14/vitl14 |
| `--device` | auto | Device: cuda, mps, or cpu |
| `--conf` | 0.25 | Detection confidence threshold |
| `--match-conf` | 0.5 | SKU match probability threshold (0–1) |
| `--match-concentration` | 0.0 | Concentration score threshold |
| `--imgsz` | 1280 | Inference image size |
| `--batch` | 16 | Batch size |
| `--all-cls` | - | Detect all 365 Objects365 classes |
| `--swap` | - | Swap to detection-only model |
| `--onnx` | - | Use ONNX backend for DINOv2 |

**build_index.py:**

| Option | Default | Description |
|--------|---------|-------------|
| `-r, --reference-dir` | data/references/ | Reference images directory |
| `-o, --output-dir` | chroma_data/ | Output directory for Chroma index |
| `-m, --model` | dinov2_vits14 | Embedding model variant |
| `--device` | auto | Device for inference |

**crop_reference.py:**

| Option | Default | Description |
|--------|---------|-------------|
| `-r, --raw-dir` | data/references_raw/ | Raw reference photos directory |
| `-o, --output-dir` | data/references/ | Output directory for cropped refs |
| `-m, --model` | models/yoloe-26l-seg.pt | YOLOE model for detection |
| `--device` | auto | Device for inference |
| `--conf` | 0.25 | Detection confidence threshold |

## Project Structure

```
sku-match/
├── main.py                   # CLI entry point
├── pyproject.toml            # v0.2.0
├── src/
│   ├── core.py               # CLI detection
│   ├── embedder.py           # DINOv2 embedder (PyTorch + optional ONNX)
│   ├── indexer.py            # Chroma-backed SKU index
│   ├── matcher.py            # Detection + matching pipeline
│   ├── types.py              # Dataclasses
│   ├── utils.py              # Shared utilities
│   ├── reference_processor.py # Crop→embed→index pipeline
│   ├── image_utils.py        # Annotation drawing, mask utilities
│   └── classes/              # Beverage class definitions
├── api/
│   ├── app.py                # FastAPI + lifespan
│   ├── config.py             # pydantic-settings
│   ├── database.py           # SQLAlchemy async
│   ├── models.py             # ORM models
│   ├── schemas.py            # Pydantic request/response
│   ├── auth.py               # API key + rate limiting
│   ├── tasks.py              # Async embedding task runner
│   ├── routes/               # API route handlers
│   └── services/             # Recognition, image storage
├── scripts/                  # CLI utilities
├── docs/                     # Documentation
├── models/                   # Model weights
├── chroma_data/              # Chroma persistence (gitignored)
└── results/                  # API result images
```

## SKU Matching Algorithm

1. **Detection**: YOLOE detects beverage containers in the input image
2. **Embedding**: DINOv2 extracts visual features from each crop (384/768/1024-dim)
3. **Scoring**: For each detected object, compute softmax probability over all SKUs
   - Top-2 cosine similarities per SKU are summed
   - Softmax normalizes scores to a probability distribution (0–1)
   - `--match-conf` filters matches below the probability threshold (default 0.5)

## Testing

```bash
python tests/test_detection.py
```

## Documentation

| Document | Description |
|----------|-------------|
| [docs/SETUP.md](docs/SETUP.md) | Environment setup guide |
| [docs/API_GUIDE.md](docs/API_GUIDE.md) | API authentication, quick test, endpoint reference |
| [docs/API.md.bak](docs/API.md.bak) | Initial Chinese requirements draft (archived) |
| [docs/PLAN.md](docs/PLAN.md) | Roadmap and future plans |
| [docs/AUDIT.md](docs/AUDIT.md) | Audit findings and issue tracker |
| [docs/PERF_PLAN.md](docs/PERF_PLAN.md) | Performance optimization notes |
| [docs/CONSIDERATIONS.md](docs/CONSIDERATIONS.md) | Evaluated improvements |
| [docs/CLEANUP_PLAN.md](docs/CLEANUP_PLAN.md) | Cleanup history (complete) |

## Notes

- **First run**: Downloads YOLOE model (~400MB). DINOv2 weights are vendored locally at `models/`.
- **FP16**: Automatic on CUDA — ~1.8x speedup over FP32.
- **GPU**: ≥2GB VRAM recommended.
- **Full config**: See [docs/SETUP.md](docs/SETUP.md) for all configuration options.
