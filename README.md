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
#    - DINOv2: downloaded from HuggingFace Hub on first use

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

# 2. Run detection with SKU matching
#    The Chroma index auto-builds on first run (crop + embed) when chroma_data/ is empty.
python main.py

# With custom options
python main.py --conf 0.3 --match-conf 0.5 --match-concentration 0.0 --match-verbose

# With custom models
python main.py --det-model models/yoloe-26l-seg.pt --emb-model facebook/dinov2-base
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

# 3. Run matching (index auto-builds from data/references/ on first run)
python main.py
```

### Choosing an Embedding Model

The DINOv2 embedding model is selected at run time and determines the dimensionality of the
(auto-built) index. **Switching models requires rebuilding the index** — delete `chroma_data/`
so it auto-rebuilds on the next run.

```bash
# Small model (faster): 384-dim embeddings
python main.py --emb-model facebook/dinov2-small

# Base model (balanced): 768-dim embeddings
python main.py --emb-model facebook/dinov2-base

# Large model (higher accuracy): 1024-dim embeddings
python main.py --emb-model facebook/dinov2-large
```

For API deployments, set `EMB_MODEL` and populate the index with `scripts/init_and_download.sh`
(see [docs/SETUP.md](docs/SETUP.md)).

### CLI Options

**main.py:**

| Option | Default | Description |
|--------|---------|-------------|
| `--detection-only` | - | Run detection without SKU matching |
| `--index` | chroma_data/ | Chroma index directory |
| `--det-model` | models/yoloe-26l-seg.pt | YOLOE detection model path |
| `--emb-model` | facebook/dinov2-small | HuggingFace DINOv2 model ID |
| `--device` | auto | Device: cuda, mps, or cpu |
| `--conf` | 0.25 | Detection confidence threshold |
| `--match-conf` | 0.5 | SKU match probability threshold (0–1) |
| `--match-concentration` | 0.0 | Concentration score threshold |
| `--imgsz` | 1280 | Inference image size |
| `--batch` | 16 | Batch size |
| `--all-cls` | - | Detect all 365 Objects365 classes |
| `--swap` | - | Swap to detection-only model |
| `--onnx` | - | Use ONNX backend for DINOv2 |

> **Deprecated:** `scripts/build_index.py` is deprecated. The CLI auto-builds the index from
> `data/references/` on first run; for API deployments use `scripts/init_and_download.sh`.

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
│   ├── embedder.py           # DINOv2 embedder (HuggingFace transformers + Optimum ONNX)
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

- **First run**: Downloads YOLOE model (~400MB). DINOv2 weights downloaded from HuggingFace Hub on first use (~85–330MB).
- **FP16**: Automatic on CUDA — ~1.8x speedup over FP32.
- **GPU**: ≥2GB VRAM recommended.
- **Full config**: See [docs/SETUP.md](docs/SETUP.md) for all configuration options.
