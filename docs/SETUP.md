# SKU Match — Environment Setup Guide

## 1. Prerequisites

- **Python 3.10–3.13**
- **GPU** (recommended): NVIDIA GPU with CUDA support. Falls back to CPU/MPS on macOS.
- **Git**: to clone the repository

## 2. Installation

```bash
# Clone the repo
git clone https://github.com/shenqshuoh/sku-match.git
cd sku-match

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS

# Install dependencies (CPU)
pip install -e ".[cpu]"

# OR: Install with CUDA 12.6 support
pip install -e ".[cu126]"
```

> **Note:** `torch` and `torchvision` are installed via optional dependency groups (`[cu126]` or `[cpu]`). They are not in the base dependencies. Choose the group matching your hardware.

## 3. Model Weights

The following model files must be present in the `models/` directory:

| File | Source | Notes |
|------|--------|-------|
| `models/yoloe-26l-seg.pt` | Download from Ultralytics / project release | YOLOE detection model (~400MB) |
| `models/dinov2_vits14_pretrain.pth` | Vendored in repo (or download from Meta) | DINOv2 embedding model |

Other DINOv2 variants are supported but require corresponding weight files:

| Variant | Dimensions | Weight File |
|---------|-----------|-------------|
| `dinov2_vits14` (default) | 384-dim | `models/dinov2_vits14_pretrain.pth` |
| `dinov2_vitb14` | 768-dim | `models/dinov2_vitb14_pretrain.pth` |
| `dinov2_vitl14` | 1024-dim | `models/dinov2_vitl14_pretrain.pth` |

## 4. Configuration

All settings are configured via **environment variables** or a **`.env` file** in the project root.

### Create `.env` file

```bash
cp .env.example .env   # if available
# OR create manually:
touch .env
```

### Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| **Server** | | |
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8000` | Bind port |
| `DEBUG` | `false` | Enable debug mode (uvicorn auto-reload) |
| **Models** | | |
| `DET_MODEL` | `models/yoloe-26l-seg.pt` | YOLOE detection model path |
| `EMB_MODEL` | `dinov2_vits14` | DINOv2 variant: `dinov2_vits14`, `dinov2_vitb14`, or `dinov2_vitl14` |
| `DEVICE` | _(auto)_ | Force device: `cuda`, `mps`, `cpu`. Auto-detects if unset. |
| `USE_ONNX` | `false` | Use ONNX Runtime for DINOv2 (requires ≥4GB VRAM) |
| **Detection** | | |
| `DET_CONF` | `0.25` | YOLOE detection confidence threshold |
| `IMGSZ` | `1280` | Inference image size |
| `MATCH_CONF` | `0.5` | SKU match confidence threshold |
| `CONCENTRATION_TOPK` | `10` | Top-K for concentration score calculation |
| **Storage** | | |
| `DATA_DIR` | `data` | Data directory |
| `CHROMA_PERSIST_DIR` | `chroma_data` | Chroma vector store persistence directory |
| `RESULTS_DIR` | `results` | Annotated result images directory |
| `DATABASE_URL` | `sqlite+aiosqlite:///./sku_match.db` | SQLAlchemy async database URL |
| `DOWNLOAD_TIMEOUT` | `30` | Image download timeout (seconds) |
| **Security** | | |
| `API_KEY` | _(empty)_ | API key for authentication. **Empty = disabled.** |
| `RATE_LIMIT` | `0` | Max requests per minute per IP. **0 = disabled.** |

### Minimal `.env` Example

```env
# No configuration needed for local development.
# All defaults work out of the box.
# Device is auto-detected (cuda > mps > cpu).
```

### Production `.env` Example

```env
DEVICE=cuda
API_KEY=your-secret-api-key-here
RATE_LIMIT=100
DOWNLOAD_TIMEOUT=60
```

## 5. Starting the Server

```bash
source .venv/bin/activate

# Start server (production)
python -m api.app

# Start with debug mode (auto-reload on code changes)
DEBUG=true python -m api.app
```

The server logs startup progress:

```
INFO api.app: Starting up — device=cuda
INFO api.app: Loading YOLOE detector: models/yoloe-26l-seg.pt
INFO api.app: Detector converted to FP16
INFO api.app: Warming up detector...
INFO api.app: Detector warm-up complete
INFO api.app: Loading DINOv2 embedder: dinov2_vits14 (onnx=False)
INFO api.app: Opening Chroma index: chroma_data
INFO api.app: Startup complete
INFO Uvicorn running on http://0.0.0.0:8000
```

First startup may take 10–20 seconds (model loading + GPU warm-up). Subsequent requests are fast.

## 6. Health Check

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

## 7. Building the Chroma Index (Optional)

If you have existing reference images in `data/references/{sku_id}/`:

```bash
python scripts/build_index.py \
  -r data/references/ \
  -o chroma_data/ \
  -m dinov2_vits14
```

This is only needed for manual index building. The API builds the index automatically when SKUs are created via `/api/v1/goods/sku/new`.

## 8. Troubleshooting

### `ModuleNotFoundError: No module named 'fastapi'`

The virtual environment is not activated:
```bash
source .venv/bin/activate
```

### Model file not found

Ensure `models/yoloe-26l-seg.pt` exists. On first run, YOLOE may attempt to download the model (~400MB). If download fails (network issues), manually place the file in `models/`.

### OOM on GPU

Set `IMGSZ=640` or ensure `retina_masks=False` (default). The 2GB VRAM GPU requires these settings. See `PERF_PLAN.md` for details.

### CUDA not available

The server auto-detects device (cuda → mps → cpu). Check logs for `device=cpu` and verify CUDA drivers:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
