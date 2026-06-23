# SKU Match — Environment Setup Guide

## 1. Prerequisites

- **Python 3.10–3.13**
- **GPU** (recommended): NVIDIA GPU with CUDA support. Falls back to CPU/MPS on macOS.
- **Git**: to clone the repository

## 2. Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. Install it first:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
# Clone the repo
git clone https://github.com/shenqshuoh/sku-match.git
cd sku-match

# Install dependencies (CPU)
uv sync

# OR: Install with CUDA 12.8 support
uv sync --extra cu128

# OR: Install with CUDA 12.6 support
uv sync --extra cu126
```

> **Note:** `uv sync` creates the `.venv` automatically. No manual venv creation or activation needed — use `uv run` to execute commands.

## 3. Running Commands

All commands use `uv run` — no venv activation required:

```bash
# CLI: SKU matching (index auto-builds from data/references/ on first run)
uv run sku-match

# CLI: Detection only
uv run sku-match --detection-only

# API server
uv run sku-match-api

# API: reinitialize DB + Chroma and bulk-add SKUs on the server
bash scripts/init_and_download.sh

# Run tests
uv run python tests/test_detection.py

# Lint
uv run ruff check .
```

## 4. Model Weights

The following model files must be present in the `models/` directory:

| File | Source | Notes |
|------|--------|-------|
| `models/yoloe-26l-seg.pt` | Download from Ultralytics / project release | YOLOE detection model (~400MB) |

DINOv2 embedding models are downloaded automatically from HuggingFace Hub on first use (~85–330MB depending on variant), cached in `~/.cache/huggingface/`.

Supported variants (selected via `EMB_MODEL` / `--emb-model` / `-m`):

| Model ID | Dimensions | Notes |
|----------|-----------|-------|
| `facebook/dinov2-small` (default API) | 384-dim | Fastest |
| `facebook/dinov2-base` (default CLI) | 768-dim | Balanced |
| `facebook/dinov2-large` | 1024-dim | Higher accuracy |
| `facebook/dinov2-giant` | 1536-dim | Best accuracy |
| `facebook/dinov2-*-with-registers` | same | Registers variants (cleaner features) |

## 5. Configuration

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
| `EMB_MODEL` | `facebook/dinov2-small` | HuggingFace DINOv2 model ID (e.g., `facebook/dinov2-base`, `facebook/dinov2-large-with-registers`) |
| `DEVICE` | _(auto)_ | Force device: `cuda`, `mps`, `cpu`. Auto-detects if unset. |
| `USE_ONNX` | `false` | Use Optimum ONNX Runtime for embedding inference |
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
| `RESULTS_MAX_AGE_HOURS` | `24` | Annotated image cleanup age (hours) |
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
RESULTS_MAX_AGE_HOURS=24
```

## 6. Starting the Server

```bash
# Start server (production)
uv run sku-match-api

# Start with debug mode (auto-reload on code changes)
DEBUG=true uv run sku-match-api
```

> **Note:** No venv activation needed. `uv run` resolves the project venv automatically.

The server logs startup progress:

```
INFO api.app: Starting up — device=cuda
INFO api.app: Loading YOLOE detector: models/yoloe-26l-seg.pt
INFO api.app: Detector converted to FP16
INFO api.app: Warming up detector...
INFO api.app: Detector warm-up complete
INFO api.app: Loading embedder: facebook/dinov2-small (onnx=False)
INFO api.app: Opening Chroma index: chroma_data
INFO api.app: Startup complete
INFO Uvicorn running on http://0.0.0.0:8000
```

First startup may take 10–20 seconds (model loading + GPU warm-up). Subsequent requests are fast.

## 7. Health Check

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

## 8. Building the Chroma Index

You normally don't build the index manually:

- **CLI** (`uv run sku-match`): the index auto-builds from `data/references/{sku_id}/` on first
  run (crop → embed → index) when `chroma_data/` is empty. The embedding model is set with
  `--emb-model`.
- **API**: SKUs are indexed automatically when created via `/api/v1/goods/sku/new`. On a server,
  use `scripts/init_and_download.sh` to reinitialize the DB + Chroma and bulk-add SKUs from a
  reference directory.

> `scripts/build_index.py` is **deprecated** — it embeds raw (uncropped) images, stores no patch
> tokens or `feature_type` metadata, and bypasses the database. Use the CLI auto-build or
> `init_and_download.sh` instead.

## 9. Troubleshooting

### `ModuleNotFoundError: No module named 'fastapi'`

Dependencies not installed:
```bash
uv sync
```

### Model file not found

Ensure `models/yoloe-26l-seg.pt` exists. On first run, YOLOE may attempt to download the model (~400MB). If download fails (network issues), manually place the file in `models/`.

### OOM on GPU

Set `IMGSZ=640` or ensure `retina_masks=False` (default). The 2GB VRAM GPU requires these settings. See `PERF_PLAN.md` for details.

### CUDA not available

The server auto-detects device (cuda → mps → cpu). Check logs for `device=cpu` and verify CUDA drivers:
```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
```
