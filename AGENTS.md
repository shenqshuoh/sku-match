# PROJECT KNOWLEDGE BASE

**Generated:** 2026-03-23
**Project:** beverage-cashier

## OVERVIEW

YOLOE-based bottle & can detection with DINOv2 SKU matching for Apple Silicon (MPS).

## STRUCTURE

```
beverage-cashier/
├── main.py                   # CLI entry point (default: SKU matching mode)
├── pyproject.toml            # Project config
├── src/
│   ├── __init__.py           # Package init
│   ├── core.py               # YOLOE detection logic
│   ├── embedder.py           # DINOv2 embedding wrapper (3 variants: s/b/l)
│   ├── indexer.py            # SKU index with model-specific files
│   ├── matcher.py            # Detection + SKU matching pipeline
│   ├── types.py              # Dataclasses
│   ├── objects365_classes.py # 365 class labels
│   └── yoloe2o365.py         # CoreML export utility
├── scripts/
│   ├── build_index.py        # Build SKU reference index with --model arg
│   └── crop_reference.py     # Crop raw reference photos using YOLOE
├── tests/
│   └── test_detection.py     # Smoke test
├── data/
│   ├── images/               # Input images
│   └── references/           # SKU reference images (sku_id/*.jpg)
├── models/                   # YOLOE weights
├── index/                    # Generated index (embeddings_{model}.npy)
└── runs/                     # Output directory
    ├── match/                # SKU matching results
    │   ├── match01/          # First run results
    │   │   ├── crops/        # All segment crops with SKU names
    │   │   └── match_results.json
    │   ├── match02/          # Second run results
    │   └── ...
    └── detect/predict*/      # Detection-only results
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| CLI entry | main.py | Default: SKU matching. Use --detection-only for detection only |
| Detection logic | src/core.py | detect() function |
| DINOv2 embedder | src/embedder.py | 3 variants: vits14(384-dim), vitb14(768-dim), vitl14(1024-dim) |
| SKU indexer | src/indexer.py | Model-specific files: embeddings_{model}.npy |
| Detection + matching | src/matcher.py | SKUMatcher class with crop saving |
| Build ref index | scripts/build_index.py | --model arg for embedding variant selection |
| Crop raw refs | scripts/crop_reference.py | Pre-process raw photos with YOLOE detection |
| Input images | data/images/ | Hardcoded path |
| Raw reference photos | data/references_raw/ | Original photos to be cropped |
| Reference photos | data/references/ | Cropped sku_id/*.jpg structure |
| Output crops | runs/match/matchXX/crops/ | Named: {image}_{idx}_{sku_id}.jpg |

## CODE MAP

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| main() | function | main.py | CLI entry (default: matching mode) |
| run_detection() | function | main.py | Detection-only mode (--detection-only) |
| run_matching() | function | main.py | SKU matching mode (default) |
| detect() | function | src/core.py | Core YOLO inference |
| DINOv2Embedder | class | src/embedder.py | DINOv2 embedding wrapper |
| DINOv2Variant | type | src/embedder.py | Literal type: vits14/vitb14/vitl14 |
| SKUIndexer | class | src/indexer.py | Index with model-specific file names |
| SKUMatcher | class | src/matcher.py | Detection + SKU matching with crop saving |
| export_yoloe_to_objects365() | function | src/yoloe2o365.py | CoreML export |

## ENTRY POINTS

| Command | Script | Purpose |
|---------|--------|---------|
| `python main.py` | main.py | SKU matching mode (default) |
| `python main.py --detection-only` | main.py | Detection only mode |
| `python scripts/build_index.py -r data/references/ -o index/ -m dinov2_vitb14` | build_index.py | Build index with specific embedding model |
| `python scripts/crop_reference.py -r data/references_raw/ -o data/references/` | crop_reference.py | Crop raw reference photos using YOLOE detection |
| `python src/yoloe2o365.py <model>` | yoloe2o365.py | Export to CoreML |
| `python tests/test_detection.py` | test_detection.py | Smoke tests |

## CONVENTIONS

- **Type hints**: Full on function signatures (Python 3.10+ union: `str | None`)
- **Imports**: Absolute imports (`from src.core import detect`)
- **Linting**: Ruff configured (line-length: 100, py310 target)
- **Device**: Default `mps` (Apple Silicon)
- **Classes**: Objects365 indices (0=Bottle, 1=Canned)
- **Index files**: Model-specific naming: `embeddings_{model}.npy`, `metadata_{model}.json`
- **Crop naming**: `{image_name}_{index}_{sku_id}.jpg`

## ANTI-PATTERNS (THIS PROJECT)

- **Hardcoded paths**: `data/images` path hardcoded
- **sys.path hack**: tests/test_detection.py manipulates sys.path
- **No docstrings**: Minimal documentation
- **No [project.scripts]**: pyproject.toml lacks console_scripts entry
- **SSL override**: `ssl._create_default_https_context` disabled in embedder.py

## COMMANDS

```bash
# Setup
source .venv/bin/activate

# Default: SKU matching mode
python main.py

# SKU matching with custom models
python main.py --det-model models/yoloe-26l-seg.pt --emb-model dinov2_vitb14

# Detection only mode
python main.py --detection-only

# Build index (one-time setup)
python scripts/build_index.py -r data/references/ -o index/ -m dinov2_vitb14

# Build index with different embedding model
python scripts/build_index.py -r data/references/ -o index/ -m dinov2_vits14

# Crop raw reference photos
python scripts/crop_reference.py -r data/references_raw/ -o data/references/ -m models/yoloe-26l-seg.pt

# Run smoke test
python tests/test_detection.py
```

## NOTES

- **Default mode**: SKU matching (not detection-only)
- **Embedding models**: dinov2_vits14 (384-dim), dinov2_vitb14 (768-dim), dinov2_vitl14 (1024-dim)
- **Index files**: Model-specific (e.g., `embeddings_dinov2_vitb14.npy`)
- **Crop output**: Saved with SKU ID in filename (e.g., `image_001_coke_330ml.jpg`)
- **Match threshold**: Default 1.5 (sum of top 2 similarities, range 0-2)
- **SSL fix**: Disabled certificate verification for macOS compatibility
- **First run**: Downloads YOLOE and DINOv2 models (~400MB)
- **Crop naming**: `{image_name}_{index}_{sku_id}.jpg`
- **Match threshold**: Default 1.4 (sum of top 2 similarities, range 0-2)
- **SSL fix**: Disabled certificate verification for macOS compatibility
- **First run**: Downloads YOLOE and DINOv2 models (~400MB)
