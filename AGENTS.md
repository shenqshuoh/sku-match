# PROJECT KNOWLEDGE BASE

**Generated:** 2026-03-23
**Project:** beverage-cashier

## OVERVIEW

YOLOv12-based bottle detection running on Apple Silicon (MPS). Python ML application with Ultralytics YOLO.

## STRUCTURE

```
beverage-cashier/
├── main.py              # CLI entry point
├── pyproject.toml       # Project config
├── src/
│   ├── __init__.py      # Package init (exports detect)
│   └── core.py          # Core detection logic
├── tests/
│   └── test_detection.py # Smoke test
├── data/images/         # Input images
├── models/              # YOLO weights (.pt files)
└── runs/                # YOLO output (auto-created)
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| CLI entry | main.py | argparse, calls src.core.detect |
| Detection logic | src/core.py | detect() function, hardcoded paths |
| Input images | data/images/ | Hardcoded in core.py:14 |
| Model weights | models/ | yolo12m.pt auto-downloads |
| Output | runs/detect/predict*/ | YOLO default output dir |
| Tests | tests/test_detection.py | Smoke test only, sys.path hack |

## CODE MAP

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| main() | function | main.py:8 | CLI entry (argparse) |
| detect() | function | src/core.py:7 | Core YOLO inference |
| test_imports() | function | tests/test_detection.py:7 | Smoke test |

## CONVENTIONS

- **Type hints**: Full on function signatures (Python 3.10+ union: `str \| None`)
- **Docstrings**: None (project has no docstrings)
- **Imports**: Absolute (`from src.core import detect`)
- **Linting**: Ruff configured (line-length: 100, py310 target)
- **Device**: Default `mps` (Apple Silicon)

## ANTI-PATTERNS (THIS PROJECT)

- **Hardcoded paths**: `data/images` path hardcoded in src/core.py:14
- **Missing CLI input arg**: Can't specify input dir via CLI
- **sys.path hack**: tests/test_detection.py manipulates sys.path instead of proper package install
- **No docstrings**: Zero docstrings across codebase
- **No [project.scripts]**: pyproject.toml lacks console_scripts entry
- **Minimal testing**: Only import smoke test, no actual assertions
- **No CI/CD**: No .github/workflows or automation

## UNIQUE STYLES

- MPS-first: Explicit Apple Silicon GPU acceleration
- YOLOv12: Community release (not official YOLO)
- uv-based: Uses uv package manager (uv.lock present)

## COMMANDS

```bash
# Setup
source .venv/bin/activate

# Run detection
python main.py

# With options
python main.py --model yolo12x --conf 0.5 --batch 4

# Run smoke test
python tests/test_detection.py
```

## NOTES

- First inference slow (model loading + MPS compilation)
- YOLO model auto-downloads on first run (yolo12m.pt)
- Output: annotated images + JSON in runs/detect/predict/
- Target class hardcoded: `classes=[39]` (bottle class from COCO)
