# Color Histogram Re-Ranking — Implementation Plan

**Status:** Approved (2026-06-24)
**Companion doc:** `docs/COLOUR_HISTOGRAMS.md` (theory + evidence)
**Scope:** API-only. CLI untouched. Strictly additive, default-off.

---

## 1. Goal

Add a color signal to SKU re-ranking to fix DINOv2's color-blindness (green Sprite vs. red Coke retrieve nearly identically). Color is fused at **score level** — not concatenated into the embedding — because color lives in a different space with a different similarity distribution.

See `docs/COLOUR_HISTOGRAMS.md` for the quantitative evidence (DINOv2 color retrieval accuracy = 40.8%) and the full rationale.

---

## 2. Resolved design decisions

| Fork | Decision | Rationale |
|---|---|---|
| **Weighting form** | Flat simplex: `final = β·patch + γ·color + (1−β−γ)·coarse_norm` | Spans the same simplex as the doc's nested form, but preserves the **existing** `RERANK_BLEND_BETA` convention (β = patch weight, per `reranker.py:154`). The doc redefines β as the *dino* weight — that would invert the existing field, so we reject it. |
| **Candidate pool** | **Minimal re-score** — compute `color_sim` only for candidates ChromaDB already returned (top-K). No brute-force color retrieval, no in-memory matrix. | The DINOv2 color-blind failure is a *re-ordering* problem within surfaced candidates, not a recall problem — the correct SKU is almost always in top-50. Re-scoring solves it at ~zero cost. Full brute-force + union (doc's architecture) only helps when the correct SKU ranks outside top-K, which is rare for beverage cans. |
| **Color source crop** | Raw YOLO crop, both sides | Product area dominates the histogram; background noise is small and consistent. Matches the doc's rationale. |
| **CLI support** | No — API-only | Consistent with patch re-ranking, which is also API-only (`src/matcher.py` uses plain `score_matches`). |

### Weighting form detail

The doc presents its formula as two phases, but reading `query()` carefully it is a **single closed-form blend** computed after all three similarities are known:

```
doc nested:  final = γ·color + (1−γ)·[β·dino + (1−β)·patch]
           = γ·color + (1−γ)β·dino + (1−γ)(1−β)·patch      ← weights sum to 1
```

Our flat form spans the **same simplex** with a different (and convention-consistent) parametrization:

```
flat:        final = β·patch + γ·color + (1−β−γ)·coarse_norm
```

where:
- `β = RERANK_BLEND_BETA` — **patch** weight (unchanged from today; γ=0 reduces exactly to current 2-way blend).
- `γ = COLOR_GAMMA` — **color** weight (new).
- `coarse_norm` — query-relative normalized ChromaDB similarity (existing behavior).

**Constraint:** `β + γ ≤ 1`, validated at `rerank()` entry (fail fast on bad config).

### Pre-existing scale note (not introduced by this work)

`coarse_norm` is query-relative (divided by max coarse in the query); `patch_score` and `color_score` are absolute cosines. Mixing a relative score with absolute ones is existing behavior. We preserve it — no new calibration introduced.

---

## 3. Architecture

Color slots in as a **parallel sibling** to the existing patch pipeline. Two new leaf modules with zero cross-coupling; the orchestrator (`reranker.py`) gains an optional color path that is inert when color is absent.

```
INDEX TIME (ReferenceProcessor)
  crop ──┬──→ DINOv2 features ──→ ChromaDB (coarse)        ┐
         │                      └→ PatchStore (patches)    │  existing
         └──→ extract_color_descriptor ──→ ColorStore       ┘  NEW

QUERY TIME (RecognitionService)
  raw crop ──┬──→ DINOv2 features ──→ ChromaDB top-K ──┐
             │                                         ├─→ rerank() 3-way blend
             └──→ extract_color_descriptor ────────────┘  NEW
                                                         (γ>0: 3-way; γ=0: 2-way, unchanged)
```

---

## 4. Module specs

### `src/color.py` (new — pure leaf, no I/O, no class)

```python
def extract_color_descriptor(image: np.ndarray, n_bins: int = 16) -> np.ndarray:
    """
    HSV histogram descriptor.

    Args:
        image: RGB uint8 ndarray (H, W, 3) — codebase convention.
               cv2 RGB→HSV conversion performed internally.
        n_bins: histogram bins per channel (16 → 49-dim descriptor).

    Returns:
        (3*n_bins + 1,) L2-normalized float32 vector:
        [hue_hist(n_bins), sat_hist(n_bins), val_hist(n_bins), dominant_hue(1)].
    """
```

- Accepts **RGB** (PIL/numpy convention used everywhere else in the codebase), converts to HSV internally.
- Per-channel 1D histograms, L2-normalized independently; dominant hue = saturation-weighted circular mean; final vector L2-normalized.
- Deterministic, pure, unit-testable in isolation.

### `src/color_store.py` (new — leaf, mirrors `PatchStore` 1:1)

```python
class ColorStore:
    def __init__(self, color_dir: str | Path) -> None: ...
    @property
    def color_dir(self) -> Path: ...
    def save(self, doc_id: str, descriptor: np.ndarray) -> Path: ...
    def load(self, doc_id: str) -> np.ndarray: ...                       # raises FileNotFoundError
    def load_batch(self, doc_ids: list[str]) -> dict[str, np.ndarray]: ...  # silent skip missing
    def delete(self, doc_id: str) -> None: ...
    def delete_sku(self, sku_id: str) -> int: ...                        # prefix match on "{sku_id}__"
    def count(self) -> int: ...
```

- Identical API and doc_id scheme (`{sku_id}__{media_id}`) to `PatchStore`.
- Stores `float32` `(D,)` arrays (D=49 for n_bins=16).
- Kept independent of `PatchStore` — no shared base — to honor the non-interference principle. (Optional future refactor: extract a generic `VectorStore` base if a third sibling appears.)

### `src/reranker.py` (extend — existing 2-way path unchanged when color absent)

Add `color_score: float` fields to `RerankCandidate` and `SKURerankResult`.

```python
def rerank(
    query_patches: NDArray,
    patch_store: PatchStore,
    top_vectors: list[VectorMatch],
    blend_beta: float = 0.3,            # patch weight
    top_n: int = 5,
    color_store: ColorStore | None = None,           # NEW
    query_color_descriptor: NDArray | None = None,   # NEW (D,)
    color_gamma: float = 0.0,                        # NEW color weight
) -> list[SKURerankResult]:
```

Behavior:
1. Validate `blend_beta + color_gamma <= 1.0` (raise `ValueError` otherwise).
2. Existing patch scoring path (unchanged).
3. If `color_store` and `query_color_descriptor` provided and `color_gamma > 0`:
   - `color_store.load_batch(doc_ids)` → compute `color_score = cosine(query_color, ref_color)` per candidate.
   - Missing color files → `color_score = 0.0` (graceful, mirrors `PatchStore` skip semantics).
   - Aggregate max `color_score` per SKU.
   - Blend: `blended = β·patch + γ·color + (1−β−γ)·coarse_norm`.
4. Else: current 2-way blend `blended = β·patch + (1−β)·coarse_norm`, byte-for-byte unchanged.

### `src/reference_processor.py` (extend — index time)

- `__init__`: add `color_store: ColorStore | None = None`.
- In `process_and_add` and `build_from_directory`: when `color_store` is set, `color_store.save(doc_id, extract_color_descriptor(np.array(crop), n_bins=...))` alongside the existing `patch_store.save`.
- Extract from the returned `crop` (raw under current default `use_mask=False`, as wired in `app.py`).

### `api/services/recognition.py` (extend — query time)

- `__init__`: add `color_store=None`, `use_color=False`, `color_gamma=0.0`, `color_bins=16`.
- In `recognize`: for each detection crop, `query_color_descriptor = extract_color_descriptor(np.array(crop), n_bins=self.color_bins)`. Thread `color_store`, `query_color_descriptor`, `color_gamma` into `rerank()` via `_score_with_reranking`.

### `api/config.py` (extend)

```python
USE_COLOR_RERANK: bool = False      # off until validated
COLOR_GAMMA: float = 0.4
COLOR_BINS: int = 16
COLOR_DIR: str = "data/colors"
```

### `api/app.py` (extend — wiring)

In lifespan, when `settings.USE_COLOR_RERANK`:
- Construct `ColorStore(color_dir=settings.COLOR_DIR)` (parallel to `PatchStore` block).
- Pass `color_store` to `ReferenceProcessor` and `RecognitionService`.
- Log the color config.

### `scripts/init_and_download.sh`

Add `data/colors/` to the wipe list alongside `data/patches/` (prevents stale doc_id mismatch on re-index).

---

## 5. Implementation order (dependency graph)

```
Wave 1 (parallel, non-overlapping writes):
  • src/color.py + src/color_store.py + tests/test_color.py        [fixer-color]
  • src/reranker.py (3-way blend) + tests/test_reranker_color.py   [fixer-reranker]
  • api/config.py (4 settings)                                     [orchestrator, trivial]

Wave 2 (after wave 1 reconciled):
  • src/reference_processor.py + api/services/recognition.py
    + api/app.py + scripts/init_and_download.sh                    [fixer-integration]
    (single coherent lane — these are the tightly-coupled live path)

Wave 3 (orchestrator):
  • AGENTS.md update (CODE MAP + CONVENTIONS + WHERE TO LOOK)
  • ruff check + pytest verification
```

---

## 6. Validation

- **`extract_color_descriptor`**: shape `(3*n_bins+1,)`; L2 norm ≈ 1.0; deterministic; handles tiny images without crashing.
- **`ColorStore`**: round-trip save/load; `delete_sku` prefix match; `load_batch` silent-skip on missing.
- **`rerank`**: 3-way blend math on a known fixture; `β+γ>1` raises `ValueError`; color-absent path identical to prior output; missing color file → `color_score=0`, no crash.
- **Regression**: `USE_COLOR_RERANK=False` (default) → existing pipeline output unchanged.

Verification commands:
```bash
uv run ruff check .
uv run pytest tests/ -q
```

---

## 7. Non-goals (deferred)

- **Grid search / weight tuning** (`tune_gamma`, `tune_all` in the doc) — requires annotated eval images we don't have yet.
- **Full brute-force color retrieval + candidate union** (doc Option A) — only if recall (correct SKU outside DINOv2 top-K) ever becomes an observed problem.
- **CLI support** — patch re-ranking is API-only; color follows suit.
- **Shared `VectorStore` base class** — premature until a third sibling appears.
- **Score calibration** (relative coarse vs absolute cosine) — pre-existing; out of scope.
