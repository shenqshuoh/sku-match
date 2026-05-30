import logging
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLOE

from src.classes.beverage_cls import BEVERAGE_CONTAINER_CLASSES
from src.core import parse_detections
from src.embedder import Embedder, EmbedderVariant
from src.image_utils import save_crop
from src.indexer import SKUIndexer, score_matches
from src.types import Detection, SKUMatch
from src.utils import detect_device, free_gpu_memory

logger = logging.getLogger(__name__)


class SKUMatcher:
    def __init__(
        self,
        indexer: SKUIndexer,
        embedder: Embedder,
        detector: YOLOE,
        confidence_threshold: float = 0.5,
        concentration_threshold: float = 0.0,
        concentration_topk: int = 10,
        det_conf: float = 0.25,
        swap_models: bool = False,
    ):
        self.indexer = indexer
        self.embedder = embedder
        self.detector = detector
        self.confidence_threshold = confidence_threshold
        self.concentration_threshold = concentration_threshold
        self.concentration_topk = concentration_topk
        self.det_conf = det_conf
        self.swap_models = swap_models
        self._device = embedder.device

    def match_images(
        self,
        image_paths: list[Path],
        output_dirs: list[Path] | None = None,
        batch_size: int = 1,
        verbose: bool = False,
    ) -> list[tuple[list[SKUMatch], float]]:
        """Match SKUs across images.

        Returns:
            List of (results, elapsed_seconds) tuples, one per image.
        """
        all_results: list[tuple[list[SKUMatch], float]] = [([], 0.0)] * len(image_paths)

        # Phase 1: Batch detection
        all_detections = self._detect_all(image_paths, batch_size)

        if self.swap_models and self._device != "cpu":
            # Unload detector from GPU, move embedder to GPU
            logger.info("Swapping: unloading detector, loading embedder to GPU")
            self.detector.model.cpu()
            del self.detector
            torch.cuda.empty_cache()
            self.embedder.to(self._device)

        free_gpu_memory()

        # Phase 2: Per-image embed + search
        for img_idx, (img_path, detections) in enumerate(zip(image_paths, all_detections)):
            t0 = time.perf_counter()

            img_array = np.array(Image.open(image_paths[img_idx]).convert("RGB"))

            if not detections:
                all_results[img_idx] = ([], 0.0)
                continue

            crops = []
            for det in detections:
                x1, y1, x2, y2 = map(int, det.bbox)
                crop = Image.fromarray(img_array[y1:y2, x1:x2])
                crops.append(crop)

            # Embed + search (chunked to avoid GPU OOM on low-VRAM devices)
            _chunk = 16
            all_embeddings: list[np.ndarray] = []
            for ci in range(0, len(crops), _chunk):
                all_embeddings.append(self.embedder.embed_batch(crops[ci : ci + _chunk]))
            embeddings_arr = np.concatenate(all_embeddings, axis=0) if all_embeddings else np.empty((0, self.embedder.dim))
            search_results = self.indexer.search_batch(embeddings_arr)

            # Build results using shared scoring
            results = score_matches(detections, search_results, self.indexer, self.concentration_topk)

            output_dir = output_dirs[img_idx] if output_dirs else None
            if output_dir:
                for crop_idx, (match, crop) in enumerate(zip(results, crops), start=1):
                    output_dir.mkdir(parents=True, exist_ok=True)
                    is_match = match.match_score >= self.confidence_threshold and match.match_concentration >= self.concentration_threshold
                    save_sku = match.sku_id if is_match else "unk"
                    save_crop(crop, f"{img_path.stem}_{crop_idx:03d}_{save_sku}.jpg", output_dir)

            elapsed = time.perf_counter() - t0
            all_results[img_idx] = (results, elapsed)

            if verbose:
                self._print_verbose(img_path.name, results)

        return all_results

    def _detect_all(
        self,
        image_paths: list[Path],
        batch_size: int = 1,
    ) -> list[list[Detection]]:
        """Run YOLOE detection on all images in batches."""
        all_detections: list[list[Detection]] = [[] for _ in image_paths]

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            results = self.detector.predict(
                source=[str(p) for p in batch_paths],
                retina_masks=False,
                verbose=False,
                conf=self.det_conf,
            )

            for j, result in enumerate(results):
                img_idx = i + j
                all_detections[img_idx] = parse_detections(result)

        return all_detections

    @staticmethod
    def _print_verbose(image_name: str, results: list[SKUMatch]) -> None:
        print(f"  {image_name}:")
        for i, match in enumerate(results, 1):
            dist = match.sku_distribution
            if dist is None:
                print(f"    detection {i}: no distribution")
                continue
            ranked = sorted(dist.items(), key=lambda x: x[1], reverse=True)
            top = ranked[:10]
            labels = "  ".join(f"{sku:>6s}" for sku, _ in top)
            probs = "  ".join(f"{p:>6.3f}" for _, p in top)
            print(f"    detection {i}: conf={match.detection.confidence:.2f} score={match.match_score:.3f} conc={match.match_concentration:.3f}, top 2 at {match.top2_ranks}")
            print(f"      {labels}")
            print(f"      {probs}")

    @classmethod
    def from_index_dir(
        cls,
        index_dir: Path,
        det_model: str = "models/yoloe-26l-seg.pt",
        emb_model: EmbedderVariant | str = "facebook/dinov2-base",
        device: str | None = None,
        confidence_threshold: float = 0.5,
        concentration_threshold: float = 0.0,
        concentration_topk: int = 10,
        det_conf: float = 0.25,
        swap_models: bool = False,
        use_onnx: bool = False,
    ) -> "SKUMatcher":
        if device is None:
            device = detect_device()

        # When swap is enabled, load embedder on CPU first — moved to GPU after detection
        emb_device = "cpu" if swap_models and device != "cpu" else device
        embedder = Embedder(model_name=emb_model, device=emb_device, use_onnx=use_onnx)
        indexer = SKUIndexer()
        indexer.init_collection(persist_dir=index_dir)

        detector = YOLOE(det_model)
        detector.set_classes(BEVERAGE_CONTAINER_CLASSES)
        if device == "cuda":
            detector.model.half()

        instance = cls(
            indexer=indexer,
            embedder=embedder,
            detector=detector,
            confidence_threshold=confidence_threshold,
            concentration_threshold=concentration_threshold,
            concentration_topk=concentration_topk,
            det_conf=det_conf,
            swap_models=swap_models,
        )
        # Store target device (not initial device, which may be CPU for swap)
        instance._device = device
        return instance

