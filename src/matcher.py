from pathlib import Path
import time

import numpy as np
from PIL import Image
from ultralytics import YOLOE

from src.embedder import DINOv2Embedder, DINOv2Variant
from src.image_utils import isolate_object, save_crop, extract_binary_masks
from src.indexer import SKUIndexer
from src.classes.beverage_cls import BEVERAGE_CONTAINER_CLASSES
from src.types import Detection, SKUMatch
from src.utils import detect_device, free_gpu_memory


class SKUMatcher:
    def __init__(
        self,
        indexer: SKUIndexer,
        embedder: DINOv2Embedder,
        detector: YOLOE,
        confidence_threshold: float = 0.5,
        ratio_threshold: float = 0.0,
        det_conf: float = 0.25,
    ):
        self.indexer = indexer
        self.embedder = embedder
        self.detector = detector
        self.confidence_threshold = confidence_threshold
        self.ratio_threshold = ratio_threshold
        self.det_conf = det_conf

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
        free_gpu_memory()

        # Phase 2: Per-image embed + search
        for img_idx, (img_path, detections) in enumerate(zip(image_paths, all_detections)):
            t0 = time.perf_counter()

            img_array = np.array(Image.open(image_paths[img_idx]).convert("RGB"))

            if not detections:
                all_results[img_idx] = ([], 0.0)
                continue

            # Crop detections
            all_masks = [d.mask for d in detections if d.mask is not None]
            crops = []
            for det in detections:
                x1, y1, x2, y2 = map(int, det.bbox)
                other_masks = [m for m in all_masks if m is not det.mask]
                isolated = isolate_object(img_array, det.mask, other_masks)
                crop = Image.fromarray(isolated[y1:y2, x1:x2])
                crops.append(crop)

            # Embed + search (batched within this image's crops)
            embeddings = self.embedder.embed_batch(crops)
            embeddings_arr = np.array(embeddings)
            search_results = self.indexer.search_batch(embeddings_arr)

            # Build results
            results = []
            output_dir = output_dirs[img_idx] if output_dirs else None
            for det, (distribution, rank_info), crop in zip(detections, search_results, crops):
                ranked = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
                sku_id, confidence = ranked[0]
                sku_name = self.indexer.get_sku_name(sku_id) or sku_id
                # Confidence ratio: top-1 / top-2 (0.0 if only 1 SKU)
                match_ratio = confidence / ranked[1][1] if len(ranked) > 1 and ranked[1][1] > 0 else 0.0
                # Rank positions of top-2 vectors for matched SKU
                positions = rank_info.get(sku_id, [])
                top2_ranks = (positions[0], positions[1]) if len(positions) >= 2 else (positions[0], 0) if len(positions) == 1 else (0, 0)
                match = SKUMatch(
                    detection=det,
                    sku_id=sku_id,
                    sku_name=sku_name,
                    match_score=confidence,
                    match_ratio=match_ratio,
                    top2_ranks=top2_ranks,
                    sku_distribution=distribution,
                )
                results.append(match)

                if output_dir:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    is_match = confidence >= self.confidence_threshold and match_ratio >= self.ratio_threshold
                    save_sku = sku_id if is_match else "unk"
                    crop_idx = len(results)
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
                retina_masks=True,
                verbose=False,
                conf=self.det_conf,
            )

            for j, result in enumerate(results):
                img_idx = i + j
                if result.boxes is None:
                    continue

                binary_masks = extract_binary_masks(result)
                for box, mask in zip(result.boxes, binary_masks):
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls_id = int(box.cls[0])
                    all_detections[img_idx].append(
                        Detection(
                            bbox=(x1, y1, x2, y2),
                            confidence=float(box.conf[0]),
                            class_name=BEVERAGE_CONTAINER_CLASSES[cls_id] if cls_id < len(BEVERAGE_CONTAINER_CLASSES) else str(cls_id),
                            class_id=cls_id,
                            mask=mask,
                        )
                    )

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
            print(f"    detection {i}: conf={match.detection.confidence:.2f} score={match.match_score:.3f} ratio={match.match_ratio:.2f}, top 2 at {match.top2_ranks}")
            print(f"      {labels}")
            print(f"      {probs}")

    @classmethod
    def from_index_dir(
        cls,
        index_dir: Path,
        det_model: str = "models/yoloe-26l-seg.pt",
        emb_model: DINOv2Variant = "dinov2_vitb14",
        device: str | None = None,
        confidence_threshold: float = 0.5,
        ratio_threshold: float = 0.0,
        det_conf: float = 0.25,
    ) -> "SKUMatcher":
        if device is None:
            device = detect_device()
        embedder = DINOv2Embedder(model_name=emb_model, device=device)
        indexer = SKUIndexer()
        indexer.load(index_dir, emb_model)

        detector = YOLOE(det_model)
        detector.set_classes(BEVERAGE_CONTAINER_CLASSES)

        return cls(
            indexer=indexer,
            embedder=embedder,
            detector=detector,
            confidence_threshold=confidence_threshold,
            ratio_threshold=ratio_threshold,
            det_conf=det_conf,
        )
