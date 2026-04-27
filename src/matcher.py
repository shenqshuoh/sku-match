import gc
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLOE

from src.embedder import DINOv2Embedder, DINOv2Variant
from src.image_utils import isolate_object, save_crop, extract_binary_masks
from src.indexer import SKUIndexer, _detect_device
from src.classes.beverage_cls import BEVERAGE_CONTAINER_CLASSES
from src.classes.objects365_classes import OBJECTS365_CLASSES
from src.types import Detection, SKUMatch


def _free_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class SKUMatcher:
    def __init__(
        self,
        indexer: SKUIndexer,
        embedder: DINOv2Embedder,
        detector: YOLOE,
        confidence_threshold: float = 1.2,
    ):
        self.indexer = indexer
        self.embedder = embedder
        self.detector = detector
        self.confidence_threshold = confidence_threshold

    def match_images(
        self,
        image_paths: list[Path],
        output_dirs: list[Path] | None = None,
        batch_size: int = 1,
    ) -> list[list[SKUMatch]]:
        all_results = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            batch_dirs: list[Path | None] = (
                list(output_dirs[i : i + batch_size]) if output_dirs else [None] * len(batch_paths)
            )
            batch_results = self._process_batch(batch_paths, batch_dirs)
            all_results.extend(batch_results)
            _free_gpu_memory()

        return all_results

    def _process_batch(
        self,
        image_paths: list[Path],
        output_dirs: list[Path | None],
    ) -> list[list[SKUMatch]]:
        images = [Image.open(p).convert("RGB") for p in image_paths]
        img_arrays = [np.array(img) for img in images]

        all_detections = self._detect_batch(image_paths)

        all_crops = []
        crop_metadata = []

        for img_idx, (img_array, detections, img_path) in enumerate(
            zip(img_arrays, all_detections, image_paths)
        ):
            all_masks = [d.mask for d in detections if d.mask is not None]
            for det in detections:
                x1, y1, x2, y2 = map(int, det.bbox)
                other_masks = [m for m in all_masks if m is not det.mask]
                isolated = isolate_object(img_array, det.mask, other_masks)
                crop = Image.fromarray(isolated[y1:y2, x1:x2])
                all_crops.append(crop)
                crop_metadata.append((img_idx, det, crop, img_path))

        if not all_crops:
            return [[] for _ in image_paths]

        all_embeddings = self.embedder.embed_batch(all_crops)

        all_embeddings_arr = np.array(all_embeddings)
        all_matches = self.indexer.search_batch(all_embeddings_arr)

        results_by_image = [[] for _ in image_paths]

        for (img_idx, det, crop, img_path), (sku_id, score) in zip(crop_metadata, all_matches):
            sku_name = self.indexer.get_sku_name(sku_id) or sku_id
            match = SKUMatch(
                detection=det,
                sku_id=sku_id,
                sku_name=sku_name,
                match_score=score,
            )
            results_by_image[img_idx].append(match)

            output_dir = output_dirs[img_idx]
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                save_sku = "unk" if score < self.confidence_threshold else sku_id
                crop_idx = len(results_by_image[img_idx])
                save_crop(crop, f"{img_path.stem}_{crop_idx:03d}_{save_sku}.jpg", output_dir)

        return results_by_image

    def _detect_batch(self, image_paths: list[Path]) -> list[list[Detection]]:
        results = self.detector.predict(
            source=[str(p) for p in image_paths],
            retina_masks=True,
            verbose=False,
        )

        all_detections = []
        for result in results:
            if result.boxes is None:
                all_detections.append([])
                continue

            detections = []
            binary_masks = extract_binary_masks(result)
            for i, (box, mask) in enumerate(zip(result.boxes, binary_masks)):
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detections.append(
                    Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=float(box.conf[0]),
                        class_name=OBJECTS365_CLASSES[int(box.cls[0])],
                        class_id=int(box.cls[0]),
                        mask=mask,
                    )
                )
            all_detections.append(detections)

        return all_detections

    @classmethod
    def from_index_dir(
        cls,
        index_dir: Path,
        det_model: str = "models/yoloe-26l-seg.pt",
        emb_model: DINOv2Variant = "dinov2_vitb14",
        device: str | None = None,
        confidence_threshold: float = 1.2,
    ) -> "SKUMatcher":
        if device is None:
            device = _detect_device()
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
        )
