"""Reference image processing: detect, crop, embed, and index."""

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLOE

from src.classes.beverage_cls import BEVERAGE_CONTAINER_CLASSES
from src.embedder import Embedder
from src.indexer import SKUIndexer
from src.masking import extract_binary_masks, mask_background
from src.patch_store import PatchStore
from src.utils import embedding_to_list, free_gpu_memory

logger = logging.getLogger(__name__)


@dataclass
class ProcessResult:
    """Result from process_and_add: success flag + optional masked crop temp file."""
    success: bool
    crop_path: Path | None = None  # Temp file containing the masked crop JPEG


def select_best_detection(boxes, img_w: int, img_h: int) -> int | None:
    """Select the best detection from results — smallest box that covers the image center.

    Returns None if no box covers the center point (no fallback).
    """
    if not boxes:
        return None

    cx, cy = img_w / 2, img_h / 2
    candidates = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            area = (x2 - x1) * (y2 - y1)
            candidates.append((i, area))

    if not candidates:
        return None

    return min(candidates, key=lambda x: x[1])[0]


class ReferenceProcessor:
    """Processes raw reference images: detect → crop → embed → index."""

    def __init__(
        self,
        detector: YOLOE,
        embedder: Embedder,
        indexer: SKUIndexer,
        device: str = "cpu",
        det_conf: float = 0.25,
        imgsz: int = 640,
        use_mask: bool = False,
        patch_store: PatchStore | None = None,
        feature_type: str | None = None,
        crop_model_path: str | None = None,
    ):
        self.detector = detector
        self.embedder = embedder
        self.indexer = indexer
        self.device = device
        self.det_conf = det_conf
        self.imgsz = imgsz
        self.use_mask = use_mask
        self.patch_store = patch_store
        self.feature_type = feature_type
        self.crop_model_path = crop_model_path

    def _crop_detect(self, image_np: np.ndarray):
        """Run detection for cropping. Uses on-demand crop model if configured."""
        if self.crop_model_path:
            temp_model = YOLOE(self.crop_model_path)
            temp_model.set_classes(BEVERAGE_CONTAINER_CLASSES)
            try:
                # On-demand crop model runs in FP32 to avoid fuse_conv_and_bn
                # crash on models with many BatchNorm layers (e.g. yoloe-26x).
                # FP16 speedup is negligible for a single on-demand inference.
                results = temp_model.predict(
                    source=image_np,
                    device=self.device,
                    conf=self.det_conf,
                    imgsz=self.imgsz,
                    retina_masks=False,
                    verbose=False,
                )
                return results[0]
            finally:
                del temp_model
                free_gpu_memory()
        else:
            # Startup detector is already FP16 via model.half() — no half= kwarg
            results = self.detector.predict(
                source=image_np,
                device=self.device,
                conf=self.det_conf,
                imgsz=self.imgsz,
                retina_masks=False,
                verbose=False,
            )
            return results[0]

    def crop_reference(self, image_path: Path) -> Image.Image | None:
        """Detect the primary object in a reference image and return a cropped PIL Image.

        Returns None if no detection is found.
        """
        image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
        image_np = np.array(image)

        result = self._crop_detect(image_np)

        if result.boxes is None or len(result.boxes) == 0:
            logger.warning("No detections in reference image: %s", image_path)
            return None

        h, w = image_np.shape[:2]

        sel_idx = select_best_detection(result.boxes, w, h)
        if sel_idx is None:
            logger.warning("Could not select best detection in: %s", image_path)
            return None

        box = result.boxes[sel_idx]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        if self.use_mask and result.masks is not None:
            binary_masks = extract_binary_masks(result)
            mask = binary_masks[sel_idx]
            other_masks = [m for j, m in enumerate(binary_masks) if j != sel_idx and m is not None]
            masked = mask_background(image_np, mask, exclude_masks=other_masks)
            crop = Image.fromarray(masked[y1:y2, x1:x2])
        else:
            crop = image.crop((x1, y1, x2, y2))

        if crop.size[0] == 0 or crop.size[1] == 0:
            logger.warning("Empty crop from: %s", image_path)
            return None

        return crop

    def process_and_add(
        self,
        sku_id: str,
        sku_name: str,
        media_id: str,
        image_path: Path,
        metadata: dict | None = None,
    ) -> ProcessResult:
        """Full pipeline: crop → embed → add to index → save crop to temp file.

        Returns ProcessResult with success flag and optional crop_path.
        crop_path is a temp file that the caller should upload and then delete.
        """
        crop = self.crop_reference(image_path)

        if crop is None:
            logger.warning(
                "No center-covering detection for %s/%s — skipping", sku_id, media_id
            )
            return ProcessResult(success=False)

        # Extract features (CLS + patches) in a single forward pass
        features = self.embedder.extract_features_batch([crop])[0]
        embedding = self.embedder.features_to_embedding(features)

        # Save patches for re-ranking if patch store is configured
        if self.patch_store is not None:
            doc_id = f"{sku_id}__{media_id}"
            self.patch_store.save(doc_id, features.patches)

        self.indexer.add_reference(sku_id, sku_name, media_id, embedding, metadata, feature_type=self.feature_type)
        logger.info("Cropped, embedded and indexed reference %s/%s", sku_id, media_id)

        # Save masked crop to temp file for Qiniu upload
        with tempfile.NamedTemporaryFile(suffix=".jpg", prefix=f"crop_{sku_id}_{media_id}_", delete=False) as tmp:
            crop_path = Path(tmp.name)
        crop.save(crop_path, format="JPEG", quality=95)

        return ProcessResult(success=True, crop_path=crop_path)

    def build_from_directory(
        self,
        reference_dir: Path,
        batch_size: int = 8,
    ) -> int:
        """Build the index from a directory of reference images (crop + embed).

        Expects ``reference_dir/{sku_id}/`` subdirectories containing images.
        Clears all existing vectors before building.

        Returns the number of reference images indexed.
        """
        reference_dir = Path(reference_dir)
        if not reference_dir.exists():
            logger.warning("Reference directory not found: %s", reference_dir)
            return 0

        all_paths: list[tuple[Path, str]] = []  # (image_path, sku_id)
        for sku_dir in sorted(reference_dir.iterdir()):
            if not sku_dir.is_dir():
                continue
            sku_id = sku_dir.name
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                for img_path in sorted(sku_dir.glob(ext)):
                    all_paths.append((img_path, sku_id))

        if not all_paths:
            logger.warning("No reference images found in %s", reference_dir)
            return 0

        logger.info("Building index from %d reference images in %s", len(all_paths), reference_dir)

        # Clear existing data
        existing = self.indexer.collection.get(include=[])
        if existing["ids"]:
            self.indexer.collection.delete(ids=existing["ids"])
            self.indexer._cache_valid = False

        # Process in batches: crop → collect images → embed batch → add batch
        total_added = 0
        for i in range(0, len(all_paths), batch_size):
            batch = all_paths[i : i + batch_size]
            crops: list[Image.Image] = []
            batch_meta: list[tuple[str, str, str]] = []  # (sku_id, sku_name, doc_id)

            for img_path, sku_id in batch:
                crop = self.crop_reference(img_path)
                if crop is not None:
                    crops.append(crop)
                    batch_meta.append((sku_id, sku_id, f"{sku_id}__{total_added + len(batch_meta):04d}"))
                else:
                    logger.warning("Skipping reference image (no center-covering detection): %s", img_path)

            if not crops:
                logger.info("All images in batch %d-%d skipped", i + 1, min(i + batch_size, len(all_paths)))
                continue

            # Extract features (CLS + patches) in a single forward pass
            features_list = self.embedder.extract_features_batch(crops)
            embeddings = np.stack([self.embedder.features_to_embedding(f) for f in features_list])

            # Save patches if patch store is configured
            if self.patch_store is not None:
                for j, f in enumerate(features_list):
                    doc_id = batch_meta[j][2]  # (sku_id, sku_name, doc_id)
                    self.patch_store.save(doc_id, f.patches)

            ids = [m[2] for m in batch_meta]
            emb_list = [embedding_to_list(e) for e in embeddings]
            metadatas = [
                {"sku_id": m[0], "sku_name": m[1], "enabled": True}
                for m in batch_meta
            ]
            if self.feature_type is not None:
                for meta in metadatas:
                    meta["feature_type"] = self.feature_type

            self.indexer.collection.add(ids=ids, embeddings=emb_list, metadatas=metadatas)
            total_added += len(crops)
            logger.info("Indexed batch %d-%d / %d", i + 1, min(i + batch_size, len(all_paths)), len(all_paths))

        self.indexer._cache_valid = False
        logger.info("Index built: %d vectors from %d SKUs", total_added, len({p[1] for p in all_paths}))
        return total_added
