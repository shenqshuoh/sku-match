"""Reference image processing: detect, crop, embed, and index."""

import logging
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLOE

from src.embedder import DINOv2Embedder
from src.image_utils import extract_binary_masks, isolate_object
from src.indexer import SKUIndexer

logger = logging.getLogger(__name__)


def select_best_detection(boxes, img_w: int, img_h: int) -> int | None:
    """Select the best detection from results — prefers center, smallest containing box."""
    if not boxes:
        return None

    cx, cy = img_w / 2, img_h / 2
    candidates = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            area = (x2 - x1) * (y2 - y1)
            candidates.append((i, area))

    if candidates:
        return min(candidates, key=lambda x: x[1])[0]

    # Fall back to closest center
    min_dist = float("inf")
    min_idx = 0
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        bx, by = (x1 + x2) / 2, (y1 + y2) / 2
        dist_sq = (bx - cx) ** 2 + (by - cy) ** 2
        if dist_sq < min_dist:
            min_dist = dist_sq
            min_idx = i

    return min_idx


class ReferenceProcessor:
    """Processes raw reference images: detect → crop → embed → index."""

    def __init__(
        self,
        detector: YOLOE,
        embedder: DINOv2Embedder,
        indexer: SKUIndexer,
        device: str = "cpu",
        det_conf: float = 0.25,
        use_mask: bool = True,
    ):
        self.detector = detector
        self.embedder = embedder
        self.indexer = indexer
        self.device = device
        self.det_conf = det_conf
        self.use_mask = use_mask

    def crop_reference(self, image_path: Path) -> Image.Image | None:
        """Detect the primary object in a reference image and return a cropped PIL Image.

        Returns None if no detection is found.
        """
        results = self.detector.predict(
            source=str(image_path),
            device=self.device,
            conf=self.det_conf,
            retina_masks=True,
            verbose=False,
        )
        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            logger.warning("No detections in reference image: %s", image_path)
            return None

        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
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
            isolated = isolate_object(image_np, mask, other_masks)
            crop = Image.fromarray(isolated[y1:y2, x1:x2])
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
    ) -> bool:
        """Full pipeline: crop → embed → add to index.

        Falls back to embedding the full image if detection fails.
        Returns True if successful.
        """
        crop = self.crop_reference(image_path)

        if crop is not None:
            embedding = self.embedder.embed(crop)
            logger.info("Cropped and embedded reference %s/%s", sku_id, media_id)
        else:
            # Fallback: embed the full image
            image = Image.open(image_path).convert("RGB")
            embedding = self.embedder.embed(image)
            logger.info(
                "No crop available, embedded full image for %s/%s", sku_id, media_id
            )

        self.indexer.add_reference(sku_id, sku_name, media_id, embedding, metadata)
        logger.info("Added reference %s/%s to index", sku_id, media_id)
        return True

    def build_from_directory(
        self,
        reference_dir: Path,
        batch_size: int = 16,
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
                else:
                    # Fallback to full image
                    crops.append(Image.open(img_path).convert("RGB"))
                batch_meta.append((sku_id, sku_id, f"{sku_id}__{total_added + len(batch_meta):04d}"))

            embeddings = self.embedder.embed_batch(crops)

            ids = [m[2] for m in batch_meta]
            emb_list = [
                e.tolist() if isinstance(e, np.ndarray) else list(e)
                for e in embeddings
            ]
            metadatas = [
                {"sku_id": m[0], "sku_name": m[1], "enabled": True}
                for m in batch_meta
            ]

            self.indexer.collection.add(ids=ids, embeddings=emb_list, metadatas=metadatas)
            total_added += len(batch)
            logger.info("Indexed batch %d-%d / %d", i + 1, min(i + batch_size, len(all_paths)), len(all_paths))

        self.indexer._cache_valid = False
        logger.info("Index built: %d vectors from %d SKUs", total_added, len({p[1] for p in all_paths}))
        return total_added

    def delete_sku_references(self, sku_id: str) -> None:
        self.indexer.delete_sku(sku_id)
        logger.info("Deleted all references for SKU %s", sku_id)

    def delete_media_reference(self, sku_id: str, media_id: str) -> None:
        self.indexer.delete_media(sku_id, media_id)
        logger.info("Deleted media %s/%s", sku_id, media_id)

    def set_sku_enabled(self, sku_id: str, enabled: bool) -> None:
        self.indexer.set_enabled(sku_id, enabled)
        logger.info("SKU %s enabled=%s", sku_id, enabled)

    def update_sku_name(self, sku_id: str, new_name: str) -> None:
        results = self.indexer.collection.get(
            where={"sku_id": sku_id}, include=["metadatas"]
        )
        if not results["ids"]:
            return
        updated = [{**m, "sku_name": new_name} for m in results["metadatas"]]
        self.indexer.collection.update(ids=results["ids"], metadatas=updated)
        self.indexer._cache_valid = False
        logger.info("Updated sku_name to '%s' for SKU %s (%d vectors)", new_name, sku_id, len(results["ids"]))
