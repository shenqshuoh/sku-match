from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLOE

from src.classes.beverage_cls import BEVERAGE_CONTAINER_CLASSES
from src.embedder import DINOv2Embedder
from src.image_utils import draw_annotations, extract_binary_masks, isolate_object
from src.indexer import SKUIndexer, concentration_score

if TYPE_CHECKING:
    from api.services.image_storage import ImageStorage

logger = logging.getLogger(__name__)


class RecognitionService:

    def __init__(
        self,
        detector: YOLOE,
        embedder: DINOv2Embedder,
        indexer: SKUIndexer,
        image_storage: "ImageStorage",
        det_conf: float = 0.25,
        imgsz: int = 1280,
        match_conf: float = 0.5,
        concentration_topk: int = 10,
    ):
        self.detector = detector
        self.embedder = embedder
        self.indexer = indexer
        self.image_storage = image_storage
        self.det_conf = det_conf
        self.imgsz = imgsz
        self.match_conf = match_conf
        self.concentration_topk = concentration_topk

    def recognize(
        self,
        image_path: Path,
        task_id: str,
        roi_rect: list[float] | None = None,
        device: str = "cpu",
    ) -> dict:
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        results = self.detector.predict(
            source=image_np,
            device=device,
            conf=self.det_conf,
            imgsz=self.imgsz,
            retina_masks=False,
            verbose=False,
        )
        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            annotated_path = self.image_storage.get_result_path(task_id)
            draw_annotations(image_np, [], annotated_path)
            return {
                "counts": {},
                "detections": [],
                "matched_image": self.image_storage.get_result_url(task_id),
                "taskId": task_id,
            }

        binary_masks = extract_binary_masks(result)
        all_masks = binary_masks

        detections = []
        for i, (box, mask) in enumerate(zip(result.boxes, binary_masks)):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            class_name = BEVERAGE_CONTAINER_CLASSES[cls_id] if cls_id < len(BEVERAGE_CONTAINER_CLASSES) else str(cls_id)

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": float(box.conf[0]),
                "class_name": class_name,
                "class_id": cls_id,
                "mask": mask,
            })

        if roi_rect is not None:
            rx1, ry1, rx2, ry2 = roi_rect
            detections = [
                d for d in detections
                if rx1 <= (d["bbox"][0] + d["bbox"][2]) / 2 <= rx2
                and ry1 <= (d["bbox"][1] + d["bbox"][3]) / 2 <= ry2
            ]

        # Pre-compute union of all masks for O(D) instead of O(D²) masking
        union_mask = None
        if all_masks and any(m is not None for m in all_masks):
            first_valid = next(m for m in all_masks if m is not None)
            union_mask = np.zeros_like(first_valid, dtype=np.uint8)
            for m in all_masks:
                if m is not None:
                    m_uint8 = (m * 255).astype(np.uint8) if m.max() <= 1 else m.astype(np.uint8)
                    union_mask = cv2.bitwise_or(union_mask, m_uint8)

        crops = []
        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = map(int, det["bbox"])
            # Exclusion = union minus own mask
            exclusion = None
            if union_mask is not None and det["mask"] is not None:
                own_mask = det["mask"]
                own_uint8 = (own_mask * 255).astype(np.uint8) if own_mask.max() <= 1 else own_mask.astype(np.uint8)
                exclusion = cv2.bitwise_and(union_mask, cv2.bitwise_not(own_uint8))
            isolated = isolate_object(image_np, det["mask"], [exclusion] if exclusion is not None else None)
            crop = Image.fromarray(isolated[y1:y2, x1:x2])
            crops.append(crop)

        if not crops:
            annotated_path = self.image_storage.get_result_path(task_id)
            draw_annotations(image_np, [], annotated_path)
            return {
                "counts": {},
                "detections": [],
                "matched_image": self.image_storage.get_result_url(task_id),
                "taskId": task_id,
            }

        embeddings = self.embedder.embed_batch(crops)
        search_results = self.indexer.search_batch(embeddings)

        counts: dict[str, int] = {}
        detection_items = []
        for i, (det, (distribution, _rank_info)) in enumerate(zip(detections, search_results), start=1):
            ranked = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
            sku_id, confidence = ranked[0]
            sku_name = self.indexer.get_sku_name(sku_id) or sku_id
            match_concentration = concentration_score(distribution, top_k=self.concentration_topk)
            counts[sku_id] = counts.get(sku_id, 0) + 1
            detection_items.append({
                "itemId": i,
                "bbox": det["bbox"],
                "class_id": det["class_id"],
                "class_name": det["class_name"],
                "detection_conf": det["confidence"],
                "sku_id": sku_id,
                "sku_name": sku_name,
                "match_score": confidence,
                "match_concentration": match_concentration,
                "sku_distribution": distribution,
            })

        annotated_path = self.image_storage.get_result_path(task_id)
        draw_annotations(image_np, detection_items, annotated_path)

        return {
            "counts": counts,
            "detections": detection_items,
            "matched_image": self.image_storage.get_result_url(task_id),
            "taskId": task_id,
        }
