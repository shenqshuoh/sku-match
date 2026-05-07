from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image
from ultralytics import YOLOE

from src.classes.beverage_cls import BEVERAGE_CONTAINER_CLASSES
from src.embedder import DINOv2Embedder
from src.image_utils import draw_annotations, extract_binary_masks, isolate_object
from src.indexer import SKUIndexer

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
    ):
        self.detector = detector
        self.embedder = embedder
        self.indexer = indexer
        self.image_storage = image_storage
        self.det_conf = det_conf
        self.imgsz = imgsz
        self.match_conf = match_conf

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
            source=str(image_path),
            device=device,
            conf=self.det_conf,
            imgsz=self.imgsz,
            retina_masks=True,
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

        crops = []
        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = map(int, det["bbox"])
            other_masks = [m for j, m in enumerate(all_masks) if j != idx and m is not None]
            isolated = isolate_object(image_np, det["mask"], other_masks)
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
        distributions = self.indexer.search_batch(embeddings)

        counts: dict[str, int] = {}
        detection_items = []
        for i, (det, distribution) in enumerate(zip(detections, distributions), start=1):
            ranked = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
            sku_id, confidence = ranked[0]
            sku_name = self.indexer.get_sku_name(sku_id) or sku_id
            match_ratio = confidence / ranked[1][1] if len(ranked) > 1 and ranked[1][1] > 0 else 0.0
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
                "match_ratio": match_ratio,
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
