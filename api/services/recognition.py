from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image
from ultralytics import YOLOE

from src.core import parse_detections
from src.embedder import DINOv2Embedder
from src.image_utils import draw_annotations
from src.indexer import SKUIndexer, score_matches

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

        detections = parse_detections(result)

        # Apply ROI filter if specified
        if roi_rect is not None and detections:
            rx1, ry1, rx2, ry2 = roi_rect
            detections = [
                d for d in detections
                if rx1 <= (d.bbox[0] + d.bbox[2]) / 2 <= rx2
                and ry1 <= (d.bbox[1] + d.bbox[3]) / 2 <= ry2
            ]

        if not detections:
            annotated_path = self.image_storage.get_result_path(task_id)
            draw_annotations(image_np, [], annotated_path)
            return {
                "counts": {},
                "detections": [],
                "matched_image": self.image_storage.get_result_url(task_id),
                "taskId": task_id,
            }

        # Crop detections from image
        crops = []
        for det in detections:
            x1, y1, x2, y2 = map(int, det.bbox)
            crop = Image.fromarray(image_np[y1:y2, x1:x2])
            crops.append(crop)

        # Embed + search + score using shared pipeline
        embeddings = self.embedder.embed_batch(crops)
        search_results = self.indexer.search_batch(embeddings)
        matches = score_matches(detections, search_results, self.indexer, self.concentration_topk)

        # Format results with API-specific threshold filtering
        counts: dict[str, int] = {}
        detection_items = []
        for i, match in enumerate(matches, start=1):
            sku_id = match.sku_id
            confidence = match.match_score

            if confidence < self.match_conf:
                sku_id = ""
                sku_name = ""
                confidence = 0.0
                match_concentration = 0.0
                matched_vector_tags = []
            else:
                sku_name = match.sku_name
                match_concentration = match.match_concentration
                matched_vector_tags = [
                    {"skuId": vm.sku_id, "score": round(vm.similarity, 6), "mediaUrl": vm.media_url}
                    for vm in (match.top_vectors or [])[:20]
                ]

            if sku_id:
                counts[sku_id] = counts.get(sku_id, 0) + 1

            ranked = sorted((match.sku_distribution or {}).items(), key=lambda x: x[1], reverse=True)
            top5_distribution = dict(ranked[:5]) if confidence >= self.match_conf else {}

            detection_items.append({
                "itemId": i,
                "bbox": list(match.detection.bbox),
                "class_id": match.detection.class_id,
                "class_name": match.detection.class_name,
                "detection_conf": match.detection.confidence,
                "sku_id": sku_id,
                "sku_name": sku_name,
                "match_score": confidence,
                "match_concentration": match_concentration,
                "sku_distribution": top5_distribution,
                "matched_vector_tags": matched_vector_tags,
            })

        annotated_path = self.image_storage.get_result_path(task_id)
        draw_annotations(image_np, detection_items, annotated_path)

        return {
            "counts": counts,
            "detections": detection_items,
            "matched_image": self.image_storage.get_result_url(task_id),
            "taskId": task_id,
        }
