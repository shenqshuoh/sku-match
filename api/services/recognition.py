from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLOE

from src.color import extract_color_descriptor
from src.core import parse_detections
from src.embedder import Embedder
from src.image_utils import draw_annotations
from src.indexer import SKUIndexer, _softmax, concentration_score, score_matches
from src.patch_store import PatchStore
from src.reranker import rerank

if TYPE_CHECKING:
    from api.services.image_storage import ImageStorage
    from src.color_store import ColorStore

logger = logging.getLogger(__name__)


class RecognitionService:

    def __init__(
        self,
        detector: YOLOE,
        embedder: Embedder,
        indexer: SKUIndexer,
        image_storage: "ImageStorage",
        det_conf: float = 0.25,
        imgsz: int = 1280,
        match_conf: float = 0.5,
        concentration_topk: int = 10,
        distribution_top_k: int = 10,
        patch_store: PatchStore | None = None,
        use_reranking: bool = True,
        rerank_top_k: int = 50,
        rerank_blend_beta: float = 0.3,
        color_store: "ColorStore | None" = None,
        use_color: bool = False,
        color_gamma: float = 0.0,
        color_bins: int = 16,
    ):
        self.detector = detector
        self.embedder = embedder
        self.indexer = indexer
        self.image_storage = image_storage
        self.det_conf = det_conf
        self.imgsz = imgsz
        self.match_conf = match_conf
        self.concentration_topk = concentration_topk
        self.distribution_top_k = distribution_top_k
        self.patch_store = patch_store
        self.use_reranking = use_reranking
        self.rerank_top_k = rerank_top_k
        self.rerank_blend_beta = rerank_blend_beta
        self.color_store = color_store
        self.use_color = use_color
        self.color_gamma = color_gamma
        self.color_bins = color_bins

        # Fail fast at startup: rerank() enforces blend_beta + color_gamma <= 1.0
        if self.use_color and (self.rerank_blend_beta + self.color_gamma > 1.0 + 1e-9):
            raise ValueError(
                f"rerank_blend_beta + color_gamma must be <= 1.0 "
                f"(got beta={self.rerank_blend_beta}, gamma={self.color_gamma})"
            )

    def _score_with_reranking(
        self,
        detections: list,
        search_results: list[tuple],
        features_list: list,
        query_color_descriptors: list | None = None,
    ) -> list:
        """Score detections using patch re-ranking with blended scoring.

        For each detection:
        1. Run patch re-ranking on the coarse candidates.
        2. Apply softmax over blended scores to get a probability distribution.
        3. Build SKUMatch results from the blended distribution.
        """
        from src.types import SKUMatch

        all_matches = []
        for i, (det, (coarse_dist, rank_info, top_vectors), features) in enumerate(
            zip(detections, search_results, features_list)
        ):
            # Run re-ranking
            rerank_results = rerank(
                query_patches=features.patches,
                patch_store=self.patch_store,
                top_vectors=top_vectors,
                blend_beta=self.rerank_blend_beta,
                color_store=self.color_store if self.use_color else None,
                query_color_descriptor=(
                    query_color_descriptors[i] if query_color_descriptors else None
                ),
                color_gamma=self.color_gamma,
            )

            if not rerank_results:
                # Fallback: no patch files available, use coarse scores
                matches = score_matches([det], [(coarse_dist, rank_info, top_vectors)], self.indexer, self.concentration_topk)
                all_matches.extend(matches)
                continue

            # Build blended distribution from rerank results
            blended_scores = {r.sku_id: r.blended_score for r in rerank_results}
            distribution = _softmax(blended_scores, self.indexer._temperature)

            # Top-1 SKU from blended distribution
            ranked = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
            sku_id, confidence = ranked[0]
            sku_name = self.indexer.get_sku_name(sku_id) or sku_id
            conc = concentration_score(distribution, top_k=self.concentration_topk)

            # Get rank positions for the top SKU from coarse results
            positions = rank_info.get(sku_id, [])
            top2_ranks = (
                (positions[0], positions[1]) if len(positions) >= 2
                else (positions[0], 0) if len(positions) == 1
                else (0, 0)
            )

            all_matches.append(SKUMatch(
                detection=det,
                sku_id=sku_id,
                sku_name=sku_name,
                match_score=confidence,
                match_concentration=conc,
                top2_ranks=top2_ranks,
                sku_distribution=distribution,
                top_vectors=top_vectors,
            ))

        return all_matches

    def recognize(
        self,
        image_path: Path,
        task_id: str,
        roi_rect: list[float] | None = None,
        device: str = "cpu",
    ) -> dict:
        image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
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
                "matchedImage": self.image_storage.get_result_url(task_id),
                "taskId": task_id,
            }

        # Crop detections from image
        crops = []
        for det in detections:
            x1, y1, x2, y2 = map(int, det.bbox)
            crop = Image.fromarray(image_np[y1:y2, x1:x2])
            crops.append(crop)

        # Extract query color descriptors when color re-ranking is active
        query_color_descriptors: list | None = None
        if self.use_color and self.color_store is not None:
            query_color_descriptors = [
                extract_color_descriptor(np.array(c), n_bins=self.color_bins) for c in crops
            ]

        # Extract features (CLS + patches in single forward pass)
        features_list = self.embedder.extract_features_batch(crops)
        embeddings = np.stack([self.embedder.features_to_embedding(f) for f in features_list])

        # Stage 1: Coarse retrieval from ChromaDB
        n_override = self.rerank_top_k if self.use_reranking and self.patch_store is not None else None
        search_results = self.indexer.search_batch(embeddings, n_results_override=n_override)

        if self.use_reranking and self.patch_store is not None:
            # Stage 2: Patch re-ranking with blended scoring
            matches = self._score_with_reranking(
                detections, search_results, features_list,
                query_color_descriptors=query_color_descriptors,
            )
        else:
            # Standard scoring pipeline
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
            top_n = ranked[:self.distribution_top_k] if confidence >= self.match_conf else []
            top_distribution = {
                sid: {"skuName": self.indexer.get_sku_name(sid) or sid, "score": prob}
                for sid, prob in top_n
            }

            detection_items.append({
                "itemId": i,
                "bbox": list(match.detection.bbox),
                "classId": match.detection.class_id,
                "className": match.detection.class_name,
                "detectionConf": match.detection.confidence,
                "skuId": sku_id,
                "skuName": sku_name,
                "matchScore": confidence,
                "matchConcentration": match_concentration,
                "skuDistribution": top_distribution,
                "matchedVectorTags": matched_vector_tags,
            })

        annotated_path = self.image_storage.get_result_path(task_id)
        draw_annotations(image_np, detection_items, annotated_path)

        return {
            "counts": counts,
            "detections": detection_items,
            "matchedImage": self.image_storage.get_result_url(task_id),
            "taskId": task_id,
        }
