from typing import Dict, List

from PIL import Image

from app.resnet_search_engine import ResNetImageSearchEngine
from app.search_engine import ImageSearchEngine


class HybridImageSearchEngine:
    """Fuse semantic CLIP and visual ResNet rankings for image retrieval."""

    def __init__(
        self,
        clip_engine: ImageSearchEngine,
        resnet_engine: ResNetImageSearchEngine,
        clip_weight: float = 0.4,
        resnet_weight: float = 0.6,
        rrf_constant: int = 60,
    ) -> None:
        if clip_weight < 0 or resnet_weight < 0 or clip_weight + resnet_weight <= 0:
            raise ValueError("Hybrid weights must be non-negative with a positive sum.")
        self.clip_engine = clip_engine
        self.resnet_engine = resnet_engine
        total_weight = clip_weight + resnet_weight
        self.clip_weight = clip_weight / total_weight
        self.resnet_weight = resnet_weight / total_weight
        self.rrf_constant = rrf_constant

    def search_by_image(self, image: Image.Image, top_k: int = 5) -> List[Dict]:
        """Retrieve a broad candidate pool and fuse model ranks without score scaling."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        candidate_count = min(max(top_k * 10, 50), self.clip_engine.index.ntotal)
        clip_results = self.clip_engine.search_by_image(image, top_k=candidate_count)
        resnet_results = self.resnet_engine.search_by_image(image, top_k=candidate_count)

        candidates: Dict[str, Dict] = {}
        for model_name, weight, results in (
            ("clip", self.clip_weight, clip_results),
            ("resnet", self.resnet_weight, resnet_results),
        ):
            for rank, product in enumerate(results, start=1):
                product_id = str(product["id"])
                candidate = candidates.setdefault(
                    product_id,
                    {
                        **product,
                        "hybrid_score": 0.0,
                        "clip_rank": None,
                        "resnet_rank": None,
                        "clip_similarity": None,
                        "resnet_similarity": None,
                    },
                )
                candidate["hybrid_score"] += weight / (self.rrf_constant + rank)
                candidate[f"{model_name}_rank"] = rank
                candidate[f"{model_name}_similarity"] = product["similarity_score"]

        ranked = sorted(
            candidates.values(),
            key=lambda product: product["hybrid_score"],
            reverse=True,
        )[:top_k]

        # Normalize against the maximum possible fused score for readable UI values.
        maximum_score = 1.0 / (self.rrf_constant + 1)
        for product in ranked:
            product["similarity_score"] = product["hybrid_score"] / maximum_score
        return ranked
