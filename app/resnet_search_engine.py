import json
from pathlib import Path
from typing import Dict, List, Union

import faiss
import numpy as np
from PIL import Image

from app.config import BASE_DIR, RESNET_FAISS_INDEX_PATH, RESNET_INDEXED_METADATA_PATH
from app.resnet_embedding_model import ResNetEmbeddingModel


class ResNetImageSearchEngine:
    """Image-to-image search using ResNet visual features."""

    def __init__(self, embedding_model: ResNetEmbeddingModel) -> None:
        self.embedding_model = embedding_model
        self.index = faiss.read_index(str(RESNET_FAISS_INDEX_PATH))
        self.products = json.loads(RESNET_INDEXED_METADATA_PATH.read_text(encoding="utf-8"))
        if self.index.ntotal != len(self.products):
            raise ValueError("ResNet index and metadata counts do not match.")

    def search_by_image(
        self,
        image: Union[Image.Image, str, Path],
        top_k: int = 5,
    ) -> List[Dict]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if isinstance(image, (str, Path)):
            path = Path(image)
            if not path.is_absolute():
                path = BASE_DIR / path
            with Image.open(path) as opened:
                query_image = opened.convert("RGB")
        elif isinstance(image, Image.Image):
            query_image = image.convert("RGB")
        else:
            raise TypeError("image must be a PIL Image or path.")

        query = self.embedding_model.encode_images(query_image)
        scores, positions = self.index.search(
            np.ascontiguousarray(query, dtype="float32"),
            min(top_k, self.index.ntotal),
        )
        results = []
        for score, position in zip(scores[0], positions[0]):
            product = dict(self.products[int(position)])
            product["similarity_score"] = float(score)
            results.append(product)
        return results
