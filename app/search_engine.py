import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import faiss
import numpy as np
from PIL import Image

from app.config import BASE_DIR, FAISS_INDEX_PATH, INDEXED_METADATA_PATH
from app.embedding_model import CLIPEmbeddingModel


class ImageSearchEngine:
    """Search an aligned FAISS image index using text or image queries."""

    def __init__(
        self,
        embedding_model: CLIPEmbeddingModel,
        index_path: Path = FAISS_INDEX_PATH,
        metadata_path: Path = INDEXED_METADATA_PATH,
    ) -> None:
        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {index_path}. Run python -m scripts.build_index first."
            )
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Indexed metadata not found: {metadata_path}. "
                "Run python -m scripts.build_index first."
            )

        self.embedding_model = embedding_model
        self.index = faiss.read_index(str(index_path))
        with metadata_path.open("r", encoding="utf-8") as file:
            self.products = json.load(file)

        if not isinstance(self.products, list):
            raise ValueError("Indexed metadata must be a JSON array.")
        if self.index.ntotal != len(self.products):
            raise ValueError("FAISS vector count and indexed metadata count do not match.")
        if self.index.d != self.embedding_model.embedding_dimension:
            raise ValueError(
                f"Index dimension {self.index.d} does not match model dimension "
                f"{self.embedding_model.embedding_dimension}."
            )

        print("Image search engine loaded successfully.")
        print(f"Searchable products: {self.index.ntotal}")

    def _search_embedding(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        exclude_product_id: Optional[str] = None,
    ) -> List[Dict]:
        """Search FAISS and attach similarity scores to aligned metadata."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if query_embedding.ndim != 2 or query_embedding.shape != (1, self.index.d):
            raise ValueError(f"Query embedding must have shape (1, {self.index.d}).")

        excluded_id = None if exclude_product_id is None else str(exclude_product_id)
        search_count = min(
            self.index.ntotal,
            top_k + (1 if excluded_id is not None else 0),
        )
        scores, positions = self.index.search(
            np.ascontiguousarray(query_embedding, dtype="float32"),
            search_count,
        )

        results = []
        for score, position in zip(scores[0], positions[0]):
            if position < 0:
                continue
            product = dict(self.products[int(position)])
            if excluded_id is not None and str(product.get("id")) == excluded_id:
                continue

            product["similarity_score"] = float(score)
            results.append(product)
            if len(results) == top_k:
                break

        return results

    def search_by_text(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve images that are semantically relevant to text."""
        query = query.strip()
        if not query:
            raise ValueError("The text query cannot be empty.")

        query_embedding = self.embedding_model.encode_texts(query)
        return self._search_embedding(query_embedding=query_embedding, top_k=top_k)

    def search_by_image(
        self,
        image: Union[Image.Image, str, Path],
        top_k: int = 5,
        exclude_product_id: Optional[str] = None,
    ) -> List[Dict]:
        """Retrieve images visually similar to a query image."""
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            if not image_path.is_absolute():
                image_path = BASE_DIR / image_path
            if not image_path.exists():
                raise FileNotFoundError(f"Query image not found: {image_path}")
            with Image.open(image_path) as opened_image:
                query_image = opened_image.convert("RGB").copy()
        elif isinstance(image, Image.Image):
            query_image = image.convert("RGB")
        else:
            raise TypeError("image must be a PIL Image or a filesystem path.")

        query_embedding = self.embedding_model.encode_images(query_image)
        return self._search_embedding(
            query_embedding=query_embedding,
            top_k=top_k,
            exclude_product_id=exclude_product_id,
        )
