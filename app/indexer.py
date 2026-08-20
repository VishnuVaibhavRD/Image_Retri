import json
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from app.config import (
    BASE_DIR,
    EMBEDDING_BATCH_SIZE,
    FAISS_INDEX_PATH,
    INDEXED_METADATA_PATH,
)
from app.embedding_model import CLIPEmbeddingModel


class FashionImageIndexer:
    """Build and save a FAISS index whose rows align with product metadata."""

    def __init__(
        self,
        embedding_model: CLIPEmbeddingModel,
        batch_size: int = EMBEDDING_BATCH_SIZE,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.embedding_model = embedding_model
        self.batch_size = batch_size

    @staticmethod
    def load_metadata(metadata_path: Path) -> List[Dict]:
        """Load product metadata from a JSON array."""
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with metadata_path.open("r", encoding="utf-8") as file:
            products = json.load(file)

        if not isinstance(products, list):
            raise ValueError("Metadata must contain a JSON array of products.")
        if not all(isinstance(product, dict) for product in products):
            raise ValueError("Every metadata entry must be a JSON object.")
        if not products:
            raise ValueError("Metadata contains no products to index.")

        return products

    def build_index(self, products: List[Dict]) -> Tuple[faiss.Index, List[Dict]]:
        """Embed valid product images in batches and add them to FAISS."""
        dimension = self.embedding_model.embedding_dimension
        index = faiss.IndexFlatIP(dimension)
        indexed_products: List[Dict] = []

        print(f"Products available: {len(products)}")
        print(f"Batch size: {self.batch_size}")
        print(f"Embedding dimension: {dimension}")

        batches = range(0, len(products), self.batch_size)
        for start in tqdm(
            batches,
            total=(len(products) + self.batch_size - 1) // self.batch_size,
            desc="Creating image embeddings",
        ):
            batch_products = products[start : start + self.batch_size]
            images = []
            valid_products = []

            for product in batch_products:
                relative_path = product.get("local_image_path")
                if not relative_path:
                    print(f"Skipping product {product.get('id', '<unknown>')}: no image path")
                    continue

                image_path = BASE_DIR / str(relative_path)
                try:
                    with Image.open(image_path) as image:
                        images.append(image.convert("RGB"))
                    valid_products.append(product)
                except (FileNotFoundError, OSError, UnidentifiedImageError) as error:
                    print(f"Skipping product {product.get('id', '<unknown>')}: {error}")

            if not images:
                continue

            embeddings = self.embedding_model.encode_images(images)
            if embeddings.ndim != 2:
                raise ValueError("Embeddings must be a two-dimensional array.")
            if embeddings.shape[0] != len(valid_products):
                raise ValueError(
                    "The number of embeddings does not match the number of valid products."
                )
            if embeddings.shape[1] != dimension:
                raise ValueError(
                    f"Expected embedding dimension {dimension}, got {embeddings.shape[1]}."
                )

            embeddings = np.ascontiguousarray(embeddings, dtype="float32")
            index.add(embeddings)
            indexed_products.extend(valid_products)

        if index.ntotal == 0:
            raise ValueError("No images were successfully added to the index.")
        if index.ntotal != len(indexed_products):
            raise ValueError("FAISS vector count and metadata count do not match.")

        print(f"Successfully indexed {index.ntotal} images.")
        return index, indexed_products

    @staticmethod
    def save_index(index: faiss.Index, indexed_products: List[Dict]) -> None:
        """Save the vector index and its matching metadata."""
        if index.ntotal != len(indexed_products):
            raise ValueError("Refusing to save a misaligned index and metadata file.")

        FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        INDEXED_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(index, str(FAISS_INDEX_PATH))
        with INDEXED_METADATA_PATH.open("w", encoding="utf-8") as file:
            json.dump(
                indexed_products,
                file,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

        print(f"FAISS index saved to: {FAISS_INDEX_PATH}")
        print(f"Indexed metadata saved to: {INDEXED_METADATA_PATH}")

    def run(self, metadata_path: Path) -> None:
        products = self.load_metadata(metadata_path)
        index, indexed_products = self.build_index(products)
        self.save_index(index, indexed_products)
