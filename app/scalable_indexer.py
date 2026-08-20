import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from app.config import (
    BACKUP_FAISS_INDEX_PATH,
    BACKUP_INDEXED_METADATA_PATH,
    BASE_DIR,
    EMBEDDING_BATCH_SIZE,
    FAISS_CHECKPOINT_PATH,
    FAISS_INDEX_PATH,
    INDEX_CHECKPOINT_BATCHES,
    INDEXED_METADATA_PATH,
    METADATA_CHECKPOINT_PATH,
)
from app.embedding_model import CLIPEmbeddingModel
from app.ingestion import atomic_json_dump


class CheckpointedFashionIndexer:
    """Build a resumable exact FAISS index while preserving row alignment."""

    def __init__(
        self,
        embedding_model: CLIPEmbeddingModel,
        batch_size: int = EMBEDDING_BATCH_SIZE,
        checkpoint_batches: int = INDEX_CHECKPOINT_BATCHES,
    ) -> None:
        if batch_size < 1 or checkpoint_batches < 1:
            raise ValueError("Batch and checkpoint sizes must be at least 1.")
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.checkpoint_batches = checkpoint_batches

    @staticmethod
    def _load_json(path: Path) -> List[Dict]:
        with path.open("r", encoding="utf-8") as file:
            value = json.load(file)
        if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
            raise ValueError(f"Expected a JSON array of objects: {path}")
        return value

    def _load_checkpoint(self, products: List[Dict]) -> Tuple[faiss.Index, List[Dict]]:
        dimension = self.embedding_model.embedding_dimension
        if not (FAISS_CHECKPOINT_PATH.exists() and METADATA_CHECKPOINT_PATH.exists()):
            return faiss.IndexFlatIP(dimension), []

        index = faiss.read_index(str(FAISS_CHECKPOINT_PATH))
        metadata = self._load_json(METADATA_CHECKPOINT_PATH)
        checkpoint_ids = [str(item.get("id")) for item in metadata]
        source_ids = [str(item.get("id")) for item in products[: len(metadata)]]
        if index.d != dimension or index.ntotal != len(metadata) or checkpoint_ids != source_ids:
            print("Checkpoint is stale or incomplete; rebuilding the index from product 0.")
            return faiss.IndexFlatIP(dimension), []

        print(f"Resuming index from {index.ntotal} checkpointed products.")
        return index, metadata

    @staticmethod
    def _write_faiss_atomic(index: faiss.Index, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        faiss.write_index(index, str(temporary_path))
        os.replace(temporary_path, path)

    def _save_checkpoint(self, index: faiss.Index, metadata: List[Dict]) -> None:
        if index.ntotal != len(metadata):
            raise ValueError("Cannot checkpoint a misaligned index and metadata list.")
        self._write_faiss_atomic(index, FAISS_CHECKPOINT_PATH)
        atomic_json_dump(metadata, METADATA_CHECKPOINT_PATH)

    @staticmethod
    def preserve_small_index() -> None:
        """Copy the original working index once, without overwriting its backup."""
        if FAISS_INDEX_PATH.exists() and not BACKUP_FAISS_INDEX_PATH.exists():
            shutil.copy2(FAISS_INDEX_PATH, BACKUP_FAISS_INDEX_PATH)
            print(f"Backed up existing index to: {BACKUP_FAISS_INDEX_PATH}")
        if INDEXED_METADATA_PATH.exists() and not BACKUP_INDEXED_METADATA_PATH.exists():
            shutil.copy2(INDEXED_METADATA_PATH, BACKUP_INDEXED_METADATA_PATH)
            print(f"Backed up existing metadata to: {BACKUP_INDEXED_METADATA_PATH}")

    def build(self, products: List[Dict]) -> Tuple[faiss.Index, List[Dict]]:
        if not products:
            raise ValueError("No products are available for indexing.")
        product_ids = [str(product.get("id")) for product in products]
        if len(product_ids) != len(set(product_ids)):
            raise ValueError("Clean metadata contains duplicate product IDs.")

        index, indexed_products = self._load_checkpoint(products)
        start = len(indexed_products)
        remaining_starts = range(start, len(products), self.batch_size)
        total_batches = (len(products) - start + self.batch_size - 1) // self.batch_size

        for batch_number, batch_start in enumerate(
            tqdm(remaining_starts, total=total_batches, desc="Indexing products"),
            start=1,
        ):
            batch = products[batch_start : batch_start + self.batch_size]
            images = []
            for product in batch:
                image_path = BASE_DIR / str(product["local_image_path"])
                try:
                    with Image.open(image_path) as image:
                        images.append(image.convert("RGB"))
                except (FileNotFoundError, OSError, UnidentifiedImageError) as error:
                    raise ValueError(
                        f"Validated image became unavailable for product {product.get('id')}: {error}"
                    ) from error

            embeddings = self.embedding_model.encode_images(images)
            expected_shape = (len(batch), self.embedding_model.embedding_dimension)
            if embeddings.shape != expected_shape:
                raise ValueError(
                    f"Expected embedding shape {expected_shape}, got {embeddings.shape}."
                )
            if not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5):
                raise ValueError("Encountered a non-normalized embedding batch.")

            index.add(np.ascontiguousarray(embeddings, dtype="float32"))
            for product in batch:
                aligned_product = dict(product)
                aligned_product["index_position"] = len(indexed_products)
                indexed_products.append(aligned_product)

            if batch_number % self.checkpoint_batches == 0:
                self._save_checkpoint(index, indexed_products)
                print(f"Checkpoint saved at {index.ntotal} vectors.")

        if index.ntotal != len(products) or index.ntotal != len(indexed_products):
            raise ValueError("Final vector and metadata counts do not match source products.")
        if any(item["index_position"] != position for position, item in enumerate(indexed_products)):
            raise ValueError("Metadata index positions are not contiguous and aligned.")

        self._save_checkpoint(index, indexed_products)
        return index, indexed_products

    def finalize(self, index: faiss.Index, indexed_products: List[Dict]) -> None:
        self.preserve_small_index()
        self._write_faiss_atomic(index, FAISS_INDEX_PATH)
        atomic_json_dump(indexed_products, INDEXED_METADATA_PATH)

        persisted_index = faiss.read_index(str(FAISS_INDEX_PATH))
        persisted_metadata = self._load_json(INDEXED_METADATA_PATH)
        if persisted_index.ntotal != len(persisted_metadata) or persisted_index.d != index.d:
            raise ValueError("Persisted final index failed alignment validation.")
        if any(item.get("index_position") != i for i, item in enumerate(persisted_metadata)):
            raise ValueError("Persisted metadata positions failed validation.")

        print(f"Final index saved with {persisted_index.ntotal} vectors: {FAISS_INDEX_PATH}")
        print(f"Aligned metadata saved to: {INDEXED_METADATA_PATH}")
