import json
import os
from typing import Dict, List, Tuple

import faiss
import numpy as np
from PIL import Image
from tqdm import tqdm

from app.config import (
    BASE_DIR,
    INDEX_CHECKPOINT_BATCHES,
    RESNET_BATCH_SIZE,
    RESNET_FAISS_CHECKPOINT_PATH,
    RESNET_FAISS_INDEX_PATH,
    RESNET_INDEXED_METADATA_PATH,
    RESNET_METADATA_CHECKPOINT_PATH,
)
from app.ingestion import atomic_json_dump
from app.resnet_embedding_model import ResNetEmbeddingModel


class ResNetFashionIndexer:
    """Build a resumable ResNet image-similarity index alongside CLIP."""

    def __init__(
        self,
        embedding_model: ResNetEmbeddingModel,
        batch_size: int = RESNET_BATCH_SIZE,
    ) -> None:
        self.embedding_model = embedding_model
        self.batch_size = batch_size

    @staticmethod
    def _write_index(index: faiss.Index, path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        faiss.write_index(index, str(temporary_path))
        os.replace(temporary_path, path)

    def _load_checkpoint(self, products: List[Dict]) -> Tuple[faiss.Index, List[Dict]]:
        dimension = self.embedding_model.embedding_dimension
        if not (
            RESNET_FAISS_CHECKPOINT_PATH.exists()
            and RESNET_METADATA_CHECKPOINT_PATH.exists()
        ):
            return faiss.IndexFlatIP(dimension), []

        index = faiss.read_index(str(RESNET_FAISS_CHECKPOINT_PATH))
        metadata = json.loads(RESNET_METADATA_CHECKPOINT_PATH.read_text(encoding="utf-8"))
        checkpoint_ids = [str(item.get("id")) for item in metadata]
        source_ids = [str(item.get("id")) for item in products[: len(metadata)]]
        if index.d != dimension or index.ntotal != len(metadata) or checkpoint_ids != source_ids:
            print("ResNet checkpoint is stale; rebuilding from product 0.")
            return faiss.IndexFlatIP(dimension), []
        print(f"Resuming ResNet index from {index.ntotal} products.")
        return index, metadata

    def _checkpoint(self, index: faiss.Index, metadata: List[Dict]) -> None:
        if index.ntotal != len(metadata):
            raise ValueError("ResNet index and metadata are misaligned.")
        self._write_index(index, RESNET_FAISS_CHECKPOINT_PATH)
        atomic_json_dump(metadata, RESNET_METADATA_CHECKPOINT_PATH)

    def build(self, products: List[Dict]) -> Tuple[faiss.Index, List[Dict]]:
        index, metadata = self._load_checkpoint(products)
        starts = range(len(metadata), len(products), self.batch_size)
        batch_count = (len(products) - len(metadata) + self.batch_size - 1) // self.batch_size

        for number, start in enumerate(
            tqdm(starts, total=batch_count, desc="Indexing ResNet features"),
            start=1,
        ):
            batch_products = products[start : start + self.batch_size]
            images = []
            for product in batch_products:
                with Image.open(BASE_DIR / product["local_image_path"]) as image:
                    images.append(image.convert("RGB"))

            embeddings = self.embedding_model.encode_images(images)
            expected = (len(batch_products), self.embedding_model.embedding_dimension)
            if embeddings.shape != expected:
                raise ValueError(f"Expected embedding shape {expected}, got {embeddings.shape}.")
            if not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5):
                raise ValueError("ResNet produced non-normalized embeddings.")

            index.add(np.ascontiguousarray(embeddings, dtype="float32"))
            for product in batch_products:
                aligned = dict(product)
                aligned["index_position"] = len(metadata)
                metadata.append(aligned)

            if number % INDEX_CHECKPOINT_BATCHES == 0:
                self._checkpoint(index, metadata)
                print(f"ResNet checkpoint saved at {index.ntotal} vectors.")

        if index.ntotal != len(products) or index.ntotal != len(metadata):
            raise ValueError("Final ResNet vector and metadata counts do not align.")
        self._checkpoint(index, metadata)
        return index, metadata

    def finalize(self, index: faiss.Index, metadata: List[Dict]) -> None:
        self._write_index(index, RESNET_FAISS_INDEX_PATH)
        atomic_json_dump(metadata, RESNET_INDEXED_METADATA_PATH)
        saved = faiss.read_index(str(RESNET_FAISS_INDEX_PATH))
        if saved.ntotal != len(metadata) or saved.d != self.embedding_model.embedding_dimension:
            raise ValueError("Persisted ResNet index failed validation.")
        print(f"ResNet index saved with {saved.ntotal} vectors: {RESNET_FAISS_INDEX_PATH}")
