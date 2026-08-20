from typing import List, Union

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from app.config import MODEL_NAME


class CLIPEmbeddingModel:
    """Create normalized CLIP embeddings for images and text."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Using device: {self.device}")
        print(f"Loading CLIP model: {model_name}")

        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.embedding_dimension = self.model.config.projection_dim

        print("CLIP model loaded successfully.")
        print(f"Embedding dimension: {self.embedding_dimension}")

    @staticmethod
    def normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
        """L2-normalize embeddings along their feature dimension."""
        return embeddings / embeddings.norm(
            p=2,
            dim=-1,
            keepdim=True,
        ).clamp(min=1e-12)

    @staticmethod
    def _feature_tensor(features: object) -> torch.Tensor:
        """Handle tensor and model-output return types across Transformers versions."""
        if isinstance(features, torch.Tensor):
            return features

        pooled_output = getattr(features, "pooler_output", None)
        if isinstance(pooled_output, torch.Tensor):
            return pooled_output

        if isinstance(features, tuple) and features:
            candidate = features[1] if len(features) > 1 else features[0]
            if isinstance(candidate, torch.Tensor):
                return candidate

        raise TypeError(f"Unexpected CLIP feature output: {type(features).__name__}")

    def encode_images(
        self,
        images: Union[Image.Image, List[Image.Image]],
    ) -> np.ndarray:
        """Convert one or more PIL images into normalized embeddings."""
        if isinstance(images, Image.Image):
            images = [images]
        if not images:
            raise ValueError("At least one image is required.")

        prepared_images = []
        for image in images:
            if not isinstance(image, Image.Image):
                raise TypeError("Every image must be a PIL Image.")
            if image.mode != "RGB":
                image = image.convert("RGB")
            prepared_images.append(image)

        inputs = self.processor(images=prepared_images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        with torch.inference_mode():
            features = self.model.get_image_features(pixel_values=pixel_values)

        embeddings = self._feature_tensor(features)
        embeddings = self.normalize_embeddings(embeddings)
        return embeddings.cpu().numpy().astype("float32")

    def encode_texts(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Convert one or more text queries into normalized embeddings."""
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            raise ValueError("At least one text query is required.")
        if not all(isinstance(text, str) for text in texts):
            raise TypeError("Every text query must be a string.")

        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.inference_mode():
            features = self.model.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        embeddings = self._feature_tensor(features)
        embeddings = self.normalize_embeddings(embeddings)
        return embeddings.cpu().numpy().astype("float32")
