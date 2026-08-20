from typing import List, Union

import numpy as np
import torch
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50

from app.config import RESNET_EMBEDDING_DIMENSION, RESNET_MODEL_NAME, RESNET_WEIGHTS_NAME


class ResNetEmbeddingModel:
    """Generate normalized visual embeddings from pretrained ResNet-50 features."""

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights = ResNet50_Weights.IMAGENET1K_V2
        self.preprocess = self.weights.transforms()

        print(f"Using device: {self.device}")
        print(f"Loading ResNet model: {RESNET_MODEL_NAME} ({RESNET_WEIGHTS_NAME})")

        classification_model = resnet50(weights=self.weights)
        self.model = torch.nn.Sequential(*list(classification_model.children())[:-1])
        self.model = self.model.to(self.device).eval()
        self.embedding_dimension = RESNET_EMBEDDING_DIMENSION

        print("ResNet model loaded successfully.")
        print(f"Embedding dimension: {self.embedding_dimension}")

    def encode_images(
        self,
        images: Union[Image.Image, List[Image.Image]],
    ) -> np.ndarray:
        if isinstance(images, Image.Image):
            images = [images]
        if not images:
            raise ValueError("At least one image is required.")
        if not all(isinstance(image, Image.Image) for image in images):
            raise TypeError("Every image must be a PIL Image.")

        batch = torch.stack(
            [self.preprocess(image.convert("RGB")) for image in images]
        ).to(self.device)
        with torch.inference_mode():
            embeddings = self.model(batch).flatten(start_dim=1)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        if embeddings.shape[1] != self.embedding_dimension:
            raise ValueError(
                f"Expected {self.embedding_dimension} features, got {embeddings.shape[1]}."
            )
        return embeddings.cpu().numpy().astype("float32")
