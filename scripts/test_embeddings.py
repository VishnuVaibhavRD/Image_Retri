import json

import numpy as np
from PIL import Image

from app.config import BASE_DIR, METADATA_DIR
from app.embedding_model import CLIPEmbeddingModel


def calculate_similarity(
    image_embedding: np.ndarray,
    text_embedding: np.ndarray,
) -> float:
    """Calculate cosine similarity between normalized single-item embeddings."""
    return float(np.dot(image_embedding[0], text_embedding[0]))


def test_embeddings() -> None:
    metadata_path = METADATA_DIR / "sample_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            "Sample metadata is missing. Run python -m scripts.inspect_dataset first."
        )

    records = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not records:
        raise ValueError("Sample metadata contains no products.")

    product = records[0]
    image_path = BASE_DIR / product["local_image_path"]
    if not image_path.exists():
        raise FileNotFoundError(f"Product image is missing: {image_path}")

    product_title = str(product.get("productDisplayName", "fashion product"))
    matching_query = "a navy blue men's shirt"
    unrelated_query = "a red women's handbag"

    embedding_model = CLIPEmbeddingModel()

    with Image.open(image_path) as image:
        image_embedding = embedding_model.encode_images(image)

    text_embeddings = embedding_model.encode_texts(
        [product_title, matching_query, unrelated_query]
    )

    title_embedding = text_embeddings[0:1]
    matching_embedding = text_embeddings[1:2]
    unrelated_embedding = text_embeddings[2:3]

    title_similarity = calculate_similarity(image_embedding, title_embedding)
    matching_similarity = calculate_similarity(image_embedding, matching_embedding)
    unrelated_similarity = calculate_similarity(image_embedding, unrelated_embedding)

    print("\nEmbedding results")
    print("-----------------")
    print(f"Image embedding shape: {image_embedding.shape}")
    print(f"Text embeddings shape: {text_embeddings.shape}")
    print("Image vector length:", np.linalg.norm(image_embedding[0]))
    print("Text vector length:", np.linalg.norm(text_embeddings[0]))

    print("\nSimilarity results")
    print("------------------")
    print(f"Product title similarity: {title_similarity:.4f}")
    print(f"Matching query similarity: {matching_similarity:.4f}")
    print(f"Unrelated query similarity: {unrelated_similarity:.4f}")

    if matching_similarity > unrelated_similarity:
        print(
            "\nSuccess: CLIP understood the relevant "
            "query better than the unrelated query."
        )
    else:
        print(
            "\nWarning: The unrelated query scored higher. "
            "This can occasionally happen for an individual image."
        )


if __name__ == "__main__":
    test_embeddings()
