from app.config import METADATA_DIR
from app.embedding_model import CLIPEmbeddingModel
from app.indexer import FashionImageIndexer


def main() -> None:
    metadata_path = METADATA_DIR / "sample_metadata.json"

    print("Starting image indexing process...")
    print(f"Source metadata: {metadata_path}")

    embedding_model = CLIPEmbeddingModel()
    indexer = FashionImageIndexer(embedding_model=embedding_model)
    indexer.run(metadata_path)

    print("\nIndexing completed successfully.")


if __name__ == "__main__":
    main()
