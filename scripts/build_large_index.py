import argparse

from app.config import CLEAN_METADATA_PATH, TARGET_DATASET_SIZE
from app.embedding_model import CLIPEmbeddingModel
from app.ingestion import FashionDatasetIngestor
from app.scalable_indexer import CheckpointedFashionIndexer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a resumable fashion image index.")
    parser.add_argument("--target", type=int, default=TARGET_DATASET_SIZE)
    parser.add_argument("--ingest-only", action="store_true")
    parser.add_argument("--index-only", action="store_true")
    args = parser.parse_args()
    if args.ingest_only and args.index_only:
        parser.error("--ingest-only and --index-only cannot be used together")
    return args


def main() -> None:
    args = parse_args()
    ingestor = FashionDatasetIngestor(target_size=args.target)

    if args.index_only:
        products = ingestor._load_json_array(CLEAN_METADATA_PATH)
        if len(products) != args.target:
            raise ValueError(
                f"Expected {args.target} clean products, found {len(products)}. "
                "Run ingestion first."
            )
    else:
        products = ingestor.ingest()

    if args.ingest_only:
        print("Ingestion completed; indexing was not requested.")
        return

    embedding_model = CLIPEmbeddingModel()
    indexer = CheckpointedFashionIndexer(embedding_model=embedding_model)
    index, indexed_products = indexer.build(products)
    indexer.finalize(index, indexed_products)
    print("Scalable ingestion and indexing completed successfully.")


if __name__ == "__main__":
    main()
