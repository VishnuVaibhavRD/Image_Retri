import json

from app.config import CLEAN_METADATA_PATH
from app.resnet_embedding_model import ResNetEmbeddingModel
from app.resnet_indexer import ResNetFashionIndexer


def main() -> None:
    products = json.loads(CLEAN_METADATA_PATH.read_text(encoding="utf-8"))
    model = ResNetEmbeddingModel()
    indexer = ResNetFashionIndexer(model)
    index, metadata = indexer.build(products)
    indexer.finalize(index, metadata)


if __name__ == "__main__":
    main()
