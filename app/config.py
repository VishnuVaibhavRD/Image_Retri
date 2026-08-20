from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Local data directories
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
INDEXES_DIR = DATA_DIR / "indexes"
METADATA_DIR = DATA_DIR / "metadata"

# Hugging Face dataset
DATASET_NAME = "ashraq/fashion-product-images-small"
DATASET_SPLIT = "train"

# Start with a small sample.
INITIAL_DATASET_SIZE = 100
TARGET_DATASET_SIZE = 5_000

# Embedding model
MODEL_NAME = "openai/clip-vit-base-patch32"

# Use this batch size when indexing images later.
EMBEDDING_BATCH_SIZE = 16

# Persisted vector index and its position-aligned metadata.
FAISS_INDEX_PATH = INDEXES_DIR / "fashion_products.faiss"
INDEXED_METADATA_PATH = METADATA_DIR / "indexed_products.json"

# Resumable large-dataset ingestion and indexing artifacts.
INGESTION_PROGRESS_PATH = METADATA_DIR / "ingestion_progress.jsonl"
CLEAN_METADATA_PATH = METADATA_DIR / "fashion_products_5000.json"
FAISS_CHECKPOINT_PATH = INDEXES_DIR / "fashion_products.checkpoint.faiss"
METADATA_CHECKPOINT_PATH = METADATA_DIR / "indexed_products.checkpoint.json"
BACKUP_FAISS_INDEX_PATH = INDEXES_DIR / "fashion_products_100_backup.faiss"
BACKUP_INDEXED_METADATA_PATH = METADATA_DIR / "indexed_products_100_backup.json"
INDEX_CHECKPOINT_BATCHES = 10

# ResNet visual-retrieval baseline. CLIP remains the text-search model.
RESNET_MODEL_NAME = "resnet50"
RESNET_WEIGHTS_NAME = "IMAGENET1K_V2"
RESNET_EMBEDDING_DIMENSION = 2_048
RESNET_BATCH_SIZE = 32
RESNET_FAISS_INDEX_PATH = INDEXES_DIR / "fashion_products_resnet50.faiss"
RESNET_INDEXED_METADATA_PATH = METADATA_DIR / "indexed_products_resnet50.json"
RESNET_FAISS_CHECKPOINT_PATH = INDEXES_DIR / "fashion_products_resnet50.checkpoint.faiss"
RESNET_METADATA_CHECKPOINT_PATH = METADATA_DIR / "indexed_products_resnet50.checkpoint.json"

# Ensure directories exist.
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
INDEXES_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)
