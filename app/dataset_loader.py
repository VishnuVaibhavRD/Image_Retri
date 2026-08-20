from datasets import load_dataset

from app.config import DATASET_NAME, DATASET_SPLIT, INITIAL_DATASET_SIZE


def load_fashion_dataset(limit: int = INITIAL_DATASET_SIZE):
    """Load a limited number of records from the Hugging Face dataset.

    Streaming mode avoids downloading the entire dataset before inspection.
    """
    if limit < 1:
        raise ValueError("limit must be at least 1")

    print(f"Loading dataset: {DATASET_NAME}")
    print(f"Split: {DATASET_SPLIT}")
    print(f"Requested records: {limit}")

    streaming_dataset = load_dataset(
        DATASET_NAME,
        split=DATASET_SPLIT,
        streaming=True,
    )
    records = list(streaming_dataset.take(limit))

    print(f"Successfully loaded {len(records)} records.")
    return records
