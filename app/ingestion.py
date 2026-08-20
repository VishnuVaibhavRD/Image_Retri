import json
import os
from pathlib import Path
from typing import Dict, Iterable, List

from datasets import load_dataset
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from app.config import (
    BASE_DIR,
    CLEAN_METADATA_PATH,
    DATASET_NAME,
    DATASET_SPLIT,
    IMAGES_DIR,
    INGESTION_PROGRESS_PATH,
    METADATA_DIR,
    TARGET_DATASET_SIZE,
)


def atomic_json_dump(value: object, path: Path) -> None:
    """Replace a JSON file only after its complete replacement is written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False, default=str)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary_path, path)


class FashionDatasetIngestor:
    """Download, validate, deduplicate, and resume fashion dataset ingestion."""

    def __init__(self, target_size: int = TARGET_DATASET_SIZE) -> None:
        if target_size < 1:
            raise ValueError("target_size must be at least 1")
        self.target_size = target_size

    @staticmethod
    def _load_json_array(path: Path) -> List[Dict]:
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as file:
            value = json.load(file)
        if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
            raise ValueError(f"Expected a JSON array of objects: {path}")
        return value

    @staticmethod
    def _valid_saved_record(record: Dict) -> bool:
        product_id = record.get("id")
        relative_path = record.get("local_image_path")
        if product_id is None or not relative_path:
            return False
        path = BASE_DIR / str(relative_path)
        try:
            with Image.open(path) as image:
                image.verify()
            return True
        except (FileNotFoundError, OSError, UnidentifiedImageError):
            return False

    def load_progress(self) -> List[Dict]:
        """Recover valid records, tolerating an interrupted final JSONL line."""
        records: List[Dict] = []
        if INGESTION_PROGRESS_PATH.exists():
            lines = INGESTION_PROGRESS_PATH.read_text(encoding="utf-8").splitlines()
            for line_number, line in enumerate(lines, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    if line_number == len(lines):
                        print("Ignoring an incomplete final ingestion progress record.")
                        break
                    raise ValueError(
                        f"Invalid ingestion progress at line {line_number}."
                    )
                if self._valid_saved_record(record):
                    records.append(record)
        else:
            # Seed the scalable run from the already validated 100-product sample.
            sample_path = METADATA_DIR / "sample_metadata.json"
            records = [
                record
                for record in self._load_json_array(sample_path)
                if self._valid_saved_record(record)
            ]
            if records:
                self._rewrite_progress(records)

        unique_records = []
        seen_ids = set()
        for record in records:
            product_id = str(record["id"])
            if product_id not in seen_ids:
                seen_ids.add(product_id)
                unique_records.append(record)
        return unique_records[: self.target_size]

    @staticmethod
    def _rewrite_progress(records: List[Dict]) -> None:
        INGESTION_PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = INGESTION_PROGRESS_PATH.with_suffix(".jsonl.tmp")
        with temporary_path.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, INGESTION_PROGRESS_PATH)

    @staticmethod
    def _stream_dataset() -> Iterable[Dict]:
        return load_dataset(DATASET_NAME, split=DATASET_SPLIT, streaming=True)

    def ingest(self) -> List[Dict]:
        records = self.load_progress()
        seen_ids = {str(record["id"]) for record in records}

        print(f"Target products: {self.target_size}")
        print(f"Recovered valid products: {len(records)}")
        if len(records) >= self.target_size:
            atomic_json_dump(records[: self.target_size], CLEAN_METADATA_PATH)
            print("Ingestion is already complete.")
            return records[: self.target_size]

        skipped_duplicates = 0
        skipped_invalid = 0
        progress = tqdm(total=self.target_size, initial=len(records), desc="Ingesting products")
        with INGESTION_PROGRESS_PATH.open("a", encoding="utf-8", buffering=1) as progress_file:
            for source_record in self._stream_dataset():
                if len(records) >= self.target_size:
                    break

                product_id = source_record.get("id")
                if product_id is None or str(product_id) in seen_ids:
                    skipped_duplicates += 1
                    continue

                image = source_record.get("image")
                if image is None:
                    skipped_invalid += 1
                    continue

                output_path = IMAGES_DIR / f"{product_id}.jpg"
                temporary_path = IMAGES_DIR / f".{product_id}.jpg.tmp"
                try:
                    rgb_image = image.convert("RGB")
                    with temporary_path.open("wb") as file:
                        rgb_image.save(file, format="JPEG", quality=90)
                        file.flush()
                        os.fsync(file.fileno())
                    os.replace(temporary_path, output_path)
                    with Image.open(output_path) as saved_image:
                        saved_image.verify()
                except (OSError, ValueError, UnidentifiedImageError) as error:
                    skipped_invalid += 1
                    temporary_path.unlink(missing_ok=True)
                    print(f"Skipping product {product_id}: {error}")
                    continue

                clean_record = {
                    key: value for key, value in source_record.items() if key != "image"
                }
                clean_record["local_image_path"] = output_path.relative_to(BASE_DIR).as_posix()
                progress_file.write(
                    json.dumps(clean_record, ensure_ascii=False, default=str) + "\n"
                )
                progress_file.flush()

                records.append(clean_record)
                seen_ids.add(str(product_id))
                progress.update(1)

        progress.close()
        if len(records) < self.target_size:
            raise RuntimeError(
                f"Dataset ended after {len(records)} valid unique products; "
                f"target was {self.target_size}."
            )

        atomic_json_dump(records, CLEAN_METADATA_PATH)
        print(f"Saved {len(records)} clean unique products to: {CLEAN_METADATA_PATH}")
        print(f"Skipped duplicates: {skipped_duplicates}")
        print(f"Skipped invalid images: {skipped_invalid}")
        return records
