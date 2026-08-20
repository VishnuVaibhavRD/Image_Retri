import json

from PIL import Image, ImageDraw

from app.config import BASE_DIR, IMAGES_DIR, METADATA_DIR
from app.dataset_loader import load_fashion_dataset


def save_sample_images(records: list[dict]) -> None:
    """Save product images and their JSON-serializable metadata locally."""
    metadata = []

    for index, record in enumerate(records):
        image = record.get("image")
        if image is None:
            continue

        product_id = str(record.get("id", index))
        output_path = IMAGES_DIR / f"{product_id}.jpg"
        image.convert("RGB").save(output_path, format="JPEG", quality=90)

        metadata_record = {
            key: value
            for key, value in record.items()
            if key != "image"
        }
        metadata_record["local_image_path"] = output_path.relative_to(BASE_DIR).as_posix()
        metadata.append(metadata_record)

    metadata_path = METADATA_DIR / "sample_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print(f"Saved {len(metadata)} sample images.")
    print(f"Metadata saved to: {metadata_path}")


def create_contact_sheet(records: list[dict]) -> None:
    """Create a labeled preview of up to the first 20 available images."""
    preview_records = [record for record in records if record.get("image") is not None][:20]
    if not preview_records:
        print("No images were available for a dataset preview.")
        return

    columns = 5
    cell_size = (180, 220)
    rows = (len(preview_records) + columns - 1) // columns
    contact_sheet = Image.new(
        "RGB",
        (columns * cell_size[0], rows * cell_size[1]),
        "white",
    )
    draw = ImageDraw.Draw(contact_sheet)

    for index, record in enumerate(preview_records):
        image = record["image"].convert("RGB")
        image.thumbnail((160, 170))

        cell_x = (index % columns) * cell_size[0]
        cell_y = (index // columns) * cell_size[1]
        image_x = cell_x + (cell_size[0] - image.width) // 2
        image_y = cell_y + 5
        contact_sheet.paste(image, (image_x, image_y))

        product_name = str(record.get("productDisplayName", f"Product {index}"))
        draw.text((cell_x + 8, cell_y + 182), product_name[:24], fill="black")

    output_path = IMAGES_DIR / "dataset_preview.jpg"
    contact_sheet.save(output_path, format="JPEG", quality=90)
    print(f"Dataset preview saved to: {output_path}")


def inspect_dataset() -> None:
    records = load_fashion_dataset(limit=100)

    if not records:
        print("The dataset returned no records.")
        return

    print("\nAvailable dataset columns:")
    for column in records[0].keys():
        print(f"- {column}")

    print("\nFirst product metadata:")
    for key, value in records[0].items():
        if key == "image":
            if value is not None:
                print(f"- image: PIL image, size={value.size}, mode={value.mode}")
            else:
                print("- image: None")
        else:
            print(f"- {key}: {value}")

    save_sample_images(records)
    create_contact_sheet(records)
    print("\nDataset inspection completed successfully.")


if __name__ == "__main__":
    inspect_dataset()
