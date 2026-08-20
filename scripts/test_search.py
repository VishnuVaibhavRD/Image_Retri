from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageDraw

from app.config import BASE_DIR, IMAGES_DIR
from app.embedding_model import CLIPEmbeddingModel
from app.search_engine import ImageSearchEngine


def print_results(title: str, results: List[Dict]) -> None:
    print(f"\n{title}")
    print("=" * len(title))
    for rank, product in enumerate(results, start=1):
        print(f"Rank: {rank}")
        print(f"Product: {product.get('productDisplayName', 'Unknown')}")
        print(f"Category: {product.get('articleType', 'Unknown')}")
        print(f"Colour: {product.get('baseColour', 'Unknown')}")
        print(f"Gender: {product.get('gender', 'Unknown')}")
        print(f"Similarity: {product['similarity_score']:.4f}")
        print(f"Image: {product.get('local_image_path', 'Unknown')}\n")


def create_results_sheet(results: List[Dict], output_path: Path, title: str) -> None:
    """Save a labeled horizontal preview of ranked search results."""
    if not results:
        print(f"No results available for preview: {output_path}")
        return

    cell_width, cell_height = 200, 250
    title_height = 40
    sheet = Image.new(
        "RGB",
        (cell_width * len(results), title_height + cell_height),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 12), title, fill="black")

    for index, product in enumerate(results):
        image_path = BASE_DIR / product["local_image_path"]
        with Image.open(image_path) as opened_image:
            image = opened_image.convert("RGB")
            image.thumbnail((180, 180))

        cell_x = index * cell_width
        image_x = cell_x + (cell_width - image.width) // 2
        image_y = title_height + 5
        sheet.paste(image, (image_x, image_y))

        name = str(product.get("productDisplayName", "Unknown"))[:25]
        score = float(product["similarity_score"])
        draw.text((cell_x + 8, title_height + 190), f"{index + 1}. {name}", fill="black")
        draw.text((cell_x + 8, title_height + 210), f"Score: {score:.4f}", fill="black")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="JPEG", quality=90)
    print(f"Results preview saved to: {output_path}")


def main() -> None:
    query = "black men's shoes"
    top_k = 5

    print(f"Search query: {query}")
    print(f"Requested results: {top_k}")

    embedding_model = CLIPEmbeddingModel()
    search_engine = ImageSearchEngine(embedding_model=embedding_model)

    text_results = search_engine.search_by_text(query=query, top_k=top_k)
    print_results(title="Text-to-image search results", results=text_results)

    text_preview_path = IMAGES_DIR / "text_search_results.jpg"
    create_results_sheet(
        results=text_results,
        output_path=text_preview_path,
        title=f'Text search: "{query}"',
    )

    if text_results:
        first_result = text_results[0]
        query_image_path = first_result["local_image_path"]
        query_product_id = first_result.get("id")

        image_results = search_engine.search_by_image(
            image=query_image_path,
            top_k=top_k,
            exclude_product_id=str(query_product_id),
        )
        print_results(title="Image-to-image search results", results=image_results)

        image_preview_path = IMAGES_DIR / "image_search_results.jpg"
        create_results_sheet(
            results=image_results,
            output_path=image_preview_path,
            title="Visually similar products",
        )


if __name__ == "__main__":
    main()
