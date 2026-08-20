import sys
from pathlib import Path
from typing import Dict, List

import streamlit as st
from PIL import Image

# Make project imports work with both supported launch commands.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import (  # noqa: E402
    BASE_DIR,
    MODEL_NAME,
    RESNET_MODEL_NAME,
    RESNET_WEIGHTS_NAME,
)
from app.embedding_model import CLIPEmbeddingModel  # noqa: E402
from app.hybrid_search_engine import HybridImageSearchEngine  # noqa: E402
from app.resnet_embedding_model import ResNetEmbeddingModel  # noqa: E402
from app.resnet_search_engine import ResNetImageSearchEngine  # noqa: E402
from app.search_engine import ImageSearchEngine  # noqa: E402


st.set_page_config(
    page_title="Fashion Image Search",
    page_icon="🔎",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def load_search_engine() -> ImageSearchEngine:
    """Load CLIP and FAISS once for the lifetime of the Streamlit process."""
    return ImageSearchEngine(embedding_model=CLIPEmbeddingModel())


@st.cache_resource(show_spinner=False)
def load_resnet_search_engine() -> ResNetImageSearchEngine:
    """Load the ResNet visual-search baseline once per Streamlit process."""
    return ResNetImageSearchEngine(embedding_model=ResNetEmbeddingModel())


@st.cache_resource(show_spinner=False)
def load_hybrid_search_engine(
    _clip_engine: ImageSearchEngine,
) -> HybridImageSearchEngine:
    """Combine the cached CLIP and ResNet engines for visual retrieval."""
    return HybridImageSearchEngine(
        clip_engine=_clip_engine,
        resnet_engine=load_resnet_search_engine(),
    )


def resolve_image_path(product: Dict) -> Path:
    image_path = Path(product["local_image_path"])
    return image_path if image_path.is_absolute() else BASE_DIR / image_path


def display_results(results: List[Dict], heading: str) -> None:
    """Render ranked products in a responsive row of cards."""
    if not results:
        st.info("No matching products were found.")
        return

    st.subheader(heading)
    columns = st.columns(min(len(results), 5))
    for rank, (column, product) in enumerate(zip(columns, results), start=1):
        with column:
            image_path = resolve_image_path(product)
            if image_path.exists():
                st.image(str(image_path), use_container_width=True)
            else:
                st.warning("Image unavailable")

            st.markdown(f"**{rank}. {product.get('productDisplayName', 'Unknown product')}**")
            st.metric("Similarity", f"{product['similarity_score']:.4f}")
            st.caption(
                " · ".join(
                    str(value)
                    for value in (
                        product.get("gender"),
                        product.get("baseColour"),
                        product.get("articleType"),
                    )
                    if value
                )
            )
            with st.expander("Technical details"):
                st.write("Product ID:", product.get("id", "Unknown"))
                st.write("Master category:", product.get("masterCategory", "Unknown"))
                st.write("Subcategory:", product.get("subCategory", "Unknown"))
                st.write("Season:", product.get("season", "Unknown"))
                st.write("Year:", product.get("year", "Unknown"))
                st.write("Usage:", product.get("usage", "Unknown"))
                if product.get("clip_rank") is not None:
                    st.write("CLIP candidate rank:", product["clip_rank"])
                    st.write("CLIP similarity:", f"{product['clip_similarity']:.4f}")
                if product.get("resnet_rank") is not None:
                    st.write("ResNet candidate rank:", product["resnet_rank"])
                    st.write("ResNet similarity:", f"{product['resnet_similarity']:.4f}")
                st.code(product.get("local_image_path", ""))


def text_search_tab(search_engine: ImageSearchEngine, top_k: int) -> None:
    st.write("Describe the fashion product you want to find.")
    with st.form("text_search_form"):
        query = st.text_input(
            "Search query",
            value=st.session_state.get("last_text_query", "black men's shoes"),
            placeholder="For example: blue casual shirt",
        )
        submitted = st.form_submit_button("Search products", use_container_width=True)

    if submitted:
        if not query.strip():
            st.warning("Enter a text query before searching.")
        else:
            with st.spinner("Searching product images..."):
                st.session_state.text_results = search_engine.search_by_text(
                    query=query,
                    top_k=top_k,
                )
                st.session_state.last_text_query = query

    results = st.session_state.get("text_results")
    if results is not None:
        display_results(results, "Text search results")


def image_search_tab(clip_search_engine: ImageSearchEngine, top_k: int) -> None:
    st.write("Upload a product image to retrieve visually similar products.")
    retrieval_model = st.radio(
        "Visual retrieval model",
        options=["Hybrid", "CLIP", "ResNet-50"],
        horizontal=True,
        help=(
            "Hybrid combines semantic CLIP and visual ResNet rankings. "
            "ResNet-50 alone does not support text queries."
        ),
    )
    uploaded_file = st.file_uploader(
        "Product image",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=False,
    )

    if uploaded_file is not None:
        try:
            query_image = Image.open(uploaded_file).convert("RGB")
            st.image(query_image, caption="Query image", width=240)
        except (OSError, ValueError):
            st.error("The uploaded file could not be read as an image.")
            return

        if st.button("Find similar images", type="primary", use_container_width=True):
            with st.spinner("Searching for visually similar products..."):
                if retrieval_model == "Hybrid":
                    selected_engine = load_hybrid_search_engine(clip_search_engine)
                elif retrieval_model == "ResNet-50":
                    selected_engine = load_resnet_search_engine()
                else:
                    selected_engine = clip_search_engine
                st.session_state.image_results = selected_engine.search_by_image(
                    image=query_image,
                    top_k=top_k,
                )
                st.session_state.image_results_model = retrieval_model

    results = st.session_state.get("image_results")
    if results is not None:
        result_model = st.session_state.get("image_results_model", "CLIP")
        st.caption(f"Results generated with {result_model}")
        display_results(results, f"Image search results — {result_model}")


def main() -> None:
    st.title("Fashion Image Search")
    st.write("Search the catalog using natural language or a product image.")

    with st.sidebar:
        st.header("Search settings")
        top_k = st.slider("Number of results", min_value=1, max_value=10, value=5)

        st.divider()
        st.subheader("Current engine")
        st.write("Text and semantic image model")
        st.code(MODEL_NAME)
        st.write("Visual image model")
        st.code(f"{RESNET_MODEL_NAME} / {RESNET_WEIGHTS_NAME}")
        st.write("Vector indexes")
        st.code("CLIP: 5,000 × 512\nResNet: 5,000 × 2,048\nFAISS IndexFlatIP")
        st.write("Retrieval modes")
        st.write("✓ Text-to-image")
        st.write("✓ Hybrid image-to-image (recommended)")
        st.write("✓ CLIP image-to-image")
        st.write("✓ ResNet image-to-image")

        st.divider()
        if st.button("Clear previous results", use_container_width=True):
            for key in [
                "text_results",
                "last_text_query",
                "image_results",
                "image_results_model",
            ]:
                st.session_state.pop(key, None)
            st.rerun()

    try:
        with st.spinner("Loading CLIP model and FAISS index..."):
            search_engine = load_search_engine()
    except Exception as error:
        st.error("The search engine could not be loaded.")
        st.exception(error)
        st.info(
            "Confirm that you have already run:\n\n"
            "`python -m scripts.build_index`"
        )
        st.stop()

    st.success(f"Engine ready — {search_engine.index.ntotal} products indexed")

    text_tab, image_tab = st.tabs(["Text search", "Image search"])
    with text_tab:
        text_search_tab(search_engine=search_engine, top_k=top_k)
    with image_tab:
        image_search_tab(clip_search_engine=search_engine, top_k=top_k)


if __name__ == "__main__":
    main()
