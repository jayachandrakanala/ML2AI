"""Extract, persist, and resolve document images for local RAG retrieval."""

import hashlib
from functools import lru_cache
from io import BytesIO
import logging
import re
from pathlib import Path

from langchain_core.documents import Document

from src.guardrails import sanitize_text

logger = logging.getLogger(__name__)

IMAGE_STORE_DIRECTORY = "./document_images"
OCR_MIN_CONFIDENCE = 0.2
SUPPORTED_IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
VISUAL_QUERY_TERMS = (
    "image",
    "picture",
    "photo",
    "diagram",
    "figure",
    "chart",
    "visualization",
)


def is_visual_query(query: str) -> bool:
    """Return whether a query explicitly asks for a local visual asset."""
    normalized_query = query.lower()
    return any(term in normalized_query for term in VISUAL_QUERY_TERMS)


@lru_cache(maxsize=1)
def _load_ocr_engine():
    """Load one CPU OCR engine for document-image indexing."""
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR()


def _extract_image_text(image_bytes: bytes) -> str:
    """Recognize searchable labels embedded inside a raster document image."""
    import numpy as np
    from PIL import Image

    with Image.open(BytesIO(image_bytes)) as image:
        image_array = np.asarray(image.convert("RGB"))
    results, _ = _load_ocr_engine()(image_array)
    if not results:
        return ""
    recognized_lines = []
    for result in results:
        if len(result) < 2 or not isinstance(result[1], str):
            continue
        try:
            confidence = float(result[2]) if len(result) > 2 else 1.0
        except (TypeError, ValueError):
            continue
        if confidence >= OCR_MIN_CONFIDENCE:
            recognized_lines.append(result[1])
    return "\n".join(recognized_lines)


def _safe_extension(extension: str) -> str:
    """Return a supported lowercase extension, or an empty string."""
    normalized = extension.lower().lstrip(".")
    if normalized == "jpeg":
        normalized = "jpg"
    return normalized if f".{normalized}" in SUPPORTED_IMAGE_EXTENSIONS else ""


def _persist_image(
    image_bytes: bytes,
    extension: str,
    source: str,
    location: str,
    store_directory: Path,
) -> str:
    """Persist an extracted image under a stable ID derived from its origin and bytes."""
    content_hash = hashlib.sha256(image_bytes).hexdigest()
    asset_id = hashlib.sha256(
        f"{source}\0{location}\0{content_hash}".encode("utf-8")
    ).hexdigest()
    target = store_directory / f"{asset_id}.{extension}"
    if target.is_symlink():
        raise RuntimeError(f"Refusing to overwrite a symlinked image asset: {target}")
    if not target.exists():
        target.write_bytes(image_bytes)
    return asset_id


def _image_document(
    *,
    image_bytes: bytes,
    extension: str,
    source: str,
    location: str,
    description: str,
    store_directory: Path,
) -> Document:
    """Save an image and create searchable text and metadata for Chroma."""
    from PIL import Image

    with Image.open(BytesIO(image_bytes)) as decoded_image:
        decoded_image.verify()
    recognized_text = _extract_image_text(image_bytes)
    asset_id = _persist_image(
        image_bytes, extension, source, location, store_directory
    )
    page_match = re.match(r"page-(\d+)", location)
    page_number = int(page_match.group(1)) if page_match else None
    safe_source = sanitize_text(source)
    combined_description = "\n".join(
        part
        for part in (description, recognized_text)
        if part
    )
    safe_description = sanitize_text(combined_description[:6000])
    caption = f"{safe_source}, page {page_number}" if page_number else safe_source
    searchable_text = (
        f"Document image: {caption}\n"
        f"Description and nearby document text:\n{safe_description}"
    )
    metadata = {
        "kind": "image",
        "source": safe_source,
        "image_asset_id": asset_id,
        "image_extension": extension,
        "image_caption": caption,
    }
    if page_number:
        metadata["image_page"] = page_number
    return Document(page_content=searchable_text, metadata=metadata)


def _extract_pdf_images(
    pdf_path: Path, documents_root: Path, store_directory: Path
) -> list[Document]:
    """Extract embedded raster images and pair them with their PDF page text."""
    from pypdf import PdfReader

    relative_source = pdf_path.relative_to(documents_root).as_posix()
    image_documents = []
    try:
        with pdf_path.open("rb") as pdf_file:
            pdf = PdfReader(pdf_file)
            for page_index, page in enumerate(pdf.pages):
                page_number = page_index + 1
                page_text = sanitize_text(page.extract_text() or "").strip()
                for image_index, image in enumerate(page.images):
                    try:
                        image_bytes = image.data
                        extension = _safe_extension(Path(image.name).suffix)
                        if not extension:
                            try:
                                from PIL import Image

                                with Image.open(BytesIO(image_bytes)) as decoded_image:
                                    output = BytesIO()
                                    image_mode = (
                                        "RGBA"
                                        if "A" in decoded_image.getbands()
                                        else "RGB"
                                    )
                                    decoded_image.convert(image_mode).save(
                                        output, format="PNG"
                                    )
                                    image_bytes = output.getvalue()
                                    extension = "png"
                            except Exception:
                                logger.exception(
                                    "Could not decode PDF image %r in %s page %s",
                                    image.name,
                                    relative_source,
                                    page_number,
                                )
                                continue
                        image_documents.append(
                            _image_document(
                                image_bytes=image_bytes,
                                extension=extension,
                                source=relative_source,
                                location=f"page-{page_number}-image-{image_index}",
                                description=page_text or pdf_path.stem.replace("_", " "),
                                store_directory=store_directory,
                            )
                        )
                    except Exception:
                        logger.exception(
                            "Could not extract image %s from %s page %s",
                            image_index + 1,
                            relative_source,
                            page_number,
                        )
    except Exception:
        logger.exception("Could not read PDF document images from %s", relative_source)
    return image_documents


def _standalone_image_document(
    image_path: Path, documents_root: Path, store_directory: Path
) -> Document | None:
    """Index a local raster image using its filename and optional text sidecar."""
    relative_source = image_path.relative_to(documents_root).as_posix()
    extension = _safe_extension(image_path.suffix)
    if not extension:
        return None

    sidecar_text = ""
    for sidecar_suffix in (".md", ".txt"):
        sidecar = image_path.with_suffix(sidecar_suffix)
        if sidecar.is_file():
            try:
                sidecar_text = sanitize_text(sidecar.read_text(encoding="utf-8"))
            except OSError:
                logger.exception("Could not read image description %s", sidecar)
            break

    try:
        image_bytes = image_path.read_bytes()
        return _image_document(
            image_bytes=image_bytes,
            extension=extension,
            source=relative_source,
            location="file",
            description=(
                f"{image_path.stem.replace('_', ' ')}\n{sidecar_text}".strip()
            ),
            store_directory=store_directory,
        )
    except Exception:
        logger.exception("Could not index document image %s", relative_source)
        return None


def index_document_images(
    documents_directory: str | Path,
    image_store_directory: str | Path = IMAGE_STORE_DIRECTORY,
) -> list[Document]:
    """Extract local image files and PDF images into a persistent image store."""
    documents_root = Path(documents_directory).resolve()
    if not documents_root.is_dir():
        return []

    store_directory = Path(image_store_directory).resolve()
    store_directory.mkdir(parents=True, exist_ok=True)
    image_documents = []

    paths = sorted(documents_root.rglob("*"))
    has_images = any(
        path.suffix.lower() == ".pdf"
        or path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        for path in paths
        if path.is_file() and not path.is_symlink()
    )
    if has_images:
        try:
            _load_ocr_engine()
        except ImportError as error:
            raise RuntimeError(
                "Image text indexing requires rapidocr_onnxruntime. Install "
                "project requirements and restart the assistant."
            ) from error

    for path in paths:
        if path.is_symlink() or not path.is_file():
            continue
        resolved_path = path.resolve()
        if not resolved_path.is_relative_to(documents_root):
            continue
        if path.suffix.lower() == ".pdf":
            image_documents.extend(
                _extract_pdf_images(path, documents_root, store_directory)
            )
        elif path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            image_document = _standalone_image_document(
                path, documents_root, store_directory
            )
            if image_document is not None:
                image_documents.append(image_document)
    return image_documents


def image_assets_for_documents(documents: list[Document]) -> list[dict[str, str | int]]:
    """Build unique UI-safe image references from retrieved Chroma documents."""
    image_assets = []
    seen_asset_ids = set()
    for document in documents:
        metadata = document.metadata
        asset_id = metadata.get("image_asset_id")
        extension = metadata.get("image_extension")
        if (
            metadata.get("kind") != "image"
            or not isinstance(asset_id, str)
            or not isinstance(extension, str)
            or asset_id in seen_asset_ids
        ):
            continue
        seen_asset_ids.add(asset_id)
        image_asset: dict[str, str | int] = {
            "asset_id": asset_id,
            "extension": extension,
            "caption": str(
                metadata.get("image_caption", metadata.get("source", "Document image"))
            ),
        }
        if isinstance(metadata.get("image_page"), int):
            image_asset["page"] = metadata["image_page"]
        image_assets.append(image_asset)
    return image_assets


def resolve_image_asset(asset_id: str, extension: str) -> Path | None:
    """Resolve a stored image only when its ID and extension are safe."""
    if not re.fullmatch(r"[a-f0-9]{64}", asset_id):
        return None
    normalized_extension = _safe_extension(extension)
    if not normalized_extension:
        return None
    store_directory = Path(IMAGE_STORE_DIRECTORY).resolve()
    image_path = (store_directory / f"{asset_id}.{normalized_extension}").resolve()
    if image_path.parent != store_directory or not image_path.is_file():
        return None
    return image_path
