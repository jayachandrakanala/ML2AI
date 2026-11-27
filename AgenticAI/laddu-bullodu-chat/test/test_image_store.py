import sys
from types import SimpleNamespace

from PIL import Image

from src import image_store


def _png_bytes():
    """Create a small valid image fixture without adding a binary test file."""
    from io import BytesIO

    output = BytesIO()
    Image.new("RGB", (4, 4), color="blue").save(output, format="PNG")
    return output.getvalue()


def test_extracts_pdf_images_with_searchable_page_context(tmp_path, monkeypatch):
    """Persist a PDF image and index its source page text."""
    documents = tmp_path / "documents"
    documents.mkdir()
    pdf_path = documents / "visualization.pdf"
    pdf_path.write_bytes(b"test pdf")
    image = SimpleNamespace(name="Image1.png", data=_png_bytes())
    page = SimpleNamespace(
        images=[image],
        extract_text=lambda: "Embedding space visualization with HR documents",
    )
    reader = SimpleNamespace(pages=[page])
    monkeypatch.setitem(sys.modules, "pypdf", SimpleNamespace(PdfReader=lambda _: reader))

    assets = image_store.index_document_images(
        documents, tmp_path / "stored_images"
    )

    assert len(assets) == 1
    asset = assets[0]
    assert "HR documents" in asset.page_content
    assert asset.metadata["source"] == "visualization.pdf"
    assert asset.metadata["image_page"] == 1
    stored_path = tmp_path / "stored_images" / (
        f"{asset.metadata['image_asset_id']}.{asset.metadata['image_extension']}"
    )
    assert stored_path.is_file()
    assert stored_path.read_bytes() == _png_bytes()


def test_indexes_standalone_images_with_sidecar_description(tmp_path, monkeypatch):
    """Use a sidecar description as searchable context for a local image."""
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "embedding-plot.png").write_bytes(_png_bytes())
    (documents / "embedding-plot.md").write_text(
        "Scatter plot showing document embeddings by HR and product category.",
        encoding="utf-8",
    )
    store = tmp_path / "stored_images"
    monkeypatch.setattr(image_store, "IMAGE_STORE_DIRECTORY", str(store))

    assets = image_store.index_document_images(documents, store)
    references = image_store.image_assets_for_documents(assets)

    assert len(assets) == 1
    assert "HR and product category" in assets[0].page_content
    assert len(references) == 1
    assert image_store.resolve_image_asset(
        references[0]["asset_id"], references[0]["extension"]
    ).is_file()


def test_rejects_unsafe_or_unknown_image_asset_ids(tmp_path, monkeypatch):
    """Do not resolve traversal strings or unrecognized file formats."""
    monkeypatch.setattr(image_store, "IMAGE_STORE_DIRECTORY", str(tmp_path))

    assert image_store.resolve_image_asset("../outside", "png") is None
    assert image_store.resolve_image_asset("a" * 64, "svg") is None


def test_skips_invalid_standalone_image_files(tmp_path):
    """Do not register corrupt or misnamed files as retrievable images."""
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "not-really-an-image.png").write_bytes(b"not an image")

    assert image_store.index_document_images(documents, tmp_path / "stored") == []


def test_extracts_ocr_text_from_rapidocr_results(monkeypatch):
    """Accept RapidOCR's string confidence values and return recognized labels."""
    monkeypatch.setattr(
        image_store,
        "_load_ocr_engine",
        lambda: lambda image: (
            [
                ([], "Embedding Space", "0.91"),
                ([], "uncertain", "0.05"),
            ],
            [],
        ),
    )

    assert "Embedding Space" in image_store._extract_image_text(_png_bytes())
    assert "uncertain" not in image_store._extract_image_text(_png_bytes())
