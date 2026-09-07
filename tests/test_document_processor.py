from collections.abc import Callable, Iterator
from typing import cast

import httpx
import pymupdf
import pytest

from bizcompass_kh.schemas.documents import SourceConfig
from bizcompass_kh.services.document_processor import (
    DocumentProcessingError,
    DocumentProcessor,
)

SOURCE_DATA: dict[str, object] = {
    "id": "business_registration_guide",
    "agency": "Ministry of Economy and Finance",
    "title": "How to Register Your Business Online",
    "url": "https://example.com/guide",
    "topic": "registration_guide",
    "category": "business_registration",
    "published_year": 2021,
    "official": True,
    "type": "html",
}


def make_source(**updates: object) -> SourceConfig:
    return SourceConfig.model_validate({**SOURCE_DATA, **updates})


def build_pdf(write_page: Callable[[pymupdf.Page], None]) -> bytes:
    pdf = pymupdf.open()
    try:
        page = pdf.new_page()
        write_page(page)
        return cast(bytes, pdf.tobytes())
    finally:
        pdf.close()


@pytest.fixture
def processor() -> Iterator[DocumentProcessor]:
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, request=request))
    ) as client:
        yield DocumentProcessor(http_client=client)


def test_process_document_fetches_and_processes_html() -> None:
    html = "<html><main><h1>Employer registration</h1><p>Required in 30 days.</p></main></html>"

    def respond(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://example.com/guide"
        assert request.headers["user-agent"].startswith("Mozilla/5.0")
        return httpx.Response(200, text=html, request=request)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        documents = DocumentProcessor(http_client=client).process_document(make_source())

    assert len(documents) == 1
    assert "Employer registration" in documents[0].page_content
    assert documents[0].metadata["extracted_method"] == "html"


def test_http_error_is_wrapped() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(404, request=request))
    with (
        httpx.Client(transport=transport) as client,
        pytest.raises(DocumentProcessingError, match="HTTP 404"),
    ):
        DocumentProcessor(http_client=client).process_document(make_source())


def test_process_pdf_preserves_non_empty_page_numbers(processor: DocumentProcessor) -> None:
    def write_pages(first_page: pymupdf.Page) -> None:
        first_page.insert_text((72, 72), "Employer registration requirements")
        pdf = cast(pymupdf.Document, first_page.parent)
        pdf.new_page()
        pdf.new_page().insert_text((72, 72), "Contribution rules")

    documents = processor.process_pdf(build_pdf(write_pages), make_source(type="pdf"))

    assert [document.page_content for document in documents] == [
        "Employer registration requirements",
        "Contribution rules",
    ]
    assert [document.metadata["page"] for document in documents] == [1, 3]


def test_process_pdf_rejects_broken_pdf(processor: DocumentProcessor) -> None:
    with pytest.raises(DocumentProcessingError, match="invalid PDF data"):
        processor.process_pdf(b"not a PDF", make_source(type="pdf"))


def test_process_html_rejects_empty_content(processor: DocumentProcessor) -> None:
    with pytest.raises(DocumentProcessingError, match="no extractable text"):
        processor.process_html("<html><body></body></html>", make_source())


def test_normalize_text_removes_blank_lines() -> None:
    assert DocumentProcessor.normalize_text(" A \n\n B ") == "A\nB"


def test_source_metadata_contains_provenance() -> None:
    metadata = DocumentProcessor.source_metadata(make_source())
    assert metadata["source_id"] == "business_registration_guide"
    assert metadata["agency"] == "Ministry of Economy and Finance"
    assert metadata["official"] is True


def test_source_config_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="type"):
        make_source(type="spreadsheet")
