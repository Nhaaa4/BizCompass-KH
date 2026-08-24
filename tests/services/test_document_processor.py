from __future__ import annotations

import asyncio
import re
from collections.abc import Callable, Iterator
from typing import cast

import httpx
import pymupdf
import pytest
from langchain_core.documents import Document

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
        return pdf.tobytes()  # pyright: ignore[reportUnknownMemberType]
    finally:
        pdf.close()


@pytest.fixture
def processor() -> Iterator[DocumentProcessor]:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, request=request)
        )
    )
    yield DocumentProcessor(http_client=client)
    asyncio.run(client.aclose())


def test_process_document_fetches_and_processes_html() -> None:
    html = """
    <html>
      <body>
        <main>
          <h1>Employer registration</h1>
          <p>Registration is required within 30 days.</p>
        </main>
      </body>
    </html>
    """

    def respond(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://example.com/guide"
        return httpx.Response(200, text=html, request=request)

    async def process() -> list[Document]:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(respond),
        ) as client:
            document_processor = DocumentProcessor(http_client=client)
            return await document_processor.process_document(make_source())

    documents = asyncio.run(process())

    assert len(documents) == 1
    assert "Employer registration" in documents[0].page_content
    assert documents[0].metadata["extracted_method"] == "http"


def test_process_document_converts_http_status_error_to_processing_error() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, request=request)

    async def fetch() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(respond),
        ) as client:
            document_processor = DocumentProcessor(http_client=client)

            with pytest.raises(
                DocumentProcessingError,
                match="http request failed",
            ):
                await document_processor.process_document(make_source())

    asyncio.run(fetch())


def test_process_document_fetches_and_processes_pdf() -> None:
    def write_page(page: pymupdf.Page) -> None:
        page.insert_text(  # pyright: ignore[reportUnknownMemberType]
            (72, 72),
            "Employer registration requirements",
        )

    pdf_content = build_pdf(write_page)

    def respond(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://example.com/guide.pdf"
        return httpx.Response(200, content=pdf_content, request=request)

    async def process() -> list[Document]:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(respond),
        ) as client:
            document_processor = DocumentProcessor(http_client=client)
            return await document_processor.process_document(
                make_source(
                    type="pdf",
                    url="https://example.com/guide.pdf",
                )
            )

    documents = asyncio.run(process())

    assert len(documents) == 1
    assert documents[0].page_content == "Employer registration requirements"
    assert documents[0].metadata["extracted_method"] == "text_layer"


def test_process_html_extracts_article_and_preserves_metadata(
    processor: DocumentProcessor,
) -> None:
    html = """
    <html>
      <body>
        <nav>Navigation menu</nav>
        <main>
          <h1>Employer registration</h1>
          <p>Registration is required within 30 days.</p>
        </main>
      </body>
    </html>
    """

    documents = processor.process_html(html, make_source())

    assert len(documents) == 1
    assert "Employer registration" in documents[0].page_content
    assert "Registration is required within 30 days" in documents[0].page_content
    assert "Navigation menu" not in documents[0].page_content
    assert documents[0].metadata["source_id"] == "business_registration_guide"
    assert documents[0].metadata["type"] == "html"
    assert documents[0].metadata["extracted_method"] == "http"
    assert re.fullmatch(
        r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}",
        str(documents[0].metadata["extracted_at"]),
    )


def test_process_html_rejects_empty_content(
    processor: DocumentProcessor,
) -> None:
    with pytest.raises(DocumentProcessingError, match="no extractable text found"):
        processor.process_html("<html><body></body></html>", make_source())


def test_process_pdf_extracts_one_document_per_non_empty_page(
    processor: DocumentProcessor,
) -> None:
    def write_pages(first_page: pymupdf.Page) -> None:
        first_page.insert_text((72, 72), "Employer registration requirements")  # pyright: ignore[reportUnknownMemberType]
        pdf = cast(pymupdf.Document, first_page.parent)
        second_page = pdf.new_page()
        second_page.insert_text((72, 72), "Contribution rules")  # pyright: ignore[reportUnknownMemberType]

    documents = processor.process_pdf(
        build_pdf(write_pages),
        make_source(type="pdf"),
    )

    assert [document.page_content for document in documents] == [
        "Employer registration requirements",
        "Contribution rules",
    ]
    assert [document.metadata["page"] for document in documents] == [1, 2]
    assert all(
        document.metadata["extracted_method"] == "text_layer"
        for document in documents
    )
    assert all(document.metadata["type"] == "pdf" for document in documents)


def test_process_pdf_skips_pages_without_text(
    processor: DocumentProcessor,
) -> None:
    documents = processor.process_pdf(
        build_pdf(lambda _page: None),
        make_source(type="pdf"),
    )

    assert documents == []


def test_process_pdf_rejects_invalid_pdf_bytes(
    processor: DocumentProcessor,
) -> None:
    with pytest.raises(DocumentProcessingError, match="invalid pdf data"):
        processor.process_pdf(b"not a PDF", make_source(type="pdf"))


def test_normalize_text_removes_blank_lines_and_surrounding_whitespace() -> None:
    value = "  Employer registration  \n\n  Contribution rules  "

    normalized = DocumentProcessor.normalize_text(value)

    assert normalized == "Employer registration\nContribution rules"


def test_source_metadata_contains_source_provenance() -> None:
    metadata = DocumentProcessor.source_metadata(make_source())

    assert metadata == {
        "source_id": "business_registration_guide",
        "agency": "Ministry of Economy and Finance",
        "title": "How to Register Your Business Online",
        "url": "https://example.com/guide",
        "topic": "registration_guide",
        "category": "business_registration",
        "published_year": 2021,
        "official": True,
        "type": "html",
    }


def test_source_config_rejects_unknown_document_type() -> None:
    with pytest.raises(ValueError, match="type"):
        make_source(type="spreadsheet")
