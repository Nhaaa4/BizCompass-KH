import unicodedata
from datetime import UTC, datetime
from typing import cast

import httpx
import pymupdf
import trafilatura
from langchain_core.documents import Document

from bizcompass_kh.schemas.documents import SourceConfig

USER_AGENT = "Mozilla/5.0"


class DocumentProcessingError(RuntimeError):
    """Raised when downloading or extracting a source fails."""


class DocumentProcessor:
    def __init__(
        self,
        *,
        http_client: httpx.Client,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._http_client = http_client
        self._timeout_seconds = timeout_seconds

    def download(self, source: SourceConfig) -> bytes:
        try:
            response = self._http_client.get(
                str(source.url),
                headers={"User-Agent": USER_AGENT},
                follow_redirects=True,
                timeout=self._timeout_seconds,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise DocumentProcessingError(
                f"source {source.id}: HTTP {error.response.status_code}"
            ) from error
        except httpx.HTTPError as error:
            raise DocumentProcessingError(f"source {source.id}: HTTP request failed") from error
        return response.content

    def process_document(self, source: SourceConfig) -> list[Document]:
        content = self.download(source)
        if source.type == "html":
            return self.process_html(content.decode("utf-8", errors="replace"), source)
        return self.process_pdf(content, source)

    def process_pdf(self, content: bytes, source: SourceConfig) -> list[Document]:
        documents: list[Document] = []
        try:
            pdf = pymupdf.open(stream=content, filetype="pdf")
        except Exception as error:
            raise DocumentProcessingError(f"source {source.id}: invalid PDF data") from error

        try:
            for page_index in range(len(pdf)):
                text = cast(str, pdf[page_index].get_text("text"))
                text = self.normalize_text(text)
                if text:
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                **self.source_metadata(source),
                                "page": page_index + 1,
                                "extracted_method": "text_layer",
                                "extracted_at": datetime.now(UTC).isoformat(),
                            },
                        )
                    )
        finally:
            pdf.close()
        return documents

    def process_html(self, content: str, source: SourceConfig) -> list[Document]:
        extracted = trafilatura.extract(
            content,
            include_tables=True,
            include_links=False,
            favor_precision=True,
        )
        text = self.normalize_text(extracted or "")
        if not text:
            raise DocumentProcessingError(f"source {source.id}: no extractable text found")
        return [
            Document(
                page_content=text,
                metadata={
                    **self.source_metadata(source),
                    "page": None,
                    "extracted_method": "html",
                    "extracted_at": datetime.now(UTC).isoformat(),
                },
            )
        ]

    @staticmethod
    def normalize_text(value: str) -> str:
        normalized = unicodedata.normalize("NFC", value)
        return "\n".join(line.strip() for line in normalized.splitlines() if line.strip())

    @staticmethod
    def source_metadata(source: SourceConfig) -> dict[str, object]:
        return {
            "source_id": source.id,
            "agency": source.agency,
            "title": source.title,
            "url": str(source.url),
            "topic": source.topic,
            "category": source.category,
            "published_year": source.published_year,
            "official": source.official,
            "type": source.type,
        }
