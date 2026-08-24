import unicodedata
from datetime import datetime
from typing import cast
from zoneinfo import ZoneInfo

import httpx
import pymupdf
import trafilatura
from langchain_core.documents import Document

from bizcompass_kh.schemas.documents import SourceConfig


class DocumentProcessingError(RuntimeError):
    """Raised when document processing fails."""

class DocumentProcessor:
    def __init__(
        self,
        *,
        http_client: httpx.AsyncClient,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._http_client = http_client
        self._timeout_seconds = timeout_seconds


    async def _fetch(self, source: SourceConfig) -> httpx.Response:
        try:
            response = await self._http_client.get(
                source.url,
                follow_redirects=True,
                timeout=self._timeout_seconds,
            )
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as e:
            raise DocumentProcessingError(f"source: {source.id} http request failed.") from e

    async def process_document(self, source: SourceConfig) -> list[Document]:
        response = await self._fetch(source)

        if source.type == "html":
            return self.process_html(response.text, source)

        return self.process_pdf(response.content, source)

    def process_pdf(self, content: bytes, source: SourceConfig) -> list[Document]:
        documents: list[Document] = []

        try:
            pdf = pymupdf.open(stream=content, filetype="pdf")
        except Exception as e:
            raise DocumentProcessingError(f"source: {source.id} invalid pdf data.") from e

        try:
            for page_index in range(len(pdf)):
                page = pdf[page_index]
                page_num = page_index + 1
                text = cast(
                    str,
                    page.get_text("text"),  # pyright: ignore[reportUnknownMemberType]
                )
                text = self.normalize_text(text)

                if not text:
                    continue

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            **self.source_metadata(source),
                            "page": page_num,
                            "extracted_method": "text_layer",
                            "extracted_at": datetime.now(ZoneInfo("Asia/Phnom_Penh")).strftime("%Y-%m-%d %H:%M:%S"),
                        }
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
            favor_precision=True
        )

        text = self.normalize_text(extracted or "")

        if not text:
            raise DocumentProcessingError(
                f"source: {source.id}, no extractable text found."
            )
        return [
            Document(
                page_content=text,
                metadata={
                    **self.source_metadata(source),
                    "extracted_method": "http",
                    "extracted_at": datetime.now(ZoneInfo("Asia/Phnom_Penh")).strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
        ]
        
    @staticmethod
    def normalize_text(value: str) -> str:
        normalize = unicodedata.normalize("NFC", value)
        return "\n".join(
            line.strip() 
            for line in normalize.splitlines()
            if line.strip()
        )

    @staticmethod
    def source_metadata(source: SourceConfig) -> dict[str, object]:
        return {
            "source_id": source.id,
            "agency": source.agency,
            "title": source.title,
            "url": source.url,
            "topic": source.topic,
            "category": source.category,
            "published_year": source.published_year,
            "official": source.official,
            "type": source.type,
        }