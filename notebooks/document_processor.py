# %%
import asyncio
from pathlib import Path

import httpx
import yaml
from langchain_core.documents import Document
from pydantic import TypeAdapter

from bizcompass_kh.schemas.documents import SourceConfig
from bizcompass_kh.services.document_processor import DocumentProcessor

# %%
project_root = Path.cwd()
if project_root.name == "notebooks":
    project_root = project_root.parent

sources_path = project_root / "data" / "sources.yaml"
raw_sources = yaml.safe_load(sources_path.read_text(encoding="utf-8"))
sources = TypeAdapter(list[SourceConfig]).validate_python(raw_sources)

print(f"Loaded {len(sources)} validated English sources")


async def process_source(
    source: SourceConfig,
    *,
    timeout: float,
) -> list[Document]:
    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=timeout,
    ) as client:
        processor = DocumentProcessor(http_client=client)
        return await processor.process_document(source)


# %%
html_source = next(source for source in sources if source.type == "html")

html_documents = asyncio.run(process_source(html_source, timeout=30))

print(f"HTML source: {html_source.title}")
print(f"Documents: {len(html_documents)}")
print(html_documents[0].page_content[:500])


# %%
pdf_source = next(source for source in sources if source.type == "pdf")

pdf_documents = asyncio.run(process_source(pdf_source, timeout=60))

print(f"PDF source: {pdf_source.title}")
print(f"Extracted pages: {len(pdf_documents)}")
print(pdf_documents[0].page_content[:500])
