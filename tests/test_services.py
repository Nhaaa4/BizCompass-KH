from typing import cast

import pytest
from langchain_core.documents import Document
from sqlalchemy import Table
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Session
from sqlalchemy.schema import CreateTable

from bizcompass_kh.databases.models import Chunk, SourceDocument
from bizcompass_kh.databases.postgres import Postgres
from bizcompass_kh.evaluation import runner
from bizcompass_kh.evaluation.answer_eval import JudgeOutputError, parse_judge_output
from bizcompass_kh.evaluation.metrics import (
    RetrievalExample,
    calculate_retrieval_metrics,
)
from bizcompass_kh.schemas.evaluation import EvaluationQuestion
from bizcompass_kh.services.chunking import ChunkingService, count_tokens
from bizcompass_kh.services.llm import LLMResponse
from bizcompass_kh.services.rag import RAGService
from bizcompass_kh.services.vector_db import RankedChunk, SearchResult, reciprocal_rank_fusion


def test_chunking_stays_within_bounds_and_overlaps() -> None:
    document = Document(page_content=" ".join(f"word{i}" for i in range(1300)))
    chunks = ChunkingService(500, 700, 50).chunk_documents([document])

    assert [count_tokens(chunk.page_content) for chunk in chunks] == [700, 650]
    assert chunks[0].page_content.split()[-50:] == chunks[1].page_content.split()[:50]


def test_chunking_keeps_short_page() -> None:
    chunks = ChunkingService().chunk_documents([Document(page_content="one two")])
    assert len(chunks) == 1
    assert chunks[0].metadata["token_count"] == 2


def test_postgres_configures_a_dimensioned_vector_column() -> None:
    database = Postgres("postgresql://postgres:postgres@localhost:5433/bizcompass", 1024)
    try:
        ddl = str(CreateTable(cast(Table, Chunk.__table__)).compile(dialect=postgresql.dialect()))
    finally:
        database.close()

    assert "VECTOR(1024)" in ddl


def ranked(chunk_id: str, official: bool) -> RankedChunk:
    chunk = Chunk(
        chunk_id=chunk_id,
        source_id=f"source-{chunk_id}",
        chunk_index=0,
        page=1,
        content=chunk_id,
        token_count=1,
        embedding=[0.0] * 3,
    )
    document = SourceDocument(
        source_id=f"source-{chunk_id}",
        title=chunk_id,
        agency="agency",
        url="https://example.com",
        topic="topic",
        category="category",
        official=official,
        source_type="pdf",
        content_hash="hash",
    )
    return RankedChunk(chunk, document, 0)


def test_rrf_combines_rankings_and_deduplicates_chunks() -> None:
    a, b, c = ranked("a", False), ranked("b", True), ranked("c", False)
    results = reciprocal_rank_fusion([[a, b], [b, c]], limit=3, rank_constant=60)
    assert [result.chunk.chunk_id for result in results] == ["b", "a", "c"]


def test_rrf_prefers_official_source_on_equal_score() -> None:
    unofficial, official = ranked("a", False), ranked("b", True)
    results = reciprocal_rank_fusion([[unofficial, official], [official, unofficial]], limit=2)
    assert results[0].document.official is True


def test_retrieval_metrics() -> None:
    examples = [
        RetrievalExample({"a", "b"}, ["x", "a", "b"]),
        RetrievalExample({"c"}, ["z"]),
    ]
    metrics = calculate_retrieval_metrics(examples, 2)
    assert metrics.hit_rate_at_k == 0.5
    assert metrics.mrr == 0.25
    assert metrics.recall_at_k == 0.25


def test_judge_parser_accepts_fenced_text_and_detects_hallucination() -> None:
    score = parse_judge_output(
        '```json\n{"relevance":5,"groundedness":2,"completeness":4,'
        '"citation_correctness":3,"unsupported_claims":["fee"]}\n```'
    )
    assert score.hallucinated is True
    assert score.average_score == 3.5


def test_judge_parser_rejects_invalid_output() -> None:
    with pytest.raises(JudgeOutputError):
        parse_judge_output("not json")


def test_evaluation_progress_includes_percentage_question_and_scores() -> None:
    question = EvaluationQuestion(
        id="q015",
        question="How do I register a business?",
        expected_source_ids=["registration-guide"],
        expected_answer="Use the online registration portal.",
    )
    score = parse_judge_output(
        '{"relevance":5,"groundedness":4,"completeness":3,'
        '"citation_correctness":5,"unsupported_claims":[]}'
    )

    message = runner.format_evaluation_progress(question, score, 15, 75)

    assert message == "[15/75 |  20.0%] q015: relevance=5 groundedness=4 completeness=3 citations=5"


def test_rag_lets_the_llm_handle_messages_without_retrieved_evidence() -> None:
    class EmptyRetriever:
        def hybrid_search(self, session: Session, query: str, k: int = 5) -> list[SearchResult]:
            del session, query, k
            return []

    class ConversationalLLM:
        provider_name = "test"
        model = "test"

        def __init__(self) -> None:
            self.prompt = ""

        def generate(self, prompt: str) -> LLMResponse:
            self.prompt = prompt
            return LLMResponse("Hello! How can I help?")

    llm = ConversationalLLM()
    service = RAGService(EmptyRetriever(), llm)
    answer = service.answer(cast(Session, None), "Hello")

    assert answer.answer == "Hello! How can I help?"
    assert answer.citations == []
    assert "No relevant evidence was found" in llm.prompt


def test_rag_prompt_requires_grounded_practical_cited_guidance() -> None:
    class Retriever:
        def hybrid_search(self, session: Session, query: str, k: int = 5) -> list[SearchResult]:
            del session, query, k
            return [
                SearchResult(
                    chunk_id="chunk-1",
                    source_id="registration-guide",
                    content="Submit an application through the registration portal.",
                    page=2,
                    title="Registration Guide",
                    agency="Ministry of Commerce",
                    url="https://example.com/guide",
                    official=True,
                    score=1.0,
                )
            ]

    class CapturingLLM:
        provider_name = "test"
        model = "test"

        def __init__(self) -> None:
            self.prompt = ""

        def generate(self, prompt: str) -> LLMResponse:
            self.prompt = prompt
            return LLMResponse("Apply through the portal. [1]")

    llm = CapturingLLM()
    RAGService(Retriever(), llm).answer(cast(Session, None), "How do I register a business?")

    assert "Use the question's language" in llm.prompt
    assert "Do not use knowledge that is not in the evidence" in llm.prompt
    assert (
        "Do not invent fees, dates, forms, agencies, eligibility requirements, or timelines"
        in llm.prompt
    )
    assert "Markdown link such as [1](https://source-url)" in llm.prompt
    assert "URL: https://example.com/guide" in llm.prompt
