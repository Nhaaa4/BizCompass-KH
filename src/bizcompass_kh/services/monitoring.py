from dataclasses import dataclass
from threading import Lock

from prometheus_client import Counter, Gauge, Histogram, start_http_server
from sqlalchemy.orm import Session

from bizcompass_kh.databases.models import Feedback, RagRequest, RagRetrieval
from bizcompass_kh.services.rag import RAGAnswer

REQUESTS = Counter("bizcompass_rag_requests_total", "RAG requests", ["provider", "status"])
LATENCY = Histogram("bizcompass_rag_latency_seconds", "RAG request latency")
ERRORS = Counter("bizcompass_rag_errors_total", "RAG errors", ["error_type"])
TOKENS = Counter("bizcompass_llm_tokens_total", "LLM tokens", ["kind", "provider"])
RETRIEVED = Histogram("bizcompass_retrieved_chunks", "Chunks retrieved per request")
FEEDBACK_SCORE = Gauge("bizcompass_feedback_score", "Most recent user feedback score")
JUDGE_SCORE = Gauge("bizcompass_judge_score", "Most recent judge score", ["metric"])
HALLUCINATIONS = Counter("bizcompass_hallucinations_total", "Judge-detected hallucinations")
EVALUATION_QUESTIONS = Counter("bizcompass_evaluation_questions_total", "Evaluated questions")
_METRICS_LOCK = Lock()
_METRICS_STARTED = False


@dataclass(frozen=True)
class RequestLog:
    question: str
    provider: str
    model: str
    latency_ms: float
    result: RAGAnswer | None = None
    error: str | None = None


def start_metrics_server(port: int) -> None:
    global _METRICS_STARTED
    with _METRICS_LOCK:
        if not _METRICS_STARTED:
            start_http_server(port)
            _METRICS_STARTED = True


def save_request(session: Session, log: RequestLog) -> int:
    result = log.result
    row = RagRequest(
        question=log.question,
        answer=result.answer if result else None,
        provider=log.provider,
        model=log.model,
        latency_ms=log.latency_ms,
        retrieved_chunks=len(result.retrieved) if result else 0,
        prompt_tokens=result.prompt_tokens if result else 0,
        completion_tokens=result.completion_tokens if result else 0,
        error=log.error,
    )
    session.add(row)
    session.flush()
    if result:
        for rank, chunk in enumerate(result.retrieved, start=1):
            session.add(
                RagRetrieval(
                    request_id=row.id,
                    chunk_id=chunk.chunk_id,
                    rank=rank,
                    score=chunk.score,
                )
            )
    status = "error" if log.error else "success"
    REQUESTS.labels(log.provider, status).inc()
    LATENCY.observe(log.latency_ms / 1000)
    RETRIEVED.observe(len(result.retrieved) if result else 0)
    if result:
        TOKENS.labels("prompt", log.provider).inc(result.prompt_tokens)
        TOKENS.labels("completion", log.provider).inc(result.completion_tokens)
    if log.error:
        ERRORS.labels(type(log.error).__name__).inc()
    return row.id


def save_feedback(session: Session, request_id: int, score: int) -> None:
    if score not in {-1, 1}:
        raise ValueError("feedback score must be -1 or 1")
    session.add(Feedback(request_id=request_id, score=score))
    FEEDBACK_SCORE.set(score)
