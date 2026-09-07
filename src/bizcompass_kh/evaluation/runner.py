import csv
import json
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any
from uuid import uuid4

from sqlalchemy.orm import Session

from bizcompass_kh.config.settings import Settings, get_settings
from bizcompass_kh.databases.models import EvaluationScore
from bizcompass_kh.databases.postgres import Postgres
from bizcompass_kh.evaluation.answer_eval import AnswerJudge
from bizcompass_kh.evaluation.metrics import (
    RetrievalExample,
    calculate_retrieval_metrics,
)
from bizcompass_kh.schemas.evaluation import EvaluationQuestion, JudgeScore
from bizcompass_kh.services.embedding import create_embedding_service
from bizcompass_kh.services.llm import create_provider
from bizcompass_kh.services.monitoring import (
    EVALUATION_QUESTIONS,
    HALLUCINATIONS,
    JUDGE_SCORE,
)
from bizcompass_kh.services.rag import RAGService
from bizcompass_kh.services.vector_db import VectorDBService

EvaluationProgressReporter = Callable[[EvaluationQuestion, JudgeScore, int, int], None]


def load_questions(path: str | Path) -> list[EvaluationQuestion]:
    with Path(path).open(encoding="utf-8") as stream:
        payload: Any = json.load(stream)
    questions = [EvaluationQuestion.model_validate(item) for item in payload]
    if len(questions) < 50:
        raise ValueError("evaluation dataset must contain at least 50 questions")
    return questions


def save_judge_score(session: Session, run_id: str, question_id: str, score: JudgeScore) -> None:
    session.add(
        EvaluationScore(
            run_id=run_id,
            question_id=question_id,
            relevance=score.relevance,
            groundedness=score.groundedness,
            completeness=score.completeness,
            citation_correctness=score.citation_correctness,
            hallucinated=score.hallucinated,
        )
    )


def format_evaluation_progress(
    question: EvaluationQuestion, score: JudgeScore, completed: int, total: int
) -> str:
    percentage = completed / total * 100 if total else 100.0
    return (
        f"[{completed}/{total} | {percentage:5.1f}%] {question.id}: "
        f"relevance={score.relevance} groundedness={score.groundedness} "
        f"completeness={score.completeness} citations={score.citation_correctness}"
    )


def run_evaluation(
    settings: Settings | None = None,
    on_progress: EvaluationProgressReporter | None = None,
) -> dict[str, Any]:
    config = settings or get_settings()
    embedder = create_embedding_service(config)
    embedder.warmup()
    answer_provider = create_provider(config)
    judge_provider = answer_provider
    database = Postgres(config.postgres_dsn, config.embedding_dimension)
    database.init_database()
    retriever = VectorDBService(
        embedder, candidate_k=config.retrieval_candidate_k, rrf_k=config.rrf_k
    )
    rag = RAGService(retriever, answer_provider, retrieval_k=config.retrieval_k)
    judge = AnswerJudge(judge_provider)
    run_id = uuid4().hex
    rows: list[dict[str, Any]] = []
    retrieval_examples: list[RetrievalExample] = []

    questions = load_questions(config.eval_questions_path)
    total = len(questions)
    for completed, question in enumerate(questions, start=1):
        with database.session() as session:
            answer = rag.answer(session, question.question)
            retrieved_ids = [result.source_id for result in answer.retrieved]
            retrieval_examples.append(
                RetrievalExample(set(question.expected_source_ids), retrieved_ids)
            )
            evidence = "\n\n".join(result.content for result in answer.retrieved)
            score = judge.evaluate(
                question.question, answer.answer, evidence, question.expected_answer
            )
            save_judge_score(session, run_id, question.id, score)
        EVALUATION_QUESTIONS.inc()
        if score.hallucinated:
            HALLUCINATIONS.inc()
        for metric in ("relevance", "groundedness", "completeness", "citation_correctness"):
            JUDGE_SCORE.labels(metric).set(getattr(score, metric))
        rows.append(
            {
                "id": question.id,
                "question": question.question,
                "answer": answer.answer,
                "expected_source_ids": question.expected_source_ids,
                "retrieved_source_ids": retrieved_ids,
                **score.model_dump(),
                "hallucinated": score.hallucinated,
            }
        )
        if on_progress:
            on_progress(question, score, completed, total)

    retrieval = calculate_retrieval_metrics(retrieval_examples, config.retrieval_k)
    summary = {
        **asdict(retrieval),
        "answer_relevance": fmean(row["relevance"] for row in rows) / 5,
        "groundedness": fmean(row["groundedness"] for row in rows) / 5,
        "citation_correctness": fmean(row["citation_correctness"] for row in rows) / 5,
        "hallucination_rate": fmean(bool(row["hallucinated"]) for row in rows),
    }
    output_dir = Path(config.eval_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"evaluation-{stamp}.json"
    csv_path = output_dir / f"evaluation-{stamp}.csv"
    payload = json.dumps({"summary": summary, "results": rows}, indent=2)
    json_path.write_text(payload, encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    database.close()
    return {"summary": summary, "json": str(json_path), "csv": str(csv_path)}


def main() -> None:
    def print_progress(
        question: EvaluationQuestion, score: JudgeScore, completed: int, total: int
    ) -> None:
        print(format_evaluation_progress(question, score, completed, total), flush=True)

    print(json.dumps(run_evaluation(on_progress=print_progress), indent=2))


if __name__ == "__main__":
    main()
