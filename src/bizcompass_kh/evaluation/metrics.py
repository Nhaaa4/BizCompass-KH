from dataclasses import dataclass
from statistics import fmean


@dataclass(frozen=True)
class RetrievalExample:
    expected_source_ids: set[str]
    retrieved_source_ids: list[str]


@dataclass(frozen=True)
class RetrievalMetrics:
    hit_rate_at_k: float
    mrr: float
    recall_at_k: float


def hit_rate_at_k(examples: list[RetrievalExample], k: int) -> float:
    if not examples:
        return 0.0
    return fmean(
        bool(example.expected_source_ids & set(example.retrieved_source_ids[:k]))
        for example in examples
    )


def mean_reciprocal_rank(examples: list[RetrievalExample]) -> float:
    if not examples:
        return 0.0
    reciprocal_ranks: list[float] = []
    for example in examples:
        rank = next(
            (
                index
                for index, source_id in enumerate(example.retrieved_source_ids, start=1)
                if source_id in example.expected_source_ids
            ),
            None,
        )
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
    return fmean(reciprocal_ranks)


def recall_at_k(examples: list[RetrievalExample], k: int) -> float:
    if not examples:
        return 0.0
    return fmean(
        len(example.expected_source_ids & set(example.retrieved_source_ids[:k]))
        / len(example.expected_source_ids)
        if example.expected_source_ids
        else 0.0
        for example in examples
    )


def calculate_retrieval_metrics(examples: list[RetrievalExample], k: int) -> RetrievalMetrics:
    return RetrievalMetrics(
        hit_rate_at_k(examples, k),
        mean_reciprocal_rank(examples),
        recall_at_k(examples, k),
    )
