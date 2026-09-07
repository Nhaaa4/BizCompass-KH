from statistics import fmean

from pydantic import BaseModel, Field


class EvaluationQuestion(BaseModel):
    id: str
    question: str
    expected_source_ids: list[str] = Field(min_length=1)
    expected_answer: str


class JudgeScore(BaseModel):
    relevance: int = Field(ge=1, le=5)
    groundedness: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    citation_correctness: int = Field(ge=1, le=5)
    unsupported_claims: list[str] = Field(default_factory=list)

    @property
    def hallucinated(self) -> bool:
        return bool(self.unsupported_claims)

    @property
    def average_score(self) -> float:
        return fmean(
            [
                self.relevance,
                self.groundedness,
                self.completeness,
                self.citation_correctness,
            ]
        )
