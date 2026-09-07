import json
import re

from bizcompass_kh.schemas.evaluation import JudgeScore
from bizcompass_kh.services.llm import LLMProvider


class JudgeOutputError(ValueError):
    pass


def parse_judge_output(text: str) -> JudgeScore:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise JudgeOutputError("judge response does not contain JSON")
    try:
        return JudgeScore.model_validate(json.loads(match.group()))
    except (json.JSONDecodeError, ValueError) as error:
        raise JudgeOutputError(f"invalid judge response: {error}") from error


class AnswerJudge:
    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

    def evaluate(
        self,
        question: str,
        answer: str,
        evidence: str,
        expected_answer: str,
    ) -> JudgeScore:
        prompt = f"""You are a strict RAG evaluator. Score each dimension from 1 to 5.
                Unsupported claims are answer claims not supported by Evidence. Return JSON only:
                {{"relevance": 1, "groundedness": 1, "completeness": 1,
                "citation_correctness": 1, "unsupported_claims": []}}
                Question: {question}
                Expected answer: {expected_answer}
                Evidence: {evidence}
                Generated answer: {answer}
                """.strip()
        return parse_judge_output(self.llm.generate(prompt).text)
