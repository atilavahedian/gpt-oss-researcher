import re
import string
from dataclasses import dataclass
from typing import List

from gpt_oss_researcher.schemas import Task


def normalize_answer(text: str) -> str:
    lowered = text.strip().lower()
    lowered = lowered.translate(str.maketrans("", "", string.punctuation))
    lowered = re.sub(r"\b(a|an|the)\b", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def is_correct(expected: str, actual: str) -> bool:
    expected_norm = normalize_answer(expected)
    actual_norm = normalize_answer(actual)
    return actual_norm == expected_norm or expected_norm in actual_norm.split()


@dataclass(frozen=True)
class VerifierResult:
    score: float
    reasons: List[str]


class VerifierSuite:
    def score(self, task: Task, answer: str) -> VerifierResult:
        reasons: List[str] = []
        exact = 1.0 if is_correct(task.answer, answer) else 0.0
        if exact:
            reasons.append("exact_answer_match")
        else:
            reasons.append("exact_answer_mismatch")

        if task.rubric_keywords:
            normalized_answer = normalize_answer(answer)
            hits = 0
            for keyword in task.rubric_keywords:
                if normalize_answer(keyword) in normalized_answer:
                    hits += 1
            keyword_score = hits / len(task.rubric_keywords)
            reasons.append(f"rubric_keyword_hits={hits}/{len(task.rubric_keywords)}")
        else:
            keyword_score = exact
            reasons.append("no_rubric_keywords")

        score = min(1.0, (0.85 * exact) + (0.15 * keyword_score))
        return VerifierResult(score=score, reasons=reasons)

