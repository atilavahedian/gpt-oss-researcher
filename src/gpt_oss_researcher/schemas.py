from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class Task:
    id: str
    question: str
    answer: str
    rubric_keywords: List[str] = field(default_factory=list)
    fixture_candidates: List[str] = field(default_factory=list)
    source: str = "synthetic_control"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "rubric_keywords": list(self.rubric_keywords),
            "fixture_candidates": list(self.fixture_candidates),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Task":
        return cls(
            id=payload["id"],
            question=payload["question"],
            answer=payload["answer"],
            rubric_keywords=list(payload.get("rubric_keywords", [])),
            fixture_candidates=list(payload.get("fixture_candidates", [])),
            source=payload.get("source", "synthetic_control"),
        )


@dataclass(frozen=True)
class ExperimentConfig:
    title: str
    research_question: str
    hypothesis: str
    tasks: List[Task]
    strategies: List[str]
    sample_count: int = 4
    adaptive_threshold: float = 0.99

    def to_dict(self, include_tasks: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "title": self.title,
            "research_question": self.research_question,
            "hypothesis": self.hypothesis,
            "strategies": list(self.strategies),
            "sample_count": self.sample_count,
            "adaptive_threshold": self.adaptive_threshold,
        }
        if include_tasks:
            payload["tasks"] = [task.to_dict() for task in self.tasks]
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ExperimentConfig":
        return cls(
            title=payload["title"],
            research_question=payload["research_question"],
            hypothesis=payload["hypothesis"],
            tasks=[Task.from_dict(task) for task in payload["tasks"]],
            strategies=list(payload["strategies"]),
            sample_count=int(payload.get("sample_count", 4)),
            adaptive_threshold=float(payload.get("adaptive_threshold", 0.99)),
        )


@dataclass(frozen=True)
class Candidate:
    text: str
    sample_index: int
    model: str

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "sample_index": self.sample_index, "model": self.model}


@dataclass(frozen=True)
class PredictionRecord:
    strategy: str
    task_id: str
    question: str
    expected_answer: str
    selected_answer: str
    selected_model: str
    verifier_score: float
    correct: bool
    samples_used: int
    candidates: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "task_id": self.task_id,
            "question": self.question,
            "expected_answer": self.expected_answer,
            "selected_answer": self.selected_answer,
            "selected_model": self.selected_model,
            "verifier_score": self.verifier_score,
            "correct": self.correct,
            "samples_used": self.samples_used,
            "candidates": self.candidates,
        }


@dataclass(frozen=True)
class StrategyOutput:
    strategy: str
    total_tasks: int
    correct_tasks: int
    accuracy: float
    avg_samples_per_task: float
    mean_verifier_score: float

    @classmethod
    def from_records(cls, strategy: str, records: List[PredictionRecord]) -> "StrategyOutput":
        total = len(records)
        correct = sum(1 for record in records if record.correct)
        samples = sum(record.samples_used for record in records)
        score = sum(record.verifier_score for record in records)
        return cls(
            strategy=strategy,
            total_tasks=total,
            correct_tasks=correct,
            accuracy=correct / total if total else 0.0,
            avg_samples_per_task=samples / total if total else 0.0,
            mean_verifier_score=score / total if total else 0.0,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "total_tasks": self.total_tasks,
            "correct_tasks": self.correct_tasks,
            "accuracy": self.accuracy,
            "avg_samples_per_task": self.avg_samples_per_task,
            "mean_verifier_score": self.mean_verifier_score,
        }

