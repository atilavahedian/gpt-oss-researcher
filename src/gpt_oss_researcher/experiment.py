from collections import Counter
from typing import Dict, List, Tuple

from gpt_oss_researcher.providers import ModelProvider
from gpt_oss_researcher.schemas import Candidate, ExperimentConfig, PredictionRecord, StrategyOutput, Task
from gpt_oss_researcher.verifiers import VerifierSuite, is_correct, normalize_answer


def _choose_majority(candidates: List[Candidate]) -> Candidate:
    counts = Counter(normalize_answer(candidate.text) for candidate in candidates)
    if not counts:
        return Candidate(text="", sample_index=0, model="none")
    winner = counts.most_common(1)[0][0]
    for candidate in candidates:
        if normalize_answer(candidate.text) == winner:
            return candidate
    return candidates[0]


def _choose_by_verifier(task: Task, candidates: List[Candidate], verifier: VerifierSuite) -> Tuple[Candidate, float]:
    best_candidate = candidates[0]
    best_score = -1.0
    for candidate in candidates:
        score = verifier.score(task, candidate.text).score
        if score > best_score:
            best_candidate = candidate
            best_score = score
    return best_candidate, best_score


def _record(
    strategy: str,
    task: Task,
    candidates: List[Candidate],
    selected: Candidate,
    score: float,
    samples_used: int,
) -> PredictionRecord:
    return PredictionRecord(
        strategy=strategy,
        task_id=task.id,
        question=task.question,
        expected_answer=task.answer,
        selected_answer=selected.text,
        selected_model=selected.model,
        verifier_score=score,
        correct=is_correct(task.answer, selected.text),
        samples_used=samples_used,
        candidates=[candidate.to_dict() for candidate in candidates[:samples_used]],
    )


def evaluate_strategies(
    config: ExperimentConfig,
    provider: ModelProvider,
    verifier: VerifierSuite,
) -> Tuple[Dict[str, StrategyOutput], List[PredictionRecord]]:
    strategies: Dict[str, StrategyOutput] = {}
    records: List[PredictionRecord] = []

    for strategy in config.strategies:
        strategy_records: List[PredictionRecord] = []
        for task in config.tasks:
            candidates = provider.generate(task, sample_count=config.sample_count)
            if strategy == "single":
                selected = candidates[0]
                score = verifier.score(task, selected.text).score
                record = _record(strategy, task, candidates, selected, score, samples_used=1)
            elif strategy == "self_consistency":
                selected = _choose_majority(candidates)
                score = verifier.score(task, selected.text).score
                record = _record(strategy, task, candidates, selected, score, samples_used=len(candidates))
            elif strategy == "verifier_rerank":
                selected, score = _choose_by_verifier(task, candidates, verifier)
                record = _record(strategy, task, candidates, selected, score, samples_used=len(candidates))
            elif strategy == "adaptive_verifier_rerank":
                best = candidates[0]
                best_score = verifier.score(task, best.text).score
                samples_used = 1
                for candidate in candidates[1:]:
                    if best_score >= config.adaptive_threshold:
                        break
                    score = verifier.score(task, candidate.text).score
                    samples_used += 1
                    if score > best_score:
                        best = candidate
                        best_score = score
                record = _record(strategy, task, candidates, best, best_score, samples_used=samples_used)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            strategy_records.append(record)
            records.append(record)
        strategies[strategy] = StrategyOutput.from_records(strategy, strategy_records)

    return strategies, records

