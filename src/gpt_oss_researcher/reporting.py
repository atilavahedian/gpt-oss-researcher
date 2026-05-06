from typing import Any, Dict, List

from gpt_oss_researcher.model_policy import ModelPolicy
from gpt_oss_researcher.schemas import ExperimentConfig


def _metrics_table(metrics: Dict[str, Any]) -> str:
    rows = ["| Strategy | Accuracy | Avg samples/task | Mean verifier score |", "| --- | ---: | ---: | ---: |"]
    for name, data in metrics["strategies"].items():
        rows.append(
            "| {name} | {accuracy:.3f} | {samples:.2f} | {score:.3f} |".format(
                name=name,
                accuracy=data["accuracy"],
                samples=data["avg_samples_per_task"],
                score=data["mean_verifier_score"],
            )
        )
    return "\n".join(rows)


def render_research_plan(config: ExperimentConfig) -> str:
    strategies = "\n".join(f"- `{strategy}`" for strategy in config.strategies)
    tasks = "\n".join(f"- `{task.id}`: {task.question}" for task in config.tasks)
    return f"""# Research Plan: {config.title}

## Research Question
{config.research_question}

## Hypothesis
{config.hypothesis}

## Experimental Design
- Benchmark tasks: {len(config.tasks)}
- Candidate samples per fixed-compute strategy: {config.sample_count}
- Adaptive stopping threshold: {config.adaptive_threshold}
- Evaluation contract: every run must write a manifest, predictions JSONL, metrics JSON, analysis note, and research report.

## Strategies
{strategies}

## Task Set
{tasks}

## Success Criteria
Verifier-guided test-time compute should beat the single-answer baseline while reporting the extra sample cost and the failure modes.
"""


def render_analysis(config: ExperimentConfig, metrics: Dict[str, Any]) -> str:
    best = metrics["best_strategy"]
    table = _metrics_table(metrics)
    return f"""# Analysis

The best observed strategy for `{config.title}` was `{best}`.

{table}

The analysis treats sample count as a first-order cost. A strategy that improves accuracy but spends many more samples is not automatically better; the report compares both.
"""


def render_research_report(
    config: ExperimentConfig,
    model_policy: ModelPolicy,
    metrics: Dict[str, Any],
    manifest: Dict[str, Any],
) -> str:
    table = _metrics_table(metrics)
    best = metrics["best_strategy"]
    return f"""# {config.title}

## Abstract
This repo studies whether verifier-guided test-time compute can improve answer quality on bounded benchmark tasks. The agent runs multiple decoding strategies, scores their outputs with a verifier suite, records reproducible artifacts, and writes this report from the recorded metrics.

## Research Question
{config.research_question}

## Hypothesis
{config.hypothesis}

## Model Policy
The preferred generator is `{model_policy.primary_model}`. Preferred verifier models are {", ".join(f"`{model}`" for model in model_policy.verifier_models)}. The local fixture provider is used for reproducible CI and does not claim to load the large models.

## Method
The experiment compares single-answer generation, self-consistency, fixed verifier reranking, and adaptive verifier reranking. Each strategy is evaluated on the same task set. The artifact contract records the provider, dataset hash, model policy, predictions, verifier scores, and metrics.

## Results
{table}

Best strategy: `{best}`.

## Interpretation
Verifier reranking is useful when the correct answer appears somewhere in the candidate set and the verifier can identify it. Adaptive reranking can reduce sample use by stopping when a high-confidence answer appears early.

## Limitations
- Fixture runs prove the research machinery, not frontier-model performance.
- Real `gpt-oss` runs require local GPU memory or a hosted inference provider.
- Answer-key verifiers are appropriate for benchmark research but are not a substitute for human review on open-ended scientific claims.
- The task set is intentionally small in the checked-in example; serious runs should expand the benchmark matrix.

## Reproducibility
- Dataset SHA-256: `{manifest["dataset_sha256"]}`
- Provider: `{manifest["provider"]}`
- Python: `{manifest["python"]}`
- Git commit: `{manifest["git_commit"]}`

Run:

```bash
python3 -m gpt_oss_researcher.cli all --provider fixture --output-dir results/latest
```
"""

