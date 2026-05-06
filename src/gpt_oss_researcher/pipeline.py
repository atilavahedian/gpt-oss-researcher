from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from gpt_oss_researcher.artifacts import build_manifest, write_json, write_jsonl
from gpt_oss_researcher.experiment import evaluate_strategies
from gpt_oss_researcher.model_policy import default_model_policy
from gpt_oss_researcher.providers import build_provider
from gpt_oss_researcher.reporting import render_analysis, render_research_plan, render_research_report
from gpt_oss_researcher.schemas import ExperimentConfig
from gpt_oss_researcher.verifiers import VerifierSuite


@dataclass
class PipelineResult:
    metrics: Dict[str, Any]
    artifacts: Dict[str, Path]


def _metrics_from_outputs(outputs: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "strategies": {name: output.to_dict() for name, output in outputs.items()},
        "best_strategy": max(outputs.values(), key=lambda output: output.accuracy).strategy,
    }


def run_research_pipeline(
    config: ExperimentConfig,
    output_dir: Path,
    provider_name: str = "fixture",
) -> PipelineResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_policy = default_model_policy()
    provider = build_provider(provider_name, model_policy)
    verifier = VerifierSuite()
    strategy_outputs, prediction_records = evaluate_strategies(config, provider, verifier)

    metrics = _metrics_from_outputs(strategy_outputs)
    manifest = build_manifest(config, model_policy, provider_name, prediction_records)
    plan = render_research_plan(config)
    analysis = render_analysis(config, metrics)
    report = render_research_report(config, model_policy, metrics, manifest)

    artifacts = {
        "research_plan": output_dir / "research_plan.md",
        "run_manifest": output_dir / "run_manifest.json",
        "predictions": output_dir / "predictions.jsonl",
        "metrics": output_dir / "metrics.json",
        "analysis": output_dir / "analysis.md",
        "research_report": output_dir / "research_report.md",
    }

    artifacts["research_plan"].write_text(plan)
    write_json(artifacts["run_manifest"], manifest)
    write_jsonl(artifacts["predictions"], [record.to_dict() for record in prediction_records])
    write_json(artifacts["metrics"], metrics)
    artifacts["analysis"].write_text(analysis)
    artifacts["research_report"].write_text(report)

    return PipelineResult(metrics=metrics, artifacts=artifacts)

