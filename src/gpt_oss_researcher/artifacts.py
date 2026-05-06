import hashlib
import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

from gpt_oss_researcher.model_policy import ModelPolicy
from gpt_oss_researcher.schemas import ExperimentConfig, PredictionRecord, Task


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def stable_json(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stable_json(data) + "\n")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def dataset_sha256(tasks: List[Task]) -> str:
    payload = json.dumps([task.to_dict() for task in tasks], sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError:
        return "unavailable"
    if completed.returncode != 0:
        return "unavailable"
    return completed.stdout.strip() or "unavailable"


def build_manifest(
    config: ExperimentConfig,
    model_policy: ModelPolicy,
    provider_name: str,
    prediction_records: List[PredictionRecord],
) -> Dict[str, Any]:
    return {
        "created_at": utc_timestamp(),
        "project": "gpt-oss-researcher",
        "provider": provider_name,
        "python": platform.python_version(),
        "git_commit": git_commit(),
        "dataset_sha256": dataset_sha256(config.tasks),
        "model_policy": model_policy.to_dict(),
        "config": config.to_dict(include_tasks=False),
        "task_count": len(config.tasks),
        "prediction_records": len(prediction_records),
        "artifact_contract": {
            "run_manifest": "exact code, model policy, provider, config, and dataset hash",
            "predictions": "per-task candidates, selected output, verifier score, and correctness",
            "metrics": "strategy-level accuracy, sample use, and verifier score statistics",
            "research_report": "paper-style summary with hypothesis, results, limitations, and reproduction",
        },
    }

