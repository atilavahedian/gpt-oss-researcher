import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from gpt_oss_researcher.artifacts import stable_json, write_json, write_jsonl
from gpt_oss_researcher.pipeline import run_research_pipeline
from gpt_oss_researcher.reporting import render_research_plan
from gpt_oss_researcher.schemas import ExperimentConfig
from gpt_oss_researcher.templates import default_experiment_config


def _load_config(path: Optional[str]) -> ExperimentConfig:
    if path is None:
        return default_experiment_config()
    payload = json.loads(Path(path).read_text())
    return ExperimentConfig.from_dict(payload)


def _cmd_init(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    config = default_experiment_config()
    write_json(output_dir / "experiment_config.json", config.to_dict())
    write_jsonl(output_dir / "tasks.jsonl", [task.to_dict() for task in config.tasks])
    (output_dir / "research_plan.md").write_text(render_research_plan(config))
    print(output_dir / "experiment_config.json")
    print(output_dir / "research_plan.md")
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_research_plan(config))
    print(output_path)
    return 0


def _cmd_all(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    result = run_research_pipeline(config=config, output_dir=Path(args.output_dir), provider_name=args.provider)
    print("Wrote research artifacts:")
    for key, path in sorted(result.artifacts.items()):
        print(f"- {key}: {path}")
    return 0


def _cmd_print_default_config(args: argparse.Namespace) -> int:
    sys.stdout.write(stable_json(default_experiment_config().to_dict()) + "\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpt-oss-researcher",
        description="Run evidence-first verifier-guided test-time compute experiments.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    init_parser = subcommands.add_parser("init", help="write a starter experiment config and plan")
    init_parser.add_argument("--output-dir", default="experiments/verifier_ttc")
    init_parser.set_defaults(func=_cmd_init)

    plan_parser = subcommands.add_parser("plan", help="render the research plan for a config")
    plan_parser.add_argument("--config", default=None)
    plan_parser.add_argument("--output", default="research_plan.md")
    plan_parser.set_defaults(func=_cmd_plan)

    all_parser = subcommands.add_parser("all", help="run plan, experiment, analysis, and report")
    all_parser.add_argument("--config", default=None)
    all_parser.add_argument("--output-dir", default="results/latest")
    all_parser.add_argument("--provider", default="fixture", choices=["fixture", "openai", "transformers"])
    all_parser.set_defaults(func=_cmd_all)

    config_parser = subcommands.add_parser("default-config", help="print the default experiment config JSON")
    config_parser.set_defaults(func=_cmd_print_default_config)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

