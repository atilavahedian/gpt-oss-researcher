# gpt-oss-researcher

An evidence-first AI research repo for verifier-guided test-time compute.

This is not a product, web app, or chatbot wrapper. It is a reproducible research codebase: an AI researcher proposes a hypothesis, runs a controlled experiment, records artifacts, analyzes the result, and writes a paper-style report.

The default model policy prefers American/open-weight models, especially `openai/gpt-oss-20b` and `openai/gpt-oss-120b`. Local verification uses a deterministic fixture provider so the repository can be tested without downloading large weights.

## Research Question

Can verifier-guided test-time compute improve answer accuracy over a single generation while making the extra sample cost explicit?

## What The Agent Does

1. Builds a research plan from an experiment config.
2. Runs multiple strategies over the same benchmark tasks.
3. Scores each candidate with a verifier suite.
4. Writes a manifest, prediction log, metrics, analysis note, and research report.
5. Keeps the artifact trail explicit enough to reproduce or challenge the claim.

## Strategies

- `single`: one model answer.
- `self_consistency`: multiple answers with majority vote.
- `verifier_rerank`: multiple answers scored by a verifier suite.
- `adaptive_verifier_rerank`: sample until the verifier is confident or the sample budget is exhausted.

## Quickstart

```bash
cd gpt-oss-researcher
PYTHONPATH=src python3 -m gpt_oss_researcher.cli all --provider fixture --output-dir results/fixture_verifier_ttc
python3 -m unittest discover -s tests
```

The fixture provider is deterministic and exists to prove the research machinery. Real model runs can use optional providers:

```bash
python3 -m pip install -e '.[models]'
OPENAI_API_KEY=... OPENAI_RESEARCH_MODEL=<openai-api-model> \
  gpt-oss-researcher all --provider openai --output-dir results/openai_run

HF_RESEARCH_MODEL=openai/gpt-oss-20b \
  gpt-oss-researcher all --provider transformers --output-dir results/hf_run
```

Large `gpt-oss` runs require suitable hosted inference or GPU memory.

## Artifact Contract

Every run writes:

- `research_plan.md`
- `run_manifest.json`
- `predictions.jsonl`
- `metrics.json`
- `analysis.md`
- `research_report.md`

The checked-in fixture run is under `results/fixture_verifier_ttc`.

## Repository Layout

```text
src/gpt_oss_researcher/
  model_policy.py      # gpt-oss-first model preference
  providers.py         # fixture, OpenAI, and Transformers providers
  experiment.py        # strategy execution
  verifiers.py         # answer-key and rubric verifier suite
  pipeline.py          # end-to-end artifact-producing research pipeline
  reporting.py         # research plan, analysis, and report rendering
experiments/
  verifier_ttc/        # starter config and task file
results/
  fixture_verifier_ttc/# reproducible checked-in fixture run
tests/
  test_*.py            # behavior tests for the research pipeline and CLI
```

## Why This Is Serious

The project does not claim frontier performance from a small fixture run. It proves the machinery needed for serious research:

- explicit hypothesis
- controlled strategies
- comparable metrics
- sample-cost accounting
- model policy provenance
- reproducible artifacts
- limitations written into the report

That gives a clean path to scale from fixture verification to real `gpt-oss` inference.
