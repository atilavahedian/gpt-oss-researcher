# Methodology

`gpt-oss-researcher` treats research claims as artifact-backed claims.

## Experiment Contract

Each experiment starts from:

- a research question
- a hypothesis
- a fixed task set
- a list of strategies
- a model policy
- a provider

The pipeline then writes enough evidence to inspect what happened without trusting terminal output.

## Model Policy

The default policy prefers American/open-weight models:

- `openai/gpt-oss-20b` as the primary generator target
- `openai/gpt-oss-120b` as the larger verifier or generator target
- Google and Meta open-weight models as lower-memory fallbacks

The policy is recorded in `run_manifest.json`.

## Verifier-Guided Test-Time Compute

The core experiment compares whether spending more inference at test time improves accuracy:

- single answer uses one sample
- self-consistency uses many samples and majority vote
- verifier reranking uses many samples and selects the highest-scored answer
- adaptive verifier reranking stops early when a candidate clears the confidence threshold

The report compares accuracy and average samples per task.

## What Fixture Runs Prove

The fixture provider proves the research pipeline:

- the same candidate pool is used across strategies
- verifier reranking can recover correct answers from a candidate set
- adaptive sampling can spend fewer samples than fixed reranking
- every claim is backed by files

Fixture runs do not prove performance of `gpt-oss` or any large model.

