# Verifier-Guided Test-Time Compute on Bounded QA

## Abstract
This repo studies whether verifier-guided test-time compute can improve answer quality on bounded benchmark tasks. The agent runs multiple decoding strategies, scores their outputs with a verifier suite, records reproducible artifacts, and writes this report from the recorded metrics.

## Research Question
Can verifier-guided test-time compute improve answer accuracy over a single generation while making the extra sample cost explicit?

## Hypothesis
If the correct answer appears in a sampled candidate set, verifier reranking will outperform a single-answer baseline; adaptive reranking will retain most of the gain while using fewer samples than fixed reranking.

## Model Policy
The preferred generator is `openai/gpt-oss-20b`. Preferred verifier models are `openai/gpt-oss-120b`, `openai/gpt-oss-20b`. The local fixture provider is used for reproducible CI and does not claim to load the large models.

## Method
The experiment compares single-answer generation, self-consistency, fixed verifier reranking, and adaptive verifier reranking. Each strategy is evaluated on the same task set. The artifact contract records the provider, dataset hash, model policy, predictions, verifier scores, and metrics.

## Results
| Strategy | Accuracy | Avg samples/task | Mean verifier score |
| --- | ---: | ---: | ---: |
| single | 0.200 | 1.00 | 0.200 |
| self_consistency | 0.800 | 4.00 | 0.800 |
| verifier_rerank | 1.000 | 4.00 | 1.000 |
| adaptive_verifier_rerank | 1.000 | 1.80 | 1.000 |

Best strategy: `verifier_rerank`.

## Interpretation
Verifier reranking is useful when the correct answer appears somewhere in the candidate set and the verifier can identify it. Adaptive reranking can reduce sample use by stopping when a high-confidence answer appears early.

## Limitations
- Fixture runs prove the research machinery, not frontier-model performance.
- Real `gpt-oss` runs require local GPU memory or a hosted inference provider.
- Answer-key verifiers are appropriate for benchmark research but are not a substitute for human review on open-ended scientific claims.
- The task set is intentionally small in the checked-in example; serious runs should expand the benchmark matrix.

## Reproducibility
- Dataset SHA-256: `18327d9f3d764c8c28b179b2297bc057442a0a24226db56a93b71790774b730f`
- Provider: `fixture`
- Python: `3.9.6`
- Git commit: `b93f0e0`

Run:

```bash
python3 -m gpt_oss_researcher.cli all --provider fixture --output-dir results/latest
```
