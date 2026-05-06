# Research Plan: Verifier-Guided Test-Time Compute on Bounded QA

## Research Question
Can verifier-guided test-time compute improve answer accuracy over a single generation while making the extra sample cost explicit?

## Hypothesis
If the correct answer appears in a sampled candidate set, verifier reranking will outperform a single-answer baseline; adaptive reranking will retain most of the gain while using fewer samples than fixed reranking.

## Experimental Design
- Benchmark tasks: 5
- Candidate samples per fixed-compute strategy: 4
- Adaptive stopping threshold: 0.99
- Evaluation contract: every run must write a manifest, predictions JSONL, metrics JSON, analysis note, and research report.

## Strategies
- `single`
- `self_consistency`
- `verifier_rerank`
- `adaptive_verifier_rerank`

## Task Set
- `arithmetic_13`: What is 8 + 5?
- `capital_california`: What is the capital of California?
- `space_needle`: Which U.S. city is the Space Needle in?
- `prime_boolean`: Is 29 a prime number? Answer true or false.
- `declaration_year`: In what year was the U.S. Declaration of Independence adopted?

## Success Criteria
Verifier-guided test-time compute should beat the single-answer baseline while reporting the extra sample cost and the failure modes.
