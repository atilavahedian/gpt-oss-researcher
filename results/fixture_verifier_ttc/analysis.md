# Analysis

The best observed strategy for `Verifier-Guided Test-Time Compute on Bounded QA` was `verifier_rerank`.

| Strategy | Accuracy | Avg samples/task | Mean verifier score |
| --- | ---: | ---: | ---: |
| single | 0.200 | 1.00 | 0.200 |
| self_consistency | 0.800 | 4.00 | 0.800 |
| verifier_rerank | 1.000 | 4.00 | 1.000 |
| adaptive_verifier_rerank | 1.000 | 1.80 | 1.000 |

The analysis treats sample count as a first-order cost. A strategy that improves accuracy but spends many more samples is not automatically better; the report compares both.
