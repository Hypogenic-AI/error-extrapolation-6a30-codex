# Paper Outline

## Title
Budget-Limited Failure, Not Factual Drift, Drives Error Scaling in Controlled Long-Context Summarization

## Abstract
- State the problem: long-context summarization is used where humans cannot inspect all input.
- Identify the gap: prior work measures factuality and coherence, but not per-error slopes under controlled ground truth.
- Describe the approach: inject known compliance incidents into QMSum and GovReport backgrounds and run two OpenAI models across length and fact-load bins.
- Preview the result: valid outputs have nearly no factual errors; the only strong length effect is two empty GPT-5-mini visible outputs under a 5,000-token cap.
- State significance: long-context risk can appear as abrupt budget failure rather than gradual factual drift.

## Introduction
- Hook: operational summarization requires knowing which errors grow with length.
- Gap: aggregate factuality scores hide mechanisms, especially omissions versus hallucinations and substitutions.
- Approach: controlled injected facts in natural long-document backgrounds.
- Quantitative preview: 48 runs, 600 target records, 536 predicted records, 0 hallucinations/duplicates, 2 action-field errors, 2 parse failures.
- Contributions: benchmark design, per-error analysis, statistical sensitivity, practical budget implication.

## Related Work
- Long-document summarization and evaluation: QMSum, GovReport, BooookScore, LongDocFACTScore.
- Factuality metrics and taxonomies: FRANK, FactCC, SummaC, AggreFact, MiniCheck.
- Stress testing and controlled scaling: position this paper as complementary because ground truth is injected and slopes are measured by error type.

## Methodology
- Define target records and predicted records.
- Describe datasets, task families, model calls, and scoring.
- Include experimental design table.
- Define omissions, hallucinations, duplicates, field errors, exact record rate, and parse failure.
- Describe baselines and statistical tests.

## Results
- Main-run summary table.
- Error-rate figure.
- Hardest 24k condition table.
- Recovery table.
- Statistical slope table and heatmap.
- Explain that the infinite odds ratios reflect separation at the largest scaled-load bin.

## Discussion
- Interpret the main result: valid outputs remain factually stable.
- Explain the visible-output budget failure and why it matters.
- State limitations: synthetic facts, small sample, constrained JSON, two models, 24k maximum source length.
- Discuss broader implications for long-context systems.

## Conclusion
- Summarize the controlled experiment and result.
- Emphasize budget-aware evaluation as the main next step.
