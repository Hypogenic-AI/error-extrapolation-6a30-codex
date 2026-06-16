# Resources Catalog

## Summary

This document catalogs all resources gathered for the Error Extrapolation project.

- Papers downloaded: 18
- Datasets downloaded: 5 logical datasets plus Hugging Face cache
- Repositories cloned: 10
- Environment: fresh `.venv` created with `uv`; dependencies recorded in `pyproject.toml`

## Papers

| Title | Year | File | Key Info |
|---|---:|---|---|
| BooookScore | 2024 | `papers/2310.00785_booookscore_book_length_summarization.pdf` | Book-length LLM summary workflows and coherence errors. |
| Stress Testing Factual Consistency Metrics | 2025 | `papers/2511.07689_stress_testing_long_document_factuality_metrics.pdf` | Robustness of factuality metrics under long-doc perturbations. |
| LongDocFACTScore | 2024 | `papers/2309.12455_longdocfactscore_long_document_factuality.pdf` | Long-document factuality metric and LongSciVerify. |
| Long-document factuality case study | 2025 | `papers/2025_findings_acl_long_document_summarization_evaluation_case_study.pdf` | Evaluator behavior across long-doc datasets/model generations. |
| Efficient Attentions / GovReport | 2021 | `papers/2104.02112_govreport_efficient_attentions_long_summarization.pdf` | GovReport dataset and long-doc baselines. |
| QMSum | 2021 | `papers/2104.05938_qmsum_query_based_meeting_summarization.pdf` | Query-based meeting summarization dataset. |
| SQuALITY | 2022 | `papers/2205.11465_squality_long_document_summarization_dataset.pdf` | High-quality question-focused story summaries. |
| SCROLLS | 2022 | `papers/2201.03533_scrolls_long_sequence_benchmark.pdf` | Standardized long-text benchmark. |
| ZeroSCROLLS | 2023 | `papers/2305.14196_zeroscrolls_long_text_benchmark.pdf` | Zero-shot long-text LLM benchmark. |
| LongBench | 2024 | `papers/2308.14508_longbench_long_context_benchmark.pdf` | Long-context benchmark with summarization subsets. |
| LongBench v2 | 2024 | `papers/2412.15204_longbench_v2_long_context_reasoning.pdf` | Harder long-context reasoning benchmark. |
| FRANK | 2021 | `papers/2104.13346_frank_factuality_error_typology.pdf` | Fine-grained factual error taxonomy. |
| AggreFact | 2023 | `papers/2023_acl_aggrefact_understanding_factual_errors.pdf` | Aggregated factuality annotations and metric analysis. |
| SummaC | 2022 | `papers/2111.09525_summac_inconsistency_detection.pdf` | NLI-based factual consistency metric. |
| FactCC | 2020 | `papers/1910.12840_factcc_factual_consistency.pdf` | Weakly supervised factuality baseline. |
| FIB | 2023 | `papers/2023_findings_acl_fib_llm_factual_consistency.pdf` | LLM factual preference benchmark for news summaries. |
| MiniCheck | 2024 | `papers/2404.10774_minicheck_llm_aggrefact.pdf` | Efficient grounded fact checker and LLM-AggreFact. |
| Summarization is Not Dead Yet | 2026 | `papers/2606.08000_summarization_is_not_dead_yet.pdf` | Recent LLM-era summary quality reassessment. |

See `papers/README.md` for detailed descriptions and source links.

## Datasets

| Name | Source | Size/Splits | Task | Location | Notes |
|---|---|---:|---|---|---|
| GovReport | `ccdv/govreport-summarization` | 17,517/973/973 | Long-document summarization | `datasets/govreport_summarization/` | Primary long-report dataset. |
| QMSum Cleaned | `pszemraj/qmsum-cleaned` | 1,257/272/281 | Query-based meeting summarization | `datasets/qmsum_cleaned/` | Includes token count fields. |
| SQuALITY v1.3 | `pszemraj/SQuALITY-v1.3` | 50/25/52 docs | Long story summarization | `datasets/squality_v1_3/` | Small, high-quality, nested questions. |
| FRANK annotations | `mtc/frank-test-set-with-faithfulness-annotation` | 671/1,575 | Factuality labels | `datasets/frank_factuality_annotations/` | Sentence-level factuality annotations. |
| LongBench | `zai-org/LongBench` | 35 JSONL files | Long-context benchmark | `datasets/longbench/` | Includes summarization subsets. |

See `datasets/README.md` for download and loading instructions.

## Code Repositories

| Name | URL | Purpose | Location |
|---|---|---|---|
| BooookScore | https://github.com/lilakk/BooookScore | Book-length summarization and coherence scoring | `code/BooookScore/` |
| LongDocFACTScore | https://github.com/jbshp/LongDocFACTScore | Long-document factuality metric | `code/LongDocFACTScore/` |
| metricEval-longSum | https://github.com/zainmujahid/metricEval-longSum | Long-summary factuality metric stress tests | `code/metricEval-longSum/` |
| MiniCheck | https://github.com/Liyan06/MiniCheck | Efficient grounded fact checker | `code/MiniCheck/` |
| AggreFact | https://github.com/Liyan06/AggreFact | Aggregated factuality benchmark | `code/AggreFact/` |
| SummaC | https://github.com/tingofurro/summac | NLI-based consistency metric | `code/summac/` |
| FactCC | https://github.com/salesforce/factCC | Weakly supervised factuality model | `code/factCC/` |
| FRANK | https://github.com/artidoro/frank | Fine-grained factuality benchmark | `code/frank/` |
| LongBench | https://github.com/THUDM/LongBench | Long-context evaluation code | `code/LongBench/` |
| QMSum | https://github.com/Yale-LILY/QMSum | Official QMSum data and processing | `code/QMSum/` |

See `code/README.md` for repository-specific notes.

## Search Strategy

1. Tried the local paper-finder service first, as required. The diligent query stalled without output and was terminated.
2. Used manual academic search across arXiv, ACL Anthology, Semantic Scholar-style web results, Hugging Face, and GitHub.
3. Prioritized papers with usable datasets, metric code, human error annotations, or direct long-document summarization relevance.
4. Downloaded datasets that are locally loadable without authentication or legacy HF dataset scripts.

## Challenges Encountered

- `uv add` initially failed because the workspace was not an installable Python package. Added `[tool.uv] package = false` to make the project dependency-only.
- Hugging Face `datasets==5.0.0` no longer supports legacy dataset scripts, so `tau/scrolls`, `tau/zero_scrolls`, and the scripted LongBench loader were not loaded via `load_dataset`.
- `lytang/LLM-AggreFact` is gated on Hugging Face without authentication, so it was documented through MiniCheck but not downloaded.
- Some tools need API keys, GPUs, or large checkpoints; no heavy inference was run during resource gathering.

## Recommendations for Experiment Design

1. Primary dataset: use GovReport as the main long-document summarization dataset.
2. Secondary datasets: use QMSum for meeting transcripts, SQuALITY for high-quality small-scale manual checking, and LongBench summarization subsets for compact length-controlled tests.
3. Baselines: compare truncation, direct full-context summarization, hierarchical map-reduce, incremental/refine, and retrieve-then-summarize.
4. Metrics: combine MiniCheck, SummaC-Conv, LongDocFACTScore, FRANK-style error typing, and BooookScore-style coherence categories.
5. Main analysis: fit error-rate trends by source length for each error type, not only aggregate factuality scores.

## Research Execution Outputs

The completed pilot experiment used QMSum and GovReport as natural background text and injected controlled compliance-incident facts as ground truth. Real OpenAI API calls were run for `gpt-4.1-mini` and `gpt-5-mini`.

Key generated outputs:

- `planning.md`: motivation, novelty, and preregistered experiment plan.
- `src/error_experiment.py`: generation, API calling, parsing, and scoring harness.
- `src/analyze_results.py`: statistical analysis and visualization script.
- `src/recovery_check.py`: high-budget recovery check for the two failed GPT-5 high-load cases.
- `results/model_outputs/`: 48 cached main-run API responses.
- `results/evaluations/`: per-run, per-record, slope, and sensitivity result tables.
- `figures/`: generated plots for length trends and slope estimates.
- `REPORT.md`: final report with actual findings.
