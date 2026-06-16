# Error Extrapolation Planning

## Motivation & Novelty Assessment

### Why This Research Matters

Long-context LLMs are increasingly used for tasks where humans cannot practically read the whole input: regulatory reports, meeting archives, litigation discovery, scientific corpora, and operational logs. A single aggregate summary score is not enough for these settings, because a model that is mostly faithful may still fail in predictable, length-dependent ways such as omitting low-salience facts, altering numbers, or inventing plausible records.

### Gap in Existing Work

The gathered literature shows strong benchmarks for long-document summarization and factuality, including BooookScore, LongDocFACTScore, FRANK, AggreFact, SummaC, and MiniCheck. The main gap is that these works usually evaluate overall quality or factuality, while the present question asks how specific error types scale continuously with source length and whether those slopes transfer to regimes beyond normal human reading.

### Our Novel Contribution

This study creates controlled long-document summarization tasks using real long-document corpora as background text and injected audit-style facts as objective ground truth. This design allows programmatic measurement of per-type error rates across length while still using real hosted LLMs for the summarization behavior.

### Experiment Justification

- Experiment 1, fixed fact load: hold the number of target facts constant while increasing irrelevant context length. This isolates context dilution and tests whether hallucinations, substitutions, and omissions grow even when the summary workload is unchanged.
- Experiment 2, scaled fact load: increase target fact count with document length. This tests the more realistic regime where longer documents contain more facts to ingest and summarize, directly addressing whether LLM advantage persists when human inspection is impractical.

## Research Question

As source documents grow from short to long contexts, which LLM summarization error types remain approximately stable and which increase with document length or fact load?

## Background and Motivation

The literature review identifies two complementary taxonomies. FRANK-style factual errors cover entity, predicate/relation, circumstance/date, coreference/discourse, out-of-source, and grammar problems. BooookScore-style long-summary errors add omissions, duplication, salience, discontinuity, and causal gaps. This project focuses on objectively measurable analogues: omissions, hallucinated records, duplicate records, entity errors, numeric errors, date errors, severity/circumstance errors, and relation/action errors.

## Hypothesis Decomposition

- H1: Omission rate increases with source length, especially when the number of target facts increases with length.
- H2: Hallucination and duplication rates are more stable over length than omission rate under deterministic, JSON-constrained prompting.
- H3: Numeric, date, and entity substitution rates grow more slowly than omissions but may increase under scaled fact load.
- H4: Error scaling differs between context dilution and fact-load scaling, so document length alone is not a sufficient explanation.

Independent variables are model, length condition, task family, source length, and target fact count. Dependent variables are per-record or per-output error rates by category. Success means producing real model outputs, objective scores, slope estimates with uncertainty, and documented limitations.

## Proposed Methodology

### Approach

Use downloaded QMSum and GovReport documents as natural distractor context. Insert synthetic but realistic audit incidents with known fields: record id, unit, action, object, count, month, and severity. Prompt real OpenAI models to summarize every audit incident as strict JSON. Parse the outputs and compare against the known facts.

This avoids simulated LLM behavior and avoids relying on fragile local factuality checkpoints. It also creates exact ground truth for error typing, which the literature review identifies as a weakness of long-document factuality work.

### Experimental Steps

1. Load and validate local QMSum/GovReport data; compute approximate token lengths and sample background passages.
2. Generate deterministic injected fact sets with fixed random seed.
3. Build two task families:
   - fixed_load: 10 target facts at 2k, 6k, 12k, and 24k approximate input tokens.
   - scaled_load: 4, 8, 16, and 32 target facts paired with 2k, 6k, 12k, and 24k approximate input tokens.
4. Run real model calls with deterministic decoding where supported. Primary models: `gpt-5-mini` and `gpt-4.1-mini`, using OpenAI's authenticated model list for availability.
5. Parse JSON outputs robustly and score each run by matching predicted records to ground-truth ids and field values.
6. Fit per-error logistic regressions against log source length and fact count; bootstrap confidence intervals for error-rate means and slopes.
7. Generate figures and tables, then write a reproducible report.

### Baselines

- Perfect extractor baseline: zero errors by construction, useful as a sanity reference for metric implementation.
- Prompt/format baseline: JSON parse failure rate and duplicate id rate, which measure whether failures are structural rather than factual.
- Cross-model comparison: `gpt-4.1-mini` versus `gpt-5-mini` to check whether slopes are model-specific.

### Evaluation Metrics

- Omission rate: missing target record ids divided by target records.
- Hallucination rate: predicted record ids not in the source divided by predicted records.
- Duplicate rate: repeated predicted ids divided by predicted records.
- Entity, action, object, count, month, and severity error rates: mismatched fields among matched records.
- Exact record accuracy: all fields correct among target records.
- Parse failure rate: invalid or unparseable JSON outputs.

These metrics map to FRANK/BooookScore-inspired categories while remaining objectively computable.

### Statistical Analysis Plan

Use descriptive means with bootstrap 95% confidence intervals. For hypothesis tests, fit logistic regressions for each error indicator using log2(input tokens), target fact count, model, and task family as predictors. Report slope direction, odds ratios, Wald p-values, and Benjamini-Hochberg adjusted p-values across error types. Because runs are small and generated tasks share templates, conclusions will emphasize effect sizes and confidence intervals over binary significance.

## Expected Outcomes

Results supporting the hypothesis would show positive slopes for omissions and possibly numeric/date errors as length or fact count increases, while hallucinations and duplicates remain flat. Results refuting it would show stable per-type error rates across length once fact count is controlled.

## Timeline and Milestones

- Resource review and planning: complete before implementation.
- Environment and data setup: verify venv, dependencies, GPU, API keys, and dataset schemas.
- Implementation: create scripts for task generation, API execution, scoring, analysis, and plotting.
- Experimentation: run a compact but complete factorial design.
- Analysis and documentation: save raw outputs, figures, tables, REPORT.md, README.md, and reproducibility notes.

## Potential Challenges

- API model parameters may differ across model families; handle unsupported temperature or response-format options with fallback logic.
- JSON parsing may fail despite instructions; use robust extraction and report parse failures instead of silently dropping runs.
- The task may be too easy if facts are too salient; mitigate by scattering facts through real background text and using varied prose, not a single table.
- The compact sample size limits inference; use it as an empirical pilot and report this clearly.

## Success Criteria

The research succeeds if it produces real model outputs, per-error scaling estimates, generated figures/tables, reproducible code, and a report that clearly states which error categories were stable versus length-sensitive in the observed run.
