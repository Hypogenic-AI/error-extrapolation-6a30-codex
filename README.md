# Error Extrapolation

This workspace contains a completed pilot study on how LLM summarization errors scale with source length. It uses QMSum and GovReport text as long natural background context, injects known compliance-incident facts, calls real OpenAI models, and scores the returned JSON against exact ground truth.

## Key Findings

- `gpt-4.1-mini` and `gpt-5-mini` made no hallucinations or duplicate-record errors across the 48 main runs.
- Ordinary field errors were almost flat: only 2 action errors appeared across 536 matched predicted records after normalization.
- The only strong length-sensitive failure was `gpt-5-mini` returning empty visible content in 2 of 3 24k-token / 32-fact scaled-load runs under a 5,000 completion-token cap.
- A 12,000-token recovery check fixed both failed high-load cases with zero omissions or field errors.
- Conclusion: in this controlled setting, error scaling was driven by output/reasoning budget thresholds more than by gradual factual degradation.

See [REPORT.md](REPORT.md) for the full methodology, results, limitations, and references.

## Reproduce

```bash
source .venv/bin/activate
python src/error_experiment.py --models gpt-4.1-mini gpt-5-mini --replicates 3 --all
python src/recovery_check.py --force
python src/analyze_results.py
```

The experiment requires `OPENAI_API_KEY` for real model calls. Cached raw outputs are already saved under `results/model_outputs/`, so deterministic rescoring can be run without new API calls:

```bash
source .venv/bin/activate
python src/error_experiment.py --models gpt-4.1-mini gpt-5-mini --score
python src/analyze_results.py
```

## File Structure

| Path | Contents |
|---|---|
| `planning.md` | Motivation, novelty assessment, hypotheses, and preregistered plan. |
| `REPORT.md` | Full research report with actual results. |
| `src/error_experiment.py` | Task generation, API execution, parsing, and scoring. |
| `src/analyze_results.py` | Statistical analysis and visualization generation. |
| `src/recovery_check.py` | High-budget GPT-5 recovery experiment. |
| `results/evaluations/` | CSV/JSONL metrics, slopes, and sensitivity summaries. |
| `figures/` | Generated plots. |
| `datasets/` | Pre-downloaded datasets used as background context. |
