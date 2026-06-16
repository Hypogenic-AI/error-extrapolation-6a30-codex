# Cloned Repositories

This directory contains cloned code and data repositories relevant to long-document summarization and factuality evaluation.

| Name | URL | Purpose | Location | Notes |
|---|---|---|---|---|
| BooookScore | https://github.com/lilakk/BooookScore | Book-length summarization workflows and coherence-error scoring | `code/BooookScore/` | Includes summaries and model/human annotations. Strongest scoring mode requires LLM API access. |
| LongDocFACTScore | https://github.com/jbshp/LongDocFACTScore | Long-document factuality scoring with sentence retrieval | `code/LongDocFACTScore/` | Includes LongSciVerify and LongEval data under `data/`; full replication may need model downloads and FactCC checkpoint. |
| metricEval-longSum | https://github.com/zainmujahid/metricEval-longSum | Stress-test data and scripts for long-summary factuality metrics | `code/metricEval-longSum/` | Includes perturbed SQuALITY, LexAbSumm, and ScholarQABench CSVs. |
| MiniCheck | https://github.com/Liyan06/MiniCheck | Efficient sentence-level grounded fact checker | `code/MiniCheck/` | Includes package code and examples; model weights download from Hugging Face on first use. |
| AggreFact | https://github.com/Liyan06/AggreFact | Aggregated factuality benchmark and metric results | `code/AggreFact/` | Includes `aggre_fact_final.csv`, SOTA subset, and unified error-type mappings. |
| SummaC | https://github.com/tingofurro/summac | NLI-based summary consistency metric | `code/summac/` | Installable as `summac`; benchmark loaders download data on first run. |
| FactCC | https://github.com/salesforce/factCC | Weakly supervised factual consistency model | `code/factCC/` | Requires checkpoint download and original CNN/DailyMail pairing for full reproduction. |
| FRANK | https://github.com/artidoro/frank | Fine-grained factual error benchmark and evaluation scripts | `code/frank/` | Includes benchmark JSONs, sentence annotations, and baseline metric outputs. |
| LongBench | https://github.com/THUDM/LongBench | Long-context benchmark code and prompts | `code/LongBench/` | Local dataset zip is also saved under `datasets/longbench/`. v2 evaluation expects model serving setup. |
| QMSum | https://github.com/Yale-LILY/QMSum | Official QMSum data and processing notes | `code/QMSum/` | Includes original JSON/JSONL data, extracted spans, and model outputs. |

## Practical Entry Points

- BooookScore:
  - `python -m booookscore.chunk`
  - `python -m booookscore.summ`
  - `python -m booookscore.score`
- LongDocFACTScore:
  - `from longdocfactscore.ldfacts import LongDocFACTScore`
  - `python run_example.py`
- MiniCheck:
  - `from minicheck.minicheck import MiniCheck`
  - Use `flan-t5-large` for a practical sub-1B baseline; use `Bespoke-MiniCheck-7B` only with sufficient GPU.
- SummaC:
  - `from summac.model_summac import SummaCZS, SummaCConv`
- FRANK:
  - `python evaluation/evaluate.py --split test`
- LongBench:
  - v1 code under `code/LongBench/LongBench/`
  - v2 top-level scripts expect OpenAI-compatible model serving.

## Validation Notes

- Repositories cloned successfully with `--depth 1`.
- READMEs and dependency files were inspected.
- No heavy model inference was run because several tools require API keys, GPU resources, or large checkpoint downloads.
