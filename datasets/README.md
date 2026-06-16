# Downloaded Datasets

This directory contains datasets for the Error Extrapolation project. Full data files are intentionally excluded from Git by `datasets/.gitignore`; small samples are retained under each dataset's `samples/` folder.

## Dataset 1: GovReport Summarization

### Overview
- Source: Hugging Face `ccdv/govreport-summarization`, config `document`
- Location: `datasets/govreport_summarization/`
- Task: long-document summarization of U.S. government reports
- Splits: train 17,517; validation 973; test 973
- Fields: `report`, `summary`
- Why useful: primary long-document dataset for measuring how error rates change as source length increases.

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("ccdv/govreport-summarization", "document")
dataset.save_to_disk("datasets/govreport_summarization")
```

### Loading

```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/govreport_summarization")
```

### Sample Data

See `datasets/govreport_summarization/samples/sample_records.json`.

## Dataset 2: QMSum Cleaned

### Overview
- Source: Hugging Face `pszemraj/qmsum-cleaned`, config `default`
- Location: `datasets/qmsum_cleaned/`
- Task: query-based meeting summarization
- Splits: train 1,257; validation 272; test 281
- Fields: `id`, `pid`, `input`, `output`, `input_token_count`, `output_token_count`
- Why useful: long meeting transcripts with query-focused summaries and token-count fields for length-binned experiments.

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("pszemraj/qmsum-cleaned", "default")
dataset.save_to_disk("datasets/qmsum_cleaned")
```

### Loading

```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/qmsum_cleaned")
```

### Sample Data

See `datasets/qmsum_cleaned/samples/sample_records.json`.

## Dataset 3: SQuALITY v1.3

### Overview
- Source: Hugging Face `pszemraj/SQuALITY-v1.3`, config `default`
- Location: `datasets/squality_v1_3/`
- Task: long-input, question-focused summarization over stories
- Splits: train 50; validation 25; test 52 documents
- Fields: `metadata`, `document`, `questions`
- Why useful: high-quality multi-reference long-document summaries for controlled small-scale experiments.

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("pszemraj/SQuALITY-v1.3", "default")
dataset.save_to_disk("datasets/squality_v1_3")
```

### Loading

```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/squality_v1_3")
```

### Sample Data

See `datasets/squality_v1_3/samples/sample_records.json`.

## Dataset 4: FRANK Factuality Annotations

### Overview
- Source: Hugging Face `mtc/frank-test-set-with-faithfulness-annotation`, config `default`
- Location: `datasets/frank_factuality_annotations/`
- Task: factuality/error annotation for generated summaries
- Splits: validation 671; test 1,575
- Fields include: `article`, `summary`, `reference`, `summary_sentences`, `summary_sentences_annotations`, `Factual`
- Why useful: fine-grained human factuality labels for training or validating error classifiers.

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("mtc/frank-test-set-with-faithfulness-annotation", "default")
dataset.save_to_disk("datasets/frank_factuality_annotations")
```

### Loading

```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/frank_factuality_annotations")
```

### Sample Data

See `datasets/frank_factuality_annotations/samples/sample_records.json`.

## Dataset 5: LongBench

### Overview
- Source: Hugging Face `zai-org/LongBench`, file `data.zip`
- Location: `datasets/longbench/`
- Task: long-context benchmark with summarization, QA, retrieval, code, and synthetic tasks
- Local summarization-related files:
  - `gov_report.jsonl`: 200 examples
  - `gov_report_e.jsonl`: 300 examples
  - `multi_news.jsonl`: 200 examples
  - `multi_news_e.jsonl`: 294 examples
  - `qmsum.jsonl`: 200 examples
  - `vcsum.jsonl`: 200 examples
- Why useful: compact length-controlled benchmark cases for evaluating degradation over longer contexts.

### Download Instructions

```python
from huggingface_hub import hf_hub_download
import zipfile

zip_path = hf_hub_download(
    repo_id="zai-org/LongBench",
    filename="data.zip",
    repo_type="dataset",
    local_dir="datasets/longbench/raw",
)
with zipfile.ZipFile(zip_path) as zf:
    zf.extractall("datasets/longbench/extracted")
```

### Loading

```python
import json
from pathlib import Path

path = Path("datasets/longbench/extracted/data/gov_report.jsonl")
records = [json.loads(line) for line in path.open()]
```

### Sample Data

See `datasets/longbench/samples/sample_records.json`.

## Notes

- `datasets/.hf_home/` contains Hugging Face cache files from the local download process and is ignored by Git.
- Hugging Face dataset repos that require legacy dataset scripts, such as `tau/scrolls`, were not loaded through `datasets==5.0.0`; LongBench and direct loadable alternatives were used instead.
- `lytang/LLM-AggreFact` is gated on Hugging Face without authentication in this environment. The MiniCheck repo was cloned and documented, but that gated dataset was not downloaded.
