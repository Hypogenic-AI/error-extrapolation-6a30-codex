# Literature Review: Error Extrapolation

## Review Scope

### Research Question

How do the types and rates of LLM summarization errors change as source length increases, and which datasets, metrics, and baselines can support controlled experiments over that scaling behavior?

### Inclusion Criteria

- Long-document or long-context summarization benchmarks.
- Factuality, hallucination, consistency, or coherence evaluation for generated summaries.
- Human-annotated error taxonomies or datasets useful for labeling error types.
- Repositories or datasets available for downstream experiments.
- LLM-era studies prioritized, with older factuality metrics included as baselines.

### Exclusion Criteria

- General long-context QA papers without summarization or factuality relevance.
- Pure model-architecture papers without reusable datasets, metrics, or evaluation protocols.
- Paywalled papers without accessible preprint or proceedings PDF.

### Search Log

| Date | Query/Source | Results | Notes |
|---|---|---:|---|
| 2026-06-16 | paper-finder: "large language model long document summarization factual errors length scaling" | 0 usable | Local service stalled and was terminated after no output. |
| 2026-06-16 | arXiv/web: long-document summarization factual consistency LLM benchmarks | 10+ | Found BooookScore, LongDocFACTScore, LongBench, stress-test paper. |
| 2026-06-16 | ACL/arXiv/web: factuality error taxonomy summarization FRANK AggreFact SummaC FactCC | 8+ | Found foundational metrics and error datasets. |
| 2026-06-16 | Hugging Face/GitHub dataset search | 5 downloaded datasets, 10 repos | Downloaded GovReport, QMSum, SQuALITY, FRANK, LongBench. |

## Key Findings

Long-document summarization needs separate treatment from short-news summarization. BooookScore shows that book-length summarization introduces coherence errors tied to chunk-and-combine workflows, especially omissions, discontinuity, duplication, salience, and causal gaps. The paper's most actionable result is that hierarchical merging tends to be more coherent than incremental updating, while incremental updating can preserve more detail. This is directly relevant to error extrapolation because the summarization workflow itself can create length-dependent error modes.

Factuality metrics do not transfer cleanly to long contexts. LongDocFACTScore and the 2025 stress-test paper both show that metrics designed for short inputs struggle with long evidence, truncation, retrieval choices, and information-dense claims. Sentence-level retrieval before scoring is a practical workaround, but metric stability still varies by domain, perturbation, and evidence dispersion.

Error types should be measured separately. FRANK provides factual error categories such as predicate, entity, circumstance, coreference, discourse-link, out-of-article, and grammar errors. BooookScore adds long-summary coherence categories such as entity omission, event omission, causal omission, discontinuity, salience, language, inconsistency, and duplication. These taxonomies are complementary: FRANK is source-grounded factuality; BooookScore is summary-internal coherence at book scale.

Benchmarks must match the target model era. AggreFact shows that apparent metric improvements are often concentrated on summaries from older systems rather than modern summarizers. Experiments for this project should generate fresh summaries from current LLMs, then evaluate by error type instead of relying only on old benchmark outputs.

## Key Papers

### BooookScore

- Contribution: First systematic study of LLM book-length summarization with hierarchical and incremental workflows.
- Data/method: 100 recently published books, GPT-4 summaries, 1,193 fine-grained human annotations.
- Error taxonomy: entity omission, event omission, causal omission, discontinuity, salience, language, inconsistency, duplication.
- Results: Hierarchical merging usually had higher BooookScore; incremental updating preserved more detail but made more coherence errors.
- Relevance: Best direct template for length-scaling experiments and error-category measurement.

### LongDocFACTScore

- Contribution: Reference-free long-document factuality framework that retrieves evidence snippets per summary sentence before applying BARTScore.
- Data/method: LongSciVerify with PubMed and ArXiv long documents, expert factuality annotations, plus LongEval PubMed.
- Results: LongDocFACTScore correlated better with human factuality judgments than ROUGE, BERTScore, FactCC, QuestEval, and vanilla BARTScore on the reported long-document datasets.
- Relevance: Provides a local long-document factuality baseline and data.

### Stress Testing Factual Consistency Metrics for Long-Document Summarization

- Contribution: Tests six reference-free metrics under seven meaning-preserving perturbations.
- Metrics: BARTScore, SummaC-Conv, SummaC-ZS, AlignScore, MiniCheck, UniEval.
- Datasets: SQuALITY, LexAbSumm, ScholarQABench.
- Findings: Many metrics change scores under paraphrase/simplification/negation despite preserved meaning. MiniCheck and UniEval are comparatively robust but still imperfect.
- Relevance: Provides perturbation ideas for measuring metric sensitivity as length and claim density increase.

### FRANK

- Contribution: Fine-grained factuality benchmark over CNN/DailyMail and XSum summaries.
- Data: 2,250 annotated summaries from 9 summarization systems.
- Error taxonomy: predicate, entity, circumstance, coreference, discourse-link, out-of-article, grammar.
- Relevance: Good training/validation source for factual error type classifiers.

### AggreFact

- Contribution: Aggregates nine factuality annotation datasets and stratifies by summarizer generation.
- Finding: No single factuality metric is uniformly best; metric performance varies by dataset, model era, and error type.
- Relevance: Warns against drawing conclusions from a single factuality metric or older summarizer outputs.

### SummaC, FactCC, MiniCheck, and FIB

- SummaC: Strong practical NLI-based consistency metric using sentence-level document-summary alignment.
- FactCC: Historical weakly supervised factual consistency baseline with generated perturbations.
- MiniCheck: Efficient sentence-level grounded fact checker with strong LLM-era performance, but model weights may need download/GPU.
- FIB: LLM factuality preference benchmark over news summaries; useful conceptually but less aligned with long documents.

## Datasets in the Literature

- GovReport: 19k government reports with long expert summaries; good primary long-document summarization dataset.
- QMSum: 1,808 query-summary pairs over 232 meetings; useful for testing long transcript summarization.
- SQuALITY: Long stories with question-focused summaries; high quality and small enough for careful evaluation.
- LongBench: Compact long-context benchmark with summarization subsets, useful for length-controlled evaluation.
- FRANK/AggreFact/LongSciVerify: Factuality/error annotations for calibrating metrics and classifiers.

## Standard Baselines

- Summarization workflows: direct full-context summarization, truncation, hierarchical map-reduce, incremental/refine, retrieve-then-summarize.
- Factuality metrics: MiniCheck, SummaC-Conv, LongDocFACTScore, FactCC, BARTScore, QuestEval, ROUGE/BERTScore as non-factuality controls.
- Error taxonomies: FRANK for factual grounding errors; BooookScore for long-summary coherence errors.

## Evaluation Metrics

- Error rate per summary sentence, grouped by error type.
- Factual support score per sentence using MiniCheck or LongDocFACTScore.
- Coherence score using BooookScore-style annotations.
- ROUGE/BERTScore only as auxiliary reference-overlap metrics, not as factuality proxies.
- Stratified analysis by source length bins, summary length, input domain, workflow, and model.

## Gaps and Opportunities

- There is little direct evidence on how each error type scales continuously with document length.
- Long-document factuality metrics are sensitive to retrieval and semantic perturbations.
- Most available error datasets are short-news or older-model outputs, so fresh LLM-generated summaries should be labeled or automatically screened.
- Book-length datasets are often inaccessible due to copyright, so GovReport, QMSum, SQuALITY, and LongBench are more reproducible starting points.

## Recommendations for Experiment Design

- Primary datasets: GovReport for long reports, QMSum for transcripts, SQuALITY for smaller high-quality long stories, LongBench for compact length-controlled subsets.
- Length bins: use token counts from QMSum directly; compute report/story token counts for GovReport and SQuALITY; include LongBench `_e` subsets where available.
- Generation workflows: compare direct full-context summaries where the model supports it, hierarchical map-reduce, incremental/refine, and truncation.
- Metrics: use at least two factuality metrics, preferably MiniCheck and LongDocFACTScore/SummaC, plus a BooookScore-inspired coherence classifier or LLM judge.
- Analysis: report total error rate and per-type slopes against source length. Treat metric disagreement as a result, not just noise.
