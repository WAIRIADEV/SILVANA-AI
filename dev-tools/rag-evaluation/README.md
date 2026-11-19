# SILVANA RAG Evaluation Harness

An isolated evaluation toolkit for measuring the retrieval and answer-generation quality of SILVANA's production Retrieval-Augmented Generation (RAG) pipeline.

The evaluation harness is intentionally kept outside the production chat pipeline. It reuses production document extraction, chunking, indexing, embedding, retrieval, and chat-processing services without introducing evaluation logic into production request handling.

## 1. Objectives

The evaluation harness measures:

- Retrieval of relevant document evidence.
- Retrieval ranking quality at different top-K values.
- Answer correctness against annotated expected answers and facts.
- Evidence-grounded abstention when documents do not support an answer.
- Citation presence, validity, and relevance.
- Consistency across supported LLM providers.

The initial benchmark establishes a reproducible baseline before any retrieval or generation tuning is attempted.

## 2. Architecture

The evaluation environment is isolated from production document storage.

```text
dev-tools/rag-evaluation/
|
+-- config.py
+-- index_corpus.py
+-- validate_dataset.py
+-- evaluate_retrieval.py
+-- evaluate_answers.py
+-- aggregate_results.py
|
+-- corpus/
|   +-- documents/
|
+-- datasets/
|   +-- baseline.json
|
+-- data/                  (local, ignored by Git)
|   +-- evaluation.sqlite3
|   +-- documents/
|
+-- results/               (generated reports)
    +-- retrieval_smoke_baseline_v1.1.json
    +-- retrieval_baseline_v1.2.json
    +-- combined_baseline_v1.2.json
```

### Isolation

The evaluation configuration defines its own:

- SQLite database.
- Document storage root.
- Corpus directory.
- Dataset path.
- Results directory.

Before creating evaluation directories, the configuration checks that evaluation database and storage paths do not overlap production document resources.

The evaluation indexing runner creates persistent documents in the isolated repository and uses the existing production indexing service.

Production data is not used as the evaluation corpus.

## 3. Requirements

Run commands from the SILVANA project root with the project virtual environment activated.

Required:

- Python environment with SILVANA backend dependencies installed.
- Access to the production document intelligence modules.
- The configured FastEmbed embedding model, `BAAI/bge-small-en-v1.5`.
- Network access when model files need to be downloaded for the first time.
- A configured Gemini or Groq API provider for answer-generation evaluation.

The answer evaluator uses the existing SILVANA provider configuration. API credentials should be supplied through the project's supported environment configuration.

Do not place API keys in the evaluation dataset, source code, or result reports.

## 4. Evaluation workflow

Run the stages in the following order.

### Step 1: Validate the dataset

```powershell
python .\dev-tools\rag-evaluation\validate_dataset.py
```

Validates the dataset structure, test case identifiers, knowledge scopes, expected answers, evidence annotations, and abstention definitions.

The baseline dataset currently contains:

- 6 test cases.
- 4 answerable questions.
- 2 expected-abstention questions.
- 2 abstention cases with explicitly annotated negative evidence.

### Step 2: Index the evaluation corpus

```powershell
python .\dev-tools\rag-evaluation\index_corpus.py
```

Indexes supported PDF, DOCX, and TXT documents from:

```text
dev-tools/rag-evaluation/corpus/documents/
```

The indexing runner:

- Uses the isolated evaluation repository and storage.
- Reuses production extraction, chunking, embedding, and indexing services.
- Uses document hashes to identify unchanged documents.
- Skips documents that are already indexed with matching content.
- Rejects conflicting content for an existing corpus filename rather than silently replacing it.

The operation is designed to be idempotent.

### Step 3: Evaluate retrieval

```powershell
python .\dev-tools\rag-evaluation\evaluate_retrieval.py
```

Uses the production `DocumentRetrievalService` against the isolated evaluation corpus.

The evaluator measures positive evidence retrieval for answerable questions and negative evidence retrieval for abstention questions.

The generated current report is:

```text
dev-tools/rag-evaluation/results/retrieval_latest.json
```

### Step 4: Evaluate answer generation

Run with Gemini:

```powershell
python .\dev-tools\rag-evaluation\evaluate_answers.py --provider gemini
```

Or Groq:

```powershell
python .\dev-tools\rag-evaluation\evaluate_answers.py --provider groq
```

Optional arguments:

```powershell
--case-id water-001
--limit 2
```

Examples:

Evaluate one case:

```powershell
python .\dev-tools\rag-evaluation\evaluate_answers.py --provider gemini --case-id water-001
```

Evaluate the first two selected cases:

```powershell
python .\dev-tools\rag-evaluation\evaluate_answers.py --provider gemini --limit 2
```

The evaluator uses SILVANA's production chat-processing pipeline and configured provider.

It performs deterministic checks for:

- Expected answer matching.
- Expected fact matching.
- Abstention behavior.
- Citation presence.
- Citation validity against returned sources.
- Citation relevance against annotated expected documents.

It does not use an LLM-as-judge as its sole quality signal.

Answer-generation evaluation makes real provider API calls and may incur API usage or cost.

The evaluator writes the current answer report to:

```text
dev-tools/rag-evaluation/results/answers_latest.json
```

Individual case reports can be retained separately for mixed-provider comparisons and reproducibility.

### Step 5: Aggregate results

```powershell
python .\dev-tools\rag-evaluation\aggregate_results.py
```

Combines:

- The validated baseline dataset.
- The latest retrieval report.
- One individual answer report for each expected test case.

The aggregator validates dataset name, version, case identifiers, duplicate reports, and missing reports.

It supports both legacy `citation_completeness` and current `citation_presence` report fields.

The generated current report is:

```text
dev-tools/rag-evaluation/results/combined_baseline_latest.json
```

The versioned baseline snapshot is maintained separately.

## 5. Metrics

### Retrieval metrics

| Metric | Meaning |
|---|---|
| Recall@K | Whether annotated relevant evidence appears within the first K retrieved chunks |
| Precision@K | Proportion of returned top-K chunks matching annotated relevant evidence |
| MRR@K | Reciprocal rank of the first relevant result |
| Negative evidence coverage | Whether annotated evidence supporting an abstention is retrieved |
| Abstention retrieval pass rate | Proportion of abstention cases where the expected negative evidence is retrieved |

Positive metrics apply to answerable questions. Negative metrics apply to abstention questions.

Precision uses the number of chunks actually returned, up to K, as its denominator. It is not necessarily calculated against a fixed K when fewer than K chunks are returned.

Evidence relevance currently uses annotated document filenames and normalized evidence-text matching. It is not a general semantic relevance judgment.

### Answer-generation metrics

| Metric | Meaning |
|---|---|
| Answer correctness rate | Proportion of answerable cases passing deterministic expected-answer/fact checks |
| Abstention accuracy | Proportion of all cases where the answer-versus-abstain behavior matches the expected behavior |
| Citation validity rate | Proportion of evaluated citations that reference returned sources |
| Citation relevance rate | Proportion of evaluated citations that match expected relevant documents |
| Citation presence rate (answerable) | Proportion of answerable cases containing citations |
| Citation presence rate (abstention) | Proportion of abstention cases containing citations |

A correct abstention does not require a citation. Citation validity and relevance can therefore be `null` when no citation was produced.

Citation presence is tracked separately from citation validity. A missing citation is not the same as an invalid citation.

## 6. Baseline v1.2

The initial consolidated baseline uses six questions over four synthetic evaluation documents.

### Answer-generation results

| Metric | Result |
|---|---:|
| Answerable cases | 4 |
| Abstention cases | 2 |
| Answer correctness | 100% |
| Abstention behavior accuracy | 100% |
| Citation validity | 100% |
| Citation relevance | 100% |
| Citation presence on answerable cases | 100% |
| Citation presence on abstention cases | 50% |

Provider distribution:

| Provider | Cases |
|---|---:|
| Gemini | 5 |
| Groq | 1 |

Models recorded:

- Gemini: `gemini-3.8-flash`
- Groq: `openai/gpt-oss-20b`

### Retrieval results

| Metric | @1 | @3 | @5 |
|---|---:|---:|---:|
| Positive Recall | 100% | 100% | 100% |
| Negative Recall | 100% | 100% | 100% |
| Positive MRR | 100% | 100% | 100% |
| Negative MRR | 100% | 100% | 100% |
| Positive Precision | 100% | 33.33% | 25% |
| Negative Precision | 100% | 33.33% | 25% |

Abstention evidence retrieval pass rate: 100%.

### Interpretation

All six cases retrieved their annotated target evidence at rank one.

All four answerable cases passed deterministic answer checks. Both abstention cases correctly declined to provide unsupported answers.

The lower Precision@3 and Precision@5 values indicate that additional retrieved chunks did not match the single annotated target evidence passage.

These results are encouraging, but the benchmark is deliberately small and synthetic. They should not be interpreted as proof of broad real-world accuracy or production readiness.

The mixed-provider results are not a controlled comparison between Gemini and Groq because they were not evaluated on the same complete set of questions.

## 7. Reproducibility

The following files define the reproducible benchmark:

- `datasets/baseline.json`
- `corpus/documents/`
- Evaluation scripts.
- Versioned retrieval and consolidated result snapshots.

The local `data/` directory contains the evaluation database and indexed document storage. It is intentionally excluded from Git.

Generated `latest` reports and individual answer reports are local execution artifacts. Versioned baseline snapshots are retained as benchmark evidence.

To reproduce the benchmark:

1. Validate the dataset.
2. Index the corpus.
3. Run retrieval evaluation.
4. Run answer evaluation for each required case.
5. Aggregate the resulting reports.
6. Compare the new metrics with the versioned baseline.

Use the same dataset version, corpus, embedding model, and evaluation settings when comparing benchmark runs.

## 8. Limitations and future improvements

Current limitations:

- Small synthetic corpus and six-question benchmark.
- Deterministic answer matching does not fully assess semantic correctness.
- Evidence matching depends on annotated passages.
- No independent semantic entailment judge.
- No broad multi-provider controlled comparison.
- No large-scale latency, throughput, or concurrency benchmark.
- Citation validation checks source references and document relevance, not complete claim-by-claim factual support.

Potential future improvements:

- Expand the corpus with realistic documents and edge cases.
- Add larger datasets covering paraphrases, conflicting documents, and insufficient evidence.
- Introduce claim-level citation evaluation.
- Add optional LLM-as-judge evaluation with deterministic safeguards.
- Compare providers across identical questions.
- Track retrieval latency and end-to-end response latency.
- Add regression thresholds and automated benchmark comparison.

Do not tune production retrieval settings based only on this initial six-question benchmark.

## 9. Safety and data boundaries

The evaluation harness must remain isolated from production document storage and database paths.

Do not:

- Point evaluation storage at production document storage.
- Reuse the production document database.
- Add generated evaluation database files to Git.
- Commit API keys or other secrets.
- Treat synthetic benchmark scores as proof of production-grade performance.

The harness is a development and measurement tool. It does not modify production RAG behavior.

---

**SILVANA — Sprint 12.1: RAG Evaluation Harness**
