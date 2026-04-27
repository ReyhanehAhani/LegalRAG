# LegalBench-RAG Evaluation

Retrieval evaluation pipeline for [zeroentropy-ai/legalbenchrag](https://github.com/zeroentropy-ai/legalbenchrag) — a character-level IR benchmark covering legal contracts (CUAD, ContractNLI, MAUD, PrivacyQA).

Unlike CLERC (which uses NDCG/MRR over ranked document IDs), LegalBench-RAG measures retrieval quality at the **exact character level**:


| Metric    | Formula |
| --------- | ------- |
| Recall    | `       |
| Precision | `       |


Retrieved chunk spans are merged per file before computing the intersection with ground-truth spans, following the methodology in the [original paper](https://arxiv.org/abs/2408.10343).

---

## Data Download

Download the benchmark data from the [official Dropbox link](https://www.dropbox.com/scl/fo/r7xfa5i3hdsbxex1w6amw/AID389Olvtm-ZLTKAPrw6k4?rlkey=5n8zrbk4c08lbit3iiexofmwg&st=0hu354cq&dl=0).

Extract to:

```
data/LegalBenchRAG/
    corpus/
        contractnli/   *.txt
        cuad/          *.txt
        maud/          *.txt
        privacy_qa/    *.txt
    benchmarks/
        contractnli.json
        cuad.json
        maud.json
        privacy_qa.json
```

---

## Prerequisites

### Ensure OpenSearch is running

```bash
curl http://localhost:9200/_cluster/health
```

```bash
# check index
curl -XGET "http://localhost:9200/_cat/indices?v&s=index"
```

If not running:

```bash
~/opensearch-3.4.0/bin/opensearch &          # start in background
```

---

## Quickstart

### Step 1 — Ingest corpus (default settings)

By default, only the corpus files referenced by benchmark test cases are ingested
(much smaller than the full corpus). This is the recommended first run.

```bash
python -m evaluation.LegalBenchRAG.ingest \
    --data-dir data/LegalBenchRAG
```

For a fast smoke test (10 test cases per sub-benchmark → minimal corpus):

```bash
python -m evaluation.LegalBenchRAG.ingest \
    --data-dir data/LegalBenchRAG \
    --limit 10
```

To ingest from a custom corpus subset (e.g. `corpus_50`):

```bash
python -m evaluation.LegalBenchRAG.ingest \
    --data-dir data/LegalBenchRAG \
    --corpus-dir data/LegalBenchRAG/corpus_50 \
    --all
```

### Step 2 — Evaluate

```bash
python -m evaluation.LegalBenchRAG.eval_precision_recall \
    --data-dir data/LegalBenchRAG \
    --ks 20 40 60
```

---

## Chunker Strategies

Two chunkers are available. Pass `--chunker` to `ingest` to select one.

### `hierarchical` (default)

Two-level structure: large **parent** chunks (~~1500 chars) contain overlapping
**child** chunks (~~512 chars). Only child chunks are embedded and retrieved;
the parent is fetched at generation time for richer context
(*small-to-big retrieval*).

- Better for generative RAG (wider context per retrieved chunk)
- `--parent-size` controls the parent window (default: 1500 chars)
- `--chunk-size` controls the child window (default: 512 chars from config)
- `--chunk-overlap` controls child overlap (default: 64 chars from config)

```bash
# Ingest with hierarchical chunker into a named index
python -m evaluation.LegalBenchRAG.ingest \
    --data-dir data/LegalBenchRAG \
    --chunker hierarchical \
    --parent-size 2048 \
    --chunk-size 512 \
    --chunk-overlap 64 \
    --index-name lbr-hier-all-mpnet-base-v2 \
    --all

# Evaluate against that index
python -m evaluation.LegalBenchRAG.eval_precision_recall \
    --data-dir data/LegalBenchRAG \
    --index-name lbr-hier-all-mpnet-base-v2 \
    --ks 2 4 6 10 15 20 40 60 \
    --trace-file logs/eval/lbr_hier_all-mpnet-base-v2.jsonl \
    --benchmarks-dir data/LegalBenchRAG/benchmarks_50
```

### `recursive`

Flat chunker that tries to split on progressively finer separators
(`\n\n` → `\n` → `.`  →  `` → character). No parent-child hierarchy —
every chunk is both ingested and retrieved directly.

- Better recall at low K (retrieved chunk boundaries align with paragraph structure)
- No `--parent-size` (ignored)
- `--chunk-size` and `--chunk-overlap` still apply

```bash
# Ingest with recursive chunker into a separate named index
python -m evaluation.LegalBenchRAG.ingest \
    --data-dir data/LegalBenchRAG \
    --chunker recursive \
    --chunk-size 512 \
    --chunk-overlap 64 \
    --index-name legalrag-lbr-recursive \
    --all


# Evaluate against that index
python -m evaluation.LegalBenchRAG.eval_precision_recall \
    --data-dir data/LegalBenchRAG \
    --index-name legalrag-lbr-recursive \
    --ks 2 4 6 10 15 20 40 60
```

## Evaluate a single sub-benchmark

```bash
# Ingest only CUAD corpus files
python -m evaluation.LegalBenchRAG.ingest \
    --data-dir data/LegalBenchRAG \
    --benchmarks cuad

# Evaluate CUAD only at K=20,40,60
python -m evaluation.LegalBenchRAG.eval_precision_recall \
    --data-dir data/LegalBenchRAG \
    --benchmarks cuad \
    --ks 20 40 60
```

---

## Embedding Providers

The `EMBEDDING_PROVIDER` in `.env` controls which embedder class is used. The `--embedding-model` flag (on both `ingest` and `eval_precision_recall`) overrides only the model name — the provider is always taken from `.env`.

| `EMBEDDING_PROVIDER`    | Class                        | Use when                                                                 |
| ----------------------- | ---------------------------- | ------------------------------------------------------------------------ |
| `sentence_transformers` | `SentenceTransformerEmbedder`| Model has a `sentence_transformers` config on HF (e.g. `all-mpnet-base-v2`, `legal-bert-base-uncased`, `Legal-Embed-bge-base-en-v1.5`) |
| `huggingface`           | `HuggingFaceEmbedder`        | Raw `AutoModel` + `AutoTokenizer` with mean pooling — for models not packaged as sentence-transformers (e.g. `jhu-clsp/BERT-DPR-CLERC-ft`) |
| `openai`                | `OpenAIEmbedder`             | OpenAI API or compatible endpoint                                        |

`HuggingFaceEmbedder` mean-pools the last hidden state over non-padding tokens and L2-normalises the result. It is GPU-aware (uses CUDA if available).

Example — ingest and evaluate with `jhu-clsp/BERT-DPR-CLERC-ft`:

```bash
# .env
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL=jhu-clsp/BERT-DPR-CLERC-ft
EMBEDDING_DIM=768

# Ingest
python -m evaluation.LegalBenchRAG.ingest \
    --data-dir data/LegalBenchRAG \
    --index-name lbr-hier-clerc \
    --all

# Evaluate
python -m evaluation.LegalBenchRAG.eval_precision_recall \
    --data-dir data/LegalBenchRAG \
    --index-name lbr-hier-clerc \
    --ks 20 40 60
```

Or override both on the command line without touching `.env`:

```bash
# Ingest
python -m evaluation.LegalBenchRAG.ingest \
    --data-dir data/LegalBenchRAG \
    --index-name lbr-hier-clerc \
    --embedding-provider huggingface \
    --embedding-model jhu-clsp/BERT-DPR-CLERC-ft \
    --all

# Evaluate
python -m evaluation.LegalBenchRAG.eval_precision_recall \
    --data-dir data/LegalBenchRAG \
    --index-name lbr-hier-clerc \
    --embedding-provider huggingface \
    --embedding-model jhu-clsp/BERT-DPR-CLERC-ft \
    --ks 20 40 60
```

---

## Re-ingestion

Delete the index and re-run ingestion when you change the embedding model or chunk size:

```bash
curl -X DELETE http://localhost:9200/legalrag-legalbenchrag
python -m evaluation.LegalBenchRAG.ingest --data-dir data/LegalBenchRAG
```

> **Note**: if you change `--embedding-model` to a model with a different output
> dimension, you *must* delete and recreate the index — the embedding dimension is
> fixed at index creation time.

---

## Options Reference

### Ingest options (`evaluation.LegalBenchRAG.ingest`)


| Flag                      | Default                  | Description                                                            |
| ------------------------- | ------------------------ | ---------------------------------------------------------------------- |
| `--data-dir PATH`         | required                 | Root of downloaded data dir (must contain `corpus/` and `benchmarks/`) |
| `--benchmarks NAME …`     | all four                 | Restrict which sub-benchmarks determine which corpus files to ingest   |
| `--limit N`               | None                     | Cap test cases per benchmark when selecting corpus files               |
| `--all`                   | false                    | Ingest every `*.txt` under `corpus/` (ignores benchmark filter)        |
| `--corpus-dir PATH`       | `<data-dir>/corpus`      | Override the corpus directory (e.g. `data/LegalBenchRAG/corpus_50`)    |
| `--chunker`               | `hierarchical`           | Chunking strategy: `hierarchical` or `recursive`                       |
| `--chunk-size N`          | 512 (config)             | Child chunk size in characters                                         |
| `--chunk-overlap N`       | 64 (config)              | Character overlap between consecutive chunks                           |
| `--parent-size N`         | 1500                     | Parent chunk size — hierarchical only                                  |
| `--embedding-provider PROVIDER` | `EMBEDDING_PROVIDER` in `.env` | Embedding provider: `sentence_transformers`, `huggingface`, or `openai` |
| `--embedding-model MODEL` | `EMBEDDING_MODEL` in `.env` | Override the embedding model name                                      |
| `--index-name NAME`       | `legalrag-legalbenchrag` | OpenSearch index to ingest into                                        |
| `--log-level`             | INFO                     | Verbosity                                                              |


### Evaluation options (`evaluation.LegalBenchRAG.eval_precision_recall`)


| Flag                      | Default                     | Description                                               |
| ------------------------- | --------------------------- | --------------------------------------------------------- |
| `--data-dir PATH`         | required                    | Root of downloaded data dir                               |
| `--benchmarks-dir PATH`   | `<data-dir>/benchmarks`     | Override benchmarks directory (e.g. for a sampled subset) |
| `--benchmarks NAME …`     | all four                    | Sub-benchmarks to evaluate                                |
| `--limit N`               | None                        | Cap test cases per benchmark (for fast iteration)         |
| `--ks K …`                | `1 5 10 20`                 | Rank cutoffs; retrieves `max(ks)` chunks per query        |
| `--index-name NAME`       | `legalrag-legalbenchrag`    | OpenSearch index to query — must match ingestion          |
| `--embedding-provider PROVIDER` | `EMBEDDING_PROVIDER` in `.env` | Embedding provider: `sentence_transformers`, `huggingface`, or `openai`. Must match ingest. |
| `--embedding-model MODEL` | `EMBEDDING_MODEL` in `.env` | Override the query embedding model. Must match ingest.                 |
| `--trace-file PATH`       | None                        | Write per-query JSONL retrieval trace (see below)         |
| `--log-level`             | WARNING                     | Verbosity                                                 |


---

## Retrieval trace file

Pass `--trace-file logs/eval_trace.jsonl` to write one JSON line per query. Useful for debugging low-scoring queries.

```bash
python -m evaluation.LegalBenchRAG.eval_precision_recall \
    --data-dir data/LegalBenchRAG \
    --benchmarks-dir data/LegalBenchRAG/benchmarks_subset \
    --index-name legalrag-lb-hierarchical \
    --ks 20 40 60 \
    --trace-file logs/eval_trace.jsonl
```

### File format

The output is **JSONL** (one JSON object per line). There are two record types, distinguished by the presence of `"_type"`:

| Record type | Condition | Description |
| ----------- | --------- | ----------- |
| *(no `_type`)* | one per query | Per-query retrieval trace |
| `"run_summary"` | last line | Aggregate metrics for the whole run |

---

### Per-query record

```json
{
  "query_idx": 1,
  "query": "Does the agreement include a limitation of liability clause?",
  "tags": ["cuad"],
  "ground_truth": [
    {"file": "cuad/contract_001.txt", "span": [12400, 12850]}
  ],
  "total_gt_chars": 450,
  "n_retrieved": 60,
  "retrieved_all": [ ... ],
  "metrics_by_k": [ ... ]
}
```

#### Top-level fields

| Field | Type | Description |
| ----- | ---- | ----------- |
| `query_idx` | int | 0-based position of this test case across all benchmarks in the run |
| `query` | string | Raw query string from the benchmark JSON |
| `tags` | string[] | Sub-benchmark name(s) this query belongs to (e.g. `["cuad"]`) |
| `ground_truth` | object[] | GT snippets — each has `file` (relative corpus path) and `span` ([char_start, char_end]) |
| `total_gt_chars` | int | Total unique GT characters after merging overlapping GT spans per file |
| `n_retrieved` | int | Number of chunks actually returned by the retriever (≤ max(ks)) |
| `retrieved_all` | object[] | All retrieved chunks in rank order (see below) |
| `metrics_by_k` | object[] | Per-K metric breakdown (see below) |

#### `retrieved_all` — one entry per retrieved chunk

| Field | Type | Description |
| ----- | ---- | ----------- |
| `rank` | int | Retrieval rank (1 = highest scored) |
| `file` | string | Relative corpus path (e.g. `cuad/contract_001.txt`) |
| `char_start` | int | Character offset of chunk start in the original document |
| `char_end` | int | Character offset of chunk end (exclusive) |
| `char_len` | int | `char_end - char_start` |
| `score` | float | Semantic (kNN) score returned by OpenSearch |
| `chunk_id` | string | MD5 chunk identifier used in the index |

#### `metrics_by_k` — one entry per K cutoff

| Field | Type | Description |
| ----- | ---- | ----------- |
| `k` | int | Rank cutoff |
| `char_recall` | float | Character-level recall: `intersection_chars / gt_chars` |
| `char_precision` | float | Character-level precision: `intersection_chars / retrieved_chars` |
| `intersection_chars` | int | Characters in the intersection of GT spans and top-K retrieved spans (after merging per file) |
| `gt_chars` | int | Same as `total_gt_chars` — total unique GT characters |
| `retrieved_chars` | int | Total unique characters covered by top-K chunks (after merging per file) |
| `chunk_recall` | float | Chunk-level recall: `# GT snippets hit by ≥1 top-K chunk / # GT snippets` |
| `chunk_precision` | float | Chunk-level precision: `# top-K chunks overlapping ≥1 GT snippet / k` |
| `n_gt_snippets` | int | Number of GT snippets for this query |
| `n_gt_hit` | int | Number of GT snippets hit by at least one top-K chunk |
| `n_chunk_hits` | int | Number of top-K chunks that overlap at least one GT snippet |
| `chunk_hits` | object[] | Per-chunk hit detail (see below) |
| `gt_snippet_hits` | object[] | Per-GT-snippet hit detail (see below) |

#### `chunk_hits` — one entry per top-K chunk

Extends all fields from `retrieved_all` plus:

| Field | Type | Description |
| ----- | ---- | ----------- |
| `is_chunk_hit` | bool | True if this chunk overlaps at least one GT snippet |
| `gt_overlaps` | object[] | One entry per GT snippet that this chunk overlaps (empty if no hit) |
| `gt_overlaps[].gt_idx` | int | 0-based index into `ground_truth` |
| `gt_overlaps[].overlap_span` | [int, int] | `[overlap_start, overlap_end]` character offsets |
| `gt_overlaps[].overlap_chars` | int | Number of overlapping characters |
| `gt_overlaps[].overlap_pct_of_chunk` | float | `overlap_chars / chunk_char_len × 100` |
| `gt_overlaps[].overlap_pct_of_gt` | float | `overlap_chars / gt_snippet_char_len × 100` |

#### `gt_snippet_hits` — one entry per GT snippet

| Field | Type | Description |
| ----- | ---- | ----------- |
| `gt_idx` | int | 0-based index into `ground_truth` |
| `file` | string | Relative corpus path of the GT snippet |
| `span` | [int, int] | `[char_start, char_end]` of the GT snippet |
| `is_hit` | bool | True if at least one top-K chunk overlaps this snippet |
| `hit_by_chunk_ranks` | int[] | Ranks of the chunks that hit this snippet (empty if not hit) |

---

### `run_summary` record (last line)

The final line has `"_type": "run_summary"` and contains aggregate metrics for the entire run.

```json
{
  "_type": "run_summary",
  "run_info": {
    "label": "all-mpnet-base-v2",
    "embedding_model": "sentence-transformers/all-mpnet-base-v2",
    "index_name": "lbr-hier-all-mpnet-base-v2",
    "ks": [20, 40, 60],
    "timestamp": "2024-11-01T14:23:05"
  },
  "benchmarks": {
    "cuad": {
      "n_queries": 50,
      "char_recall_at_k":    {"20": 0.154, "40": 0.333, "60": 0.641},
      "char_precision_at_k": {"20": 0.0077, "40": 0.0135, "60": 0.0141},
      "chunk_recall_at_k":   {"20": 0.12, "40": 0.28, "60": 0.52},
      "chunk_precision_at_k":{"20": 0.006, "40": 0.007, "60": 0.009}
    }
  },
  "overall": {
    "char_recall_at_k":    {"20": 0.195, "40": 0.332, "60": 0.482},
    "char_precision_at_k": {"20": 0.015, "40": 0.013, "60": 0.011},
    "chunk_recall_at_k":   {"20": 0.11, "40": 0.23, "60": 0.41},
    "chunk_precision_at_k":{"20": 0.006, "40": 0.006, "60": 0.007}
  }
}
```

`run_summary` fields:

| Field | Description |
| ----- | ----------- |
| `run_info.label` | Display name for this run (defaults to embedding model name; override with `--label`) |
| `run_info.embedding_model` | Embedding model used |
| `run_info.index_name` | OpenSearch index queried |
| `run_info.ks` | K cutoffs evaluated |
| `run_info.timestamp` | ISO-8601 timestamp of the run |
| `benchmarks.<name>.n_queries` | Number of queries evaluated for this sub-benchmark |
| `benchmarks.<name>.*_at_k` | Average metric across queries; keys are stringified K values |
| `overall.*_at_k` | Macro-average of per-benchmark scores (not weighted by n_queries) |

---

### Quick analysis with `jq`

```bash
# Queries with zero character recall at K=20
jq 'select(.metrics_by_k != null) | select(.metrics_by_k[] | select(.k==20) | .char_recall == 0)' logs/eval_trace.jsonl

# Top-5 retrieved chunks for query index 3
jq 'select(.query_idx==3) | .retrieved_all[:5]' logs/eval_trace.jsonl

# Queries where the rank-1 chunk hit a GT span
jq 'select(.metrics_by_k != null) | select(.metrics_by_k[0].chunk_hits[0].is_chunk_hit == true) | .query' logs/eval_trace.jsonl

# GT snippets that were never hit at K=40 (hard negatives)
jq 'select(.metrics_by_k != null) | .metrics_by_k[] | select(.k==40) | .gt_snippet_hits[] | select(.is_hit == false)' logs/eval_trace.jsonl

# Print the run summary
jq 'select(._type == "run_summary")' logs/eval_trace.jsonl
```


---

## How chunk-to-span mapping works

1. Each corpus document is ingested with its **relative file path** (e.g. `cuad/contract_001.txt`) stored as `metadata.citation`.
2. The chunker records `char_start` / `char_end` (character offsets in the original document text) on every child chunk.
3. At evaluation time, retrieved chunks are grouped by `metadata.citation` (= file path).
4. Spans within the same file are **sorted and merged** (overlapping spans collapsed).
5. Merged retrieved spans are intersected with ground-truth spans from the benchmark JSON.
6. Recall and precision are computed from total character counts.

---

## File Reference


| File                       | Description                                                                              |
| -------------------------- | ---------------------------------------------------------------------------------------- |
| `loader.py`                | `LegalBenchRAGCorpusLoader`, `load_benchmark()`, `BenchmarkTestCase`, `BenchmarkSnippet` |
| `pipeline.py`              | `LegalBenchRAGIngestionPipeline` — chunker → embed → OpenSearch                          |
| `ingest.py`                | CLI for corpus ingestion                                                                 |
| `eval_precision_recall.py` | CLI for character-level Precision & Recall evaluation                                    |


---

## Sub-benchmarks


| Name          | Source                                          | Domain                                  |
| ------------- | ----------------------------------------------- | --------------------------------------- |
| `cuad`        | [CUAD](https://arxiv.org/abs/2103.06268)        | Commercial contracts (expert-annotated) |
| `contractnli` | [ContractNLI](https://arxiv.org/abs/2110.01799) | NDA / contract NLI                      |
| `maud`        | [MAUD](https://arxiv.org/abs/2301.00876)        | Merger agreements                       |
| `privacy_qa`  | [PrivacyQA](https://aclanthology.org/D19-1500/) | App privacy policies                    |


---

## Citation

```bibtex
@article{pipitone2024legalbenchrag,
  title={LegalBench-RAG: A Benchmark for Retrieval-Augmented Generation in the Legal Domain},
  author={Pipitone, Nicholas and Houir Alami, Ghita},
  journal={arXiv preprint arXiv:2408.10343},
  year={2024},
  url={https://arxiv.org/abs/2408.10343}
}
```

