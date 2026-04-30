# LegalBench-RAG Evaluation

Retrieval evaluation pipeline for [zeroentropy-ai/legalbenchrag](https://github.com/zeroentropy-ai/legalbenchrag) — a character-level IR benchmark covering legal contracts (CUAD, ContractNLI, MAUD, PrivacyQA).

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

## NOTE
- `legalrag/core/config.py` and `.env` files are implemented for the sake of centeralized configs control for STABLE SYSTEM in the future.
- For evaluation and experiments, use CLI flags to contorl embedding models, embedding provider, chunk size, etc.
- The only setting that cannot be bypassed by CLI flags and use the `legalrag/core/config.py` is the `opensearch` set up

---
## End-to-End Workflow

This section walks through a full experiment: ingest → evaluate → analyse results.
The example uses `all-mpnet-base-v2` with hierarchical chunking on the 50-query subset.

### Step 1 — Ingest the corpus

```bash
python -m evaluation.LegalBenchRAG.ingest \
    --data-dir data/LegalBenchRAG \
    --index-name lbr-hier-all-mpnet-base-v2 \
    --chunker hierarchical \
    --parent-size 2048 \
    --chunk-size 512 \
    --chunk-overlap 64 \
    --embedding-provider sentence_transformers \
    --embedding-model sentence-transformers/all-mpnet-base-v2 \
    --all
```

Verify the index was created and has documents:

```bash
curl http://localhost:9200/lbr-hier-all-mpnet-base-v2/_count
```

### Step 2 — Run evaluation

Evaluate on the **50-query sampled subset** (`data/LegalBenchRAG/benchmarks_50/`), using
K cutoffs `2 4 6 10 15 20 40 60` and writing a per-query **trace file**:

**trace file**: detailed logs of each run, it's very important for debugging, result reproducbility and analytics.


```bash
python -m evaluation.LegalBenchRAG.eval_precision_recall \
    --data-dir data/LegalBenchRAG \
    --index-name lbr-hier-all-mpnet-base-v2 \
    --embedding-provider sentence_transformers \
    --embedding-model sentence-transformers/all-mpnet-base-v2 \
    --benchmarks-dir data/LegalBenchRAG/benchmarks_50 \
    --ks 2 4 6 10 15 20 40 60 \
    --trace-file logs/eval/original/lbr_hier_all-mpnet.jsonl
```

The terminal prints a results table like:

```
benchmark       R@6    R@20   R@60   P@6    P@20   P@60
contractnli     0.231  0.308  0.462  0.0308 0.0154 0.0090
cuad            0.154  0.333  0.641  0.0154 0.0077 0.0141
maud            0.083  0.125  0.167  0.0083 0.0042 0.0056
privacy_qa      0.313  0.563  0.646  0.0417 0.0333 0.0167
OVERALL         0.195  0.332  0.482  0.0240 0.0150 0.0113
```

### Step 3 — Analyse with notebooks

Analyics is done via loading trace files in notebooks(`notebooks/`), some notebooks includes:

| Notebook | Purpose |
| -------- | ------- |
| [chunk_hits_eda.ipynb](../../notebooks/chunk_hits_eda.ipynb) | EDA on chunk-level hit/miss patterns across benchmarks |
| [eval_trace_inspector.ipynb](../../notebooks/eval_trace_inspector.ipynb) | Interactive per-query trace inspector; filter by recall, drill into retrieved chunks |
| [plot_eval_results.ipynb](../../notebooks/plot_eval_results.ipynb) | Multi-run comparison charts (recall/precision vs K, model vs model) |

Pass your `--trace-file` path into the notebooks to load results.


---

## Experiment Matrix

The scripts in `Tian_scripts/` cover a full sweep across embedding models, chunkers, and
query-rewriting variants. The table below shows what each script targets.

### Embedding models

| Index label | Model | Provider |
| ----------- | ----- | -------- |
| `*-clerc` | `jhu-clsp/BERT-DPR-CLERC-ft` | `huggingface` |
| `*-legalbert` | `nlpaueb/legal-bert-base-uncased` | `huggingface` |
| `*-legal-embed-bge-base-en-v1.5` | `axondendriteplus/Legal-Embed-bge-base-en-v1.5` | `sentence_transformers` |
| `*-all-mpnet-base-v2` | `sentence-transformers/all-mpnet-base-v2` | `sentence_transformers` |
| `*-octen` | `Octen/Octen-Embedding-0.6B` | `sentence_transformers` |
| `*-qwen3` | `Qwen/Qwen3-Embedding-0.6B` | `sentence_transformers` |

### Query-rewriting variants

Each script evaluates on a different LLM-rewritten version of the benchmark queries.
The `--benchmarks-dir` path determines which variant is used.

| Script | Benchmarks dir | Query rewriter |
| ------ | -------------- | -------------- |
| `Tian_eval_original.sh` | `benchmarks_50/` | None (original queries) |
| `Tian_eval_qwen2.5.sh` | `benchmark_50_reformated_proccessed/qwen72b` | Qwen2.5-72B |
| `Tian_eval_qwen3.5.sh` | `benchmark_50_reformated_proccessed/qwen35_9b` | Qwen3.5-9B |
| `Tian_eval_mistral.sh` | `benchmark_50_reformated_proccessed/mistral` | Mistral |
| `Tian_eval_gemini_3.sh` | `benchmark_50_reformated_proccessed/gemini-3-flash` | Gemini 3 Flash |
| `Tian_eval_mini.sh` | *(uses `data/legalbenchrag-mini`)* | Original (mini dataset) |

Trace files land in `logs/eval/{variant}/`, e.g. `logs/eval/qwen3.5/lbr_hier_clerc.jsonl`.

### Running a full sweep (one script)

```bash
bash evaluation/LegalBenchRAG/Tian_scripts/Tian_eval_original.sh
```

Each script ingests every model × chunker combination and immediately evaluates it,
so indices accumulate in OpenSearch across runs.

---


## Chunker Strategies

Two chunkers are available. Pass `--chunker` to `ingest` to select one.

### `hierarchical` (default)

Two-level structure: large **parent** chunks (~~2048 chars) contain overlapping
**child** chunks (~~512 chars). Only child chunks are embedded and retrieved;
the parent is fetched at generation time for richer context
(*small-to-big retrieval*).

- Better for generative RAG (wider context per retrieved chunk)
- `--parent-size` controls the parent window (using 2048)
- `--chunk-size` controls the child window (using 512 chars)
- `--chunk-overlap` controls child overlap (using 64 chars)

```bash
# Ingest with hierarchical chunker into a named index
python -m evaluation.LegalBenchRAG.ingest \
    --data-dir data/LegalBenchRAG \
    --chunker hierarchical \
    --parent-size 2048 \
    --chunk-size 512 \
    --chunk-overlap 64 \
    --index-name lbr-hier-all-mpnet-base-v2 \
    --embedding-provider sentence_transformers \
    --embedding-model sentence-transformers/all-mpnet-base-v2 \
    --all

# Evaluate against that index
python -m evaluation.LegalBenchRAG.eval_precision_recall \
    --data-dir data/LegalBenchRAG \
    --index-name lbr-hier-all-mpnet-base-v2 \
    --embedding-provider sentence_transformers \
    --embedding-model sentence-transformers/all-mpnet-base-v2 \
    --benchmarks-dir data/LegalBenchRAG/benchmarks_50 \
    --trace-file logs/eval/original/lbr_hier_all-mpnet.jsonl \
    --ks 2 4 6 10 15 20 40 60
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
    --index-name lbr-rec-all-mpnet-base-v2 \
    --embedding-provider sentence_transformers \
    --embedding-model sentence-transformers/all-mpnet-base-v2 \
    --all

# Evaluate against that index
python -m evaluation.LegalBenchRAG.eval_precision_recall \
    --data-dir data/LegalBenchRAG \
    --index-name lbr-rec-all-mpnet-base-v2 \
    --embedding-provider sentence_transformers \
    --embedding-model sentence-transformers/all-mpnet-base-v2 \
    --benchmarks-dir data/LegalBenchRAG/benchmarks_50 \
    --trace-file logs/eval/original/lbr_rec_all-mpnet.jsonl \
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

The `EMBEDDING_PROVIDER` in `.env` controls which embedder class is used. The `--embedding-model` and `--embedding-provider` flags (on both `ingest` and `eval_precision_recall`) override the `.env` values per-run — no file edits needed for experiments.

| `EMBEDDING_PROVIDER`    | Class                        | Use when                                                                 |
| ----------------------- | ---------------------------- | ------------------------------------------------------------------------ |
| `sentence_transformers` | `SentenceTransformerEmbedder`| Model has a `sentence_transformers` config on HF (e.g. `all-mpnet-base-v2`, `legal-bert-base-uncased`, `Legal-Embed-bge-base-en-v1.5`) |
| `huggingface`           | `HuggingFaceEmbedder`        | Raw `AutoModel` + `AutoTokenizer` with mean pooling — for models not packaged as sentence-transformers (e.g. `jhu-clsp/BERT-DPR-CLERC-ft`) |
| `openai`                | `OpenAIEmbedder`             | OpenAI API or compatible endpoint                                        |

`HuggingFaceEmbedder` mean-pools the last hidden state over non-padding tokens and L2-normalises the result. It is GPU-aware (uses CUDA if available).

Example — ingest and evaluate with `jhu-clsp/BERT-DPR-CLERC-ft` (CLERC model):

```bash
python -m evaluation.LegalBenchRAG.ingest \
    --data-dir data/LegalBenchRAG \
    --index-name lbr-hier-clerc \
    --chunker hierarchical \
    --parent-size 2048 \
    --chunk-size 512 \
    --chunk-overlap 64 \
    --embedding-provider huggingface \
    --embedding-model jhu-clsp/BERT-DPR-CLERC-ft \
    --all

python -m evaluation.LegalBenchRAG.eval_precision_recall \
    --data-dir data/LegalBenchRAG \
    --index-name lbr-hier-clerc \
    --embedding-provider huggingface \
    --embedding-model jhu-clsp/BERT-DPR-CLERC-ft \
    --benchmarks-dir data/LegalBenchRAG/benchmarks_50 \
    --trace-file logs/eval/original/lbr_hier_clerc.jsonl \
    --ks 2 4 6 10 15 20 40 60
```

---

## Re-ingestion

Delete the index and re-run ingestion when you change the embedding model or chunk size:

```bash
curl -X DELETE http://localhost:9200/lbr-hier-all-mpnet-base-v2
python -m evaluation.LegalBenchRAG.ingest \
    --data-dir data/LegalBenchRAG \
    --index-name lbr-hier-all-mpnet-base-v2 \
    --embedding-provider sentence_transformers \
    --embedding-model sentence-transformers/all-mpnet-base-v2 \
    --all
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
| `--chunk-size N`          | required                 | Child chunk size in characters                                         |
| `--chunk-overlap N`       | required                 | Character overlap between consecutive chunks                           |
| `--parent-size N`         | 1500                     | Parent chunk size — hierarchical only                                  |
| `--embedding-provider PROVIDER` | required           | Embedding provider: `sentence_transformers`, `huggingface`, or `openai` |
| `--embedding-model MODEL` | required                 | Embedding model name                                                   |
| `--index-name NAME`       | `legalrag-legalbenchrag` | OpenSearch index to ingest into                                        |
| `--log-level`             | INFO                     | Verbosity                                                              |


### Evaluation options (`evaluation.LegalBenchRAG.eval_precision_recall`)


| Flag                      | Default                     | Description                                               |
| ------------------------- | --------------------------- | --------------------------------------------------------- |
| `--data-dir PATH`         | required                    | Root of downloaded data dir                               |
| `--benchmarks-dir PATH`   | `<data-dir>/benchmarks`     | Override benchmarks directory (e.g. for a sampled subset or a rewritten-query variant) |
| `--benchmarks NAME …`     | all four                    | Sub-benchmarks to evaluate                                |
| `--limit N`               | None                        | Cap test cases per benchmark (for fast iteration)         |
| `--ks K …`                | `1 5 10 20`                 | Rank cutoffs; retrieves `max(ks)` chunks per query        |
| `--index-name NAME`       | required                    | OpenSearch index to query — must match ingestion          |
| `--embedding-provider PROVIDER` | required                    | Embedding provider: `sentence_transformers`, `huggingface`, or `openai`. Must match ingest. |
| `--embedding-model MODEL` | required                    | Embedding model name. Must match ingest.                               |
| `--trace-file PATH`       | required                    | Write per-query JSONL retrieval trace (see below)         |
| `--log-level`             | WARNING                     | Verbosity                                                 |


---

## Retrieval trace file

Pass `--trace-file logs/eval_trace.jsonl` to write one JSON line per query. Useful for debugging low-scoring queries and as input to the analysis notebooks.

```bash
python -m evaluation.LegalBenchRAG.eval_precision_recall \
    --data-dir data/LegalBenchRAG \
    --benchmarks-dir data/LegalBenchRAG/benchmarks_50 \
    --index-name lbr-hier-all-mpnet-base-v2 \
    --embedding-provider sentence_transformers \
    --embedding-model sentence-transformers/all-mpnet-base-v2 \
    --ks 2 4 6 10 15 20 40 60 \
    --trace-file logs/eval/original/lbr_hier_all-mpnet.jsonl
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

## How chunk-to-span mapping works for char-level

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
| `Tian_scripts/`            | Full sweep scripts — one per query-rewriting variant; each covers all embedding models   |


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
