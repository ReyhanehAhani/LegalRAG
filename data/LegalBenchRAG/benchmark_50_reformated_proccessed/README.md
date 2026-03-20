# benchmark_50_reformated_proccessed

Benchmark JSONs in the **same shape as** [`benchmarks_50`](../benchmarks_50): each file has a `tests` array with `query`, `snippets` (`file_path`, `span`, `answer`, …). The **`query` field is the rewritten text** from [`benchmark_50_reformated`](../benchmark_50_reformated) (`v4_reddit_style` by default). Snippets and spans are **unchanged** from `benchmarks_50`.

## Layout

```
benchmark_50_reformated_proccessed/
├── mistral/     # one JSON per dataset (50 tests each)
├── qwen72b/
└── README.md
```

Each JSON also includes a top-level `metadata` block (for traceability). Loaders that only read `tests` (e.g. `evaluation.LegalBenchRAG.loader.load_benchmark`) still work.

## Regenerate

From repo root:

```bash
python3 scripts/build_benchmark_50_reformated_processed.py
python3 scripts/build_benchmark_50_reformated_processed.py --models mistral qwen72b
python3 scripts/build_benchmark_50_reformated_processed.py --variant v4_reddit_style
```

If a rewrite is missing or is an `[ERROR:...]` placeholder in the reformated file, the **original** `benchmarks_50` query is kept; see `metadata.stats` per file.

## Eval example

Point `--benchmarks-dir` at a model subfolder:

```bash
python3 -m evaluation.LegalBenchRAG.eval_precision_recall \
  --data-dir data/LegalBenchRAG \
  --benchmarks-dir benchmark_50_reformated_proccessed/mistral \
  --ks 1 5 10 20
```

(No `--rewrites-file` needed: the query text is already rewritten in the benchmark files.)
