# Benchmark 50 — Reformatted Query Rewrites

Query rewrites for the LegalBench-RAG benchmark_50 subset (200 queries: 50 per dataset), split by dataset for easier use.

**Source:** Rewrites from `results/query_rewrites_compare_1vars_*.json`  
**Prompt variant:** v4_reddit_style (Reddit-style plain-language rewrites)

## Structure

```
benchmark_50_reformated/
├── mistral/           # Mistral Small Latest
│   ├── contractnli.json
│   ├── cuad.json
│   ├── maud.json
│   └── privacy_qa.json
└── qwen72b/           # Qwen 2.5 72B Instruct
    ├── contractnli.json
    ├── cuad.json
    ├── maud.json
    └── privacy_qa.json
```

Each JSON file contains 50 `{original, rewrites}` entries for that dataset.

## Full benchmark JSONs (rewritten `query` + original snippets)

The folder [`benchmark_50_reformated_proccessed/`](../benchmark_50_reformated_proccessed) mirrors the `benchmarks_50` schema: same `snippets` / spans / answers, but each `query` is the `v4_reddit_style` rewrite (per model: `mistral/`, `qwen72b/`). Use for retrieval eval without a separate `--rewrites-file`.

Regenerate (requires a local `benchmarks_50/` tree aligned with `original` keys):

```bash
python3 scripts/build_benchmark_50_reformated_processed.py
```
