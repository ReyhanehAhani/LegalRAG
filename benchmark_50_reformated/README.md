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
