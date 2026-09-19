# Yelp-SQL Benchmark

A zero-dependency, reproducible benchmark for evaluating how well LLMs (and a
rule-based baseline) translate natural-language questions into SQL over a Yelp
`business_small` schema.

This is the evaluation artifact backing the arXiv-style study. It is a
*hand-written* question set (not LLM-generated), so it is a clean eval set.

## Layout

```
benchmark/
├── schema.sql          # business_small schema (mirrors app/core/openai_client.py)
├── generate_data.py    # builds a deterministic synthetic dataset
├── questions.json      # 50 NL -> gold SQL pairs (categories + difficulty)
├── db.py               # sqlite helpers + SELECT-only safety guard
├── runner.py           # model adapters (OpenAI, Ollama, rule-based, mock)
├── evaluate.py         # harness: execution rate + exact-match accuracy
├── report.md           # generated per-model error analysis
└── data/               # generated: business_small.sqlite + .csv
```

## Quick start

Generate the dataset and run the offline baseline (no API keys or network):

```bash
python3 benchmark/generate_data.py
python3 benchmark/evaluate.py --model rule-based
python3 benchmark/evaluate.py --limit 10          # smoke test
```

## Running a real model

Set `OPENAI_API_KEY` (already in the repo `.env`) and run:

```bash
python3 benchmark/evaluate.py --model openai
```

For a local open-weight model, install [Ollama](https://ollama.com) and pull a
model, then run:

```bash
ollama pull llama3.1
python3 benchmark/evaluate.py --model ollama
# or: OLLAMA_BASE_URL=http://localhost:11434/v1 python3 benchmark/evaluate.py --model ollama
```

Swap models by editing `build_models()` in `runner.py` (e.g. `qwen2.5-coder`,
`llama3.1:8b`, `gpt-4o-mini`).

## Metrics

- **Execution rate** — fraction of generated queries that run without error.
- **Exact match (execution accuracy)** — fraction whose result set equals the
  gold result set (the standard Spider-style metric).

The report also breaks accuracy down by query category
(`filter`, `ranking`, `aggregation`, `grouping`, `combined`) and difficulty, and
produces an error taxonomy (hallucinated column, syntax error, wrong table,
non-select, forbidden keyword, wrong result).

## Notes

- The dataset is **synthetic but faithful** to the real `business_small`
  schema, generated with a fixed seed (`generate_data.py`) so results are
  reproducible. To use real Yelp data, replace
  `benchmark/data/business_small.sqlite` with a dump of the same schema.
- `gold_sql` is executed to compute the ground-truth result, so the answers are
  exact by construction (no hand-computed expected values to drift).
