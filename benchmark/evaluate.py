"""Evaluation harness for the Yelp text-to-SQL benchmark.

Computes two standard metrics per model, plus per-category / per-difficulty
breakdowns and an error taxonomy:

- Execution rate  : fraction of generated queries that run without error.
- Execution match : fraction whose result set exactly matches the gold result.

Usage:
    python3 benchmark/evaluate.py                 # all models
    python3 benchmark/evaluate.py --model mock    # a single model
    python3 benchmark/evaluate.py --limit 5       # quick smoke test
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark import db
from benchmark.runner import BaseModel, MockModel, build_models

HERE = Path(__file__).parent
QUESTIONS_PATH = HERE / "questions.json"
REPORT_PATH = HERE / "report.md"


def load_questions() -> list[dict]:
    with open(QUESTIONS_PATH) as f:
        data = json.load(f)
    return data["questions"]


def classify_error(err: Exception) -> str:
    msg = str(err).lower()
    if "no such column" in msg:
        return "hallucinated column"
    if "no such table" in msg:
        return "wrong table"
    if "syntax error" in msg or "near \"" in msg:
        return "syntax error"
    if "forbidden" in msg:
        return "forbidden keyword"
    if "only select" in msg:
        return "non-select query"
    return "other"


def evaluate_model(model: BaseModel, questions: list[dict]) -> dict:
    results = []
    for q in questions:
        gold = db.gold_result(q["gold_sql"])
        entry = {
            "id": q["id"],
            "category": q["category"],
            "difficulty": q["difficulty"],
            "question": q["question"],
            "gold_sql": q["gold_sql"],
        }
        try:
            sql = model.generate_sql(q["question"])
            entry["sql"] = sql
        except Exception as e:  # model call failed (network, key, etc.)
            entry["error"] = f"model error: {e}"
            entry["status"] = "model_error"
            results.append(entry)
            continue

        try:
            rows = db.run_query(sql)
        except Exception as e:
            entry["error"] = str(e)
            entry["status"] = "execution_error"
            entry["error_type"] = classify_error(e)
            results.append(entry)
            continue

        entry["status"] = "correct" if set(rows) == set(gold) else "wrong_result"
        results.append(entry)

    return summarize(results), results


def summarize(results: list[dict]) -> dict:
    n = len(results)
    executed = [r for r in results if r["status"] in ("correct", "wrong_result")]
    correct = [r for r in results if r["status"] == "correct"]

    summary = {
        "total": n,
        "executed": len(executed),
        "correct": len(correct),
        "execution_rate": round(len(executed) / n, 4) if n else 0.0,
        "exact_match": round(len(correct) / n, 4) if n else 0.0,
        "exact_match_of_executed": round(len(correct) / len(executed), 4) if executed else 0.0,
        "by_category": _group(results, "category"),
        "by_difficulty": _group(results, "difficulty"),
        "error_types": _error_counts(results),
    }
    return summary


def _group(results: list[dict], key: str) -> dict:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        buckets[r[key]].append(r)
    out = {}
    for k, items in sorted(buckets.items()):
        correct = sum(1 for r in items if r["status"] == "correct")
        out[k] = {
            "total": len(items),
            "correct": correct,
            "exact_match": round(correct / len(items), 4),
        }
    return out


def _error_counts(results: list[dict]) -> dict:
    counts: dict[str, int] = defaultdict(int)
    for r in results:
        if r["status"] == "execution_error":
            counts[r.get("error_type", "other")] += 1
        elif r["status"] == "model_error":
            counts["model_error"] += 1
        elif r["status"] == "wrong_result":
            counts["wrong_result"] += 1
    return dict(counts)


def render_report(model_name: str, summary: dict, results: list[dict] | None = None) -> str:
    lines = []
    lines.append(f"# Model: {model_name}")
    lines.append("")
    lines.append(f"- Total questions: {summary['total']}")
    lines.append(f"- Executed without error: {summary['executed']} ({summary['execution_rate']:.1%})")
    lines.append(f"- Exact match (execution accuracy): {summary['correct']} ({summary['exact_match']:.1%})")
    lines.append("")
    lines.append("## By category")
    lines.append("")
    lines.append("| Category | Total | Correct | Exact match |")
    lines.append("|---|---|---|---|")
    for k, v in summary["by_category"].items():
        lines.append(f"| {k} | {v['total']} | {v['correct']} | {v['exact_match']:.1%} |")
    lines.append("")
    lines.append("## By difficulty")
    lines.append("")
    lines.append("| Difficulty | Total | Correct | Exact match |")
    lines.append("|---|---|---|---|")
    for k, v in summary["by_difficulty"].items():
        lines.append(f"| {k} | {v['total']} | {v['correct']} | {v['exact_match']:.1%} |")
    lines.append("")
    lines.append("## Error breakdown")
    lines.append("")
    if summary["error_types"]:
        lines.append("| Error type | Count |")
        lines.append("|---|---|")
        for k, v in sorted(summary["error_types"].items(), key=lambda x: -x[1]):
            lines.append(f"| {k} | {v} |")
    else:
        lines.append("No errors.")
    lines.append("")
    if results:
        failures = [r for r in results if r["status"] != "correct"]
        if failures:
            lines.append("## Failure detail")
            lines.append("")
            lines.append("| # | Question | Status | Detail |")
            lines.append("|---|---|---|---|")
            for r in failures:
                detail = r.get("error") or r.get("sql") or ""
                detail = detail.replace("|", "\\|")[:120]
                lines.append(f"| {r['id']} | {r['question'][:60]} | {r['status']} | {detail} |")
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Only run this model name")
    parser.add_argument("--limit", type=int, default=None, help="Run only first N questions")
    parser.add_argument("--out", default=None, help="Write markdown report to file")
    args = parser.parse_args()

    questions = load_questions()
    if args.limit:
        questions = questions[: args.limit]

    if args.model == "mock":
        models: list[BaseModel] = [MockModel()]
    elif args.model:
        models = [m for m in build_models() if args.model in m.name]
        if not models:
            raise SystemExit(f"No model matched '{args.model}'")
    else:
        models = build_models()

    report_parts = []
    all_results = {}
    for model in models:
        summary, results = evaluate_model(model, questions)
        report_parts.append(render_report(model.name, summary, results))
        all_results[model.name] = results
        print(f"{model.name:<20} exact={summary['exact_match']:.1%}  "
              f"executed={summary['execution_rate']:.1%}  "
              f"({summary['correct']}/{summary['total']})")

    full = "\n\n".join(report_parts)
    out = args.out or str(REPORT_PATH)
    Path(out).write_text(full)
    print(f"\nReport written to {out}")

    results_path = HERE / "results.json"
    results_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"Per-question results written to {results_path}")


if __name__ == "__main__":
    main()
