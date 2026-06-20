"""Eval runner using execution accuracy.

Reads evals/eval_set.jsonl, calls the agent at AGENT_URL on each question,
then compares the agent's SQL output to the gold SQL by *executed rows*
(canonicalized: sorted, stringified, None-coerced to empty).

Helpers (run_sql / canonicalize / matches) are provided. You implement
eval_one() and summarize().

Run:
    uv run python evals/run_eval.py --out results/eval_baseline.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_FILE = ROOT / "evals" / "eval_set.jsonl"
DEFAULT_OUT_FILE = ROOT / "results" / "eval_baseline.json"
DB_DIR = ROOT / "data" / "bird"
AGENT_URL_DEFAULT = "http://localhost:8001/answer"


# ---------- Helpers (provided) -----------------------------------------

def run_sql(db_id: str, sql: str, timeout: float = 5.0) -> tuple[bool, list[tuple] | None, str | None]:
    """Run sql against db_id in read-only mode. Returns (ok, rows, error)."""
    path = DB_DIR / f"{db_id}.sqlite"
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout) as conn:
            cur = conn.execute(sql)
            rows = cur.fetchall()
            return True, rows, None
    except Exception as e:  # noqa: BLE001
        return False, None, f"{type(e).__name__}: {e}"


def canonicalize(rows: list[tuple] | None) -> list[tuple] | None:
    """Sort rows; coerce cells to str; None -> ''."""
    if rows is None:
        return None
    return sorted(tuple("" if c is None else str(c) for c in row) for row in rows)


def matches(gold_rows: list[tuple] | None, pred_rows: list[tuple] | None) -> bool:
    if gold_rows is None or pred_rows is None:
        return False
    return canonicalize(gold_rows) == canonicalize(pred_rows)


# ---------- Implement these (Phase 5) ----------------------------------

def _sqls_from_history(history: list[dict]) -> list[str]:
    """Ordered SQL strings across iterations (generate + each revise)."""
    return [h["sql"] for h in history if isinstance(h, dict) and "sql" in h]


def eval_one(question: dict, agent_url: str) -> dict:
    """Score one question. Return a dict capturing per-iteration correctness."""
    db_id = question["db_id"]
    gold_ok, gold_rows, _ = run_sql(db_id, question["gold_sql"])

    try:
        resp = httpx.post(
            agent_url,
            json={"question": question["question"], "db": db_id},
            timeout=120.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:  # noqa: BLE001 -- HTTP boundary: record and move on
        return {
            "db_id": db_id,
            "question": question["question"],
            "iterations": 0,
            "correct_per_iter": [],
            "final_correct": False,
            "final_sql": "",
            "agent_ok": False,
            "error": f"{type(e).__name__}: {e}",
        }

    sqls = _sqls_from_history(data.get("history", []))
    if not sqls and data.get("sql"):
        sqls = [data["sql"]]

    correct_per_iter: list[bool] = []
    for sql in sqls:
        ok, rows, _ = run_sql(db_id, sql)
        correct_per_iter.append(bool(gold_ok and ok and matches(gold_rows, rows)))

    return {
        "db_id": db_id,
        "question": question["question"],
        "iterations": data.get("iterations", len(sqls)),
        "correct_per_iter": correct_per_iter,
        "final_correct": correct_per_iter[-1] if correct_per_iter else False,
        "final_sql": data.get("sql", ""),
        "agent_ok": bool(data.get("ok", False)),
        "error": data.get("error"),
    }


def summarize(results: list[dict]) -> dict:
    """Aggregate per-question results.

    Per-iteration carry-forward: if the agent terminated at iteration j < k
    (verify said ok at j, or it hit MAX_ITERATIONS at j < k), treat the
    question's iteration-k result as identical to its iteration-j result.
    The agent stopped emitting; whatever it had at termination is what
    would have been served had we polled at iteration k.
    """
    total = len(results)
    overall = sum(1 for r in results if r["final_correct"]) / total if total else 0.0

    n_iters = max((len(r["correct_per_iter"]) for r in results), default=0)
    per_iteration: list[float] = []
    for k in range(n_iters):
        correct = 0
        for r in results:
            cpi = r["correct_per_iter"]
            if not cpi:
                continue  # agent never produced SQL -> wrong at every iteration
            correct += int(cpi[k] if k < len(cpi) else cpi[-1])  # carry-forward
        per_iteration.append(correct / total if total else 0.0)

    return {
        "n": total,
        "overall_pass_rate": overall,
        "pass_rate_per_iteration": per_iteration,
    }


# ---------- Main (provided) --------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_FILE)
    parser.add_argument("--agent-url", default=AGENT_URL_DEFAULT)
    args = parser.parse_args()

    questions = [json.loads(line) for line in args.eval_set.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(questions)} eval questions from {args.eval_set}")

    results: list[dict] = []
    t0 = time.monotonic()
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['db_id']}: {q['question'][:60]}...", flush=True)
        results.append(eval_one(q, args.agent_url))
    elapsed = time.monotonic() - t0

    summary = summarize(results)
    out = {
        "summary": summary,
        "wall_clock_seconds": elapsed,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
