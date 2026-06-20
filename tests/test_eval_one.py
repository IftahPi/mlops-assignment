import sqlite3

import evals.run_eval as run_eval


def _make_db(path):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t (name TEXT, country TEXT)")
    conn.executemany("INSERT INTO t VALUES (?, ?)", [("Alice", "US"), ("Bob", "UK")])
    conn.commit()
    conn.close()


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_eval_one_computes_per_iteration_correctness(tmp_path, monkeypatch):
    _make_db(tmp_path / "demo.sqlite")
    monkeypatch.setattr(run_eval, "DB_DIR", tmp_path)  # point run_sql at the temp DB

    gold = "SELECT name FROM t WHERE country='US'"      # -> [('Alice',)]
    wrong = "SELECT name FROM t WHERE country='ZZ'"     # -> []
    right = "SELECT name FROM t WHERE country='US'"     # -> [('Alice',)]
    payload = {
        "sql": right,
        "ok": True,
        "iterations": 2,
        "history": [
            {"node": "generate_sql", "sql": wrong},
            {"node": "verify", "ok": False, "issue": "no rows"},
            {"node": "revise", "sql": right},
        ],
    }
    monkeypatch.setattr(run_eval.httpx, "post", lambda *a, **k: _Resp(payload))

    out = run_eval.eval_one(
        {"question": "US names?", "db_id": "demo", "gold_sql": gold},
        "http://x/answer",
    )
    assert out["correct_per_iter"] == [False, True]
    assert out["final_correct"] is True
