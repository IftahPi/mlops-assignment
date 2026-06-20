from evals.run_eval import _sqls_from_history


def test_sqls_from_history_keeps_order_and_skips_non_sql_entries():
    history = [
        {"node": "generate_sql", "sql": "SELECT 1"},
        {"node": "verify", "ok": False, "issue": "x"},
        {"node": "revise", "sql": "SELECT 2"},
    ]
    assert _sqls_from_history(history) == ["SELECT 1", "SELECT 2"]


def test_sqls_from_history_empty():
    assert _sqls_from_history([]) == []
