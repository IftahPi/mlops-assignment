"""Tests for agent.trace module."""
import pytest

from agent.execution import ExecutionResult
from agent.trace import _oneline, debug_enabled, format_step


def test_format_step_generate_includes_symbol_and_sql() -> None:
    """Test that generate_sql step includes the emoji symbol and SQL."""
    result = format_step("generate_sql", {"sql": "SELECT name FROM drivers", "iteration": 1})
    assert "🧭" in result
    assert "SELECT name FROM drivers" in result


def test_format_step_verify_failure_shows_issue() -> None:
    """Test that verify step failure includes the issue description."""
    result = format_step("verify", {"verify_ok": False, "verify_issue": "returned 0 rows"})
    assert "🔎" in result
    assert "returned 0 rows" in result
    assert "ok=false" in result


def test_format_step_execute_includes_rows() -> None:
    """Test that execute step includes execution result with rows."""
    er = ExecutionResult(ok=True, rows=[("a",), ("b",)], columns=["name"], row_count=2)
    result = format_step("execute", {"execution": er})
    assert "📊" in result
    assert "rows" in result


def test_oneline_collapses_and_truncates() -> None:
    """Test that _oneline collapses whitespace and truncates with ellipsis."""
    # Test collapse
    assert _oneline("a\n\n   b") == "a b"

    # Test truncation
    long_text = "x" * 300
    truncated = _oneline(long_text, limit=50)
    assert len(truncated) <= 51
    assert truncated[-1] == "…"


def test_debug_enabled_respects_env_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that debug_enabled respects AGENT_DEBUG env var."""
    monkeypatch.setenv("AGENT_DEBUG", "0")
    assert debug_enabled() is False

    monkeypatch.setenv("AGENT_DEBUG", "1")
    assert debug_enabled() is True

    monkeypatch.delenv("AGENT_DEBUG", raising=False)
    assert debug_enabled() is True
