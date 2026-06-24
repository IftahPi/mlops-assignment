"""The agent sync-endpoint threadpool limit (Phase-6 Iteration 4)."""
import agent.server


def test_threadpool_limit_defaults_to_40(monkeypatch):
    monkeypatch.delenv("AGENT_MAX_THREADS", raising=False)
    assert agent.server._threadpool_limit() == 40


def test_threadpool_limit_reads_env(monkeypatch):
    monkeypatch.setenv("AGENT_MAX_THREADS", "100")
    assert agent.server._threadpool_limit() == 100
