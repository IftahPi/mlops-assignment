"""Tests for Prometheus metrics instrumentation."""

from prometheus_client import generate_latest

from agent.server import AGENT_LATENCY


def test_histogram_exists_and_is_named_correctly():
    """The AGENT_LATENCY histogram should exist and have the right name."""
    assert AGENT_LATENCY._name == "agent_request_duration_seconds"


def test_histogram_observes_and_exposes_metric():
    """Observing a value should show up in Prometheus exposition text."""
    AGENT_LATENCY.observe(0.5)
    metrics_text = generate_latest()
    assert b"agent_request_duration_seconds_bucket" in metrics_text


def test_histogram_has_5s_bucket():
    """The 5.0 bucket (SLO boundary) should be present in the exposition."""
    metrics_text = generate_latest()
    assert b'le="5.0"' in metrics_text
