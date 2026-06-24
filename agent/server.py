"""FastAPI wrapper exposing the agent over HTTP.

Run:
    uv run uvicorn agent.server:app --host 0.0.0.0 --port 8001

The /answer endpoint accepts {question, db, tags?} and returns the
agent's final SQL, the result rows, and per-iteration history.
"""
import os
from contextlib import asynccontextmanager
from typing import Any

import anyio
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel

load_dotenv()

from agent.graph import AgentState, graph  # noqa: E402
from agent.schema import available_dbs  # noqa: E402
from agent.trace import configure_logging, format_run_start, langfuse_metadata, logger  # noqa: E402

# Per-step trace logging in the uvicorn console (on by default; AGENT_DEBUG=0 to quiet).
configure_logging()

# Snapshot the valid db set once at startup. Validating req.db against this (instead of
# globbing the filesystem on every request) keeps the path-traversal guard but avoids a
# per-request stat/glob that, under load, hits the open-FD limit and makes Path.exists()
# silently return False - which rejected ~2/3 of valid requests with HTTP 400 (Phase 6 baseline).
VALID_DBS = frozenset(available_dbs())

# Langfuse callback handler. If keys are set we initialize it; failures
# are NOT swallowed - a misconfigured Langfuse should not silently
# produce zero traces.
_lf_handler: Any = None
if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
    from langfuse.langchain import CallbackHandler

    _lf_handler = CallbackHandler()


def _threadpool_limit() -> int:
    """Max threads for sync endpoints. Default 40 = Starlette's default. Raising it lets
    more /answer runs execute concurrently - Phase-6 Iteration 4: the agent's sync handler
    (not vLLM) was the throughput cap. Tunable via AGENT_MAX_THREADS."""
    return int(os.environ.get("AGENT_MAX_THREADS", "40"))


@asynccontextmanager
async def lifespan(app: "FastAPI"):
    # Resize the threadpool that runs sync endpoints (the /answer handler is sync, so each
    # in-flight request holds one thread for its whole multi-call duration).
    limiter = anyio.to_thread.current_default_thread_limiter()
    limiter.total_tokens = _threadpool_limit()
    logger.info("agent sync threadpool limit = %d", limiter.total_tokens)
    yield


app = FastAPI(lifespan=lifespan)

# End-to-end /answer latency = the Phase-6 SLO metric. Buckets straddle the 5s
# SLO boundary so histogram_quantile is accurate right where it matters.
AGENT_LATENCY = Histogram(
    "agent_request_duration_seconds",
    "End-to-end /answer handler latency (the Phase-6 SLO metric).",
    buckets=(0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 30.0, 60.0, 120.0),
)


class AnswerRequest(BaseModel):
    question: str
    db: str
    tags: dict[str, str] = {}


class AnswerResponse(BaseModel):
    sql: str
    rows: list[list[Any]] | None
    iterations: int
    ok: bool
    error: str | None = None
    history: list[dict[str, Any]] = []


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/answer", response_model=AnswerResponse)
@AGENT_LATENCY.time()
def answer(req: AnswerRequest) -> AnswerResponse:
    # Validate db against the known set before it reaches db_path() / the schema
    # loader - db is attacker-controllable and feeds a filesystem path, so an
    # unvalidated value like "../.." would traverse out of the data directory.
    if req.db not in VALID_DBS:
        raise HTTPException(status_code=400, detail=f"unknown db: {req.db!r}")

    logger.info(format_run_start(req.question, req.db))
    state = AgentState(question=req.question, db_id=req.db)
    config: dict[str, Any] = {
        "callbacks": [_lf_handler] if _lf_handler is not None else [],
        # Forward tags as Langfuse trace tags (filterable in the list, used in Phase 6),
        # not just plain metadata. langfuse_metadata() adds the magic langfuse_tags key.
        "metadata": langfuse_metadata(req.tags),
    }
    try:
        final = graph.invoke(state, config=config)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    sql = final.get("sql", "")
    iteration = final.get("iteration", 0)
    history = final.get("history", [])
    execution = final.get("execution")

    if execution is None:
        return AnswerResponse(
            sql=sql,
            rows=None,
            iterations=iteration,
            ok=False,
            error="agent produced no execution result",
            history=history,
        )
    if not execution.ok:
        return AnswerResponse(
            sql=sql,
            rows=None,
            iterations=iteration,
            ok=False,
            error=execution.error,
            history=history,
        )

    return AnswerResponse(
        sql=sql,
        rows=[list(r) for r in (execution.rows or [])],
        iterations=iteration,
        ok=True,
        history=history,
    )
