"""Pure formatter + logging configuration helper for the agent's per-step trace."""
import logging
import os
import re

logger = logging.getLogger("agent")

# Horizontal rule printed once per complete trace, between questions.
RUN_SEPARATOR = "-" * 140


def langfuse_metadata(tags: dict[str, str]) -> dict[str, str | list[str]]:
    """Build the LangGraph config 'metadata' so Langfuse records FILTERABLE trace tags.

    The LangChain CallbackHandler turns metadata['langfuse_tags'] (a list of strings) into the
    trace's tags - the chips shown in the trace LIST and filterable in the UI (needed for Phase 6).
    Plain metadata keys, by contrast, are only visible once you open a single trace. So we keep the
    raw key/values as metadata AND derive 'key:value' tag strings. Empty tags -> empty metadata
    (no empty langfuse_tags key).
    """
    metadata: dict[str, str | list[str]] = dict(tags)
    if tags:
        metadata["langfuse_tags"] = [f"{key}:{value}" for key, value in tags.items()]
    return metadata


def debug_enabled() -> bool:
    """Read AGENT_DEBUG env var. Default is ON (True).

    Return False only when the value (lowercased, stripped) is one of:
    "0", "false", "". Otherwise True. If unset, return True (default on).
    """
    if "AGENT_DEBUG" not in os.environ:
        return True

    value = os.environ.get("AGENT_DEBUG", "").strip().lower()
    if value in ("0", "false", ""):
        return False
    return True


def _oneline(text: str, limit: int = 120) -> str:
    """Collapse ALL runs of whitespace (including newlines) into single spaces.

    Strip ends, and if longer than limit, truncate to limit chars and append
    a unicode ellipsis character. Private helper.
    """
    # Replace all runs of whitespace with a single space
    collapsed = re.sub(r"\s+", " ", text).strip()

    if len(collapsed) > limit:
        return collapsed[:limit] + "…"
    return collapsed


def format_run_start(question: str, db_id: str) -> str:
    """Pure. One header line announcing the run, logged before any node executes.

    Gives the trace a clear starting marker: which db and which question the agent
    is about to answer, so the generate/execute/verify/revise lines that follow have
    context. Leads with two blank lines and a horizontal rule (RUN_SEPARATOR) so
    each complete trace is clearly divided from the previous one in the console.

    Both fields are passed through _oneline so a value containing newlines can't
    forge extra log lines (log injection) - the only newlines are the structural
    separators here.
    """
    return f"\n\n{RUN_SEPARATOR}\n❓ [{_oneline(db_id, limit=64)}] {_oneline(question)}"


def format_step(node: str, update: dict) -> str:
    """Pure. One emoji-prefixed summary line for a node update, led by a blank line
    so the steps within a question are visually separated in the console.
    """
    return "\n" + _format_step_line(node, update)


def _format_step_line(node: str, update: dict) -> str:
    """Maps a LangGraph node's returned update dict to one emoji-prefixed line.

    Behavior by node name:
    - "generate_sql" → 🧭
    - "revise" → 💭
    - "execute" → 📊
    - "verify" → 🔎
    - any other node → •
    """
    if node == "generate_sql":
        return f"🧭 generate_sql (iter {update.get('iteration')}) → {_oneline(update.get('sql', ''))}"

    if node == "revise":
        return f"💭 revise (iter {update.get('iteration')}) → {_oneline(update.get('sql', ''))}"

    if node == "execute":
        execution = update.get("execution")
        if execution is None:
            render_text = "no execution result"
        else:
            try:
                render_text = execution.render()
            except Exception:  # noqa: BLE001 - rendering an arbitrary object, fall back to str
                render_text = str(execution)
        return f"📊 execute → {_oneline(render_text)}"

    if node == "verify":
        if update.get("verify_ok"):
            return "🔎 verify → ok=true"
        return f"🔎 verify → ok=false: {_oneline(update.get('verify_issue', ''))}"

    return f"• {node} → {_oneline(str(update))}"


def configure_logging(debug: bool | None = None) -> None:
    """Idempotent. Configure the module logger for per-step trace output.

    If debug is None, use debug_enabled(). Ensure exactly one StreamHandler,
    set a simple formatter, and set log level. Also set propagate=False
    to avoid doubling by the root logger.
    """
    if debug is None:
        debug = debug_enabled()

    # Idempotent: only attach a handler the first time, so repeat calls don't stack them.
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

    logger.setLevel(logging.INFO if debug else logging.WARNING)
    logger.propagate = False
