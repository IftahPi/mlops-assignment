"""Interactive single-question runner for the text-to-SQL agent.

Streams the per-step trace (generate → execute → verify → revise) to the
console via the agent's logging, then prints the final SQL and rows.

Usage:
    python -m agent.run_cli --db formula_1 "How many drivers are there?"
    python -m agent.run_cli --db formula_1 --quiet "..."   # suppress the step trace
"""
import argparse

from dotenv import load_dotenv

# Load .env BEFORE importing agent.graph: graph.py reads VLLM_BASE_URL / VLLM_MODEL
# at import time, so the env must be populated first (same ordering as server.py).
load_dotenv()

from agent.graph import AgentState, graph  # noqa: E402
from agent.trace import configure_logging  # noqa: E402


def main() -> None:
    """Run the agent on a single question and display the result."""
    parser = argparse.ArgumentParser(
        description="Run the text-to-SQL agent on a single question."
    )
    parser.add_argument(
        "question",
        type=str,
        help="Natural language question to convert to SQL.",
    )
    parser.add_argument(
        "--db",
        type=str,
        required=True,
        help="Database ID (e.g., 'formula_1').",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-step trace output.",
    )

    args = parser.parse_args()

    # Configure logging based on --quiet flag (.env already loaded at import).
    configure_logging(debug=not args.quiet)

    # Invoke the graph with the question and database ID
    final = graph.invoke(AgentState(question=args.question, db_id=args.db))

    # Pretty-print the result
    print()
    print(f"Final SQL (iteration {final.get('iteration')}):")
    print(final.get("sql", "(no SQL generated)"))
    print()

    execution = final.get("execution")
    if execution is not None and getattr(execution, "ok", False):
        row_count = getattr(execution, "row_count", "?")
        print(f"Result: {row_count} rows")
        print(execution.render())
    else:
        error_msg = getattr(execution, "error", "agent produced no execution result")
        print(f"Error: {error_msg}")


if __name__ == "__main__":
    main()
