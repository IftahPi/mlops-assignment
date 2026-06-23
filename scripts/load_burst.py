"""Concurrent load burst against the vLLM OpenAI-compatible endpoint.

Drives sustained concurrent chat-completions so the serving stack does real
work - used to make the Grafana panels (Phase 2) react and as a quick,
dependency-light load source while developing. The graded Phase 6 SLO
measurement uses the dedicated harness under load_test/; this is the simple
"fire traffic and watch the dashboard" tool.

Reads the endpoint from the same env vars the agent uses:
  VLLM_BASE_URL (default http://localhost:8000/v1), VLLM_MODEL.
Prompts are drawn from load_test/perf_pool.jsonl.

Examples:
  uv run python scripts/load_burst.py                       # 120s @ 48 concurrency
  uv run python scripts/load_burst.py -d 150 -c 384 -m 512  # heavy burst
"""
import argparse
import json
import os
import random
import threading
import time

from openai import OpenAI

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_POOL = os.path.join(REPO, "load_test", "perf_pool.jsonl")


def load_prompts(path: str) -> list[str]:
    prompts: list[str] = []
    with open(path) as handle:
        for line in handle:
            try:
                prompts.append(json.loads(line)["question"])
            except (json.JSONDecodeError, KeyError):
                continue
    return prompts or ["Write a short SQL query that counts rows in a table."]


def run_burst(duration: int, concurrency: int, max_tokens: int, pool: list[str]) -> None:
    client = OpenAI(
        base_url=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
        api_key=os.environ.get("OPENAI_API_KEY", "dummy-local-vllm"),
    )
    model = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
    stop_at = time.time() + duration
    counts = {"sent": 0, "ok": 0, "err": 0}
    lock = threading.Lock()

    def worker() -> None:
        while time.time() < stop_at:
            question = random.choice(pool)
            with lock:
                counts["sent"] += 1
            try:
                client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": f"Answer concisely: {question}"}],
                    temperature=0.7,
                    max_tokens=max_tokens,
                )
                key = "ok"
            except Exception:  # noqa: BLE001 - count failures, keep firing
                key = "err"
            with lock:
                counts[key] += 1

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(concurrency)]
    print(
        f"burst: {duration}s @ concurrency {concurrency}, max_tokens {max_tokens}, "
        f"pool={len(pool)} prompts",
        flush=True,
    )
    started = time.time()
    for thread in threads:
        thread.start()
    while time.time() < stop_at:
        time.sleep(5)
        print(f"  t={int(time.time() - started)}s {counts}", flush=True)
    for thread in threads:
        thread.join(timeout=30)
    print(f"done: {time.time() - started:.0f}s {counts}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-d", "--duration", type=int, default=120, help="seconds (default 120)")
    parser.add_argument("-c", "--concurrency", type=int, default=48, help="worker threads (default 48)")
    parser.add_argument("-m", "--max-tokens", type=int, default=256, help="max output tokens (default 256)")
    parser.add_argument("--pool", default=DEFAULT_POOL, help="prompts jsonl (default load_test/perf_pool.jsonl)")
    args = parser.parse_args()
    run_burst(args.duration, args.concurrency, args.max_tokens, load_prompts(args.pool))


if __name__ == "__main__":
    main()
