# MLOps Assignment — Report

Text-to-SQL LangGraph agent + observability over BIRD-bench, served on an H100 with vLLM.

**Environment (all authoritative numbers below come from this setup):**

| | |
|---|---|
| Hardware | 1× NVIDIA H100 80GB HBM3 (Nebius VM) |
| Model | `Qwen/Qwen3-30B-A3B-Instruct-2507` — BF16, MoE (~30B total / ~3B active params) |
| Serving | vLLM 0.10.2, OpenAI-compatible API on `:8000` |
| Observability | Langfuse (`:3001`) + Prometheus (`:9090`) + Grafana (`:3000`) via `docker compose` |

---

## 1. Serving configuration (Phase 1)

### Launch command

```bash
# scripts/start_vllm.sh
MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"

exec uv run python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --max-num-batched-tokens 32768
```

### Changes from the example script

The shipped `start_vllm.sh` launched with only `--model`, `--host`, `--port`. I added three flags:

| Flag | Value | Justification |
|---|---|---|
| `--max-model-len` | `32768` | **Required to start at all.** The model's native context is 262,144 tokens; vLLM sizes the KV cache to `max-model-len`, and at 262K that allocation exceeds 80 GB and vLLM refuses to start. 32K fits comfortably and is far more than text-to-SQL needs (schema + question + SQL is a few thousand tokens). |
| `--gpu-memory-utilization` | `0.90` | Use up to 90% of the 80 GB for weights + KV cache. Matches vLLM's default; set explicitly so the config is self-documenting. Verified at runtime: **74.4 GB used ≈ 0.90 × 80 GB**. |
| `--max-num-batched-tokens` | `32768` | Per-scheduler-step token budget. Set equal to `max-model-len` so one full-length prompt prefills in a single step; also bounds batch size for the Phase 6 load test. **A knob I expect to revisit during SLO tuning.** |

No quantization and no tensor parallelism (single H100 → TP=1, full BF16 weights). Model id, host, and port unchanged.

### Prerequisites outside the launch script

Two environment fixes were needed before vLLM would start (documenting for reproducibility):

1. **Pinned `transformers<5.0.0`** (resolved 5.9.0 → 4.57.6). vLLM 0.10.2 fails to initialize against transformers 5.x. Added as an explicit dependency in `pyproject.toml`, then `uv lock && uv sync`.
2. **Installed `python3.12-dev`.** Torch Inductor JIT-compiles a CUDA helper at startup and needs `Python.h`; without the dev headers the engine core dies with `fatal error: Python.h: No such file or directory`.

### Verification

- `GET /health` → **200**; GPU **74.4 GB** used, idle util 0%.
- Prometheus scrapes vLLM `/metrics` → 200 (Phase 2 pipeline live).
- Full agent end-to-end confirmed: the formula_1 coordinates question generated duplicate rows → `verify ok=false` → `revise` added `DISTINCT` → `verify ok=true` (the loop fixing a real case on the authoritative backend).

**Manual smoke test** — first 5 questions from `evals/eval_set.jsonl` fired *one-shot* at the served model (single generate, no agent loop), then each query executed against its sqlite:

| # | db_id | question | generated SQL (truncated) | executed? |
|---|---|---|---|---|
| 1 | `formula_1` | What is the coordinates location of the circuits for Australian… | `SELECT lat, lng FROM circuits JOIN races ON … WHERE races.name = 'Australian Grand Prix'` | OK: 11 rows |
| 2 | `superhero` | List down Ajax's superpowers. | `SELECT sp.power_name FROM superhero s JOIN hero_power hp … JOIN superpower sp …` | OK: 5 rows |
| 3 | `california_schools` | List the top five schools, by descending order, from the highest… | `SELECT s.NCESSchool FROM schools s JOIN frpm f … ORDER BY f.\`Enrollment (Ages 5-17)\` DESC LIMIT 5` | OK: 5 rows |
| 4 | `financial` | What is the average number of crimes committed in 1995 in regions… | `SELECT AVG(d.A14) … FROM district d WHERE d.A14 > 4000 AND EXISTS (…)` | OK: 1 row |
| 5 | `financial` | How many male clients in 'Hl.m. Praha' district? | `SELECT COUNT(*) FROM client c JOIN district d … WHERE c.gender = 'm' AND d.A2 = 'Hl.m. Praha'` | OK: 1 row |

**5/5 produced executable SQL** — the served model is healthy and generates reasonable SQLite. Notably, two of these *execute cleanly but are wrong*: #1 returns 11 duplicate rows (missing `DISTINCT`) and #5 filters `gender = 'm'` where the stored value is `'M'` (SQLite is case-sensitive). This is the **"200 OK ≠ correct"** gap precisely — and exactly what the agent's verify→revise loop (Phase 3/5) is built to catch.

---

## 2. Baseline eval results (Phase 5)

**Setup.** Execution accuracy: run the agent's final SQL and the gold SQL against the target sqlite,
compare canonicalized row sets (sorted, column-name case ignored; duplicates *not* deduped). 30 curated
BIRD questions, agent → H100 vLLM (`Qwen/Qwen3-30B-A3B-Instruct-2507`), `MAX_ITERATIONS=3`. Harness:
`evals/run_eval.py` → `results/eval_baseline.json`. 31 s wall clock, 0 agent errors.

| Metric | Value |
|---|---|
| Overall pass rate | **10/30 = 0.333** |
| Per-iteration pass rate (carry-forward) | **[0.30, 0.367, 0.333]** |
| Loop fired (≥2 iterations) | 12/30 |
| fail→pass flips | 1 (0 regressions vs iter-0) |

**Commentary — the loop does real work, but the second revise doesn't pay.** The per-iteration curve is
**not flat** (so it is *not* the README's "architecture doing nothing" case): iter-0 `0.30` → iter-1
`0.367`, +2 questions — the verify→revise loop earns its keep on the first revise. But the curve **peaks
at iter-1 then dips** (`0.367 → 0.333`): the third attempt re-breaks one question that was correct after
the first revise, so the final pass rate (`0.333`) beats iter-0 by only 1 question.

**Noise caveat.** ~0.033 (one question on n=30) is the noise floor — Qwen3-30B-A3B is a mixture-of-experts
and hosted greedy decoding is not bit-reproducible. The +1 *final* lift is within noise, but the +2
iter-0→iter-1 jump and the non-flat shape are real signal that the loop works. The peak-then-dip strongly
suggests **`MAX_ITERATIONS=2` is the sweet spot** — revisited as a knob in Phase 6. (Pre-H100 Nebius
experiments exploring this — verify-sees-schema, cap depth, revise temperature — are in
`planning/experiment-findings.md`.)

---

## 3. Hitting the SLO (Phase 6)

**SLO:** p95 end-to-end **agent** latency < 5 s at 10 RPS over a 5-minute window (1 RPS = 1 full agent
run). The SLO metric's source of truth is the load driver (`load_test/driver.py`), which times the full
`/answer` round-trip.

**Observability gap (a finding in itself):** the Grafana dashboard instruments **vLLM only** —
per-LLM-call latency, KV, queue. It has **no panel for end-to-end agent latency**, so it confirms
serving health but cannot display the SLO metric. We therefore read the SLO off the driver and use the
dashboard to locate *where* the time is *not* going. (Agent-level latency/queue instrumentation → §5.)

**Baseline (MAX_ITERATIONS=3, + the db bug, 10 RPS, 150 s):**
- p95 **86.5 s** (p50 29.9 s, p99 95.9 s, max 117.9 s) — **~17× over the 5 s SLO.**
- achieved only **7.14 RPS** (could not sustain 10); **507 / 1500 ok** (http 127, client 586, timeouts 280).
- Dashboard during the run: vLLM `running` peaked ~33, KV ~22%, **`waiting` stayed 0** — vLLM never
  saturated. So the latency is **not** in serving; it is upstream (agent-server queuing + the sequential
  call chain).

### Iteration log

**Iteration 1 — fix the correctness bug so the measurement is valid**
- **saw:** the SLO didn't just fail, it failed catastrophically (p95 86.5 s), and only 507/1500 requests
  succeeded — a suspiciously high error rate (HTTP 400s + "unable to open database file"). Errors return
  instantly, so the latency distribution itself is untrustworthy.
- **hypothesized:** a correctness bug, not (yet) a perf limit — the per-request `available_dbs()`
  filesystem glob hits the open-FD limit under load, so `Path.exists()` returns False → valid dbs
  rejected with 400, and FD exhaustion surfaces as "unable to open database file".
- **changed:** snapshot the valid-db set once at startup (`VALID_DBS`) instead of globbing on every
  request — keeps the path-traversal guard, removes the per-request filesystem dependency.
- **result:** _<!-- fill after re-run + screenshot 2 -->_

---

## 4. Agent value (did the loop help?)

**The verify → revise loop is wired and bounded (Phase 3).** Firing all 30 eval questions through the
agent on the H100 (`MAX_ITERATIONS=3`):

- **13/30 triggered at least one revise** — verify rejected the first attempt and the loop
  re-generated. The textbook fix-on-loop case: `formula_1` *"coordinates of the Australian Grand Prix
  circuit"* returned 11 duplicate rows → verify flagged the missing `DISTINCT` → revise added it →
  re-verify passed (`generate → verify(false) → revise → verify(true)`, ends at iteration 2).
- **9/30 exhausted the cap** — three iterations (generate + 2 revises), all three verifies `ok=false`,
  then `route_after_verify` terminated at `iteration == MAX_ITERATIONS` instead of looping forever.
  Concrete cap-eater: `financial` — *"the average number of crimes committed in 1995 in regions where
  the number exceeds 4000 and the region has accounts opened from 1997"* — a multi-condition aggregate
  the model never satisfied the verifier on. The Langfuse trace shows `execute (3/3)` and a root
  `verify_ok=false`. This is the cap doing its job: an unfixable question costs a bounded 3 iterations,
  not an infinite loop.

<!-- TODO (Phase 5, authoritative): cite the per-iteration pass rate (carry-forward) to quantify
     whether the loop raises ACCURACY, not just whether it fires. Source: planning/experiment-findings.md. -->
_Whether the loop raises **accuracy** (not just fires) is the per-iteration pass rate — filled from the
Phase 5 eval._

---

## 5. What I'd do with more time

<!-- TODO: be specific (not "add Kubernetes"). Candidates from experiment-findings.md §5:
     - multi-seed eval to beat the MoE noise floor (~0.033 on n=30 at temp 0);
     - let generate/revise inspect distinct column values (the 'm' vs 'M' class can't be fixed blind);
     - early-stop / confidence gate so the loop doesn't over-revise (the MAX=5 oscillation);
     - LLM-as-judge on the trace for near-misses where execution-accuracy is too strict. -->

_To be filled._
