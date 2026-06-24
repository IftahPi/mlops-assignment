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

<!-- TODO (authoritative, on H100): regenerate results/eval_baseline.json multi-seed.
     Report overall pass rate + per-iteration pass rate (carry-forward) + brief commentary.
     Source material / methodology: planning/experiment-findings.md (currently Nebius, non-authoritative). -->

_To be filled from `results/eval_baseline.json` (H100). Methodology and the Nebius dry-run story are in `planning/experiment-findings.md`._

---

## 3. Hitting the SLO (Phase 6)

<!-- TODO (H100): baseline latency vs SLO; iteration log entries of the form
     "saw X → hypothesized Y → changed Z → result was W"; final numbers; honest verdict.
     Pair: screenshots/grafana_before.png + grafana_after.png; results/eval_after_tuning.json. -->

_To be filled from the H100 load test + tuning iterations._

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
