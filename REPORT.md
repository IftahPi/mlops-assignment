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

**Observability gap (a finding in itself):** the Grafana dashboard *initially* instrumented **vLLM
only** — per-LLM-call latency, KV, queue. It had **no panel for end-to-end agent latency**, so it
confirmed serving health but could not display the SLO metric. We *initially* read the SLO off the
driver and used the dashboard only to locate *where* the time is *not* going. **We later closed this
gap** — see *"Tooling: make the SLO visible"* below.

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
- **result:** the targeted metric moved — **HTTP 400s went 685 → 0** (status codes are now clean 200/500
  only). But the **SLO did not improve; p95 rose 86.5 s → 116.6 s.** The ~685 requests formerly rejected
  *instantly* now actually run, adding real load to the already-saturated agent queue (failures shift to
  500s/timeouts). Textbook *"a metric improved and the SLO didn't"* — and it **confirms the bottleneck is
  agent-side queuing, not the bug.** → Iteration 2.

**Iteration 2 — cut the per-request work (`MAX_ITERATIONS` 3 → 2)**
- **saw:** with the bug fixed, p95 was 116.6 s, and the Phase-5 per-iteration eval curve peaks at
  iter-1 then dips — i.e. the *third* attempt adds latency for essentially no accuracy.
- **hypothesized:** each request is a chain of sequential LLM calls, so dropping the cap 3 → 2 removes
  one generate→verify→revise round per request — less service time per run, and the peak-then-dip says
  iteration-2 doesn't pay for itself anyway.
- **changed:** `AGENT_MAX_ITERATIONS=2`.
- **result:** p95 **116.6 → 76.7 s (−34%)** and success rose to **1304 / 1500 (87%)** (400s still 0).
  **But achieved RPS was unchanged at 7.14** and we are still **~15× over the 5 s SLO.** Latency
  improved; the *throughput ceiling did not move* — so the iteration cap is **not** the ceiling. → Iteration 3.

**Iteration 3 — diagnose the ceiling: measure the uncontended floor**
- **saw:** across baseline, Iteration 1 and Iteration 2, achieved RPS was pinned at **7.14 regardless of
  the knob**, with p95 in the tens of seconds. A throughput ceiling that ignores tuning smells like a
  concurrency limit — but it could equally mean each run is simply slow.
- **hypothesized:** maybe the SLO is *architecturally impossible* — if a single agent run (a chain of
  sequential LLM calls) already exceeds 5 s **uncontended**, no amount of tuning reaches p95 < 5 s.
- **changed (diagnostic, not a perf change):** fired 6 real eval questions **strictly sequentially**
  (one at a time, zero overlap) and timed each end-to-end run.
- **result:** the floor is **0.4–1.6 s per run** — *including* the revise loop firing (iters=2) — well
  **under** the 5 s SLO. **Hypothesis refuted:** the SLO is reachable; the 77 s p95 is **~50× queue
  inflation, not compute.** Bottleneck localized to the **agent server**: `/answer` is a *sync*
  endpoint, so it runs in Starlette's ~40-thread pool, capping throughput at ≈ `40 ÷ (per-run latency
  under load) ≈ 7 RPS` while **vLLM sits idle** (KV ~22%, `waiting` 0). → Iteration 4: raise agent
  concurrency.

**Tooling (between Iterations 3 and 4) — make the SLO visible (close the observability gap)**
- **saw:** every iteration's verdict so far was read off the **load driver's** client-side percentiles,
  because the Grafana dashboard instrumented **vLLM only** — the metric we actually grade against (p95
  end-to-end `/answer` latency) had **no panel**. During a run we could see vLLM was *idle* but not the
  agent-side number we were chasing; we were effectively blind to the SLO on the dashboard.
- **changed:** instrumented the agent server with a Prometheus histogram
  `agent_request_duration_seconds` (buckets straddling the SLO: …2, **3, 4, 5**, 10…), exposed a
  `/metrics` endpoint, added a Prometheus scrape job for the agent (`:8001`), and added a Grafana panel
  *"Agent end-to-end latency (the SLO)"* — p50/p95/p99 with a red 5 s threshold line — pinned to the top
  of the dashboard.
- **result:** the SLO is now a **first-class, live** panel, verified end-to-end (a 5-request smoke
  produced a queryable p95). The **driver stays the source of truth** — the panel's `histogram_quantile`
  is a bucket-interpolated *approximation* (with only a few samples it reads high; the 3 s/4 s buckets
  sharpen it near the boundary) — but the dashboard now shows the SLO in real time, so the remaining
  concurrency iterations can be watched live and the before/after pair can show p95 crossing the 5 s
  line. *(This also delivers the "agent-latency instrumentation" item once parked in §5.)*

**Iteration 4 — raise the agent's concurrency (threadpool 40 → 100)**
- **saw:** the floor diagnostic (Iteration 3) showed a single run costs ~1 s while vLLM sits idle, and
  `/answer` is a **sync** endpoint → it runs in Starlette's **40-thread pool**, so each in-flight
  request holds a thread for its whole multi-call duration.
- **hypothesized (stated before the run):** the binding constraint is the **agent's 40-thread pool, not
  vLLM.** Raising it (40 → 100) will drain the agent-side queue and **drop p95 sharply.** *Falsifier:*
  if vLLM were the real limit, raising threads would just **move the queue into vLLM** — KV → ~100%,
  preemptions appear, `waiting` lifts off 0 — and p95 wouldn't improve.
- **changed:** added an env-tunable threadpool limiter (`AGENT_MAX_THREADS`, lifespan hook resizing
  anyio's `total_tokens`); ran the agent at **`AGENT_MAX_THREADS=100`**. Both runs are full 300 s @ 10
  RPS. (This restart also activated the sharper 3 s/4 s SLO buckets.)
- **result — the hypothesis was confirmed *and* we overshot; the answer landed in *both* columns:**

  | Metric | Before (40 threads) | After (100 threads) | Δ |
  |---|---|---|---|
  | p50 | 54.7 s | **6.45 s** | **−88%** |
  | **p95 (SLO)** | 67.7 s | **20.1 s** | **−70%** |
  | p99 | 72.2 s | 26.9 s | −63% |
  | achieved RPS | 8.33 | **8.33** | unchanged |
  | vLLM KV cache | ~22% | **~80–100%** | saturated |

  The agent threadpool **was** a major cap ✅: p50 fell 8.5×, p95 fell 3.4×, and the in-system backlog
  collapsed from ~456 to ~54 requests (Little's law: `L = throughput × latency`). **But 100 threads
  pushed the bottleneck *into* vLLM** 🔴 — exactly the falsifier: **KV jumped 22% → ~80–100% with
  preemptions firing (~0.05/s)**, vLLM's own e2e latency spiked to *minutes*, and the SLO panel ran
  flat-low (~6–20 s) for most of the window **then spiked at the end** — i.e. **100 threads is
  *unstable*: it overshoots vLLM's KV headroom and degrades as the cache fills.** Achieved RPS stayed
  pinned at 8.33 because the limiter is now vLLM's KV-bound capacity, not the threads. **Net: p95 cut
  ~70% but still ~4× over the 5 s SLO — and we've relocated the ceiling from the agent to vLLM's KV
  cache.** Before/after evidence: **screenshot 5** (before) vs **image 6** (after — KV pinned near
  100%, preemptions visible).

  **Why KV saturated — and why better caching is *not* the fix.** vLLM's automatic prefix caching is on
  (`enable_prefix_caching=True`) and *working*: measured hit rate **≈52%** (`prefix_cache_hits /
  queries` = 121M / 232M), because every prompt shares the system + schema prefix for a given `db_id`.
  So the shared prefix is already largely deduped/cached — the blocks that *filled* KV are the
  **per-request *unique* tokens + the growing *decode* KV, × ~100 concurrent sequences**, which prefix
  caching does not bound. (Cache-aware *routing* doesn't apply here — a single vLLM instance, no router
  to steer shared-prefix requests between replicas.) Also, honestly: my pre-run guess of ~65–70% KV was
  a crude *linear* extrapolation from ~22%@~33-concurrent; near the saturation cliff it is
  **superlinear** — as latency rises, sequences stay resident longer (Little's law), so KV climbs faster
  than the thread count. Hence ~80–100%, not ~66%.

  → Iteration 5: dial threads to the concurrency that sits *just below* the KV-preemption knee (between
  the 40 that under-uses vLLM and the 100 that saturates it) for a *stable* p95, and/or cut KV pressure
  per request (cap generation `max_tokens` / `--max-num-seqs`) to raise that knee.

**Iteration 5 — kill the preemptions: cap admission + generation (KV levers)**
- **saw:** Iteration 4 hit p95 20.1 s but *unstable* — vLLM KV pinned ~80–100% with preemptions firing
  and a tail that spiked late in the window. The instability, not the agent, was now the problem.
- **hypothesized (stated before the run):** the late-window spike is **KV-preemption thrash**. Bounding
  KV two ways — admission (`--max-num-seqs 64`, so vLLM never tries to co-run more sequences than it has
  comfortable headroom for) and decode (`AGENT_MAX_TOKENS=512`, so no runaway generation hogs KV) — plus
  backing threads off 100 → **70**, will hold **KV below ~75%, preemptions ≈ 0, and a *stable* p95**.
  *Falsifier:* if admission-capping just relocates the backlog from vLLM's run-queue into its *waiting*
  queue, preemptions vanish but tail latency does **not** improve (queueing delay replaces preemption).
- **changed:** `--max-num-seqs 64` (`scripts/start_vllm.sh`), `AGENT_MAX_TOKENS=512` wired into the
  `llm()` factory (`agent/graph.py`), agent run at `AGENT_MAX_THREADS=70`. One coordinated step, full
  300 s @ 10 RPS. (Committed `4e314fa`.)
- **result — the falsifier fired: hypothesis confirmed *on its own terms* but the SLO regressed:**

  | Metric | Iter 4 (threads 100, no KV cap) | Iter 5 (threads 70, max-seqs 64, max-tok 512) | Δ |
  |---|---|---|---|
  | p50 | 6.45 s | 7.09 s | +10% |
  | **p95 (SLO)** | **20.1 s** | **30.3 s** | **+50% (worse)** |
  | p99 | 26.9 s | 36.6 s | +36% |
  | achieved RPS | 8.33 | **9.19** | **+10%** |
  | vLLM preemptions | many (KV 80–100%) | **0** (KV bounded) | ✅ eliminated |
  | ok / http-500 | 2571 / 372 | 2583 / 376 | unchanged |

  The KV lever **did exactly what it promised** ✅: **preemptions → 0**, KV no longer saturates, and
  goodput actually *rose* (8.33 → **9.19 RPS** — no preempt-recompute waste, so every admitted token is
  useful work). **But the SLO got *worse*** 🔴: p95 20→30 s. The falsifier was right — capping admission
  at 64 didn't remove the backlog, it **moved it from vLLM's *preempting* run-queue into its *waiting*
  queue**, and the admission queue at 64 is a *deeper* wait than the unbounded-but-preempting regime. **We
  traded preemption thrash for a longer queueing delay, and the queueing delay was bigger.** Evidence:
  **screenshot 7** (KV flat and well off 100%, preemptions panel flat at 0 — visibly stable, yet the SLO
  panel higher than image 6).

  **An honest aside on the 12.5% HTTP-500s.** They are **not** an Iteration-5 regression: Iter 4 already
  had **372** and Iter 5 has **376**, spread *evenly* across the run (not a startup artifact). The agent
  surfaces them from `graph.invoke` → `HTTPException(500)` (`server.py:116`); under sustained ~70-way
  concurrency the LLM client errors out (httpx pool / vLLM API-server connection pressure at 70 threads ×
  3 sequential calls), independent of the KV levers. **This is the next thing to chase** — see §5.

  **Verdict on Phase 6.** Across five iterations p95 fell **86 s → ~20–30 s** (and the system is now
  *stable* — preemptions gone), but the **5 s SLO @ 10 RPS is not reachable by tuning alone**: this
  workload is **3 sequential LLM calls per request**, vLLM tops out near **8–9 RPS** of that, and the
  remaining latency is *queueing for the model*, not agent overhead. The two regimes we can pick between
  are now clear — **Iter 4 (unstable, lower p95)** vs **Iter 5 (stable, higher p95, higher goodput)** —
  and neither is a config away from 5 s. Closing the gap needs an **architectural** change: **fewer LLM
  calls per request** (the dominant lever — fold verify into generate, or only revise on a real execution
  error), async pipelining, a smaller/faster model, or speculative decoding. → §5.

### Final configuration (the rollback decision)

The two regimes were a genuine choice, so we picked on the **graded metric — p95.** Iteration 4 holds the
lower p95 (**20.1 s** vs Iteration 5's 30.3 s), so we **reverted the Iteration 5 KV levers** and restored
Iteration 4 as the final configuration (commit `130771a` reverts `4e314fa`):

| Setting | Final value | Why |
|---|---|---|
| `AGENT_MAX_ITERATIONS` | **2** | Iteration 2 — banks the +2 first-revise accuracy lift, drops the iter-2 regression. |
| `AGENT_MAX_THREADS` | **100** | Iteration 4 — the single change that moved p95 (67 → 20 s) by draining the agent-side queue. |
| vLLM `--max-num-seqs` | *(default, unset)* | Iteration 5 admission cap reverted — capping at 64 deepened the *waiting* queue and **raised** p95. |
| `max_tokens` | *(unset)* | Iteration 5 decode cap reverted — the KV bound wasn't worth the SLO regression, and full generation removes any truncation risk to answer quality. |

We accept Iteration 4's known cost: KV runs hot (~80–100%) with occasional preemptions, so this p95 is
**lower but less stable** than Iteration 5's. Since the SLO is graded on **p95**, the lower-p95 regime
wins; the instability is documented, not hidden.

**Did the rollback cost accuracy? No — it improved.** Re-ran the full 30-question eval against this final
config (`AGENT_MAX_ITERATIONS=2`, reverted levers, agent → H100 vLLM) → `results/eval_after_tuning.json`
(41.5 s wall, 0 agent errors):

| | Baseline (pre-tuning, `MAX_ITERATIONS=3`) | **Final config** (`MAX_ITERATIONS=2`, reverted) |
|---|---|---|
| Overall pass rate | 0.333 (10/30) | **0.400 (12/30)** |
| Per-iteration (carry-forward) | [0.30, 0.367, 0.333] | **[0.333, 0.400]** |

**Quality survived — it rose +2 questions (0.333 → 0.400).** This is the Phase-5 prediction playing out:
the loop's *first* revise still earns its keep (iter-0 0.333 → iter-1 **0.400, +2**), and capping at
`MAX_ITERATIONS=2` banks that lift while dropping baseline's iter-2 regression. The reverted KV levers
were pure serving/concurrency knobs with no path to correctness (if anything, removing the 512-token cap
eliminates a truncation risk), so latency stability changed but accuracy did not suffer. The +2 sits just
above the ~0.033 single-question MoE noise floor, so the honest read is **quality held and modestly
improved, with zero regression** from the Phase-6 tuning.

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

**But firing isn't helping — the per-iteration pass rate is the proof.** Carry-forward accuracy across
iterations is **[0.30, 0.367, 0.333]** (Phase 5, n=30): generate-alone (iter-0) scores **0.30**; the
**first revise lifts it to 0.367 (+2 questions)**; the second revise *dips* to 0.333 (it re-breaks one
question that was already correct). So the loop **does** raise accuracy — but only the **first** revise
pays. The curve being **non-flat** is what rules out the README's "architecture doing nothing" case (a
flat line would mean revise never changes an outcome); the **peak-then-dip** is precisely why the final
config caps at **`MAX_ITERATIONS=2`** — it banks the +2 lift and skips the iter-2 regression. The +2 at
the peak sits above the ~0.033 single-question noise floor (n=30, temp-0 MoE), so the loop earns its
keep — and the eval told us not just *that* it helps but *how much* loop to keep.

---

## 5. What I'd do with more time

In priority order, each tied to a finding above:

1. **Cut the LLM calls per request — the dominant SLO lever.** Phase 6 proved the residual latency is
   *queueing for the model*, not agent overhead, and the workload is **3 sequential calls/request**. Fold
   the verifier into the generator (one structured response that emits SQL **and** a self-check), and
   spend a separate `revise` call **only when execution actually errors or returns empty rows** — not on
   every run. That removes 1–2 sequential model calls from the hot path, the one change with a real path
   to p95 → 5 s. Re-measure with the same driver to confirm.
2. **Chase the ~12.5% HTTP-500s** (`server.py:116`, ~376/3000, evenly spread). Reproduce the LLM-client
   failures under ~70-way concurrency, add a bounded retry + explicit httpx connection-pool sizing on the
   client, and read the agent `/metrics` to confirm whether the cause is client-pool exhaustion or
   vLLM's API-server accept queue. That's real goodput currently thrown away.
3. **Beat the eval noise floor with multi-seed eval.** n=30 at temp 0 on an MoE has a ~0.033 (one
   question) noise floor, so the +1 *final* lift is inside the noise. Run k seeds (or a small temperature
   sweep) and report mean ± std, so the loop's accuracy claim is statistically honest rather than a
   single coin flip.
4. **Let generate/revise see distinct column values.** The `gender = 'm'` vs stored `'M'` bug (and
   enum/category mismatches generally) **cannot be fixed blind** — give the agent a cheap
   `SELECT DISTINCT <col> LIMIT k` probe on filtered columns so it conditions on real stored values
   instead of guessing their spelling/case.
5. **Make `/answer` async + add a faster decode path.** A sync endpoint holds a threadpool slot across
   the whole sequential chain (the Iteration 4 bottleneck); going async frees the slot during model
   waits, and pairing it with speculative decoding (or a small draft model) cuts per-call decode time —
   both attack the queueing delay directly, complementing #1.
