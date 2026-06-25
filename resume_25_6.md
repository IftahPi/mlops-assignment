# RESUME — 2026-06-25 (Module 3 MLOps assignment, Phases 6–7 wrap-up)

**One-line status:** Phases 1–5 done. Phase 6 tuning complete and **reverted to the final
configuration**. Phase 7 report 95% written. **The only work left is blocked on the VM being down** —
one eval run + slotting one number into REPORT + zipping. Everything else is committed.

Authoritative context lives in:
- `REPORT.md` — the graded writeup (read it; it's near-final).
- `README.md` — the assignment spec; **§ "Final deliverables"** (table) and **§ "Grading"** (rubric).
- Memory index: `~/.claude/projects/.../memory/MEMORY.md` (esp. `m3-mlops-assignment`,
  `let-user-verify-before-testing`, `env-install-policy`).

---

## 1. The FINAL configuration (definitive — this is what to run/ship)

Iteration 5's KV levers regressed the SLO, so they were **reverted** (commit `130771a` reverts
`4e314fa`). Final = the Iteration 4 regime:

| Setting | Final value | Set where |
|---|---|---|
| `AGENT_MAX_ITERATIONS` | **2** | env on the agent process (Phase-6 Iteration 2) |
| `AGENT_MAX_THREADS` | **100** | env on the agent process (Phase-6 Iteration 4) |
| `AGENT_MAX_TOKENS` | **unset** | reverted — `agent/graph.py` no longer passes `max_tokens` |
| vLLM `--max-num-seqs` | **unset/default** | reverted — `scripts/start_vllm.sh` |
| vLLM flags kept | `--max-model-len 32768 --gpu-memory-utilization 0.90 --max-num-batched-tokens 32768` | `scripts/start_vllm.sh` |

Rationale (full version in REPORT §3 "Final configuration"): we picked on the graded metric **p95** —
Iter4 = 20.1 s (lower) vs Iter5 = 30.3 s. Iter4's cost is hot KV / occasional preemptions (less stable);
accepted and documented.

---

## 2. What was DONE this session (all committed on branch `phase3-5-local`)

| Commit | What |
|---|---|
| `130771a` | **Task 3:** revert Iter5 KV levers (graph.py `max_tokens`, start_vllm.sh `--max-num-seqs`) → Iter4 is final |
| `5c65fcb` | **Task 7:** before/after Grafana pair — copied `grafana/restart/5.png`→`screenshots/grafana_before.png`, `6.png`→`grafana_after.png` (force-added; screenshots are gitignored) |
| `70b780f` | **Task 8 fix:** force-add `screenshots/vllm_manual_query.png` (required deliverable, was untracked) |
| `18ff792` | **Tasks 5/6:** REPORT §3 rollback subsection, §4 per-iteration pass rate, §5 "what I'd do with more time" |
| (earlier) `9414310` | result JSONs `load_iter4_*`, `load_iter5` (fetched from VM, now local too) |

Local `phase3-5-local` HEAD = `18ff792`. It is **ahead of the VM** (VM `main` = `9414310`) and ahead of
GitHub `mine/main`.

---

## 3. REMAINING WORK — all blocked on the VM (do this when it's back up)

The VM (`iftahp@<IP>`, IP is **dynamic**; last seen `89.169.115.42`, currently **DOWN/unreachable on :22**)
hosts vLLM (:8000) + the agent (:8001) + Langfuse/Prometheus/Grafana (docker compose). Steps in order:

### Step A — get IP, sync the reverted config to the VM
```bash
# update the remote if the IP changed:
#   git remote set-url vm iftahp@<NEW_IP>:mlops-assignment
git push vm phase3-5-local:incoming --force
ssh iftahp@<IP> 'cd ~/mlops-assignment && git checkout main && git merge --ff-only incoming'
# VM main should now be at 18ff792 (reverted graph.py + start_vllm.sh present)
```

### Step B — restart serving + observability with the final config (Task 4)
```bash
ssh iftahp@<IP>
cd ~/mlops-assignment
# vLLM (reverted start_vllm.sh — no --max-num-seqs):
#   run scripts/start_vllm.sh (tmux/background); wait for GET :8000/health = 200
# observability:
docker compose -f infra/docker-compose.yml up -d   # (confirm compose file path)
# agent server with FINAL config (sync endpoint, threadpool 100, 2 iters):
AGENT_MAX_THREADS=100 AGENT_MAX_ITERATIONS=2 <the uvicorn/agent start cmd used in Phase 4-6>
# VM .env must point VLLM_BASE_URL=http://localhost:8000/v1 (NOT the local Mac .env, which points at
# hosted Nebius Token Factory). Verify GET :8001/health, then one POST /answer smoke.
```

### Step C — run the eval against the final config → `results/eval_after_tuning.json` (Task 4)
```bash
# agent must be running with AGENT_MAX_ITERATIONS=2 (final). Eval is sequential, so threads don't matter.
cd ~/mlops-assignment
uv run python evals/run_eval.py --out results/eval_after_tuning.json
# prints summary: overall_pass_rate + pass_rate_per_iteration. EXPECTED ~0.367 (11/30) — i.e. +1 vs
# baseline's 0.333 (baseline was MAX_ITERATIONS=3). If it regressed, that's fine — ANALYZE it in REPORT.
```
Then pull the file back and commit:
```bash
# from Mac:
git fetch vm && git merge --ff-only vm/main      # or scp the json; then force-add (results/*.json gitignored)
git add -f results/eval_after_tuning.json && git commit -m "Phase 6: eval_after_tuning (final config)"
```

### Step D — fill the ONE pending number in REPORT (Task 5/6)
In `REPORT.md` §3, subsection **"Final configuration (the rollback decision)"**, replace the
`⏳ PENDING` block (search for "PENDING — re-run on the VM") with the real
`overall_pass_rate` / `pass_rate_per_iteration` from `eval_after_tuning.json`, and one honest sentence on
whether quality survived vs baseline (`results/eval_baseline.json` = 0.333 / [0.30,0.367,0.333]).

### Step E — zip for submission (Task 8)
From the repo root (`mlops-assignment/`), once `eval_after_tuning.json` exists and REPORT is filled:
```bash
cd /Users/iftahp/PycharmProjects/NebiusAcademy/AIPerformanceEngineering/Module3MLOps/m3-assignment-2/mlops-assignment
zip -r ../iftah_piasetzky_mlops_assignment_2.zip . \
  -x '.git/*' -x '.venv/*' -x '**/__pycache__/*' -x '.pytest_cache/*' \
  -x 'data/bird/*.sqlite'   # decide: BIRD dbs are large — include only if the grader needs them
```
**Before zipping, re-check README §"Final deliverables" table** (every row must exist) and §"Grading".

---

## 4. Deliverables checklist (status now)

| File | Status |
|---|---|
| `REPORT.md` | ✅ written; **1 pending number** (Step D) |
| `infra/grafana/provisioning/dashboards/serving.json` | ✅ |
| `agent/graph.py`, `agent/prompts.py` | ✅ (graph.py reverted) |
| `evals/run_eval.py` | ✅ |
| `results/eval_baseline.json` | ✅ |
| `results/eval_after_tuning.json` | ❌ **Step C produces this** |
| `screenshots/vllm_manual_query.png` | ✅ (now tracked) |
| `screenshots/grafana_serving.png` | ✅ |
| `screenshots/langfuse_trace.png`, `langfuse_tags.png` | ✅ |
| `screenshots/grafana_eval_run.png` | ✅ |
| `screenshots/grafana_before.png`, `grafana_after.png` | ✅ (restart #5/#6) |

---

## 5. Git remotes cheat-sheet
- `mine` = `https://github.com/IftahPi/mlops-assignment.git` (your GitHub fork; has working creds **from
  the Mac only** — the VM has none over HTTPS).
- `vm` = `iftahp@<IP>:mlops-assignment` (SSH; the live working copy).
- `origin` = `GlebBerjoskin/mlops-assignment` (course upstream — don't push here).
- To back up to GitHub: from the **Mac**, `git push mine phase3-5-local:main`.

## 6. Workflow reminders (from memory)
- Install only into the project `.venv`; track deps in the project requirements file.
- The Mac `.env` points at **hosted Nebius Token Factory**, not local vLLM — don't confuse with the VM `.env`.
- Phase-6 SLO source of truth = the load driver JSONs (`results/load_iter*.json`), not the Grafana
  histogram (interpolated approximation).
