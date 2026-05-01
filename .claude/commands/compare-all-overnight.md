---
description: Overnight unattended /compare-all — GPU-first, smoke-tested, retry-once-then-skip, resume-capable. Auto-approves training scripts. Run before bed.
allowed-tools: Bash, Read, Write, Edit
argument-hint: [--baseline-run PATH] [--top-k 30]
---

# Compare All — Overnight Mode

Runs the full /compare-all pipeline unattended. Designed to start before bed
and finish overnight. **All training scripts auto-approved** via
`.claude/settings.json` permissions; non-training tasks (pip install, git push,
rm) still require approval.

## What this command does (vs /compare-all)

Same 5 stages, but with operational guarantees for unattended execution:

- **GPU verification first**: halts if no CUDA available (use `--allow-cpu`
  to override for testing)
- **Smoke-test gate**: 3-min cnn1d run; halts entire pipeline if it fails
- **Retry-once-then-skip**: failed steps retried once, then skipped
- **Resume-capable**: each stage skips already-completed runs (those with
  `cv_summary.json`)
- **File logging**: all stdout/stderr tee'd to `logs/overnight_<ts>.log`
- **Status JSON updated** after every step
- **Auto-arch selection for ablation**: best Stage A architecture by mean rank
- **Baselines NOT retrained**: LightGBM/XGBoost from /train reused as-is

## Preconditions

- `/train` completed (LightGBM + XGBoost baseline run exists)
- `/feature-pipeline` completed (recommended K identified)
- `configs/splits.csv` from /train present
- GPU available (NVIDIA + CUDA + working PyTorch)
- Tests pass: `pytest tests/ -x`

## Steps

### Step 0 — User confirmation

Print plan and wait for go-ahead:

```
Overnight run plan:
  GPU check:                ~5 sec
  Smoke-test:               ~3 min
  Stage A (4 archs full):   ~M GPU-hours
  Stage B (4 archs top-K):  ~M GPU-hours
  Stage C (6 modalities):   ~M GPU-hours
  Stage D (plots):          ~5 min
  Stage E (comparison):     ~1 min

Total: ~total GPU-hours
Logs:   logs/overnight_<ts>.log
Status: logs/overnight_status_<ts>.json
```

### Step 1 — Launch

```bash
python scripts/run_overnight.py \
    --baseline-run runs/<lgbm_xgb_run> \
    --top-k <K_from_feature_pipeline> \
    --seeds 42 1337 7 \
    --epochs 50
```

Run in foreground from inside Claude Code; the script handles its own
logging to disk so you can read progress later. Safe to leave running.

### Step 2 — Brief user

Print before disconnecting:

```
Overnight run started.

Logs:    logs/overnight_<ts>.log
Status:  logs/overnight_status_<ts>.json
Output:  runs/<ts>_overnight-comparison/comparison.md (final report)

Safe to close laptop / sleep. The script writes status after every step.

In the morning:
  1. cat logs/overnight_status_<ts>.json | python -m json.tool
  2. Open runs/<ts>_overnight-comparison/comparison.md
  3. Read logs/overnight_<ts>.log if anything looks off
```

## Resume after a crash

If anything killed the run, just rerun the SAME command:

```bash
python scripts/run_overnight.py \
    --baseline-run runs/<lgbm_xgb_run> \
    --top-k <K> \
    --seeds 42 1337 7
```

Each stage detects already-complete runs by glob-matching `runs/*_<slug>`
and checking for `cv_summary.json`. Completed runs are skipped automatically.
Partial runs are restarted from scratch (no per-fold checkpoint resume).

## What's auto-approved

Listed in `.claude/settings.json` `permissions.allow`:

```
Bash(python scripts/train_*.py:*)
Bash(python scripts/compare_*.py:*)
Bash(python scripts/ablate_*.py:*)
Bash(python scripts/generate_*.py:*)
Bash(python scripts/train_with_top_k.py:*)
Bash(python scripts/sweep_top_k.py:*)
Bash(python scripts/run_overnight.py:*)
Bash(mkdir -p runs/*)
Bash(mkdir -p logs/*)
Bash(nvidia-smi:*)
Bash(git status), Bash(git diff:*), Bash(git log:*)
Read, Write, Edit, Glob, Grep
```

## What's NOT auto-approved (still requires your approval)

In `permissions.ask`:
- `pip install` / `uv pip install`
- `git push` / `git commit`
- `rm`

In `permissions.deny`:
- `rm -rf`, `sudo`, `curl`, `wget`
- Reading `.env`, `secrets/`, `data/private/`

## Hard rules

- **NEVER auto-install packages.** If `import xgboost` fails at runtime,
  the step retries once then skips. Fix in the morning.
- **NEVER push to git overnight.** Version control is explicit only.
- **NEVER delete `runs/` contents.** Resume requires existing artifacts.
- **GPU memory**: if OOM persists across retries, lower `--batch-size` to 32
  and rerun (will resume from where it stopped).
