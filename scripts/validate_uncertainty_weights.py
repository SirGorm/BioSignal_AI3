"""Sanity check: do Kendall (2018) uncertainty weights actually learn?

Trains ONE feat-MLP fold for ~150 epochs with `use_uncertainty=True` and
records per-epoch:
  * raw per-task losses (exercise CE, phase KL, fatigue L1, reps SmoothL1)
  * log_var per task   (the learnable Kendall parameters)
  * implied weight w_i = 0.5 * exp(-log_var_i)

Plus a parallel control run with `use_uncertainty=False` and equal fixed
weights, on the same fold/seed, to confirm uncertainty weighting actually
changes optimisation behaviour.

Outputs: runs/uncertainty_validation/
  history.json     — per-epoch records, both runs
  log_var.png      — log_var traces per task
  weights.png      — implied weights per task
  losses.png       — per-task raw losses, both runs
  verdict.md       — pass/fail criteria + summary

Pass criteria:
  1. log_var values move away from zero (gradients flow into Kendall params)
  2. log_var differs between tasks at the end (heterogeneous weighting)
  3. Total loss curve looks healthy (no NaN, descending trend)
  4. Final implied weights are not all uniform within ~5%
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.datasets import WindowFeatureDataset
from src.models.mlp import MLPMultiTask
from src.training.cv import load_or_generate_splits
from src.training.loop import set_deterministic
from src.training.losses import MultiTaskLoss

# ------------------------------------------------------------------
# Config — match v13soft (clean, 1 s window, multi-task, soft labels)
# Use the best HPs Optuna already found in the v13soft trial we killed.
# ------------------------------------------------------------------
LABELED_ROOT = ROOT / "data" / "labeled_clean"
SPLITS = ROOT / "configs" / "splits_clean_loso.csv"
EXCLUDE = ["recording_004", "recording_005", "recording_007",
           "recording_008", "recording_009"]
OUT = ROOT / "runs" / "uncertainty_validation"
OUT.mkdir(parents=True, exist_ok=True)

WINDOW_S = 1.0
EPOCHS = 150
BATCH = 64
LR = 4e-3
WD = 2e-5
SEED = 42
FOLD_INDEX = 0  # which CV fold to use as the validation training fold


def build_dataset():
    files = sorted(LABELED_ROOT.rglob("window_features.parquet"))
    files = [p for p in files if not any(ex in str(p) for ex in EXCLUDE)]
    target_modes = {'reps': 'soft_overlap', 'phase': 'soft'}
    ds = WindowFeatureDataset(
        window_parquets=files, active_only=True,
        target_modes=target_modes, window_s=WINDOW_S,
    )
    return ds


def make_model(ds):
    return MLPMultiTask(
        n_features=ds.n_features,
        n_exercise=ds.n_exercise,
        n_phase=ds.n_phase,
        repr_dim=64, hidden_dim=64, dropout=0.22,
    )


def make_loss(use_uncertainty: bool):
    return MultiTaskLoss(
        w_exercise=1.0, w_phase=1.0, w_fatigue=1.0, w_reps=0.5,
        use_uncertainty_weighting=use_uncertainty,
        target_modes={'reps': 'soft_overlap', 'phase': 'soft'},
        enabled_tasks=['exercise', 'phase', 'fatigue', 'reps'],
    )


def make_optimizer(model, loss_fn, use_uncertainty: bool):
    if use_uncertainty:
        return torch.optim.AdamW([
            {'params': list(model.parameters())},
            {'params': list(loss_fn.parameters()), 'weight_decay': 0.0},
        ], lr=LR, weight_decay=WD)
    return torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)


def run_one(use_uncertainty: bool, ds, train_idx, test_idx, device):
    set_deterministic(SEED)
    model = make_model(ds).to(device)
    loss_fn = make_loss(use_uncertainty).to(device)
    opt = make_optimizer(model, loss_fn, use_uncertainty)
    scaler = GradScaler(enabled=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    train_loader = DataLoader(Subset(ds, train_idx), batch_size=BATCH,
                               shuffle=True, num_workers=0, pin_memory=False,
                               drop_last=True)
    val_loader = DataLoader(Subset(ds, test_idx), batch_size=BATCH * 2,
                             shuffle=False, num_workers=0, pin_memory=False)

    history = []
    for ep in range(EPOCHS):
        model.train()
        sums = {'total': 0.0, 'exercise': 0.0, 'phase': 0.0,
                'fatigue': 0.0, 'reps': 0.0}
        n_seen = 0
        for batch in train_loader:
            x = (batch['x'].to(device) if not isinstance(batch['x'], dict)
                 else {k: v.to(device) for k, v in batch['x'].items()})
            tgts = {k: v.to(device) for k, v in batch['targets'].items()}
            msks = {k: v.to(device) for k, v in batch['masks'].items()}

            opt.zero_grad(set_to_none=True)
            with autocast(enabled=True):
                preds = model(x)
                total, parts = loss_fn(preds, tgts, msks)
            has_signal = any(bool(msks[k].any().item())
                             for k in loss_fn.enabled if k in msks)
            if has_signal:
                scaler.scale(total).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            n = next(iter(tgts.values())).shape[0]
            n_seen += n
            sums['total'] += total.item() * n
            for k in parts: sums[k] += parts[k].item() * n
        sched.step()
        train_avg = {k: v / max(n_seen, 1) for k, v in sums.items()}

        model.eval()
        vsums = {'total': 0.0, 'exercise': 0.0, 'phase': 0.0,
                 'fatigue': 0.0, 'reps': 0.0}
        vn = 0
        with torch.no_grad():
            for batch in val_loader:
                x = (batch['x'].to(device) if not isinstance(batch['x'], dict)
                     else {k: v.to(device) for k, v in batch['x'].items()})
                tgts = {k: v.to(device) for k, v in batch['targets'].items()}
                msks = {k: v.to(device) for k, v in batch['masks'].items()}
                with autocast(enabled=True):
                    preds = model(x)
                    total, parts = loss_fn(preds, tgts, msks)
                n = next(iter(tgts.values())).shape[0]
                vn += n
                vsums['total'] += total.item() * n
                for k in parts: vsums[k] += parts[k].item() * n
        val_avg = {k: v / max(vn, 1) for k, v in vsums.items()}

        rec = {
            'epoch': ep,
            'train': train_avg,
            'val':   val_avg,
        }
        if use_uncertainty:
            lv = loss_fn.log_var.detach().cpu().numpy().tolist()
            keys = list(loss_fn._uw_keys)
            rec['log_var'] = dict(zip(keys, lv))
            rec['weight'] = {k: float(0.5 * np.exp(-v))
                             for k, v in zip(keys, lv)}
        history.append(rec)

        if ep % 10 == 0 or ep == EPOCHS - 1:
            extra = ''
            if use_uncertainty:
                lvs = ' '.join(f'{k[:3]}={rec["log_var"][k]:+.3f}'
                               for k in rec['log_var'])
                extra = f'  log_var: {lvs}'
            print(f"  [u={int(use_uncertainty)}] ep {ep:3d}  "
                  f"train_total={train_avg['total']:.4f}  "
                  f"val_total={val_avg['total']:.4f}  "
                  f"ex={train_avg['exercise']:.3f} "
                  f"ph={train_avg['phase']:.3f} "
                  f"fa={train_avg['fatigue']:.3f} "
                  f"re={train_avg['reps']:.3f}"
                  f"{extra}")

    return history


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    t0 = time.time()

    print("=== Kendall uncertainty-weight validation ===")
    print(f"  window={WINDOW_S}s  epochs={EPOCHS}  batch={BATCH}  lr={LR}")
    print(f"  fold_index={FOLD_INDEX}  seed={SEED}")
    print(f"  outdir={OUT}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  device={device}")

    ds = build_dataset()
    print(f"  dataset: {len(ds)} windows  features={ds.n_features}  "
          f"n_ex={ds.n_exercise}  n_ph={ds.n_phase}")
    if torch.cuda.is_available():
        ds.materialize_to_device("cuda")

    folds = load_or_generate_splits(np.array(ds.subject_ids), splits_path=SPLITS)
    fold = folds[FOLD_INDEX]
    train_idx = np.asarray(fold['train_idx'])
    test_idx = np.asarray(fold['test_idx'])
    print(f"  fold {FOLD_INDEX}: train={len(train_idx)} test={len(test_idx)}  "
          f"test subjects={fold.get('test_subjects')}")

    print("\n[run A] use_uncertainty=True (Kendall)")
    hist_u = run_one(True, ds, train_idx, test_idx, device)
    print("\n[run B] use_uncertainty=False (fixed weights)")
    hist_f = run_one(False, ds, train_idx, test_idx, device)

    # ----- Save raw history --------------------------------------------------
    payload = {'uncertainty': hist_u, 'fixed': hist_f,
               'config': {'window_s': WINDOW_S, 'epochs': EPOCHS,
                          'batch': BATCH, 'lr': LR, 'wd': WD, 'seed': SEED,
                          'fold_index': FOLD_INDEX}}
    (OUT / 'history.json').write_text(json.dumps(payload, indent=2))

    # ----- Plots -------------------------------------------------------------
    eps = [r['epoch'] for r in hist_u]

    # log_var trace
    fig, ax = plt.subplots(figsize=(10, 5))
    for k in ('exercise', 'phase', 'fatigue', 'reps'):
        ys = [r['log_var'][k] for r in hist_u]
        ax.plot(eps, ys, label=k, linewidth=2)
    ax.axhline(0, color='black', linestyle=':', alpha=0.5,
               label='log_var=0 (init)')
    ax.set_xlabel('epoch'); ax.set_ylabel('log_var')
    ax.set_title('Kendall uncertainty: log_var per task vs epoch')
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / 'log_var.png', dpi=140); plt.close(fig)

    # implied weight
    fig, ax = plt.subplots(figsize=(10, 5))
    for k in ('exercise', 'phase', 'fatigue', 'reps'):
        ys = [r['weight'][k] for r in hist_u]
        ax.plot(eps, ys, label=k, linewidth=2)
    ax.set_xlabel('epoch'); ax.set_ylabel('implied weight  0.5·exp(−log_var)')
    ax.set_title('Kendall uncertainty: implied per-task weights')
    ax.set_yscale('log')
    ax.legend(); ax.grid(alpha=0.3, which='both')
    fig.tight_layout(); fig.savefig(OUT / 'weights.png', dpi=140); plt.close(fig)

    # per-task loss comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, k in zip(axes.flat, ['exercise', 'phase', 'fatigue', 'reps']):
        ax.plot(eps, [r['train'][k] for r in hist_u], label='kendall (train)',
                color='#1f77b4')
        ax.plot(eps, [r['val'][k] for r in hist_u], label='kendall (val)',
                color='#1f77b4', linestyle='--')
        ax.plot(eps, [r['train'][k] for r in hist_f], label='fixed (train)',
                color='#d62728')
        ax.plot(eps, [r['val'][k] for r in hist_f], label='fixed (val)',
                color='#d62728', linestyle='--')
        ax.set_title(f'{k} — raw loss');  ax.set_xlabel('epoch')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle('Per-task raw loss: Kendall vs fixed weights')
    fig.tight_layout(); fig.savefig(OUT / 'losses.png', dpi=140); plt.close(fig)

    # ----- Verdict ----------------------------------------------------------
    last = hist_u[-1]
    log_vars = last['log_var']
    weights = last['weight']
    moved_from_zero = max(abs(v) for v in log_vars.values()) > 0.05
    spread = max(log_vars.values()) - min(log_vars.values())
    heterogeneous = spread > 0.1
    no_nan = all(np.isfinite(v) for v in log_vars.values()) \
              and np.isfinite(last['train']['total'])
    train_descended = (hist_u[0]['train']['total']
                       > hist_u[-1]['train']['total'])

    pass_all = moved_from_zero and heterogeneous and no_nan and train_descended

    md = []
    md.append('# Kendall uncertainty-weight validation — verdict\n')
    md.append(f'Run dir: `runs/uncertainty_validation/`\n')
    md.append(f'Total wall-time: {time.time() - t0:.1f} s\n')
    md.append(f'Architecture: feat-MLP @ {WINDOW_S}s, '
              f'{EPOCHS} epochs, fold {FOLD_INDEX}, seed {SEED}\n')
    md.append(f'Test subjects: {fold.get("test_subjects")}\n')
    md.append('')
    md.append('## Final state (epoch ' + str(EPOCHS - 1) + ')\n')
    md.append('| Task | log_var | implied weight 0.5·exp(−log_var) | '
              'final train loss | final val loss |')
    md.append('|---|---|---|---|---|')
    for k in ('exercise', 'phase', 'fatigue', 'reps'):
        md.append(f'| {k} | {log_vars[k]:+.4f} | {weights[k]:.4f} | '
                  f'{last["train"][k]:.4f} | {last["val"][k]:.4f} |')
    md.append('')

    md.append('## Pass criteria\n')
    md.append(f'1. Kendall params moved from initial zero — '
              f'`{moved_from_zero}` (max |log_var| = '
              f'{max(abs(v) for v in log_vars.values()):.3f})')
    md.append(f'2. log_var differs across tasks (spread > 0.1) — '
              f'`{heterogeneous}` (spread = {spread:.3f})')
    md.append(f'3. No NaN/Inf in final state — `{no_nan}`')
    md.append(f'4. Train total loss descended — '
              f'`{train_descended}` ({hist_u[0]["train"]["total"]:.3f} → '
              f'{hist_u[-1]["train"]["total"]:.3f})')
    md.append('')
    md.append(f'**Overall: {"PASS" if pass_all else "FAIL"}**\n')

    md.append('## Compared to fixed-weight control')
    f_last = hist_f[-1]
    md.append(f'- Kendall final val total: {last["val"]["total"]:.4f}')
    md.append(f'- Fixed   final val total: {f_last["val"]["total"]:.4f}')
    md.append(f'- Δ (fixed − kendall): {f_last["val"]["total"] - last["val"]["total"]:+.4f} '
              '(positive = kendall better)')

    (OUT / 'verdict.md').write_text('\n'.join(md))
    print('\n=== VERDICT ===')
    print('\n'.join(md[-15:]))
    print(f'\nWrote {OUT}/verdict.md, log_var.png, weights.png, losses.png, '
          f'history.json')


if __name__ == '__main__':
    main()
