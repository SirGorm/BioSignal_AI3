---
name: multimodal-fusion
description: Use when designing how multiple biosignal modalities are combined in a neural network. Covers early fusion (stacked channels), late fusion (per-modality encoders), and hybrid approaches. Critical for this project because the 6 modalities have wildly different sample rates and temporal characteristics — naive concatenation often fails.
---

# Multimodal Fusion (6 modalities, mixed sample rates)

The 6 modalities in this project have different temporal characteristics:

| Modality | Sample rate (typical) | Temporal scale of information |
|----------|----------------------|------------------------------|
| ECG      | 500 Hz | Beat-to-beat (~1 s) |
| EMG      | 1000 Hz | Spectral content per ~250 ms |
| EDA      | 32 Hz | Slow phasic (~1–10 s) |
| Temp     | 4 Hz | Very slow (~30 s+) |
| Acc-mag  | 100 Hz | Per-rep (~1–3 s) |
| PPG-grn  | 64 Hz | Per-pulse (~1 s) |

You must decide how to align them in time AND how to fuse them. Three strategies:

## Strategy 1: Early fusion (resample-to-common-rate)

Resample all modalities to a single rate (typically 100 Hz, matching acc) and stack as channels. Simplest; works well when modalities carry related information at the chosen rate.

```python
# In the data loader / dataset class
def to_common_rate(signals_dict, target_fs=100, target_t=None):
    """signals_dict: {name: (timestamps_unix, values, native_fs)}
    Returns: tensor (n_channels, n_samples) on common time grid."""
    import numpy as np
    from scipy.interpolate import interp1d

    if target_t is None:
        # Use ECG/EMG range bracketed by all-modality coverage
        t_starts = [s[0][0] for s in signals_dict.values()]
        t_ends   = [s[0][-1] for s in signals_dict.values()]
        t_start, t_end = max(t_starts), min(t_ends)
        target_t = np.arange(t_start, t_end, 1.0 / target_fs)

    out = []
    for name, (ts, vals, fs) in signals_dict.items():
        f = interp1d(ts, vals, bounds_error=False,
                      fill_value=(vals[0], vals[-1]))
        out.append(f(target_t))
    return np.stack(out, axis=0), target_t  # (C, T)
```

**Costs**: EMG decimated from 1000 Hz to 100 Hz loses spectral content above 50 Hz — the entire fatigue-relevant band (Dimitrov et al. 2006). For this reason, **early fusion at low rate is a bad fit for this project**. Use it only as the simplest sanity-check baseline.

## Strategy 2: Late fusion (per-modality encoders)

Each modality goes through its own encoder at its native rate, then the per-modality representations are concatenated and passed to task heads. Preserves spectral content of each modality but adds parameters.

```python
import torch
import torch.nn as nn

class PerModalityEncoder(nn.Module):
    """A small 1D-CNN encoder appropriate to a single modality's rate."""
    def __init__(self, in_ch=1, out_dim=64, fs=100):
        super().__init__()
        # Kernel sizes scale with fs so receptive field is roughly equivalent in seconds
        k = max(3, int(0.05 * fs))  # ~50 ms first kernel
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=k, stride=2, padding=k//2),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=k, padding=k//2),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):  # (B, in_ch, T_modality)
        return self.net(x)


class LateFusionEncoder(nn.Module):
    """Encodes each modality separately, concatenates representations.
    Modalities arrive as a dict of tensors (different T per modality)."""
    MODALITIES = {
        'ecg':       {'in_ch': 1, 'fs': 500},
        'emg':       {'in_ch': 1, 'fs': 1000},
        'eda':       {'in_ch': 1, 'fs': 32},
        'temp':      {'in_ch': 1, 'fs': 4},
        'acc_mag':   {'in_ch': 1, 'fs': 100},
        'ppg_green': {'in_ch': 1, 'fs': 64},
    }
    def __init__(self, repr_dim=128, per_mod_dim=64):
        super().__init__()
        self.encoders = nn.ModuleDict({
            m: PerModalityEncoder(cfg['in_ch'], per_mod_dim, cfg['fs'])
            for m, cfg in self.MODALITIES.items()
        })
        self.fuse = nn.Sequential(
            nn.Linear(per_mod_dim * len(self.MODALITIES), repr_dim),
            nn.ReLU(),
        )

    def forward(self, x_dict):
        zs = [self.encoders[m](x_dict[m]) for m in self.MODALITIES]
        return self.fuse(torch.cat(zs, dim=-1))
```

**Costs**: more parameters, more careful data-loading (you can't stack into a single 3D tensor; you need a dict of tensors). Worth it because EMG keeps its 1 kHz spectral information.

## Strategy 3: Hybrid — group by temporal scale

Cluster modalities by how fast they change, then early-fuse within each group at the group's appropriate rate. Three groups for this project:

| Group | Modalities | Rate | Purpose |
|-------|-----------|------|---------|
| **Fast** | EMG | 1000 Hz | Spectral fatigue features (MNF, Dimitrov) |
| **Medium** | ECG, Acc-mag, PPG | 100 Hz | Cardiac, motion, pulse |
| **Slow** | EDA, Temp | 32 Hz | Tonic/thermal context |

Then each group has its own encoder and the three group-representations are fused. Compromise between simplicity and respecting modality-specific timescales.

```python
class HybridFusion(nn.Module):
    GROUPS = {
        'fast':   {'channels': ['emg'],                  'fs': 1000, 'group_dim': 96},
        'medium': {'channels': ['ecg', 'acc_mag', 'ppg_green'], 'fs': 100, 'group_dim': 96},
        'slow':   {'channels': ['eda', 'temp'],          'fs': 32,  'group_dim': 32},
    }
    def __init__(self, repr_dim=128):
        super().__init__()
        self.group_encoders = nn.ModuleDict()
        for g, cfg in self.GROUPS.items():
            self.group_encoders[g] = PerModalityEncoder(
                in_ch=len(cfg['channels']),
                out_dim=cfg['group_dim'],
                fs=cfg['fs'],
            )
        total = sum(cfg['group_dim'] for cfg in self.GROUPS.values())
        self.fuse = nn.Linear(total, repr_dim)

    def forward(self, x_groups):
        # x_groups: dict of (B, C_g, T_g) tensors keyed by group name
        zs = [self.group_encoders[g](x_groups[g]) for g in self.GROUPS]
        return self.fuse(torch.cat(zs, dim=-1))
```

## Recommendation for this project

**Start with hybrid grouping.** It captures most of the benefit of late fusion (preserving EMG spectral content) at lower complexity. If hybrid clearly beats early fusion in offline experiments, run a full late-fusion ablation. If late fusion beats hybrid, that's interesting and ship-worthy.

Compare the same architecture (e.g., TCN) across all three fusion strategies as the cleanest ablation. Don't change architecture AND fusion at the same time — you won't know what helped.

## Dataset implementation note

The labeled parquet stores all signals at native rate aligned by Unix time (per `data-loader` skill). For early fusion, resample to 100 Hz at dataset construction time. For late/hybrid fusion, return a dict of per-modality tensors per window:

```python
class StrengthRTDataset(torch.utils.data.Dataset):
    def __init__(self, parquet_paths, window_s=2.0, hop_s=0.1, mode='hybrid'):
        ...
    def __getitem__(self, idx):
        # Returns: (x, targets) where x is a dict (hybrid/late) or tensor (early)
        ...
```

## Per-task masking

Some tasks don't have ground truth on every window:
- `phase`, `reps`: only valid `in_active_set == True`
- `fatigue`: RPE per set, broadcast to all windows of that set; masked when `in_active_set == False` (rest periods don't have RPE)
- `exercise`: include `rest` as a class so all windows are usable

In the loss function:

```python
def masked_loss(pred, target, mask, loss_fn):
    """Loss computed only where mask is True."""
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return loss_fn(pred[mask], target[mask])
```

## References

When documenting fusion strategy in code comments and deliverables, cite from these (full entries in `literature-references` skill):

- **Ordóñez & Roggen 2016** — DeepConvLSTM, canonical example of early-fusion success when modalities share rate
- **Baltrušaitis et al. 2019** — multimodal ML taxonomy (early/late/hybrid fusion definitions)
- **Ramachandram & Taylor 2017** — practical deep multimodal architectures survey
- **Dimitrov et al. 2006** — for justifying NOT downsampling EMG below ~1 kHz (spectral fatigue features require it)

All are in the central `literature-references` skill. Never invent.
