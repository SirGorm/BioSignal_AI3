---
name: neural-architectures
description: Use when implementing 1D-CNN, LSTM, CNN-LSTM, or TCN architectures for the multi-task biosignal pipeline. Provides PyTorch implementations of each architecture with shared encoder + 4 task heads (fatigue regression, exercise classification, phase classification, rep counting). All architectures take multimodal raw signal windows as input. Use AFTER the LightGBM baseline is established and documented in a model_card.md.
---

# Neural Architectures (multi-task, multimodal)

Four architectures to compare against the LightGBM baseline: **1D-CNN**, **LSTM**, **CNN-LSTM**, and **TCN**. All share the same multi-task structure: one encoder produces a representation, four task-specific heads predict their targets.

## Common contract

Every architecture takes the same input shape and produces the same output structure, so they're interchangeable in the training loop:

```python
# Input
x: torch.Tensor  # (batch, channels, time)
                 # channels = 9 (ecg, emg, eda, temp, ax, ay, az, ppg_green, acc_mag)
                 # time = window_samples (e.g., 200 samples = 2 s at 100 Hz)

# Output (dict, one entry per task)
{
    'exercise':  (batch, n_classes_exercise),     # logits
    'phase':     (batch, n_classes_phase),        # logits
    'fatigue':   (batch, 1),                      # regression
    'reps':      (batch, 1),                      # regression (per-window rep activity)
}
```

Targets at training time:
- `exercise`: long, class index per window
- `phase`: long, class index per window
- `fatigue`: float, RPE 1–10 (broadcast from per-set RPE — see masked loss below)
- `reps`: float, instantaneous "rep velocity" or rep-event indicator (regress, then aggregate at inference)

## Shared base class

```python
import torch
import torch.nn as nn

class MultiTaskModel(nn.Module):
    """Common interface — subclasses implement encode(x) -> (B, D)."""
    def __init__(self, n_channels=9, repr_dim=128,
                 n_exercise=5, n_phase=4, dropout=0.3):
        super().__init__()
        self.repr_dim = repr_dim
        self.dropout = nn.Dropout(dropout)
        # Task-specific heads
        self.head_exercise = nn.Linear(repr_dim, n_exercise)
        self.head_phase    = nn.Linear(repr_dim, n_phase)
        self.head_fatigue  = nn.Linear(repr_dim, 1)
        self.head_reps     = nn.Linear(repr_dim, 1)

    def encode(self, x):
        raise NotImplementedError

    def forward(self, x):
        h = self.dropout(self.encode(x))
        return {
            'exercise': self.head_exercise(h),
            'phase':    self.head_phase(h),
            'fatigue':  self.head_fatigue(h),
            'reps':     self.head_reps(h),
        }
```

## 1D-CNN

Pure convolutional encoder. Fast, parallel, good for short-time patterns. Most likely to match LightGBM in this regime.

```python
class CNN1D(MultiTaskModel):
    def __init__(self, n_channels=9, repr_dim=128, **kw):
        super().__init__(n_channels, repr_dim, **kw)
        self.encoder = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, repr_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(repr_dim), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # global avg pool -> (B, repr_dim, 1)
            nn.Flatten(),             # (B, repr_dim)
        )

    def encode(self, x):
        return self.encoder(x)
```

Reference: Yang et al. 2015 introduced 1D-CNNs for human activity recognition from sensors; widely used since.

## LSTM

Pure recurrent. Good when temporal order matters but not for very long windows (vanishing gradients despite gating). For 2 s windows at 100 Hz (200 timesteps) it's fine.

```python
class LSTMNet(MultiTaskModel):
    def __init__(self, n_channels=9, repr_dim=128, hidden=128, n_layers=2, **kw):
        super().__init__(n_channels, repr_dim, **kw)
        self.lstm = nn.LSTM(
            input_size=n_channels, hidden_size=hidden,
            num_layers=n_layers, batch_first=True,
            bidirectional=True, dropout=0.2 if n_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(hidden * 2, repr_dim)  # *2 for bidirectional

    def encode(self, x):
        # x: (B, C, T) -> (B, T, C) for LSTM
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)        # (B, T, 2*hidden)
        # Mean-pool across time for window-level representation
        h = out.mean(dim=1)
        return self.proj(h)
```

Note: bidirectional LSTM is fine for **offline training** but cannot be deployed in the strict real-time pipeline (it sees future timesteps). At deployment, switch to unidirectional. Document this in model_card.md.

Reference: Hochreiter & Schmidhuber 1997 (LSTM). Schuster & Paliwal 1997 (BiLSTM).

## CNN-LSTM

Convolutional front-end extracts local features, LSTM models their temporal evolution. Often the strongest of the four for biosignal classification of medium-length windows.

```python
class CNN_LSTM(MultiTaskModel):
    def __init__(self, n_channels=9, repr_dim=128, lstm_hidden=128, **kw):
        super().__init__(n_channels, repr_dim, **kw)
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(
            input_size=128, hidden_size=lstm_hidden,
            num_layers=1, batch_first=True, bidirectional=True,
        )
        self.proj = nn.Linear(lstm_hidden * 2, repr_dim)

    def encode(self, x):
        f = self.conv(x)             # (B, 128, T')
        f = f.transpose(1, 2)        # (B, T', 128)
        out, _ = self.lstm(f)
        h = out.mean(dim=1)
        return self.proj(h)
```

Reference: Karpathy et al. 2014 (CNN+LSTM for video; same idea applies to 1D signals). Ordóñez & Roggen 2016 (DeepConvLSTM for HAR — closest analogue to your task).

## TCN (Temporal Convolutional Network)

Dilated causal convolutions. Same temporal modeling power as LSTM but parallelizable, with explicit receptive field control. Good fit for real-time deployment because layers are inherently causal.

```python
class TemporalBlock(nn.Module):
    """Causal dilated conv block with residual."""
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.2):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size,
                                padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size,
                                padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.pad = pad
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def _causal_trim(self, x):
        # Remove right padding to keep causality
        return x[..., :-self.pad] if self.pad > 0 else x

    def forward(self, x):
        out = self.relu(self.bn1(self._causal_trim(self.conv1(x))))
        out = self.dropout(out)
        out = self.relu(self.bn2(self._causal_trim(self.conv2(out))))
        out = self.dropout(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(MultiTaskModel):
    def __init__(self, n_channels=9, repr_dim=128,
                 channels=(64, 64, 128, 128), kernel_size=5, dropout=0.2, **kw):
        super().__init__(n_channels, repr_dim, **kw)
        layers = []
        ch_in = n_channels
        for i, ch_out in enumerate(channels):
            layers.append(TemporalBlock(ch_in, ch_out, kernel_size,
                                          dilation=2**i, dropout=dropout))
            ch_in = ch_out
        self.tcn = nn.Sequential(*layers)
        self.proj = nn.Linear(channels[-1], repr_dim)

    def encode(self, x):
        h = self.tcn(x)              # (B, C_last, T)
        # Use last timestep — natural for causal model in real-time
        h = h[..., -1]
        return self.proj(h)
```

The TCN here is **strictly causal** — receptive field grows exponentially with depth via dilated convolutions, but no layer sees the future. This is the only architecture in this skill that can be deployed in `src/streaming/` directly without modification.

Reference: Bai et al. 2018 (An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling). The canonical TCN paper.

## Multi-task structure: hard vs soft parameter sharing

This project compares two structures (default = hard, soft = ablation on winner only):

### Hard sharing (default, recommended for low-data)

One encoder produces a single representation; 4 task heads project independently. Lower parameter count, better in low-data regimes (Caruana 1997). All architectures above use this structure.

### Soft sharing (ablation only)

Separate encoder per task. Useful when tasks have conflicting representational needs (e.g., fatigue wants long-range trends, phase wants sharp transitions).

```python
class SoftSharingModel(nn.Module):
    """4 separate encoders, one per task. Same architecture, independent weights."""
    def __init__(self, encoder_class, encoder_kwargs, n_exercise=5, n_phase=4):
        super().__init__()
        self.enc_exercise = encoder_class(**encoder_kwargs)
        self.enc_phase    = encoder_class(**encoder_kwargs)
        self.enc_fatigue  = encoder_class(**encoder_kwargs)
        self.enc_reps     = encoder_class(**encoder_kwargs)
        D = encoder_kwargs.get('repr_dim', 128)
        self.head_exercise = nn.Linear(D, n_exercise)
        self.head_phase    = nn.Linear(D, n_phase)
        self.head_fatigue  = nn.Linear(D, 1)
        self.head_reps     = nn.Linear(D, 1)

    def forward(self, x):
        return {
            'exercise': self.head_exercise(self.enc_exercise.encode(x)),
            'phase':    self.head_phase(self.enc_phase.encode(x)),
            'fatigue':  self.head_fatigue(self.enc_fatigue.encode(x)),
            'reps':     self.head_reps(self.enc_reps.encode(x)),
        }
```

Run soft sharing only on the winning architecture from the hard-sharing comparison. Reasons:
- 4× the compute and parameters
- Loses the regularization-via-shared-representation benefit of MTL
- Useful diagnostic: if soft sharing significantly outperforms hard sharing on a specific task, that task's representation conflicted with the others (negative transfer; Ruder 2017)

## Input variants: features vs raw signals

The project compares both input types. Same architectures, different input adapters.

### Variant B (raw) — default

Architectures take `(B, C, T)` as defined above. Use `multimodal-fusion` skill for hybrid grouping.

### Variant A (features) — comparison input

When input is per-window engineered features (same as LightGBM), architectures degrade to MLPs (the time dimension is degenerate). Wrap each architecture so it accepts features:

```python
class FeatureInputWrapper(MultiTaskModel):
    """Adapter for feature-input variant. Architecture identity preserved on paper,
    but in practice this is an MLP — that's the point of the comparison."""
    def __init__(self, n_features, hidden=128, repr_dim=128, dropout=0.3, **kw):
        super().__init__(n_channels=1, repr_dim=repr_dim, dropout=dropout, **kw)
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, repr_dim),
        )

    def encode(self, x):
        return self.encoder(x)


# In code that builds models:
def build_model(arch_name, input_variant, n_features=None, **kwargs):
    if input_variant == 'features':
        # All "architectures" become MLPs of varying depth
        return FeatureInputWrapper(n_features=n_features, **kwargs)
    elif input_variant == 'raw':
        return {
            'cnn1d':    CNN1D,
            'lstm':     LSTMNet,
            'cnn_lstm': CNN_LSTM,
            'tcn':      TCN,
        }[arch_name](**kwargs)
    raise ValueError(f"Unknown input variant: {input_variant}")
```

## Why these four

- **1D-CNN**: simplest, fastest, lowest parameters. Sets the floor for what neural representation gives you.
- **LSTM**: tests whether sequential ordering beyond local windows helps.
- **CNN-LSTM**: combination of local feature extraction and temporal modeling. Often best on medium-length biosignal windows (Ordóñez & Roggen 2016).
- **TCN**: causal-by-design, parallelizable, deployable. The architecture you'll likely ship if NN beats LightGBM.

## Deployment compatibility

| Architecture | Train offline | Deploy real-time |
|--------------|---------------|------------------|
| 1D-CNN       | ✓             | ✓ (causal padding required — pad on left only at inference) |
| LSTM (uni)   | ✓             | ✓ (carry hidden state between windows) |
| BiLSTM       | ✓             | ✗ (sees future — for analysis only) |
| CNN-LSTM     | ✓             | ✓ if LSTM is unidirectional |
| TCN          | ✓             | ✓ (already causal) |

If you want a paper-ready comparison, train all four with bidirectional/non-causal variants. If you want a deployable model, mark BiLSTM and any non-causal CNN-LSTM as **research-only** in model_card.md and ship the TCN.

## Parameter counts (approximate, default config)

| Architecture | Parameters | Notes |
|--------------|------------|-------|
| 1D-CNN       | ~250k      | Smallest |
| LSTM         | ~400k      | BiLSTM doubles forward params |
| CNN-LSTM     | ~500k      | Conv front-end + BiLSTM |
| TCN          | ~350k      | Depends on `channels` tuple |

With ~24 subjects × ~3500 windows/subject = ~84k training windows, even the largest of these is reasonable. LightGBM with ~50 features still has a structural advantage on the per-set fatigue task (only ~216 rows). NN is more likely to win on per-window tasks (exercise, phase) where you have plenty of data.

## How heads handle each task

- `head_exercise`: standard CE loss on per-window class
- `head_phase`: same
- `head_fatigue`: MAE loss on RPE; mask out windows during rest (where set RPE doesn't apply); broadcast set RPE to all windows in the set during training
- `head_reps`: regress an instantaneous rep-activity signal derived from joint angles (e.g., `1` at peak velocity, `0` elsewhere) — at inference, integrate / threshold to count reps

The actual loss combination, masking, and weighting is in the `deep-learning-training` skill.

## References

When documenting architecture choices in code comments and deliverables, cite from these (full entries in `literature-references` skill):

- **Bai et al. 2018** — TCN paper (canonical reference for dilated causal convolutions)
- **Hochreiter & Schmidhuber 1997** — LSTM paper
- **Schuster & Paliwal 1997** — bidirectional RNN paper
- **Yang et al. 2015** — 1D-CNN for multichannel sensor data
- **Ordóñez & Roggen 2016** — DeepConvLSTM, the closest analogue to this project's setup
- **Karpathy et al. 2014** — origin of CNN-LSTM hybrid architectures

All are now in the central `literature-references` skill. Cite using `(Author Year)` inline and full entry in `## References` section. Never invent.
