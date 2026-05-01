"""Label verification visualizer.

Generates per-session PNG (overview) + HTML (interactive) plots showing:
- Joint angle traces with phase overlay (eccentric / concentric / pause / transition)
- Vertical markers at rep onsets
- Exercise band at the bottom (colored regions per exercise, grey for rest)
- Auto-generated sanity flags

Usage:
    python scripts/visualize_labels.py
    python scripts/visualize_labels.py --subjects S001 S002
    python scripts/visualize_labels.py --output-dir runs/<ts>_label-verification

Outputs:
    runs/<ts>_label-verification/
        verification_summary.md          # global summary + sanity flags table
        S001_session1_overview.png       # static overview, fast to scan
        S001_session1_interactive.html   # zoomable for detail inspection
        S002_session1_overview.png
        S002_session1_interactive.html
        ...

References:
- González-Badillo & Sánchez-Medina 2010 — phase definitions from velocity sign
- Bonomi et al. 2009 — active-set detection from acc magnitude
- Saeb et al. 2017 — per-subject inspection before aggregation
"""

from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from src.eval.plot_style import apply_style, despine

apply_style()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PHASE_COLORS = {
    'eccentric':    '#e74c3c',  # red
    'concentric':   '#27ae60',  # green
    'bottom_pause': '#7f8c8d',  # grey
    'top_pause':    '#bdc3c7',  # light grey
    'transition':   '#f39c12',  # orange
    'rest':         '#ecf0f1',  # very light
}

# Fallback colors for an arbitrary number of exercise classes
EXERCISE_PALETTE = [
    '#3498db', '#9b59b6', '#1abc9c', '#e67e22', '#34495e',
    '#16a085', '#2980b9', '#8e44ad', '#d35400', '#2c3e50',
]

# Expected ranges for sanity flags — adjust per protocol
EXPECTED_REPS_PER_SET = (7, 11)
EXPECTED_RPE_RANGE = (1, 10)
EXPECTED_SETS_PER_SESSION = (6, 12)   # e.g., 3 exercises × 3 sets


# ---------------------------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------------------------

def find_aligned_files(labeled_root: Path,
                        subject_filter: Optional[List[str]] = None) -> List[Path]:
    """Find aligned_features.parquet for each session, optionally filtered."""
    files = sorted(labeled_root.rglob('aligned_features.parquet'))
    if subject_filter:
        files = [f for f in files
                 if any(s in str(f) for s in subject_filter)]
    return files


def load_session(parquet_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    # Ensure required columns exist; otherwise raise informative error
    required = ['t_unix', 'subject_id', 'session_id']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{parquet_path} missing required columns: {missing}. "
            f"Verify data-labeler output schema.")
    return df


def detect_joint_angle_columns(df: pd.DataFrame) -> List[str]:
    """Find columns that look like joint-angle measurements."""
    candidates = [c for c in df.columns
                  if c.startswith('joint_')
                  or c.endswith('_angle')
                  or c in ('knee_angle', 'hip_angle', 'elbow_angle',
                            'shoulder_angle')]
    return candidates


def detect_rep_onsets(df: pd.DataFrame) -> List[float]:
    """Identify rep onset times: where rep_index increments."""
    if 'rep_index' not in df.columns:
        return []
    rep = df['rep_index'].fillna(-1).astype(int).to_numpy()
    t = df['t_unix'].to_numpy()
    onsets = []
    prev = -1
    for i, r in enumerate(rep):
        if r != prev and r > 0:
            onsets.append(float(t[i]))
        prev = r
    return onsets


def get_phase_segments(df: pd.DataFrame) -> List[Tuple[float, float, str]]:
    """Return contiguous (t_start, t_end, phase_label) segments."""
    if 'phase_label' not in df.columns:
        return []
    phase = df['phase_label'].fillna('rest').astype(str).to_numpy()
    t = df['t_unix'].to_numpy()
    segments = []
    if len(phase) == 0:
        return segments
    cur_phase = phase[0]
    cur_start = t[0]
    for i in range(1, len(phase)):
        if phase[i] != cur_phase:
            segments.append((float(cur_start), float(t[i]), str(cur_phase)))
            cur_phase = phase[i]
            cur_start = t[i]
    segments.append((float(cur_start), float(t[-1]), str(cur_phase)))
    return segments


def get_exercise_segments(df: pd.DataFrame) -> List[Tuple[float, float, str]]:
    """Return contiguous (t_start, t_end, exercise_name) segments."""
    if 'exercise' not in df.columns:
        return []
    ex = df['exercise'].fillna('rest').astype(str).to_numpy()
    t = df['t_unix'].to_numpy()
    segments = []
    if len(ex) == 0:
        return segments
    cur = ex[0]; cur_start = t[0]
    for i in range(1, len(ex)):
        if ex[i] != cur:
            segments.append((float(cur_start), float(t[i]), str(cur)))
            cur = ex[i]; cur_start = t[i]
    segments.append((float(cur_start), float(t[-1]), str(cur)))
    return segments


def reps_per_set(df: pd.DataFrame) -> Dict[int, int]:
    """Count reps per set."""
    if 'set_number' not in df.columns or 'rep_index' not in df.columns:
        return {}
    out: Dict[int, int] = {}
    for set_num, sub in df.groupby('set_number'):
        if pd.isna(set_num):
            continue
        max_rep = sub['rep_index'].max()
        if pd.notna(max_rep):
            out[int(set_num)] = int(max_rep)
    return out


# ---------------------------------------------------------------------------
# Sanity flags
# ---------------------------------------------------------------------------

def compute_sanity_flags(df: pd.DataFrame,
                          rep_onsets: List[float]) -> List[Dict]:
    """Return a list of {'level': 'warn'|'ok', 'msg': '...'} sanity flags."""
    flags: List[Dict] = []
    n_total = len(df)

    # Active vs rest fraction
    if 'in_active_set' in df.columns:
        active_frac = df['in_active_set'].astype(bool).mean()
        if active_frac < 0.2:
            flags.append({'level': 'warn',
                           'msg': f"Only {active_frac:.0%} of session is active "
                                  f"(expected 30–70% for typical strength session)"})
        elif active_frac > 0.85:
            flags.append({'level': 'warn',
                           'msg': f"{active_frac:.0%} of session is active "
                                  f"(expected 30–70%; check rest detection)"})
        else:
            flags.append({'level': 'ok',
                           'msg': f"Active fraction OK: {active_frac:.0%}"})

    # Reps per set
    rps = reps_per_set(df)
    if rps:
        n_sets = len(rps)
        lo, hi = EXPECTED_SETS_PER_SESSION
        if n_sets < lo or n_sets > hi:
            flags.append({'level': 'warn',
                           'msg': f"{n_sets} sets detected (expected {lo}–{hi})"})
        else:
            flags.append({'level': 'ok',
                           'msg': f"{n_sets} sets detected (within expected range)"})

        rep_lo, rep_hi = EXPECTED_REPS_PER_SET
        for set_num, n in sorted(rps.items()):
            if n < rep_lo or n > rep_hi:
                flags.append({'level': 'warn',
                               'msg': f"Set {set_num}: {n} reps "
                                      f"(expected {rep_lo}–{rep_hi})"})

    # RPE range
    rpe_col = 'rpe_for_this_set' if 'rpe_for_this_set' in df.columns else 'rpe'
    if rpe_col in df.columns:
        rpe_vals = df[rpe_col].dropna().unique()
        out_of_range = [v for v in rpe_vals
                         if v < EXPECTED_RPE_RANGE[0] or v > EXPECTED_RPE_RANGE[1]]
        if out_of_range:
            flags.append({'level': 'warn',
                           'msg': f"RPE values outside 1–10 scale: "
                                  f"{sorted(set(out_of_range))[:5]}"})
        else:
            flags.append({'level': 'ok',
                           'msg': f"All RPE values within 1–10 scale"})

    # Joint-angle gaps
    joint_cols = detect_joint_angle_columns(df)
    if joint_cols:
        for jc in joint_cols:
            null_frac = df[jc].isna().mean()
            if null_frac > 0.1:
                flags.append({'level': 'warn',
                               'msg': f"{jc}: {null_frac:.0%} missing — "
                                      f"possible motion-capture dropout"})

    # Joint-angle range sanity (typical bounds 0–180°)
    for jc in joint_cols:
        v = df[jc].dropna()
        if len(v) > 0:
            vmin, vmax = float(v.min()), float(v.max())
            if vmin < -10 or vmax > 200:
                flags.append({'level': 'warn',
                               'msg': f"{jc}: extreme range "
                                      f"({vmin:.0f}°, {vmax:.0f}°) — units?"})

    # Time monotonicity
    t = df['t_unix'].to_numpy()
    if len(t) > 1:
        gaps = np.diff(t)
        median_gap = float(np.median(gaps))
        big_gaps = gaps[gaps > median_gap * 10]
        if len(big_gaps) > 0:
            flags.append({'level': 'warn',
                           'msg': f"{len(big_gaps)} large time gaps detected "
                                  f"(>10× median) — possible sensor disconnect"})

    if not any(f['level'] == 'warn' for f in flags):
        flags.append({'level': 'ok',
                       'msg': "No automated sanity flags raised"})

    return flags


# ---------------------------------------------------------------------------
# PNG plotting
# ---------------------------------------------------------------------------

def plot_session_png(df: pd.DataFrame,
                       joint_cols: List[str],
                       phase_segments: List[Tuple[float, float, str]],
                       exercise_segments: List[Tuple[float, float, str]],
                       rep_onsets: List[float],
                       title: str,
                       out_path: Path):
    """Generate static overview PNG."""
    n_joints = len(joint_cols)
    if n_joints == 0:
        # No joint angles available — produce a minimal plot showing
        # exercise/phase bands only
        fig, axes = plt.subplots(1, 1, figsize=(14, 3))
        axes = [axes]
        joint_cols = []
    else:
        fig, axes = plt.subplots(
            n_joints + 1, 1,
            figsize=(14, 2.2 * n_joints + 1.5),
            sharex=True,
            gridspec_kw={'height_ratios': [3] * n_joints + [0.5]},
        )
        if n_joints == 1:
            axes = [axes[0], axes[1]]

    t = df['t_unix'].to_numpy()
    t_rel = t - t[0]   # display as seconds-into-session

    def to_rel(unix_t):
        return unix_t - t[0]

    # Joint-angle traces with phase background
    for ax, jc in zip(axes[:n_joints], joint_cols):
        # Phase background bands
        for seg_start, seg_end, phase_name in phase_segments:
            color = PHASE_COLORS.get(phase_name, '#ffffff')
            ax.axvspan(to_rel(seg_start), to_rel(seg_end),
                       facecolor=color, alpha=0.25, zorder=0)
        # Joint angle line
        valid = df[jc].notna()
        ax.plot(t_rel[valid.to_numpy()], df[jc][valid].to_numpy(),
                color='#2c3e50', lw=1.0, zorder=2)
        # Rep-onset markers
        for r in rep_onsets:
            ax.axvline(to_rel(r), color='#e67e22', lw=0.8, alpha=0.7,
                        zorder=1)
        ax.set_ylabel(jc, fontsize=9)
        ax.grid(alpha=0.2)

    # Exercise band (bottom row)
    ex_ax = axes[-1]
    unique_exercises = sorted({s[2] for s in exercise_segments})
    palette = {ex: EXERCISE_PALETTE[i % len(EXERCISE_PALETTE)]
               for i, ex in enumerate(unique_exercises) if ex != 'rest'}
    palette['rest'] = '#ecf0f1'
    for seg_start, seg_end, ex_name in exercise_segments:
        color = palette.get(ex_name, '#bdc3c7')
        ex_ax.axvspan(to_rel(seg_start), to_rel(seg_end),
                       facecolor=color, alpha=0.85)
        # Add label at segment midpoint if wide enough
        width = to_rel(seg_end) - to_rel(seg_start)
        if width > 30:   # only label segments > 30 sec
            mid = to_rel(seg_start) + width / 2
            ex_ax.text(mid, 0.5, ex_name, ha='center', va='center',
                        fontsize=8, color='black')
    ex_ax.set_xlabel('Time (s, session-relative)', fontsize=10)
    ex_ax.set_yticks([])
    ex_ax.set_ylabel('Exercise', fontsize=9)
    ex_ax.set_ylim(0, 1)

    # Combined legend at top
    phase_handles = [Patch(facecolor=PHASE_COLORS[p], alpha=0.4, label=p)
                      for p in PHASE_COLORS if p in {s[2] for s in phase_segments}]
    if rep_onsets:
        phase_handles.append(plt.Line2D([0], [0], color='#e67e22', lw=1,
                                         label='rep onset'))
    if phase_handles:
        axes[0].legend(handles=phase_handles, loc='upper right',
                        fontsize=8, ncol=min(4, len(phase_handles)))

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    despine(fig=fig)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Interactive HTML plotting (Plotly if available; falls back to matplotlib)
# ---------------------------------------------------------------------------

def plot_session_html(df: pd.DataFrame,
                        joint_cols: List[str],
                        phase_segments: List[Tuple[float, float, str]],
                        exercise_segments: List[Tuple[float, float, str]],
                        rep_onsets: List[float],
                        title: str,
                        out_path: Path):
    """Generate interactive HTML plot. Uses Plotly; if unavailable, writes
    a minimal HTML wrapping the PNG."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        # Fallback: link the PNG inside HTML
        png_name = out_path.with_suffix('').name + '_overview.png'
        out_path.write_text(
            f"<html><head><title>{title}</title></head>"
            f"<body><h1>{title}</h1>"
            f"<p>Plotly not installed. Install with <code>pip install plotly"
            f"</code> for interactive view.</p>"
            f'<img src="{png_name}" style="max-width:100%"></body></html>'
        )
        return

    t = df['t_unix'].to_numpy()
    t_rel = t - t[0]

    def to_rel(unix_t):
        return unix_t - t[0]

    n_joints = max(len(joint_cols), 1)
    fig = make_subplots(
        rows=n_joints + 1, cols=1, shared_xaxes=True,
        row_heights=[3.0] * n_joints + [0.5],
        vertical_spacing=0.03,
        subplot_titles=joint_cols + ['Exercise'] if joint_cols
                        else ['Exercise'],
    )

    # Phase-colored shapes (background bands) on each joint subplot
    for row_i in range(1, n_joints + 1):
        for seg_start, seg_end, phase_name in phase_segments:
            color = PHASE_COLORS.get(phase_name, 'rgba(255,255,255,0)')
            fig.add_vrect(
                x0=to_rel(seg_start), x1=to_rel(seg_end),
                fillcolor=color, opacity=0.25, line_width=0,
                row=row_i, col=1,
                annotation_text=phase_name if (seg_end - seg_start) > 5 else "",
                annotation_position="top left",
                annotation_font_size=8,
            )

    # Joint-angle traces
    for i, jc in enumerate(joint_cols):
        valid = df[jc].notna()
        fig.add_trace(
            go.Scatter(
                x=t_rel[valid.to_numpy()],
                y=df[jc][valid].to_numpy(),
                mode='lines',
                name=jc,
                line=dict(color='#2c3e50', width=1.2),
                hovertemplate=f"{jc}: %{{y:.1f}}°<br>t=%{{x:.1f}}s<extra></extra>",
            ),
            row=i + 1, col=1,
        )

    # Rep-onset vertical lines
    for r in rep_onsets:
        for row_i in range(1, n_joints + 1):
            fig.add_vline(x=to_rel(r), line_color='#e67e22',
                          line_width=1, opacity=0.7,
                          row=row_i, col=1)

    # Exercise band as a heatmap-style plot
    if exercise_segments:
        unique_ex = sorted({s[2] for s in exercise_segments})
        palette = {ex: EXERCISE_PALETTE[i % len(EXERCISE_PALETTE)]
                   for i, ex in enumerate(unique_ex) if ex != 'rest'}
        palette['rest'] = '#ecf0f1'
        for seg_start, seg_end, ex_name in exercise_segments:
            color = palette.get(ex_name, '#bdc3c7')
            fig.add_shape(
                type='rect', xref=f"x{n_joints + 1}", yref=f"y{n_joints + 1}",
                x0=to_rel(seg_start), x1=to_rel(seg_end),
                y0=0, y1=1,
                fillcolor=color, line_width=0, opacity=0.85,
                row=n_joints + 1, col=1,
            )
            width = to_rel(seg_end) - to_rel(seg_start)
            if width > 30:
                mid = to_rel(seg_start) + width / 2
                fig.add_annotation(
                    x=mid, y=0.5, text=ex_name,
                    showarrow=False, xref=f"x{n_joints + 1}",
                    yref=f"y{n_joints + 1}",
                    font=dict(size=10),
                    row=n_joints + 1, col=1,
                )
        fig.update_yaxes(showticklabels=False, range=[0, 1],
                          row=n_joints + 1, col=1)

    fig.update_layout(
        title=title,
        height=300 * n_joints + 200,
        showlegend=True,
        hovermode='x unified',
    )
    fig.update_xaxes(title_text='Time (s, session-relative)',
                      row=n_joints + 1, col=1)

    fig.write_html(out_path, include_plotlyjs='cdn')


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def render_summary_md(per_session_results: List[Dict],
                       out_path: Path):
    lines = [
        "# Label Verification Summary",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Sessions inspected: {len(per_session_results)}",
        "",
        "## Quick scan",
        "",
        "| Subject | Session | Active % | Sets | Reps total | Warnings |",
        "|---------|---------|----------|------|-----------|----------|",
    ]
    for r in per_session_results:
        lines.append(
            f"| {r['subject']} | {r['session']} | "
            f"{r['active_frac']:.0%} | {r['n_sets']} | "
            f"{r['n_reps']} | {r['n_warnings']} |"
        )

    lines += ["", "## Per-session detail", ""]
    for r in per_session_results:
        lines.append(f"### {r['subject']} / {r['session']}")
        lines.append("")
        lines.append(f"- PNG: `{r['png']}`")
        lines.append(f"- Interactive: `{r['html']}`")
        lines.append("")
        lines.append("**Sanity flags:**")
        lines.append("")
        for f in r['flags']:
            symbol = '⚠️' if f['level'] == 'warn' else '✓'
            lines.append(f"- {symbol} {f['msg']}")
        lines.append("")

    lines += [
        "## How to use",
        "",
        "1. Scan the table above for sessions with many warnings — those need attention first.",
        "2. Open the PNG for a fast visual check (joint angle + phases + reps + exercise bands).",
        "3. If something looks suspicious in the PNG, open the corresponding HTML and zoom in.",
        "4. If labels are wrong, fix the labeling pipeline (`/label`) and rerun this command.",
        "",
        "## Phase color legend",
        "",
        "- 🟥 **eccentric** (red) — lengthening phase",
        "- 🟩 **concentric** (green) — shortening phase",
        "- ⬜ **bottom_pause / top_pause** (grey) — static holds",
        "- 🟧 **transition** (orange) — between phases",
        "- ⬜ **rest** (very light grey) — between sets",
        "",
        "Vertical orange lines mark detected rep onsets.",
        "",
        "## References",
        "",
        "- González-Badillo, J. J., & Sánchez-Medina, L. (2010). Movement velocity "
          "as a measure of loading intensity in resistance training. "
          "*International Journal of Sports Medicine*, 31(5), 347–352.",
        "- Bonomi, A. G., Goris, A. H. C., Yin, B., & Westerterp, K. R. (2009). "
          "Detection of type, duration, and intensity of physical activity "
          "using an accelerometer. *Medicine & Science in Sports & Exercise*, "
          "41(9), 1770–1777.",
        "- Saeb, S., Lonini, L., Jayaraman, A., Mohr, D. C., & Kording, K. P. "
          "(2017). The need to approximate the use-case in clinical machine "
          "learning. *GigaScience*, 6(5), gix019.",
    ]
    out_path.write_text('\n'.join(lines), encoding='utf-8')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--labeled-root', type=Path, default=Path('data/labeled'))
    p.add_argument('--subjects', type=str, nargs='+', default=None,
                    help='Filter to specific subject IDs (substring match)')
    p.add_argument('--output-dir', type=Path, default=None,
                    help='Default: runs/<ts>_label-verification')
    p.add_argument('--runs-root', type=Path, default=Path('runs'))
    return p.parse_args()


def main():
    args = parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.output_dir is None:
        args.output_dir = args.runs_root / f"{timestamp}_label-verification"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[verify] Output: {args.output_dir}")

    parquets = find_aligned_files(args.labeled_root, args.subjects)
    if not parquets:
        raise SystemExit(f"No aligned_features.parquet under {args.labeled_root}. "
                          f"Run /label first.")

    print(f"[verify] Inspecting {len(parquets)} session(s)")
    per_session_results = []

    for p in parquets:
        df = load_session(p)
        subject = str(df['subject_id'].iloc[0])
        session = str(df['session_id'].iloc[0])
        title = f"{subject} / {session} — label verification"
        slug = f"{subject}_{session}".replace('/', '_')
        png_path = args.output_dir / f"{slug}_overview.png"
        html_path = args.output_dir / f"{slug}_interactive.html"

        joint_cols = detect_joint_angle_columns(df)
        phase_segments = get_phase_segments(df)
        exercise_segments = get_exercise_segments(df)
        rep_onsets = detect_rep_onsets(df)
        flags = compute_sanity_flags(df, rep_onsets)

        try:
            plot_session_png(df, joint_cols, phase_segments,
                              exercise_segments, rep_onsets,
                              title, png_path)
            print(f"[verify]  PNG  -> {png_path.name}")
        except Exception as e:
            print(f"[verify]  PNG  FAILED ({e}) for {slug}")
            flags.append({'level': 'warn', 'msg': f'PNG generation failed: {e}'})

        try:
            plot_session_html(df, joint_cols, phase_segments,
                                exercise_segments, rep_onsets,
                                title, html_path)
            print(f"[verify]  HTML -> {html_path.name}")
        except Exception as e:
            print(f"[verify]  HTML FAILED ({e}) for {slug}")
            flags.append({'level': 'warn', 'msg': f'HTML generation failed: {e}'})

        active_frac = (df['in_active_set'].astype(bool).mean()
                        if 'in_active_set' in df.columns else float('nan'))
        rps = reps_per_set(df)
        per_session_results.append({
            'subject': subject,
            'session': session,
            'active_frac': active_frac,
            'n_sets': len(rps),
            'n_reps': sum(rps.values()),
            'n_warnings': sum(1 for f in flags if f['level'] == 'warn'),
            'flags': flags,
            'png': png_path.name,
            'html': html_path.name,
        })

    summary_path = args.output_dir / 'verification_summary.md'
    render_summary_md(per_session_results, summary_path)
    print(f"\n[verify] Summary: {summary_path}")

    # Aggregate stats
    total_warnings = sum(r['n_warnings'] for r in per_session_results)
    n_with_warnings = sum(1 for r in per_session_results if r['n_warnings'] > 0)
    print(f"[verify] {n_with_warnings}/{len(per_session_results)} sessions "
          f"have warnings ({total_warnings} total)")
    if n_with_warnings > 0:
        print("[verify] Open verification_summary.md to triage warnings.")


if __name__ == '__main__':
    main()
