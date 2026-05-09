"""Generate LaTeX results tables from results/Final/v17_v20_thesis_master.csv."""
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "results/Final/v17_v20_thesis_master.csv"
OUT = ROOT / "results/thesis_results_tables.tex"


df = pd.read_csv(CSV)
df["window_s"] = df["window_s"].astype(str).str.replace(".0", "", regex=False)
NUM_COLS = [
    "exercise_f1_mean", "exercise_f1_std",
    "phase_f1_mean", "phase_f1_std",
    "fatigue_mae_mean", "fatigue_mae_std",
    "fatigue_pearson_mean", "fatigue_pearson_std",
    "reps_mae_mean", "reps_mae_std",
]
for c in NUM_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce")


def get(arch, variant, w, task):
    sub = df[(df["arch"] == arch) & (df["variant"] == variant) & (df["window_s"] == str(w)) & (df["tasks"] == task)]
    return sub.iloc[0] if not sub.empty else None


ARCH_LABEL = {
    "mlp": "\\gls{mlp}",
    "lstm": "\\gls{lstm}",
    "cnn1d_raw": "1D-\\gls{cnn}",
    "lstm_raw": "\\gls{lstm}",
    "cnn_lstm_raw": "\\gls{cnn}-\\gls{lstm}",
    "tcn_raw": "\\gls{tcn}",
}
FEATURE_ARCHS = ["mlp", "lstm"]
RAW_ARCHS = ["cnn1d_raw", "lstm_raw", "cnn_lstm_raw", "tcn_raw"]
WINDOWS = [1, 2, 5]


def cell(mean, std, bold=False):
    if mean is None or std is None or (isinstance(mean, float) and math.isnan(mean)) or (isinstance(std, float) and math.isnan(std)):
        return "---"
    body = f"{mean:.3f} \\pm {std:.3f}"
    return f"$\\mathbf{{{body}}}$" if bold else f"${body}$"


def get_metric(row, task):
    if row is None:
        return (None, None, None)
    if task == "exercise":
        return (row["exercise_f1_mean"], row["exercise_f1_std"], None)
    if task == "phase":
        return (row["phase_f1_mean"], row["phase_f1_std"], None)
    if task == "fatigue":
        return (row["fatigue_mae_mean"], row["fatigue_mae_std"], (row["fatigue_pearson_mean"], row["fatigue_pearson_std"]))
    if task == "reps":
        return (row["reps_mae_mean"], row["reps_mae_std"], None)
    return (None, None, None)


def task_row_for(arch, variant, w, scope, task):
    return get(arch, variant, w, "all 4" if scope == "MT" else task)


rf = df[df["category"] == "rf_baseline"].iloc[0]
RF = {
    "exercise": (rf["exercise_f1_mean"], rf["exercise_f1_std"]),
    "phase": (rf["phase_f1_mean"], rf["phase_f1_std"]),
    "fatigue": (rf["fatigue_mae_mean"], rf["fatigue_mae_std"], rf["fatigue_pearson_mean"]),
    "reps": (rf["reps_mae_mean"], rf["reps_mae_std"]),
}


def best_indices(task):
    bests = {}
    for scope in ["MT", "ST"]:
        for w in WINDOWS:
            entries = []
            for arch in FEATURE_ARCHS:
                row = task_row_for(arch, "features", w, scope, task)
                m, s, extra = get_metric(row, task)
                if m is not None and not (isinstance(m, float) and math.isnan(m)):
                    entries.append({"arch": arch, "variant": "features", "mean": m, "std": s, "extra": extra})
            for arch in RAW_ARCHS:
                row = task_row_for(arch, "raw", w, scope, task)
                m, s, extra = get_metric(row, task)
                if m is not None and not (isinstance(m, float) and math.isnan(m)):
                    entries.append({"arch": arch, "variant": "raw", "mean": m, "std": s, "extra": extra})
            if not entries:
                bests[(scope, w)] = (None, None)
                continue
            lower = task in ("fatigue", "reps")
            best_primary = min(entries, key=lambda e: e["mean"]) if lower else max(entries, key=lambda e: e["mean"])
            best_r = None
            if task == "fatigue":
                with_r = [e for e in entries if e["extra"] is not None and not math.isnan(e["extra"][0])]
                if with_r:
                    best_r = max(with_r, key=lambda e: e["extra"][0])
            bests[(scope, w)] = (best_primary, best_r)
    return bests


def render_section_simple(lines, label, scope, archs, variant, task, bests):
    nrows = len(archs)
    for i, a in enumerate(archs):
        cells = []
        for w in WINDOWS:
            row = task_row_for(a, variant, w, scope, task)
            m, s, _ = get_metric(row, task)
            bp = bests[(scope, w)][0]
            is_best = bp is not None and bp["arch"] == a and bp["variant"] == variant
            cells.append(cell(m, s, bold=is_best))
        mr = f"\\multirow{{{nrows}}}{{*}}{{{label}}}" if i == 0 else ""
        lines.append(f"  {mr}\n    & {scope} & {ARCH_LABEL[a]} & " + " & ".join(cells) + " \\\\")


def render_simple_table(task, label_suffix):
    bests = best_indices(task)
    if task == "exercise":
        cap = ("Exercise classification macro-F1 per window size. "
               "MT = multi-task, ST = single-task. Best neural macro-F1 per window is highlighted.")
    elif task == "phase":
        cap = ("Movement phase detection macro-F1 per window size. "
               "MT = multi-task, ST = single-task. Best value per window is highlighted.")
    elif task == "reps":
        cap = ("Repetition counting soft-overlap \\gls{mae} per window size "
               "(lower is better). MT = multi-task, ST = single-task. "
               "Best value per window is highlighted.")
    L = []
    L.append("\\begin{table}[htbp]")
    L.append("\\centering")
    L.append(f"\\caption{{{cap}}}")
    L.append(f"\\label{{tab:results-{label_suffix}}}")
    L.append("\\small")
    L.append("\\setlength{\\tabcolsep}{4pt}")
    L.append("\\begin{tabular}{lllccc}")
    L.append("\\toprule")
    L.append("\\textbf{Variant} & \\textbf{Type} & \\textbf{Architecture} & \\textbf{1\\,s} & \\textbf{2\\,s} & \\textbf{5\\,s} \\\\")
    L.append("\\midrule")
    render_section_simple(L, "Feature", "MT", FEATURE_ARCHS, "features", task, bests)
    L.append("\\midrule")
    render_section_simple(L, "Feature (ST)", "ST", FEATURE_ARCHS, "features", task, bests)
    L.append("\\midrule")
    render_section_simple(L, "Raw (MT)", "MT", RAW_ARCHS, "raw", task, bests)
    L.append("\\midrule")
    render_section_simple(L, "Raw (ST)", "ST", RAW_ARCHS, "raw", task, bests)
    L.append("\\midrule")
    rf_m, rf_s = RF[task][:2]
    rf_str = f"${rf_m:.3f} \\pm {rf_s:.3f}$"
    if task == "reps":
        L.append(f"\\multicolumn{{3}}{{l}}{{RF baseline}} & \\multicolumn{{3}}{{c}}{{{rf_str} (per-set count, not directly comparable)}} \\\\")
    else:
        L.append(f"\\multicolumn{{3}}{{l}}{{RF baseline}} & \\multicolumn{{3}}{{c}}{{{rf_str}}} \\\\")
    L.append("\\bottomrule")
    L.append("\\end{tabular}")
    L.append("\\end{table}")
    return "\n".join(L)


def render_fatigue_table():
    bests = best_indices("fatigue")
    L = []
    L.append("\\begin{table}[htbp]")
    L.append("\\centering")
    L.append("\\caption{Fatigue estimation \\gls{mae} in \\gls{rpe} points (lower is better) and Pearson $r$ (higher is better). MT = multi-task, ST = single-task. Best MT and ST values per window are highlighted separately.}")
    L.append("\\label{tab:results-fatigue}")
    L.append("\\small")
    L.append("\\setlength{\\tabcolsep}{3pt}")
    L.append("\\begin{tabular}{lllcccccc}")
    L.append("\\toprule")
    L.append("& & & \\multicolumn{2}{c}{\\textbf{1\\,s}} & \\multicolumn{2}{c}{\\textbf{2\\,s}} & \\multicolumn{2}{c}{\\textbf{5\\,s}} \\\\")
    L.append("\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}")
    L.append("\\textbf{Variant} & \\textbf{Type} & \\textbf{Architecture} & MAE $\\downarrow$ & $r$ $\\uparrow$ & MAE $\\downarrow$ & $r$ $\\uparrow$ & MAE $\\downarrow$ & $r$ $\\uparrow$ \\\\")
    L.append("\\midrule")

    def render(label, scope, archs, variant):
        nrows = len(archs)
        for i, a in enumerate(archs):
            cells = []
            for w in WINDOWS:
                row = task_row_for(a, variant, w, scope, "fatigue")
                m, s, extra = get_metric(row, "fatigue")
                bp, br = bests[(scope, w)]
                is_best_mae = bp is not None and bp["arch"] == a and bp["variant"] == variant
                is_best_r = br is not None and br["arch"] == a and br["variant"] == variant
                cells.append(cell(m, s, bold=is_best_mae))
                if extra is None or (isinstance(extra[0], float) and math.isnan(extra[0])):
                    cells.append("---")
                else:
                    cells.append(cell(extra[0], extra[1], bold=is_best_r))
            mr = f"\\multirow{{{nrows}}}{{*}}{{{label}}}" if i == 0 else ""
            L.append(f"  {mr}\n    & {scope} & {ARCH_LABEL[a]} & " + " & ".join(cells) + " \\\\")

    render("Feature", "MT", FEATURE_ARCHS, "features")
    L.append("\\midrule")
    render("Feature (ST)", "ST", FEATURE_ARCHS, "features")
    L.append("\\midrule")
    render("Raw (MT)", "MT", RAW_ARCHS, "raw")
    L.append("\\midrule")
    render("Raw (ST)", "ST", RAW_ARCHS, "raw")
    L.append("\\midrule")
    rf_m, rf_s, rf_r = RF["fatigue"]
    L.append(f"\\multicolumn{{3}}{{l}}{{RF baseline}} & \\multicolumn{{6}}{{c}}{{${rf_m:.3f} \\pm {rf_s:.3f}$,\\ $r = {rf_r:.3f}$}} \\\\")
    L.append("\\bottomrule")
    L.append("\\end{tabular}")
    L.append("\\end{table}")
    return "\n".join(L)


header = ("% Thesis results tables -- fatigue, reps, phase, exercise.\n"
          "% Auto-generated from results/Final/v17_v20_thesis_master.csv\n"
          "% via scripts/_gen_thesis_tables.py.\n"
          "% Requires: \\usepackage{booktabs, multirow, glossaries}\n")
tex = header + "\n"
tex += "% --- Table: Fatigue ---\n" + render_fatigue_table() + "\n\n"
tex += "% --- Table: Reps ---\n" + render_simple_table("reps", "reps") + "\n\n"
tex += "% --- Table: Phase ---\n" + render_simple_table("phase", "phase") + "\n\n"
tex += "% --- Table: Exercise ---\n" + render_simple_table("exercise", "exercise") + "\n"

OUT.write_text(tex, encoding="utf-8")
print(f"Wrote {len(tex)} chars to {OUT}")
