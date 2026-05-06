"""Retrofit existing cv_summary.json: mark non-enabled task metrics as
``{'untrained': True}`` so downstream plots/tables don't compare random-init
heads against trained baselines.

Reads ``phase2/train_config.json["enabled_tasks"]`` and rewrites
``phase2/<arch>/cv_summary.json`` in place. Original is backed up to
``cv_summary.raw.json`` next to it.

Run:
    python scripts/retrofit_untrained_summaries.py runs/optuna_clean_v10*
    python scripts/retrofit_untrained_summaries.py runs/optuna_clean_v10restwin-w3s-fatigue-raw-tcn
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ALL_TASKS = ('exercise', 'phase', 'fatigue', 'reps')


def retrofit(run_dir: Path) -> None:
    cfg_path = run_dir / 'phase2' / 'train_config.json'
    if not cfg_path.exists():
        print(f"  SKIP {run_dir.name}: no phase2/train_config.json")
        return
    enabled = set(json.loads(cfg_path.read_text()).get('enabled_tasks',
                                                          ALL_TASKS))

    summaries = list((run_dir / 'phase2').rglob('cv_summary.json'))
    if not summaries:
        print(f"  SKIP {run_dir.name}: no cv_summary.json under phase2/")
        return

    for summ_path in summaries:
        data = json.loads(summ_path.read_text())
        summary = data.get('summary', {})
        changed = []
        for task in ALL_TASKS:
            if task in summary and task not in enabled:
                if summary[task].get('untrained') is True:
                    continue  # already retrofitted
                summary[task] = {'untrained': True}
                changed.append(task)
        if not changed:
            print(f"  OK   {summ_path.relative_to(run_dir)}: nothing to do "
                  f"(enabled={sorted(enabled)})")
            continue
        backup = summ_path.with_name('cv_summary.raw.json')
        if not backup.exists():
            backup.write_text(json.dumps(data, indent=2))
        data['summary'] = summary
        data['enabled_tasks'] = sorted(enabled)
        summ_path.write_text(json.dumps(data, indent=2))
        print(f"  FIX  {summ_path.relative_to(run_dir)}: marked {changed} "
              f"untrained (enabled={sorted(enabled)})")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    targets = []
    for arg in sys.argv[1:]:
        p = Path(arg)
        if '*' in arg:
            targets.extend(sorted(Path('.').glob(arg)))
        elif p.is_dir():
            targets.append(p)
    if not targets:
        print("No matching run dirs.")
        sys.exit(1)
    for d in targets:
        print(f"\n[retrofit] {d}")
        retrofit(d)


if __name__ == '__main__':
    main()
