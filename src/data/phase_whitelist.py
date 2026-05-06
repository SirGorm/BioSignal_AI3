"""Per-(recording, set) whitelist for the phase head.

Phase labels are derived from Kinect joint angles, which are noisy on some sets
(skeleton occlusion, frame drops, miscalibrated angle range). When a curated
whitelist is supplied, only listed (recording_id, set_number) pairs contribute
to the phase loss/eval; all other heads (exercise, fatigue, reps) keep using
the full data.

CSV format:

    recording_id,set_number
    recording_006,1
    recording_006,3
    recording_007,2

- Header row is required.
- One row per included set. Sets not listed are masked out for phase only.
- Lines starting with `#` are skipped as comments.
- `set_number` may be int or float-like; it is normalised to int internally.
- An empty whitelist file (header only) means "no sets included" — phase
  head will not train. A `None` path means "no whitelist" — full data used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Set, Tuple

import pandas as pd


WhitelistKey = Tuple[str, int]


def load_phase_whitelist(path: Optional[Path]) -> Optional[Set[WhitelistKey]]:
    """Load a phase-quality whitelist CSV.

    Returns None if path is None (signal: no filtering). Returns a set of
    (recording_id, set_number) tuples otherwise. Raises FileNotFoundError
    if the path is given but missing, ValueError if columns are wrong.
    """
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"phase whitelist not found: {p}")
    df = pd.read_csv(p, comment='#')
    required = {'recording_id', 'set_number'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"phase whitelist {p} missing columns: {missing}. "
            f"Required header: recording_id,set_number"
        )
    out: Set[WhitelistKey] = set()
    for _, row in df.iterrows():
        rid = str(row['recording_id']).strip()
        if not rid or rid.lower() == 'nan':
            continue
        try:
            s = int(round(float(row['set_number'])))
        except (TypeError, ValueError):
            continue
        out.add((rid, s))
    return out


def whitelist_mask(
    recording_ids,
    set_numbers,
    whitelist: Optional[Set[WhitelistKey]],
):
    """Return a boolean array: True where (recording_id, set_number) is in
    `whitelist`. If whitelist is None, returns an all-True array (no filter).

    `recording_ids` and `set_numbers` may be array-likes or pandas Series.
    NaN set_numbers are treated as not-in-whitelist.
    """
    import numpy as np

    n = len(recording_ids)
    if whitelist is None:
        return np.ones(n, dtype=bool)

    rids = pd.Series(recording_ids).astype(str).to_numpy()
    sets_raw = pd.to_numeric(pd.Series(set_numbers), errors='coerce').to_numpy()
    out = np.zeros(n, dtype=bool)
    for i in range(n):
        s = sets_raw[i]
        if s != s:  # NaN
            continue
        out[i] = (rids[i], int(round(s))) in whitelist
    return out
