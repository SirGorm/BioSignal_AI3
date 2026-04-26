"""
Parse markers.json for set/rep timing ground truth.

markers.json structure (all recordings):
  {
    "markers": [
      {"unix_time": <float>, "time": <float>, "label": "<label>", "color": "<str>"},
      ...
    ],
    "total_markers": <int>
  }

Label patterns:
  "Set:N_Start"   — set N begins (N is 1-indexed)
  "Set_N_End"     — set N ends   (note: Start uses colon, End uses underscore)
  "Set:N_Rep:M"   — rep M of set N (not present in early recordings like rec_001)
  "Rest:K"        — rest markers (present in some recordings, not used for labeling)

Some recordings have extra/short sets beyond the canonical 12.  The caller
is responsible for deciding which sets map to Participants.xlsx slots.

References
----------
- Using annotated event markers as ground truth for set/rep detection is
  methodologically superior to accelerometer-based segmentation when markers
  are available (analogous to annotation-based ground truth in activity
  recognition: Bulling et al. 2014, IEEE Pervasive Computing).
"""

from __future__ import annotations

import json
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class RepMarker:
    __slots__ = ("set_num", "rep_num", "unix_time")

    def __init__(self, set_num: int, rep_num: int, unix_time: float):
        self.set_num = set_num
        self.rep_num = rep_num
        self.unix_time = unix_time

    def __repr__(self):
        return f"RepMarker(set={self.set_num}, rep={self.rep_num}, t={self.unix_time:.3f})"


class SetMarker:
    __slots__ = ("set_num", "start_unix", "end_unix", "rep_markers")

    def __init__(
        self,
        set_num: int,
        start_unix: float,
        end_unix: float,
        rep_markers: list[RepMarker],
    ):
        self.set_num = set_num
        self.start_unix = start_unix
        self.end_unix = end_unix
        self.rep_markers = rep_markers

    @property
    def duration_s(self) -> float:
        return self.end_unix - self.start_unix

    @property
    def n_reps(self) -> int:
        return len(self.rep_markers)

    def __repr__(self):
        return (
            f"SetMarker(set={self.set_num}, "
            f"dur={self.duration_s:.1f}s, reps={self.n_reps})"
        )


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

_RE_START = re.compile(r"Set:(\d+)_Start")
_RE_END = re.compile(r"Set_(\d+)_End")
_RE_REP = re.compile(r"Set:(\d+)_Rep:(\d+)")


def parse_markers(markers_path: Path) -> list[SetMarker]:
    """Parse markers.json and return a list of SetMarker objects, sorted by set_num.

    Parameters
    ----------
    markers_path: Path to the recording's markers.json file.

    Returns
    -------
    list[SetMarker] sorted by set_num.

    Raises
    ------
    FileNotFoundError if the file does not exist.
    ValueError if a set has a Start but no End or vice-versa.
    """
    if not markers_path.exists():
        raise FileNotFoundError(f"markers.json not found: {markers_path}")

    with open(markers_path) as fh:
        data = json.load(fh)

    raw_markers = data.get("markers", [])

    # Collect starts, ends, reps
    starts: dict[int, float] = {}
    ends: dict[int, float] = {}
    reps: dict[int, list[RepMarker]] = {}

    for m in raw_markers:
        label = m["label"]
        unix_t = float(m["unix_time"])

        ms = _RE_START.match(label)
        if ms:
            starts[int(ms.group(1))] = unix_t
            continue

        me = _RE_END.match(label)
        if me:
            ends[int(me.group(1))] = unix_t
            continue

        mr = _RE_REP.match(label)
        if mr:
            sn, rn = int(mr.group(1)), int(mr.group(2))
            reps.setdefault(sn, []).append(RepMarker(sn, rn, unix_t))
            continue
        # Other labels (Rest:K) are ignored

    # Build SetMarker list
    all_set_nums = sorted(set(starts.keys()) | set(ends.keys()))
    set_markers: list[SetMarker] = []

    for sn in all_set_nums:
        if sn not in starts:
            raise ValueError(
                f"{markers_path}: Set {sn} has End marker but no Start marker."
            )
        if sn not in ends:
            raise ValueError(
                f"{markers_path}: Set {sn} has Start marker but no End marker."
            )

        rep_list = sorted(reps.get(sn, []), key=lambda r: r.rep_num)
        set_markers.append(
            SetMarker(sn, starts[sn], ends[sn], rep_list)
        )

    return sorted(set_markers, key=lambda s: s.set_num)


def select_canonical_sets(
    all_sets: list[SetMarker],
    expected_n: int = 12,
    min_reps: int = 5,
    min_duration_s: float = 15.0,
) -> list[SetMarker]:
    """Filter to the canonical N sets from a recording that may have extras.

    Strategy (applied in order):
    1. Discard sets with exactly 0 reps AND duration < 5 s (truly empty
       accidental triggers — nothing was recorded).
    2. If still more than expected_n, discard sets with fewer than min_reps
       reps AND duration < min_duration_s (aborted practice attempts).
    3. If still more than expected_n, remove the fewest-rep / shortest sets
       until exactly expected_n remain.
    4. Return whatever is left (caller checks length and halts if < expected_n).

    This two-stage approach correctly handles rec_005 where set 10 has 0 reps
    and 3.5 s (true empty set) while sets 4-6 have 3 reps each at 11-17 s
    (legitimate short sets of one exercise block).

    Parameters
    ----------
    all_sets:       All SetMarker objects from parse_markers().
    expected_n:     Target canonical set count (default 12).
    min_reps:       Threshold for stage-2 filtering (default 5).
    min_duration_s: Duration threshold for stage-2 filtering (default 15 s).

    Returns
    -------
    list[SetMarker] of length <= expected_n, sorted by set_num.
    """
    # Stage 1: remove truly empty sets (0 reps + very short duration)
    filtered = [
        s for s in all_sets
        if not (s.n_reps == 0 and s.duration_s < 5.0)
    ]

    # Stage 2: if still over target, remove clearly aborted sets
    if len(filtered) > expected_n:
        filtered = [
            s for s in filtered
            if not (s.n_reps < min_reps and s.duration_s < min_duration_s)
        ]

    # Stage 3: if still over target, iteratively remove worst set
    while len(filtered) > expected_n:
        worst = min(filtered, key=lambda s: (s.n_reps, s.duration_s))
        filtered.remove(worst)

    return sorted(filtered, key=lambda s: s.set_num)
