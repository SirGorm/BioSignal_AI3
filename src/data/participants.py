"""
Parser for the dual-row Participants.xlsx schema.

Schema (one block per recording):
  Row A: | <recording_num> | <Name> | <exercise_set1> | ... | <exercise_set12> |
  Row B: | NaN             | fatigue | <rpe_set1>     | ... | <rpe_set12>      |

Recordings 15-21 have no data yet (RPE = NaN throughout). We only load
recordings that have at least one non-NaN RPE value.

All subject names are preserved exactly as written in the sheet (e.g. 'Tias',
'lucas 2') — the caller must use Name: as the canonical subject_id.

Returns
-------
load_participants() -> dict mapping int recording_num to a dict with keys:
  'subject_id': str  (from Name: column)
  'exercises':  list[str]  length 12
  'rpe':        list[int | None]  length 12 (None when NaN in sheet)
"""

from __future__ import annotations
from pathlib import Path

import pandas as pd


_SET_COLS = [f"set{i}" for i in range(1, 13)]


def load_participants(xlsx_path: Path) -> dict[int, dict]:
    """Parse Participants.xlsx and return per-recording metadata.

    Parameters
    ----------
    xlsx_path: Path to dataset/Participants/Participants.xlsx

    Returns
    -------
    dict: {recording_num (int): {'subject_id': str, 'exercises': list,
                                  'rpe': list}}
    Only recordings with a valid Name and at least some data are included.
    Recordings 15+ (no data collected yet) are excluded.
    """
    raw = pd.read_excel(xlsx_path, header=None)

    # Row 0 is the header row: Recording: | Name: | set1 | ... | set12
    # Subsequent rows alternate: exercise row / fatigue row
    result: dict[int, dict] = {}

    for row_idx in range(1, len(raw), 2):
        exercise_row = raw.iloc[row_idx]
        if row_idx + 1 >= len(raw):
            break
        fatigue_row = raw.iloc[row_idx + 1]

        # Recording number is in col 0 of the exercise row
        rec_num_raw = exercise_row.iloc[0]
        try:
            rec_num = int(rec_num_raw)
        except (ValueError, TypeError):
            continue  # skip malformed rows

        name = exercise_row.iloc[1]
        if pd.isna(name):
            continue  # no participant data yet

        name = str(name).strip()

        # Extract exercises (cols 2-13)
        exercises = []
        for col_idx in range(2, 14):
            val = exercise_row.iloc[col_idx]
            if pd.isna(val):
                exercises.append(None)
            else:
                exercises.append(str(val).strip().lower())

        # Extract RPE (fatigue row, cols 2-13)
        rpe = []
        for col_idx in range(2, 14):
            val = fatigue_row.iloc[col_idx]
            if pd.isna(val):
                rpe.append(None)
            else:
                try:
                    rpe.append(int(float(val)))
                except (ValueError, TypeError):
                    rpe.append(None)

        result[rec_num] = {
            "subject_id": name,
            "exercises": exercises,
            "rpe": rpe,
        }

    return result


def get_recording_info(
    participants: dict[int, dict], recording_num: int
) -> dict | None:
    """Return the participant info for one recording, or None if not found."""
    return participants.get(recording_num)
