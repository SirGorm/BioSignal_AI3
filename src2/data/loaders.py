"""Thin re-export of the existing biosignal CSV loaders.

`src/data/loaders.py` is already library-only (pandas + scipy) — there is no
duplication to win by rewriting it. Importing from `src.*` is the cleanest
form of "do not modify existing code": we depend on it instead.
"""

from src.data.loaders import (  # noqa: F401
    load_biosignal,
    load_temperature,
    load_imu,
    load_metadata,
    load_all_biosignals,
)
