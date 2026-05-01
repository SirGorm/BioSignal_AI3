"""
Renumber dataset_clean for 3 recordings with confirmed false-start sets.

Per user 2026-05-01 + joint-file unix-time analysis:

  rec_006: drop SOURCE sets [4, 6, 7]
           - Set 4 (6.69s, 0 reps) — false start
           - Set 6 (1.98s) — auto-trigger glitch right at set 5's end
           - Set 7 (15.25s, markers ok) — NO Kinect tracking (file 04..12
             cover source sets [1,2,3,5,8,9,10,11,12,13,14,15])
           Joint files: NO RENAME — recording_NN_joints.json content is
             already aligned to NEW set N because source's filename was
             based on capture-index, not source set_number.

  rec_008: drop SOURCE set [8]
           - Set 8 (15.27s, 2 reps) — aborted; restarted as set 9.
           Joint files: NO RENAME (same reason as rec_006).

  rec_009: drop SOURCE sets [10, 11]
           - Set 10 (eff_fps=4.8) — heavy Kinect tracking failure.
           - Set 11 (eff_fps=8.7) — same.
           Joint files DO need rename: recording_10/11 deleted;
             recording_12 → 10, recording_13 → 11, recording_14 → 12.

Modifies dataset_clean/ in-place. Backup of every modified file goes to
dataset_clean/_backup_pre_renumber/<rel>.

Idempotent: re-running on already-fixed dirs is a no-op (set_number contiguity
check skips). Originals (dataset/, dataset_memory/) are READ-ONLY.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALIGNED = ROOT / "dataset_clean"
BACKUP = ALIGNED / "_backup_pre_renumber"

# Per-recording action plan.
PLAN = {
    "006": {
        "drop_source_sets": [4, 6, 7],
        "joint_file_action": "no_rename",   # file content already maps correctly
        "note": "Source joint files cover sets [1,2,3,5,8,9,10,11,12,13,14,15]; renaming would corrupt content alignment.",
    },
    "008": {
        "drop_source_sets": [8],
        "joint_file_action": "no_rename",
        "note": "Source joint files cover sets [1..7, 9..13]; file 08 already contains source set 9 data.",
    },
    "009": {
        "drop_source_sets": [10, 11],
        "joint_file_action": "rename",      # files map 1-to-1 to source set_number
        "note": "Source joint files match set_number 1-to-1; delete files for dropped sets, then renumber the rest.",
    },
}


def _backup(p: Path) -> None:
    """Copy p to BACKUP/<relative-to-ALIGNED> before editing."""
    rel = p.relative_to(ALIGNED)
    dst = BACKUP / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(p, dst)


def renumber_recording(rec: str, drop: list[int], joint_action: str) -> dict:
    rec_dir = ALIGNED / f"recording_{rec}"
    if not rec_dir.exists():
        return {"recording": rec, "status": "missing"}

    md_p = rec_dir / "metadata.json"
    mk_p = rec_dir / "markers.json"
    if not md_p.exists() or not mk_p.exists():
        return {"recording": rec, "status": "missing_md_or_mk"}

    md = json.loads(md_p.read_text())
    sets = md.get("kinect_sets", [])
    src_set_nums = sorted(s["set_number"] for s in sets)

    # Idempotency: if all set_numbers are already contiguous 1..N AND the dropped
    # ones are NOT present, skip.
    if all(sn not in src_set_nums for sn in drop):
        return {"recording": rec, "status": "already_renumbered_or_no_op",
                "current_set_numbers": src_set_nums}

    # Build mapping: source_set_number -> new_set_number
    keep = [sn for sn in src_set_nums if sn not in drop]
    mapping = {old: new for new, old in enumerate(keep, start=1)}

    # 1. Update metadata.kinect_sets
    _backup(md_p)
    new_sets = []
    for s in sets:
        sn = s["set_number"]
        if sn in drop:
            continue
        s2 = dict(s)
        s2["set_number"] = mapping[sn]
        new_sets.append(s2)
    md["kinect_sets"] = sorted(new_sets, key=lambda x: x["set_number"])
    md["total_kinect_sets"] = len(new_sets)
    md_p.write_text(json.dumps(md, indent=2))

    # 2. Update markers.json
    _backup(mk_p)
    raw = json.loads(mk_p.read_text())
    if isinstance(raw, dict):
        markers = raw.get("markers", []); wrap = True
    else:
        markers = raw; wrap = False

    pat_start = re.compile(r"^Set:(\d+)_Start$")
    pat_end = re.compile(r"^Set_(\d+)_End$")
    pat_rep = re.compile(r"^Set:(\d+)_Rep:(\d+)$")
    new_markers = []
    dropped_labels = []
    renumbered_count = 0
    for e in markers:
        lbl = e.get("label", "")
        m = pat_start.match(lbl) or pat_end.match(lbl)
        m_rep = pat_rep.match(lbl)
        sn = None
        if m: sn = int(m.group(1))
        elif m_rep: sn = int(m_rep.group(1))
        if sn is None:
            new_markers.append(e); continue
        if sn in drop:
            dropped_labels.append(lbl); continue
        new_sn = mapping.get(sn)
        if new_sn is None:
            dropped_labels.append(lbl); continue
        e2 = dict(e)
        if pat_start.match(lbl):
            e2["label"] = f"Set:{new_sn}_Start"
        elif pat_end.match(lbl):
            e2["label"] = f"Set_{new_sn}_End"
        elif pat_rep.match(lbl):
            rep_num = pat_rep.match(lbl).group(2)
            e2["label"] = f"Set:{new_sn}_Rep:{rep_num}"
        if e2["label"] != lbl:
            renumbered_count += 1
        new_markers.append(e2)

    if wrap:
        raw["markers"] = new_markers
        raw["total_markers"] = len(new_markers)
        out_mk = raw
    else:
        out_mk = new_markers
    mk_p.write_text(json.dumps(out_mk, indent=2))

    # 3. Joint files
    joint_renames: list[tuple[str, str]] = []
    joint_deletes: list[str] = []
    if joint_action == "rename":
        # Source joint files map by source set_number.
        # Step 1: delete files for dropped sets.
        for sn in drop:
            jp = rec_dir / f"recording_{sn:02d}_joints.json"
            if jp.exists():
                _backup(jp)
                jp.unlink()
                joint_deletes.append(jp.name)
        # Step 2: rename remaining keep[k] → mapping[keep[k]]. Two-phase to
        # avoid name collisions (rename to _tmp suffix first, then strip).
        tmp_pairs = []
        for old in keep:
            new = mapping[old]
            if old == new:
                continue
            old_p = rec_dir / f"recording_{old:02d}_joints.json"
            tmp_p = rec_dir / f"recording_{new:02d}_joints.json.TMP"
            if old_p.exists():
                _backup(old_p)
                old_p.rename(tmp_p)
                tmp_pairs.append((old, new, tmp_p))
        for old, new, tmp_p in tmp_pairs:
            new_p = rec_dir / f"recording_{new:02d}_joints.json"
            tmp_p.rename(new_p)
            joint_renames.append((f"recording_{old:02d}_joints.json",
                                   f"recording_{new:02d}_joints.json"))
    # else: joint_action == "no_rename" — leave files as-is.

    # 4. Regenerate set_quality.json (re-import build script's flagger, but to
    # avoid coupling we inline a small version here).
    _regenerate_set_quality(rec_dir, rec)

    return {
        "recording": rec,
        "status": "renumbered",
        "drop_source_sets": drop,
        "kept_source_sets": keep,
        "mapping": mapping,
        "n_kinect_sets_after": len(new_sets),
        "n_markers_dropped": len(dropped_labels),
        "n_markers_renumbered": renumbered_count,
        "joint_renames": joint_renames,
        "joint_deletes": joint_deletes,
        "joint_action": joint_action,
    }


def _regenerate_set_quality(rec_dir: Path, rec: str) -> None:
    md = json.loads((rec_dir / "metadata.json").read_text())
    mk_raw = json.loads((rec_dir / "markers.json").read_text())
    mk = mk_raw if isinstance(mk_raw, list) else mk_raw.get("markers", [])

    rep_counts: dict[int, int] = {}
    rep_present = False
    for e in mk:
        m = re.match(r"^Set:(\d+)_Rep:\d+$", e.get("label", ""))
        if m:
            sn = int(m.group(1))
            rep_counts[sn] = rep_counts.get(sn, 0) + 1
            rep_present = True

    out = {"recording": rec,
           "policy": "flag-only; downstream labeler decides ok_for_training",
           "thresholds": {"very_short_s": 10.0, "short_s": 15.0, "low_reps": 3},
           "rep_markers_present": rep_present,
           "renumbered_at_utc": datetime.now(timezone.utc).isoformat(),
           "sets": []}
    for s in md.get("kinect_sets", []):
        sn = s["set_number"]
        dur = float(s["end_unix_time"] - s["start_unix_time"])
        nr = rep_counts.get(sn, 0)
        joint_p = rec_dir / f"recording_{sn:02d}_joints.json"
        has_joint = joint_p.exists()
        flags = []
        if nr == 0 and rep_present:
            flags.append("no_reps_in_markers")
        if not rep_present:
            flags.append("rep_markers_unavailable_in_protocol")
        if dur < 10.0: flags.append("very_short_<10s")
        elif dur < 15.0: flags.append("short_<15s")
        if 0 < nr < 3: flags.append("low_reps_<3")
        if not has_joint: flags.append("no_joint_file")
        ok = (dur >= 10.0 and has_joint
              and (nr >= 3 or not rep_present)
              and "no_reps_in_markers" not in flags)
        out["sets"].append({
            "set_number": sn, "duration_s": round(dur, 3),
            "rep_count_markers": nr, "has_joint_file": has_joint,
            "flags": flags, "ok_for_training": ok,
        })
    (rec_dir / "set_quality.json").write_text(json.dumps(out, indent=2))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"=== renumber_dataset_clean ===\nBackup root: {BACKUP}\n")

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "plan": PLAN,
        "recordings": {},
    }
    for rec, plan in PLAN.items():
        print(f"--- recording_{rec} ---")
        print(f"   drop source sets: {plan['drop_source_sets']}")
        print(f"   joint file action: {plan['joint_file_action']}")
        rep = renumber_recording(rec, plan["drop_source_sets"], plan["joint_file_action"])
        report["recordings"][rec] = rep
        if rep["status"] == "renumbered":
            print(f"   kept_source: {rep['kept_source_sets']}")
            print(f"   mapping (src→new): {rep['mapping']}")
            print(f"   markers: {rep['n_markers_dropped']} dropped, {rep['n_markers_renumbered']} renumbered")
            if rep["joint_deletes"]:
                print(f"   joint deletes: {rep['joint_deletes']}")
            if rep["joint_renames"]:
                for old, new in rep["joint_renames"]:
                    print(f"   joint rename: {old} → {new}")
        else:
            print(f"   status: {rep['status']}")
        print()

    (ALIGNED / "_renumber_report.json").write_text(json.dumps(report, indent=2))
    print(f"=== DONE ===\nReport: {ALIGNED/'_renumber_report.json'}")


if __name__ == "__main__":
    main()
