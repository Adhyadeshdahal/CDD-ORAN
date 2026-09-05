"""Plan 007: reproducible multi-seed E2 RECOVERY envelope sweep (implemented, NOT run this phase).

Runs the FROZEN §12 seed envelope -- replicates ``0..9``, all retained -- of the E2 discovery
pipeline (NO training arm): per replicate ``r`` (``seed = r``, ``sampling_seed = 0``) it runs
``generate -> discover -> recover`` on the frozen §14 constants and aggregates a deterministic
recovery envelope plus the per-edge stability-selection frequency (§13). ``recover`` reads E2 truth
(§9), so this sweep is NOT executed during the implementation phase -- the orchestrator runs it after
review.

It reuses the Plan 004 sweep runner's generic, tested discipline machinery (atomic journal, artifact
hashing, semantic cross-artifact validators, seed-match guards, single-writer lock, deterministic
aggregation) WITHOUT modifying ``e1_slice_sweep.py`` -- only the stage chain and the aggregate are
E2-recovery-specific.

Discipline (do NOT violate): never discard a failed replicate, never change a setting mid-run, never
select a favorable subset. A failed replicate is marked failed and later replicates continue; the
process exits nonzero if any replicate failed or the aggregate is incomplete.

Usage::

    uv run python -m scripts.e2_slice_recovery_sweep --out runs/e2slice-recovery --dry-run
    uv run python -m scripts.e2_slice_recovery_sweep --out runs/e2slice-recovery
    uv run python -m scripts.e2_slice_recovery_sweep --out runs/e2slice-recovery --aggregate-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from cdd_oran.e1slice.aggregate import envelope
from cdd_oran.e2slice import SCHEMA_VERSION
from cdd_oran.e2slice.discovery import (
    FROZEN_B_PERM,
    FROZEN_PERMUTATION_SEED,
    FROZEN_Q,
    PROTOCOL_COMMIT,
)

# Reuse the Plan 004 sweep's GENERIC (plan-agnostic) discipline machinery. e1_slice_sweep.py is
# left untouched.
from scripts.e1_slice_sweep import (
    REPO_ROOT,
    Stage,
    _atomic_write_text,
    _load_json,
    _now,
    _run,
    _uv_lock_sha256,
    acquire_lock,
    load_journal,
    release_lock,
    replicate_dir,
    run_replicate,
    stage_valid,
)

PLAN = "007-e2-discovery"

# FROZEN §12/§14 matrix (recovery only, so no model block).
N_REPLICATES = 10
DATASET: dict[str, Any] = {"rows": 4000, "sampling_seed": 0, "obs_noise_scale": 0.0}
GRAPH_BLOCKS = ("overall", "ncp_kpi", "kpi_kpi")

# Candidate-graph layout for the stability table (§13). Local constants, not env truth.
NODE_NAMES_IN = ["P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "K0", "K1", "K2", "K3", "K4", "K5"]
NODE_NAMES_OUT = ["K0", "K1", "K2", "K3", "K4", "K5"]
CANDIDATE_SHAPE = (len(NODE_NAMES_OUT), len(NODE_NAMES_IN))  # (6, 14)


def replicate_seeds(r: int) -> dict[str, int]:
    """The frozen §12 per-replicate seed assignment (``seed = r``, frozen sampling base)."""
    return {"seed": int(r), "sampling_seed": int(DATASET["sampling_seed"])}


def _e2(*args: str) -> list[str]:
    return [sys.executable, "-m", "scripts.e2_slice", *args]


# --- stage argv builders (E2 recovery-only chain) ----------------------------
def _stage_generate(d: Path, r: int, s: dict[str, int]) -> list[str]:
    return _e2(
        "generate",
        "--rows", str(DATASET["rows"]),
        "--seed", str(s["seed"]),
        "--sampling-seed", str(s["sampling_seed"]),
        "--obs-noise-scale", repr(float(DATASET["obs_noise_scale"])),
        "--out", str(d),
        "--force",
    )


def _stage_discover(d: Path, r: int, s: dict[str, int]) -> list[str]:
    return _e2("discover", "--dataset", str(d), "--force")


def _stage_recover(d: Path, r: int, s: dict[str, int]) -> list[str]:
    return _e2("recover", "--dataset", str(d))


# --- semantic validators: cross-artifact provenance bindings must hold --------
def _v_discover(d: Path) -> bool:
    m = _load_json(d / "manifest.json")
    dc = _load_json(d / "discovery.json")
    return dc["dataset_hash"] == m["dataset_hash"] and dc["protocol_commit"] == PROTOCOL_COMMIT


def _v_recover(d: Path) -> bool:
    m = _load_json(d / "manifest.json")
    dc = _load_json(d / "discovery.json")
    rc = _load_json(d / "recovery.json")
    return (
        rc["dataset_hash"] == m["dataset_hash"]
        and rc["discovery_hash"] == dc["content_hash"]
        and rc["protocol_commit"] == PROTOCOL_COMMIT
    )


def build_stages() -> tuple[Stage, ...]:
    """The frozen 3-stage E2 recovery-only pipeline for one replicate (no training, no split)."""
    return (
        Stage("generate", _stage_generate, ("manifest.json", "rows.npz")),
        Stage("discover", _stage_discover, ("discovery.json",), _v_discover),
        Stage("recover", _stage_recover, ("recovery.json",), _v_recover),
    )


def frozen_settings() -> dict[str, Any]:
    return {
        "n_replicates": N_REPLICATES,
        "dataset": DATASET,
        "graph_blocks": list(GRAPH_BLOCKS),
        "protocol_commit": PROTOCOL_COMMIT,
        "b_perm": FROZEN_B_PERM,
        "permutation_seed": FROZEN_PERMUTATION_SEED,
        "q": FROZEN_Q,
        "recovery_only": True,
        "seed_matrix": {str(r): replicate_seeds(r) for r in range(N_REPLICATES)},
    }


def capture_provenance(out: Path) -> dict[str, Any]:
    """Write ``run_provenance.json`` once at launch; on resume keep the original bytes."""
    path = out / "run_provenance.json"
    if path.exists():
        return _load_json(path)
    import numpy as np

    prov = {
        "plan": PLAN,
        "launch_utc": _now(),
        "protocol_commit": PROTOCOL_COMMIT,
        "git_sha": _run("rev-parse", "HEAD"),
        "git_dirty": bool(_run("status", "--porcelain")),
        "git_status_short": _run("status", "--short"),
        "uv_lock_sha256": _uv_lock_sha256(),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "settings": frozen_settings(),
    }
    _atomic_write_text(path, json.dumps(prov, indent=2, sort_keys=True))
    return prov


def _replicate_record(out: Path, r: int, stages: tuple[Stage, ...]) -> dict[str, Any]:
    """Read one replicate's verified recovery outputs into a flat record. status success/failed."""
    rep_dir = replicate_dir(out, r)
    journal = load_journal(rep_dir, r)
    all_valid = all(stage_valid(rep_dir, st, journal) for st in stages)
    rec: dict[str, Any] = {"replicate": r, "seeds": replicate_seeds(r)}
    if not all_valid:
        failed_stage = None
        for st in stages:
            if not stage_valid(rep_dir, st, journal):
                failed_stage = st.name
                break
        j = journal.get("stages", {}).get(failed_stage, {}) if failed_stage else {}
        rec.update({"status": "failed", "failed_stage": failed_stage, "returncode": j.get("returncode")})
        return rec

    manifest = _load_json(rep_dir / "manifest.json")
    discovery = _load_json(rep_dir / "discovery.json")
    recovery = _load_json(rep_dir / "recovery.json")
    rec.update({
        "status": "succeeded",
        "graph": {
            block: {
                "precision": float(recovery["recovery"][block]["precision"]),
                "recall": float(recovery["recovery"][block]["recall"]),
                "f1": float(recovery["recovery"][block]["f1"]),
                "tp": int(recovery["recovery"][block]["tp"]),
                "fp": int(recovery["recovery"][block]["fp"]),
                "fn": int(recovery["recovery"][block]["fn"]),
            }
            for block in GRAPH_BLOCKS
        },
        "kpi_kpi_fp": int(recovery["kpi_kpi_fp"]),
        "kpi_kpi_rejection_rate": float(recovery["kpi_kpi_rejection_rate"]),
        "n_guarded": int(sum(sum(row) for row in discovery["guarded_mask"])),
        "missed": recovery["recovery"]["missed"],
        "binary_mask": [[int(v) for v in row] for row in discovery["binary_mask"]],
        "child_hashes": {
            "dataset_hash": manifest["dataset_hash"],
            "scm_hash": manifest["scm_hash"],
            "discovery_hash": discovery["content_hash"],
            "recovery_hash": recovery["content_hash"],
        },
    })
    return rec


def stability_frequency(succeeded: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-edge selection frequency across the succeeded replicates (§13: reported diagnostic)."""
    n = len(succeeded)
    k, d = CANDIDATE_SHAPE
    counts = [[0 for _ in range(d)] for _ in range(k)]
    for rec in succeeded:
        for j in range(k):
            for i in range(d):
                counts[j][i] += int(rec["binary_mask"][j][i])
    freq = [[(counts[j][i] / n if n else 0.0) for i in range(d)] for j in range(k)]
    selected = [
        {"child": NODE_NAMES_OUT[j], "parent": NODE_NAMES_IN[i],
         "count": counts[j][i], "frequency": freq[j][i]}
        for j in range(k) for i in range(d) if counts[j][i] > 0
    ]
    selected.sort(key=lambda e: (-float(e["frequency"]), str(e["child"]), str(e["parent"])))
    return {"b": n, "counts": counts, "frequency": freq, "selected_edges": selected}


def aggregate(out: Path, stages: tuple[Stage, ...] | None = None) -> dict[str, Any]:
    """Build the canonical recovery summary from persisted, verified replicate artifacts."""
    if stages is None:
        stages = build_stages()
    provenance = _load_json(out / "run_provenance.json")

    records = [_replicate_record(out, r, stages) for r in range(N_REPLICATES)]
    succeeded = [rec for rec in records if rec["status"] == "succeeded"]
    failed = [
        {"replicate": rec["replicate"], "seeds": rec["seeds"],
         "failed_stage": rec.get("failed_stage"), "returncode": rec.get("returncode")}
        for rec in records
        if rec["status"] != "succeeded"
    ]
    complete = len(succeeded) == N_REPLICATES

    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "plan": PLAN,
        "protocol_commit": PROTOCOL_COMMIT,
        "provenance": provenance,
        "settings": frozen_settings(),
        "n_replicates_planned": N_REPLICATES,
        "n_replicates_succeeded": len(succeeded),
        "complete": complete,
        "failed_replicates": failed,
        "replicates": records,
    }

    if not succeeded:
        summary["graph_recovery"] = {}
        summary["kpi_kpi"] = {}
        summary["stability_selection"] = stability_frequency([])
        return summary

    summary["graph_recovery"] = {
        block: {
            metric: envelope([rec["graph"][block][metric] for rec in succeeded])
            for metric in ("precision", "recall", "f1")
        }
        for block in GRAPH_BLOCKS
    }
    summary["kpi_kpi"] = {
        "fp_count": envelope([float(rec["kpi_kpi_fp"]) for rec in succeeded]),
        "rejection_rate": envelope([float(rec["kpi_kpi_rejection_rate"]) for rec in succeeded]),
    }
    summary["stability_selection"] = stability_frequency(succeeded)
    return summary


def write_outputs(out: Path, summary: dict[str, Any]) -> tuple[Path, Path]:
    """Write ``replicates.jsonl`` and canonical ``summary.json`` (sorted-key, stable floats)."""
    jsonl = out / "replicates.jsonl"
    lines = [json.dumps(rec, sort_keys=True) for rec in summary["replicates"]]
    _atomic_write_text(jsonl, "\n".join(lines) + ("\n" if lines else ""))

    summ = out / "summary.json"
    _atomic_write_text(summ, json.dumps(summary, indent=2, sort_keys=True))
    return jsonl, summ


def print_dry_run(out: Path) -> None:
    dirs = [replicate_dir(out, r) for r in range(N_REPLICATES)]
    unique = sorted({str(d) for d in dirs})
    print(f"DRY RUN: {len(unique)} unique run dirs (of {N_REPLICATES} planned)")
    for r in range(N_REPLICATES):
        s = replicate_seeds(r)
        print(f"  {replicate_dir(out, r)}  seed={s['seed']} sampling_seed={s['sampling_seed']}")
    print("FROZEN SETTINGS (E2 recovery-only):")
    print(json.dumps(frozen_settings(), indent=2, sort_keys=True))
    if len(unique) != N_REPLICATES:
        raise SystemExit(f"collision: only {len(unique)} unique dirs for {N_REPLICATES} replicates")


def run_all(out: Path, force_unlock: bool) -> int:
    stages = build_stages()
    lock = acquire_lock(out, force_unlock)
    try:
        capture_provenance(out)
        statuses = [run_replicate(out, r, stages) for r in range(N_REPLICATES)]
        summary = aggregate(out, stages)
        jsonl, summ = write_outputs(out, summary)
    finally:
        release_lock(lock)

    failed = [s for s in statuses if s["status"] != "succeeded"]
    print("\n==== E2 recovery sweep summary ====")
    print(f"  succeeded {summary['n_replicates_succeeded']}/{N_REPLICATES}, complete={summary['complete']}")
    print(f"  -> {summ}")
    print(f"  -> {jsonl}")
    for s in failed:
        print(f"  FAILED replicate-{s['replicate']:02d} at stage '{s['failed_stage']}' (rc {s['returncode']})")
    if failed or not summary["complete"]:
        return 1
    return 0


def aggregate_only(out: Path) -> int:
    summary = aggregate(out)
    jsonl, summ = write_outputs(out, summary)
    print(f"aggregated {summary['n_replicates_succeeded']}/{N_REPLICATES} "
          f"(complete={summary['complete']}) -> {summ}")
    return 0 if summary["complete"] else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plan 007: reproducible E2 recovery sweep (not run this phase).")
    p.add_argument("--out", required=True, help="output sweep directory (holds replicate-00..09)")
    p.add_argument("--dry-run", action="store_true", help="print the 10 run dirs + frozen settings")
    p.add_argument("--aggregate-only", action="store_true",
                   help="re-aggregate existing replicate artifacts (determinism check)")
    p.add_argument("--force-unlock", action="store_true", help="break a stale .sweep.lock")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.out)
    if not out.is_absolute():
        out = (REPO_ROOT / out).resolve()
    if args.dry_run:
        print_dry_run(out)
        return 0
    if args.aggregate_only:
        return aggregate_only(out)
    return run_all(out, args.force_unlock)


if __name__ == "__main__":
    raise SystemExit(main())
