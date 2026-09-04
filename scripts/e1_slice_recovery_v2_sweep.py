"""Plan 006 phase 2b: reproducible multi-seed E1 v2 RECOVERY envelope sweep.

Runs the FROZEN Plan 004 seed matrix -- replicates ``0..9``, all retained -- but only the
RECOVERY-relevant stages of the v2 partial-correlation pipeline (NO MLP training): per replicate
``r`` (``env_seed = weight_seed = r``, ``split_seed = 0``; dataset 48 ep / 16 steps / warmup 2 /
noise 0; split test_fraction 0.25) it runs ``generate -> split -> discover-v2 -> recover-v2`` and
aggregates a deterministic recovery envelope plus the per-edge stability-selection frequency (§8).

This closes the provenance gap flagged by the adversarial review: the phase-2 10-seed numbers were
produced ad-hoc; this committed driver regenerates them from the repo. It reuses the Plan 004 sweep
runner's generic, tested discipline machinery (atomic journal, artifact hashing, semantic
cross-artifact validators, seed-match guards, single-writer lock, deterministic aggregation) WITHOUT
modifying ``e1_slice_sweep.py`` -- only the stage chain and the aggregate are v2-recovery-specific.

Discipline (do NOT violate): never discard a failed replicate, never change a setting mid-run, never
select a favorable subset. A failed replicate is marked failed and later replicates continue; the
process exits nonzero if any replicate failed or the aggregate is incomplete.

Usage::

    uv run python -m scripts.e1_slice_recovery_v2_sweep --out runs/e1slice-v2-recovery --dry-run
    uv run python -m scripts.e1_slice_recovery_v2_sweep --out runs/e1slice-v2-recovery
    uv run python -m scripts.e1_slice_recovery_v2_sweep --out runs/e1slice-v2-recovery --aggregate-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from cdd_oran.e1slice import SCHEMA_VERSION
from cdd_oran.e1slice.aggregate import envelope
from cdd_oran.e1slice.discovery_v2 import PROTOCOL_COMMIT

# Reuse the Plan 004 sweep's GENERIC (plan-agnostic) discipline machinery. These are provenance /
# resumability primitives, not Plan 004 policy; e1_slice_sweep.py is left untouched.
from scripts.e1_slice_sweep import (
    REPO_ROOT,
    Stage,
    _atomic_write_text,
    _e1,
    _load_json,
    _now,
    _run,
    _uv_lock_sha256,
    _v_split,
    acquire_lock,
    load_journal,
    release_lock,
    replicate_dir,
    replicate_seeds,
    run_replicate,
    stage_valid,
)

PLAN = "006-e1-kpi-kpi-discovery"

# FROZEN matrix (identical seed/dataset/split policy to Plan 004; recovery only, so no model block).
N_REPLICATES = 10
DATASET: dict[str, Any] = {"episodes": 48, "steps": 16, "warmup": 2, "obs_noise_scale": 0.0}
SPLIT: dict[str, Any] = {"test_fraction": 0.25, "split_seed": 0}
GRAPH_BLOCKS = ("overall", "ncp_kpi", "kpi_kpi")

# Candidate-graph layout for the stability table (§8). Local constants, not env truth.
NODE_NAMES_IN = ["P0", "P1", "P2", "P3", "K0", "K1", "K2", "K3"]
NODE_NAMES_OUT = ["K0", "K1", "K2", "K3"]
CANDIDATE_SHAPE = (len(NODE_NAMES_OUT), len(NODE_NAMES_IN))


# --- stage argv builders (recovery-only v2 chain) ----------------------------
def _stage_generate(d: Path, r: int, s: dict[str, int]) -> list[str]:
    return _e1(
        "generate",
        "--episodes", str(DATASET["episodes"]),
        "--steps", str(DATASET["steps"]),
        "--warmup", str(DATASET["warmup"]),
        "--seed", str(s["env_seed"]),
        "--obs-noise-scale", repr(float(DATASET["obs_noise_scale"])),
        "--out", str(d),
        "--force",
    )


def _stage_split(d: Path, r: int, s: dict[str, int]) -> list[str]:
    return _e1(
        "split", "--dataset", str(d),
        "--test-fraction", repr(float(SPLIT["test_fraction"])),
        "--split-seed", str(s["split_seed"]),
        "--force",
    )


def _stage_discover_v2(d: Path, r: int, s: dict[str, int]) -> list[str]:
    return _e1("discover-v2", "--dataset", str(d), "--force")


def _stage_recover_v2(d: Path, r: int, s: dict[str, int]) -> list[str]:
    return _e1("recover-v2", "--dataset", str(d))


# --- semantic validators: cross-artifact provenance bindings must hold --------
def _v_discover_v2(d: Path) -> bool:
    m = _load_json(d / "manifest.json")
    sp = _load_json(d / "split.json")
    dc = _load_json(d / "discovery_v2.json")
    return dc["dataset_hash"] == m["dataset_hash"] and dc["split_hash"] == sp["split_hash"]


def _v_recover_v2(d: Path) -> bool:
    m = _load_json(d / "manifest.json")
    sp = _load_json(d / "split.json")
    dc = _load_json(d / "discovery_v2.json")
    rc = _load_json(d / "recovery_v2.json")
    return (
        rc["dataset_hash"] == m["dataset_hash"]
        and rc["split_hash"] == sp["split_hash"]
        and rc["discovery_hash"] == dc["content_hash"]
        and rc["protocol_commit"] == PROTOCOL_COMMIT
    )


def build_stages() -> tuple[Stage, ...]:
    """The frozen 4-stage v2 recovery-only pipeline for one replicate (no training)."""
    return (
        Stage("generate", _stage_generate, ("manifest.json", "rows.npz")),
        Stage("split", _stage_split, ("split.json",), _v_split),
        Stage("discover-v2", _stage_discover_v2, ("discovery_v2.json",), _v_discover_v2),
        Stage("recover-v2", _stage_recover_v2, ("recovery_v2.json",), _v_recover_v2),
    )


def frozen_settings() -> dict[str, Any]:
    return {
        "n_replicates": N_REPLICATES,
        "dataset": DATASET,
        "split": SPLIT,
        "graph_blocks": list(GRAPH_BLOCKS),
        "protocol_commit": PROTOCOL_COMMIT,
        "recovery_only": True,
        "seed_matrix": {str(r): replicate_seeds(r) for r in range(N_REPLICATES)},
    }


def capture_provenance(out: Path) -> dict[str, Any]:
    """Write ``run_provenance.json`` once at launch; on resume keep the original bytes.

    Aggregation reads this file (never wall-clock ``now()``), so re-running aggregation reproduces
    byte-identical summary content.
    """
    path = out / "run_provenance.json"
    if path.exists():
        return _load_json(path)
    import numpy as np
    import torch

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
        "torch_version": torch.__version__,
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
    discovery = _load_json(rep_dir / "discovery_v2.json")
    recovery = _load_json(rep_dir / "recovery_v2.json")
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
        "missed": recovery["recovery"]["missed"],
        "binary_mask": [[int(v) for v in row] for row in discovery["binary_mask"]],
        "child_hashes": {
            "dataset_hash": manifest["dataset_hash"],
            "scm_hash": manifest["scm_hash"],
            "split_hash": recovery["split_hash"],
            "discovery_v2_hash": discovery["content_hash"],
            "recovery_v2_hash": recovery["content_hash"],
        },
    })
    return rec


def stability_frequency(succeeded: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-edge selection frequency across the succeeded replicates (§8: reported diagnostic).

    Pure integer/rational arithmetic (count / B) so the summary is byte-stable; never the selector.
    """
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
    """Build the canonical recovery summary from persisted, verified replicate artifacts.

    Deterministic: every number derives from on-disk artifacts and the persisted provenance file;
    no wall-clock is read. Requires the exact frozen seed matrix (all of ``0..9`` succeeded) for a
    ``complete`` summary.
    """
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
        summary["stability_selection"] = stability_frequency([])
        return summary

    summary["graph_recovery"] = {
        block: {
            metric: envelope([rec["graph"][block][metric] for rec in succeeded])
            for metric in ("precision", "recall", "f1")
        }
        for block in GRAPH_BLOCKS
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
        print(f"  {replicate_dir(out, r)}  env_seed={s['env_seed']} split_seed={s['split_seed']}")
    print("FROZEN SETTINGS (recovery-only v2):")
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
    print("\n==== E1 v2 recovery sweep summary ====")
    print(f"  succeeded {summary['n_replicates_succeeded']}/{N_REPLICATES}, complete={summary['complete']}")
    print(f"  -> {summ}")
    print(f"  -> {jsonl}")
    for s in failed:
        print(f"  FAILED replicate-{s['replicate']:02d} at stage '{s['failed_stage']}' (rc {s['returncode']})")
    # Exit nonzero on ANY replicate failure OR an incomplete aggregate.
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
    p = argparse.ArgumentParser(description="Plan 006 phase 2b: reproducible E1 v2 recovery sweep.")
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
