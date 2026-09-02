"""Plan 004: resumable multi-seed E1 envelope sweep.

Runs the FROZEN Plan 004 matrix -- replicates ``0..9``, all retained -- over the Plan 003
three-arm E1 slice pipeline, then aggregates a deterministic effect *envelope* (no
hypothesis-significance claim). For replicate ``r``: ``env_seed=r``, ``weight_seed=r``,
``split_seed=0``; dataset 48 ep / 16 steps / warmup 2 / noise 0; split test_fraction 0.25;
model hidden (16,) / lr 1e-2 / 300 epochs / batch 64; arms oracle, dense, discovered.

Discipline (do NOT violate): never discard a failed replicate, never change a setting
mid-run, never select a favorable subset. A failed replicate is marked failed and later
replicates continue; the process exits nonzero if any replicate failed.

Resumability: each replicate is a chain of stages (generate -> split -> discover -> train
-> eval -> verify -> recover). After every stage an *atomic* journal record is written with
the command, UTC start/end, return code, log path, and the SHA-256 of each output artifact.
On resume a stage is skipped ONLY when its journal record returned 0, every recorded artifact
still hashes to the journaled digest, AND the cross-artifact provenance bindings hold -- never
from file existence alone. The first stage that fails to validate forces itself and all later
stages in that replicate to re-run.

Usage::

    uv run python -m scripts.e1_slice_sweep --out runs/e1slice-v2-envelope --dry-run
    uv run python -m scripts.e1_slice_sweep --out runs/e1slice-v2-envelope
    uv run python -m scripts.e1_slice_sweep --out runs/e1slice-v2-envelope --aggregate-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cdd_oran.e1slice import SCHEMA_VERSION
from cdd_oran.e1slice.aggregate import envelope, paired_summary

REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN = "004-e1-multiseed-envelope"

# ---------------------------------------------------------------------------
# FROZEN Plan 004 matrix. Changing any value here after execution starts is a
# discipline violation; the run records these settings into summary.json.
# ---------------------------------------------------------------------------
N_REPLICATES = 10
DATASET: dict[str, Any] = {"episodes": 48, "steps": 16, "warmup": 2, "obs_noise_scale": 0.0}
SPLIT: dict[str, Any] = {"test_fraction": 0.25, "split_seed": 0}
MODEL: dict[str, Any] = {"hidden": [16], "lr": 1e-2, "epochs": 300, "batch_size": 64}
VERIFY_TOL = 1e-6
ARMS = ("oracle", "dense", "discovered")
GRAPH_BLOCKS = ("overall", "ncp_kpi", "kpi_kpi")
BOOTSTRAP: dict[str, Any] = {"resample_seed": 0, "n_resamples": 10_000, "alpha": 0.05}
PAIRS = (
    ("dense_minus_oracle", "dense", "oracle"),
    ("discovered_minus_oracle", "discovered", "oracle"),
    ("dense_minus_discovered", "dense", "discovered"),
)


def replicate_seeds(r: int) -> dict[str, int]:
    """The frozen paired-seed assignment for replicate ``r``."""
    return {"env_seed": r, "weight_seed": r, "split_seed": int(SPLIT["split_seed"])}


def replicate_dir(out: Path, r: int) -> Path:
    """Collision-free per-replicate directory ``<out>/replicate-00 .. replicate-09``."""
    return out / f"replicate-{r:02d}"


# ---------------------------------------------------------------------------
# Small deterministic helpers (atomic writes, hashing).
# ---------------------------------------------------------------------------
def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Stage model.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Stage:
    """One step of a replicate pipeline.

    ``argv`` builds the child command (run from ``REPO_ROOT`` so ``-m scripts...`` and the
    ``cdd_oran`` import resolve). ``artifacts`` lists the output files whose bytes are hashed
    into the journal and re-checked on resume. ``validate`` is an optional semantic check of
    cross-artifact provenance bindings; it must NOT trust file existence for correctness.
    """

    name: str
    argv: Callable[[Path, int, dict[str, int]], list[str]]
    artifacts: tuple[str, ...] = ()
    validate: Callable[[Path], bool] | None = None


def _e1(*args: str) -> list[str]:
    return [sys.executable, "-m", "scripts.e1_slice", *args]


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


def _stage_discover(d: Path, r: int, s: dict[str, int]) -> list[str]:
    return _e1("discover", "--dataset", str(d), "--force")


def _stage_train(d: Path, r: int, s: dict[str, int]) -> list[str]:
    hidden = [str(w) for w in MODEL["hidden"]]
    return _e1(
        "train", "--dataset", str(d),
        "--hidden", *hidden,
        "--lr", repr(float(MODEL["lr"])),
        "--epochs", str(MODEL["epochs"]),
        "--batch-size", str(MODEL["batch_size"]),
        "--weight-seed", str(s["weight_seed"]),
        "--force",
    )


def _stage_eval(d: Path, r: int, s: dict[str, int]) -> list[str]:
    return _e1("eval", "--dataset", str(d))


def _stage_verify(d: Path, r: int, s: dict[str, int]) -> list[str]:
    return _e1("verify", "--dataset", str(d), "--tol", repr(float(VERIFY_TOL)))


def _stage_recover(d: Path, r: int, s: dict[str, int]) -> list[str]:
    return _e1("recover", "--dataset", str(d))


# --- semantic validators: cross-artifact provenance bindings must hold ------
def _v_split(d: Path) -> bool:
    m = _load_json(d / "manifest.json")
    sp = _load_json(d / "split.json")
    return sp["dataset_hash"] == m["dataset_hash"]


def _v_discover(d: Path) -> bool:
    m = _load_json(d / "manifest.json")
    sp = _load_json(d / "split.json")
    dc = _load_json(d / "discovery.json")
    return dc["dataset_hash"] == m["dataset_hash"] and dc["split_hash"] == sp["split_hash"]


def _v_train(d: Path) -> bool:
    m = _load_json(d / "manifest.json")
    sp = _load_json(d / "split.json")
    dc = _load_json(d / "discovery.json")
    for arm in ARMS:
        meta = _load_json(d / "arms" / arm / "arm_meta.json")
        if meta["dataset_hash"] != m["dataset_hash"] or meta["split_hash"] != sp["split_hash"]:
            return False
        if meta["model_sha256"] != _sha256_file(d / "arms" / arm / "model.pt"):
            return False
        if meta["ref_sha256"] != _sha256_file(d / "arms" / arm / "test_pred_ref.npz"):
            return False
        if arm == "discovered" and meta.get("discovery_hash") != dc["content_hash"]:
            return False
    return True


def _v_eval(d: Path) -> bool:
    m = _load_json(d / "manifest.json")
    sp = _load_json(d / "split.json")
    mt = _load_json(d / "metrics.json")
    return mt["dataset_hash"] == m["dataset_hash"] and mt["split_hash"] == sp["split_hash"]


def _v_recover(d: Path) -> bool:
    m = _load_json(d / "manifest.json")
    sp = _load_json(d / "split.json")
    dc = _load_json(d / "discovery.json")
    rc = _load_json(d / "recovery.json")
    return (
        rc["dataset_hash"] == m["dataset_hash"]
        and rc["split_hash"] == sp["split_hash"]
        and rc["discovery_hash"] == dc["content_hash"]
    )


_ARM_FILES = tuple(
    f"arms/{arm}/{name}"
    for arm in ARMS
    for name in ("model.pt", "test_pred_ref.npz", "arm_meta.json")
)


def build_stages() -> tuple[Stage, ...]:
    """The frozen 7-stage E1 pipeline for one replicate."""
    return (
        Stage("generate", _stage_generate, ("manifest.json", "rows.npz")),
        Stage("split", _stage_split, ("split.json",), _v_split),
        Stage("discover", _stage_discover, ("discovery.json",), _v_discover),
        Stage("train", _stage_train, _ARM_FILES, _v_train),
        Stage("eval", _stage_eval, ("metrics.json",), _v_eval),
        Stage("verify", _stage_verify, ()),
        Stage("recover", _stage_recover, ("recovery.json",), _v_recover),
    )


# ---------------------------------------------------------------------------
# Journal (atomic, per replicate).
# ---------------------------------------------------------------------------
def _journal_path(rep_dir: Path) -> Path:
    return rep_dir / "journal.json"


def load_journal(rep_dir: Path, r: int) -> dict[str, Any]:
    path = _journal_path(rep_dir)
    if path.exists():
        try:
            data = _load_json(path)
            if isinstance(data.get("stages"), dict):
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return {"replicate": r, "seeds": replicate_seeds(r), "stages": {}}


def save_journal(rep_dir: Path, journal: dict[str, Any]) -> None:
    _atomic_write_text(_journal_path(rep_dir), json.dumps(journal, indent=2, sort_keys=True))


def stage_valid(rep_dir: Path, stage: Stage, journal: dict[str, Any]) -> bool:
    """True iff the stage completed cleanly and its full artifact chain still validates.

    Existence alone is never sufficient: every journaled artifact must re-hash to its
    recorded digest, and the stage's semantic binding check (if any) must pass.
    """
    rec = journal.get("stages", {}).get(stage.name)
    if not rec or rec.get("returncode") != 0:
        return False
    recorded = rec.get("artifacts", {})
    for name in stage.artifacts:
        f = rep_dir / name
        if not f.exists() or name not in recorded:
            return False
        if _sha256_file(f) != recorded[name]:
            return False
    if stage.validate is not None:
        try:
            if not stage.validate(rep_dir):
                return False
        except (OSError, KeyError, json.JSONDecodeError):
            return False
    return True


# ---------------------------------------------------------------------------
# Stage execution.
# ---------------------------------------------------------------------------
def _stream_subprocess(argv: list[str], log_path: Path, prefix: str) -> int:
    """Run ``argv`` from REPO_ROOT, tee combined stdout/stderr to ``log_path`` and console."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"$ {' '.join(argv)}\n")
        log.flush()
        proc = subprocess.Popen(
            argv,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            log.flush()
            sys.stdout.write(f"{prefix} {line}")
            sys.stdout.flush()
        return proc.wait()


def run_stage(
    rep_dir: Path, r: int, idx: int, stage: Stage, journal: dict[str, Any]
) -> bool:
    """Execute one stage, capture its log, and journal it atomically. Returns success."""
    seeds = replicate_seeds(r)
    argv = stage.argv(rep_dir, r, seeds)
    log_rel = f"logs/{idx:02d}-{stage.name}.log"
    log_path = rep_dir / log_rel
    start = _now()
    prefix = f"[replicate-{r:02d} {stage.name}]"
    rc = _stream_subprocess(argv, log_path, prefix)
    end = _now()

    artifacts: dict[str, str] = {}
    missing: list[str] = []
    for name in stage.artifacts:
        f = rep_dir / name
        if f.exists():
            artifacts[name] = _sha256_file(f)
        else:
            missing.append(name)

    ok = rc == 0 and not missing
    journal.setdefault("stages", {})[stage.name] = {
        "command": argv,
        "start_utc": start,
        "end_utc": end,
        "returncode": rc,
        "log": log_rel,
        "artifacts": artifacts,
        "missing_artifacts": missing,
        "ok": ok,
    }
    save_journal(rep_dir, journal)
    if missing:
        print(f"{prefix} FAILED: missing artifacts {missing}")
    return ok


def run_replicate(out: Path, r: int, stages: Iterable[Stage]) -> dict[str, Any]:
    """Run (or resume) one replicate. Returns a status record; never raises on a stage failure."""
    rep_dir = replicate_dir(out, r)
    rep_dir.mkdir(parents=True, exist_ok=True)
    journal = load_journal(rep_dir, r)
    # Persist the (possibly freshly initialised) journal so a crash still leaves a record.
    save_journal(rep_dir, journal)

    must_run = False
    for idx, stage in enumerate(stages, start=1):
        if not must_run and stage_valid(rep_dir, stage, journal):
            print(f"[replicate-{r:02d} {stage.name}] skip (validated)")
            continue
        must_run = True
        ok = run_stage(rep_dir, r, idx, stage, journal)
        if not ok:
            return {
                "replicate": r,
                "seeds": replicate_seeds(r),
                "status": "failed",
                "failed_stage": stage.name,
                "returncode": journal["stages"][stage.name]["returncode"],
            }
    return {"replicate": r, "seeds": replicate_seeds(r), "status": "succeeded", "failed_stage": None}


# ---------------------------------------------------------------------------
# Launch provenance (recorded once; re-read verbatim so aggregation stays byte-stable).
# ---------------------------------------------------------------------------
def _run(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=str(REPO_ROOT), capture_output=True, text=True, check=False
    )
    return result.stdout.rstrip("\n") if result.returncode == 0 else "unknown"


def _uv_lock_sha256() -> str:
    lock = REPO_ROOT / "uv.lock"
    return _sha256_file(lock) if lock.exists() else "absent"


def capture_provenance(out: Path) -> dict[str, Any]:
    """Write ``run_provenance.json`` once at launch; on resume keep the original bytes.

    Aggregation reads this file (never wall-clock ``now()``), so re-running aggregation
    reproduces byte-identical summary content.
    """
    path = out / "run_provenance.json"
    if path.exists():
        return _load_json(path)
    import numpy as np
    import torch

    prov = {
        "plan": PLAN,
        "launch_utc": _now(),
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


def frozen_settings() -> dict[str, Any]:
    return {
        "n_replicates": N_REPLICATES,
        "dataset": DATASET,
        "split": SPLIT,
        "model": MODEL,
        "verify_tol": VERIFY_TOL,
        "arms": list(ARMS),
        "graph_blocks": list(GRAPH_BLOCKS),
        "bootstrap": BOOTSTRAP,
        "seed_matrix": {str(r): replicate_seeds(r) for r in range(N_REPLICATES)},
    }


# ---------------------------------------------------------------------------
# Deterministic aggregation.
# ---------------------------------------------------------------------------
def _replicate_record(out: Path, r: int, stages: tuple[Stage, ...]) -> dict[str, Any]:
    """Read one replicate's verified outputs into a flat record. status success/failed."""
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
        rec.update({
            "status": "failed",
            "failed_stage": failed_stage,
            "returncode": j.get("returncode"),
        })
        return rec

    metrics = _load_json(rep_dir / "metrics.json")
    recovery = _load_json(rep_dir / "recovery.json")
    manifest = _load_json(rep_dir / "manifest.json")
    discovery = _load_json(rep_dir / "discovery.json")
    rec.update({
        "status": "succeeded",
        "metrics": {
            arm: {
                "mse": float(metrics["arms"][arm]["mse"]),
                "mae": float(metrics["arms"][arm]["mae"]),
            }
            for arm in ARMS
        },
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
        "child_hashes": {
            "dataset_hash": manifest["dataset_hash"],
            "scm_hash": manifest["scm_hash"],
            "split_hash": metrics["split_hash"],
            "discovery_hash": discovery["content_hash"],
            "metrics_hash": metrics["metrics_hash"],
            "recovery_hash": recovery["content_hash"],
            "arms": {
                arm: {
                    "model_sha256": metrics["arms"][arm]["model_sha256"],
                    "ref_sha256": metrics["arms"][arm]["ref_sha256"],
                }
                for arm in ARMS
            },
        },
    })
    return rec


def aggregate(out: Path, stages: tuple[Stage, ...] | None = None) -> dict[str, Any]:
    """Build the canonical summary dict from persisted, verified replicate artifacts.

    Deterministic: every number derives from on-disk artifacts and the persisted
    provenance file; no wall-clock is read here. Requires the exact frozen seed matrix
    (all of ``0..9`` succeeded) for a ``complete`` summary.
    """
    if stages is None:
        stages = build_stages()
    provenance = _load_json(out / "run_provenance.json")

    records = [_replicate_record(out, r, stages) for r in range(N_REPLICATES)]
    succeeded = [rec for rec in records if rec["status"] == "succeeded"]
    failed = [
        {
            "replicate": rec["replicate"],
            "seeds": rec["seeds"],
            "failed_stage": rec.get("failed_stage"),
            "returncode": rec.get("returncode"),
        }
        for rec in records
        if rec["status"] != "succeeded"
    ]
    complete = len(succeeded) == N_REPLICATES

    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "plan": PLAN,
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
        summary["prediction"] = {}
        summary["paired_mse_diff"] = {}
        return summary

    # Graph recovery envelopes (per block, per metric).
    graph_env: dict[str, Any] = {}
    for block in GRAPH_BLOCKS:
        graph_env[block] = {
            metric: envelope([rec["graph"][block][metric] for rec in succeeded])
            for metric in ("precision", "recall", "f1")
        }
    summary["graph_recovery"] = graph_env

    # Per-arm prediction-error envelopes.
    pred_env: dict[str, Any] = {}
    for arm in ARMS:
        pred_env[arm] = {
            "mse": envelope([rec["metrics"][arm]["mse"] for rec in succeeded]),
            "mae": envelope([rec["metrics"][arm]["mae"] for rec in succeeded]),
        }
    summary["prediction"] = pred_env

    # Paired MSE differences with deterministic bootstrap CI.
    paired: dict[str, Any] = {}
    for label, a_arm, b_arm in PAIRS:
        a = [rec["metrics"][a_arm]["mse"] for rec in succeeded]
        b = [rec["metrics"][b_arm]["mse"] for rec in succeeded]
        paired[label] = paired_summary(
            a, b,
            resample_seed=BOOTSTRAP["resample_seed"],
            n_resamples=BOOTSTRAP["n_resamples"],
            alpha=BOOTSTRAP["alpha"],
        )
    summary["paired_mse_diff"] = paired
    return summary


def write_outputs(out: Path, summary: dict[str, Any]) -> tuple[Path, Path]:
    """Write ``replicates.jsonl`` and canonical ``summary.json`` (sorted-key, stable floats)."""
    jsonl = out / "replicates.jsonl"
    lines = [json.dumps(rec, sort_keys=True) for rec in summary["replicates"]]
    _atomic_write_text(jsonl, "\n".join(lines) + ("\n" if lines else ""))

    summ = out / "summary.json"
    _atomic_write_text(summ, json.dumps(summary, indent=2, sort_keys=True))
    return jsonl, summ


# ---------------------------------------------------------------------------
# Lock (guards the collision STOP condition: two processes / one out dir).
# ---------------------------------------------------------------------------
def _lock_path(out: Path) -> Path:
    return out / ".sweep.lock"


def acquire_lock(out: Path, force_unlock: bool) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = _lock_path(out)
    if force_unlock and path.exists():
        path.unlink()
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        holder = path.read_text(encoding="utf-8") if path.exists() else "<unknown>"
        raise SystemExit(
            f"STOP: {path} already held -> another sweep may be writing {out}.\n"
            f"  holder: {holder}\n"
            f"  If it is stale, re-run with --force-unlock."
        ) from exc
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"pid": os.getpid(), "host": os.environ.get("COMPUTERNAME", "?"),
                             "utc": _now()}))
    return path


def release_lock(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def print_dry_run(out: Path) -> None:
    dirs = [replicate_dir(out, r) for r in range(N_REPLICATES)]
    unique = sorted({str(d) for d in dirs})
    print(f"DRY RUN: {len(unique)} unique run dirs (of {N_REPLICATES} planned)")
    for r in range(N_REPLICATES):
        s = replicate_seeds(r)
        print(
            f"  {replicate_dir(out, r)}  "
            f"env_seed={s['env_seed']} weight_seed={s['weight_seed']} split_seed={s['split_seed']}"
        )
    print("FROZEN SETTINGS:")
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
    print("\n==== E1 sweep summary ====")
    print(f"  succeeded {summary['n_replicates_succeeded']}/{N_REPLICATES}, complete={summary['complete']}")
    print(f"  -> {summ}")
    print(f"  -> {jsonl}")
    if failed:
        for s in failed:
            print(f"  FAILED replicate-{s['replicate']:02d} at stage '{s['failed_stage']}' "
                  f"(rc {s['returncode']})")
        return 1
    return 0


def aggregate_only(out: Path) -> int:
    summary = aggregate(out)
    jsonl, summ = write_outputs(out, summary)
    print(f"aggregated {summary['n_replicates_succeeded']}/{N_REPLICATES} "
          f"(complete={summary['complete']}) -> {summ}")
    return 0 if summary["complete"] else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plan 004 resumable multi-seed E1 envelope sweep.")
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
