"""Tests for the Plan 004 multi-seed envelope sweep runner + aggregation.

The expensive real pipeline (300-epoch training x 3 arms x 10 seeds) is NOT exercised
here. Instead:

* resume / reject / failure logic runs a *fake* fast stage list (each stage is a tiny
  subprocess that writes an artifact and bumps an external counter), so we can assert
  exactly which stages re-ran;
* aggregation determinism runs against synthetic per-replicate JSON artifacts;
* the bootstrap CI and pairing math are pinned against small fixed arrays.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from cdd_oran.e1slice.aggregate import (
    bootstrap_ci_mean,
    envelope,
    paired_diff,
    paired_summary,
)
from scripts import e1_slice_sweep as sweep


# ---------------------------------------------------------------------------
# Fake fast stages (no torch, no training).
# ---------------------------------------------------------------------------
def _fake_stage(name: str, counter: Path, *, exit_code: int = 0) -> sweep.Stage:
    """A stage whose subprocess bumps ``counter`` and writes ``<rep>/<name>.json``."""

    def argv(rep_dir: Path, r: int, seeds: dict[str, int]) -> list[str]:
        art = rep_dir / f"{name}.json"
        code = (
            "import pathlib;"
            f"cp=pathlib.Path({str(counter)!r});"
            "n=int(cp.read_text()) if cp.exists() else 0;"
            "cp.write_text(str(n+1));"
            f"pathlib.Path({str(art)!r}).write_text('{name}-ok');"
            f"raise SystemExit({exit_code})"
        )
        return [sys.executable, "-c", code]

    return sweep.Stage(name, argv, (f"{name}.json",))


def _read_counter(counter: Path) -> int:
    return int(counter.read_text()) if counter.exists() else 0


# ---------------------------------------------------------------------------
# Frozen matrix / dry run.
# ---------------------------------------------------------------------------
def test_plan_names_exactly_ten_collision_free_dirs(tmp_path: Path) -> None:
    dirs = [sweep.replicate_dir(tmp_path, r) for r in range(sweep.N_REPLICATES)]
    assert len(dirs) == 10
    assert len({str(d) for d in dirs}) == 10
    assert [d.name for d in dirs] == [f"replicate-{r:02d}" for r in range(10)]


def test_seed_matrix_is_frozen_paired() -> None:
    for r in range(sweep.N_REPLICATES):
        assert sweep.replicate_seeds(r) == {"env_seed": r, "weight_seed": r, "split_seed": 0}


def test_dry_run_prints_ten_unique_dirs_and_settings(tmp_path: Path, capsys) -> None:
    sweep.print_dry_run(tmp_path)
    out = capsys.readouterr().out
    assert "10 unique run dirs" in out
    for r in range(10):
        assert f"replicate-{r:02d}" in out
    assert "FROZEN SETTINGS" in out
    settings = sweep.frozen_settings()
    assert settings["dataset"] == {"episodes": 48, "steps": 16, "warmup": 2, "obs_noise_scale": 0.0}
    assert settings["split"] == {"test_fraction": 0.25, "split_seed": 0}
    assert settings["model"] == {"hidden": [16], "lr": 1e-2, "epochs": 300, "batch_size": 64}


def test_build_stages_is_the_frozen_seven_stage_chain() -> None:
    names = [s.name for s in sweep.build_stages()]
    assert names == ["generate", "split", "discover", "train", "eval", "verify", "recover"]


# ---------------------------------------------------------------------------
# Journal + resume.
# ---------------------------------------------------------------------------
def test_full_run_journals_every_stage(tmp_path: Path) -> None:
    counter = tmp_path / "counter"
    stages = [_fake_stage(n, counter) for n in ("a", "b", "c")]
    status = sweep.run_replicate(tmp_path, 0, stages)
    assert status["status"] == "succeeded"
    assert _read_counter(counter) == 3

    journal = sweep.load_journal(sweep.replicate_dir(tmp_path, 0), 0)
    assert set(journal["stages"]) == {"a", "b", "c"}
    for name in ("a", "b", "c"):
        rec = journal["stages"][name]
        assert rec["returncode"] == 0
        assert rec["ok"] is True
        assert f"{name}.json" in rec["artifacts"]
        assert rec["log"].startswith("logs/")


def test_resume_skips_validated_stages(tmp_path: Path) -> None:
    counter = tmp_path / "counter"
    stages = [_fake_stage(n, counter) for n in ("a", "b", "c")]
    sweep.run_replicate(tmp_path, 0, stages)
    assert _read_counter(counter) == 3
    # Re-run: everything already validates -> nothing recomputes.
    sweep.run_replicate(tmp_path, 0, stages)
    assert _read_counter(counter) == 3


def test_resume_after_interrupt_completes_without_recomputing(tmp_path: Path) -> None:
    counter = tmp_path / "counter"
    a, b, c = (_fake_stage(n, counter) for n in ("a", "b", "c"))
    # Simulate a crash after stage b: only [a, b] ran.
    sweep.run_replicate(tmp_path, 0, [a, b])
    assert _read_counter(counter) == 2
    # Resume the full chain: a, b validated + skipped, only c runs.
    status = sweep.run_replicate(tmp_path, 0, [a, b, c])
    assert status["status"] == "succeeded"
    assert _read_counter(counter) == 3  # +1 for c only


def test_resume_rejects_a_corrupted_artifact(tmp_path: Path) -> None:
    counter = tmp_path / "counter"
    stages = [_fake_stage(n, counter) for n in ("a", "b", "c")]
    sweep.run_replicate(tmp_path, 0, stages)
    assert _read_counter(counter) == 3
    # Corrupt b's artifact bytes: b no longer hashes to its journaled digest.
    rep = sweep.replicate_dir(tmp_path, 0)
    (rep / "b.json").write_text("tampered")
    sweep.run_replicate(tmp_path, 0, stages)
    # a stays valid (skip); b + c re-run.
    assert _read_counter(counter) == 5


def test_resume_rejects_a_deleted_artifact(tmp_path: Path) -> None:
    counter = tmp_path / "counter"
    stages = [_fake_stage(n, counter) for n in ("a", "b")]
    sweep.run_replicate(tmp_path, 0, stages)
    assert _read_counter(counter) == 2
    (sweep.replicate_dir(tmp_path, 0) / "a.json").unlink()
    sweep.run_replicate(tmp_path, 0, stages)
    assert _read_counter(counter) == 4  # both re-run (a invalid forces b too)


def test_stage_valid_requires_a_zero_return_code(tmp_path: Path) -> None:
    counter = tmp_path / "counter"
    good = _fake_stage("a", counter)
    bad = _fake_stage("b", counter, exit_code=3)
    status = sweep.run_replicate(tmp_path, 0, [good, bad])
    assert status["status"] == "failed"
    assert status["failed_stage"] == "b"
    assert status["returncode"] == 3
    journal = sweep.load_journal(sweep.replicate_dir(tmp_path, 0), 0)
    # 'a' recorded ok; 'b' recorded with the nonzero rc; no stage after 'b' ran.
    assert journal["stages"]["a"]["ok"] is True
    assert journal["stages"]["b"]["returncode"] == 3


def test_failed_replicate_does_not_block_later_replicates(tmp_path: Path) -> None:
    counter = tmp_path / "counter"
    bad = _fake_stage("a", counter, exit_code=1)
    good = _fake_stage("a", counter)
    s0 = sweep.run_replicate(tmp_path, 0, [bad])
    s1 = sweep.run_replicate(tmp_path, 1, [good])
    assert s0["status"] == "failed"
    assert s1["status"] == "succeeded"


# ---------------------------------------------------------------------------
# Bootstrap / pairing math (fixed arrays, no SciPy).
# ---------------------------------------------------------------------------
def test_envelope_on_fixed_array() -> None:
    env = envelope([1.0, 2.0, 3.0, 4.0])
    assert env["n"] == 4
    assert env["mean"] == pytest.approx(2.5)
    assert env["median"] == pytest.approx(2.5)
    assert env["min"] == 1.0 and env["max"] == 4.0
    # sample std (ddof=1) of 1..4 = sqrt(5/3)
    assert env["std"] == pytest.approx(math.sqrt(5.0 / 3.0))
    assert env["values"] == [1.0, 2.0, 3.0, 4.0]


def test_envelope_single_value_has_zero_std() -> None:
    env = envelope([7.5])
    assert env["std"] == 0.0
    assert env["mean"] == 7.5 == env["min"] == env["max"]


def test_paired_diff_is_elementwise() -> None:
    assert paired_diff([3.0, 5.0, 9.0], [1.0, 2.0, 4.0]) == [2.0, 3.0, 5.0]


def test_paired_diff_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        paired_diff([1.0, 2.0], [1.0])


def test_bootstrap_ci_is_deterministic_and_ordered() -> None:
    values = [0.10, 0.12, 0.09, 0.11, 0.13, 0.08, 0.10, 0.12, 0.09, 0.11]
    a = bootstrap_ci_mean(values, resample_seed=0, n_resamples=10_000)
    b = bootstrap_ci_mean(values, resample_seed=0, n_resamples=10_000)
    assert a == b  # byte-identical across re-runs given the same seed
    assert a["ci_low"] <= a["mean"] <= a["ci_high"]
    assert a["n_resamples"] == 10_000 and a["resample_seed"] == 0


def test_bootstrap_ci_matches_reference_percentiles() -> None:
    values = [0.10, 0.12, 0.09, 0.11, 0.13, 0.08, 0.10, 0.12, 0.09, 0.11]
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, arr.size, size=(10_000, arr.size))
    means = arr[idx].mean(axis=1)
    ci = bootstrap_ci_mean(values, resample_seed=0, n_resamples=10_000)
    assert ci["ci_low"] == pytest.approx(float(np.percentile(means, 2.5)))
    assert ci["ci_high"] == pytest.approx(float(np.percentile(means, 97.5)))


def test_paired_summary_bundles_envelope_and_ci() -> None:
    ps = paired_summary([2.0, 4.0, 6.0], [1.0, 1.0, 1.0])
    assert ps["diffs"] == [1.0, 3.0, 5.0]
    assert ps["envelope"]["mean"] == pytest.approx(3.0)
    assert "ci_low" in ps["bootstrap_ci"] and "ci_high" in ps["bootstrap_ci"]


# ---------------------------------------------------------------------------
# Aggregation determinism against synthetic artifacts.
# ---------------------------------------------------------------------------
def _write_synthetic_replicate(out: Path, r: int) -> None:
    rep = sweep.replicate_dir(out, r)
    (rep / "arms").mkdir(parents=True, exist_ok=True)
    base = 1e-6 * (1.0 + r)
    (rep / "manifest.json").write_text(json.dumps({
        "dataset_hash": f"ds{r:02d}", "scm_hash": f"scm{r:02d}",
    }))
    (rep / "discovery.json").write_text(json.dumps({"content_hash": f"disc{r:02d}"}))
    (rep / "metrics.json").write_text(json.dumps({
        "split_hash": f"sp{r:02d}", "metrics_hash": f"mt{r:02d}",
        "arms": {
            "oracle": {"mse": base, "mae": base * 10, "model_sha256": f"o{r}", "ref_sha256": f"or{r}"},
            "dense": {"mse": base * 3, "mae": base * 30, "model_sha256": f"d{r}", "ref_sha256": f"dr{r}"},
            "discovered": {"mse": base * 100, "mae": base * 1000,
                           "model_sha256": f"v{r}", "ref_sha256": f"vr{r}"},
        },
    }))
    # replicate 0..8 recover both K->K missed (partial); replicate 9 recovers everything (varies).
    if r < 9:
        overall = {"precision": 1.0, "recall": 2.0 / 3.0, "f1": 0.8, "tp": 4, "fp": 0, "fn": 2}
        kk = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 2}
    else:
        overall = {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 6, "fp": 0, "fn": 0}
        kk = {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 2, "fp": 0, "fn": 0}
    (rep / "recovery.json").write_text(json.dumps({
        "content_hash": f"rec{r:02d}",
        "recovery": {
            "overall": overall,
            "ncp_kpi": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 4, "fp": 0, "fn": 0},
            "kpi_kpi": kk,
        },
    }))


def _synthetic_sweep(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_provenance.json").write_text(json.dumps({
        "plan": sweep.PLAN, "git_sha": "deadbeef", "git_dirty": False,
        "python_version": "3.12.13", "numpy_version": "2.4.2", "torch_version": "2.10.0",
        "uv_lock_sha256": "abc", "launch_utc": "2026-09-03T00:00:00+00:00",
    }))
    for r in range(sweep.N_REPLICATES):
        _write_synthetic_replicate(out, r)


def test_aggregation_is_byte_identical_across_reruns(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(sweep, "stage_valid", lambda *a, **k: True)
    out = tmp_path / "sweep"
    _synthetic_sweep(out)
    stages = sweep.build_stages()
    s1 = json.dumps(sweep.aggregate(out, stages), indent=2, sort_keys=True)
    s2 = json.dumps(sweep.aggregate(out, stages), indent=2, sort_keys=True)
    assert s1 == s2


def test_aggregation_envelopes_and_completeness(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(sweep, "stage_valid", lambda *a, **k: True)
    out = tmp_path / "sweep"
    _synthetic_sweep(out)
    summary = sweep.aggregate(out, sweep.build_stages())
    assert summary["complete"] is True
    assert summary["n_replicates_succeeded"] == 10
    assert summary["failed_replicates"] == []
    # graph envelope: kpi_kpi recall varies (nine 0.0, one 1.0) -> min 0, max 1, mean 0.1
    kk = summary["graph_recovery"]["kpi_kpi"]["recall"]
    assert kk["min"] == 0.0 and kk["max"] == 1.0
    assert kk["mean"] == pytest.approx(0.1)
    # paired diff present with bootstrap CI
    diff = summary["paired_mse_diff"]["dense_minus_oracle"]
    assert "bootstrap_ci" in diff and len(diff["diffs"]) == 10
    # every child hash carried through
    assert summary["replicates"][0]["child_hashes"]["dataset_hash"] == "ds00"


def test_aggregation_marks_incomplete_when_a_replicate_fails(tmp_path: Path, monkeypatch) -> None:
    out = tmp_path / "sweep"
    _synthetic_sweep(out)

    def fake_valid(rep_dir: Path, stage, journal) -> bool:
        return not (rep_dir.name == "replicate-07" and stage.name == "recover")

    monkeypatch.setattr(sweep, "stage_valid", fake_valid)
    summary = sweep.aggregate(out, sweep.build_stages())
    assert summary["complete"] is False
    assert summary["n_replicates_succeeded"] == 9
    assert [f["replicate"] for f in summary["failed_replicates"]] == [7]
    assert summary["failed_replicates"][0]["failed_stage"] == "recover"


def test_write_outputs_emits_jsonl_and_summary(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(sweep, "stage_valid", lambda *a, **k: True)
    out = tmp_path / "sweep"
    _synthetic_sweep(out)
    summary = sweep.aggregate(out, sweep.build_stages())
    jsonl, summ = sweep.write_outputs(out, summary)
    assert summ.exists() and jsonl.exists()
    lines = jsonl.read_text().strip().splitlines()
    assert len(lines) == 10
    assert all(json.loads(line)["replicate"] == i for i, line in enumerate(lines))


# ---------------------------------------------------------------------------
# Lock (collision STOP guard).
# ---------------------------------------------------------------------------
def test_lock_blocks_a_second_holder(tmp_path: Path) -> None:
    out = tmp_path / "sweep"
    lock = sweep.acquire_lock(out, force_unlock=False)
    with pytest.raises(SystemExit):
        sweep.acquire_lock(out, force_unlock=False)
    sweep.release_lock(lock)
    # released -> can re-acquire
    lock2 = sweep.acquire_lock(out, force_unlock=False)
    sweep.release_lock(lock2)


def test_force_unlock_breaks_a_stale_lock(tmp_path: Path) -> None:
    out = tmp_path / "sweep"
    sweep.acquire_lock(out, force_unlock=False)  # leak it (simulate a crash)
    lock2 = sweep.acquire_lock(out, force_unlock=True)
    sweep.release_lock(lock2)
