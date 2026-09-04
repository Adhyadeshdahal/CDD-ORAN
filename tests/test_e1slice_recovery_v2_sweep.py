"""Tests for the Plan 006 phase-2b reproducible v2 RECOVERY sweep driver.

The real pipeline (generate -> split -> discover-v2 -> recover-v2 across 10 seeds) is NOT exercised
here; it is validated by the end-to-end run. These tests pin the driver's contract:

* the frozen 4-stage recovery-only chain and the frozen paired-seed matrix;
* aggregation is deterministic / byte-identical on re-run, from synthetic per-replicate artifacts;
* a complete summary requires the full frozen matrix; a non-complete run exits nonzero;
* the per-edge stability-selection frequency (§8) is exact count/B arithmetic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cdd_oran.e1slice.discovery_v2 import PROTOCOL_COMMIT
from scripts import e1_slice_recovery_v2_sweep as rsweep

# The frozen 6-edge E1 mask: K0<-P0, K1<-P1, K2<-{P2,K0}, K3<-{P3,K1}.
_TRUE_MASK = [
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 0, 0],
]


def test_build_stages_is_the_frozen_four_stage_recovery_chain() -> None:
    names = [s.name for s in rsweep.build_stages()]
    assert names == ["generate", "split", "discover-v2", "recover-v2"]


def test_seed_matrix_is_the_frozen_paired_matrix() -> None:
    for r in range(rsweep.N_REPLICATES):
        assert rsweep.replicate_seeds(r) == {"env_seed": r, "weight_seed": r, "split_seed": 0}


def test_frozen_settings_pin_protocol_commit_and_recovery_only() -> None:
    s = rsweep.frozen_settings()
    assert s["protocol_commit"] == PROTOCOL_COMMIT
    assert s["recovery_only"] is True
    assert s["dataset"] == {"episodes": 48, "steps": 16, "warmup": 2, "obs_noise_scale": 0.0}
    assert s["split"] == {"test_fraction": 0.25, "split_seed": 0}


def test_dry_run_prints_ten_unique_dirs_and_settings(tmp_path: Path, capsys) -> None:
    rsweep.print_dry_run(tmp_path)
    out = capsys.readouterr().out
    assert "10 unique run dirs" in out
    for r in range(10):
        assert f"replicate-{r:02d}" in out
    assert "FROZEN SETTINGS" in out and PROTOCOL_COMMIT in out


def _write_synthetic_replicate(out: Path, r: int, *, recall_full: bool = True) -> None:
    rep = rsweep.replicate_dir(out, r)
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "manifest.json").write_text(json.dumps({
        "dataset_hash": f"ds{r:02d}", "scm_hash": f"scm{r:02d}", "seeds": {"env_seed": r},
    }))
    (rep / "split.json").write_text(json.dumps({
        "dataset_hash": f"ds{r:02d}", "split_hash": f"sp{r:02d}", "config": {"split_seed": 0},
    }))
    (rep / "discovery_v2.json").write_text(json.dumps({
        "content_hash": f"disc{r:02d}", "dataset_hash": f"ds{r:02d}", "split_hash": f"sp{r:02d}",
        "binary_mask": _TRUE_MASK,
    }))
    if recall_full:
        overall = {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 6, "fp": 0, "fn": 0}
        kk = {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 2, "fp": 0, "fn": 0}
        missed: list = []
    else:
        overall = {"precision": 1.0, "recall": 2.0 / 3.0, "f1": 0.8, "tp": 4, "fp": 0, "fn": 2}
        kk = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 2}
        missed = [{"child_name": "K2", "parent_name": "K0", "type": "kpi_kpi"}]
    (rep / "recovery_v2.json").write_text(json.dumps({
        "content_hash": f"rec{r:02d}", "dataset_hash": f"ds{r:02d}", "split_hash": f"sp{r:02d}",
        "discovery_hash": f"disc{r:02d}", "protocol_commit": PROTOCOL_COMMIT,
        "recovery": {
            "overall": overall,
            "ncp_kpi": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 4, "fp": 0, "fn": 0},
            "kpi_kpi": kk,
            "missed": missed,
        },
    }))


def _synthetic_sweep(out: Path, **kw) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_provenance.json").write_text(json.dumps({
        "plan": rsweep.PLAN, "protocol_commit": PROTOCOL_COMMIT, "git_sha": "deadbeef",
        "git_dirty": False, "python_version": "3.12.13", "numpy_version": "2.4.2",
        "torch_version": "2.10.0", "uv_lock_sha256": "abc", "launch_utc": "2026-09-04T00:00:00+00:00",
    }))
    for r in range(rsweep.N_REPLICATES):
        _write_synthetic_replicate(out, r, **kw)


def test_aggregation_is_byte_identical_across_reruns(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(rsweep, "stage_valid", lambda *a, **k: True)
    out = tmp_path / "sweep"
    _synthetic_sweep(out)
    stages = rsweep.build_stages()
    s1 = json.dumps(rsweep.aggregate(out, stages), indent=2, sort_keys=True)
    s2 = json.dumps(rsweep.aggregate(out, stages), indent=2, sort_keys=True)
    assert s1 == s2


def test_aggregation_perfect_recovery_and_stability(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(rsweep, "stage_valid", lambda *a, **k: True)
    out = tmp_path / "sweep"
    _synthetic_sweep(out)
    summary = rsweep.aggregate(out, rsweep.build_stages())
    assert summary["complete"] is True
    assert summary["n_replicates_succeeded"] == 10
    assert summary["failed_replicates"] == []
    # Every block's recall is a flat 1.0 envelope.
    for block in ("overall", "ncp_kpi", "kpi_kpi"):
        rc = summary["graph_recovery"][block]["recall"]
        assert rc["min"] == 1.0 and rc["max"] == 1.0 and rc["mean"] == 1.0
    # Stability: exactly the 6 true edges at frequency 1.0, both KPI->KPI included, no others.
    stab = summary["stability_selection"]
    assert stab["b"] == 10
    assert stab["frequency"] == [[float(v) for v in row] for row in _TRUE_MASK]
    sel = {(e["child"], e["parent"]): e["frequency"] for e in stab["selected_edges"]}
    assert sel[("K2", "K0")] == 1.0 and sel[("K3", "K1")] == 1.0
    assert len(sel) == 6
    assert summary["replicates"][0]["child_hashes"]["discovery_v2_hash"] == "disc00"


def test_aggregation_marks_incomplete_when_a_replicate_fails(tmp_path: Path, monkeypatch) -> None:
    out = tmp_path / "sweep"
    _synthetic_sweep(out)

    def fake_valid(rep_dir: Path, stage, journal) -> bool:
        return not (rep_dir.name == "replicate-07" and stage.name == "recover-v2")

    monkeypatch.setattr(rsweep, "stage_valid", fake_valid)
    summary = rsweep.aggregate(out, rsweep.build_stages())
    assert summary["complete"] is False
    assert summary["n_replicates_succeeded"] == 9
    assert [f["replicate"] for f in summary["failed_replicates"]] == [7]
    assert summary["failed_replicates"][0]["failed_stage"] == "recover-v2"


def test_stability_frequency_is_exact_count_over_b() -> None:
    # Two of three replicates select edge K2<-K0; one selects nothing -> frequency 2/3 there.
    m_full = [[0] * 8 for _ in range(4)]
    m_full[2][4] = 1  # K2 <- K0
    m_empty = [[0] * 8 for _ in range(4)]
    recs = [{"binary_mask": m_full}, {"binary_mask": m_full}, {"binary_mask": m_empty}]
    stab = rsweep.stability_frequency(recs)
    assert stab["b"] == 3
    assert stab["counts"][2][4] == 2
    assert stab["frequency"][2][4] == pytest.approx(2.0 / 3.0)
    assert stab["selected_edges"] == [
        {"child": "K2", "parent": "K0", "count": 2, "frequency": pytest.approx(2.0 / 3.0)}
    ]


def test_write_outputs_emits_jsonl_and_summary(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(rsweep, "stage_valid", lambda *a, **k: True)
    out = tmp_path / "sweep"
    _synthetic_sweep(out)
    summary = rsweep.aggregate(out, rsweep.build_stages())
    jsonl, summ = rsweep.write_outputs(out, summary)
    assert summ.exists() and jsonl.exists()
    lines = jsonl.read_text().strip().splitlines()
    assert len(lines) == 10
    assert all(json.loads(line)["replicate"] == i for i, line in enumerate(lines))


def _run_all_with(monkeypatch, out: Path, *, complete: bool) -> int:
    monkeypatch.setattr(rsweep, "capture_provenance", lambda o: {})
    monkeypatch.setattr(
        rsweep, "run_replicate",
        lambda o, r, stages: {
            "replicate": r, "seeds": rsweep.replicate_seeds(r),
            "status": "succeeded", "failed_stage": None,
        },
    )
    summary = {"n_replicates_succeeded": 10 if complete else 9, "complete": complete, "replicates": []}
    monkeypatch.setattr(rsweep, "aggregate", lambda o, stages: summary)
    monkeypatch.setattr(
        rsweep, "write_outputs", lambda o, s: (out / "replicates.jsonl", out / "summary.json"),
    )
    return rsweep.run_all(out, force_unlock=False)


def test_run_all_exits_nonzero_when_aggregate_incomplete(tmp_path: Path, monkeypatch) -> None:
    assert _run_all_with(monkeypatch, tmp_path / "bad", complete=False) == 1


def test_run_all_exits_zero_only_when_complete(tmp_path: Path, monkeypatch) -> None:
    assert _run_all_with(monkeypatch, tmp_path / "good", complete=True) == 0
