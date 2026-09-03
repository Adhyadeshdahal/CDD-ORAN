"""Tiny CPU end-to-end tests for the decoupled discovery stage.

These prove discovery recovers a non-empty, right-shaped structure AND -- critically -- that it
never reads the TARGET environment's true adjacency on the discovery/selection path (no target
leakage, Plan 002). A calibrated, downstream-sampleable posterior is produced only from a
DISTINCT held-out calibration run; the no-calibration output is an uncalibrated diagnostic that
robust structure sampling and staging reject.
"""

import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import cdd_oran.experiments.discover as discover_module
from cdd_oran.analysis.graph_posterior import GraphPosterior
from cdd_oran.config import load_config
from cdd_oran.envs import get_env
from cdd_oran.experiments.discover import run_discovery
from cdd_oran.models import (
    PosteriorStructureSampler,
    load_enumeration_graph,
    node_names,
    stage_run_artifacts,
    validate_enumeration_artifact,
    validate_posterior_artifact,
)


def _tiny_cfg(config="env_i_cdl.yaml"):
    cfg = load_config(config)
    # Point at NON-EXISTENT posterior/enum artifacts: discovery must run from scratch and
    # ignore them (it is decoupled from the dynamics artifacts it produces).
    return replace(
        cfg,
        device="cpu",
        seed=0,
        model=replace(
            cfg.model,
            posterior_artifact="does/not/exist_posterior.json",
            enumeration_graph="does/not/exist_enum.json",
        ),
    )


def _discover(cfg, out_dir, **overrides):
    params = {"train_steps": 120, "pool_size": 512, "cmi_batches": 8, "B": 5, "n_transitions": 256}
    params.update(overrides)
    return run_discovery(cfg, out_dir=out_dir, **params)


def test_discovery_no_calibration_writes_raw_uncalibrated_diagnostic(tmp_path):
    """Default (no --calibration-run) discovery recovers structure but persists only a RAW,
    explicitly-uncalibrated diagnostic -- it must NOT emit a file accepted as a calibrated
    posterior (Plan 002 Step 1.1)."""
    cfg = _tiny_cfg()
    env = get_env(cfg)
    fd = env.get_state_dim()

    result = _discover(cfg, tmp_path)

    # No calibrated posterior file is emitted; only the raw diagnostic + enumeration graph.
    raw_path = tmp_path / "EnvironmentI_raw_posterior.json"
    enum_path = tmp_path / "EnvironmentI_enum.json"
    assert not (tmp_path / "EnvironmentI_posterior.json").exists(), (
        "no-calibration discovery must NOT emit a calibrated posterior file"
    )
    assert raw_path.exists() and enum_path.exists()
    assert result["raw_posterior"] == str(raw_path)
    assert result["calibrated_posterior"] is None
    assert result["calibrated"] is False

    # Enumeration graph: right-shaped (fd, fd+1), NON-EMPTY, recovers real structure.
    graph, meta = load_enumeration_graph(enum_path)
    assert tuple(graph.shape) == (fd, fd + 1)
    pred = graph[:, :fd].numpy().astype(int)
    gt = env.true_adj_matrix.astype(int)
    assert pred.sum() > 0, "enumeration graph must be non-empty"
    assert result["state_edges"] == int(pred.sum())
    assert int(((pred == 1) & (gt == 1)).sum()) > 0, "discovery should recover at least one true edge"
    validate_enumeration_artifact(graph, meta, env, cfg.environment)

    # Raw posterior: right-shaped marginals in [0, 1], correct node layout, NOT calibrated.
    raw = GraphPosterior.load(raw_path)
    assert raw.is_calibrated is False, "no-calibration diagnostic must be marked calibrated=false"
    marg = raw.marginals()
    assert marg.shape == (fd, fd)
    assert np.all((marg >= 0.0) & (marg <= 1.0))
    assert np.all(np.diag(marg) == 0.0)
    assert raw.node_names == node_names(env)
    assert raw.meta.get("environment") == cfg.environment
    assert raw.meta["calibration"]["mode"] == "uncalibrated"


def test_no_calibration_run_removes_stale_calibrated_posterior(tmp_path):
    """Finding #8: a no-cal discovery into a dir that still holds a CALIBRATED
    ``{env}_posterior.json`` from an earlier calibrated run must REMOVE that stale calibrated
    artifact. Configs consume ``posterior_artifact`` and ``enumeration_graph`` as INDEPENDENT
    paths, so leaving the old posterior beside this run's new enum graph would silently pair two
    different runs."""
    cfg = _tiny_cfg()

    # Simulate a prior CALIBRATED run into the SAME --out dir.
    stale = tmp_path / "EnvironmentI_posterior.json"
    stale.write_text('{"stale": "prior calibrated posterior"}', encoding="utf-8")

    result = _discover(cfg, tmp_path)  # no calibration_run -> no-cal branch

    assert result["calibrated"] is False
    assert result["calibrated_posterior"] is None
    # The mismatched calibrated posterior is gone; only this run's fresh raw/enum pair remains.
    assert not stale.exists(), "stale calibrated posterior must be removed by a no-cal run"
    assert (tmp_path / "EnvironmentI_raw_posterior.json").exists()
    assert (tmp_path / "EnvironmentI_enum.json").exists()


def test_no_calibration_partial_failure_still_removes_stale_posterior(tmp_path, monkeypatch):
    """R2-1: the stale-posterior removal must precede the enum refresh + bootstrap, so a partial
    failure cannot leave a NEW enum paired with the OLD calibrated posterior. Here the bootstrap
    raises AFTER the enum is refreshed; the stale calibrated posterior must ALREADY be gone."""
    cfg = _tiny_cfg()
    stale = tmp_path / "EnvironmentI_posterior.json"
    stale.write_text('{"stale": "prior calibrated posterior"}', encoding="utf-8")

    def _boom(*args, **kwargs):
        raise RuntimeError("bootstrap failed mid-run")

    monkeypatch.setattr(GraphPosterior, "from_bootstrap", _boom)

    with pytest.raises(RuntimeError, match="bootstrap failed"):
        _discover(cfg, tmp_path)

    # The failure landed AFTER the enum was refreshed (proving removal HAD to precede it)...
    assert (tmp_path / "EnvironmentI_enum.json").exists()
    # ...yet the stale calibrated posterior is already gone -- no new-enum + old-posterior pair
    # survives the failure window (removal is hoisted before the enum overwrite).
    assert not stale.exists(), "stale calibrated posterior must be removed before the failure window"


def test_raw_diagnostic_is_rejected_by_downstream_sampler_and_staging(tmp_path):
    """The raw diagnostic loads for analysis but robust structure sampling / staging refuse it
    (Plan 002 Step 4)."""
    cfg = _tiny_cfg()
    env = get_env(cfg)
    result = _discover(cfg, tmp_path)
    raw_path = result["raw_posterior"]

    # Loadable for analysis...
    assert GraphPosterior.load(raw_path).is_calibrated is False
    # ...but robust structure sampling refuses it.
    with pytest.raises(ValueError, match="(?i)not calibrated"):
        PosteriorStructureSampler.from_artifact(raw_path, env=env, environment=cfg.environment)
    # ...and staging it into a run refuses it too.
    staged_cfg = replace(
        cfg,
        model=replace(
            cfg.model,
            posterior_artifact=raw_path,
            enumeration_graph=result["enumeration_graph"],
        ),
    )
    run_dir = tmp_path / "stage_run"
    run_dir.mkdir()
    with pytest.raises(ValueError, match="(?i)not calibrated"):
        stage_run_artifacts(staged_cfg, run_dir, env=env)


class _TruthGuardEnv:
    """Wrap an env and RAISE if its true adjacency is read -- proof that the discovery/selection
    path never touches target ground truth. Every other attribute delegates to the real env."""

    def __init__(self, env):
        object.__setattr__(self, "_env", env)

    def __getattr__(self, name):
        if name == "true_adj_matrix":
            raise AssertionError(
                "target env.true_adj_matrix must not be read on the discovery/selection path"
            )
        return getattr(object.__getattribute__(self, "_env"), name)


def test_discovery_does_not_read_target_truth_without_calibration(tmp_path, monkeypatch):
    """Training, thresholding, the crisp enumeration graph, the bootstrap frequency and the raw
    diagnostic all complete with the target env's true_adj_matrix accessor booby-trapped to raise
    (Plan 002 Step 2, round-2 Finding 2).

    The booby-trapped env must reach EVERY production path: discovery builds its env through
    ``discover.get_env``, but the bootstrap runs inside ``edge_stability`` which builds its OWN env
    through ``edge_stability.get_env`` -- so BOTH are patched. If the old truth-reading bootstrap
    were restored, ``edge_stability`` would hit ``env.true_adj_matrix`` here and this test fails."""
    import cdd_oran.analysis.edge_stability as edge_stability_module

    cfg = _tiny_cfg()
    real_get_env = discover_module.get_env  # captured before patching -> the real (unguarded) env
    guard = lambda c: _TruthGuardEnv(real_get_env(c))  # noqa: E731
    monkeypatch.setattr(discover_module, "get_env", guard)
    monkeypatch.setattr(edge_stability_module, "get_env", guard)

    result = _discover(cfg, tmp_path)  # raises if any production path reads env.true_adj_matrix

    assert result["calibrated"] is False
    assert Path(result["raw_posterior"]).exists()
    assert result["state_edges"] > 0

    # Ground truth is readable only AFTER the artifact is frozen, by a separate scorer path.
    real_env = get_env(cfg)  # real (unguarded) env
    assert real_env.true_adj_matrix.shape[0] == real_env.get_state_dim()


def test_same_target_calibration_is_rejected(tmp_path):
    """A calibration run in the SAME environment as the target is rejected up front with a
    precise error -- its labels equal the target's true adjacency (Plan 002 Step 2)."""
    cfg = _tiny_cfg()
    cal_run = tmp_path / "cal_run"
    cal_run.mkdir()
    shutil.copyfile(
        Path(discover_module.__file__).parents[2] / "configs" / "env_i_cdl.yaml",
        cal_run / "config.yaml",
    )
    with pytest.raises(ValueError, match="same-target"):
        _discover(cfg, tmp_path, calibration_run=str(cal_run))


def test_held_out_calibration_produces_calibrated_posterior(tmp_path):
    """With a DISTINCT held-out environment's run, discovery fits an isotonic map on that run's
    scores + labels and writes a CALIBRATED, downstream-sampleable posterior whose provenance
    records the held-out identity + checkpoint hash (Plan 002 Step 1.2)."""
    cal_cfg = _tiny_cfg("env_ii_cdl.yaml")
    # Env II needs a slightly larger budget than Env I to freeze a non-empty enumeration graph.
    cal_result = _discover(cal_cfg, tmp_path / "cal_out", train_steps=400, cmi_batches=16)
    cal_run = cal_result["run_dir"]

    cfg = _tiny_cfg()  # target = EnvironmentI
    env = get_env(cfg)
    result = _discover(cfg, tmp_path / "out", calibration_run=cal_run)

    assert result["calibrated"] is True
    assert result["raw_posterior"] is None
    cal_path = tmp_path / "out" / "EnvironmentI_posterior.json"
    assert result["calibrated_posterior"] == str(cal_path)
    assert cal_path.exists()
    assert not (tmp_path / "out" / "EnvironmentI_raw_posterior.json").exists()

    post = GraphPosterior.load(cal_path)
    assert post.is_calibrated
    assert post.marginals().shape == (env.get_state_dim(), env.get_state_dim())
    assert post.node_names == node_names(env)
    validate_posterior_artifact(post, env, cfg.environment)

    provenance = post.meta["calibration"]
    assert provenance["mode"] == "held_out"
    assert provenance["environment"] == "EnvironmentII"
    assert provenance["run"] == str(cal_run)
    assert provenance["checkpoint_sha256"] and len(provenance["checkpoint_sha256"]) == 64
    assert post.meta["calibration_runs"] == [str(cal_run)]

    # The calibrated posterior IS accepted by the downstream structure sampler.
    sampler = PosteriorStructureSampler.from_artifact(
        cal_path, env=env, environment=cfg.environment
    )
    assert sampler is not None
