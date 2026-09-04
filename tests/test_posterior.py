"""CPU tests for the calibrated graph posterior (Phase 1). No GPU, no training.

Covers the three checks named in the P1 brief:
  (a) sampled structures are valid binary graphs of the right shape;
  (b) marginal probabilities are in [0, 1];
  (c) calibration is fitted on separate labels and scored on held-out labels.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest

import cdd_oran.analysis.graph_posterior as posterior_module
from cdd_oran.analysis.graph_posterior import (
    GraphPosterior,
    IsotonicCalibrator,
    calibration_summary,
    reliability_curve,
)


def _tiny_posterior():
    """A 4-node posterior with three true edges and mildly noisy marginals."""
    rng = np.random.default_rng(0)
    fd = 4
    gt = np.zeros((fd, fd))
    gt[1, 0] = gt[2, 0] = gt[3, 1] = 1.0
    probs = np.full((fd, fd), 0.3)
    probs[gt == 1.0] = 0.75
    probs += rng.normal(scale=0.03, size=probs.shape)
    probs = np.clip(probs, 0.0, 1.0)
    np.fill_diagonal(probs, 0.0)
    return GraphPosterior.from_frequencies(probs, node_names=[f"n{i}" for i in range(fd)]), gt


def test_sampled_structures_are_valid_binary_graphs():
    post, _ = _tiny_posterior()
    rng = np.random.default_rng(1)

    single = post.sample(rng)
    assert single.shape == (post.n_nodes, post.n_nodes), "(a) sample must be (fd, fd)"
    assert set(np.unique(single)).issubset({0, 1}), "(a) sample must be binary"
    assert np.all(np.diag(single) == 0), "(a) sampled graph must have no self-loops"

    batch = post.sample(rng, size=16)
    assert batch.shape == (16, post.n_nodes, post.n_nodes), "(a) batched sample shape wrong"
    assert set(np.unique(batch)).issubset({0, 1}), "(a) batched sample must be binary"
    for graph in batch:
        assert np.all(np.diag(graph) == 0), "(a) each sampled graph must have no self-loops"


def test_sample_frequency_tracks_marginal():
    """Independent-Bernoulli sampling: empirical inclusion converges to the marginal."""
    post, _ = _tiny_posterior()
    rng = np.random.default_rng(2)
    draws = post.sample(rng, size=5000)
    emp = draws.mean(axis=0)
    assert np.max(np.abs(emp - post.marginals())) < 0.05, "sample frequencies must track marginals"


def test_marginals_are_probabilities():
    post, _ = _tiny_posterior()
    marg = post.marginals()
    assert marg.shape == (post.n_nodes, post.n_nodes)
    assert np.all((marg >= 0.0) & (marg <= 1.0)), "(b) marginals must be in [0, 1]"
    assert np.all(np.diag(marg) == 0.0), "(b) self-loops must be 0"
    # Constructor must clamp out-of-range inputs into [0, 1].
    bad = GraphPosterior.from_frequencies(np.array([[0.0, 2.0], [-1.0, 0.0]]))
    assert np.all((bad.marginals() >= 0.0) & (bad.marginals() <= 1.0))


def test_calibration_check_returns_finite_numbers():
    post, gt = _tiny_posterior()
    summary = post.calibration_check(gt)
    for key in ("ece", "mce", "brier", "coverage"):
        assert np.isfinite(summary[key]), f"(c) {key} must be finite"
    assert 0.0 <= summary["ece"] <= 1.0
    assert 0.0 <= summary["coverage"] <= 1.0
    assert summary["n_edges"] == gt.size - len(gt)
    assert 1.0 <= summary["avg_credible_set_cardinality"] <= 2.0
    assert 0.0 <= summary["sharpness"] <= 1.0
    assert 0.0 <= summary["sampled_structure_hpd_coverage"] <= 1.0
    assert isinstance(summary["reliability_curve"], list)
    for row in summary["reliability_curve"]:
        assert np.isfinite(row["mean_pred"]) and np.isfinite(row["empirical_freq"])


def test_calibration_reduces_ece_and_stays_finite():
    """The isotonic map is fit on one label set and scored on another."""
    post, gt = _tiny_posterior()
    heldout = gt.copy()
    heldout[1, 0] = 0.0
    calibrated, summary = post.calibrate(gt, evaluation_labels=heldout)
    assert np.isfinite(summary["before"]["ece"]) and np.isfinite(summary["after"]["ece"])
    assert summary["after"]["n_edges"] == gt.size - len(gt)
    assert summary["calibrator"]["x"] is not None
    marg = calibrated.marginals()
    assert np.all((marg >= 0.0) & (marg <= 1.0)), "calibrated marginals must stay in [0, 1]"
    # A calibrated posterior still samples valid binary structures.
    graph = calibrated.sample(np.random.default_rng(3))
    assert graph.shape == (post.n_nodes, post.n_nodes)
    assert set(np.unique(graph)).issubset({0, 1})


def test_isotonic_calibrator_is_monotone_and_bounded():
    cal = IsotonicCalibrator()
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    labels = np.array([0, 0, 1, 0, 1, 1, 1, 1], dtype=float)
    cal.fit(scores, labels)
    grid = np.linspace(0.0, 1.0, 21)
    out = cal.predict(grid)
    assert np.all((out >= 0.0) & (out <= 1.0)), "calibrated probs must be in [0, 1]"
    assert np.all(np.diff(out) >= -1e-9), "isotonic map must be non-decreasing"


def test_isotonic_tied_scores_are_order_invariant_and_weighted():
    expected = np.array([1.0 / 3.0, 1.0 / 3.0])
    first = IsotonicCalibrator().fit([0.0, 0.0, 1.0], [0.0, 1.0, 0.0])
    second = IsotonicCalibrator().fit([0.0, 0.0, 1.0], [1.0, 0.0, 0.0])
    assert np.array_equal(first.x_, [0.0, 1.0])
    assert np.allclose(first.y_, expected)
    assert np.allclose(second.y_, expected)


def test_calibration_metrics_exclude_forced_zero_diagonal():
    probs = np.array([[0.0, 0.2], [0.8, 0.0]])
    labels = np.array([[1.0, 0.0], [1.0, 1.0]])
    summary = calibration_summary(probs, labels, n_bins=2)
    assert summary["n_edges"] == 2
    assert np.isclose(summary["brier"], (0.2**2 + 0.2**2) / 2.0)
    assert np.isclose(summary["coverage"], 1.0)


def test_calibrate_requires_heldout_labels():
    post, gt = _tiny_posterior()
    with pytest.raises(ValueError, match="held out"):
        post.calibrate(gt)


def test_heldout_self_check_fixture_has_known_metrics_and_hpd():
    raw = np.array([[0.0, 0.1, 0.1], [0.5, 0.0, 0.5], [0.9, 0.9, 0.0]])
    calibration_labels = np.array([[0, 0, 1], [0, 0, 1], [1, 1, 0]], dtype=float)
    heldout_labels = np.array([[0, 0, 0], [1, 0, 1], [1, 0, 0]], dtype=float)
    post = GraphPosterior.from_frequencies(raw)
    calibrated, summary = post.calibrate(
        calibration_labels, evaluation_labels=heldout_labels
    )
    assert np.allclose(summary["calibrator"]["x"], [0.1, 0.5, 0.9])
    assert np.allclose(summary["calibrator"]["y"], [0.5, 0.5, 1.0])
    assert np.isclose(summary["after"]["ece"], 1.0 / 6.0)
    assert np.isclose(summary["after"]["brier"], 1.0 / 3.0)
    assert 0.0 <= summary["after"]["sampled_structure_hpd_coverage"] <= 1.0
    assert np.all(np.diag(calibrated.marginals()) == 0.0)


def test_uninformative_scores_do_not_create_discrimination():
    labels = np.array([[0, 0, 1], [0, 0, 1], [1, 1, 0]], dtype=float)
    post = GraphPosterior.from_frequencies(np.full((3, 3), 0.5))
    calibrated, summary = post.calibrate(labels, evaluation_labels=labels)
    mask = ~np.eye(3, dtype=bool)
    assert np.ptp(calibrated.marginals()[mask]) < 1e-12
    assert np.isfinite(summary["after"]["ece"])


def test_reliability_curve_bins_are_consistent():
    probs = np.array([0.05, 0.15, 0.95, 0.85, 0.5])
    labels = np.array([0, 0, 1, 1, 1], dtype=float)
    curve = reliability_curve(probs, labels, n_bins=10)
    assert sum(row["count"] for row in curve) == probs.size, "every edge must land in one bin"
    for row in curve:
        assert row["bin_lower"] <= row["mean_pred"] <= row["bin_upper"] + 1e-9


def test_calibration_summary_handles_empty_input():
    summary = calibration_summary(np.array([]), np.array([]))
    assert summary["n_edges"] == 0
    for key in ("ece", "mce", "brier", "coverage"):
        assert np.isfinite(summary[key])


def test_transformed_cmi_posterior_builds_and_checks():
    """A raw transformed-CMI posterior (the 'not a probability' baseline) builds, yields
    marginals in [0, 1], and produces a finite calibration check on synthetic data.
    The size of its calibration GAP is a run-time deliverable, reported in the handoff,
    not asserted here (it depends on the trained model, not this fixture)."""
    post, gt = _tiny_posterior()
    fd = post.n_nodes
    rng = np.random.default_rng(4)
    cmi = np.zeros((fd, fd + 1))
    cmi[:, :fd] = gt * 80.0 + rng.random((fd, fd)) * 2.0
    tcmi = GraphPosterior.from_transformed_cmi(cmi)
    marg = tcmi.marginals()
    assert np.all((marg >= 0.0) & (marg <= 1.0)), "transformed-CMI marginals must be in [0, 1]"
    tcmi_summary = tcmi.calibration_check(gt)
    assert np.isfinite(tcmi_summary["ece"]) and np.isfinite(tcmi_summary["brier"])


def test_from_bootstrap_preserves_asymmetric_child_parent_alignment(monkeypatch):
    frequency = np.array([[0.0, 0.2, 0.8], [0.7, 0.0, 0.1], [0.3, 0.9, 0.0]])

    def fake_bootstrap_frequency(run_dir, B, n_transitions, seed, device):
        context = {
            "run_dir": str(run_dir),
            "environment": "EnvironmentI",
            "config_threshold": 0.2,
            "B": B,
            "n_transitions": n_transitions,
            "batch_size": 32,
            "seed": seed,
            "device": device,
            "baseline_pred": np.zeros_like(frequency, dtype=int),
        }
        return frequency, context

    # The producing path calls the truth-FREE bootstrap_frequency, never edge_stability.
    monkeypatch.setattr(posterior_module, "bootstrap_frequency", fake_bootstrap_frequency)
    post = GraphPosterior.from_bootstrap("calibration-run", B=3, n_transitions=7, seed=2)
    assert np.array_equal(post.marginals(), frequency)
    assert post.marginals()[0, 2] == 0.8
    assert post.marginals()[2, 1] == 0.9
    assert post.meta["environment"] == "EnvironmentI"


def test_calibrated_posterior_save_load_roundtrip_and_flag(tmp_path):
    """Artifact for P2 (review blocker #2): save/load preserves marginals, node layout and
    the fitted isotonic knots, and carries an explicit calibrated flag."""
    post, gt = _tiny_posterior()
    heldout = gt.copy()
    heldout[1, 0] = 0.0
    calibrated, _ = post.calibrate(gt, evaluation_labels=heldout)
    assert calibrated.is_calibrated

    path = tmp_path / "posterior.json"
    calibrated.save(path)
    loaded = GraphPosterior.load(path)

    assert loaded.is_calibrated
    np.testing.assert_allclose(loaded.marginals(), calibrated.marginals())
    assert loaded.node_names == calibrated.node_names
    assert loaded.calibrator is not None
    np.testing.assert_allclose(loaded.calibrator.x_, calibrated.calibrator.x_)
    np.testing.assert_allclose(loaded.calibrator.y_, calibrated.calibrator.y_)
    graph = loaded.sample(np.random.default_rng(0))
    assert graph.shape == (post.n_nodes, post.n_nodes)


def test_raw_posterior_is_not_flagged_calibrated(tmp_path):
    """A raw (uncalibrated) posterior must round-trip as NOT calibrated so P2 can reject it."""
    post, _ = _tiny_posterior()
    assert not post.is_calibrated
    path = tmp_path / "raw.json"
    post.save(path)
    assert not GraphPosterior.load(path).is_calibrated


def test_graph_posterior_cli_rejects_same_environment_calibration(tmp_path):
    """graph_posterior.main() must reject a --calibration-run from the SAME environment as the
    target run (even a DIFFERENT run path), not only an identical path: same-environment labels
    ARE the target's true adjacency, so fitting the calibrator on them leaks the answer (Plan 002
    Finding 1). Rejection happens before any bootstrap, so only config.yaml files are needed."""
    configs = Path(posterior_module.__file__).parents[2] / "configs"
    run = tmp_path / "target_run"
    run.mkdir()
    cal = tmp_path / "cal_run"
    cal.mkdir()
    shutil.copyfile(configs / "env_i_cdl.yaml", run / "config.yaml")
    shutil.copyfile(configs / "env_i_cdl.yaml", cal / "config.yaml")  # SAME environment, other path
    with pytest.raises(SystemExit):
        posterior_module.main(["--run", str(run), "--calibration-run", str(cal)])


def test_raw_posterior_artifact_is_rejected_by_structure_sampler(tmp_path):
    """The P1<->P2 boundary refuses to sample an uncalibrated posterior artifact: this is what
    stops a no-calibration discovery diagnostic from ever reaching robust structure sampling."""
    from cdd_oran.models import PosteriorStructureSampler

    post, _ = _tiny_posterior()
    path = tmp_path / "raw.json"
    post.save(path)
    with pytest.raises(ValueError, match="not calibrated"):
        PosteriorStructureSampler.from_artifact(path)
