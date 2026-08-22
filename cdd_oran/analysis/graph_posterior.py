"""Calibrated graph posterior: turn CMI edge evidence into a DISTRIBUTION over
structures (Phase 1 of the journal upgrade, see .temp/new_arch/04_experiment_plan.md).

This is an ADDITIVE, opt-in code path. It does NOT touch the hard-threshold graph
(`CDL.get_binary_graph`) that enumerates conflicts -- that crisp discrete decision
stays byte-identical. Here we build a per-edge inclusion probability, CALIBRATE it,
CHECK the calibration (reliability / coverage), and expose a clean way to SAMPLE a
binary structure and to read per-edge marginals. Phase 2's dynamics ensemble consumes
the samples; Phase 1 only has to produce a calibrated, checkable distribution.

Why calibration is needed (reviewer critique 06, ranks #2/#4): a transformed CMI score
is NOT a probability, and raw bootstrap inclusion frequencies are "not a calibrated
posterior under dependent data / weak interventions." So we fit a monotone map from raw
evidence to empirical inclusion probability and REPORT the reliability curve + a coverage
number rather than assuming the raw scores are calibrated.

Bootstrap primitive is reused from `cdd_oran.analysis.edge_stability` (frozen predictor,
resample the transition pool, refit CMI, aggregate per-edge selection frequency).

Usage:
    uv run python -m cdd_oran.analysis.graph_posterior --run RUN_DIR \\
        --calibration-run CALIBRATION_RUN [--B 50] \\
        [--n-transitions 2048] [--seed 0] [--device cuda] [--out OUT.json]
    uv run python -m cdd_oran.analysis.graph_posterior --self-check

No new heavy dependency: numpy + torch only. sklearn is NOT installed in this repo, so
the isotonic calibrator is a small pool-adjacent-violators implementation below.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from cdd_oran.analysis.edge_stability import edge_stability


# --------------------------------------------------------------------------- #
# Calibration primitives (pure numpy; sklearn is not available in this repo).
# --------------------------------------------------------------------------- #
def _candidate_mask(shape):
    """Return the estimable edge mask; forced self-loops are never candidates."""
    if len(shape) >= 2 and shape[-1] == shape[-2]:
        mask = np.ones(shape, dtype=bool)
        diagonal = np.arange(shape[-1])
        mask[..., diagonal, diagonal] = False
        return mask
    return np.ones(shape, dtype=bool)


def _flatten_candidates(scores, labels, candidate_mask=None):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if scores.shape != labels.shape:
        raise ValueError(f"scores shape {scores.shape} != labels shape {labels.shape}")
    if candidate_mask is None:
        candidate_mask = _candidate_mask(scores.shape)
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    if candidate_mask.shape != scores.shape:
        raise ValueError(
            f"candidate_mask shape {candidate_mask.shape} != scores shape {scores.shape}"
        )
    return scores[candidate_mask].ravel(), labels[candidate_mask].ravel()


def _pav(y, w):
    """Pool-adjacent-violators: least-squares non-decreasing fit of ``y`` (already
    sorted by the predictor) with weights ``w``. Returns the fitted values."""
    values, weights, sizes = [], [], []
    for yi, wi in zip(y, w, strict=True):
        v, cw, sz = float(yi), float(wi), 1
        while values and values[-1] >= v:
            pv, pw, ps = values.pop(), weights.pop(), sizes.pop()
            v = (pv * pw + v * cw) / (pw + cw)
            cw += pw
            sz += ps
        values.append(v)
        weights.append(cw)
        sizes.append(sz)
    out = np.empty(len(y), dtype=float)
    idx = 0
    for v, sz in zip(values, sizes, strict=True):
        out[idx : idx + sz] = v
        idx += sz
    return out


class IsotonicCalibrator:
    """Monotone (non-decreasing) map from a raw edge score to a calibrated probability.

    Fit against a labelled reference (e.g. ``env.true_adj_matrix`` on a validation
    environment). Prediction linearly interpolates the PAV fit and clips to [0, 1].
    """

    def __init__(self):
        self.x_ = np.array([0.0, 1.0])
        self.y_ = np.array([0.0, 1.0])

    def fit(self, scores, labels, candidate_mask=None):
        scores, labels = _flatten_candidates(scores, labels, candidate_mask)
        if scores.size == 0:
            return self
        order = np.argsort(scores, kind="mergesort")
        xs = scores[order]
        ys = labels[order]
        # Equal predictor scores are one function value. Aggregate their labels before
        # PAV and use the multiplicity as the least-squares weight; otherwise the fit
        # depends on the stable-sort order within a tied score.
        uniq, first, cnt = np.unique(xs, return_index=True, return_counts=True)
        sums = np.add.reduceat(ys, first)
        means = sums / cnt
        fitted = _pav(means, cnt)
        self.x_ = uniq
        self.y_ = np.clip(fitted, 0.0, 1.0)
        return self

    def predict(self, scores):
        scores = np.asarray(scores, dtype=float)
        flat = np.interp(scores.ravel(), self.x_, self.y_)
        return np.clip(flat, 0.0, 1.0).reshape(scores.shape)


def reliability_curve(probs, labels, n_bins=10, candidate_mask=None):
    """Binned reliability data: for each populated bin, the mean predicted probability
    and the empirical inclusion frequency. Equal-width bins over [0, 1]."""
    probs, labels = _flatten_candidates(probs, labels, candidate_mask)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # rightmost edge inclusive so p == 1.0 lands in the last bin
    idx = np.clip(np.digitize(probs, edges[1:-1], right=False), 0, n_bins - 1)
    curve = []
    for b in range(n_bins):
        sel = idx == b
        count = int(sel.sum())
        if count == 0:
            continue
        curve.append(
            {
                "bin_lower": float(edges[b]),
                "bin_upper": float(edges[b + 1]),
                "mean_pred": float(probs[sel].mean()),
                "empirical_freq": float(labels[sel].mean()),
                "count": count,
            }
        )
    return curve


def _sampled_structure_hpd_coverage(
    edge_probs, heldout_graphs, level=0.9, n_samples=4096, seed=0
):
    """Coverage of held-out structures by an empirical sampled-structure HPD set."""
    probs = np.asarray(edge_probs, dtype=float)
    graphs = np.asarray(heldout_graphs, dtype=float)
    if graphs.ndim == 2:
        graphs = graphs[None, ...]
    if graphs.size == 0:
        return 0.0
    mask = _candidate_mask(probs.shape)
    rng = np.random.default_rng(seed)
    draws = (rng.random((n_samples, *probs.shape)) < probs).astype(float)
    draws[..., ~mask] = 0.0
    p = np.clip(probs, 1e-12, 1.0 - 1e-12)
    log_prob = (
        draws * np.log(p) + (1.0 - draws) * np.log1p(-p)
    )[..., mask].sum(axis=-1)
    cutoff = float(np.quantile(log_prob, 1.0 - level))
    heldout = np.clip(graphs, 0.0, 1.0)
    heldout[..., ~mask] = 0.0
    heldout_log_prob = (
        heldout * np.log(p) + (1.0 - heldout) * np.log1p(-p)
    )[..., mask].sum(axis=-1)
    return float(np.mean(heldout_log_prob >= cutoff))


def calibration_summary(probs, labels, n_bins=10, level=0.9, candidate_mask=None):
    """Reliability curve + scalar calibration diagnostics on labelled edges.

    Returns finite numbers for any non-empty input:
      * ``ece``   -- expected calibration error (count-weighted |mean_pred - freq|).
      * ``mce``   -- maximum calibration error over populated bins.
      * ``brier`` -- mean squared error of probs vs labels.
      * ``coverage`` -- held-out fraction of candidate edges whose true label lies in the per-edge
         ``level`` central credible set of Bernoulli(p): {1} if p>=level, {0} if
         p<=1-level, else {0,1}. A confident, well-calibrated posterior covers ~level.
      * ``avg_credible_set_cardinality`` / ``sharpness`` -- set size and its normalized
        complement, excluding impossible self-loops.
      * ``sampled_structure_hpd_coverage`` -- held-out graph coverage by an empirical
        HPD set of sampled structures.
    """
    probs_array = np.asarray(probs, dtype=float)
    labels_array = np.asarray(labels, dtype=float)
    probs, labels = _flatten_candidates(probs_array, labels_array, candidate_mask)
    n = probs.size
    curve = reliability_curve(probs_array, labels_array, n_bins, candidate_mask)
    if n == 0:
        return {
            "n_edges": 0,
            "ece": 0.0,
            "mce": 0.0,
            "brier": 0.0,
            "coverage": 0.0,
            "marginal_coverage": 0.0,
            "avg_credible_set_cardinality": 0.0,
            "sharpness": 0.0,
            "sampled_structure_hpd_coverage": 0.0,
            "level": float(level),
            "reliability_curve": curve,
        }
    ece = sum(bin_["count"] * abs(bin_["mean_pred"] - bin_["empirical_freq"]) for bin_ in curve) / n
    mce = max((abs(b["mean_pred"] - b["empirical_freq"]) for b in curve), default=0.0)
    brier = float(np.mean((probs - labels) ** 2))
    alpha = 1.0 - level
    lower_only = probs <= alpha  # credible set {0}
    upper_only = probs >= level  # credible set {1}
    covered = np.where(
        upper_only, labels == 1.0, np.where(lower_only, labels == 0.0, True)
    )
    cardinality = np.where(lower_only | upper_only, 1.0, 2.0)
    avg_cardinality = float(np.mean(cardinality))
    return {
        "n_edges": int(n),
        "ece": float(ece),
        "mce": float(mce),
        "brier": brier,
        "coverage": float(np.mean(covered)),
        "marginal_coverage": float(np.mean(covered)),
        "avg_credible_set_cardinality": avg_cardinality,
        "sharpness": float(2.0 - avg_cardinality),
        "sampled_structure_hpd_coverage": _sampled_structure_hpd_coverage(
            probs_array, labels_array, level=level
        ),
        "level": float(level),
        "reliability_curve": curve,
    }


def transformed_cmi_probs(cmi_matrix, temperature=1.0):
    """A DELIBERATELY naive baseline: squash raw CMI into [0, 1] with a logistic of the
    z-scored score. Used only to DEMONSTRATE that a transformed CMI is not calibrated --
    it is never fed to the planner. Returns a (fd, fd) matrix (action column dropped,
    diagonal zeroed), matching the marginals layout.

    # ponytail: fixed z-score+logistic squash; there is no reason its output matches
    # inclusion probability -- that is exactly the gap the calibration check quantifies.
    """
    cmi = np.asarray(cmi_matrix, dtype=float)
    core = cmi[:, :-1] if cmi.shape[1] > cmi.shape[0] else cmi
    flat = core[~np.eye(core.shape[0], dtype=bool)]
    mean, std = float(flat.mean()), float(flat.std())
    std = std if std > 1e-8 else 1.0
    probs = 1.0 / (1.0 + np.exp(-(core - mean) / (std * temperature)))
    np.fill_diagonal(probs, 0.0)
    return probs


# --------------------------------------------------------------------------- #
# The posterior.
# --------------------------------------------------------------------------- #
class GraphPosterior:
    """A calibrated distribution over binary one-step dependency structures.

    Independent per-edge Bernoulli marginals ``edge_probs`` (shape ``(fd, fd)``, child
    rows x parent cols, self-loops forced to 0). ``sample`` draws valid binary
    one-step dependency structures for the Phase 2 dynamics ensemble; ``marginals`` returns per-edge
    probabilities; ``map_graph`` gives the point (MAP) structure WITHOUT touching the
    enumeration graph.

    These are temporal dependency masks from t to t+1, not instantaneous SEM DAGs;
    reciprocal dependencies across time are therefore semantically valid.
    """

    def __init__(self, edge_probs, node_names=None, calibrator=None, meta=None):
        probs = np.asarray(edge_probs, dtype=float).copy()
        if probs.ndim != 2 or probs.shape[0] != probs.shape[1]:
            raise ValueError(f"edge_probs must be square (fd, fd); got {probs.shape}")
        np.fill_diagonal(probs, 0.0)
        self.edge_probs = np.clip(probs, 0.0, 1.0)
        self.node_names = list(node_names) if node_names is not None else None
        self.calibrator = calibrator
        self.meta = dict(meta) if meta else {}

    @property
    def n_nodes(self):
        return self.edge_probs.shape[0]

    def marginals(self):
        """Per-edge inclusion probabilities, ``(fd, fd)``, in [0, 1]."""
        return self.edge_probs.copy()

    def sample(self, rng=None, size=None):
        """Draw binary structure(s) from the posterior (independent Bernoulli per edge).

        Returns a ``(fd, fd)`` int array, or ``(size, fd, fd)`` when ``size`` is given.
        Self-loops are always 0, so every draw is a valid off-diagonal binary graph.
        """
        rng = np.random.default_rng() if rng is None else rng
        shape = self.edge_probs.shape if size is None else (size, *self.edge_probs.shape)
        draws = (rng.random(shape) < self.edge_probs).astype(int)
        if size is None:
            np.fill_diagonal(draws, 0)
        else:
            for k in range(size):
                np.fill_diagonal(draws[k], 0)
        return draws

    def map_graph(self, threshold=0.5):
        """Point (MAP) structure at a marginal cut. Provided for convenience only; the
        conflict-enumeration graph remains ``CDL.get_binary_graph`` and is unchanged."""
        graph = (self.edge_probs >= threshold).astype(int)
        np.fill_diagonal(graph, 0)
        return graph

    def calibration_check(self, labels, n_bins=10, level=0.9):
        """Held-out reliability / coverage against a labelled reference graph."""
        return calibration_summary(self.edge_probs, self._labels(labels), n_bins, level)

    def calibrate(
        self,
        labels,
        n_bins=10,
        level=0.9,
        *,
        evaluation_labels=None,
        calibration_scores=None,
    ):
        """Fit on calibration labels and score only on separate evaluation labels.

        ``calibration_scores`` is used when the labelled calibration environment has a
        different posterior score matrix than this target posterior. Refuse to report
        in-sample metrics: target labels must be supplied separately as
        ``evaluation_labels``. When calibration and evaluation runs share an environment,
        this is within-environment score-to-label calibration, not held-out structural
        coverage or graph-topology generalization.
        """
        if evaluation_labels is None:
            raise ValueError("evaluation_labels is required; calibration metrics must be held out")
        lab = np.asarray(labels, dtype=float)
        score_source = self.edge_probs if calibration_scores is None else np.asarray(calibration_scores)
        if lab.shape != score_source.shape:
            raise ValueError(f"calibration labels shape {lab.shape} != scores shape {score_source.shape}")
        cal = IsotonicCalibrator().fit(score_source, lab)
        posterior = self.apply_calibrator(cal)
        eval_lab = self._labels(evaluation_labels)
        before = calibration_summary(self.edge_probs, eval_lab, n_bins, level)
        after = calibration_summary(posterior.edge_probs, eval_lab, n_bins, level)
        summary = {
            "before": before,
            "after": after,
            "ece_reduction": before["ece"] - after["ece"],
            "brier_reduction": before["brier"] - after["brier"],
            "calibrator": {"x": cal.x_, "y": cal.y_},
        }
        return posterior, summary

    def apply_calibrator(self, calibrator):
        """Apply a pre-fitted calibration map without using target labels."""
        if not isinstance(calibrator, IsotonicCalibrator):
            raise TypeError("calibrator must be an IsotonicCalibrator")
        return GraphPosterior(
            calibrator.predict(self.edge_probs),
            node_names=self.node_names,
            calibrator=calibrator,
            meta={**self.meta, "calibrated": True},
        )

    @property
    def is_calibrated(self):
        """True only if this posterior carries an explicit calibrated=True flag (set by
        ``apply_calibrator``). P2 requires this before consuming the artifact."""
        return bool(self.meta.get("calibrated", False))

    # ---- artifact persistence (review blocker #2) --------------------------- #
    def save(self, path):
        """Serialize the held-out-CALIBRATED posterior artifact to JSON: marginals + the
        fitted isotonic knots + metadata (incl. the ``calibrated`` flag and node layout).
        P2's boundary loads this and rejects any posterior that is not calibrated."""
        payload = {
            "edge_probs": self.edge_probs.tolist(),
            "node_names": self.node_names,
            "calibrator": (
                {"x": self.calibrator.x_.tolist(), "y": self.calibrator.y_.tolist()}
                if self.calibrator is not None
                else None
            ),
            "meta": self.meta,
        }
        Path(path).write_text(json.dumps(payload, indent=2))
        return path

    @classmethod
    def load(cls, path):
        """Load a posterior artifact written by ``save``. Restores the fitted calibrator and
        the ``calibrated`` metadata flag; does NOT touch any RNG."""
        payload = json.loads(Path(path).read_text())
        calibrator = None
        cal = payload.get("calibrator")
        if cal is not None:
            calibrator = IsotonicCalibrator()
            calibrator.x_ = np.asarray(cal["x"], dtype=float)
            calibrator.y_ = np.asarray(cal["y"], dtype=float)
        return cls(
            np.asarray(payload["edge_probs"], dtype=float),
            node_names=payload.get("node_names"),
            calibrator=calibrator,
            meta=payload.get("meta") or {},
        )

    def _labels(self, labels):
        lab = np.asarray(labels, dtype=float)
        if lab.shape != self.edge_probs.shape:
            raise ValueError(
                f"labels shape {lab.shape} != marginals shape {self.edge_probs.shape}"
            )
        if not np.all(np.isin(lab, (0.0, 1.0))):
            raise ValueError("labels must be binary")
        return lab

    # ---- constructors -------------------------------------------------------- #
    @classmethod
    def from_frequencies(cls, freq, node_names=None, meta=None):
        """Build directly from a per-edge inclusion-frequency matrix (e.g. the
        ``frequency_matrix`` returned by ``edge_stability``)."""
        return cls(freq, node_names=node_names, meta=meta)

    @classmethod
    def from_transformed_cmi(cls, cmi_matrix, node_names=None, temperature=1.0):
        """UNCALIBRATED baseline posterior from raw CMI (for the gap demonstration)."""
        return cls(
            transformed_cmi_probs(cmi_matrix, temperature),
            node_names=node_names,
            meta={"source": "transformed_cmi", "calibrated": False},
        )

    @classmethod
    def from_bootstrap(cls, run_dir, B=50, n_transitions=2048, seed=0, device=None):
        """Bootstrap-with-calibration entry point: reuse ``edge_stability`` to get
        per-edge inclusion frequencies from the FROZEN predictor, then wrap them as a
        posterior. Requires a trained CDL run; not exercised by the CPU test."""
        result = edge_stability(
            run_dir, B=B, n_transitions=n_transitions, seed=seed, device=device
        )
        freq = np.asarray(result["frequency_matrix"], dtype=float)
        return cls(
            freq,
            meta={
                "source": "bootstrap",
                "run_dir": result["run_dir"],
                "environment": result["environment"],
                "B": result["B"],
                "n_transitions": result["n_transitions"],
            },
        )


# --------------------------------------------------------------------------- #
# Self-check + CLI.
# --------------------------------------------------------------------------- #
def _self_check():
    """CPU-only calibration check with separate, overlapping calibration and test labels."""
    rng = np.random.default_rng(0)
    fd = 3
    raw = np.array([[0.0, 0.1, 0.1], [0.5, 0.0, 0.5], [0.9, 0.9, 0.0]])
    calibration_labels = np.array([[0, 0, 1], [0, 0, 1], [1, 1, 0]], dtype=float)
    heldout_labels = np.array([[0, 0, 0], [1, 0, 1], [1, 0, 0]], dtype=float)
    post = GraphPosterior.from_frequencies(raw, node_names=[f"n{i}" for i in range(fd)])

    tie = IsotonicCalibrator().fit([0.0, 0.0, 1.0], [0.0, 1.0, 0.0])
    assert np.allclose(tie.x_, [0.0, 1.0])
    assert np.allclose(tie.y_, [1.0 / 3.0, 1.0 / 3.0])

    marg = post.marginals()
    assert marg.shape == (fd, fd), "marginals must be (fd, fd)"
    assert np.all((marg >= 0.0) & (marg <= 1.0)), "marginals must be in [0, 1]"
    assert np.all(np.diag(marg) == 0.0), "self-loops must be 0"

    single = post.sample(rng)
    assert single.shape == (fd, fd) and set(np.unique(single)).issubset({0, 1}), "sample must be binary (fd,fd)"
    assert np.all(np.diag(single) == 0), "sampled self-loops must be 0"
    batch = post.sample(rng, size=32)
    assert batch.shape == (32, fd, fd), "batched sample shape wrong"
    # Empirical inclusion of a high-prob edge over many draws tracks its marginal.
    big = post.sample(rng, size=4000)
    emp = big[:, 1, 0].mean()
    assert abs(emp - marg[1, 0]) < 0.05, f"sample frequency {emp:.3f} != marginal {marg[1,0]:.3f}"

    calibrated, summary = post.calibrate(
        calibration_labels, evaluation_labels=heldout_labels
    )
    for key in ("ece", "brier", "coverage", "mce"):
        assert np.isfinite(summary["before"][key]), f"before.{key} not finite"
        assert np.isfinite(summary["after"][key]), f"after.{key} not finite"
    assert summary["after"]["n_edges"] == fd * (fd - 1)
    assert np.isclose(summary["after"]["ece"], 1.0 / 6.0)
    assert np.isclose(summary["after"]["brier"], 1.0 / 3.0)
    assert 1.0 <= summary["after"]["avg_credible_set_cardinality"] <= 2.0
    assert np.isfinite(summary["after"]["sampled_structure_hpd_coverage"])
    assert np.all((calibrated.marginals() >= 0.0) & (calibrated.marginals() <= 1.0))

    # An uninformative score must remain uninformative after calibration.
    uninformative = GraphPosterior.from_frequencies(np.full((fd, fd), 0.5))
    uncalibrated, negative_summary = uninformative.calibrate(
        calibration_labels, evaluation_labels=heldout_labels
    )
    candidate = _candidate_mask((fd, fd))
    assert np.ptp(uncalibrated.marginals()[candidate]) < 1e-12
    assert np.isfinite(negative_summary["after"]["brier"])

    # Show the uncalibrated transformed-CMI baseline is measurable without diagonals.
    gt = calibration_labels
    cmi = np.zeros((fd, fd + 1))
    cmi[:, :fd] = gt * 80.0 + rng.random((fd, fd)) * 2.0
    tcmi = GraphPosterior.from_transformed_cmi(cmi)
    tcmi_summary = tcmi.calibration_check(heldout_labels)
    assert np.isfinite(tcmi_summary["ece"]), "transformed-CMI ECE not finite"

    print(
        "graph_posterior self-check passed: "
        f"raw ECE {summary['before']['ece']:.3f} -> calibrated {summary['after']['ece']:.3f}, "
        f"held-out coverage {summary['after']['coverage']:.2f}, "
        f"raw-transformed-CMI ECE {tcmi_summary['ece']:.3f}"
    )


def _jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run")
    p.add_argument(
        "--calibration-run",
        action="append",
        help=(
            "Separate labelled run used only to fit within-environment calibration "
            "(repeatable; not structural coverage)"
        ),
    )
    p.add_argument("--B", type=int, default=50)
    p.add_argument("--n-transitions", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out", help="Write the calibration summary + marginals JSON here")
    p.add_argument(
        "--save-posterior",
        help="Write the CALIBRATED posterior artifact (GraphPosterior.save) here for P2 to load",
    )
    p.add_argument("--self-check", action="store_true", help="Run the synthetic check and exit")
    args = p.parse_args(argv)
    if args.self_check:
        _self_check()
        return 0
    if not args.run:
        p.error("--run is required unless --self-check is given")
    if not args.calibration_run:
        p.error("at least one --calibration-run is required; target labels must be held out")

    # Build the target posterior first. Its labels are not exposed until after the
    # calibration map has been fitted on separate labelled calibration runs.
    from dataclasses import replace

    from cdd_oran.config import load_config
    from cdd_oran.envs import get_env

    post = GraphPosterior.from_bootstrap(
        args.run, B=args.B, n_transitions=args.n_transitions, seed=args.seed, device=args.device
    )
    calibration_scores = []
    calibration_labels = []
    for calibration_run in args.calibration_run:
        if Path(calibration_run).resolve() == Path(args.run).resolve():
            p.error("--calibration-run must be different from --run")
        reference = GraphPosterior.from_bootstrap(
            calibration_run,
            B=args.B,
            n_transitions=args.n_transitions,
            seed=args.seed,
            device=args.device,
        )
        reference_cfg = load_config(Path(calibration_run) / "config.yaml")
        if args.device:
            reference_cfg = replace(reference_cfg, device=args.device)
        reference_env = get_env(reference_cfg)
        calibration_scores.append(reference.marginals())
        calibration_labels.append(np.asarray(reference_env.true_adj_matrix, dtype=float))

    cal_scores = np.concatenate(
        [scores[_candidate_mask(scores.shape)] for scores in calibration_scores]
    )
    cal_labels = np.concatenate(
        [labels[_candidate_mask(labels.shape)] for labels in calibration_labels]
    )
    calibrator = IsotonicCalibrator().fit(cal_scores, cal_labels)
    calibrated = post.apply_calibrator(calibrator)

    cfg = load_config(Path(args.run) / "config.yaml")
    if args.device:
        cfg = replace(cfg, device=args.device)
    env = get_env(cfg)
    gt = np.asarray(env.true_adj_matrix, dtype=float)
    before = post.calibration_check(gt)
    after = calibrated.calibration_check(gt)
    summary = {
        "before": before,
        "after": after,
        "ece_reduction": before["ece"] - after["ece"],
        "brier_reduction": before["brier"] - after["brier"],
        "calibrator": {"x": calibrator.x_, "y": calibrator.y_},
        "calibration_runs": [str(Path(run)) for run in args.calibration_run],
    }

    payload = {
        "run_dir": args.run,
        "meta": post.meta,
        "calibration": summary,
        "raw_marginals": post.marginals(),
        "calibrated_marginals": calibrated.marginals(),
        "calibrator": {"x": calibrator.x_, "y": calibrator.y_},
    }
    print(json.dumps({"meta": post.meta, "calibration": summary}, indent=2, default=_jsonable))
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2, default=_jsonable))
    if args.save_posterior:
        # `calibrated` carries meta calibrated=True from apply_calibrator; save it as the
        # artifact P2 consumes. Its node layout comes from the target run's env.
        calibrated.node_names = [f"param{i}" for i in range(env.num_params)] + [
            kpi.name for kpi in env.kpis
        ]
        calibrated.meta = {**calibrated.meta, "calibration_runs": summary["calibration_runs"]}
        calibrated.save(args.save_posterior)
        print(f"calibrated posterior artifact written to {args.save_posterior}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
