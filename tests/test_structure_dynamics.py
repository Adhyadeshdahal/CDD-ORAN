"""CPU tests for Phase 2 structure-conditioned dynamics + bounded residual.

No GPU, no training job. The headline is the FOCUSED identity test (spec sec 5): a case
where two parents must NOT be pooled. The legacy max-pool over the source axis is
permutation-invariant, so it cannot tell "parent 0 active" from "parent 1 active"; the
structure-conditioned head must distinguish them. Plus: residual is zero at init, bounded,
its contribution fraction is finite, and the sampled-structure forward runs on CPU.
"""

import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch

from cdd_oran.config import load_config
from cdd_oran.conflicts import detect_conflict_edges
from cdd_oran.envs import get_env
from cdd_oran.models import (
    get_model,
    load_enumeration_graph,
    node_names,
    stage_run_artifacts,
    verify_checkpoint_manifest,
    verify_run_artifacts,
)
from cdd_oran.models.cdl import CDL, StatePredictor

# state_dim=3 (source 0,1 = parents, source 2 = the KPI self slot), action_dim=3,
# feature_dim=2 -> ns = state_dim + 1 = 4 source slots (0,1,2 state + 3 action).
STATE_DIM, KPI_START, ACTION_DIM, FEATURE_DIM = 3, 1, 3, 2
NS = STATE_DIM + 1


def _linear_predictor():
    """A structure-conditioned predictor whose head is a SINGLE linear map with explicit,
    analytically-known weights: mu = sum_j w_j * gated_feat_j[0], w = [1,2,3,4] over the four
    source slots, gated_feat_j = feat_j * presence_j. log_std weight is 0 -> std ~ 1."""
    torch.manual_seed(0)
    sp = StatePredictor(STATE_DIM, ACTION_DIM, FEATURE_DIM, [])
    lin = cast(torch.nn.Linear, sp.structure_predictor.net[-1])  # Linear(ns*(f+1)=12, 2)
    with torch.no_grad():
        lin.weight.zero_()
        lin.bias.zero_()
        for j, w in enumerate([1.0, 2.0, 3.0, 4.0]):
            lin.weight[0, j * (FEATURE_DIM + 1)] = w  # first feature channel of source slot j
    return sp


def _feats_first_channel(values):
    """(1, ns, f) features: source j carries ``values[j]`` in the WEIGHTED first channel and
    a large junk value in the zero-weighted second channel."""
    feats = torch.zeros(1, NS, FEATURE_DIM)
    for j, v in enumerate(values):
        feats[0, j, 0] = v
        feats[0, j, 1] = 7.0  # zero-weighted -> must not affect mu
    return feats


def test_structure_enforces_parent_exclusion_with_known_weights():
    """Strengthened identity test (review MAJOR #6): explicit weights, analytic outputs,
    direction, slot-swap, both-parents-on, AND exact invariance to every ABSENT slot (the
    proof of blocker #1: a gated-off parent cannot reach the predictor)."""
    sp = _linear_predictor()

    def mu_of(values, presence):
        feats = _feats_first_channel(values)
        structure = torch.tensor([presence], dtype=torch.float32)  # (1, ns)
        mu, std = sp.head(feats, structure)
        assert torch.allclose(std, torch.full_like(std, 1.0 + 1e-4)), "log_std weight is 0"
        return float(mu.detach().reshape(()))

    # (a) ABSENT-slot invariance (blocker #1): slot 1 is OFF -> changing its raw feature to an
    # arbitrary value leaves the output EXACTLY unchanged (not just approximately).
    absent_small = mu_of([2.0, 5.0, 0.0, 0.0], [1, 0, 0, 0])
    absent_huge = mu_of([2.0, -1e6, 0.0, 0.0], [1, 0, 0, 0])
    assert absent_small == absent_huge, "an absent parent's feature must not affect the output"
    assert absent_small == 2.0, "known-weight output mu = 1*v0 = 2.0"

    # (b) direction: the present parent's feature drives mu by exactly its slot weight.
    assert mu_of([3.0, 0.0, 0.0, 0.0], [1, 0, 0, 0]) == 3.0

    # (c) slot-swap: the SAME value on slot 1 (weight 2) instead of slot 0 (weight 1) doubles
    # the output -> slots keep identity; the head is not pooling.
    only0 = mu_of([4.0, 0.0, 0.0, 0.0], [1, 0, 0, 0])  # 1 * 4
    only1 = mu_of([0.0, 4.0, 0.0, 0.0], [0, 1, 0, 0])  # 2 * 4
    assert only0 == 4.0 and only1 == 8.0 and only1 == 2 * only0

    # (d) both-parents-on = the exact sum of the singles, distinct from either.
    both = mu_of([4.0, 4.0, 0.0, 0.0], [1, 1, 0, 0])  # 1*4 + 2*4
    assert both == only0 + only1 == 12.0
    assert both != only0 and both != only1

    # (e) legacy max-pool over the source axis is permutation-invariant (the defect): the
    # structure path above is not, so it genuinely conditions on parent identity.
    feats = _feats_first_channel([1.0, -1.0, 0.0, 0.0])
    swapped = _feats_first_channel([-1.0, 1.0, 0.0, 0.0])
    mu_pool_a, _ = sp.head(feats)
    mu_pool_b, _ = sp.head(swapped)
    torch.testing.assert_close(mu_pool_a, mu_pool_b)


def test_absent_parent_input_invariance_end_to_end():
    """Blocker #1 at the model level: with a parent absent from a KPI child's structure,
    changing that parent's raw STATE input leaves the child's GRAPH mean exactly unchanged.
    (The bounded residual is deliberately dense and may see all inputs; the graph branch is
    what the structure must constrain.)"""
    model = _model()
    torch.manual_seed(0)
    s = torch.rand(4, STATE_DIM)
    a = torch.rand(4, ACTION_DIM)
    kpi_rows = torch.arange(KPI_START, STATE_DIM)
    # KPI child 0 is state row kpi_start (=1); make source 0 (an NCP parent) ABSENT for it.
    structs = torch.ones(1, STATE_DIM, NS, dtype=torch.bool)
    structs[0, KPI_START, 0] = False
    struct_k = model._normalize_structures(structs, kpi_rows)  # forces self/action, keeps 0 off
    assert not bool(struct_k[0, 0, 0]), "source 0 must remain absent for KPI child 0"

    mu_before, _ = model._graph_means(s, a, kpi_rows, struct_k)  # (1, k, batch, 1)
    s_changed = s.clone()
    s_changed[:, 0] = s_changed[:, 0] + 100.0  # perturb the ABSENT parent's input
    mu_after, _ = model._graph_means(s_changed, a, kpi_rows, struct_k)

    # KPI child 0's graph mean must be EXACTLY invariant to the absent parent's input.
    torch.testing.assert_close(mu_before[:, 0], mu_after[:, 0], rtol=0.0, atol=0.0)
    # A present parent (source 0 IS a parent of KPI child 1) DOES change that child.
    assert not torch.allclose(mu_before[:, 1], mu_after[:, 1]), "present parent must matter"


def test_absent_parent_std_invariance_with_known_logstd_weight():
    """Review v2 MINOR #4: with a NONZERO log-std weight on an absent slot, changing that
    absent parent's raw feature leaves the predicted STD exactly unchanged (gating applies to
    the variance head too, not just the mean)."""
    torch.manual_seed(0)
    sp = StatePredictor(STATE_DIM, ACTION_DIM, FEATURE_DIM, [])
    lin = cast(torch.nn.Linear, sp.structure_predictor.net[-1])
    with torch.no_grad():
        lin.weight.zero_()
        lin.bias.zero_()
        lin.weight[1, 1 * (FEATURE_DIM + 1)] = 0.5  # log_std depends on slot 1's feature

    def std_of(v1, presence):
        feats = _feats_first_channel([2.0, v1, 0.0, 0.0])
        _, std = sp.head(feats, torch.tensor([presence], dtype=torch.float32))
        return std

    # slot 1 ABSENT -> its feature must not move std.
    torch.testing.assert_close(
        std_of(3.0, [1, 0, 0, 0]), std_of(-500.0, [1, 0, 0, 0]), rtol=0.0, atol=0.0
    )
    # sanity: slot 1 PRESENT -> its feature DOES move std.
    assert not torch.allclose(std_of(3.0, [1, 1, 0, 0]), std_of(-3.0, [1, 1, 0, 0]))


def test_residual_probe_does_not_perturb_sampler_sequence():
    """Review v2 BLOCKER A: the eval probe draws a structure from the sampler's PRIVATE numpy
    Generator; wrapped in preserve_probe_rng it must leave that generator's sequence intact, so
    planners see identical structures (hence identical utilities) with vs without the probe."""
    from cdd_oran.analysis.graph_posterior import GraphPosterior
    from cdd_oran.experiments.evaluate import preserve_probe_rng
    from cdd_oran.models import PosteriorStructureSampler

    freq = np.array([[0.0, 0.4, 0.6], [0.7, 0.0, 0.2], [0.3, 0.8, 0.0]])
    sampler = PosteriorStructureSampler(GraphPosterior.from_frequencies(freq), seed=7)
    model = _model(sampler=sampler)
    s = torch.rand(5, STATE_DIM)
    a = torch.rand(5, ACTION_DIM)

    snapshot = sampler.rng.bit_generator.state
    base_next = sampler.sample_structures(2, device=torch.device("cpu")).clone()

    sampler.rng.bit_generator.state = snapshot
    with preserve_probe_rng(model):
        model.residual_diagnostics(s, a)  # draws from the sampler, advancing it
    probe_next = sampler.sample_structures(2, device=torch.device("cpu")).clone()
    assert torch.equal(base_next, probe_next), "probe must not shift the sampler sequence"

    # Negative control: WITHOUT the wrapper the probe advances the sampler and shifts it.
    sampler.rng.bit_generator.state = snapshot
    model.residual_diagnostics(s, a)
    shifted = sampler.sample_structures(2, device=torch.device("cpu"))
    assert not torch.equal(base_next, shifted)


def _model(sampler=None):
    torch.manual_seed(0)
    return CDL(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        kpi_start=KPI_START,
        feature_fc_dims=[FEATURE_DIM],
        generative_fc_dims=[4],
        lr=1e-3,
        cmi_threshold=0.2,
        eval_tau=0.99,
        grad_clip=10.0,
        device="cpu",
        node_names=["p0", "k0", "k1"],
        eval_steps=2,
        residual_hidden=[4],
        sampler=sampler,
    )


class _FixedSampler:
    """Return a fixed set of structures, exercising the sampler contract shape (m,fd,ns)."""

    def __init__(self, structures):
        self.structures = structures

    def sample_structures(self, n_members, *, device):
        return self.structures[:n_members].to(device)


def _structs(m):
    torch.manual_seed(1)
    return (torch.rand(m, STATE_DIM, NS) > 0.5)


def test_predict_shapes_single_and_multi_member():
    model = _model()
    s = torch.rand(5, STATE_DIM)
    a = torch.rand(5, ACTION_DIM)
    k = STATE_DIM - KPI_START

    dist1 = model.predict_next_state(s, a, structures=_structs(1))
    assert dist1.mean.shape == (5, k), "m=1 must squeeze to planner-compatible (batch, k)"

    dist2 = model.predict_next_state(s, a, structures=_structs(2))
    assert dist2.mean.shape == (2, 5, k), "m=2 must expose the member axis (m, batch, k)"


def test_predict_accepts_full_and_sliced_structures():
    model = _model()
    s = torch.rand(4, STATE_DIM)
    a = torch.rand(4, ACTION_DIM)
    k = STATE_DIM - KPI_START
    full = _structs(1)  # (1, fd, ns)
    sliced = full[:, KPI_START:, :]  # (1, k, ns)
    m_full = model.predict_next_state(s, a, structures=full).mean
    m_sliced = model.predict_next_state(s, a, structures=sliced).mean
    torch.testing.assert_close(m_full, m_sliced)
    assert m_full.shape == (4, k)


def test_default_sampler_is_used_when_structures_omitted():
    model = _model(sampler=_FixedSampler(_structs(1)))
    s = torch.rand(3, STATE_DIM)
    a = torch.rand(3, ACTION_DIM)
    dist = model.predict_next_state(s, a)  # no structures -> injected sampler, m=1
    assert dist.mean.shape == (3, STATE_DIM - KPI_START)


def test_missing_sampler_fails_loudly():
    model = _model(sampler=None)
    s = torch.rand(2, STATE_DIM)
    a = torch.rand(2, ACTION_DIM)
    try:
        model.predict_next_state(s, a)  # no structures, no sampler
    except ValueError as error:
        assert "sampler" in str(error)
    else:
        raise AssertionError("new mode must fail loudly without a sampler or structures")


def test_residual_is_zero_at_init_and_bounded():
    model = _model()
    s = torch.rand(6, STATE_DIM)
    a = torch.rand(6, ACTION_DIM)
    delta = model.residual(s, a)
    assert torch.allclose(delta, torch.zeros_like(delta)), "zero-init residual must start at 0"
    assert torch.all(delta.abs() <= model.residual_bound + 1e-6), "residual must be tanh-bounded"

    # A zero residual leaves the total mean equal to the graph-only mean.
    structs = _structs(1)
    kpi_rows = torch.arange(KPI_START, STATE_DIM)
    struct_k = model._normalize_structures(structs, kpi_rows)
    mu_graph, _ = model._graph_means(s, a, kpi_rows, struct_k)
    mu_total = model.predict_next_state(s, a, structures=structs).mean
    torch.testing.assert_close(mu_total, mu_graph.squeeze(0).squeeze(-1).transpose(0, 1))


def test_residual_stays_bounded_after_a_nonzero_step():
    """After a training step the residual is no longer zero, but the tanh bound holds."""
    model = _model()
    s_batch = torch.rand(8, 2, STATE_DIM)
    a_batch = torch.rand(8, ACTION_DIM)
    loss = model.train_step(s_batch, a_batch, structures=_structs(1))
    assert torch.isfinite(loss) and loss.shape == ()
    s = torch.rand(6, STATE_DIM)
    a = torch.rand(6, ACTION_DIM)
    delta = model.residual(s, a)
    assert torch.all(delta.abs() <= model.residual_bound + 1e-6), "residual must remain bounded"


def test_residual_diagnostics_are_finite_fractions():
    model = _model()
    s = torch.rand(16, STATE_DIM)
    a = torch.rand(16, ACTION_DIM)
    diag = model.residual_diagnostics(s, a, structures=_structs(1))
    k = STATE_DIM - KPI_START
    assert diag["fraction_per_kpi"].shape == (k,)
    assert torch.all(torch.isfinite(diag["fraction_per_kpi"]))
    assert torch.all((diag["fraction_per_kpi"] >= 0.0) & (diag["fraction_per_kpi"] <= 1.0))
    assert torch.isfinite(diag["fraction_aggregate"])
    assert 0.0 <= float(diag["fraction_aggregate"]) <= 1.0
    # At init the residual is zero, so the contribution fraction is exactly zero.
    assert float(diag["fraction_aggregate"]) == 0.0
    assert diag["alert"] is False

    # Multi-member diagnostics report a max-member fraction too, never averaging first.
    multi = model.residual_diagnostics(s, a, structures=_structs(3))
    assert torch.isfinite(multi["fraction_max_member"])
    assert float(multi["fraction_max_member"]) >= float(multi["fraction_aggregate"]) - 1e-6


def test_residual_diagnostics_return_components_with_correct_algebra():
    """Review MINOR #8: residual_diagnostics returns mu_graph, delta, mu_total, and
    mu_total == mu_graph + delta (broadcast over members) with |delta| <= bound."""
    model = _model()
    # A training step makes the residual non-zero so the algebra is a real check.
    model.train_step(torch.rand(8, 2, STATE_DIM), torch.rand(8, ACTION_DIM), structures=_structs(1))
    s = torch.rand(5, STATE_DIM)
    a = torch.rand(5, ACTION_DIM)
    diag = model.residual_diagnostics(s, a, structures=_structs(2))

    mu_graph = diag["mu_graph"]  # (m, k, batch, 1)
    delta = diag["delta"]  # (batch, k)
    mu_total = diag["mu_total"]  # (m, k, batch, 1)
    m, k, batch, _ = mu_graph.shape
    assert delta.shape == (batch, k)
    assert torch.all(delta.abs() <= model.residual_bound + 1e-6), "delta must be tanh-bounded"

    delta_rows = delta.transpose(0, 1).unsqueeze(-1).unsqueeze(0)  # (1, k, batch, 1)
    torch.testing.assert_close(mu_total, mu_graph + delta_rows)
    # The reported abs residual matches delta.
    torch.testing.assert_close(diag["abs_mean_residual"], delta.abs().mean(dim=0))


def test_train_step_runs_and_updates_residual_params():
    model = _model()
    s_batch = torch.rand(8, 2, STATE_DIM)
    a_batch = torch.rand(8, ACTION_DIM)
    before = [p.detach().clone() for p in model.residual.parameters()]
    loss = model.train_step(s_batch, a_batch, structures=_structs(2))
    assert torch.isfinite(loss)
    after = list(model.residual.parameters())
    assert any(not torch.allclose(b, a) for b, a in zip(before, after, strict=True)), (
        "the residual must receive gradients and update"
    )


# --------------------------------------------------------------------------- #
# Enumeration-artifact validation (review v2 BLOCKER B).
# --------------------------------------------------------------------------- #
def _env_i_cfg(enum_path=None, posterior_path=None):
    cfg = load_config("env_i_cdl.yaml")
    model = cfg.model
    if enum_path is not None:
        model = replace(model, enumeration_graph=str(enum_path))
    if posterior_path is not None:
        model = replace(model, posterior_artifact=str(posterior_path))
    return replace(cfg, device="cpu", model=model)


def _write_enum(path, state_graph, env, *, environment="EnvironmentI",
                node_names_override=None, model_kind="cdl", dynamics_mode="hard_mask"):
    fd = env.get_state_dim()
    full = np.zeros((fd, fd + 1), dtype=bool)
    full[:, :fd] = np.asarray(state_graph, dtype=bool)
    payload = {
        "graph": full.tolist(),
        "shape": [fd, fd + 1],
        "source_run": "runs/source",
        "environment": environment,
        "node_names": node_names_override if node_names_override is not None else node_names(env),
        "state_edge_count": int(np.asarray(state_graph, dtype=bool).sum()),
        "model_kind": model_kind,
        "dynamics_mode": dynamics_mode,
    }
    Path(path).write_text(json.dumps(payload))
    return path


def test_valid_enumeration_artifact_accepted_and_enumerates_conflicts(tmp_path):
    env = get_env(_env_i_cfg())
    enum = _write_enum(tmp_path / "enum.json", env.true_adj_matrix, env)
    model = get_model(_env_i_cfg(enum_path=enum), env)  # validates against the env
    graph = model.get_binary_graph()[:, :-1].cpu().numpy()
    edges = detect_conflict_edges(graph, env)
    assert len(edges) > 0, "a valid enumeration graph must yield at least one conflict"


def test_zero_edge_enumeration_artifact_rejected(tmp_path):
    env = get_env(_env_i_cfg())
    fd = env.get_state_dim()
    enum = _write_enum(tmp_path / "zero.json", np.zeros((fd, fd), dtype=bool), env)
    with pytest.raises(ValueError, match="(?i)zero state edges"):
        load_enumeration_graph(enum)
    with pytest.raises(ValueError):
        get_model(_env_i_cfg(enum_path=enum), env)


def test_wrong_node_order_enumeration_artifact_rejected(tmp_path):
    env = get_env(_env_i_cfg())
    enum = _write_enum(
        tmp_path / "wrong.json", env.true_adj_matrix, env,
        node_names_override=list(reversed(node_names(env))),
    )
    with pytest.raises(ValueError, match="node-name"):
        get_model(_env_i_cfg(enum_path=enum), env)


def test_wrong_env_enumeration_artifact_rejected(tmp_path):
    env = get_env(_env_i_cfg())
    bad_env = _write_enum(tmp_path / "badenv.json", env.true_adj_matrix, env, environment="EnvironmentII")
    with pytest.raises(ValueError, match="environment"):
        get_model(_env_i_cfg(enum_path=bad_env), env)
    bad_kind = _write_enum(
        tmp_path / "badkind.json", env.true_adj_matrix, env, model_kind="mlp"
    )
    with pytest.raises(ValueError, match="CDL source"):
        get_model(_env_i_cfg(enum_path=bad_kind), env)


# --------------------------------------------------------------------------- #
# Self-contained run artifacts: stage + hash-verify (review v2 MAJOR).
# --------------------------------------------------------------------------- #
def _calibrated_posterior_file(path, env):
    from cdd_oran.analysis.graph_posterior import GraphPosterior

    fd = env.get_state_dim()
    probs = np.full((fd, fd), 0.2)
    np.fill_diagonal(probs, 0.0)
    gt = np.asarray(env.true_adj_matrix, dtype=float)
    post = GraphPosterior.from_frequencies(
        probs,
        node_names=node_names(env),
        meta={"environment": "EnvironmentI"},
    )
    calibrated, _ = post.calibrate(gt, evaluation_labels=gt)
    calibrated.save(path)
    return path


def test_stage_and_verify_run_artifacts_roundtrip_and_tamper(tmp_path):
    env = get_env(_env_i_cfg())
    posterior = _calibrated_posterior_file(tmp_path / "post.json", env)
    enum = _write_enum(tmp_path / "enum.json", env.true_adj_matrix, env)
    cfg = _env_i_cfg(enum_path=enum, posterior_path=posterior)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest = stage_run_artifacts(cfg, run_dir, env=env)
    assert (run_dir / "artifacts" / "posterior.json").exists()
    assert (run_dir / "artifacts" / "enumeration_graph.json").exists()
    assert manifest["posterior"]["calibrated"] is True

    verified, resolved = verify_run_artifacts(run_dir, env=env, environment=cfg.environment)
    assert verified == manifest
    assert Path(resolved["posterior"]).exists()

    # Tampering with a staged file must be detected by the hash check.
    (run_dir / "artifacts" / "enumeration_graph.json").write_text('{"graph": []}')
    with pytest.raises(ValueError, match="hash changed"):
        verify_run_artifacts(run_dir)


def test_wrong_layout_posterior_rejected_before_staging(tmp_path):
    env = get_env(_env_i_cfg())
    posterior = _calibrated_posterior_file(tmp_path / "post.json", env)
    payload = json.loads(Path(posterior).read_text())
    payload["node_names"] = list(reversed(payload["node_names"]))
    wrong = tmp_path / "wrong_layout.json"
    wrong.write_text(json.dumps(payload))
    enum = _write_enum(tmp_path / "enum.json", env.true_adj_matrix, env)
    cfg = _env_i_cfg(enum_path=enum, posterior_path=wrong)

    with pytest.raises(ValueError, match="posterior node-name"):
        from cdd_oran.models import make_structure_sampler

        make_structure_sampler(cfg, seed=0, env=env)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(ValueError, match="posterior node-name"):
        stage_run_artifacts(cfg, run_dir, env=env)
    assert not (run_dir / "artifacts").exists(), "invalid posterior must fail before staging"


def test_resume_checkpoint_manifest_equality_is_required(tmp_path):
    manifest = {
        "posterior": {"filename": "posterior.json", "sha256": "a" * 64},
        "enumeration_graph": {"filename": "enumeration_graph.json", "sha256": "b" * 64},
    }
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"artifact_manifest": manifest}, checkpoint)
    assert verify_checkpoint_manifest(checkpoint, manifest) == manifest

    changed = {**manifest, "posterior": {**manifest["posterior"], "sha256": "c" * 64}}
    with pytest.raises(ValueError, match="exactly match"):
        verify_checkpoint_manifest(checkpoint, changed)

    torch.save({"artifact_manifest": None}, checkpoint)
    with pytest.raises(ValueError, match="exactly match"):
        verify_checkpoint_manifest(checkpoint, manifest)


def test_verify_run_artifacts_missing_manifest_fails_loudly(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="manifest missing"):
        verify_run_artifacts(run_dir)
