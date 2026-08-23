"""CPU tests for the attribution ablation infra (knobs + prediction harness), post-review.

No GPU, no training job. Covers the review fixes:
  * oracle uses env.true_adj_matrix and REJECTS a caller-supplied sampler (BLOCKER #3);
  * residual disabled -> representation-level identity mu_total IS mu_graph (MAJOR #7),
    residual magnitude ratio exactly 0, no learnable residual path;
  * ensemble-mean MSE + Normal-mixture NLL, single- AND two-member closed forms (BLOCKER #1,
    MINOR #9);
  * paired residual delta dMSE/dNLL, 0 for structure_only/oracle/dense (BLOCKER #2);
  * run_attribution/load_run integration: identical transition + structure inputs, and rejection
    of mismatched posterior identity, incompatible ranges, and duplicate/incomplete sets
    (BLOCKER #4, MAJOR #6, MINOR #10);
  * runtime validation of both config knobs (BLOCKER #8).
"""

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.distributions import Normal

from cdd_oran.analysis.attribution import (
    VARIANT_ORDER,
    draw_attribution_structures,
    held_out_transitions,
    load_run,
    mixture_metrics,
    run_attribution,
    score_variant,
    write_report,
)
from cdd_oran.analysis.graph_posterior import GraphPosterior
from cdd_oran.config import load_config
from cdd_oran.envs import get_env
from cdd_oran.models import (
    PosteriorStructureSampler,
    build_oracle_enumeration_graph,
    get_model,
    make_structure_sampler,
    node_names,
    stage_run_artifacts,
)
from cdd_oran.models.cdl import CDL
from cdd_oran.utils.runs import write_metadata

STATE_DIM, KPI_START, ACTION_DIM = 3, 1, 3
NS = STATE_DIM + 1


def _cfg(**model_over):
    cfg = load_config("env_i_cdl.yaml")
    defaults = {"posterior_artifact": None, "enumeration_graph": None, "predict_members": 1}
    defaults.update(model_over)
    return replace(cfg, device="cpu", model=replace(cfg.model, **defaults))


def _tiny_cdl(residual_enabled, sampler=None):
    torch.manual_seed(0)
    return CDL(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        kpi_start=KPI_START,
        feature_fc_dims=[2],
        generative_fc_dims=[4],
        lr=1e-3,
        cmi_threshold=0.2,
        eval_tau=0.99,
        grad_clip=10.0,
        device="cpu",
        node_names=["p0", "k0", "k1"],
        eval_steps=2,
        residual_hidden=[4],
        residual_enabled=residual_enabled,
        sampler=sampler,
    )


def _structs(m=1):
    torch.manual_seed(1)
    return torch.rand(m, STATE_DIM, NS) > 0.5


def _disc_sampler(env, seed=0, fill=0.3):
    fd = env.get_state_dim()
    probs = np.full((fd, fd), fill)
    np.fill_diagonal(probs, 0.0)
    return PosteriorStructureSampler(
        GraphPosterior.from_frequencies(probs, node_names=node_names(env)), seed=seed
    )


# --------------------------------------------------------------------------- #
# (a/BLOCKER 3) oracle uses the true adjacency; rejects a supplied sampler.
# --------------------------------------------------------------------------- #
def test_oracle_uses_true_adjacency_without_artifacts():
    cfg = _cfg(structure_source="oracle")
    env = get_env(cfg)
    fd = env.get_state_dim()
    true_adj = np.asarray(env.true_adj_matrix, dtype=bool)

    assert cfg.model.posterior_artifact is None and cfg.model.enumeration_graph is None
    model = get_model(cfg, env, sampler=None)

    enum_state = model.enumeration_graph[:, :fd].cpu().numpy().astype(bool)
    assert np.array_equal(enum_state, true_adj)
    built = build_oracle_enumeration_graph(env)[:, :fd].cpu().numpy().astype(bool)
    assert np.array_equal(built, true_adj)

    draw = model.sampler.sample_structures(4, device=torch.device("cpu"))
    assert draw.shape == (4, fd, fd + 1)
    for member in range(4):
        assert np.array_equal(draw[member, :, :fd].cpu().numpy().astype(bool), true_adj)
        assert bool(draw[member, :, -1].all())


def test_oracle_rejects_supplied_discovered_sampler():
    """BLOCKER #3: a discovered (dense) sampler passed in oracle mode must NOT leak into
    prediction -- get_model rebuilds the oracle sampler and still draws env.true_adj_matrix."""
    cfg = _cfg(structure_source="oracle")
    env = get_env(cfg)
    fd = env.get_state_dim()
    true_adj = np.asarray(env.true_adj_matrix, dtype=bool)

    leak = _disc_sampler(env, seed=0, fill=0.9)  # dense discovered posterior
    model = get_model(cfg, env, sampler=leak)
    assert model.sampler is not leak, "oracle must ignore the caller-supplied sampler"

    draw = model.sampler.sample_structures(3, device=torch.device("cpu"))
    for member in range(3):
        assert np.array_equal(draw[member, :, :fd].cpu().numpy().astype(bool), true_adj)


# --------------------------------------------------------------------------- #
# (b/MAJOR 7) residual disabled -> representation-level identity + zero.
# --------------------------------------------------------------------------- #
def test_residual_disabled_is_representation_level_identity():
    sampler = _disc_sampler(get_env(_cfg()))
    model = _tiny_cdl(residual_enabled=False, sampler=sampler)
    assert model.residual is None, "disabled residual must be absent (no learnable path)"

    opt_params = {id(p) for group in model.opt.param_groups for p in group["params"]}
    assert opt_params == {id(p) for p in model.models.parameters()}

    model.train_step(torch.rand(8, 2, STATE_DIM), torch.rand(8, ACTION_DIM), structures=_structs(1))

    s, a = torch.rand(6, STATE_DIM), torch.rand(6, ACTION_DIM)
    diag = model.residual_diagnostics(s, a, structures=_structs(2))
    # Representation-level identity: same tensor object, not just numeric equality.
    assert diag["mu_total"].data_ptr() == diag["mu_graph"].data_ptr()
    assert float(diag["residual_magnitude_ratio"]) == 0.0
    assert float(diag["fraction_aggregate"]) == 0.0  # legacy alias
    assert float(diag["delta"].abs().sum()) == 0.0

    comps = model.predict_components(s, a, structures=_structs(1))
    assert comps["mu_total"].data_ptr() == comps["mu_graph"].data_ptr()


def test_residual_enabled_default_has_residual_module():
    model = _tiny_cdl(residual_enabled=True, sampler=_disc_sampler(get_env(_cfg())))
    assert model.residual is not None
    residual_ids = {id(p) for p in model.residual.parameters()}
    opt_ids = {id(p) for group in model.opt.param_groups for p in group["params"]}
    assert residual_ids <= opt_ids
    # With a residual, the two component tensors are distinct objects.
    s, a = torch.rand(5, STATE_DIM), torch.rand(5, ACTION_DIM)
    comps = model.predict_components(s, a, structures=_structs(2))
    assert comps["mu_total"].data_ptr() != comps["mu_graph"].data_ptr()


# --------------------------------------------------------------------------- #
# (BLOCKER 1 / MINOR 9) ensemble-mean MSE + mixture NLL closed forms.
# --------------------------------------------------------------------------- #
def test_single_member_mixture_collapses_to_normal_nll():
    target = torch.tensor([[1.0, 2.0]])
    mu = torch.zeros(1, 1, 2)
    std = torch.ones(1, 1, 2)
    metrics = mixture_metrics(mu, std, target)
    assert abs(float(metrics["mse"]) - 2.5) < 1e-6
    expected_nll = 0.5 * math.log(2 * math.pi) + 0.5 * 2.5
    assert abs(float(metrics["nll"]) - expected_nll) < 1e-6
    np.testing.assert_allclose([float(x) for x in metrics["mse_per_kpi"]], [1.0, 4.0], atol=1e-6)


def test_two_member_ensemble_mean_mse_and_mixture_nll():
    """MINOR #9: distinct means AND variances catch the ensemble-mean MSE (not mean of squared
    errors) and the mixture NLL (not mean of component NLLs)."""
    mu = torch.tensor([[[0.0]], [[2.0]]])  # (2, 1, 1)
    std = torch.tensor([[[1.0]], [[2.0]]])
    target = torch.tensor([[1.0]])
    metrics = mixture_metrics(mu, std, target)

    mu_bar = 1.0  # (0 + 2)/2
    assert abs(float(metrics["mse"]) - (mu_bar - 1.0) ** 2) < 1e-6  # == 0, NOT mean(1,1)=1

    n0 = math.exp(float(Normal(torch.tensor(0.0), torch.tensor(1.0)).log_prob(torch.tensor(1.0))))
    n1 = math.exp(float(Normal(torch.tensor(2.0), torch.tensor(2.0)).log_prob(torch.tensor(1.0))))
    expected_nll = -math.log(0.5 * (n0 + n1))
    assert abs(float(metrics["nll"]) - expected_nll) < 1e-6
    # Mean-of-component-NLLs would differ; confirm we are NOT computing that.
    mean_component_nll = 0.5 * (-math.log(n0) - math.log(n1))
    assert abs(expected_nll - mean_component_nll) > 1e-3


# --------------------------------------------------------------------------- #
# (BLOCKER 2) paired residual delta in a scored row; 0 for no-residual variants.
# --------------------------------------------------------------------------- #
def test_score_variant_reports_paired_residual_delta():
    cfg = _cfg(predict_members=2)
    env = get_env(cfg)
    full = get_model(_cfg(structure_source="discovered", residual_enabled=True, predict_members=2),
                     env, sampler=_disc_sampler(env))
    structure_only = get_model(
        _cfg(structure_source="discovered", residual_enabled=False, predict_members=2),
        env, sampler=_disc_sampler(env),
    )
    transitions = held_out_transitions(env, n_transitions=16, seed=0)

    full_row = score_variant("full", full, transitions, attribution_seed=3)
    so_row = score_variant("structure_only", structure_only, transitions, attribution_seed=3)

    assert full_row["n_members"] == 2
    assert "dmse_residual" in full_row and "dnll_residual" in full_row
    assert np.isfinite(full_row["dmse_residual"]) and np.isfinite(full_row["dnll_residual"])
    # residual OFF -> paired delta and magnitude ratio are exactly 0.
    assert so_row["dmse_residual"] == 0.0 and so_row["dnll_residual"] == 0.0
    assert so_row["residual_magnitude_ratio"] == 0.0


def test_comparison_table_has_all_four_variant_rows(tmp_path):
    cfg = _cfg()
    env = get_env(cfg)
    full = get_model(_cfg(structure_source="discovered", residual_enabled=True), env,
                     sampler=_disc_sampler(env))
    structure_only = get_model(_cfg(structure_source="discovered", residual_enabled=False), env,
                               sampler=_disc_sampler(env))
    oracle = get_model(_cfg(structure_source="oracle", residual_enabled=False), env, sampler=None)
    dense = get_model(replace(load_config("env_i_mlp.yaml"), device="cpu"), env)

    transitions = held_out_transitions(env, n_transitions=16, seed=0)
    rows = [
        score_variant("full", full, transitions),
        score_variant("structure_only", structure_only, transitions),
        score_variant("oracle", oracle, transitions),
        score_variant("dense", dense, transitions),
    ]
    out_md = tmp_path / "attribution_table.md"
    out_json = tmp_path / "attribution_table.json"
    markdown, payload = write_report(rows, out_md=out_md, out_json=out_json)

    variants = [row["variant"] for row in payload["variants"]]
    assert variants == list(VARIANT_ORDER)
    for row in payload["variants"]:
        assert np.isfinite(row["mse"]) and row["mse"] >= 0.0
        assert np.isfinite(row["nll"])
        assert row["param_count"] > 0
    by_name = {row["variant"]: row for row in payload["variants"]}
    for name in ("structure_only", "oracle", "dense"):
        assert by_name[name]["dmse_residual"] == 0.0
        assert by_name[name]["dnll_residual"] == 0.0
        assert by_name[name]["residual_magnitude_ratio"] == 0.0
    assert out_md.exists() and out_json.exists()
    assert "dMSE_residual" in markdown


# --------------------------------------------------------------------------- #
# (BLOCKER 8) runtime validation of both knobs.
# --------------------------------------------------------------------------- #
def test_config_rejects_bad_structure_source():
    with pytest.raises(ValueError, match="structure_source"):
        load_config("env_i_cdl.yaml", ["model.structure_source=bogus"])


def test_config_rejects_nonbool_residual_enabled():
    # A quoted string must NOT silently become True.
    with pytest.raises(ValueError, match="residual_enabled"):
        load_config("env_i_cdl.yaml", ['model.residual_enabled="false"'])


def test_set_residual_enabled_false_is_python_false():
    cfg = load_config("env_i_cdl.yaml", ["model.residual_enabled=false"])
    assert cfg.model.residual_enabled is False


# --------------------------------------------------------------------------- #
# (BLOCKER 4 / MAJOR 6 / MINOR 10) run_attribution + load_run integration.
# --------------------------------------------------------------------------- #
def _write_enum(path, env):
    fd = env.get_state_dim()
    full = np.zeros((fd, fd + 1), dtype=bool)
    full[:, :fd] = np.asarray(env.true_adj_matrix, dtype=bool)
    payload = {
        "graph": full.tolist(),
        "shape": [fd, fd + 1],
        "source_run": "runs/source",
        "environment": "EnvironmentI",
        "node_names": node_names(env),
        "state_edge_count": int(np.asarray(env.true_adj_matrix, dtype=bool).sum()),
        "model_kind": "cdl",
        "dynamics_mode": "structure_conditioned",
    }
    Path(path).write_text(json.dumps(payload))
    return path


def _write_posterior(path, env, fill=0.2):
    fd = env.get_state_dim()
    probs = np.full((fd, fd), fill)
    np.fill_diagonal(probs, 0.0)
    gt = np.asarray(env.true_adj_matrix, dtype=float)
    post = GraphPosterior.from_frequencies(
        probs, node_names=node_names(env), meta={"environment": "EnvironmentI"}
    )
    calibrated, _ = post.calibrate(gt, evaluation_labels=gt)
    calibrated.save(path)
    return path


def _make_cdl_run(run_dir, env, base_cfg, *, residual_enabled=True,
                  structure_source="discovered", post_fill=0.2, predict_members=2):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    model_over = replace(
        base_cfg.model,
        residual_enabled=residual_enabled,
        structure_source=structure_source,
        predict_members=predict_members,
    )
    if structure_source == "discovered":
        post = _write_posterior(run_dir / "src_post.json", env, fill=post_fill)
        enum = _write_enum(run_dir / "src_enum.json", env)
        model_over = replace(model_over, posterior_artifact=str(post), enumeration_graph=str(enum))
    else:
        model_over = replace(model_over, posterior_artifact=None, enumeration_graph=None)
    cfg = replace(base_cfg, device="cpu", model=model_over)
    write_metadata(run_dir, cfg)

    if structure_source == "discovered":
        manifest = stage_run_artifacts(cfg, run_dir, env=env)
        staged = replace(cfg, model=replace(
            cfg.model,
            posterior_artifact=str(run_dir / "artifacts" / "posterior.json"),
            enumeration_graph=str(run_dir / "artifacts" / "enumeration_graph.json"),
        ))
        sampler = make_structure_sampler(staged, seed=staged.seed, env=env)
        model = get_model(staged, env, sampler=sampler)
        model.artifact_manifest = manifest
    else:
        model = get_model(cfg, env, sampler=None)
    model.save_model(run_dir / "checkpoint.pt")
    return run_dir


def _make_mlp_run(run_dir, env, mlp_cfg):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = replace(mlp_cfg, device="cpu")
    write_metadata(run_dir, cfg)
    model = get_model(cfg, env)
    model.save_model(run_dir / "checkpoint.pt")
    return run_dir


def test_run_attribution_integration_and_shared_inputs(tmp_path):
    base = replace(load_config("env_i_cdl.yaml"), device="cpu")
    env = get_env(base)
    mlp = replace(load_config("env_i_mlp.yaml"), device="cpu")

    full = _make_cdl_run(tmp_path / "full", env, base, residual_enabled=True)
    so = _make_cdl_run(tmp_path / "so", env, base, residual_enabled=False)  # same posterior bytes
    oracle = _make_cdl_run(tmp_path / "oracle", env, base, structure_source="oracle")
    dense = _make_mlp_run(tmp_path / "dense", env, mlp)

    rows, markdown = run_attribution(
        [full, so, oracle, dense],
        device="cpu",
        n_transitions=16,
        seed=0,
        attribution_seed=7,
        require_full_set=True,
    )
    by = {row["variant"]: row for row in rows}
    assert set(by) == set(VARIANT_ORDER)
    for name in ("structure_only", "oracle", "dense"):
        assert by[name]["dmse_residual"] == 0.0
        assert by[name]["dnll_residual"] == 0.0
        assert by[name]["residual_magnitude_ratio"] == 0.0
    assert by["full"]["n_members"] == 2 and by["structure_only"]["n_members"] == 2
    assert by["dense"]["n_members"] == 1
    assert np.isfinite(by["full"]["dmse_residual"])
    assert "dMSE_residual" in markdown

    # Discovered pair shares posterior identity AND draws IDENTICAL structures at one seed.
    loaded_full = load_run(full)
    loaded_so = load_run(so)
    assert loaded_full.posterior_sha == loaded_so.posterior_sha
    struct_full = draw_attribution_structures(loaded_full.model, 7)
    struct_so = draw_attribution_structures(loaded_so.model, 7)
    assert torch.equal(struct_full, struct_so)


def test_run_attribution_rejects_mismatched_posterior(tmp_path):
    base = replace(load_config("env_i_cdl.yaml"), device="cpu")
    env = get_env(base)
    full = _make_cdl_run(tmp_path / "full", env, base, residual_enabled=True, post_fill=0.2)
    so = _make_cdl_run(tmp_path / "so", env, base, residual_enabled=False, post_fill=0.4)
    with pytest.raises(ValueError, match="posterior identity"):
        run_attribution([full, so], device="cpu", n_transitions=8, seed=0)


def test_run_attribution_rejects_incompatible_ranges(tmp_path):
    base = replace(load_config("env_i_cdl.yaml"), device="cpu")
    env = get_env(base)
    full = _make_cdl_run(tmp_path / "full", env, base, residual_enabled=True)
    base_train = replace(base, evaluation_param_ranges="train")
    so = _make_cdl_run(tmp_path / "so", env, base_train, residual_enabled=False)
    with pytest.raises(ValueError, match="incompatible"):
        run_attribution([full, so], device="cpu", n_transitions=8, seed=0)


def test_run_attribution_rejects_mismatched_predict_members(tmp_path):
    # Same posterior bytes so the posterior-identity check passes; different member counts must be
    # rejected so the discovered pair cannot draw different structure tensors (review BLOCKER #4).
    base = replace(load_config("env_i_cdl.yaml"), device="cpu")
    env = get_env(base)
    full = _make_cdl_run(tmp_path / "full", env, base, residual_enabled=True, predict_members=2)
    so = _make_cdl_run(tmp_path / "so", env, base, residual_enabled=False, predict_members=4)
    with pytest.raises(ValueError, match="predict_members"):
        run_attribution([full, so], device="cpu", n_transitions=8, seed=0)


def test_run_attribution_rejects_mismatched_train_ranges(tmp_path):
    # Different training param_ranges confounds the ablation and must be rejected (review MAJOR #6).
    base = replace(load_config("env_i_cdl.yaml"), device="cpu")
    env = get_env(base)
    full = _make_cdl_run(tmp_path / "full", env, base, residual_enabled=True)
    base_ood = replace(base, param_ranges="ood")
    so = _make_cdl_run(tmp_path / "so", env, base_ood, residual_enabled=False)
    with pytest.raises(ValueError, match="incompatible"):
        run_attribution([full, so], device="cpu", n_transitions=8, seed=0)


def test_run_attribution_rejects_duplicate_and_incomplete_sets(tmp_path):
    base = replace(load_config("env_i_cdl.yaml"), device="cpu")
    env = get_env(base)
    full = _make_cdl_run(tmp_path / "full", env, base, residual_enabled=True)
    with pytest.raises(ValueError, match="duplicate"):
        run_attribution([full, full], device="cpu", n_transitions=8, seed=0)
    with pytest.raises(ValueError, match="incomplete"):
        run_attribution([full], device="cpu", n_transitions=8, seed=0, require_full_set=True)
