"""CPU tests for the attribution ablation infra (knobs + prediction harness).

No GPU, no training job. Three checks the brief names:
  (a) structure_source=oracle uses the env's TRUE adjacency (enum graph == oracle edges) and
      needs NO discovery artifacts;
  (b) residual disabled -> residual fraction exactly 0 and mu_total == mu_graph (no learnable
      residual path, not merely bound=0);
  (c) the prediction harness computes MSE/NLL on a known tiny case and produces the comparison
      table with all four variant rows.
"""

from dataclasses import replace

import numpy as np
import torch

from cdd_oran.analysis.attribution import (
    VARIANT_ORDER,
    evaluate_variant,
    held_out_transitions,
    prediction_metrics,
    write_report,
)
from cdd_oran.analysis.graph_posterior import GraphPosterior
from cdd_oran.config import load_config
from cdd_oran.envs import get_env
from cdd_oran.models import (
    PosteriorStructureSampler,
    build_oracle_enumeration_graph,
    get_model,
    node_names,
)
from cdd_oran.models.cdl import CDL

STATE_DIM, KPI_START, ACTION_DIM = 3, 1, 3
NS = STATE_DIM + 1


def _cfg(**model_over):
    cfg = load_config("env_i_cdl.yaml")
    model = replace(
        cfg.model,
        posterior_artifact=None,
        enumeration_graph=None,
        predict_members=1,
        **model_over,
    )
    return replace(cfg, device="cpu", model=model)


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


def _disc_sampler(env, seed=0):
    fd = env.get_state_dim()
    probs = np.full((fd, fd), 0.3)
    np.fill_diagonal(probs, 0.0)
    return PosteriorStructureSampler(
        GraphPosterior.from_frequencies(probs, node_names=node_names(env)), seed=seed
    )


# --------------------------------------------------------------------------- #
# (a) oracle uses the true adjacency, no artifacts needed.
# --------------------------------------------------------------------------- #
def test_oracle_uses_true_adjacency_without_artifacts():
    cfg = _cfg(structure_source="oracle")
    env = get_env(cfg)
    fd = env.get_state_dim()
    true_adj = np.asarray(env.true_adj_matrix, dtype=bool)

    # No posterior/enumeration artifacts are configured, yet the oracle model builds.
    assert cfg.model.posterior_artifact is None and cfg.model.enumeration_graph is None
    model = get_model(cfg, env, sampler=None)

    # The frozen enumeration graph's state block equals the env's true adjacency.
    assert model.enumeration_graph is not None
    enum_state = model.enumeration_graph[:, :fd].cpu().numpy().astype(bool)
    assert np.array_equal(enum_state, true_adj)
    # ... and matches the standalone oracle builder exactly.
    built = build_oracle_enumeration_graph(env)[:, :fd].cpu().numpy().astype(bool)
    assert np.array_equal(built, true_adj)

    # The oracle sampler draws exactly the true adjacency (degenerate one-hot posterior).
    assert model.sampler is not None
    draw = model.sampler.sample_structures(4, device=torch.device("cpu"))
    assert draw.shape == (4, fd, fd + 1)
    for member in range(4):
        assert np.array_equal(draw[member, :, :fd].cpu().numpy().astype(bool), true_adj)
        assert bool(draw[member, :, -1].all())  # action source always on


# --------------------------------------------------------------------------- #
# (b) residual disabled -> fraction exactly 0 and mu_total == mu_graph.
# --------------------------------------------------------------------------- #
def test_residual_disabled_contributes_exactly_zero():
    sampler = _disc_sampler(get_env(_cfg()))
    model = _tiny_cdl(residual_enabled=False, sampler=sampler)
    assert model.residual is None, "disabled residual must be absent (no learnable path)"

    # Only the structure predictors are in the optimizer -- no residual parameters.
    opt_params = {id(p) for group in model.opt.param_groups for p in group["params"]}
    assert opt_params == {id(p) for p in model.models.parameters()}

    # A training step must not create a residual contribution.
    model.train_step(torch.rand(8, 2, STATE_DIM), torch.rand(8, ACTION_DIM), structures=_structs(1))

    s, a = torch.rand(6, STATE_DIM), torch.rand(6, ACTION_DIM)
    diag = model.residual_diagnostics(s, a, structures=_structs(2))
    assert float(diag["fraction_aggregate"]) == 0.0
    assert float(diag["delta"].abs().sum()) == 0.0
    torch.testing.assert_close(diag["mu_total"], diag["mu_graph"], rtol=0.0, atol=0.0)

    # Prediction mean equals the structure-conditioned graph mean (residual adds exact zero).
    kpi_rows = torch.arange(KPI_START, STATE_DIM)
    struct_k = model._normalize_structures(_structs(1), kpi_rows)
    mu_graph, _ = model._graph_means(s, a, kpi_rows, struct_k)
    mu_total = model.predict_next_state(s, a, structures=_structs(1)).mean
    torch.testing.assert_close(
        mu_total, mu_graph.squeeze(0).squeeze(-1).transpose(0, 1), rtol=0.0, atol=0.0
    )


def test_residual_enabled_default_has_residual_module():
    model = _tiny_cdl(residual_enabled=True, sampler=_disc_sampler(get_env(_cfg())))
    assert model.residual is not None
    residual_ids = {id(p) for p in model.residual.parameters()}
    opt_ids = {id(p) for group in model.opt.param_groups for p in group["params"]}
    assert residual_ids <= opt_ids, "enabled residual must be trainable"


# --------------------------------------------------------------------------- #
# (c) prediction harness: known tiny MSE/NLL + all four variant rows.
# --------------------------------------------------------------------------- #
def test_prediction_metrics_match_closed_form_on_known_case():
    from torch.distributions import Normal

    class _Stub:
        kpi_start = 1
        device = torch.device("cpu")
        residual_enabled = False

        def predict_next_state(self, s, a):
            return Normal(torch.zeros(s.shape[0], 2), torch.ones(s.shape[0], 2))

    s = torch.zeros(4, 3)
    a = torch.zeros(4, 3)
    s_next = torch.zeros(4, 3)
    s_next[:, 1:] = torch.tensor([1.0, 2.0])  # per-KPI targets 1 and 2
    metrics = prediction_metrics(_Stub(), s, a, s_next)

    assert abs(metrics["mse"] - 2.5) < 1e-6  # (1^2 + 2^2)/2
    expected_nll = 0.5 * np.log(2 * np.pi) + 0.5 * 2.5
    assert abs(metrics["nll"] - expected_nll) < 1e-6
    np.testing.assert_allclose(metrics["mse_per_kpi"], [1.0, 4.0], atol=1e-6)


def test_comparison_table_has_all_four_variant_rows(tmp_path):
    cfg = _cfg()
    env = get_env(cfg)

    full = get_model(
        _cfg(structure_source="discovered", residual_enabled=True), env, sampler=_disc_sampler(env)
    )
    structure_only = get_model(
        _cfg(structure_source="discovered", residual_enabled=False), env, sampler=_disc_sampler(env)
    )
    oracle = get_model(_cfg(structure_source="oracle", residual_enabled=False), env, sampler=None)
    mlp_cfg = replace(load_config("env_i_mlp.yaml"), device="cpu")
    dense = get_model(mlp_cfg, env)

    transitions = held_out_transitions(env, n_transitions=16, seed=0)
    rows = [
        evaluate_variant("full", full, transitions),
        evaluate_variant("structure_only", structure_only, transitions),
        evaluate_variant("oracle", oracle, transitions),
        evaluate_variant("dense", dense, transitions),
    ]

    out_md = tmp_path / "attribution_table.md"
    out_json = tmp_path / "attribution_table.json"
    markdown, payload = write_report(rows, out_md=out_md, out_json=out_json)

    variants = [row["variant"] for row in payload["variants"]]
    assert variants == list(VARIANT_ORDER), "table rows must be the four named variants, in order"
    for row in payload["variants"]:
        assert np.isfinite(row["mse"]) and row["mse"] >= 0.0
        assert np.isfinite(row["nll"])
        assert row["param_count"] > 0
    by_name = {row["variant"]: row for row in payload["variants"]}
    for name in ("structure_only", "oracle", "dense"):
        assert by_name[name]["residual_fraction"] == 0.0
    assert np.isfinite(by_name["full"]["residual_fraction"])
    assert by_name["full"]["residual_fraction"] >= 0.0

    # Both artifacts were written and the markdown carries every variant.
    assert out_md.exists() and out_json.exists()
    for name in VARIANT_ORDER:
        assert name in markdown
