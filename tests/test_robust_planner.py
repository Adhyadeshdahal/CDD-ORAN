"""CPU tests for Phase 3 robust planning over per-model returns. No GPU, no training.

(a) the risk-averse aggregator picks a different (safer) action than the mean aggregator;
(b) the disagreement signal is finite, >0 when members differ, 0 when identical;
(c) all four planners run end-to-end with a 2-member ensemble and return valid actions;
(d) the m=1 scoring path is byte-identical to the pre-Phase-3 score_batch path.
"""

from dataclasses import replace

import torch
from torch.distributions import Normal

from cdd_oran.config import DEFAULT_CONFIG
from cdd_oran.conflicts import detect_conflict_edges, state_to_tensor
from cdd_oran.envs import get_env
from cdd_oran.models.cdl import CDL
from cdd_oran.planners.cem import ModelBasedCEM
from cdd_oran.planners.cost import score_batch
from cdd_oran.planners.ensemble import (
    EnsembleAggregator,
    member_disagreement,
    robust_returns_from_dist,
    robust_score_batch,
)
from cdd_oran.planners.mcts import ModelBasedMCTS
from cdd_oran.planners.mppi import ModelBasedMPPI
from cdd_oran.planners.qacm import QACM


# --------------------------------------------------------------------------- #
# (a) risk-averse vs mean aggregator select different actions.
# --------------------------------------------------------------------------- #
def test_risk_averse_aggregator_picks_safer_action_than_mean():
    # Two candidates scored under two members (rows = members, cols = candidates).
    # Candidate 0 is SAFE (identical cost across members); candidate 1 is RISKY
    # (low mean cost but high spread).
    per_member = torch.tensor([[1.0, 0.0], [1.0, 1.8]])  # (m=2, n=2)

    mean_agg = EnsembleAggregator(method="mean")
    risk_agg = EnsembleAggregator(method="quantile", quantile=0.9)  # the default is risk-averse

    mean_choice = int(torch.argmin(mean_agg.combine(per_member)))
    risk_choice = int(torch.argmin(risk_agg.combine(per_member)))

    assert mean_choice == 1, "mean cost prefers the low-mean risky candidate"
    assert risk_choice == 0, "risk-averse (upper-cost-quantile) prefers the safe candidate"
    assert mean_choice != risk_choice, "aggregators must diverge on this constructed case"

    # kappa aggregator (mean + kappa*std) is also risk-averse and picks the safe one.
    kappa_agg = EnsembleAggregator(method="kappa", kappa=1.0)
    assert int(torch.argmin(kappa_agg.combine(per_member))) == 0


def test_single_member_aggregation_is_passthrough():
    per_member = torch.tensor([[0.3, 0.7, 0.1]])  # (m=1, n=3)
    for method in ("mean", "quantile", "kappa"):
        agg = EnsembleAggregator(method=method)
        torch.testing.assert_close(agg.combine(per_member), per_member[0])


# --------------------------------------------------------------------------- #
# (b) disagreement signal.
# --------------------------------------------------------------------------- #
def test_disagreement_is_positive_when_members_differ_and_zero_when_identical():
    std = torch.ones(2, 1, 1)
    differ = Normal(torch.tensor([[[1.0]], [[3.0]]]), std)
    identical = Normal(torch.tensor([[[2.0]], [[2.0]]]), std)

    d_diff = member_disagreement(differ)
    d_same = member_disagreement(identical)
    assert torch.isfinite(d_diff).all() and float(d_diff[0]) > 0.0
    assert float(d_same[0]) == 0.0

    # m=1 dist (batch, k) -> no member axis -> zero disagreement.
    single = Normal(torch.tensor([[2.0]]), torch.tensor([[1.0]]))
    assert torch.all(member_disagreement(single) == 0.0)


def test_disagreement_penalty_raises_cost_and_ood_flag_fires():
    mean = torch.tensor([[[0.0]], [[4.0]]])  # members disagree a lot (batch=1, k=1)
    dist = Normal(mean, torch.ones(2, 1, 1))
    xapps, weights, tau, dev, kappa = _stub_xapps()

    plain = EnsembleAggregator(method="mean", disagreement_penalty=0.0, ood_threshold=1.0)
    penalized = EnsembleAggregator(method="mean", disagreement_penalty=10.0, ood_threshold=1.0)

    torch.manual_seed(0)
    s_plain, dis_plain = robust_returns_from_dist(dist, xapps, weights, tau, dev, kappa, plain)
    torch.manual_seed(0)
    s_pen, dis_pen = robust_returns_from_dist(dist, xapps, weights, tau, dev, kappa, penalized)

    assert float(dis_plain[0]) > 0.0
    assert float(s_pen[0]) > float(s_plain[0]), "penalty must increase the cost of a risky candidate"
    assert penalized.ood(dis_pen) is True, "OOD flag must fire above threshold"
    assert plain.ood(torch.zeros(1)) is False


def _stub_xapps():
    """A single affine xApp reading KPI 0; enough to exercise score_batch."""
    from types import SimpleNamespace

    xapp = SimpleNamespace(
        direction=0, threshold=0.0, mean=0.0, std=1.0,
        compute_utility=lambda k: float(k[0]),
    )
    return [xapp], [1.0], 1.0, "cpu", 0.0


# --------------------------------------------------------------------------- #
# (c) + (d) end-to-end with a real env and a 2-member ensemble model.
# --------------------------------------------------------------------------- #
class _FixedSampler:
    def __init__(self, structures):
        self.structures = structures

    def sample_structures(self, n_members, *, device):
        return self.structures[:n_members].to(device)


def _env_and_state():
    cfg = replace(DEFAULT_CONFIG, device="cpu", model_kind="cdl")
    torch.manual_seed(0)
    env = get_env(cfg)
    env.reset()
    state = state_to_tensor(env.get_state()).to("cpu")
    edges = detect_conflict_edges(env.true_adj_matrix, env)
    assert edges, "need a conflict edge to plan against"
    return env, state, edges[0]


def _ensemble_model(env, predict_members):
    state_dim = env.get_state_dim()
    ns = state_dim + 1
    torch.manual_seed(1)
    structs = torch.rand(4, state_dim, ns) > 0.5
    torch.manual_seed(2)
    return CDL(
        state_dim=state_dim,
        action_dim=env.get_action_dim(),
        kpi_start=env.num_params,
        feature_fc_dims=[8],
        generative_fc_dims=[8],
        lr=1e-3,
        cmi_threshold=0.2,
        eval_tau=0.99,
        grad_clip=10.0,
        device="cpu",
        node_names=[f"n{i}" for i in range(state_dim)],
        eval_steps=2,
        residual_hidden=[8],
        sampler=_FixedSampler(structs),
        predict_members=predict_members,
    )


def test_all_four_planners_run_end_to_end_with_ensemble():
    env, state, edge = _env_and_state()
    model = _ensemble_model(env, predict_members=2)
    pi = edge["param_id"]
    xapps = edge["xapps_in_conflict"]
    weights = [1.0] * len(xapps)
    agg = EnsembleAggregator(method="quantile", quantile=0.9, ood_threshold=0.0)

    planners = [
        QACM(model=model, env=env, aggregator=agg),
        ModelBasedCEM(model=model, env=env, n_candidate=8, n_top=4, n_iter=2, aggregator=agg),
        ModelBasedMPPI(
            model=model, env=env, n_samples=8, temperature=0.6, noise_sigma=0.1, aggregator=agg
        ),
        ModelBasedMCTS(model=model, env=env, n_simulations=6, ucb_c=1.5, aggregator=agg),
    ]

    for planner in planners:
        torch.manual_seed(7)
        action = planner.act(state.clone(), pi, xapps, weights, 10)
        assert isinstance(action, list) and len(action) == 3, f"{planner.name} bad action shape"
        assert action[0] == pi
        assert 0 <= action[1] <= env.action_space[1], f"{planner.name} bin out of range"
        assert 0 <= action[2] <= env.action_space[pi + 2], f"{planner.name} index out of range"
        # The member axis was actually consumed: disagreement was recorded and is finite.
        assert torch.isfinite(torch.tensor(planner.last_disagreement))


def test_m1_scoring_is_byte_identical_to_pre_phase3_path():
    """(d) With m=1 the robust path must reproduce the exact score_batch call/RNG."""
    env, state, edge = _env_and_state()
    model = _ensemble_model(env, predict_members=1)  # single-member -> (batch, k) Normal
    pi = edge["param_id"]
    xapps = edge["xapps_in_conflict"]
    weights = [1.0] * len(xapps)

    n = 8
    samples = torch.randint(0, 3, (n, 2))
    pi_col = torch.full((n, 1), pi, dtype=torch.long)
    action_batch = torch.cat([pi_col, samples], dim=1).float()
    s_batch = state.unsqueeze(0).expand(n, -1).float()
    agg = EnsembleAggregator()

    # Reference: the pre-Phase-3 block (predict -> sample -> score_batch).
    torch.manual_seed(123)
    dist = model.predict_next_state(s_batch, action_batch)
    ref = score_batch(dist.sample(), xapps, weights, 10, "cpu", dist.stddev, 0.0)

    # Phase 3 shared path with the identical RNG seed.
    torch.manual_seed(123)
    scores, disagreement = robust_score_batch(
        model, s_batch, action_batch, xapps, weights, 10, "cpu", 0.0, agg
    )
    torch.testing.assert_close(scores, ref, rtol=0.0, atol=0.0)
    assert torch.all(disagreement == 0.0), "m=1 disagreement must be exactly zero"


class _RandomSampler:
    """Returns a DIFFERENT structure set on each call, so the lifecycle scope is observable."""

    def __init__(self, state_dim, seed=0):
        self.state_dim = state_dim
        self.generator = torch.Generator().manual_seed(seed)

    def sample_structures(self, n_members, *, device):
        ns = self.state_dim + 1
        return torch.rand(n_members, self.state_dim, ns, generator=self.generator) > 0.5


def test_structures_are_fixed_within_a_decision_and_fresh_across_decisions():
    """Lifecycle contract: structures are drawn ONCE per decision (act) and held FIXED for
    every candidate/rollout inside it; a fresh set is drawn at the next decision."""
    from cdd_oran.planners.ensemble import fixed_structures

    env, _, _ = _env_and_state()
    model = _ensemble_model(env, predict_members=2)
    model.sampler = _RandomSampler(env.get_state_dim(), seed=0)
    s = torch.rand(3, env.get_state_dim())
    a = torch.rand(3, env.get_action_dim())

    # Within ONE decision scope, repeated predictions reuse the SAME members -> equal means.
    with fixed_structures(model):
        m1 = model.predict_next_state(s, a).mean.clone()
        m2 = model.predict_next_state(s, a).mean.clone()
    torch.testing.assert_close(m1, m2)

    # A SECOND decision draws a fresh member set -> different structures -> different means.
    with fixed_structures(model):
        m3 = model.predict_next_state(s, a).mean.clone()
    assert not torch.allclose(m1, m3), "each decision must draw fresh structures"

    # Outside any decision scope, bare predictions also redraw every call.
    b1 = model.predict_next_state(s, a).mean
    b2 = model.predict_next_state(s, a).mean
    assert not torch.allclose(b1, b2)


def test_structure_conditioned_m1_uses_one_fixed_structure_per_decision():
    """Review v2 MINOR #5: for structure_conditioned predict_members=1, fixed_structures is
    NOT a byte-identical no-op -- it draws ONE structure per decision (the intended fair-
    comparison semantics) and holds it across candidates/rollout, then draws fresh next time."""
    from cdd_oran.planners.ensemble import fixed_structures

    env, state, _ = _env_and_state()
    model = _ensemble_model(env, predict_members=1)  # single-member structure_conditioned
    model.sampler = _RandomSampler(env.get_state_dim(), seed=3)

    with fixed_structures(model):
        first = model._decision_structures.clone()
        assert first.shape[0] == 1, "predict_members=1 draws a single member"
        for _ in range(4):
            model.predict_next_state(state.unsqueeze(0), torch.zeros(1, env.get_action_dim()))
            assert torch.equal(model._decision_structures, first), "structure changed mid-decision"
    assert model._decision_structures is None, "decision scope must clear on exit"

    with fixed_structures(model):
        second = model._decision_structures.clone()
    assert not torch.equal(first, second), "the next decision must draw a fresh structure"


def test_every_candidate_in_one_act_call_uses_the_same_members():
    """The whole point of the fixed scope: a QACM act call, which predicts once PER candidate
    action, scores every candidate under the identical member set (no cross-graph compare)."""
    from cdd_oran.planners.ensemble import fixed_structures

    env, state, edge = _env_and_state()
    model = _ensemble_model(env, predict_members=2)
    model.sampler = _RandomSampler(env.get_state_dim(), seed=1)

    with fixed_structures(model):
        assert model._decision_structures is not None
        first = model._decision_structures.clone()
        # Simulate several candidate predictions within the same decision.
        for _ in range(5):
            model.predict_next_state(state.unsqueeze(0), torch.zeros(1, env.get_action_dim()))
            assert torch.equal(model._decision_structures, first), "members changed mid-decision"
    assert model._decision_structures is None, "decision scope must clear on exit"


def test_m1_planner_action_matches_baseline_scoring():
    """A CEM run on an m=1 model gives the same action whether or not the aggregator is
    present -- the Phase 3 wrapper does not perturb the single-model decision."""
    env, state, edge = _env_and_state()
    model = _ensemble_model(env, predict_members=1)
    pi = edge["param_id"]
    xapps = edge["xapps_in_conflict"]
    weights = [1.0] * len(xapps)

    default_agg = ModelBasedCEM(model=model, env=env, n_candidate=8, n_top=4, n_iter=2)
    explicit_agg = ModelBasedCEM(
        model=model, env=env, n_candidate=8, n_top=4, n_iter=2,
        aggregator=EnsembleAggregator(method="kappa", kappa=5.0, disagreement_penalty=9.0),
    )
    torch.manual_seed(5)
    a1 = default_agg.act(state.clone(), pi, xapps, weights, 10)
    torch.manual_seed(5)
    a2 = explicit_agg.act(state.clone(), pi, xapps, weights, 10)
    assert a1 == a2, "m=1 action must not depend on the ensemble aggregator settings"
