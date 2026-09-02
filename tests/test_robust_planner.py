"""CPU tests for Phase 3 robust planning over per-model returns. No GPU, no training.

(a) the risk-averse aggregator picks a different (safer) action than the mean aggregator;
(b) the disagreement signal is finite, >0 when members differ, 0 when identical;
(c) all four planners run end-to-end with a 2-member ensemble and return valid actions;
(d) the m=1 scoring path is byte-identical to the pre-Phase-3 score_batch path.
"""

from dataclasses import replace

import numpy as np
import torch
from torch.distributions import Normal

from cdd_oran.config import DEFAULT_CONFIG
from cdd_oran.conflicts import detect_conflict_edges, state_to_tensor
from cdd_oran.envs import get_env
from cdd_oran.envs.base import XApp
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


# --------------------------------------------------------------------------- #
# (e) Lever 2 / Option A: the utility_weight term in the PER-MEMBER cost.
# --------------------------------------------------------------------------- #
def test_utility_weight_zero_is_byte_identical_to_pre_term_path():
    """lambda=0 parity: the new term is a strict no-op, so an explicit 0.0 reproduces the
    pre-change score_batch call bit-for-bit -- on the vectorised path, on the reference
    oracle, and end-to-end through the m=1 robust path used by every planner."""
    from cdd_oran.planners.cost import _score_batch_reference

    env, state, edge = _env_and_state()
    model = _ensemble_model(env, predict_members=1)
    pi = edge["param_id"]
    xapps = edge["xapps_in_conflict"]
    weights = [1.0] * len(xapps)

    n = 8
    samples = torch.randint(0, 3, (n, 2))
    pi_col = torch.full((n, 1), pi, dtype=torch.long)
    action_batch = torch.cat([pi_col, samples], dim=1).float()
    s_batch = state.unsqueeze(0).expand(n, -1).float()

    torch.manual_seed(123)
    dist = model.predict_next_state(s_batch, action_batch)
    kpis, stds = dist.sample(), dist.stddev

    # The pre-change call signature (no utility_weight argument at all).
    pre = score_batch(kpis, xapps, weights, 10, "cpu", stds, 0.0)
    explicit_zero = score_batch(kpis, xapps, weights, 10, "cpu", stds, 0.0, 0.0)
    assert torch.equal(explicit_zero, pre), "utility_weight=0.0 must be a bit-identical no-op"
    ref = _score_batch_reference(kpis, xapps, weights, 10, "cpu", stds, 0.0, 0.0)
    torch.testing.assert_close(explicit_zero, ref, rtol=0.0, atol=0.0)

    # End-to-end through the shared robust path with an aggregator carrying utility_weight=0.
    torch.manual_seed(123)
    scores, _ = robust_score_batch(
        model, s_batch, action_batch, xapps, weights, 10, "cpu", 0.0,
        EnsembleAggregator(utility_weight=0.0),
    )
    torch.testing.assert_close(scores, pre, rtol=0.0, atol=0.0)

    # ... and the fast/reference paths stay byte-consistent once the term is ON.
    fast_on = score_batch(kpis, xapps, weights, 10, "cpu", stds, 0.0, 0.3)
    ref_on = _score_batch_reference(kpis, xapps, weights, 10, "cpu", stds, 0.0, 0.3)
    torch.testing.assert_close(fast_on, ref_on, rtol=0.0, atol=0.0)


def test_utility_weight_picks_the_higher_utility_satisfied_candidate():
    """The diagnosed flat region: candidates that are ALL satisfied (distance clamped to 0,
    equal satisfaction count) score identically today, so the argmin keeps the first one
    scanned regardless of how much utility is left on the table. With utility_weight>0 the
    higher-utility candidate must win -- for a maximiser and a minimiser alike."""
    for direction in (0, 1):
        # A real XApp (not a SimpleNamespace) so the score_batch(Sequence[XApp]) contract is
        # actually type-checked. mean_std=(0, 1) is the identity standardisation this test wants.
        xapp = XApp(
            threshold=0.0,
            utility_fn=lambda k: float(k[0]),
            name="t",
            params=(),
            direction=direction,
            mean_std=(0.0, 1.0),
        )
        # dir 0 (maximiser): utilities 1 < 2, both >= threshold 0 -> both satisfied.
        # dir 1 (minimiser): utilities -1 > -2, both <= threshold 0 -> both satisfied.
        worse, better = (1.0, 2.0) if direction == 0 else (-1.0, -2.0)
        kpis = torch.tensor([[worse], [better]])
        stds = torch.ones(2, 1)

        flat = score_batch(kpis, [xapp], [1.0], 1.0, "cpu", stds, 0.0, 0.0)
        assert float(flat[0]) == float(flat[1]) == -1.0, "candidates must be equally satisfied"
        assert int(flat.argmin()) == 0, "utility_weight=0 ties and keeps the FIRST candidate"

        rewarded = score_batch(kpis, [xapp], [1.0], 1.0, "cpu", stds, 0.0, 0.5)
        assert float(rewarded[1]) < float(rewarded[0]), (
            f"dir {direction}: the higher-utility satisfied candidate must cost strictly less"
        )
        assert int(rewarded.argmin()) == 1, f"dir {direction}: argmin must move off the tie"


def test_risk_averse_aggregator_still_diverges_from_mean_with_utility_weight():
    """Risk-aversion invariant: the utility term lives INSIDE each member's cost, so the
    quantile aggregator still takes a pessimistic view of the utility-inclusive per-member
    totals -- it must NOT collapse into expected utility."""
    # Candidate 0 is SAFE (utility 1 under both members). Candidate 1 is RISKY: member 0 sees
    # a big satisfied utility (8), member 1 sees an UNSATISFIED -1.
    mean = torch.tensor([[[1.0], [8.0]], [[1.0], [-1.0]]])  # (m=2, batch=2, k=1)
    dist = Normal(mean, torch.full_like(mean, 1e-6))
    xapps, weights, tau, dev, kappa = _stub_xapps()

    mean_agg = EnsembleAggregator(method="mean", utility_weight=0.5)
    risk_agg = EnsembleAggregator(method="quantile", quantile=0.9, utility_weight=0.5)

    torch.manual_seed(0)
    mean_scores, _ = robust_returns_from_dist(dist, xapps, weights, tau, dev, kappa, mean_agg)
    torch.manual_seed(0)
    risk_scores, _ = robust_returns_from_dist(dist, xapps, weights, tau, dev, kappa, risk_agg)

    assert int(mean_scores.argmin()) == 1, "expected-cost still chases the high-mean-utility gamble"
    assert int(risk_scores.argmin()) == 0, "risk-averse must still prefer the safe candidate"
    assert int(mean_scores.argmin()) != int(risk_scores.argmin()), (
        "aggregators must still diverge with utility_weight>0 (risk aversion preserved)"
    )

    # Direct PLACEMENT assertion (review point 4): the actions above are also reachable if the
    # utility term were applied AFTER aggregation, so pin the VALUES, not just the argmins.
    # (i) the realised score IS combine(utility-inclusive per-member costs) ...
    inside = torch.stack(
        [score_batch(mean[j], xapps, weights, tau, dev, None, kappa, 0.5) for j in range(2)]
    )  # (m, batch) -- exactly what reaches EnsembleAggregator.combine
    torch.testing.assert_close(risk_scores, risk_agg.combine(inside), rtol=1e-4, atol=1e-4)

    # ... and (ii) it is NOT "aggregate the plain costs, then subtract the EXPECTED utility",
    # which is the expected-utility collapse this placement exists to avoid.
    plain = torch.stack(
        [score_batch(mean[j], xapps, weights, tau, dev, None, kappa, 0.0) for j in range(2)]
    )
    after = risk_agg.combine(plain) - 0.5 * mean[..., 0].mean(dim=0)  # direction 0 -> sign +1
    assert not torch.allclose(risk_scores, after, rtol=1e-3, atol=1e-3), (
        "utility must ride INSIDE each member cost, not be subtracted from the aggregate"
    )


# --------------------------------------------------------------------------- #
# Deterministic m=1 grid harness for the QACM / MCTS *manual* scoring branches.
# Both bypass score_batch when m=1, so they need their own coverage; a fully
# deterministic model makes the expected pick exactly predictable.
# --------------------------------------------------------------------------- #
_NUM_BINS, _MAX_INDEX = 2, 2
_GRID = [(b, i) for b in range(_NUM_BINS) for i in range(_MAX_INDEX + 1)]  # QACM scan order


class _DeterministicDist:
    """Degenerate (batch, k) Normal-like stub: ``sample()`` returns the mean exactly."""

    def __init__(self, mean):
        self.mean = mean
        self.stddev = torch.ones_like(mean)

    def sample(self):
        return self.mean


def _rising_offset(bin_id, index):
    """Signed utility strictly increases with (bin, index) -> unique best is (1, 2)."""
    return 1.0 + 3.0 * bin_id + index


def _index_only_offset(bin_id, index):
    """Signed utility depends only on the index and DECREASES with it -> best is index 0."""
    return 3.0 - index


def _index_only_rising(bin_id, index):
    """Signed utility depends only on the index and INCREASES with it -> best is the LAST index.

    Used by the end-to-end MCTS move test: the batched planner breaks a flat tie deterministically
    to the first-scanned leaf (index 0), so the utility-driven pick must land on a DIFFERENT index
    for the move to be observable. A rising offset puts the best index at ``_MAX_INDEX``."""
    return 1.0 + index


class _GridModel:
    """An m=1 world model whose next-KPI vector is a pure lookup on (bin_id, index).

    ``sign`` flips the raw KPI for a minimiser xApp, so the SIGNED utility ``sign_i * u_i``
    follows ``offset`` in BOTH directions: a flipped sign in the planner under test therefore
    ranks the candidates in the opposite order and the test fails."""

    def __init__(self, direction, offset):
        self.sign = 1.0 if direction == 0 else -1.0
        self.offset = offset
        self.device = "cpu"
        self.predict_members = 1

    def kpi(self, bin_id, index):
        return self.sign * self.offset(bin_id, index)

    def predict_next_state(self, states, actions):
        rows = [[self.kpi(int(a[1]), int(a[2]))] for a in actions]
        return _DeterministicDist(torch.tensor(rows, dtype=torch.float32))


def _flat_xapp(direction):
    """One affine xApp on KPI 0 with threshold/mean/std = 0/0/1, so utility == kpis[0]."""
    from types import SimpleNamespace

    return SimpleNamespace(
        direction=direction, threshold=0.0, mean=0.0, std=1.0,
        compute_utility=lambda k: float(k[0]),
    )


def _grid_env(xapp):
    from types import SimpleNamespace

    # action_space = [num_params-1, num_bins-1, *max_bin_lengths]; one param -> pi = 0.
    return SimpleNamespace(
        xapps=[xapp], num_bins=_NUM_BINS, num_params=1,
        action_space=[0, _NUM_BINS - 1, _MAX_INDEX],
    )


def _grid_scores(model, xapp, utility_weight, grid):
    """The INDEPENDENT expectation: the same candidates scored by the vectorised
    ``score_batch`` that the manual m=1 branches are supposed to mirror."""
    kpis = torch.tensor([[model.kpi(b, i)] for b, i in grid], dtype=torch.float32)
    return score_batch(
        kpis, [xapp], [1.0], 1.0, "cpu", torch.ones_like(kpis), 0.0, utility_weight
    )


def _qacm_action(model, env, xapp, aggregator):
    return QACM(model=model, env=env, aggregator=aggregator).act(
        torch.zeros(1), 0, [xapp], [1.0], 1.0
    )


def _mcts_action(model, env, xapp, aggregator, n_simulations=60):
    np.random.seed(0)          # _act draws a rollout index for freshly expanded bin nodes
    torch.manual_seed(0)
    planner = ModelBasedMCTS(
        model=model, env=env, n_simulations=n_simulations, ucb_c=1.5, aggregator=aggregator
    )
    return planner.act(torch.zeros(1), 0, [xapp], [1.0], 1.0)


def test_utility_weight_moves_the_qacm_m1_pick_to_the_expected_candidate():
    """The m=1 manual scoring branch in QACM must mirror score_batch's term exactly, so m=1
    and m>1 agree. Six candidates, ALL satisfied -> at lambda=0 they tie at cost -1 and QACM
    keeps the first scanned, (0, 0). At lambda>0 the pick must land on the candidate the
    independent score_batch expectation names -- the highest SIGNED utility, (1, 2) -- for a
    maximiser and a minimiser alike. A flipped sign or an arbitrary perturbation lands
    elsewhere and fails."""
    for direction in (0, 1):
        xapp = _flat_xapp(direction)
        env = _grid_env(xapp)
        model = _GridModel(direction, _rising_offset)

        flat = _grid_scores(model, xapp, 0.0, _GRID)
        assert torch.all(flat == -1.0), f"dir {direction}: candidates are not equally satisfied"
        flat_action = [0, *_GRID[int(flat.argmin())]]
        assert flat_action == [0, 0, 0], "lambda=0 ties and keeps the first-scanned candidate"

        rewarded = _grid_scores(model, xapp, 0.5, _GRID)
        rewarded_action = [0, *_GRID[int(rewarded.argmin())]]
        assert rewarded_action == [0, 1, 2], f"dir {direction}: (1, 2) is the best-utility pick"

        baseline = _qacm_action(model, env, xapp, EnsembleAggregator())  # pre-change default
        assert baseline == flat_action, f"dir {direction}: lambda=0 must sit on the flat tie"
        assert _qacm_action(model, env, xapp, EnsembleAggregator(utility_weight=0.0)) == baseline, (
            f"dir {direction}: utility_weight=0.0 must leave the m=1 QACM decision unchanged"
        )
        assert _qacm_action(model, env, xapp, EnsembleAggregator(utility_weight=0.5)) == (
            rewarded_action
        ), (
            f"dir {direction}: m=1 QACM must land on the score_batch argmin {rewarded_action} "
            f"(the highest-signed-utility candidate); a flipped sign would pick (0, 0)"
        )

    # ... and on the REAL env/CDL model the m=1 QACM decision is untouched at lambda = 0.
    env, state, edge = _env_and_state()
    model = _ensemble_model(env, predict_members=1)  # (batch, k) -> QACM manual numpy branch
    pi = edge["param_id"]
    xapps = edge["xapps_in_conflict"]
    weights = [1.0] * len(xapps)

    def real_act(aggregator):
        torch.manual_seed(11)
        return QACM(model=model, env=env, aggregator=aggregator).act(
            state.clone(), pi, xapps, weights, 10
        )

    assert real_act(EnsembleAggregator(utility_weight=0.0)) == real_act(EnsembleAggregator()), (
        "utility_weight=0.0 must leave the real-model m=1 QACM decision unchanged"
    )


def test_mcts_m1_evaluate_mirrors_score_batch_with_the_utility_weight_term():
    """MCTS is the fourth planner with a hand-rolled m=1 branch (only m>1 is routed through
    robust_returns_from_dist). Its per-candidate cost must equal score_batch's: flat -1 for
    every equally-satisfied candidate at lambda=0, utility-ordered at lambda>0. Before the
    fix the lambda>0 case still returned the flat pre-term cost and this assertion failed."""
    grid = [(0, i) for i in range(_MAX_INDEX + 1)]
    for direction in (0, 1):
        xapp = _flat_xapp(direction)
        env = _grid_env(xapp)
        model = _GridModel(direction, _index_only_offset)

        for lam in (0.0, 0.5):
            planner = ModelBasedMCTS(
                model=model, env=env, n_simulations=1, ucb_c=1.5,
                aggregator=EnsembleAggregator(utility_weight=lam),
            )
            got = torch.tensor(
                [
                    planner._evaluate(action, 0, torch.zeros(1), [xapp], [1.0], 1.0)
                    for action in grid
                ],
                dtype=torch.float32,
            )
            want = _grid_scores(model, xapp, lam, grid)
            torch.testing.assert_close(got, want, rtol=0.0, atol=0.0)
            if lam == 0.0:
                assert torch.all(got == -1.0), f"dir {direction}: lambda=0 must be flat at -1"
            else:
                assert int(got.argmin()) == 0, (
                    f"dir {direction}: lambda>0 must rank index 0 (highest signed utility) best"
                )


def test_utility_weight_moves_the_mcts_m1_pick_off_the_flat_region():
    """MCTS m=1 end-to-end. Signed utility depends only on the INDEX (so the outcome does not
    depend on which bins the batched search evaluated) and INCREASES with it, making the last
    index (``_MAX_INDEX``) the unique best. At lambda=0 the cost surface is flat, so the batched
    search breaks the tie deterministically to the first-scanned leaf (index 0); at lambda>0 the
    chosen index must move to ``_MAX_INDEX`` -- for a maximiser AND a minimiser, so a flipped sign
    (which would choose index 0) cannot pass, and cannot masquerade as the flat-tie baseline."""
    for direction in (0, 1):
        xapp = _flat_xapp(direction)
        env = _grid_env(xapp)
        model = _GridModel(direction, _index_only_rising)

        baseline = _mcts_action(model, env, xapp, EnsembleAggregator())  # pre-change default
        assert _mcts_action(model, env, xapp, EnsembleAggregator(utility_weight=0.0)) == baseline, (
            f"dir {direction}: utility_weight=0.0 must leave the m=1 MCTS decision unchanged"
        )
        assert torch.all(_grid_scores(model, xapp, 0.0, _GRID) == -1.0), (
            "the lambda=0 baseline above must be a pick off a genuinely FLAT cost surface"
        )
        assert baseline[2] == 0, (
            f"dir {direction}: the flat-tie baseline must sit on the first-scanned index 0 "
            f"(got {baseline}); the utility-driven move below is measured against it"
        )

        rewarded = _mcts_action(model, env, xapp, EnsembleAggregator(utility_weight=0.5))
        assert rewarded[2] == _MAX_INDEX, (
            f"dir {direction}: m=1 MCTS must move to the highest-signed-utility index {_MAX_INDEX} "
            f"(a flipped sign would pick index 0); got {rewarded}"
        )
        assert rewarded[2] != baseline[2], (
            f"dir {direction}: the flat lambda=0 search must not already sit on index {_MAX_INDEX} "
            f"(baseline {baseline}), or the move proves nothing"
        )


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


def test_m1_planner_action_is_independent_of_ensemble_aggregation_settings():
    """A CEM run on an m=1 model gives the same action whatever the AGGREGATION settings
    (method / quantile / kappa / disagreement_penalty) are -- with one member there is
    nothing to aggregate, so the Phase 3 wrapper does not perturb the decision.

    Scoped to the aggregation settings on purpose: ``utility_weight`` also rides on
    EnsembleAggregator but is NOT an aggregation setting -- it is a per-member cost term and
    it DOES change the m=1 decision (see the utility_weight tests above)."""
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
    assert a1 == a2, (
        "m=1 action must not depend on the ensemble AGGREGATION settings "
        "(method/quantile/kappa/disagreement_penalty)"
    )
