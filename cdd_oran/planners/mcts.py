import numpy as np
import torch

from cdd_oran.planners.base import Planner
from cdd_oran.planners.cost import risk_adjust, weighted_distance
from cdd_oran.planners.ensemble import (
    EnsembleAggregator,
    fixed_structures,
    robust_returns_from_dist,
)


class ModelBasedMCTS(Planner):
    """Batched model-based search over the one-step (bin, index) action grid.

    The action tree for a single conflict decision is depth-2 and fully enumerable (root -> bin ->
    (bin, index)), so the historical Monte-Carlo tree search reduced, with n_simulations >> the leaf
    count, to picking the leaf with the lowest expected cost -- while paying for up to
    ``n_simulations`` SEQUENTIAL, batch-1 model forwards per decision (~59 s on an 8-member ensemble)
    and returning a run-to-run unstable action, because the leaf-cost surface is flat relative to
    prediction noise. This planner spends the same ``n_simulations`` evaluation budget as ``k``
    samples per leaf, scored in ``k`` BATCHED forwards over all leaves, and returns the argmin of the
    per-leaf mean cost: the same decision at 40-100x less compute. Same act() interface as QACM,
    ModelBasedCEM, ModelBasedMPPI.
    """

    def __init__(
        self,
        model,
        env,
        n_simulations,
        ucb_c,
        risk_kappa: float = 0.0,
        aggregator: EnsembleAggregator | None = None,
    ):
        self.risk_kappa = risk_kappa
        self.model = model
        self.env = env
        self.xapps = env.xapps
        self.num_bins = env.num_bins
        self.num_params = env.num_params
        self.action_space = env.action_space
        self.n_simulations = n_simulations
        self.ucb_c = ucb_c
        self.device = model.device
        self.name = "ModelBasedMCTS"
        self.aggregator = aggregator or EnsembleAggregator()
        self.last_disagreement = 0.0
        self.last_ood = False

    def act(self, current_state, conflict_param_index, xapps_under_conflict, weights_per_xapps, scaling_term):
        # One structure set per decision; every tree simulation/rollout shares it.
        with fixed_structures(self.model):
            return self._act(
                current_state, conflict_param_index, xapps_under_conflict,
                weights_per_xapps, scaling_term,
            )

    def _act(
        self,
        current_state,
        conflict_param_index,
        xapps_under_conflict,
        weights_per_xapps,
        scaling_term,
    ):
        # The action tree is depth-2 and fully enumerable (root -> bin -> (bin, index)), so the
        # search reduces to picking the leaf with the lowest expected cost. The historical
        # per-simulation implementation did up to ``n_simulations`` SEQUENTIAL, batch-1 model
        # forwards (~n_simulations/n_leaves redundant visits per leaf) -- ~59 s per decision on
        # an 8-member ensemble -- and was not reproducible run-to-run because the leaf-cost
        # surface is flat relative to prediction noise. We instead spend the same evaluation
        # budget as ``k`` samples per leaf, scored in ``k`` BATCHED forwards over all leaves, and
        # take the argmin of the per-leaf mean. Same decision, 40-100x less compute, stable.
        s0 = current_state
        pi = conflict_param_index
        xapps = xapps_under_conflict
        w = weights_per_xapps
        tau = scaling_term

        max_index = self.env.action_space[pi + 2]
        n_bins = self.action_space[1] + 1
        leaves = [(b, i) for b in range(n_bins) for i in range(max_index + 1)]
        n_leaves = len(leaves)
        k = max(1, round(self.n_simulations / n_leaves))

        actions = torch.tensor(
            [[pi, b, i] for (b, i) in leaves], dtype=torch.float32, device=self.device
        )
        s_rep = s0.unsqueeze(0).float().to(self.device).expand(n_leaves, -1)

        totals = np.zeros(n_leaves)
        for _ in range(k):
            totals += self._score_actions(actions, s_rep, xapps, w, tau)
        means = totals / k

        best = int(np.argmin(means))
        best_bin, best_idx = leaves[best]
        return [pi, best_bin, best_idx]

    def _score_actions(self, actions, s_rep, xapps, w, tau) -> np.ndarray:
        """Cost for a batch of actions ``(N, 3)`` from state rows ``s_rep`` ``(N, state_dim)``.

        Returns an ``(N,)`` numpy array. Mirrors ``_evaluate`` exactly (ensemble and m=1 paths,
        risk adjustment, the Option-A utility term) but scores the whole leaf set in one forward.
        """
        next_state_dist = self.model.predict_next_state(s_rep, actions)
        if next_state_dist.mean.ndim == 3:
            scores, disagreement = robust_returns_from_dist(
                next_state_dist, xapps, w, tau, self.model.device,
                self.risk_kappa, self.aggregator,
            )
            self.last_disagreement = float(disagreement.mean())
            self.last_ood = self.aggregator.ood(disagreement)
            return scores.reshape(-1).cpu().detach().numpy()

        next_state = next_state_dist.sample()
        stds = next_state_dist.stddev if self.risk_kappa != 0.0 else None
        kpis = next_state.cpu().detach().numpy()
        out = np.zeros(kpis.shape[0])
        for r in range(kpis.shape[0]):
            std_row = stds[r].cpu().detach().numpy() if stds is not None else None
            cost_vec = np.zeros(len(xapps))
            sat_vec = np.zeros(len(xapps))
            signed_u = np.zeros(len(xapps))
            for i, xapp in enumerate(xapps):
                u = xapp.compute_utility(risk_adjust(kpis[r], std_row, xapp.direction, self.risk_kappa))
                d, s = weighted_distance(xapp, u)
                cost_vec[i] = w[i] * d * tau
                sat_vec[i] = s
                signed_u[i] = (1.0 if xapp.direction == 0 else -1.0) * u
            f_cost = cost_vec.sum() - (sat_vec.sum()) ** 2
            if self.aggregator.utility_weight:
                f_cost = f_cost - self.aggregator.utility_weight * signed_u.sum()
            out[r] = f_cost
        return out

    def _evaluate(self, action_2d, pi, s0, xapps, w, tau) -> float:
        """Cost of a single (bin, index) action -- the scalar counterpart of ``_score_actions``.

        Retained for direct single-candidate scoring and unit tests; ``_act`` uses the batched
        ``_score_actions`` path. ``ucb_c`` from the config is accepted for wiring compatibility but
        no longer drives a tree search."""
        bin_id, idx = action_2d
        action_tensor = (
            torch.tensor([pi, bin_id, idx], dtype=torch.float32).unsqueeze(0).to(self.device)
        )

        next_state_dist = self.model.predict_next_state(s0.unsqueeze(0).float(), action_tensor)
        if next_state_dist.mean.ndim == 3:
            # Ensemble (m>1): robust aggregation over per-member returns.
            scores, disagreement = robust_returns_from_dist(
                next_state_dist, xapps, w, tau, self.model.device,
                self.risk_kappa, self.aggregator,
            )
            self.last_disagreement = float(disagreement.mean())
            self.last_ood = self.aggregator.ood(disagreement)
            return float(scores.reshape(-1)[0])
        # Model returns KPI portion only (m=1: unchanged manual scoring).
        next_state = next_state_dist.sample().squeeze(0)
        kpis = next_state.cpu().detach().numpy()
        stds = (
            next_state_dist.stddev.squeeze(0).cpu().detach().numpy()
            if self.risk_kappa != 0.0
            else None
        )

        cost_vec = np.zeros(len(xapps))
        sat_vec = np.zeros(len(xapps))
        signed_u = np.zeros(len(xapps))
        for i, xapp in enumerate(xapps):
            u = xapp.compute_utility(risk_adjust(kpis, stds, xapp.direction, self.risk_kappa))
            d, s = weighted_distance(xapp, u)
            cost_vec[i] = w[i] * d * tau
            sat_vec[i] = s
            signed_u[i] = (1.0 if xapp.direction == 0 else -1.0) * u
        f_cost = cost_vec.sum() - (sat_vec.sum()) ** 2
        # Mirror of score_batch's Lever 2 / Option A term so m=1 and m>1 agree.
        # Guarded so utility_weight == 0.0 leaves this path bit-identical.
        if self.aggregator.utility_weight:
            f_cost = f_cost - self.aggregator.utility_weight * signed_u.sum()
        return float(f_cost)
