import math

import numpy as np
import torch

from cdd_oran.planners.base import Planner
from cdd_oran.planners.cost import risk_adjust, weighted_distance
from cdd_oran.planners.ensemble import (
    EnsembleAggregator,
    fixed_structures,
    robust_returns_from_dist,
)


class MCTSNode:
    """A single node in the search tree, representing one (bin_id, index) action."""

    __slots__ = ("action", "parent", "children", "visits", "total_cost", "untried")

    def __init__(self, action, parent, untried_actions):
        self.action = action
        self.parent = parent
        self.children = []
        self.visits = 0
        self.total_cost = 0.0
        self.untried = list(untried_actions)


class ModelBasedMCTS(Planner):
    """
    Monte Carlo Tree Search planning.
    Same act() interface as QACM, ModelBasedCEM, ModelBasedMPPI.
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
        s0 = current_state
        pi = conflict_param_index
        xapps = xapps_under_conflict
        w = weights_per_xapps
        tau = scaling_term

        max_index = self.env.action_space[pi + 2]

        bin_ids = list(range(self.action_space[1] + 1))
        root = MCTSNode(action=None, parent=None, untried_actions=bin_ids)

        for _ in range(self.n_simulations):
            node = root

            while not node.untried and node.children:
                node = self._ucb_select(node)

            if node.untried:
                if node.action is None:
                    bin_id = node.untried.pop()
                    index_choices = list(range(max_index + 1))
                    child = MCTSNode(
                        action=bin_id,
                        parent=node,
                        untried_actions=index_choices,
                    )
                else:
                    index = node.untried.pop()
                    child = MCTSNode(
                        action=(node.action, index),
                        parent=node,
                        untried_actions=[],
                    )
                node.children.append(child)
                node = child

            if isinstance(node.action, tuple):
                cost = self._evaluate(node.action, pi, s0, xapps, w, tau)
            else:
                if node.action is not None:
                    rand_idx = np.random.randint(0, max_index + 1)
                    cost = self._evaluate((node.action, rand_idx), pi, s0, xapps, w, tau)
                else:
                    cost = 0.0

            while node is not None:
                node.visits += 1
                node.total_cost += cost
                node = node.parent

        best_cost = float("inf")
        best_bin = 0
        best_idx = 0
        for bin_node in root.children:
            for leaf in bin_node.children:
                if leaf.visits == 0:
                    continue
                avg = leaf.total_cost / leaf.visits
                if avg < best_cost:
                    best_cost = avg
                    best_bin, best_idx = leaf.action
        return [pi, best_bin, best_idx]

    def _ucb_select(self, node: MCTSNode) -> MCTSNode:
        """UCB1 adapted for cost minimisation."""
        log_parent = math.log(node.visits + 1)

        def ucb_score(child):
            if child.visits == 0:
                return 1e9
            avg_cost = child.total_cost / child.visits
            explore = self.ucb_c * math.sqrt(log_parent / child.visits)
            return -(avg_cost - explore)  # higher = better (lower cost)

        return max(node.children, key=ucb_score)

    def _evaluate(self, action_2d, pi, s0, xapps, w, tau) -> float:
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
