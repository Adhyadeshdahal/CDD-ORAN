import numpy as np
import torch

from cdd_oran.planners.base import Planner
from cdd_oran.planners.cost import risk_adjust, weighted_distance
from cdd_oran.planners.ensemble import (
    EnsembleAggregator,
    fixed_structures,
    robust_returns_from_dist,
)


class QACM(Planner):
    def __init__(self, model, env, risk_kappa: float = 0.0, aggregator: EnsembleAggregator | None = None):
        self.risk_kappa = risk_kappa
        self.model = model
        self.env = env
        self.xapps = env.xapps
        self.num_bins = env.num_bins
        self.num_params = env.num_params
        self.action_space = env.action_space  # [num_params-1, num_bins-1, *max_bin_lengths]
        self.name = "QACM"
        self.aggregator = aggregator or EnsembleAggregator()
        self.last_disagreement = 0.0
        self.last_ood = False

    def compute_utility(self, xapp, kpis):
        return xapp.compute_utility(kpis)

    def obtain_weighted_distance(self, xapp, utility):
        return weighted_distance(xapp, utility)

    def act(self, current_state, conflict_param_index, xapps_under_conflict, weights_per_xapps, scaling_term):
        # One structure set per decision: every (bin, index) candidate below is scored under
        # the IDENTICAL members (fixes the QACM cross-graph comparison, review OPEN finding).
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

        pl_opt = None
        min_cost = float("inf")

        for bin_id in range(self.action_space[1] + 1):
            for index in range(max_index + 1):
                action = [pi, bin_id, index]

                action_tensor = (
                    torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(self.model.device)
                )

                next_state_dist = self.model.predict_next_state(s0.unsqueeze(0), action_tensor)
                if next_state_dist.mean.ndim == 3:
                    # Ensemble (m>1): robust aggregation over per-member returns.
                    scores, disagreement = robust_returns_from_dist(
                        next_state_dist, xapps, w, tau, self.model.device,
                        self.risk_kappa, self.aggregator,
                    )
                    f_cost = float(scores.reshape(-1)[0])
                    self.last_disagreement = float(disagreement.mean())
                    self.last_ood = self.aggregator.ood(disagreement)
                else:
                    # Model returns KPI portion only (m=1: unchanged manual scoring).
                    next_state = next_state_dist.sample().squeeze(0)
                    next_kpis = next_state.cpu().numpy()
                    stds = (
                        next_state_dist.stddev.squeeze(0).cpu().numpy()
                        if self.risk_kappa != 0.0
                        else None
                    )

                    cost = np.zeros(len(xapps))
                    s = np.zeros(len(xapps))

                    for i, xapp in enumerate(xapps):
                        u_i = self.compute_utility(
                            xapp, risk_adjust(next_kpis, stds, xapp.direction, self.risk_kappa)
                        )
                        d_i, s_i = self.obtain_weighted_distance(xapp, u_i)
                        cost[i] = w[i] * d_i * tau
                        s[i] = s_i

                    f_cost = cost.sum() - (s.sum()) ** 2

                if f_cost < min_cost:
                    min_cost = f_cost
                    pl_opt = action

        return pl_opt
