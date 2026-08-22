import torch

from cdd_oran.planners.base import Planner
from cdd_oran.planners.ensemble import EnsembleAggregator, fixed_structures, robust_score_batch


class ModelBasedCEM(Planner):
    def __init__(
        self, model, env, n_candidate: int, n_top: int, n_iter: int, risk_kappa: float = 0.0,
        aggregator: EnsembleAggregator | None = None,
    ):
        self.model = model
        self.env = env
        self.xapps = env.xapps
        self.num_bins = env.num_bins
        self.num_params = env.num_params
        self.action_space = env.action_space
        self.name = "ModelBasedCEM"
        self.n_candidate = n_candidate
        self.n_top = n_top
        self.n_iter = n_iter
        self.risk_kappa = risk_kappa
        self.aggregator = aggregator or EnsembleAggregator()
        self.last_disagreement = 0.0
        self.last_ood = False
        self.device = model.device

    def act(self, current_state, conflict_param_index, xapps_under_conflict, weights_per_xapps, scaling_term):
        # Draw ONE structure set for this decision; all candidates/iters below share it.
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
        weights = weights_per_xapps
        max_index = self.env.action_space[pi + 2]
        lo = torch.zeros(2, device=self.device)
        hi = torch.tensor(
            [self.action_space[1], max_index], dtype=torch.float32, device=self.device
        )
        mu = (lo + hi) / 2.0
        std = (hi - lo) / 6.0

        for _ in range(self.n_iter):
            noise = torch.randn(self.n_candidate, 2, device=self.device)
            samples = (mu + std * noise).round().long()
            samples = torch.stack(
                [
                    samples[:, 0].clamp(0, self.action_space[1]),
                    samples[:, 1].clamp(0, max_index),
                ],
                dim=1,
            )
            pi_col = torch.full((self.n_candidate, 1), pi, dtype=torch.long, device=self.device)
            action_batch = torch.cat([pi_col, samples], dim=1).float()
            s_batch = s0.unsqueeze(0).expand(self.n_candidate, -1).float()
            # Robust score over ensemble members (m=1 -> byte-identical to score_batch).
            scores, disagreement = robust_score_batch(
                self.model, s_batch, action_batch, xapps, weights, scaling_term,
                self.device, self.risk_kappa, self.aggregator,
            )
            self.last_disagreement = float(disagreement.mean())
            self.last_ood = self.aggregator.ood(disagreement)
            elites = samples[torch.argsort(scores)[: self.n_top]].float()
            mu = elites.mean(dim=0)
            std = elites.std(dim=0).clamp(min=1.0)

        best_bin = mu[0].round().long().clamp(0, self.action_space[1]).item()
        best_idx = mu[1].round().long().clamp(0, max_index).item()
        return [pi, best_bin, best_idx]
