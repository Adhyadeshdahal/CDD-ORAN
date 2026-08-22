import torch

from cdd_oran.planners.base import Planner
from cdd_oran.planners.ensemble import EnsembleAggregator, fixed_structures, robust_score_batch


class ModelBasedMPPI(Planner):
    """
    Model Predictive Path Integral planning.
    Same act() interface as QACM and ModelBasedCEM.

    Key difference from CEM: all N samples contribute to the update
    via softmax weights — no hard elite selection, no iteration loop.
    One forward pass through the model is enough.
    """

    def __init__(
        self,
        model,
        env,
        n_samples,
        temperature,
        noise_sigma,
        risk_kappa: float = 0.0,
        aggregator: EnsembleAggregator | None = None,
    ):
        self.model = model
        self.env = env
        self.xapps = env.xapps
        self.num_bins = env.num_bins
        self.num_params = env.num_params
        self.action_space = env.action_space
        self.name = "ModelBasedMPPI"

        self.n_samples = n_samples
        self.temperature = temperature
        self.noise_sigma = noise_sigma
        self.risk_kappa = risk_kappa
        self.aggregator = aggregator or EnsembleAggregator()
        self.last_disagreement = 0.0
        self.last_ood = False
        self.device = model.device

    def act(self, current_state, conflict_param_index, xapps_under_conflict, weights_per_xapps, scaling_term):
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

        hi = torch.tensor(
            [self.action_space[1], max_index],
            dtype=torch.float32,
            device=self.device,
        )

        mu = hi / 2.0

        sigma = torch.tensor(
            [self.action_space[1] / 4.0, max_index / 4.0],
            dtype=torch.float32,
            device=self.device,
        )

        noise = torch.randn(self.n_samples, 2, device=self.device) * sigma
        samples = (mu + noise).round().long()
        samples = torch.stack(
            [
                samples[:, 0].clamp(0, self.action_space[1]),
                samples[:, 1].clamp(0, max_index),
            ],
            dim=1,
        )

        pi_col = torch.full((self.n_samples, 1), pi, dtype=torch.long, device=self.device)
        action_batch = torch.cat([pi_col, samples], dim=1).float()

        s_batch = s0.unsqueeze(0).expand(self.n_samples, -1).float()
        # Robust score over ensemble members (m=1 -> byte-identical to score_batch).
        costs, disagreement = robust_score_batch(
            self.model, s_batch, action_batch, xapps, w, tau,
            self.device, self.risk_kappa, self.aggregator,
        )
        self.last_disagreement = float(disagreement.mean())
        self.last_ood = self.aggregator.ood(disagreement)

        log_weights = -costs / self.temperature
        log_weights -= log_weights.max()  # numerical stability
        weights = torch.exp(log_weights)
        weights = weights / weights.sum()

        best_2d = (weights.unsqueeze(1) * samples.float()).sum(dim=0)
        best_bin = best_2d[0].round().long().clamp(0, self.action_space[1]).item()
        best_idx = best_2d[1].round().long().clamp(0, max_index).item()
        return [pi, best_bin, best_idx]
