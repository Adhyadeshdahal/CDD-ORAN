"""Multi-step receding-horizon planner (Task A of brief S3).

Rolls the CDL world model forward H steps autoregressively for a single conflict
edge. The chosen NCP (param) is held fixed along the edge; predicted KPIs are fed
back as the next state's KPI slots. Candidates are scored by the discounted sum of
per-step costs and the first (held) action of the best trajectory is returned.

n_horizon == 1 delegates to a plain ModelBasedCEM, so H=1 reproduces the baseline
EXACTLY (same code path, same RNG stream). Only H>1 uses the rollout below.
"""

import torch

from cdd_oran.planners.base import Planner
from cdd_oran.planners.cem import ModelBasedCEM
from cdd_oran.planners.cost import score_batch
from cdd_oran.planners.ensemble import EnsembleAggregator, fixed_structures, member_count


class RecedingHorizonCEM(Planner):
    def __init__(
        self, model, env, n_horizon, n_candidate, n_top, n_iter, gamma=0.9, risk_kappa: float = 0.0,
        aggregator: EnsembleAggregator | None = None,
    ):
        self.risk_kappa = risk_kappa
        self.aggregator = aggregator or EnsembleAggregator()
        self.last_disagreement = 0.0
        self.last_ood = False
        self.base = ModelBasedCEM(
            model=model, env=env, n_candidate=n_candidate, n_top=n_top, n_iter=n_iter,
            risk_kappa=risk_kappa, aggregator=self.aggregator,
        )
        self.model = model
        self.env = env
        self.action_space = env.action_space
        self.num_bins = env.num_bins
        self.device = model.device
        self.n_horizon = int(n_horizon)
        self.n_candidate = n_candidate
        self.n_top = n_top
        self.n_iter = n_iter
        self.gamma = gamma
        self.name = (
            "ModelBasedCEM"
            if self.n_horizon <= 1
            else f"RecedingHorizonCEM_H{self.n_horizon}"
        )

    def act(
        self,
        current_state,
        conflict_param_index,
        xapps_under_conflict,
        weights_per_xapps,
        scaling_term,
    ):
        # ponytail: H=1 delegates to the baseline -> bit-identical, no RNG reasoning.
        # (base.act already fixes one structure set for its own decision.)
        if self.n_horizon <= 1:
            return self.base.act(
                current_state,
                conflict_param_index,
                xapps_under_conflict,
                weights_per_xapps,
                scaling_term,
            )
        # H>1: fix ONE structure set for the whole multi-step decision (all candidates +
        # every rollout step share it); a fresh set is drawn on the next act.
        with fixed_structures(self.model):
            return self._act_horizon(
                current_state, conflict_param_index, xapps_under_conflict,
                weights_per_xapps, scaling_term,
            )

    def _act_horizon(
        self,
        current_state,
        conflict_param_index,
        xapps_under_conflict,
        weights_per_xapps,
        scaling_term,
    ):
        s0 = current_state.to(self.device).float()
        pi = conflict_param_index
        max_index = self.action_space[pi + 2]

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
            total = self._rollout_cost(
                s0, pi, samples, xapps_under_conflict, weights_per_xapps, scaling_term
            )
            elites = samples[torch.argsort(total)[: self.n_top]].float()
            mu = elites.mean(dim=0)
            std = elites.std(dim=0).clamp(min=1.0)

        best_bin = mu[0].round().long().clamp(0, self.action_space[1]).item()
        best_idx = mu[1].round().long().clamp(0, max_index).item()
        return [pi, best_bin, best_idx]

    def _rollout_cost(self, s0, pi, samples, xapps, weights, scaling_term):
        n = samples.shape[0]
        pi_col = torch.full((n, 1), pi, dtype=torch.float32, device=self.device)
        action_batch = torch.cat([pi_col, samples.float()], dim=1)

        m = member_count(self.model)
        if m == 1:
            return self._rollout_member(s0, action_batch, n, None, xapps, weights, scaling_term)
        # Roll EACH member's own trajectory forward and aggregate the per-member RETURNS
        # (never averaging transitions across members).
        per_member = torch.stack(
            [
                self._rollout_member(s0, action_batch, n, j, xapps, weights, scaling_term)
                for j in range(m)
            ],
            dim=0,
        )  # (m, n)
        disagreement = per_member.std(dim=0)
        self.last_disagreement = float(disagreement.mean())
        self.last_ood = self.aggregator.ood(disagreement)
        combined = self.aggregator.combine(per_member)
        if self.aggregator.disagreement_penalty:
            combined = combined + self.aggregator.disagreement_penalty * disagreement
        return combined

    def _rollout_member(self, s0, action_batch, n, member, xapps, weights, scaling_term):
        """Discounted rollout cost for ONE member (``member=None`` = single model, the exact
        pre-Phase-3 path). Predicts from the UNMODIFIED s0 + action; later steps feed only
        predicted KPIs back and hold the NCP params fixed (CDL is trained on pre-action
        state + action, so injecting the post-value would double-count / go OOD)."""
        num_params = self.env.num_params
        state = s0.unsqueeze(0).expand(n, -1).clone()
        total = torch.zeros(n, device=self.device)
        discount = 1.0
        for _ in range(self.n_horizon):
            dist = self.model.predict_next_state(state, action_batch)
            sample = dist.sample()
            if member is None:
                next_kpis, std = sample, dist.stddev
            else:
                next_kpis, std = sample[member], dist.stddev[member]
            # Risk-depth limit: the kappa shift uses THIS step's one-step aleatoric std only.
            total = total + discount * score_batch(
                next_kpis, xapps, weights, scaling_term, self.device, std, self.risk_kappa,
            )
            state = state.clone()
            state[:, num_params:] = next_kpis  # feed predicted KPIs forward
            discount *= self.gamma
        return total


def _self_check():
    from dataclasses import replace

    from cdd_oran.config import DEFAULT_CONFIG
    from cdd_oran.conflicts import detect_conflict_edges, state_to_tensor
    from cdd_oran.envs import get_env
    from cdd_oran.models import get_model

    cfg = replace(DEFAULT_CONFIG, device="cpu", model_kind="cdl")
    torch.manual_seed(0)
    env = get_env(cfg)
    model = get_model(cfg, env)  # random init, untrained -- we only check plumbing
    env.reset()
    state = state_to_tensor(env.get_state()).to("cpu")
    edges = detect_conflict_edges(env.true_adj_matrix, env)
    assert edges, "no conflict edges to test against"
    pi = edges[0]["param_id"]
    xapps = edges[0]["xapps_in_conflict"]
    weights = [1.0] * len(xapps)

    base = ModelBasedCEM(model=model, env=env, n_candidate=32, n_top=16, n_iter=3)
    h1 = RecedingHorizonCEM(
        model=model, env=env, n_horizon=1, n_candidate=32, n_top=16, n_iter=3
    )
    torch.manual_seed(123)
    a_base = base.act(state.clone(), pi, xapps, weights, 10)
    torch.manual_seed(123)
    a_h1 = h1.act(state.clone(), pi, xapps, weights, 10)
    assert a_h1 == a_base, f"H=1 must equal baseline: {a_h1} != {a_base}"

    h3 = RecedingHorizonCEM(
        model=model, env=env, n_horizon=3, n_candidate=32, n_top=16, n_iter=3
    )
    torch.manual_seed(7)
    a_h3 = h3.act(state.clone(), pi, xapps, weights, 10)
    assert a_h3[0] == pi
    assert 0 <= a_h3[1] <= env.action_space[1]
    assert 0 <= a_h3[2] <= env.action_space[pi + 2]
    print("horizon self-check OK:", {"baseline": a_base, "H1": a_h1, "H3": a_h3})


if __name__ == "__main__":
    _self_check()
