"""Joint multi-NCP resolution planner (Task B of brief S3).

The per-edge planners resolve one conflict (one NCP) at a time. This planner
optimises SEVERAL conflicting NCPs at once via CEM over a concatenated action
vector: each candidate is a (K, 2) block of (bin, index) pairs, one per NCP.

Coupling is captured by applying the K NCPs one after another through the
one-step world model, feeding predicted KPIs forward between applications, so a
later NCP sees the KPI shifts caused by earlier ones. The final KPI vector is
scored by a single joint cost over ALL affected xApps.

`act` (the Planner interface) is the single-NCP fallback used by the standard
per-edge evaluation loop; `act_joint` is the multi-NCP entry point.
"""

import torch

from cdd_oran.planners.base import Planner
from cdd_oran.planners.cost import score_batch


class JointMultiNCPPlanner(Planner):
    def __init__(self, model, env, n_candidate=256, n_top=64, n_iter=10, risk_kappa: float = 0.0):
        self.risk_kappa = risk_kappa
        self.model = model
        self.env = env
        self.action_space = env.action_space
        self.num_bins = env.num_bins
        self.device = model.device
        self.n_candidate = n_candidate
        self.n_top = n_top
        self.n_iter = n_iter
        self.name = "JointMultiNCP"

    def act(
        self,
        current_state,
        conflict_param_index,
        xapps_under_conflict,
        weights_per_xapps,
        scaling_term,
    ):
        actions = self.act_joint(
            current_state,
            [conflict_param_index],
            xapps_under_conflict,
            weights_per_xapps,
            scaling_term,
        )
        return actions[0]

    def act_joint(
        self,
        current_state,
        param_ids,
        xapps_under_conflict,
        weights_per_xapps,
        scaling_term,
    ):
        s0 = current_state.to(self.device).float()
        k_count = len(param_ids)
        bin_hi = self.action_space[1]
        idx_hi = [self.action_space[p + 2] for p in param_ids]

        hi = torch.tensor(
            [[bin_hi, idx_hi[k]] for k in range(k_count)],
            dtype=torch.float32,
            device=self.device,
        )
        lo = torch.zeros(k_count, 2, device=self.device)
        mu = (lo + hi) / 2.0
        std = (hi - lo) / 6.0

        for _ in range(self.n_iter):
            noise = torch.randn(self.n_candidate, k_count, 2, device=self.device)
            samples = (mu + std * noise).round().long()
            samples[..., 0].clamp_(0, bin_hi)
            for k in range(k_count):
                samples[:, k, 1].clamp_(0, idx_hi[k])
            cost = self._joint_cost(
                s0, param_ids, samples, xapps_under_conflict, weights_per_xapps, scaling_term
            )
            elites = samples[torch.argsort(cost)[: self.n_top]].float()
            mu = elites.mean(dim=0)
            std = elites.std(dim=0).clamp(min=1.0)

        out = []
        for k, pi in enumerate(param_ids):
            best_bin = mu[k, 0].round().long().clamp(0, bin_hi).item()
            best_idx = mu[k, 1].round().long().clamp(0, idx_hi[k]).item()
            out.append([pi, best_bin, best_idx])
        return out

    def _joint_cost(self, s0, param_ids, samples, xapps, weights, scaling_term):
        n = samples.shape[0]
        num_params = self.env.num_params
        # Predict from the UNMODIFIED s0 + the joint action; between NCP applications
        # feed only predicted KPIs forward. Do NOT overwrite the param slots with the
        # applied NCP value -- CDL is trained on (pre-action state, action), so that
        # double-counts the action and is OOD.
        state = s0.unsqueeze(0).expand(n, -1).clone()
        next_kpis = None
        next_stds = None
        for k, pi in enumerate(param_ids):
            pi_col = torch.full((n, 1), pi, dtype=torch.float32, device=self.device)
            action_batch = torch.cat([pi_col, samples[:, k, :].float()], dim=1)
            dist = self.model.predict_next_state(state, action_batch)
            next_kpis = dist.sample()
            next_stds = dist.stddev
            state = state.clone()
            state[:, num_params:] = next_kpis  # feed forward -> next NCP sees the shift
        # Risk-depth limit: only the FINAL hop's std is risk-adjusted below.
        # Intermediate NCP hops feed unadjusted sampled KPIs forward, so joint
        # pessimism reflects last-hop aleatoric std only.
        return score_batch(
            next_kpis, xapps, weights, scaling_term, self.device, next_stds, self.risk_kappa
        )


def _self_check():
    from dataclasses import replace

    from cdd_oran.config import DEFAULT_CONFIG
    from cdd_oran.conflicts import state_to_tensor
    from cdd_oran.envs import get_env
    from cdd_oran.models import get_model

    cfg = replace(DEFAULT_CONFIG, device="cpu", model_kind="cdl")
    torch.manual_seed(0)
    env = get_env(cfg)
    model = get_model(cfg, env)  # random init, untrained -- plumbing check only
    env.reset()
    state = state_to_tensor(env.get_state()).to("cpu")

    planner = JointMultiNCPPlanner(model=model, env=env, n_candidate=32, n_top=16, n_iter=3)
    param_ids = list(range(min(2, env.num_params)))
    xapps = env.xapps
    weights = [1.0] * len(xapps)

    torch.manual_seed(1)
    actions = planner.act_joint(state.clone(), param_ids, xapps, weights, 10)
    assert len(actions) == len(param_ids), "one action per NCP expected"
    for (pi, b, v) in actions:
        assert pi in param_ids
        assert 0 <= b <= env.action_space[1], f"bin {b} out of bounds"
        assert 0 <= v <= env.action_space[pi + 2], f"index {v} out of bounds for p{pi}"

    # single-NCP fallback is a valid Planner action
    single = planner.act(state.clone(), param_ids[0], xapps, weights, 10)
    assert single[0] == param_ids[0]
    print("joint self-check OK:", {"joint": actions, "single_fallback": single})


if __name__ == "__main__":
    _self_check()
