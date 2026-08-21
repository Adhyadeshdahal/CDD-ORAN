from collections.abc import Sequence

import numpy as np
import torch

from cdd_oran.envs.base import XApp


def risk_adjust(kpis, stds, direction: int, kappa: float):
    """Pessimistic (lower-confidence-bound) shift of a KPI vector.

    The CDL world model emits Normal(mu, std) per KPI. ``kpis`` is a point draw
    (or mu); ``stds`` is the matching std. A conflicting xApp that MAXIMISES its
    utility (direction 0) is pessimistic about a LOW KPI, so subtract kappa*std;
    an xApp that MINIMISES (direction 1) is pessimistic about a HIGH KPI, so add
    kappa*std. This is the closed-form Gaussian LCB / CVaR view of risk.

    kappa == 0.0 (or stds is None) returns ``kpis`` unchanged, so the planner
    keeps its exact current behaviour -- the regression-safety guarantee.
    """
    if kappa == 0.0 or stds is None:
        return kpis
    # Sign assumes one KPI direction per xApp (true for all current envs); a mixed
    # xApp would need a per-KPI sign vector here instead of a single scalar.
    sign = -1.0 if direction == 0 else 1.0
    return kpis + sign * kappa * stds


def weighted_distance(xapp: XApp, utility: float) -> tuple[float, int]:
    mean, std = xapp.mean, xapp.std
    norm_threshold = (xapp.threshold - mean) / std

    if xapp.direction == 0:
        if utility < norm_threshold:
            return norm_threshold - utility, 0
        return 0.0, 1
    if utility > norm_threshold:
        return utility - norm_threshold, 0
    return 0.0, 1


def _xapp_kpi_indices(xapp: XApp, width: int) -> list[int]:
    """Recover the KPI column indices an xApp's utility reads, by probing it.

    Env xApps are affine in the KPI vector (see base.py ``_utility_for_indices``:
    ``(sum(kpis[i] for i in indices)/len - mean)/std``). A unit probe on column j
    moves the utility iff j is one of the xApp's KPI indices. This lets score_batch
    vectorise over candidates without XApp exposing the indices its utility closure
    captured -- and without re-deriving the utility (which would risk rounding drift):
    the batched path below re-uses the SAME ``sum/len`` then ``(-mean)/std`` ops.
    """
    base = xapp.compute_utility(np.zeros(width, dtype=np.float32))
    indices: list[int] = []
    for j in range(width):
        probe = np.zeros(width, dtype=np.float32)
        probe[j] = 1.0
        if xapp.compute_utility(probe) != base:
            indices.append(j)
    return indices


def score_batch(
    next_kpis_batch: torch.Tensor,
    xapps: Sequence[XApp],
    weights: Sequence[float],
    scaling_term: float,
    device: torch.device | str,
    stds_batch: torch.Tensor | None = None,
    kappa: float = 0.0,
) -> torch.Tensor:
    # Vectorised over candidates (the ~5min/run hot loop). Kept in numpy on purpose:
    # the original promotes float32 KPI sums to float64 via `float32 array - python
    # float`; torch keeps float32 on that op, which flips argmins on the 2-KPI xApp
    # and every utility-mean subtraction. numpy is the same engine as the old loop,
    # so results are bit-identical. Parity is asserted in _self_check against the
    # reference loop below.
    count, width = next_kpis_batch.shape
    kpis_np = next_kpis_batch.cpu().detach().numpy()
    stds_np = (
        stds_batch.cpu().detach().numpy() if (stds_batch is not None and kappa != 0.0) else None
    )
    cost_matrix = np.zeros((count, len(xapps)))
    sat_matrix = np.zeros((count, len(xapps)))
    for xapp_index, xapp in enumerate(xapps):
        indices = _xapp_kpi_indices(xapp, width)
        mean, std = xapp.mean, xapp.std
        norm_threshold = (xapp.threshold - mean) / std
        adj = risk_adjust(kpis_np, stds_np, xapp.direction, kappa)  # (count, width)
        value = adj[:, indices].sum(axis=1) / len(indices)          # matches sum/len order
        utility = (value - mean) / std                             # float32 -> float64, as before
        if xapp.direction == 0:
            unsat = utility < norm_threshold
            distance = np.where(unsat, norm_threshold - utility, 0.0)
        else:
            unsat = utility > norm_threshold
            distance = np.where(unsat, utility - norm_threshold, 0.0)
        cost_matrix[:, xapp_index] = weights[xapp_index] * distance * scaling_term
        sat_matrix[:, xapp_index] = np.where(unsat, 0.0, 1.0)
    costs = cost_matrix.sum(axis=1) - (sat_matrix.sum(axis=1)) ** 2
    return torch.tensor(costs, dtype=torch.float32, device=device)


def _score_batch_reference(
    next_kpis_batch: torch.Tensor,
    xapps: Sequence[XApp],
    weights: Sequence[float],
    scaling_term: float,
    device: torch.device | str,
    stds_batch: torch.Tensor | None = None,
    kappa: float = 0.0,
) -> torch.Tensor:
    """Original per-candidate loop, kept only as the parity oracle for _self_check."""
    count = next_kpis_batch.shape[0]
    kpis_np = next_kpis_batch.cpu().detach().numpy()
    stds_np = (
        stds_batch.cpu().detach().numpy() if (stds_batch is not None and kappa != 0.0) else None
    )
    costs = np.zeros(count)
    for index in range(count):
        kpis = kpis_np[index]
        stds = stds_np[index] if stds_np is not None else None
        cost_vec = np.zeros(len(xapps))
        sat_vec = np.zeros(len(xapps))
        for xapp_index, xapp in enumerate(xapps):
            utility = xapp.compute_utility(risk_adjust(kpis, stds, xapp.direction, kappa))
            distance, satisfied = weighted_distance(xapp, utility)
            cost_vec[xapp_index] = weights[xapp_index] * distance * scaling_term
            sat_vec[xapp_index] = satisfied
        costs[index] = cost_vec.sum() - (sat_vec.sum()) ** 2
    return torch.tensor(costs, dtype=torch.float32, device=device)


def _self_check() -> None:
    """Risk knob: kappa=0 is a no-op; kappa>0 is pessimistic per xApp direction."""
    from types import SimpleNamespace

    mu = np.array([0.5, -0.5])
    std = np.array([1.0, 2.0])

    # risk_adjust: kappa=0 identity; direction 0 subtracts, direction 1 adds kappa*std.
    assert np.array_equal(risk_adjust(mu, std, 0, 0.0), mu)
    assert np.array_equal(risk_adjust(mu, None, 0, 3.0), mu)  # no std -> no-op
    assert np.allclose(risk_adjust(mu, std, 0, 2.0), mu - 2.0 * std)  # maximiser: worse-low
    assert np.allclose(risk_adjust(mu, std, 1, 2.0), mu + 2.0 * std)  # minimiser: worse-high

    # score_batch: single candidate, single KPI, at the (normalised) threshold.
    # mean=0,std=1,threshold=0 -> norm_threshold=0; utility = KPI[0].
    def make(direction):
        return SimpleNamespace(
            direction=direction, threshold=0.0, mean=0.0, std=1.0,
            compute_utility=lambda k: float(k[0]),
        )

    kpi = torch.tensor([[0.0]])       # exactly at threshold -> satisfied
    sig = torch.tensor([[1.0]])
    dev = "cpu"

    for direction in (0, 1):
        xapps = [make(direction)]
        base = score_batch(kpi, xapps, [1.0], 1.0, dev, sig, 0.0).item()
        # kappa=0 -> satisfied (distance 0, sat 1): cost = 0 - 1 = -1.
        assert base == -1.0, f"kappa=0 baseline changed: {base}"
        risky = score_batch(kpi, xapps, [1.0], 1.0, dev, sig, 2.0).item()
        # kappa=2 shifts the effective KPI by 2 std into the unsatisfied region:
        # distance 2, sat 0 -> cost = 2. Pessimism raises the cost.
        assert risky == 2.0, f"dir {direction}: expected 2.0, got {risky}"
        assert risky > base

    # Parity: vectorised score_batch must EQUAL the reference loop -- both allclose
    # (bit-identical here) and, the real gate, identical argmin over the candidates.
    from cdd_oran.envs.env_ii import ORANEnvironment2  # 2-KPI xApp (3,4) exercises float32 sums

    env = ORANEnvironment2()
    xapps = env.xapps
    weights_ref = [1.0, 2.0, 0.5, 1.5, 1.0][: len(xapps)]
    rng = np.random.default_rng(0)
    kpis = torch.tensor(rng.standard_normal((2000, env.num_kpis)), dtype=torch.float32)
    stds = torch.tensor(rng.uniform(0.1, 2.0, (2000, env.num_kpis)), dtype=torch.float32)
    for kap in (0.0, 0.7):
        new = score_batch(kpis, xapps, weights_ref, 1.3, "cpu", stds, kap)
        old = _score_batch_reference(kpis, xapps, weights_ref, 1.3, "cpu", stds, kap)
        assert torch.allclose(new, old, atol=0.0, rtol=0.0), f"kappa={kap}: values differ"
        assert torch.equal(new.argmin(), old.argmin()), f"kappa={kap}: argmin flipped"

    print("cost self-check OK")


if __name__ == "__main__":
    _self_check()
