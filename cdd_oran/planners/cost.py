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


def score_batch(
    next_kpis_batch: torch.Tensor,
    xapps: Sequence[XApp],
    weights: Sequence[float],
    scaling_term: float,
    device: torch.device | str,
    stds_batch: torch.Tensor | None = None,
    kappa: float = 0.0,
) -> torch.Tensor:
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

    print("cost self-check OK")


if __name__ == "__main__":
    _self_check()
