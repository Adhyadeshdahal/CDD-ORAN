"""Phase 3: robust planning over per-model RETURNS (shared path for all planners).

The Phase 2 world model can return an ensemble: ``predict_next_state`` gives a Normal with
mean/std shaped ``(m, batch, k)`` for ``m>1`` structure members, or ``(batch, k)`` for
``m=1``. This module scores each candidate action under EACH member and aggregates the
per-member RETURNS (costs) into one score. It NEVER averages transitions/next-states across
members (reviewer red flag / P2 spec risk 5) -- only the scalar returns are aggregated.

Cross-member disagreement is exposed as (a) an OPTIONAL additive planning penalty and (b) an
OOD warning flag. Both are OFF by default and reported even when the penalty weight is 0.

m=1 invariant: the ``sample.ndim == 2`` branch below reproduces the exact pre-Phase-3 call
sequence (one ``dist.sample()`` then one ``score_batch``), so the hard-mask ablation stays
byte-identical and the golden regression holds.
"""

from contextlib import contextmanager
from dataclasses import dataclass

import torch

from cdd_oran.planners.cost import score_batch


@contextmanager
def fixed_structures(model):
    """Structure lifecycle for ONE planner decision (review OPEN finding + P3 reconcile).

    On entry, draw ONE ensemble member set and cache it on the model; every
    ``predict_next_state`` inside the block reuses it, so all candidate actions and the full
    rollout within a single ``act`` call are scored under the IDENTICAL structures, and a
    FRESH set is drawn at the next ``act`` (decision step). Phase 3's per-member aggregation
    therefore runs over exactly this fixed member set.

    Scope of "no-op": a TRUE no-op only for hard-mask / single-model models (no
    ``sample_decision_structures`` or it returns None) -- there the m=1 path is byte-identical.
    For a structure_conditioned model with ``predict_members=1`` this is NOT a no-op: it draws
    ONE structure per decision instead of redrawing per prediction. That is the INTENDED,
    correct semantics (fair candidate comparison under one graph), not a regression."""
    drew = hasattr(model, "sample_decision_structures") and (
        model.sample_decision_structures() is not None
    )
    try:
        yield
    finally:
        if drew:
            model.clear_decision_structures()


@dataclass(frozen=True)
class EnsembleAggregator:
    """How to turn per-member returns (costs, lower = better) into one candidate score.

    ``method``:
      * ``"mean"``     -- risk-neutral expected cost.
      * ``"quantile"`` -- risk-AVERSE: an UPPER cost quantile (default q=0.9), i.e. a lower
                          quantile of the RETURN. This is the default (pessimistic).
      * ``"kappa"``    -- ``mean + kappa*std`` in cost space (mean - kappa*std of return);
                          reuse the planner's ``risk_kappa`` as ``kappa``.
    ``disagreement_penalty`` adds ``w * disagreement`` to the cost (0 = off). ``ood_threshold``
    raises an OOD flag when disagreement exceeds it (None = never).

    ``utility_weight`` (Lever 2 / Option A) is NOT an aggregation setting: it is the signed
    standardized-utility reward applied INSIDE each member's cost by ``score_batch``, and it
    therefore also bites on the m=1 path. It rides on this dataclass only because this is the
    one planner-knob carrier already threaded to every ``score_batch`` call site. 0.0 = off.
    """

    method: str = "quantile"
    quantile: float = 0.9
    kappa: float = 0.0
    disagreement_penalty: float = 0.0
    utility_weight: float = 0.0
    ood_threshold: float | None = None

    def combine(self, per_member: torch.Tensor) -> torch.Tensor:
        """(m, batch) per-member costs -> (batch,) aggregated cost."""
        if per_member.shape[0] == 1:
            return per_member[0]  # single member: exact passthrough (byte-identity)
        if self.method == "mean":
            return per_member.mean(dim=0)
        if self.method == "quantile":
            return torch.quantile(per_member, self.quantile, dim=0)
        if self.method == "kappa":
            return per_member.mean(dim=0) + self.kappa * per_member.std(dim=0)
        raise ValueError(f"unknown ensemble aggregator method: {self.method!r}")

    def ood(self, disagreement: torch.Tensor) -> bool:
        if self.ood_threshold is None:
            return False
        return bool(torch.as_tensor(disagreement).max().item() > self.ood_threshold)


def member_count(model) -> int:
    """Default ensemble size a model's ``predict_next_state`` returns (1 for a single-model
    world model such as the MLP baseline). Lets the multi-step rollout planners decide single
    vs per-member rollout WITHOUT an extra forward pass, so the m=1 path stays byte-identical."""
    return int(getattr(model, "predict_members", 1))


def member_disagreement(dist) -> torch.Tensor:
    """Epistemic cross-member disagreement per candidate: std over members of the predicted
    next-state MEAN, averaged over KPIs. Returns ``(batch,)``; zeros for m=1 (no ensemble).

    # ponytail: uses the predicted-mean spread (structure epistemic signal). An alternative
    # is the std of the per-member RETURNS; swap here if return-space disagreement is wanted.
    """
    mean = dist.mean
    if mean.ndim == 2:  # (batch, k) -> no member axis
        return torch.zeros(mean.shape[0], device=mean.device)
    return mean.std(dim=0).mean(dim=-1)  # (m, batch, k) -> (batch,)


def robust_returns_from_dist(dist, xapps, weights, scaling_term, device, kappa, aggregator):
    """Score a (possibly ensemble) prediction. Returns ``(scores (batch,), disagreement
    (batch,))``. For m=1 this is exactly ``score_batch(dist.sample(), ..., dist.stddev,
    kappa)`` with zero disagreement -- the pre-Phase-3 behaviour."""
    sample = dist.sample()
    std = dist.stddev
    if sample.ndim == 2:  # m = 1: unchanged single-model path
        costs = score_batch(
            sample, xapps, weights, scaling_term, device, std, kappa, aggregator.utility_weight
        )
        return costs, torch.zeros_like(costs)

    members = sample.shape[0]
    per_member = torch.stack(
        [
            score_batch(
                sample[j], xapps, weights, scaling_term, device, std[j], kappa,
                aggregator.utility_weight,
            )
            for j in range(members)
        ],
        dim=0,
    )  # (m, batch) -- per-member RETURNS, never averaged transitions
    combined = aggregator.combine(per_member)
    disagreement = member_disagreement(dist)
    if aggregator.disagreement_penalty:
        combined = combined + aggregator.disagreement_penalty * disagreement
    return combined, disagreement


def robust_score_batch(model, s_batch, action_batch, xapps, weights, scaling_term, device, kappa, aggregator):
    """Predict then score a batch of candidate actions robustly over ensemble members.

    Drop-in for the planners' ``dist = predict; next_kpis = dist.sample(); score_batch(...)``
    block. m=1 is byte-identical to that block."""
    dist = model.predict_next_state(s_batch, action_batch)
    return robust_returns_from_dist(dist, xapps, weights, scaling_term, device, kappa, aggregator)
