"""World-model protocol for the WG3 arbiter + the privileged true-simulator reference.

``score(ctx, plans) -> [Score]``: objective (lower is better) of holding each plan over the next ``ctx.H`` s,
J = violated UE-s + w_ll * LL-violated UE-s + lam_e * kWh, as a mean and a std (spread of the model's belief).
Batch interface so learned models can vectorise over candidates.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .plans import Plan, decide


@dataclass(frozen=True)
class Score:
    mean: float
    std: float = 0.0


@dataclass
class DecisionContext:
    obs: Mapping                      # the second's obs (requests still pending on the env)
    site: Any                         # cell -> region map
    regions: Sequence[int]
    H: int                            # horizon, control seconds
    D: float                          # plan hold / lock duration, s
    lam_e: float                      # price of energy per kWh (in violated UE-s)
    w_ll: float                       # extra weight of LL-violated UE-s
    last_change: Mapping = field(default_factory=dict)   # knob -> time of its latest applied change
    rb_at: Mapping = field(default_factory=dict)         # knob -> change time we rolled back (no rollback of those)
    env: Any = None                   # PRIVILEGED env handle between step_propose and step_apply (TrueSimWM only)


@runtime_checkable
class WorldModel(Protocol):
    privileged: bool

    def score(self, ctx: DecisionContext, plans: Sequence[Plan]) -> list[Score]: ...


def objective(sla0: Mapping, sla1: Mapping, lam_e: float, w_ll: float) -> float:
    d = {k: sla1[k] - sla0[k] for k in ("viol_ue_s", "ll_viol", "energy_j")}
    return d["viol_ue_s"] + w_ll * d["ll_viol"] + lam_e * d["energy_j"] / 3.6e6


class TrueSimWM:
    """Privileged: env.copy() between the phases, replay the plan for H s on the true tape (std = 0). NOT deployable."""
    privileged = True

    def __init__(self):
        self.n_roll = 0

    def _rollout(self, ctx: DecisionContext, plan: Plan) -> float:
        sim = ctx.env.copy()
        sla0 = dict(sim.plant.sla)
        sim.step_apply(decide(plan, ctx.obs, ctx.site, ctx.D, sim.last_change, ctx.rb_at, True))
        for _ in range(ctx.H - 1):
            if sim.sec >= sim.total_s:
                break
            o = sim.step_propose()
            sim.step_apply(decide(plan, o, ctx.site, ctx.D, sim.last_change, ctx.rb_at, False))
        self.n_roll += 1
        return objective(sla0, sim.plant.sla, ctx.lam_e, ctx.w_ll)

    def score(self, ctx: DecisionContext, plans: Sequence[Plan]) -> list[Score]:
        if ctx.env is None:
            raise ValueError("TrueSimWM needs the privileged env handle (ctx.env)")
        return [Score(self._rollout(ctx, p)) for p in plans]
