"""WG3 arbiter: budgeted per-region receding-horizon plan search, scored through a swappable world model.

Every D s (from ``start_s``): stage 1 = best of accept-all, freeze, incumbent, n_glob random network-wide plans (one
``score`` batch); stage 2 = one coordinate-descent pass over the regions in random order, n_loc random local
alternatives per region (one batch per region), keeping only exact improvements (mean J lower by > 1e-9). Between
epochs the held plan is applied (rollbacks only on the plan's first second). RNG keyed (seed, int(t), 13).
Hooks (default off):
  confidence(Score_candidate, Score_accept_all) -> bool: a plan other than accept-all may be picked only if True;
  prune(obs) -> set of (xapp, region): only those may be vetoed/modified (others forced to accept; a region's
    rollback flag survives only if the region has an allowed pair).
With ``TrueSimWM`` this is the privileged gate-B oracle (scratchpad/e6_dev/wg3_oracle.py), decision-for-decision.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from . import plans as P
from .world_model import DecisionContext, Score, WorldModel


class WG3Arbiter:
    def __init__(self, world_model: WorldModel, regions, lam_e: float, w_ll: float, D: int = 20, H: int = 90,
                 n_glob: int = 6, n_loc: int = 2, confidence: Callable[[Score, Score], bool] | None = None,
                 prune: Callable[[dict], set] | None = None, seed: int = 0, start_s: float = 0.0):
        """``regions`` = cell -> region map (e.g. ``adapters.e6.region_map(env)``)."""
        self.wm, self.site = world_model, np.asarray(regions)
        self.regions = sorted({int(x) for x in self.site})
        self.lam_e, self.w_ll, self.D, self.H = lam_e, w_ll, D, H
        self.n_glob, self.n_loc = n_glob, n_loc
        self.confidence, self.prune = confidence, prune
        self.seed, self.start_s = seed, start_s
        self.plan = P.accept_all(self.regions)
        self.until, self.picks = -1.0, {}
        self.rb_at = {}                   # knob -> change time we rolled back (never roll back our own rollback)
        self.half_state = {}              # knob -> single-quantum "half" requests seen (plans.half_step toggle)

    def _ok(self, s: Score, s_acc: Score, plan) -> bool:
        return self.confidence is None or P.is_accept_all(plan) or bool(self.confidence(s, s_acc))

    def choose(self, ctx: DecisionContext):
        r = np.random.default_rng([self.seed, int(ctx.obs["t"]), 13])
        allowed = self.prune(ctx.obs) if self.prune is not None else None
        fix = (lambda p: P.restrict(p, allowed, self.regions)) if allowed is not None else (lambda p: p)
        cands = [P.accept_all(self.regions), P.network(self.regions, P.uniform("reject")), dict(self.plan)]
        cands += [P.network(self.regions, P.random_region(r)) for _ in range(self.n_glob)]
        cands = [fix(c) for c in cands]
        sc = self.wm.score(ctx, cands)
        s_acc = sc[0]
        i_best = min((s.mean, i) for i, s in enumerate(sc) if i == 0 or self._ok(s, s_acc, cands[i]))[1]
        best, s_best, changed = dict(cands[i_best]), sc[i_best], 0
        for g in r.permutation(self.regions):
            trials = []
            for _ in range(self.n_loc):   # all trials of a region differ from best only at g -> batch is exact
                t = dict(best)
                t[int(g)] = P.random_region(r)
                trials.append(fix(t))
            for t, s in zip(trials, self.wm.score(ctx, trials), strict=True):
                if s.mean < s_best.mean - 1e-9 and self._ok(s, s_acc, t):
                    best, s_best, changed = t, s, changed + 1
        key = f"stage1={['accept', 'freeze', 'incumbent'][i_best] if i_best < 3 else 'random'}|local={min(changed, 5)}"
        self.picks[key] = self.picks.get(key, 0) + 1
        return best

    def act(self, obs: dict, last_change: dict, env=None) -> dict:
        """Decision dict for this second; re-plans at decision epochs. ``env`` only for privileged world models."""
        observe = getattr(self.wm, "observe", None)       # learned models track reports every second (serve parity)
        if observe is not None:
            observe(obs)
        first = False
        if obs["t"] >= self.start_s and obs["t"] >= self.until:
            ctx = DecisionContext(obs, self.site, self.regions, self.H, float(self.D), self.lam_e, self.w_ll,
                                  last_change, self.rb_at, env if getattr(self.wm, "privileged", False) else None,
                                  dict(self.half_state))
            self.plan, self.until, first = self.choose(ctx), obs["t"] + self.D, True
        return P.decide(self.plan, obs, self.site, self.D, last_change, self.rb_at, first, self.half_state)

    def record(self, dec: dict, last_change: dict) -> None:
        """After apply: remember which change each rollback created (so it is never rolled back itself)."""
        for k in dec["rollback"]:
            if k in last_change:
                self.rb_at[k] = last_change[k]


def run_episode(env, arbiter: WG3Arbiter) -> dict:
    """Two-phase loop: propose -> arbiter (held plan between epochs) -> apply. Returns env.score()."""
    while env.sec < env.total_s:
        obs = env.step_propose()
        dec = arbiter.act(obs, env.last_change, env)
        env.step_apply(dec)
        arbiter.record(dec, env.last_change)
    return env.score()
