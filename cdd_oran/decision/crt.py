"""mscr-crt-v1: design-based conditional randomization test (CRT) for knob-family -> KPI-family links from the DEV
probe campaign (``probe.py``); SOL_DATA_DESIGN.md section b.

WHY: frozen MSCR-v2 draws its null by permuting ROWS; on E6 traces rows are serially dependent, spatially coupled and
fed back through the xApps, so those p-values (and the BY declarations built on them) are invalid there
(REPORT_H.md). This wrapper keeps MSCR's STATISTIC and replaces its null.

Hypothesis (family f, KPI family k, scope own|nbr): SHARP NULL of no effect of the assigned probe arm f (vs sham) on
the predeclared outcome window of block b: y_b = KPI over [t0, t0 + obs] minus KPI over the pre window
[t0 - pre_s, t0] (region = the treated region, or its neighbourhood for scope nbr). Rows = blocks whose arm is f or
sham and where both are eligible; every other block's assignment is held FIXED (conditioning event), outcomes are
held fixed (true under the null), and the candidate column is re-drawn from the LOGGED assignment mechanism
(``probe.AssignmentMechanism.conditional_draws``: P(arm = f | arm in {f, sham}, slot type, eligibility), then the
level). Applied values and KPI rows are NEVER shuffled. Because the schedule is predeclared from the seed and the
skeleton (slots, units, cap) does not depend on the arm, the conditional law of the assignment given the pre-assignment
history/strata is exactly this mechanism. p = (1 + #{T_draw >= T_obs}) / (B + 1) is then finite-sample valid under
the sharp null for ANY statistic T (it need not be calibrated itself).

Statistics (both recomputed in full on every draw):
  mscr    MSCR's S* = max over conditioners g of the pooled conditional correlation ratio of y on the candidate
          (x = level for f blocks, 0 for sham). The observed value is computed by REUSING ``mscr._strata_plan`` and
          ``mscr._s_star`` unchanged; the draws use a vectorised replica of ``_s_star`` (same stable-argsort
          equal-count binning, same strata), checked against ``_s_star`` on every run (``check_draws``). The conditioner
          search (max over g) is recomputed per draw. Constants are NOT MSCR's frozen ones (its min_stratum 40 x nc 6
          needs >= 240 rows; blocks are far fewer): ``CRTConfig`` nc=3, nb=3, min_stratum=10. Conditioners = the
          pre-window values of every KPI family (both scopes) + is_macro region + load + scenario + time in episode
          (all pre-assignment). Degenerate conditioners are dropped (``features.degenerate_columns``).
  signed  episode-cluster aggregate: T = |mean over episodes of mean_b x_b * (y_b - episode mean of y over the
          hypothesis rows)|; each episode weighs equally (the cluster is the episode, not the block).
Primary = mscr; signed is reported alongside.
Declarations: statuses "declared" (BY step-up at q ACROSS the supported hypotheses of one call; labelled
"BY-across-hypotheses (mscr-crt-v1)": NO FDR claim is inherited from MSCR-v2), "not_detected" (supported, not
declared: NOT evidence of absence), "undetermined" (unsupported: too few f / sham blocks or episodes, degenerate
outcome, or no conditioner stratum). Every result carries a strict ``claim`` string (``CLAIMS``); no output ever
says "no edge" / "absent". Graph weights stay soft priors. Row-permutation p-values (``discovery.py``) are never a
substitute for these.
Validity scope: the p-value is exact for the sharp null of no effect of the family's assignment ON THE WHOLE
TRAJECTORY (carryover into later windows is then also absent). Carryover does not break validity, but it blurs the
window-local estimand; ``lag=1`` tests the previous block's arm against this block's outcome/baseline (audit), and
``randomization="episode"`` (probe config) makes the episode the randomized unit (groups re-drawn together).
"""
from __future__ import annotations

import dataclasses

import numpy as np

from cdd_oran.discovery import mscr

from .features import degenerate_columns
from .probe import ARMS, FAMILIES, KPI_FAMILIES, AssignmentMechanism, ProbeConfig, block_outcomes

CRT_VERSION = "mscr-crt-v1"


@dataclasses.dataclass(frozen=True)
class CRTConfig:
    nc: int = 3
    nb: int = 3
    min_stratum: int = 10
    B: int = 999
    alpha: float = 0.05
    q: float = 0.05             # BY across hypotheses
    min_f: int = 20             # support rule: blocks with arm f
    min_sham: int = 20
    min_episodes: int = 5       # episodes contributing an f block

    def mscr_config(self) -> mscr.MSCRConfig:
        return mscr.MSCRConfig(nc=self.nc, nb=self.nb, min_stratum=self.min_stratum, n_perm=self.B, q=self.q)


@dataclasses.dataclass
class BlockData:
    """Pooled blocks of one or more probe episodes (one row per block)."""
    episode: np.ndarray        # (n,) int
    slot_type: np.ndarray      # (n,) int
    elig: np.ndarray           # (n, len(ARMS)) bool
    arm: np.ndarray            # (n,) int index into ARMS
    level: np.ndarray          # (n,) float
    prop: np.ndarray           # (n,) logged propensity
    y: dict                    # outcome name -> (n,) post - pre
    z: dict                    # conditioner name -> (n,) pre-assignment
    pcfg: ProbeConfig
    slot: np.ndarray | None = None   # (n,) slot index within the episode (lagged carryover test)

    @property
    def n(self) -> int:
        return len(self.arm)


def build_block_data(traces, schedule_from=None) -> BlockData:
    """From probe traces. ``schedule_from`` (list, same length): probe traces whose schedule is applied to
    ``traces`` (placebo: no-probe references evaluated under the probe schedule -> a true sharp null)."""
    rows = {k: [] for k in ("episode", "slot_type", "elig", "arm", "level", "prop", "slot")}
    ys, zs, pcfg = {}, {}, None
    for e, tr in enumerate(traces):
        src = schedule_from[e] if schedule_from is not None else tr
        a, pm = src.arrays, src.meta["probe"]
        if pm.get("replay") or tr.meta.get("probe", {}).get("replay"):
            raise ValueError("audit replay traces (schedule overrides) never enter inference")
        fields = {f.name for f in dataclasses.fields(ProbeConfig)}      # older probe versions: drop retired keys
        pc = ProbeConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in pm["config"].items() if k in fields})
        if pcfg is not None and pc != pcfg:
            raise ValueError("episodes with different probe configurations cannot be pooled")
        pcfg = pc
        oc = block_outcomes(tr, {"arrays": a, "meta": pm} if schedule_from is not None else None)
        n = len(a["blk_t0"])
        rows["episode"].append(np.full(n, e))
        rows["slot_type"].append(a["blk_type"].astype(int))
        rows["elig"].append(a["blk_elig"].astype(bool).reshape(n, len(ARMS)))
        rows["arm"].append(a["blk_arm"].astype(int))
        rows["level"].append(a["blk_level"].astype(float))
        rows["prop"].append(a["blk_prop"].astype(float))
        rows["slot"].append(a["blk_slot"].astype(int))
        for k in oc["post"]:
            ys.setdefault(k, []).append(oc["post"][k] - oc["pre"][k])
            zs.setdefault("pre_" + k, []).append(oc["pre"][k])
        cfg = tr.meta["cfg"]
        zs.setdefault("is_macro", []).append((a["blk_type"] == 1) | (a["blk_region"] < 7))
        zs.setdefault("load_high", []).append(np.full(n, float(cfg["load"] == "high")))
        scn = cfg.get("scenario", "base")
        zs.setdefault("scenario", []).append(np.full(n, float(("base", "surge", "mistune").index(scn))))
        span = float(pm["end_s"] - pm["start_s"]) or 1.0
        zs.setdefault("t_frac", []).append((a["blk_t0"] - pm["start_s"]) / span)
    cat = {k: np.concatenate(v) if v else np.zeros(0) for k, v in rows.items()}
    return BlockData(episode=cat["episode"].astype(int), slot_type=cat["slot_type"].astype(int),
                     elig=cat["elig"].reshape(-1, len(ARMS)).astype(bool), arm=cat["arm"].astype(int),
                     level=cat["level"].astype(float), prop=cat["prop"].astype(float),
                     y={k: np.concatenate(v).astype(float) for k, v in ys.items()},
                     z={k: np.concatenate(v).astype(float) for k, v in zs.items()}, pcfg=pcfg or ProbeConfig(),
                     slot=cat["slot"].astype(int))


# ------------------------------------------------------------------------------------------------ statistics
def _conditioners(z: dict, rows: np.ndarray) -> tuple[np.ndarray, list]:
    cols = {k: v[rows].copy() for k, v in z.items()}
    for v in cols.values():                        # NaN (no report) -> column median: fixed, pre-assignment
        bad = ~np.isfinite(v)
        if bad.any():
            v[bad] = np.nanmedian(v) if (~bad).any() else 0.0
    bad = degenerate_columns(cols)
    names = [k for k in cols if k not in bad]
    return (np.column_stack([cols[k] for k in names]) if names else np.zeros((len(rows), 0))), names


class MSCRStat:
    """MSCR S* of candidate x vs y given fixed conditioners Z. ``observed`` reuses frozen ``mscr`` functions;
    ``draws`` is the vectorised replica used for the null (checked against ``observed``)."""

    def __init__(self, Z: np.ndarray, y: np.ndarray, cfg: CRTConfig):
        self.mcfg = cfg.mscr_config()
        self.y = y
        plan = mscr._strata_plan(Z, self.mcfg) if Z.shape[1] else []
        # index 0 = the candidate slot (skipped by _s_star since g == i); conditioners are 1..m
        self.denom = np.zeros(len(plan) + 1)
        self.strata = [[]]
        self.flat = []                             # per conditioner: list of (rows, y_s, sizes, offsets)
        for g, kept in enumerate(plan):
            info, flat, tss = [], [], 0.0
            for rows, sizes, offsets in kept:
                y_s = y[rows]
                t_s = y_s.sum()
                tss += float(np.dot(y_s, y_s)) - t_s * t_s / len(y_s)
                info.append((rows, y_s))
                # equal_count_labels(x, nb) bins = consecutive runs of the stable sort order, sizes (i*nb)//ns
                flat.append((rows, y_s, sizes, offsets))
            self.denom[g + 1] = tss
            self.strata.append(info)
            self.flat.append(flat)
        self.valid = bool(np.any(self.denom[1:] > 0))

    def observed(self, x: np.ndarray) -> float:
        return float(mscr._s_star(np.asarray(x, float), 0, self.denom, self.strata, self.mcfg))

    def draws(self, X: np.ndarray) -> np.ndarray:
        """S* for each row of X (n_draws, n)."""
        best = np.full(X.shape[0], -np.inf)
        for g, flat in enumerate(self.flat):
            den = self.denom[g + 1]
            if den <= 0:
                continue
            bss = np.zeros(X.shape[0])
            for rows, y_s, sizes, offsets in flat:
                order = np.argsort(X[:, rows], axis=1, kind="stable")
                gs = np.add.reduceat(y_s[order], offsets, axis=1)
                bss += np.sum(gs ** 2 / sizes[None, :], axis=1) - y_s.sum() ** 2 / len(y_s)
            best = np.maximum(best, bss / den)
        return best


def signed_stat(X: np.ndarray, y: np.ndarray, episode: np.ndarray) -> np.ndarray:
    """|mean_e mean_{b in e} x_b (y_b - ybar_e)| for each row of X (n_draws, n) (or a single (n,) x)."""
    X2 = np.atleast_2d(X)
    eps = np.unique(episode)
    yc = y.copy()
    acc = np.zeros(X2.shape[0])
    for e in eps:
        m = episode == e
        yc[m] = y[m] - y[m].mean()
        acc += (X2[:, m] * yc[m][None, :]).mean(axis=1)
    out = np.abs(acc / len(eps))
    return out if np.ndim(X) == 2 else out[:1]


def check_draws(stat: MSCRStat, X: np.ndarray, k: int = 3) -> float:
    """Max |replica - mscr._s_star| over the first k draws (replica fidelity; should be ~1e-15)."""
    d = stat.draws(X[:k])
    return float(max((abs(d[i] - stat.observed(X[i])) for i in range(min(k, len(X)))
                      if np.isfinite(d[i])), default=0.0))


# ------------------------------------------------------------------------------------------------ test
CLAIMS = {
    "declared": "sharp null of no ASSIGNED-probe effect (incl. xApp reactions) on this KPI window rejected; "
                "BY across the tested hypotheses of this call (mscr-crt-v1), no MSCR FDR claim; soft prior only",
    "not_detected": "not detected at this support: this is NOT evidence that the edge is absent",
    "undetermined": "undetermined ({reason}): the edge is unknown, neither present nor absent",
}


def _prev_rows(data: BlockData) -> np.ndarray:
    """Row of the block in the previous slot of the same episode (-1 if none or not unique)."""
    prev = np.full(data.n, -1)
    if data.slot is None:
        return prev
    for i in range(data.n):
        m = np.nonzero((data.episode == data.episode[i]) & (data.slot == data.slot[i] - 1))[0]
        if len(m) == 1:
            prev[i] = m[0]
    return prev


def _groups(data: BlockData, idx: np.ndarray):
    """Arm-draw groups for the rows ``idx``: one per block (block randomization) or per (episode, slot type)."""
    if data.pcfg.randomization == "episode":
        return data.episode[idx] * 2 + data.slot_type[idx]
    return idx


def crt_test(data: BlockData, family: str, outcome: str, cfg: CRTConfig | None = None, seed: int = 0,
             lag: int = 0) -> dict:
    """One hypothesis (family -> outcome). ``lag=0``: the block's own arm vs its own outcome window. ``lag=1``
    (CARRYOVER AUDIT): the arm of the block in the PREVIOUS slot vs this block's outcome (``outcome`` may be a y
    name or a pre-window conditioner name, e.g. ``pre_own_prb_util`` = contamination of the next baseline);
    conditioners are then the previous block's pre-assignment values. Returns p_mscr, p_signed, statistics,
    support and status "tested" | "undetermined" (+ reason, claim)."""
    cfg = cfg or CRTConfig()
    mech = AssignmentMechanism(data.pcfg)
    fi, si = ARMS.index(family), ARMS.index("sham")
    y_all = data.y[outcome] if outcome in data.y else data.z[outcome]
    if lag == 0:
        src = np.arange(data.n)
    elif lag == 1:
        src = _prev_rows(data)
    else:
        raise ValueError("lag must be 0 or 1")
    ok = src >= 0
    s_ = np.where(ok, src, 0)
    rows = np.nonzero(ok & ((data.arm[s_] == fi) | (data.arm[s_] == si)) & data.elig[s_, fi] & data.elig[s_, si]
                      & np.isfinite(y_all))[0]
    srows = src[rows]                                   # the assignment rows re-sampled
    n_f = int((data.arm[srows] == fi).sum())
    n_s = len(rows) - n_f
    n_ep = len(np.unique(data.episode[srows][data.arm[srows] == fi]))
    res = {"version": CRT_VERSION, "family": family, "outcome": outcome, "lag": lag, "n_rows": len(rows),
           "n_f": n_f, "n_sham": n_s, "n_episodes_f": n_ep, "randomization": data.pcfg.randomization,
           "p_mscr": np.nan, "p_signed": np.nan, "s_obs": np.nan, "t_obs": np.nan, "B": cfg.B,
           "status": "undetermined", "reason": ""}

    def undetermined(reason):
        res["reason"] = reason
        res["claim"] = CLAIMS["undetermined"].format(reason=reason)
        return res

    if n_f < cfg.min_f or n_s < cfg.min_sham or n_ep < cfg.min_episodes:
        return undetermined("support")
    y = y_all[rows]
    if degenerate_columns({"y": y}):
        return undetermined("degenerate_outcome")
    Z, names = _conditioners(data.z, srows)
    stat = MSCRStat(Z, y, cfg)
    if not stat.valid:
        return undetermined("no_conditioner_stratum")
    x = np.where(data.arm[srows] == fi, data.level[srows], 0.0)
    rng = np.random.default_rng([int(seed), 5151, fi, sum(map(ord, outcome)), lag])
    uniq, inv = np.unique(srows, return_inverse=True)   # a previous block may precede several rows
    Xu = mech.conditional_draws(rng, data.slot_type[uniq], data.elig[uniq], family, cfg.B,
                                groups=_groups(data, uniq))
    X = Xu[:, inv]
    s_obs_frozen = stat.observed(x)
    s_obs = float(stat.draws(x[None, :])[0])
    s_null = stat.draws(X)
    ep = data.episode[rows]
    t_obs = float(signed_stat(x, y, ep)[0])
    t_null = signed_stat(X, y, ep)
    tol = 1e-12 * (1.0 + abs(s_obs))
    res.update({"p_mscr": (1.0 + np.count_nonzero(s_null >= s_obs - tol)) / (cfg.B + 1.0),
                "p_signed": (1.0 + np.count_nonzero(t_null >= t_obs * (1 - 1e-12))) / (cfg.B + 1.0),
                "s_obs": s_obs, "s_obs_mscr_reuse": s_obs_frozen, "replica_err": max(abs(s_obs - s_obs_frozen),
                                                                                       check_draws(stat, X)),
                "t_obs": t_obs, "conditioners": names, "status": "tested"})
    return res


def run_crt(data: BlockData, families=FAMILIES, outcomes=None, cfg: CRTConfig | None = None, seed: int = 0,
            lag: int = 0) -> dict:
    """All (family, outcome) hypotheses + BY across the tested ones (labelled; not MSCR's FDR claim).
    Statuses: declared | not_detected | undetermined; each result carries its strict ``claim`` wording."""
    cfg = cfg or CRTConfig()
    outcomes = outcomes or [f"own_{k}" for k in KPI_FAMILIES]
    res = [crt_test(data, f, o, cfg, seed, lag) for f in families for o in outcomes]
    tested = [r for r in res if r["status"] == "tested"]
    if tested:
        dec = mscr.by_declare([r["p_mscr"] for r in tested], cfg.q)
        for r, d in zip(tested, dec, strict=True):
            r["status"] = "declared" if d else "not_detected"
            r["claim"] = CLAIMS[r["status"]]
    first_stage = {f: {"blocks": int((data.arm == ARMS.index(f)).sum()),
                       "episodes": int(len(np.unique(data.episode[data.arm == ARMS.index(f)])))}
                   for f in (*families, "sham")}
    return {"version": CRT_VERSION, "selection": f"BY-across-hypotheses (mscr-crt-v1), q={cfg.q:g}; NO MSCR FDR claim",
            "wording": "declared = assigned-probe association only; not_detected and undetermined never mean "
                       "'no edge'", "config": dataclasses.asdict(cfg), "first_stage": first_stage, "results": res}


def placebo_rejection(data: BlockData, n_rep: int = 50, families=FAMILIES, outcomes=None,
                      cfg: CRTConfig | None = None, seed: int = 0) -> dict:
    """SHAM-PLACEBO rejection on REAL outcomes: ``data`` must come from no-probe references evaluated under probe
    schedules (``build_block_data(refs, schedule_from=probes)``: a true sharp null). Each replicate re-draws every
    block's arm from the mechanism (fresh placebo assignment, same skeleton) and runs every hypothesis; returns the
    rejection rate of p_mscr / p_signed at alpha over (replicate, hypothesis) pairs that were testable."""
    cfg = cfg or CRTConfig()
    outcomes = outcomes or [f"own_{k}" for k in KPI_FAMILIES]
    mech = AssignmentMechanism(data.pcfg)
    rej_m, rej_s, n = 0, 0, 0
    for r in range(n_rep):
        rng = np.random.default_rng([int(seed), 7373, r])
        arm, level = data.arm.copy(), data.level.copy()
        drawn = {}
        for i in range(data.n):
            key = (data.episode[i], data.slot_type[i]) if data.pcfg.randomization == "episode" else i
            if key not in drawn:
                drawn[key] = mech.sample(rng, data.slot_type[i], data.elig[i])[0]
            a = drawn[key]
            lv = mech.levels(a)
            arm[i], level[i] = ARMS.index(a), lv[int(rng.integers(len(lv)))] if len(lv) > 1 else lv[0]
        d = dataclasses.replace(data, arm=arm, level=level)
        for f in families:
            for o in outcomes:
                t = crt_test(d, f, o, cfg, seed=r)
                if t["status"] != "tested":
                    continue
                n += 1
                rej_m += t["p_mscr"] <= cfg.alpha
                rej_s += t["p_signed"] <= cfg.alpha
    return {"n_tests": n, "rej_mscr": rej_m / n if n else np.nan, "rej_signed": rej_s / n if n else np.nan,
            "alpha": cfg.alpha, "n_rep": n_rep}


# ------------------------------------------------------------------------------------------------ calibration
def synthetic_block_data(rng: np.random.Generator, n_ep: int = 40, effect: float = 0.0, family: str = "hys",
                         gated: bool = False, nuisance: float = 1.0, pcfg: ProbeConfig | None = None,
                         carry: float = 0.0) -> BlockData:
    """Synthetic data with the probe design's block/assignment structure: per episode a random skeleton of type-A/B
    slots (1 unit each), arms from the real ``AssignmentMechanism``; KPI outcomes with an episode random effect, AR(1)
    persistence across blocks, heavy (t3) tails, a pre-window correlated with the post window, and a NUISANCE effect of
    another family (cio -> y0; carrier -> y1) so the conditional null (other arms fixed) is exercised. ``effect``
    adds effect * level to y0 for ``family`` blocks (``gated``: only when the pre-window of y1 is above 0); ``carry``
    adds carry * (previous block's level if its arm is ``family``) to the NEXT block's y0 (carryover)."""
    pcfg = pcfg or ProbeConfig()
    mech = AssignmentMechanism(pcfg)
    rows = {k: [] for k in ("episode", "slot_type", "elig", "arm", "level", "prop", "slot")}
    y = {"y0": [], "y1": []}
    z = {"pre_y0": [], "pre_y1": [], "is_macro": [], "t_frac": []}
    for e in range(n_ep):
        u = rng.normal(size=2)
        n_b = int(rng.integers(3, 7))
        state = rng.normal(size=2)
        prev_x, ep_arm = 0.0, {}
        for b in range(n_b):
            typ = int(rng.uniform() < pcfg.p_slot_b)
            macro = typ == 1 or rng.uniform() < 0.7
            arms_ok = pcfg.arms_b if typ else pcfg.arms_a
            elig = np.array([a in arms_ok and (a != "carrier" or macro) for a in ARMS])
            if pcfg.randomization == "episode":           # one arm per (episode, slot type); carrier-eligible
                elig = np.array([a in arms_ok for a in ARMS])
                ep_arm.setdefault(typ, mech.sample(rng, typ, elig)[0])
                arm = ep_arm[typ]
                lv = mech.levels(arm)
                level = float(lv[int(rng.integers(len(lv)))]) if len(lv) > 1 else lv[0]
            else:
                arm, level = mech.sample(rng, typ, elig)
            state = 0.8 * state + 0.6 * rng.standard_t(3, size=2)
            pre = u + state
            post = u + 0.8 * state + 0.5 * rng.standard_t(3, size=2) + 0.3 * macro
            if arm == "cio":
                post[0] += nuisance * level
            if arm == "carrier":
                post[1] += nuisance
            if arm == family and (not gated or pre[1] > 0):
                post[0] += effect * level
            post[0] += carry * prev_x
            prev_x = level if arm == family else 0.0
            for k, v in (("episode", e), ("slot_type", typ), ("elig", elig), ("arm", ARMS.index(arm)),
                         ("level", level), ("prop", mech.prob(typ, elig, arm, level)), ("slot", b)):
                rows[k].append(v)
            y["y0"].append(post[0] - pre[0])
            y["y1"].append(post[1] - pre[1])
            z["pre_y0"].append(pre[0])
            z["pre_y1"].append(pre[1])
            z["is_macro"].append(float(macro))
            z["t_frac"].append(b / n_b)
    return BlockData(episode=np.array(rows["episode"]), slot_type=np.array(rows["slot_type"]),
                     elig=np.array(rows["elig"], bool), arm=np.array(rows["arm"]), level=np.array(rows["level"]),
                     prop=np.array(rows["prop"]), y={k: np.array(v) for k, v in y.items()},
                     z={k: np.array(v, float) for k, v in z.items()}, pcfg=pcfg, slot=np.array(rows["slot"]))


def calibrate(n_rep: int = 200, effects=(0.0, 0.5, 1.0), family: str = "hys", gated: bool = False,
              n_ep: int = 40, cfg: CRTConfig | None = None, seed: int = 0) -> dict:
    """Rejection rates at cfg.alpha of p_mscr / p_signed over ``n_rep`` synthetic datasets per effect size
    (effect 0 = type-I error; the nuisance cio effect on y0 is present throughout)."""
    cfg = cfg or CRTConfig(B=199, min_f=5, min_sham=5, min_episodes=3)
    out = {}
    for eff in effects:
        rm, rs, n_t = 0, 0, 0
        for r in range(n_rep):
            rng = np.random.default_rng([int(seed), 6262, r, int(round(eff * 1000))])
            d = synthetic_block_data(rng, n_ep=n_ep, effect=eff, family=family, gated=gated)
            t = crt_test(d, family, "y0", cfg, seed=r)
            if t["status"] != "tested":
                continue
            n_t += 1
            rm += t["p_mscr"] <= cfg.alpha
            rs += t["p_signed"] <= cfg.alpha
        out[eff] = {"n": n_t, "rej_mscr": rm / max(n_t, 1), "rej_signed": rs / max(n_t, 1)}
    return out
