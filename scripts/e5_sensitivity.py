"""E5 PRE-DECLARED SENSITIVITY SWEEPS (DEV, plan 015 §5; sol adversarial lane "circularity/rigging").

Disclosure that the E5 result is a bounded operating region, NOT a knife-edge reverse-engineered point.
Every sweep below is PRE-DECLARED here (grids fixed in code) and reports how each layer's load-bearing
metric moves as one construction knob varies, holding the others at the DEV design values. Run BEFORE any
confirmatory comparison; the confirmatory run uses a SEPARATE untouched seed set (see GATE_CONTRACT_E5.md).

Sweeps and the claim each guards:
  1. STRUCTURE (core): SHAP P0 relative importance for K_harm over occupancy x subdom x base-width. Shows
     BOTH occupancy AND amplitude-subdominance suppress SHAP (ratio<0.30 => SHAP-DAG prunes) — the miss is
     not attributable to occupancy alone (sol #5).
  2. DECISION-CRITICALITY: miss-direct-edge regret over subdom (edge amplitude) and over the K_harm
     threshold. Shows the band where the subdominant edge actually flips the action (too small => not
     decision-useful; threshold too high/low => not decision-critical).
  3. CONFOUND: oracle-gate in-stratum OLS OBS/DO slopes + difference-in-differences over theta (confound
     strength). Shows DiD->0 at theta=0 (negative control) and grows smoothly with |theta| — robust, not a
     fragile cancellation.
  4. SAMPLE SIZE: stratified OBS/DO slope stability over n (mean +/- range across seeds). Shows the
     identification result is not a small-sample artifact.
  5. CHAIN: miss-chain-edge H=2 regret over chain_gamma. Shows the mediated-path regret scales with the
     chain coefficient (and is zero at chain_gamma=0).

CORE otherwise; in-gate-conditional regret; mean-SCM surrogate. Nothing frozen. SHAP section is slow
(~1-2 min); pass --no-shap to skip it.
"""
from __future__ import annotations

import contextlib
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from cdd_oran.envs.v2.e5 import E5V2Env  # noqa: E402
from scripts.e5_confound import est_stratified_ingate, gen_corpus, regret_for_slope  # noqa: E402
from scripts.e5_spine import (  # noqa: E402
    CHAIN_EDGE,
    HARMFUL_EDGE,
    build_ingate_bank,
    e5_panel,
    spine_regret_e5,
    true_edges,
)

SUBDOM, CHAIN, THETA = 0.20, 0.20, -2.5
G1_LO, G1_HI = E5V2Env.id_ranges[E5V2Env.G1]
G2_LO, G2_HI = E5V2Env.id_ranges[E5V2Env.G2]
TAU1, C2, A_BASE, MU_BASE, B1 = E5V2Env.TAU1, E5V2Env.C2, E5V2Env.A_BASE, E5V2Env.MU_BASE, E5V2Env.B1


@contextlib.contextmanager
def class_attr(cls, name, value):
    old = getattr(cls, name)
    setattr(cls, name, value)
    try:
        yield
    finally:
        setattr(cls, name, old)


# ---- 1. STRUCTURE: SHAP P0 ratio over occupancy x subdom x base width (analytic core K_harm) ---------
def w_gate_for_occupancy(occ):
    p_g1 = (TAU1 - G1_LO) / (G1_HI - G1_LO)
    return occ / p_g1 * (G2_HI - G2_LO) / 2.0


def _shap_sweep(n=4000):
    from scripts.e2_baseline_shap_dag import fit_shap_importances
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.uniform(*E5V2Env.id_ranges[0], n), rng.uniform(G1_LO, G1_HI, n),
                         rng.uniform(G2_LO, G2_HI, n), rng.uniform(*E5V2Env.id_ranges[3], n)])
    p0, g1, g2 = X[:, 0], X[:, 1], X[:, 2]

    def kharm(subdom, w_gate, w_base):
        base = A_BASE * np.exp(-((g2 - MU_BASE) ** 2) / (2.0 * w_base ** 2)) + B1 * g1
        gate = (g1 <= TAU1) & (np.abs(g2 - C2) <= w_gate)
        return base + subdom * p0 * gate

    print("\n[1] STRUCTURE — SHAP P0 relative importance for K_harm (ratio<0.30 => SHAP-DAG PRUNES)")
    print("    occupancy x subdom (base width W_BASE=25 [design]):")
    print(f"    {'occ':>6} {'realocc':>8} | " + " ".join(f"sd={s:<4}" for s in (0.1, 0.2, 0.5, 1.0)))
    for occ in (0.05, 0.10, 0.20, 0.40):
        w = w_gate_for_occupancy(occ)
        real = ((g1 <= TAU1) & (np.abs(g2 - C2) <= w)).mean()
        row = [fit_shap_importances(X, kharm(sd, w, 25.0), seed=0)[0] for sd in (0.1, 0.2, 0.5, 1.0)]
        print(f"    {occ:>6} {real:>8.3f} | " + "  ".join(f"{imp[0]/imp.max():>6.3f}" for imp in row))
    print("    base width sweep (occ=0.05 [design], subdom=0.2): smaller W_BASE => less dominant => recovers")
    w = w_gate_for_occupancy(0.05)
    for wb in (15.0, 25.0, 40.0, 100.0):
        imp = fit_shap_importances(X, kharm(0.2, w, wb), seed=0)[0]
        print(f"      W_BASE={wb:>6}: P0 ratio {imp[0]/imp.max():.3f}")


# ---- 2. DECISION-CRITICALITY: miss-direct regret over subdom and threshold ---------------------------
def _decision_sweep(bank, panel):
    print("\n[2] DECISION-CRITICALITY — miss-direct-edge regret (H=1); oracle regret is 0 throughout")
    p_true, k_true = true_edges(E5V2Env(subdom=SUBDOM, chain_gamma=0.0))
    print("    subdom (edge amplitude):   ", end="")
    for sd in (0.05, 0.10, 0.20, 0.50):
        r = spine_regret_e5(p_true - {HARMFUL_EDGE}, k_true, bank, panel, subdom=sd, chain_gamma=0.0, H=1)
        print(f"sd={sd}:{r['mean_norm']:.3f} ", end="")
    print("\n    K_harm threshold:          ", end="")
    for thr in (75.0, 78.0, 80.0, 82.0, 85.0):
        with class_attr(E5V2Env, "kpi_thresholds", (55.0, thr, 0.0, 55.0)):
            panel_t = e5_panel()
            r = spine_regret_e5(p_true - {HARMFUL_EDGE}, k_true, bank, panel_t,
                                subdom=SUBDOM, chain_gamma=0.0, H=1)
        print(f"thr={thr:g}:{r['mean_norm']:.3f} ", end="")
    print()


# ---- 3. CONFOUND: stratified OBS/DO slope + DiD over theta -------------------------------------------
def _confound_sweep(bank, panel, n=6000):
    print("\n[3] CONFOUND — oracle-gate in-stratum OLS slope + difference-in-differences over theta")
    print(f"    {'theta':>6} {'OBSslope':>9} {'DOslope':>8} {'R_OBS':>7} {'R_DO':>6} {'DiD':>7}")
    Xd0, Yd0 = gen_corpus(n, E5V2Env.MODE_DO, 0, theta=0.0)
    Xo0, Yo0 = gen_corpus(n, E5V2Env.MODE_OBS, 0, theta=0.0)
    r_do0 = regret_for_slope(est_stratified_ingate(Xd0, Yd0), bank, panel)["norm"]
    r_ob0 = regret_for_slope(est_stratified_ingate(Xo0, Yo0), bank, panel)["norm"]
    base0 = r_ob0 - r_do0
    for th in (0.0, -0.5, -1.0, -2.5, -5.0):
        Xo, Yo = gen_corpus(n, E5V2Env.MODE_OBS, 0, theta=th)
        Xd, Yd = gen_corpus(n, E5V2Env.MODE_DO, 0, theta=th)
        a_o, a_d = est_stratified_ingate(Xo, Yo), est_stratified_ingate(Xd, Yd)
        r_o = regret_for_slope(a_o, bank, panel)["norm"]
        r_d = regret_for_slope(a_d, bank, panel)["norm"]
        print(f"    {th:>6} {a_o:>+9.3f} {a_d:>+8.3f} {r_o:>7.3f} {r_d:>6.3f} {r_o - r_d - base0:>+7.3f}")


# ---- 4. SAMPLE SIZE: stratified slope stability -----------------------------------------------------
def _samplesize_sweep():
    print("\n[4] SAMPLE SIZE — oracle-gate in-stratum OLS slope (mean +/- range over seeds 0-3)")
    print(f"    true +{SUBDOM}; {'n':>6} {'OBSslope(mean[min,max])':>28} {'DOslope(mean[min,max])':>28}")
    for n in (2000, 4000, 8000, 20000):
        obs, do = [], []
        for s in range(4):
            Xo, Yo = gen_corpus(n, E5V2Env.MODE_OBS, s, theta=THETA)
            Xd, Yd = gen_corpus(n, E5V2Env.MODE_DO, s, theta=THETA)
            obs.append(est_stratified_ingate(Xo, Yo)); do.append(est_stratified_ingate(Xd, Yd))
        print(f"    {n:>6} {np.mean(obs):>+9.3f}[{min(obs):+.3f},{max(obs):+.3f}]      "
              f"{np.mean(do):>+9.3f}[{min(do):+.3f},{max(do):+.3f}]")


# ---- 5. CHAIN: miss-chain H=2 regret over chain_gamma ------------------------------------------------
def _chain_sweep(panel):
    print("\n[5] CHAIN — miss-chain-edge regret at H=2 over chain_gamma (0 => no chain, regret 0)")
    print("    ", end="")
    for cg in (0.0, 0.10, 0.20, 0.50):
        bank = build_ingate_bank(subdom=SUBDOM, chain_gamma=cg, n=30)
        p_true, k_true = true_edges(E5V2Env(subdom=SUBDOM, chain_gamma=cg))
        miss = (p_true, frozenset(k_true) - {CHAIN_EDGE})
        r = spine_regret_e5(*miss, bank, panel, subdom=SUBDOM, chain_gamma=cg, H=2)
        print(f"cg={cg}:{r['mean_norm']:.3f} ", end="")
    print()


def main():
    do_shap = "--no-shap" not in sys.argv
    panel = e5_panel()
    bank = build_ingate_bank(subdom=SUBDOM, chain_gamma=0.0, n=30)
    print(f"E5 SENSITIVITY SWEEPS (DEV, pre-declared) — bank {len(bank)} in-gate states")
    if do_shap:
        _shap_sweep()
    _decision_sweep(bank, panel)
    _confound_sweep(bank, panel)
    _samplesize_sweep()
    _chain_sweep(panel)


if __name__ == "__main__":
    main()
