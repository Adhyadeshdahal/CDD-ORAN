"""E5 CONFOUND LAYER — obs/do effect estimation through the decision spine (DEV, plan 015 §3).

The confound layer is where causal EFFECT estimation (not just binary structure) becomes load-bearing.
The core layer showed SHAP/GNN prune the subdominant edge at the STRUCTURE level; pooled correlation
recovers the structure and — because the current spine hands every recovered edge the TRUE mechanism —
takes the right action. This layer removes that crutch: each method estimates the P0->K_harm EFFECT from
a corpus, and the world model uses the ESTIMATED effect. Under a latent confounder Z, an observational
corpus yields a biased effect; a randomized do(P0) corpus identifies the truth.

Scope = Option A (5-lane research decision, 2026-09-23; scratchpad/e5_design/confound_scope_lane*.md):
an ORACLE-NUISANCE, ORACLE-FORM, SCALAR-COEFFICIENT ablation. We grant the TRUE base(G1,G2), the true
gate, the true functional form, and every other KPI mechanism, and CONTEST ONLY the scalar in-gate
P0->K_harm coefficient. The fitted world model is literally ``E5V2Env(subdom=alpha_hat, theta=0)`` — the
real frozen class with one estimated coefficient, so NO mechanism is copied (drift-safe). NOTE (sol
review, 2026-09-23): granting base is NOT a free additive constant — although base is constant across
candidate P0 at a fixed state, it sets K_harm's level relative to the satisfy-below threshold and thus
the discrete satisfied-count geometry. So this grants decision-critical nuisance; the result is an
effect-estimation-bias isolation, NOT a learned-world-model result.

DEFENSIBLE CLAIM (sol-scoped): the confound demonstration is the OBS-vs-DO contrast of the ORACLE-GATE
IN-STRATUM OLS estimator. On the OBSERVATIONAL corpus (P0 confounded by latent Z) it estimates a biased
in-gate slope (~+0.117 vs true +0.200) and the planner walks into the trap; on a RANDOMIZED do(P0) corpus
(P0 ⟂ Z) it identifies the true gated effect (~+0.201) and takes the oracle action. The winning arm
REQUIRES interventional data (~thousands of randomized P0 interventions over the full range) — this is
the value of intervention capability for a fully-latent confounder, NOT a superiority claim over
observational methods. Report the OBS-DO regret gap as an "observed contrast, analytically explained by
latent omitted-variable bias" and back it with a paired theta=0 difference-in-differences control + a
corpus-seed panel (both computed below), NOT as a single "attributable cost".

HONESTY guardrails (sol adversarial lanes):
  * marginal pooled OLS IS confounded (its OBS slope sign-reverses to ~-0.073) but adds NO extra DECISION
    contrast: gate dilution (occupancy x effect ~ +0.011) already breaks its DO arm, so OBS and DO land on
    the same saturated action. Its zero OBS-DO gap is metric saturation, not absence of confounding.
  * a covariate-aware GBDT fit attenuates (~50%) and is decision-broken even on DO (intrinsic shrinkage,
    DO regret > 0). Report it diagnostically with its DO floor; do NOT cite its OBS-DO gap as a
    confounding win, and do NOT generalize the marginal sign-flip to "correlational methods" as a category.
  * the stratified arm here uses the TRUE gate to isolate the ESTIMAND; truth-free gate DISCOVERY feeding
    effect estimation END-TO-END (propagating discovery uncertainty) is DEFERRED (overlaps M3
    productization, plan 014 #4). Until then this is "oracle-gate in-stratum OLS", not the deployed method.

CORE layer (chain/noise off), H=1, in-gate-conditional regret at ~5% occupancy. Nothing here is frozen.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from cdd_oran.analysis.v2_regret import P0_GRID, score_grid  # noqa: E402
from cdd_oran.envs.v2.e5 import E5V2Env, _gate  # noqa: E402
from scripts.e5_spine import build_ingate_bank, e5_panel  # noqa: E402

NP, NK = E5V2Env.num_params, E5V2Env.num_kpis
SUBDOM = 0.20                 # true gated P0->K_harm slope (POSITIVE)
THETA = -2.5                  # confound Z->K_harm (NEGATIVE => masks the +subdom association; env default)
LAM = 1.0                     # Z->P0_behavior (positive, natural)
CHAIN = 0.0                   # CORE layer (chain off)
MIN_CELL = 20                 # minimum in-gate rows before an OLS slope is trusted


def _in_gate(g1, g2):
    return _gate(g1, g2, E5V2Env.TAU1, E5V2Env.C2, E5V2Env.W_GATE)


def gen_corpus(n, mode, seed, theta=THETA, lam=LAM, return_z=False):
    """Confounded corpus via the TRUE E5 mechanism with a PER-ROW committed Z.

    Context params uniform; P0 = Z-confounded behavior (obs) or randomized do-grid (do, ON THE ENV's
    discrete ``_DO_GRID`` so it matches ``E5V2Env._draw_do_p0``). K_harm carries theta*Z (ungated) +
    subdom*(P0-C0) inside the gate (~5% of rows). At a FIXED seed the exogenous draws (G1,G2,P3,Z,eta)
    are identical across mode and theta, so obs/do and theta/theta=0 corpora are PAIRED for a clean
    difference-in-differences control (sol #5). K_harm is computed by the env's own ``_update_kpis``."""
    env = E5V2Env(env_seed=0, subdom=SUBDOM, chain_gamma=CHAIN, theta=theta, lam=lam, mode=mode)
    rng = np.random.default_rng(seed)
    g1 = rng.uniform(*E5V2Env.id_ranges[1], n)
    g2 = rng.uniform(*E5V2Env.id_ranges[2], n)
    p3 = rng.uniform(*E5V2Env.id_ranges[3], n)
    Z = rng.standard_normal(n)
    eta = rng.standard_normal(n) * E5V2Env.ETA_SCALE
    grid_idx = rng.integers(0, len(E5V2Env._DO_GRID), n)   # consume rng identically regardless of mode
    if mode == E5V2Env.MODE_OBS:
        p0 = np.clip(lam * 30.0 * Z + eta, *E5V2Env.id_ranges[0])   # matches E5V2Env.behavior_p0
    else:
        p0 = E5V2Env._DO_GRID[grid_idx]                             # matches E5V2Env._draw_do_p0
    X = np.column_stack([p0, g1, g2, p3])
    Y = np.empty((n, NK))
    for i in range(n):
        env.prev_Z = float(Z[i])
        Y[i] = env._update_kpis(X[i], np.zeros(NK))
    return (X, Y, Z) if return_z else (X, Y)


def _base_residual(X, Y):
    """K_harm minus the GRANTED true base(G1,G2) (Option A) — isolates the P0-driven part."""
    env0 = E5V2Env(subdom=SUBDOM, chain_gamma=CHAIN, theta=0.0)
    base = np.array([env0._base(X[i, 1], X[i, 2]) for i in range(len(X))])
    return Y[:, 1] - base


# --- effect estimators (each returns a scalar in-gate P0 slope alpha_hat) ----------------------------
def est_marginal_pooled(X, Y):
    """Marginal ungated pooled OLS slope of the base-residual on P0 (signed pooled-correlation family)."""
    r = _base_residual(X, Y)
    return float(np.polyfit(X[:, 0], r, 1)[0])


def est_stratified_ingate(X, Y):
    """Oracle-gate in-stratum OLS slope of the base-residual on P0 (gate GIVEN = estimand isolation;
    truth-free gate discovery is deferred). On DO this identifies +subdom; on OBS it is confound-biased."""
    r = _base_residual(X, Y)
    m = np.array([_in_gate(X[i, 1], X[i, 2]) for i in range(len(X))])
    if int(m.sum()) < MIN_CELL:
        return 0.0
    return float(np.polyfit(X[m, 0], r[m], 1)[0])


def est_gbdt_ingate_pd(X, Y, delta=1.0):
    """Covariate-aware HistGradientBoosting on (P0,G1,G2)->K_harm; in-gate local partial-dependence
    slope of P0 (finite-difference on in-gate rows). Diagnostic only (see docstring)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    gb = HistGradientBoostingRegressor(random_state=0, max_iter=400, learning_rate=0.05)
    gb.fit(X[:, :3], Y[:, 1])
    m = np.array([_in_gate(X[i, 1], X[i, 2]) for i in range(len(X))])
    Xin = X[m, :3]
    if len(Xin) < MIN_CELL:
        return 0.0
    lo, hi = Xin.copy(), Xin.copy()
    lo[:, 0] -= delta
    hi[:, 0] += delta
    return float(np.mean(gb.predict(hi) - gb.predict(lo)) / (2 * delta))


# --- decision: fitted-slope world model = E5V2Env(subdom=alpha_hat, theta=0) (drift-safe) -----------
def regret_for_slope(alpha_hat, bank, panel):
    norm, raw, mism, viol = [], [], 0, 0
    for seed, snap in bank:
        true_env = E5V2Env(env_seed=seed, theta=0.0, subdom=SUBDOM, chain_gamma=CHAIN)
        wm = E5V2Env(env_seed=seed, theta=0.0, subdom=float(alpha_hat), chain_gamma=CHAIN)
        s_true = score_grid(true_env, snap, E5V2Env.P0, panel)
        s_wm = score_grid(wm, snap, E5V2Env.P0, panel)
        g_star = float(s_true.max())
        d = g_star - float(s_true.min())
        a_wm = int(np.argmax(s_wm))
        r = g_star - float(s_true[a_wm])
        raw.append(r)
        norm.append(r / d if d > 1e-9 else 0.0)
        mism += int(a_wm != int(np.argmax(s_true)))
        te = E5V2Env(env_seed=seed, theta=0.0, subdom=SUBDOM, chain_gamma=CHAIN)
        te.restore(snap); te.apply_action(E5V2Env.P0, float(P0_GRID[a_wm]))
        te.advance(); k = te.advance()
        viol += int(k[E5V2Env.K_HARM] > E5V2Env.kpi_thresholds[E5V2Env.K_HARM])
    return dict(norm=float(np.mean(norm)), raw=float(np.mean(raw)),
                mism=mism, viol=viol, n=len(bank))


ESTIMATORS = [
    ("marginal pooled OLS", est_marginal_pooled),      # confounded (sign-reversed) but decision-saturated
    ("oracle-gate in-stratum OLS", est_stratified_ingate),  # DO identifies; OBS confound-biased (headline)
    ("GBDT covariate-aware", est_gbdt_ingate_pd),      # diagnostic only (intrinsic shrinkage on DO)
]


def _matrix(bank, panel, seed, theta=THETA, gbdt=True):
    """The estimator x corpus matrix at one paired seed (obs/do share exogenous draws)."""
    Xo, Yo = gen_corpus(8000, E5V2Env.MODE_OBS, seed, theta=theta)
    Xd, Yd = gen_corpus(8000, E5V2Env.MODE_DO, seed, theta=theta)
    out = {}
    for name, fn in ESTIMATORS:
        if name.startswith("GBDT") and not gbdt:
            continue
        for corpus, (X, Y) in [("OBS", (Xo, Yo)), ("DO", (Xd, Yd))]:
            a = fn(X, Y)
            out[(name, corpus)] = (a, regret_for_slope(a, bank, panel))
    return out


def main():
    panel = e5_panel()
    bank = build_ingate_bank(subdom=SUBDOM, chain_gamma=CHAIN)
    print(f"E5 CONFOUND layer (Option A: oracle nuisances + form, scalar P0 coeff contested) — "
          f"{len(bank)} in-gate states")
    print(f"  true gated slope +{SUBDOM} | confound theta={THETA}, lam={LAM} | H=1, chain/noise off | "
          f"in-gate-conditional regret\n")

    m = _matrix(bank, panel, seed=0)
    r_or = regret_for_slope(SUBDOM, bank, panel)
    print(f"  {'estimator':28s} {'corpus':4s} {'slope':>8s} {'norm':>7s} {'raw':>7s} {'mism':>7s} {'viol':>7s}")
    print(f"  {'oracle (true)':28s} {'--':4s} {SUBDOM:+8.3f} {r_or['norm']:7.4f} {r_or['raw']:7.2f} "
          f"{r_or['mism']:>3d}/{r_or['n']:<3d} {r_or['viol']:>3d}/{r_or['n']:<3d}  (oracle K_harm viol baseline)")
    for name, _ in ESTIMATORS:
        for corpus in ("OBS", "DO"):
            a, r = m[(name, corpus)]
            print(f"  {name:28s} {corpus:4s} {a:+8.3f} {r['norm']:7.4f} {r['raw']:7.2f} "
                  f"{r['mism']:>3d}/{r['n']:<3d} {r['viol']:>3d}/{r['n']:<3d}")

    # --- difference-in-differences confound control (sol #5): paired theta vs theta=0 ---------------
    print("\n  DIFFERENCE-IN-DIFFERENCES (oracle-gate in-stratum OLS; paired exogenous draws):")
    print("    DiD = [R_OBS(theta) - R_DO(theta)] - [R_OBS(0) - R_DO(0)]  isolates confounding from")
    print("    treatment-distribution/dilution effects (theta=0 has NO confound; any residual gap = other).")
    m0 = _matrix(bank, panel, seed=0, theta=0.0, gbdt=False)
    name = "oracle-gate in-stratum OLS"
    diff_th = m[(name, "OBS")][1]["norm"] - m[(name, "DO")][1]["norm"]
    diff_0 = m0[(name, "OBS")][1]["norm"] - m0[(name, "DO")][1]["norm"]
    print(f"    theta={THETA}: slopes OBS {m[(name,'OBS')][0]:+.3f} / DO {m[(name,'DO')][0]:+.3f}; "
          f"R_OBS-R_DO = {diff_th:+.4f}")
    print(f"    theta= 0.0: slopes OBS {m0[(name,'OBS')][0]:+.3f} / DO {m0[(name,'DO')][0]:+.3f}; "
          f"R_OBS-R_DO = {diff_0:+.4f}")
    print(f"    DiD (confound-attributable contrast) = {diff_th - diff_0:+.4f}")

    # --- corpus-seed panel (sol #6/tests): uncertainty on the headline contrast --------------------
    print("\n  CORPUS-SEED PANEL (oracle-gate in-stratum OLS, 5 paired seeds):")
    obs_s, do_s, gap_s = [], [], []
    for s in range(5):
        ms = _matrix(bank, panel, seed=s, gbdt=False)
        obs_s.append(ms[(name, "OBS")][1]["norm"])
        do_s.append(ms[(name, "DO")][1]["norm"])
        gap_s.append(obs_s[-1] - do_s[-1])
    print(f"    OBS regret : mean {np.mean(obs_s):.4f}  range [{min(obs_s):.4f}, {max(obs_s):.4f}]")
    print(f"    DO  regret : mean {np.mean(do_s):.4f}  range [{min(do_s):.4f}, {max(do_s):.4f}]")
    print(f"    OBS-DO gap : mean {np.mean(gap_s):.4f}  range [{min(gap_s):.4f}, {max(gap_s):.4f}]  "
          f"(observed contrast, analytically explained by latent OVB)")


if __name__ == "__main__":
    main()
