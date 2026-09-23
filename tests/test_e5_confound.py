"""E5 confound-layer regression tests (DEV env; plan 015 §3, sol reviews #6 + confound-layer lane).

Locks the corrected confound design and its HONEST attribution:
(a) default confound sign is masking (theta<0, lam*theta<0 opposes +subdom);
(b) the harness obs policy matches E5V2Env.behavior_p0 and the do policy matches the env do-grid (drift);
(c) obs P0 is Z-driven (corr(P0,Z) high) while do P0 is Z-independent;
(d) the confound MASKS: marginal pooled OBS slope is sign-reversed vs true +subdom;
(e) the do corpus IDENTIFIES (in-gate slope ~ +subdom) while in-gate OBS slope is attenuated (biased);
(f) drift-safe fitted world model E5V2Env(subdom=true) reproduces the oracle (regret ~0);
(g) THE confound-layer claim + its DiD control: stratified OBS regret >> DO regret ~0 under confounding,
    but the theta=0 factorial control shows OBS==DO~0 (so the gap IS the confound, not treatment
    distribution); the marginal estimator's OBS-DO gap is ~0 (saturation) yet its OBS/DO slopes DIFFER
    materially (it IS confounded — the zero gap is metric saturation, not absence of confounding).
"""
from __future__ import annotations

import numpy as np

from cdd_oran.envs.v2.e5 import E5V2Env, _gate
from scripts.e5_confound import (
    LAM,
    SUBDOM,
    THETA,
    build_ingate_bank,
    e5_panel,
    est_marginal_pooled,
    est_stratified_ingate,
    gen_corpus,
    regret_for_slope,
)

N = 4000


def test_default_confound_sign_masks():
    """(a): default theta is negative and lam*theta<0 opposes the positive true subdom (masking, not
    reinforcing). Locks the sol-#6 fix as a direct env-default invariant."""
    env = E5V2Env()
    assert env.theta < 0.0
    assert env.lam > 0.0
    assert env.lam * env.theta < 0.0 < SUBDOM


def test_harness_policies_match_env_api():
    """(b): drift guard. The harness obs policy equals E5V2Env.behavior_p0 (30-scale, lam), and the do
    corpus draws from the env's own _DO_GRID (not a private continuous uniform)."""
    env = E5V2Env(mode=E5V2Env.MODE_OBS, lam=LAM)
    env.reset(episode=0)
    env.Z = 1.3
    # behavior_p0 = clip(lam*30*Z + eta); with |Z| large the lam*30 scale dominates the eta noise (0.5).
    assert abs(env.behavior_p0() - LAM * 30.0 * 1.3) < 3.0
    # do corpus values must all lie on the env's discrete do-grid.
    Xd, _ = gen_corpus(500, E5V2Env.MODE_DO, 0)
    assert np.all(np.isin(Xd[:, 0], E5V2Env._DO_GRID))


def test_obs_confounded_do_independent():
    """(c): obs P0 correlates with the latent Z (confounding channel); do P0 does not."""
    Xo, _, Zo = gen_corpus(N, E5V2Env.MODE_OBS, 0, return_z=True)
    Xd, _, Zd = gen_corpus(N, E5V2Env.MODE_DO, 0, return_z=True)
    assert abs(np.corrcoef(Xo[:, 0], Zo)[0, 1]) > 0.9
    assert abs(np.corrcoef(Xd[:, 0], Zd)[0, 1]) < 0.05


def test_confound_masks_marginal_slope_sign_reversed():
    """(d): with theta<0 the marginal pooled OBS slope is NEGATIVE — the confound MASKS the true +subdom."""
    Xo, Yo = gen_corpus(N, E5V2Env.MODE_OBS, 0)
    assert est_marginal_pooled(Xo, Yo) < 0.0 < SUBDOM


def test_do_corpus_identifies_obs_is_biased():
    """(e): in-gate DO slope ~ +subdom (identified); in-gate OBS slope attenuated in (0, subdom)."""
    Xo, Yo = gen_corpus(N, E5V2Env.MODE_OBS, 0)
    Xd, Yd = gen_corpus(N, E5V2Env.MODE_DO, 0)
    a_do = est_stratified_ingate(Xd, Yd)
    a_obs = est_stratified_ingate(Xo, Yo)
    assert abs(a_do - SUBDOM) < 0.05, f"DO in-gate slope {a_do} should identify +{SUBDOM}"
    assert 0.0 < a_obs < SUBDOM, f"OBS in-gate slope {a_obs} should be attenuated in (0, {SUBDOM})"
    # analytic OVB check: OBS bias ~ theta*lam*30/var(P0_ingate) added to true subdom.
    assert abs(a_obs - (SUBDOM + THETA * LAM * 30.0 / (LAM * 30.0) ** 2)) < 0.03


def test_fitted_slope_wm_with_true_slope_is_oracle():
    """(f): the drift-safe fitted world model E5V2Env(subdom=true) reproduces the oracle (regret ~0)."""
    bank = build_ingate_bank(subdom=SUBDOM, chain_gamma=0.0)[:12]
    r = regret_for_slope(SUBDOM, bank, e5_panel())
    assert r["norm"] < 1e-6 and r["mism"] == 0


def test_confound_layer_claim_and_did_control():
    """(g): the confound-layer claim + difference-in-differences control.

    Under confounding (theta<0): stratified DO identifies (regret ~0), OBS is biased (regret >> 0). The
    theta=0 control gives OBS==DO~0 => the OBS-DO gap IS confounding (DiD large, not a treatment-dist
    artifact). Marginal: OBS-DO regret gap ~0 (metric saturation) BUT its OBS/DO slopes differ materially
    (it IS confounded), so the zero gap must not be read as absence of confounding."""
    bank = build_ingate_bank(subdom=SUBDOM, chain_gamma=0.0)[:12]
    panel = e5_panel()
    # paired exogenous draws (same seed) for obs/do and theta vs theta=0
    Xo, Yo = gen_corpus(N, E5V2Env.MODE_OBS, 0)
    Xd, Yd = gen_corpus(N, E5V2Env.MODE_DO, 0)
    Xo0, Yo0 = gen_corpus(N, E5V2Env.MODE_OBS, 0, theta=0.0)
    Xd0, Yd0 = gen_corpus(N, E5V2Env.MODE_DO, 0, theta=0.0)

    strat_obs = regret_for_slope(est_stratified_ingate(Xo, Yo), bank, panel)["norm"]
    strat_do = regret_for_slope(est_stratified_ingate(Xd, Yd), bank, panel)["norm"]
    strat_obs0 = regret_for_slope(est_stratified_ingate(Xo0, Yo0), bank, panel)["norm"]
    strat_do0 = regret_for_slope(est_stratified_ingate(Xd0, Yd0), bank, panel)["norm"]

    assert strat_do < 0.05, "DO identifies -> near-zero regret"
    assert strat_obs - strat_do > 0.10, "OBS confound-biased -> large regret gap"
    # theta=0 control: no confound -> OBS and DO both identify, gap ~ 0 -> DiD isolates confounding
    assert strat_obs0 < 0.05 and strat_do0 < 0.05
    did = (strat_obs - strat_do) - (strat_obs0 - strat_do0)
    assert did > 0.10, f"DiD {did} should attribute the gap to confounding"

    # marginal: OBS-DO regret gap saturates to ~0, but its slopes DIFFER (still confounded)
    a_obs_m = est_marginal_pooled(Xo, Yo)
    a_do_m = est_marginal_pooled(Xd, Yd)
    marg_obs = regret_for_slope(a_obs_m, bank, panel)["norm"]
    marg_do = regret_for_slope(a_do_m, bank, panel)["norm"]
    assert abs(marg_obs - marg_do) < 0.05, "marginal regret gap saturates"
    assert abs(a_obs_m - a_do_m) > 0.05, "but marginal OBS/DO slopes differ -> it IS confounded"
