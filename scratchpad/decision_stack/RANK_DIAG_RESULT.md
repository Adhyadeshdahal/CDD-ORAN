# Why the policy-effect model can't rank WG3 candidates (2026-09-28; scripts diag_rank/a-j, no commits)

Two causes, each sufficient for rho~0: (1) E6 candidate effects are unpredictable from the model's context features even
with noise-free paired oracle labels; (2) estimator overdispersion (pred contrasts 13-33x wider than oracle) + candidate
panel out of training support. No estimand/off-by-one bug.

1 Signal/noise: J(accept-all) ~5418/slot; within-slot SD 45.5 (0.9%); headroom mean 65.5 median 43.6; best cand spread
over types (random 120, SLICE-single 76, TS 45, freeze 37, rollback 33, accept 24); best-2nd gap median 4.8. Best
context-free rule TS:lock only +3.5/slot; SLICE reject/half/lock SD 72-76, P(better) 0.49. Headroom = eMBB 39.6, LL 22.4,
energy 2.9. Effects structured within slot (same-behaviour cands corr 0.93-0.97; cross-xApp additivity R2 0.87) but
lag-1 slot autocorr ~0.1; between-episode var share 0.12-0.22. v3 labels: region SD 587, context R2 0.83, residual SD
~250/region; policy features add nothing (CV R2 0.829->0.828); main effects SE ~10.7, all |z|<=2.5. Per-region effects
+-2..49 (SD~20), mixed signs. Per-row SNR ~0.006. Pooled network error ~107 > 78 needed for rho 0.5; CATE p=50 needs ~15x
data (~1800 eps) AND a feature set that carries signal (none found).
2 Representation/support: within-slot pred SD 575/1076/1510 (none/topology/all) vs oracle 45; member SD ~0.8 of cross-
cand spread -> picks = noise extremes (hence -6..-9). Ridge 449-629 cols on 7200 rows. 82% panel cands network-uniform;
0/840 fit slots had all regions same nonzero code; neighbour-policy aggregate 1.0 vs train max 0.56-0.67 (~8 SD). "none"
selector still rho 0.006. Ceilings w/ oracle labels (CV by episode): context-free network-summed rho 0.136; + stratum
0.121; paired oracle labels + model context predicting 14 fixed-cand effects: CV R2<=0 all 14 (ridge, HGB), rho 0.076;
even future request-stream aggregates R2<=0.09. True per-slot single-xApp effects spread additively -> rho 0.465 on
random/search cands (a correct per-slot CATE would rank; model can't learn it).
3 Estimand: matches (v3 t..t+19 then accept-all, labels (t,t+H], F4 holds, region labels sum exactly to panel J in 360/360).
Minor: panel resets half_state/rb_at per slot.
4 Additivity: across xApps R2 0.87; across regions partial (SLICE:reject net +9.6 vs sum +5.6; +42.3 vs +81.0), concentrated
in 2-5 regions, mixed signs.
5 Physics vs defect: robust to 1e-6 m nudge and D 19 vs 20 (r 0.96-0.99). Replication local numpy 2.4.2 vs Kaggle 2.0.2
same commit b00920b: trajectories diverge (SVR 872.1 vs 877.0); effect corr SLICE 0.53, TS 0.90, freeze 0.45, pooled
reliability 0.56 (n=18 slots, 54 pairs); argmin agreement 0.83 -> ~44% of oracle effect variance is tape-specific; 65-unit
headroom partly max-over-noise. SIDE: oracle panels not bit-reproducible across numpy versions -> PIN numpy / record it.
Single ll_ratio quantum reject costs 1-17 units, sign follows request direction; single TS CIO rejects usually exactly 0;
a 20 s policy = sum of ~44 SLICE + ~34 TS micro-effects; region-policy abstraction discards request-level structure.

Ranked causes: #1 no learnable signal in representation (mostly E6-specific; the region-policy abstraction part is
structural); #2 estimator defect (any sim); #3 SNR; #4 oracle headroom overstated (reliability 0.56); estimand not a cause.
Fixes: EB shrink contrasts toward 0 + "beat zero-contrast" precondition (regret -9 -> ~0, no gain); restrict candidates to
support or collect uniform/joint slots; paired CRN labels (noise 250 -> 20-25) only with better features; REQUEST-LEVEL
decision unit/features (direction, target cell, local state) = most promising, conjecture -> test on existing panels with
~50 per-request oracle rollouts; D/H sweep report replication reliability; re-test in conflict-rich sim; pin numpy.
