# xTRUCE plant re-implementation: parameter and interface specification

Status: **DRAFT, 2026-09-29.** Code: `cdd_oran/envs/xtruce/` (pure numpy plant; scipy is used only by the optional
xTRUCE arbiter). Tests: `tests/test_xtruce_env.py`.

Source: L. Xia, R. Q. Hu, P. S. Kudyba, Z. An, H. Sun, "xTRUCE: A Provably Safe Arbiter for Multi-xApp Conflict
Mitigation in Agentic O-RAN", arXiv:2608.28532**v2** (25 Sep 2026). The authors' code is not public ("will be
released on GitHub upon publication", p. 1). Page numbers below are the v2 PDF's.

Provenance tags: **[P]** stated in the paper (location given). **[A]** assumption: either a value taken from a named
external source, or a declared modelling choice with a stated reason. **[A\*]** assumption whose value was chosen after
looking at plant outputs on seeds 0-2 (disclosed below). The screen that uses this plant must therefore use fresh
seeds.

## 0. Corrections to our earlier notes

| our note | what the paper says |
|---|---|
| "3GPP UMa pathloss" (read as TR 38.901) | Table I cites "UMa path loss [38]", and [38] is **3GPP TR 36.814 V9.0.0** (p. 13). We implement TR 36.814 Table B.1.2.1-1 UMa, which is the ITU-R M.2135 UMa model. |
| knobs = RB share / tx power / cell sleep / UE association, commanded by xApps | These are the **action variables** of eqs. (12)-(13) (p. 6-7): alpha_b (sleep), z_u (association), x_{u,k} (RB share), p_{u,k} (power). **xApps never command them.** They post KPI *targets* (hard or soft), eq. (2) (p. 4), and the arbiter maps the targets to actions (Sec. IV-D, p. 7). Our knob-level request interface is therefore an adaptation (Sec. 3). |
| "65-92 %" | This is the **Clipping benchmark's** e1 violation ratio: "at least one user below the rate floor in 65 to 92 % of the epochs" across hallucination levels 0-1 (Fig. 3(b), p. 10). Direct: 30-86 %. xTRUCE: 0 %. |
| "CVXPY static arbiter" | CVXPY is **xTRUCE's own** solver: "parameterized exponential-cone programs in CVXPY ... CLARABEL, ECOS, and SCS" (Sec. V-A, p. 8). The benchmarks are Direct and Clipping (p. 9). There is no static-priority baseline in the paper. The "Flat" scheme (Fig. 4) is xTRUCE with all targets in one priority class. |
| direct / indirect conflict scenarios | The paper cites the direct / indirect / implicit taxonomy ([18], p. 1) but defines **no** conflict scenarios of that kind. Its experiments are a hallucination sweep (Fig. 3), an overload sweep (Fig. 4), LLM renegotiation (Fig. 5) and scaling (Fig. 6). The knob-level conflict map in Sec. 4 is ours. |

## 1. Parameters (`XConfig`, `cdd_oran/envs/xtruce/config.py`)

### 1.1 Network, channel, traffic, power

| field | value | tag | source / reason |
|---|---|---|---|
| `n_cells`, `n_ues` | 4, 20 | [P] | Table I (p. 8) "4 / 20 (hexagonal, 500-m ISD)" |
| `isd_m` | 500 | [P] | Table I |
| layout | centre + first 3 ring-1 sites (compact cluster), omni, no wrap-around | [A] | Paper says only "hexagonal". Ring order also gives the Fig. 6 sizes of 1-6 cells. |
| UE drop | uniform over the union of the cells' hexagons, ≥ `min_dist_m` from every site, static (no mobility) | [A] | The paper does not describe mobility or UE placement. |
| `min_dist_m` | 35 | [A] | TR 36.814 Table A.2.1.1-2 (minimum UE-cell distance) |
| `n_rb`, `rb_hz` | 12, 360 kHz | [P] | Table I "12 x 360 kHz" |
| `pathloss` | `"36814_uma"`: TR 36.814 Table B.1.2.1-1 UMa, LOS/NLOS by Table B.1.2.1-2 probability (one static draw per link) | [P] family / [A] variant | Table I "UMa path loss [38]". `"36814_macro"` (128.1 + 37.6 log10 R) is also available. |
| `fc_ghz` | 2.0 | [A] | ITU-R M.2135 UMa test environment carrier (2 GHz). The paper gives no simulation carrier; its OTA link is n48 at 3.61 GHz. |
| `h_bs_m`, `h_ut_m`, `street_w_m`, `bldg_h_m` | 25, 1.5, 20, 20 | [A] | TR 36.814 Table B.1.2.1-1 UMa defaults |
| `antenna_gain_db` | 0 | [A] | omni cells, no gain (paper silent; the plant is interference-limited, so a common gain barely matters) |
| `shadow_db` | 8 | [P] | Table I "+ 8-dB shadowing" (log-normal, static per link) |
| `shadow_site_corr` | 0.5 | [A] | TR 36.814 Table A.2.1.1-2 inter-site shadowing correlation |
| `fading_rho` | 0.349 | [P] | Table I "AR(1) Rayleigh (rho = 0.349) [39]" |
| fading model | h(t) = rho h(t-1) + sqrt(1 - rho^2) w, w ~ CN(0,1), per (cell, UE, RB) | [P] model / [A] per-RB independence | [39] Baddour & Beaulieu. The paper does not say whether RBs fade independently. `fading_per_rb=False` gives one coefficient per link. |
| `noise_w` | 1.15e-14 W per RB | [P] | Table I. It equals kT * 360 kHz + 9 dB (the TR 36.814 UE noise figure). |
| `traffic_bps` | 6 Mbit/s per UE | [P] | Table I "E[lambda_u(t)]/tau = 6 Mbps" |
| `arrivals` | lambda_u(t) ~ Exp(mean 6 Mbit * tau), i.i.d. per UE and epoch | [P] "exponential arrivals" / [A] reading | Our reading is that the per-epoch amount is exponential. A Poisson packet process is the other possible reading. |
| `epoch_s` | 1 s | [P] | Table I tau |
| `p_max_w`, `p_cir_w`, `p_rb_w` | 10, 50, 2 W | [P] | Table I "Powers P_b^max / P_b^cir / P^rb = 10/50/2 W" |
| energy model | E_b = alpha_b (P_cir + delta_p * P_tx,b) + (1 - alpha_b) P_sleep, with `delta_p` = 1 and `p_sleep_w` = 0 | [P] | eq. (16) (p. 7): E_b = P_cir alpha_b + sum_k P_b,k. This is the EARTH form P0 + Δp P_out with Δp = 1 and no sleep power. `delta_p` and `p_sleep_w` are exposed for an EARTH variant. |
| rate | R_u = sum_k x_u W log2(1 + G_{u,k} p_{u,k} / (x_u (sigma^2 + I_{u,k}))) | [P] | eq. (14) (p. 7) |
| fast action granularity | x_{u,k} = x_u, p_{u,k} = p_u / K for all k | [P] | Sec. V-A (p. 8): "the simulations do not differentiate the fast actions of (13) over individual RBs". Rates are still summed over the per-RB fading. |
| interference | realised: from the current-epoch cell powers P_b,k = P_b / K and current gains; exposed to xApps / arbiters: the value measured in t-1 | [P] | Sec. IV-A (p. 7): I_{u,k}(t) "measured at epoch t-1 and treated as known at epoch t". Channel gains G(t) are known at t (Sec. IV-B, footnote 1). |
| queue | Q_u(t+1) = [Q_u(t) - tau R_u]^+ + lambda_u(t) | [P] | eq. (15) (p. 7) |
| load, caused interference | rho_b = sum_{u in U_b} x_u; I_out_b = sum_k sum_{u not in U_b} G_{b,u,k} P_b,k | [P] | eqs. (17)-(18) (p. 7) |

### 1.2 Operator rules, policy, arbitration

| field | value | tag | source / reason |
|---|---|---|---|
| `n_prot`, `rmin_bps` | 3 UEs, 2 Mbit/s | [P] | Table I. Which UEs are protected is not stated: drawn uniformly at construction [A]. |
| `dx`, `dp_w`, `d_act`, `d_str` | 0.25, 0.25 W, 1 cell, 3 users | [P] | Table I; eq. (20) e2-e5 (p. 7). e2 is per (u, k), so the per-UE total limit is K * 0.25 W = 3 W [A reading]. |
| e1-e5 enforcement | only the xTRUCE arbiter enforces them. The plant logs violations (`e1_viol_epochs`, `oper_viol_epochs`). | [P] | These are operator rules in H_t, and the benchmarks do not obey them (p. 9) |
| `phys` | `"clip"`: the plant enforces c1-c4 (19) and the cell cap `pcap` by per-UE clipping and per-cell proportional rescaling (the paper's Clipping operation). `"none"`: the arbiter's action is executed as is. | [A] | The paper evaluates Direct's c1-c4-violating actions (99 % of epochs, p. 9) but never says how its simulator realises them. `"none"` reproduces that. `"clip"` is physical, and our screen needs it so that accept-all cannot draw power it does not have. |
| `t_cfg` | 10 epochs | [A] | The paper says only T_cfg > 1 (p. 4). 10 s is the E6-P ES / PowerES cadence and keeps sleep and steering on a slower clock than the fast loop. |
| `proposal_ttl` | 2 epochs | [P] | Sec. V-A (p. 8) "each accepted proposal stays valid for two epochs unless renewed" |
| priority classes | protected rate kappa = 1; non-protected rate, energy, interference kappa = 2; load kappa = 3 | [P] | Sec. V-A (p. 8) |
| `all_hard` | False: only protected-rate targets are hard (Sec. IV-D, p. 7). True: all ranked targets are hard (the Fig. 4 style, p. 10). | [A] | The paper is ambiguous here, see Sec. 6. |
| `beta` | 1 for all targets | [P] | Table I |
| `eps_rel` | epsilon_l = 1e-4 (1 + v_l*) | [P] | Table I |
| `eta` | 0.01 | [A] | Stage-II action-change weight. The paper says "operator-assigned" (p. 5) and gives no value. |
| xTRUCE KPI units | rate in Mbit/s, energy in W, interference and load relative to their cap | [A] | The paper gives no units or scaling. In raw SI units the interference term (~1e-20 W^2) would vanish. |

### 1.3 xApps (Sec. IV-D roles [P]; rules [A]; `cdd_oran/envs/xtruce/xapps.py`)

The paper says that its sweeps use "deterministic rules" (p. 8) but does not publish them. The roles, KPIs, target
types and the 53-W cap come from the paper. Everything else is [A].

| xApp | proposal, eq. (2) | direct action (knob requests) | constants |
|---|---|---|---|
| `QoS` | hard R_u ≥ `qos_prot_bps` for protected UEs (kappa 1) [P]; soft R_u ≥ arrival-rate EWMA for the others (kappa 2) [A: the paper says "to reduce the backlog Q_u" (p. 7); the offered rate is the smallest target that stops Q_u from growing] | per active cell, at uniform PSD (budget / K): protected shares = target * (1 + margin) / rate, the rest of the band split by demand min(1, target / nominal rate); power = share * budget. Asks for `pcap` = P_max when the floors do not fit under the current cap. | `qos_prot_bps` 2e6 [P, Table I floor; Fig. 4 uses 3e6]; `qos_margin` 0.30 [A\*]; `qos_deadband` 0.02 [A] (protected increases are never dead-banded) |
| `ES` | soft E_b ≤ `es_cap_w` (kappa 2) | `pcap` = (cap - P_cir) / delta_p when above it (fast). At configuration epochs: sleep the least-loaded cell with demand load < 0.30 if its UEs fit below 0.80 in the other cells (or any cell, if cap < P_cir); wake a cell when an active cell's load > 0.80; at most one change per epoch; never the last cell. | `es_cap_w` 53 W [P, Fig. 4 value (p. 10), reused as the default; the paper gives no Fig. 3 default]; `es_sleep_load` 0.30 [A, TR 38.864 Annex A light/medium boundary]; `es_wake_load` 0.80 [A, E6-P ES wake threshold] |
| `IC` | soft I_out_b ≤ n_victims * K * sigma^2 * 10^(20/10) (kappa 2) | lower `pcap` to the power that meets the cap, using an EWMA of the caused-interference gain; floor 0.5 W; only ever lowers | `ic_iot_db` 20 [A, no source]; `ic_min_w` 0.5 [A]; `ic_ewma` 0.5 [A] |
| `LB` | soft rho_b ≤ 0.80 (kappa 3) | At configuration epochs: move the UE of the most loaded cell with the smallest gain gap (≤ 6 dB) to an active cell whose load is ≥ 0.10 lower; one move per epoch. The load here is **demand** load (sum of EWMA arrival rate / nominal full-band rate). The paper's rho_b (eq. 18) counts allocated share, which any work-conserving allocation holds at 1. | `lb_cap` 0.80, `lb_hyst` 0.10, `lb_offset_db` 6 [A: MLB-style CIO span], `load_ewma` 0.3 [A] |
| hallucination | each target is corrupted with probability h; a corrupted theta is multiplied by 10^(2 h s), s ~ U(-1, 1) | requests are derived from the corrupted targets | the level h is [P] (Fig. 3, following [44]); the corruption model is [A] |

Default RRM ([A]): the initial action is an equal share and power in each cell, with association to the strongest mean
gain. A UE that changes cell hands its share and power to the old cell's remaining UEs in proportion, gets 1/n_b of the
new cell, and the new cell's other UEs are scaled by (1 - 1/n_b). Both cells' totals are therefore unchanged. When a
cell sleeps, its UEs go to their best active cell, and they return when it wakes.

**[A\*] disclosure.** `qos_margin` was set to 0.30 after comparing 0.1 / 0.3 / 0.5 on seeds 0-2 (200 epochs). The
criterion was that QoS alone keeps protected violations ≤ 5 % (the 95 % availability rule of E6-P). Freeze, QoS-alone
and accept-all outputs were visible at the time. Two structural changes were also made after looking at seeds 0-2:
(a) moves became work-conserving, because the old rule left the source cell's share unused, which made LB look like an
energy saver; (b) QoS switched from boosting protected UEs up to P_rb per RB to uniform PSD, because the boost let
protected UEs take the whole capped budget and starve the others. **No value was tuned toward a screen outcome, but
the screen must use seeds ≥ 1000** (or seeds registered in `SEED_REGISTRY.json`).

## 2. Paper arbiters (`arbiters.py`, `xtruce_arbiter.py`)

- **Direct** [P] (p. 9): "directly turns the most demanding target of each xApp into the corresponding gNB action
  without checking its joint feasibility". Instance [A]: each rate target gives x_u = theta_u / rate at P_rb per RB,
  with p_u = x_u K P_rb. Each energy cap scales its cell's powers down to (cap - P_cir) / delta_p. Interference and
  load targets are ignored, as the paper's footnote 5 states (p. 9). Run with `phys="none"` for the paper's semantics.
- **Clipping** [P] (p. 9): Direct followed by clipping onto c1-c4 and per-cell proportional rescaling.
- **XTruce** [P] structure, scipy SLSQP in place of CVXPY (no new dependency). Stage I, eqs. (8)/(21), is a
  lexicographic minimisation over the hard classes. Stage II, eqs. (9)/(22), adds soft shortfalls + eta D. H_t is
  c1-c4 plus e1, with the floors planned against measured interference plus the largest increase e2 allows (p. 8),
  and e2/e3 as bounds around the previous executed action. At configuration epochs the candidates are the current
  configuration plus each configuration request alone [A instance of Omega^cfg], compared lexicographically on v*. The
  fallback chain is rule (11): a* → previous action if still in H_t → baseline (min sum p s.t. H_t) → previous action,
  flagged uncertified. Not implemented: certificates / prices (10), the wall-clock budget tau, the standby action.
  Speed is 70-300 ms per epoch, so use it as a reference arm, not inside an oracle.

## 3. API

```python
from cdd_oran.envs.xtruce import XEnv, XConfig, AcceptAll, RejectAll, Subset, conflict_groups, half, modify
env = XEnv(XConfig(), seed)            # deterministic in (cfg, seed)
kpi = env.step(arbiter)                # one 1-s epoch; arbiter.decide(env, requests) -> accepted list (None = all)
reqs = env.step_propose()              # two-phase form: xApps post proposals + knob requests
kpi = env.step_apply(accepted)         # list order = application order (last writer wins)
c = env.copy()                         # exact replay of the future (also valid between the two phases)
c = env.copy(reseed=k)                 # same state, future draws re-keyed by (seed, stream, TAG, k, t)
env.lock(knob, epochs)                 # requests on knob are dropped while t < now + epochs
env.summary()                          # energy_j, prot_viol_ue_s, prot_viol_thr_ue_s, e1/phys/oper violation epochs, ...
```

- **Request**: `{"xapp", "knob", "cur", "prop", "t", "kind": "fast"|"cfg", "target": (kpi, idx)}`. Knobs:
  `("share", u)` in [0,1]; `("power", u)` W; `("pcap", b)` W in [0, P_max]; `("sleep", b)` in {0,1};
  `("assoc", u)` = cell. Configuration knobs take effect only at configuration epochs (`t % t_cfg == 0`); at other
  epochs they are dropped and counted in `cfg_offcycle`.
- **Proposals**: `env.proposals[xapp] = (t, [Target])`. `env.valid_proposals()` applies the 2-epoch TTL.
- **Per-epoch KPIs** (returned by `step` and appended to `env.log` if `cfg.log`): `rate` (capacity, eq. 14),
  `thr` (delivered), `prot_below` (paper e1: protected capacity < R_min), `prot_below_thr` (backlogged ≥ R_min tau
  and delivered < R_min), `power_w` per cell (eq. 16), `ptx_w`, `sleep`, `intf_out_w`, `load`, `queue_bits`, plus
  the executed `x`, `p`, `assoc`, `pcap`, and the `phys_viol` / `oper_viol` flags (the configured action broke
  c1-c4; e2-e5 broken relative to the previous epoch).
- **Random streams**: `fade`, `traffic` and `xapp` (hallucination) streams are independent, and their draw counts do
  not depend on the action. All arms on one seed therefore see the same channel and arrival tape (common random
  numbers). The layout, LOS, shadowing and protected set are construction draws: they are state, and `reseed` keeps
  them. A reseeded copy taken between the phases also re-draws the current epoch's arrivals, while the current
  channel is kept.
- **Oracle helpers**: `conflict_groups(env, reqs)` groups requests by shared cell. It returns only groups with ≥ 2
  xApps and flags `direct` when two xApps request the same knob. `half(req)` moves a fast knob halfway and returns
  `None` for configuration knobs. `modify(req, v)`. `StaticPriority(order)` resolves conflicts per knob;
  `CellPriorityLock(order, lease)` locks a whole cell for a lease; `Subset(names)` accepts only the named xApps.
- **Seed screen** [P] (p. 8, "20 screened seeds whose network realizations keep the protected-user rate floors
  physically reachable"): `env.floors_reachable()` [A operationalisation: on mean gains, in every cell the protected
  floors fit into one band at P_rb per RB with all other cells at full power]; `screened_seeds(n, start)`.

## 4. Knob-level conflict map (ours; not in the paper)

| pair | type | mechanism |
|---|---|---|
| ES ↔ QoS on `pcap` | direct | ES lowers the cap. QoS asks for P_max when the protected floors no longer fit. |
| IC ↔ QoS on `pcap` | direct | same as above; IC lowers the cap of a single cell |
| ES `pcap` ↔ QoS shares | indirect | the cap scales the PSD of every UE in the cell, protected UEs included |
| ES `sleep` ↔ QoS | indirect | protected UEs move to a farther cell (low load only) |
| LB `assoc` ↔ QoS | indirect | LB can move a protected edge UE to a weaker cell |
| LB ↔ ES | indirect | LB loads a cell that ES would otherwise put to sleep; ES sleep moves UEs that LB placed |

## 5. Verification

- Tests (`tests/test_xtruce_env.py`, 19 tests, about 2 s): paper defaults; determinism; exact `copy()` replay, also
  between the phases; `copy(reseed)` keeps state and re-draws the future (same k gives the same future); common random
  numbers across arbiters; sleep cuts cell power to P_sleep, moves the UEs and returns them on wake; off-cycle
  configuration writes are dropped; the last cell is never switched off; a higher share raises that UE's rate with no
  spill-over; more cell power raises neighbour interference and lowers neighbour rates, with E_b = 50 + P_tx; the plant
  clips to c1-c4; locks; ES alone meets its cap; QoS alone cuts protected violations; ES sleeps at low load; conflict
  groups and priority; Direct violates c1-c4 while Clipping never does; xTRUCE keeps the floors and the limits; the
  seed screen; association moves are work-conserving.
- Speed (laptop, single thread, default config, 1000 epochs): about 2,300-2,900 epochs/s for freeze and accept-all.
  `copy()` takes about 0.35 ms.

Sanity run: seeds 0-2 × 300 epochs, default config except traffic. PV = protected-floor violation UE-s out of 2,700
protected UE-s. "matched" means saving ≥ 0.90 × the ES-alone saving.

| arm | 6 Mb/s: mean W | saving | matched | PV | 1 Mb/s: mean W | saving | matched | PV |
|---|---|---|---|---|---|---|---|---|
| freeze | 240.0 | 0 % | no | 611 (22.6 %) | 240.0 | 0 % | no | 611 |
| QoS alone | 237.5 | 1.1 % | no | 96 (3.6 %) | 237.5 | 1.1 % | no | 99 |
| ES alone | 212.0 | 11.7 % | yes | 633 (23.4 %) | 133.7 | 44.3 % | yes | 1042 |
| IC alone | 222.7 | 7.2 % | no | 577 | 222.7 | 7.2 % | no | 577 |
| LB alone | 240.0 | 0 % | no | 571 | 240.0 | 0 % | no | 580 |
| ES + QoS | 211.2 | 12.0 % | yes | 93 (3.4 %) | 129.9 | 45.9 % | yes | 125 |
| accept-all | 209.2 | 12.9 % | yes | 126 (4.7 %) | 120.0 | 50.0 % | yes | 90 |

At the paper's 6 Mb/s load, cells never sleep: every cell's demand load is above 1. ES then acts only through the
53-W cap, and the conflict is mild. Accept-all is energy-matched and loses 35 % more protected UE-s than ES + QoS
(126 vs 93). It is better than the only energy-matched single xApp, ES (633). At 1 Mb/s, ES sleeps cells and
accept-all (90) beats ES + QoS (125).

Paper benchmarks on this plant (`phys="none"`, 3 seeds × 200 epochs): Direct breaks c1-c4 in 100 % of epochs, as
the paper reports (99 %). Clipping breaks them in 0 %, also as reported. Both leave at least one protected UE below
the floor in 98-100 % of epochs at every hallucination level. The paper reports 30-86 % (Direct) and 65-92 %
(Clipping). With our Direct map, targets are planned at exactly 2 Mbit/s and no hedge is applied. Tested alone, the
energy cap (53 → 60 W) does not remove the gap, and the paper's own rules are unpublished. xTRUCE (SLSQP): 0 % e1
violation and 0 % c1-c4 violation (20 epochs, seed 0).

## 6. Known deviations and ambiguities

1. **Hard vs ranked targets.** Sec. IV-D (p. 7) makes only the protected rate hard. Sec. V-A (p. 8) "ranks"
   non-protected rate, energy and interference at kappa 2 and load at kappa 3, but priorities apply only to hard
   targets (p. 4). The default follows Sec. IV-D; `all_hard=True` gives the ranked reading, which is also the Fig. 4
   setting (p. 10).
2. **Default energy cap.** 53 W appears only in Fig. 4, and 50 W in Fig. 5. The Fig. 3 default is not given. The
   `fig5_energy` preset (50 W) literally maps to `pcap` = 0: the "most demanding" action is to transmit nothing.
3. **xApp rules and the non-protected target** are not published. All of Sec. 1.3 is [A].
4. **e2 granularity.** 0.25 W is read per (u, k), because e2 in eq. (20) is per RB. That makes the per-UE total
   limit 3 W.
5. **Plant enforcement of c1-c4** (`phys`). The paper does not say how an infeasible Direct action is realised.
6. **KPI units in xTRUCE's objective** and **eta** are not given.
7. **Configuration candidates Omega^cfg** are "operator-prescribed" (p. 5) but not listed.
8. **Seed screening.** The paper's screen is not specified. Ours is `floors_reachable` [A].
9. **Not modelled:** mobility, shadowing drift, HARQ/BLER, a per-RB scheduler (paper-consistent); the OTA testbed;
   certificates and prices; the wall-clock time budget of rule (11); LLM agents.
10. **Carrier frequency and antenna gains** are not given for the simulation (2 GHz and 0 dBi assumed).
11. **Load KPI.** The paper's rho_b (eq. 18) counts allocated share, so the LB xApp reads demand load instead
    (Sec. 1.3). The paper's rho_b is still what `step` logs as `load`.

Provenance count (`XConfig` fields plus model-level choices): **32 values are [P]** (network, channel, traffic,
power, noise, change limits, policy classes, weights, tolerance, TTL, floor, 53-W cap, hallucination range). **24
field values are [A]**: 7 from named 3GPP/ITU sources (fc, BS and UE heights, street width, building height, minimum
distance, shadowing correlation), 1 from TR 38.864 (ES sleep load), 2 inherited from E6-P, 13 with no external source,
and 1 [A\*] (`qos_margin`). There are also 12 model-level [A] choices (layout, UE drop, per-RB fading, arrival reading,
default RRM, moves, xApp rules, hallucination model, xTRUCE units, Omega^cfg, the e2/e3 exemption for UEs that change
cell, the seed screen).
