"""Parameter registry of the xTRUCE re-implementation (arXiv:2608.28532v2, Xia et al., 2026).

Every default is either stated in the paper ([P], location given) or a declared assumption ([A], named source or
reason). The full provenance table is ``docs/benchmark/XTRUCE_SIM_SPEC.md``; the comment on each field is its short
form. Changing a default changes the benchmark: build a new ``XConfig`` instead of editing this file.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

XAPP_NAMES = ("QoS", "ES", "IC", "LB")          # [P] Sec. IV-D: four agents (QoS, energy, interference, load)


@dataclass(frozen=True)
class XConfig:
    # --- network (Table I, simulation column) ---
    n_cells: int = 4                  # [P] Table I "4 / 20 (hexagonal, 500-m ISD)"
    n_ues: int = 20                   # [P] Table I
    isd_m: float = 500.0              # [P] Table I
    n_rb: int = 12                    # [P] Table I "12 x 360 kHz"
    rb_hz: float = 360e3              # [P] Table I
    n_prot: int = 3                   # [P] Table I "3 protected users / 2 Mbps"
    rmin_bps: float = 2e6             # [P] Table I; operator rule e1, eq. (20)
    # --- operator change limits (eq. 20, e2-e5; enforced only by the xTRUCE arbiter, never by the plant) ---
    dx: float = 0.25                  # [P] Table I Delta^x (per-RB share change)
    dp_w: float = 0.25                # [P] Table I Delta^p, per RB (e2 is per (u,k)); per UE total = n_rb * dp_w
    d_act: int = 1                    # [P] Table I Delta^act (cells per configuration epoch)
    d_str: int = 3                    # [P] Table I Delta^str (users per configuration epoch)
    # --- channel ---
    pathloss: str = "36814_uma"       # [P] Table I "UMa path loss [38]", [38] = TR 36.814 V9.0.0; model = Table B.1.2.1-1 UMa
    fc_ghz: float = 2.0               # [A] ITU-R M.2135 / TR 36.814 B.1.2.1 UMa evaluation carrier (2 GHz)
    h_bs_m: float = 25.0              # [A] TR 36.814 Table B.1.2.1-1 UMa default
    h_ut_m: float = 1.5               # [A] TR 36.814 Table B.1.2.1-1 UMa default
    street_w_m: float = 20.0          # [A] TR 36.814 Table B.1.2.1-1 UMa NLOS default W
    bldg_h_m: float = 20.0            # [A] TR 36.814 Table B.1.2.1-1 UMa NLOS default h
    min_dist_m: float = 35.0          # [A] TR 36.814 Table A.2.1.1-2 minimum UE-cell distance
    shadow_db: float = 8.0            # [P] Table I "+ 8-dB shadowing" (log-normal)
    shadow_site_corr: float = 0.5     # [A] TR 36.814 Table A.2.1.1-2 shadowing correlation between sites
    fading_rho: float = 0.349         # [P] Table I "AR(1) Rayleigh (rho = 0.349) [39]"
    fading_per_rb: bool = True        # [A] independent Rayleigh per RB (paper silent)
    antenna_gain_db: float = 0.0      # [A] omni 0 dBi at both ends (paper silent; 4 omni hexagonal cells)
    # --- traffic ---
    traffic_bps: float = 6e6          # [P] Table I E[lambda_u(t)]/tau = 6 Mbps
    arrivals: str = "exp"             # [P] Table I "exponential arrivals": lambda_u(t) ~ Exp(mean traffic_bps * tau)
    epoch_s: float = 1.0              # [P] Table I tau = 1 s
    # --- power (Table I; eq. 16) ---
    p_max_w: float = 10.0             # [P] Table I P_b^max
    p_cir_w: float = 50.0             # [P] Table I P_b^cir (eq. 16, circuit power of an active cell)
    p_rb_w: float = 2.0               # [P] Table I P^rb (c4)
    noise_w: float = 1.15e-14         # [P] Table I sigma^2 per RB (= kT*360 kHz + 9 dB NF, TR 36.814)
    delta_p: float = 1.0              # [P] eq. (16) E_b = P_cir*alpha_b + sum_k P_b,k (EARTH slope 1)
    p_sleep_w: float = 0.0            # [P] eq. (16): a sleeping cell (alpha_b = 0) consumes 0 W
    # --- control timescales ---
    t_cfg: int = 10                   # [A] paper: T_cfg > 1 only; 10 epochs = 10 s (see spec)
    proposal_ttl: int = 2             # [P] Sec. V-A "each accepted proposal stays valid for two epochs"
    phys: str = "clip"                # [A] plant enforces c1-c4 by proportional rescaling ("none" = paper Direct)
    # --- operator policy (Sec. V-A) ---
    prio_prot: int = 1                # [P] Sec. V-A: protected-rate targets at kappa = 1
    prio_np_e_i: int = 2              # [P] Sec. V-A: non-protected rate, energy, interference at kappa = 2
    prio_load: int = 3                # [P] Sec. V-A: load caps at kappa = 3
    all_hard: bool = False            # [A] Sec. IV-D types (only protected rate is hard); True = Fig. 4 style
    beta: float = 1.0                 # [P] Table I target weight beta_{i,m} = 1
    eps_rel: float = 1e-4             # [P] Table I epsilon_l = 1e-4 (1 + v_l*)
    eta: float = 1e-2                 # [A] Stage-II action-change weight (paper: operator-assigned, no value)
    # --- xApps (deterministic rules; Sec. IV-D roles, thresholds [A]) ---
    xapps: tuple = XAPP_NAMES
    qos_prot_bps: float = 2e6         # [P] hard target = the Table I floor (Fig. 4 uses 3 Mbps)
    qos_margin: float = 0.30          # [A*] QoS plans 30 % above the target (hedge: interference measured at t-1)
    qos_deadband: float = 0.02        # [A] no request for share changes below 0.02 / power changes below 2 %
    es_cap_w: float = 53.0            # [P] Sec. V-B(ii): 53-W per-cell energy cap
    es_sleep_load: float = 0.30       # [A] TR 38.864 Annex A light/medium load boundary (as E6-P ES)
    es_wake_load: float = 0.80        # [A] E6-P / v1 ES wake threshold
    ic_iot_db: float = 20.0           # [A] caused-interference cap: per victim RB <= noise + ic_iot_db
    ic_min_w: float = 0.5             # [A] IC never asks for less than 0.5 W of cell power
    ic_ewma: float = 0.5              # [A] IC smooths the caused-interference gain over epochs
    lb_cap: float = 0.80              # [A] load cap theta^rho (TR 38.864 / E6-P wake threshold)
    lb_hyst: float = 0.10             # [A] move only if the target cell is 0.10 less loaded
    lb_offset_db: float = 6.0         # [A] candidate within 6 dB of the serving cell (MLB-style CIO span)
    load_ewma: float = 0.3            # [A] arrival-rate EWMA weight used by ES / LB load estimates
    hallucination: float = 0.0        # [P] Fig. 3 knob in [0, 1]; corruption model [A] (see spec)
    log: bool = True

    def with_(self, **kw) -> XConfig:
        return replace(self, **kw)


# Paper experiment presets (Sec. V-B). Values [P] unless noted.
SCENARIOS = {
    "default": {},
    # Fig. 3: hallucination sweep (600 epochs per level); pass hallucination=h.
    "fig3": {"hallucination": 0.5},
    # Fig. 4: overload sweep, three hard targets (3 Mbps protected, np target 1.5..24 Mbps, 53 W cap at kappa=3).
    "fig4": {"all_hard": True, "qos_prot_bps": 3e6, "prio_np_e_i": 2, "prio_load": 3},
    # Fig. 5 renegotiation initial intents (QoS: 20 Mbps target with 5.5 Mbps floor; energy: 50 W cap).
    "fig5_qos": {"qos_prot_bps": 20e6, "rmin_bps": 5.5e6},
    "fig5_energy": {"es_cap_w": 50.0},
}


def scenario(name: str = "default", **kw) -> XConfig:
    """Config for a paper experiment preset, with overrides. Fig. 6 scaling: ``scenario(n_cells=b, n_ues=5*b)``."""
    return XConfig(**{**SCENARIOS[name], **kw})


@dataclass
class Target:
    """One entry of a structured proposal, eq. (2): (theta_{i,m}, type_{i,m}) plus the policy's class and weight.

    ``kpi`` in {"rate", "energy", "intf", "load"}; ``idx`` = user (rate) or cell (others); ``d`` = +1 if larger is
    better (eq. 3). ``prio`` is the operator class kappa (only meaningful for hard targets)."""
    xapp: str
    kpi: str
    idx: int
    theta: float
    hard: bool
    prio: int
    beta: float = 1.0
    d: int = field(default=1)
