"""E6 configuration and parameter registry.

Every numeric default carries a provenance tag in its comment:
  [S] sourced (document/table named), [A] modelling assumption, [V] sourced but still to verify against the document.
Design reference: scratchpad/e6_design/E6_SPEC_DRAFT.md (compiled from three independent designs).
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------------------------------- timing
TICK_S = 0.04                 # [A] single simulation tick (radio, HO, scheduler, queues); TTT quantised to it
CONTROL_S = 1.0               # [S] near-RT control interval, upper end of 10 ms-1 s (O-RAN WG3 RICARCH)
TICKS_PER_CONTROL = int(round(CONTROL_S / TICK_S))

# ---------------------------------------------------------------------------------------------------- radio
FREQ_GHZ = 2.0                # [V] TS 38.101-1 band n1-like, co-channel macro + pico
BW_HZ = 20e6
N_PRB = 106                   # [S] TS 38.101-1 Tab 5.3.2-1 (20 MHz, 15 kHz)
PRB_HZ = 180e3
OVERHEAD = 0.25               # [A] control/RS overhead
SE_MAX = 4.4                  # [S] TR 36.942 A.2 truncated Shannon ceiling
SE_ALPHA = 0.6                # [S] TR 36.942 A.2
SINR_MIN_DB = -10.0           # [S] TR 36.942 A.2
NOISE_DBM_HZ = -174.0         # [S]
UE_NF_DB = 9.0                # [S] TR 36.814
MACRO_PTX_DBM = 46.0          # [V] TR 36.814 Tab A.2.1.1-2
PICO_PTX_DBM = 30.0           # [V] TR 36.872
MACRO_GAIN_DBI = 14.0         # [S] TR 36.814 Case 1 BS antenna gain (incl. cable loss folded in [A])
PICO_GAIN_DBI = 5.0           # [A]
MACRO_H, PICO_H, UE_H = 25.0, 10.0, 1.5   # [S] TR 38.901 Tab 7.2-1
HPBW_DEG, AM_DB = 70.0, 20.0  # [S] TR 36.814 Case 1 2D sector pattern (calibration reference)
VERTICAL = False              # [S] 36.814 Case 1 uses no vertical pattern; True -> 38.901-style below
VBW_DEG, SLAV_DB, TILT_DEG = 10.0, 30.0, 12.0   # [A] used only if VERTICAL
ISD_M = 500.0                 # [S] TR 36.814 Case 1
SF_SIGMA_MACRO = (4.0, 6.0)   # [S] TR 38.901 Tab 7.5-6 UMa LOS/NLOS
SF_SIGMA_PICO = (4.0, 7.82)   # [S] UMi-SC LOS/NLOS
SF_DECORR_M = 50.0            # [S] ~37/50 m; one value for the grid filter [A]
O2I_DB = 20.0                 # [A] simplified O2I penetration
GRID_M = 10.0                 # [A] gain-map resolution

# ---------------------------------------------------------------------------------------------------- mobility / HO
L3_K = 4                      # [S] TS 38.331 filterCoefficient (a = 1/2^(k/4) = 0.5)
MEAS_ERR_DB = 2.0             # [A] within TS 38.133 relative accuracy
HO_EXEC_S = 0.05              # [A] HO execution / interruption
HO_FAIL_SINR_DB = -8.0        # [A]
QOUT_DB, QIN_DB = -8.0, -6.0  # [V] TR 36.839 sim assumptions
T310_S = 1.0                  # [V]
RLF_OUTAGE_S = 1.0            # [A] re-establishment outage
PINGPONG_S = 1.0              # [S] TR 36.839 MTS
TOO_EARLY_S = 1.0             # [S]
TTT_SET_MS = (40, 80, 160, 256, 320, 480, 640)   # [S] TS 38.331 subset
CIO_RANGE = (-6.0, 6.0)       # [S] subset of Q-OffsetRange
HYS_RANGE = (0.0, 5.0)

# ---------------------------------------------------------------------------------------------------- energy (EARTH)
MACRO_P0_W, MACRO_DP, MACRO_PMAX_W, MACRO_SLEEP_W, MACRO_NTRX = 130.0, 4.7, 20.0, 75.0, 2   # [S] Auer 2011, per sector
PICO_P0_W, PICO_DP, PICO_PMAX_W, PICO_SLEEP_W = 6.8, 4.0, 0.13, 4.3                       # [S]
PICO_WAKE_S = 5.0             # [A]
# Macro carrier shutdown (the ES knob): each macro sector runs MACRO_NTRX co-located component carriers of equal width
# (a coverage carrier + capacity carrier(s)); shutting one leaves coverage unchanged but removes its PRBs. An idle
# (switched-off) carrier draws MACRO_SLEEP_W / MACRO_NTRX. Interference is load-coupled through the occupied share of
# the full band (used PRBs / N_PRB) [A: carrier-averaged approximation, not per-carrier SINR].
CARRIER_ON_S = 2.0            # [A] reactivation delay before the capacity carrier serves traffic

# ---------------------------------------------------------------------------------------------------- traffic / SLA
LL_PKT_BYTES = 200            # [A]
LL_RATE_PPS = (50.0, 200.0)   # [A] per LL UE, drawn per UE
LL_DELAY_TARGET_S = 0.10      # [A] coarse RAN delay target (>= 2.5 ticks; NOT a 3GPP URLLC guarantee)
EMBB_FILE_BYTES = 0.5e6       # [S] FTP model 3 (TR 36.814)
EMBB_THP_TARGET_BPS = 2e6     # [A] user-perceived throughput target while backlogged
HOL_PROC_S = 0.004            # [A] HARQ / processing


# ---------------------------------------------------------------------------------------------------- stress scenarios
# Reserved HOLDOUT variants (E6_STRESS_SCENARIOS.md). Each changes ONE factor (timing / location / intensity) of the
# DEV default; they are declared here, not used for any development run, and never tuned.
SCENARIO_VERSION = "E6-scn-v1"   # bump on ANY change to scenario mechanics/parameters (see E6_SCENARIO_CONTRACT.json)
SCENARIO_NAMES = ("base", "surge", "mistune")
SCENARIO_HOLDOUT = {
    "surge": {
        "H-time": {"surge_onset_frac": 0.45, "surge_ramp_frac": 0.05},     # later, sharper flash crowd
        "H-loc": {"surge_loc": "vertex"},                                    # at a 3-site corner
        "H-int": {"surge_mult": 4.0},                                        # [A] above the sourced DL x3
        "H-move": {"surge_speed_mps": 3.0 / 3.6},                            # [S] pedestrian 3 km/h moving crowd
    },
    "mistune": {
        "H-time": {"mis_onset_frac": 0.35},
        "H-loc": {"corr_loc": "vertex"},
        "H-int": {"mis_hys_db": 5.0, "mis_ttt_ms": 640},                     # [A] E6 actuator maxima
        "H-dir": {"mis_hys_db": 0.0, "mis_ttt_ms": 40},                      # [A] approximation of 36.839 Set 5
        # (TTT 40 ms [S]; Set 5's A3 offset -1 dB cannot be expressed as Hys >= 0, clipped to Hys 0 dB: NOT Set 5)
    },
}


@dataclass(frozen=True)
class E6PConfig:
    """E6-P: additive power / protected-slice extension (default OFF; with every flag False and no xApps the plant, RIC,
    knob list and every RNG stream are bit-identical to E6-scn-v1). Values and provenance: docs/benchmark/E6P_SPEC.md
    (section in brackets); the field names are that document's interface.

    Tx power ("ptx", c): total-carrier Tx-power offset in dB on cell c's whole gain column (0 = nominal
    MACRO/PICO_PTX_DBM), so serving RSRP, SINR, measured L3 (hence the A3 border) and the interference c causes all
    follow. EARTH power per active carrier: P0 + DP * rho * PMAX * 10^(offset/10) (P0 does not scale; offset > 0 is an
    extrapolation beyond EARTH's P_out <= PMAX, flagged not corrected).
    Protected slice: a fraction of eMBB UEs carries a per-UE rate floor; ("prot_min", c) is a WORK-CONSERVING minimum
    PRB share (TS 28.541 rRMPolicyMinRatio: guaranteed when protected UEs need it, borrowed otherwise).
    Pico sleep (only when enabled): a sleeping pico hands its served UEs over to their best remaining cell
    (``pico_sleep_ho``) and the v1 "ES" xApp gains a pico-sleep part (``es_pico``; xapps_p.ESPico) [spec 4.1, 6.1-6.2].
    """
    # ---- Tx-power knob [spec 2.1, 3]
    ptx_on: bool = False
    ptx_scope: str = "macro"              # [S->] macro sectors only (EARTH: pico load dependency negligible) | "all"
    ptx_range_db: tuple = (-9.0, 3.0)     # [S] TR 38.864 Tab 6.4.1.2-1 -9 dB; Tab 5.1-1 Set 2 49 dBm = +3 dB
    ptx_init_db: float = 0.0              # [I] E6 nominal 46 dBm
    ptx_grid_db: float = 1.0              # [S] TS 38.331 integer-dB power parameters
    ptx_max_step_db: float = 3.0          # [S->] one TR 38.864 evaluation step per write
    ptx_min_interval_s: float = 10.0      # [A] = power xApps' cadence
    # ---- protected slice (subset of eMBB UEs) + work-conserving min PRB share [spec 2.2-2.3, 3]
    prot_on: bool = False
    prot_frac: float = 0.2                # [A] share of eMBB UEs flagged protected (own RNG stream "e6p")
    prot_floor_bps: float = 2e6           # [S]+[I] TR 26.925 720p lower end = E6 EMBB_THP_TARGET_BPS
    prot_min_backlog_s: float = 0.2       # [I] E6 eMBB SLA activity gate
    prot_min_range: tuple = (0.0, 0.5)    # [I] E6 ll_ratio actuator range
    prot_min_init: float = 0.0            # [S] TS 28.541 rRMPolicyMinRatio defaultValue 0
    prot_min_grid: float = 0.05           # [I] E6 ll_ratio quantum
    prot_min_max_step: float = 0.10       # [I] E6 ll_ratio max step
    prot_min_interval_s: float = 5.0      # [A] = SliceGuarantee cadence
    edge_pct: float = 5.0                 # [S] ITU-R M.2410 5th-percentile (edge) user
    # ---- pico sleep (effective only when the extension is enabled) [spec 4.1, 6.1-6.2]
    pico_sleep_ho: bool = True            # a sleeping pico hands its UEs over (HO_EXEC_S interruption), no forced RLF
    es_pico: bool = True                  # the "ES" xApp of the base mix also sleeps / wakes picos (xapps_p.ESPico)
    # ---- E6-P xApps appended after the base mix (names from xapps_p.P_XAPPS); () = none
    xapps: tuple = ()
    # PowerES: lower ptx on lightly loaded macros, restore when busy [spec 4.2]
    pes_cadence_s: float = 10.0           # [I] v1 ES cadence
    pes_u_low: float = 0.30               # [I]/[S->] full-band load below which the cell is light
    pes_u_high: float = 0.80              # [I] v1 ES u_on: ACTIVE-capacity util above which power is restored
    pes_hold_s: float = 60.0              # [I] v1 ES low-load spell
    pes_step_db: float = 3.0              # [S->] one TR 38.864 grid step (floor = ptx_range_db[0])
    # Coverage: raise ptx where the edge SINR is weak (absolute and relative to neighbours) or protected floors fail;
    # release a positive offset toward nominal when comfortable [spec 4.3]
    cov_cadence_s: float = 10.0           # [A] same clock as PowerES
    cov_sinr_low_db: float = -6.0         # [S->] Q_in: edge-SINR percentile below which power is raised
    cov_sinr_ok_db: float = -3.0          # [S->] Q_in + one step: above it a positive offset is released
    cov_margin_db: float = 3.0            # [S->] raise only if the edge is >= one step below the neighbours' mean edge
    cov_floor_frac: float = 0.05          # [S->] protected below-floor share that also triggers a raise
    cov_step_db: float = 3.0              # [S->] TR 38.864 grid; ceiling ptx_range_db[1]
    # SliceGuarantee: raise prot_min when protected floors are violated, lower when the share is slack [spec 4.4]
    sg_cadence_s: float = 5.0             # [I] v1 SLICE decision window
    sg_viol_hi: float = 0.05              # [S->] 95 % availability rule
    sg_viol_lo: float = 0.01              # [A] release hysteresis (1/5 of the trigger)
    sg_slack: float = 0.5                 # [I] v1 SLICE "reservation mostly wasted" ratio
    sg_hold_s: float = 10.0               # [I] v1 SLICE good spell
    sg_step: float = 0.05                 # [I] v1 SLICE step

    @property
    def enabled(self) -> bool:
        return bool(self.ptx_on or self.prot_on or self.xapps)


@dataclass(frozen=True)
class E6Config:
    seed: int = 0
    # scenario factors
    load: str = "medium"          # medium | high
    mobility: str = "ped"         # ped (90/10) | mixed (60/40 vehicles)
    mix: str = "M2"               # M2 = MRO+TS+ES | M4 = M2 + Slice | M_TS_MRO = MRO+TS only
    update: bool = False          # one hidden xApp version update
    kpm: str = "nominal"          # nominal | degraded
    # sizes
    n_ue: int = 300               # [A] fixed population; load level scales the offered traffic
    n_pico: int = 3               # [A] 3 co-channel picos (one per hotspot cluster)
    warmup_s: float = 300.0       # [A]
    scored_s: float = 1800.0      # [A]
    # slice mix (fractions of UEs) [A]
    frac_ll: float = 0.15
    frac_embb: float = 0.55
    # traffic
    embb_files_per_s: float = 0.18    # [A] per eMBB UE, ~0.7 Mb/s offered (scaled by load factor and hidden m)
    be_files_per_s: float = 0.11      # [A] ~0.44 Mb/s offered
    load_factor: float = 0.0          # 0 -> from load: medium 1.8, high 2.4 (x ramp 0.4->1.0) [A, DEV-calibrated]
    ramp: tuple = (0.4, 1.0)          # [A] compressed diurnal ramp of offered traffic over the episode
    m_tau_s: float = 300.0            # [A] log-OU hidden load, per cell
    m_sigma: float = 0.3              # [A]
    n_hotspots: int = 3               # [A]
    hotspot_sigma_m: float = 60.0     # [A]
    # KPM
    kpm_delay_s: tuple = (0.1, 0.5)   # [A] nominal; degraded (2, 5) + 5% drop
    kpm_drop: float = 0.0
    # plant-family perturbation (TEST only; 0 = nominal DEV plant) [A]
    pl_offset_db: float = 0.0
    xapp_threshold_scale: float = 1.0
    # stress scenarios (docs/benchmark/E6_STRESS_SCENARIOS.md; fixed before any arbitration outcome was inspected).
    # Values below are the DEV defaults; reserved HOLDOUT values are in SCENARIO_HOLDOUT. Times are fractions of the
    # scored window (onset = warmup_s + frac * scored_s) so the event always falls inside scoring.
    scenario: str = "base"            # base (calm E6, unchanged) | surge (S1, MLB) | mistune (S2, MRO)
    # S1 surge: event / moving-hotspot traffic surge (TR 36.902 cl. 4.6 MLB use case)
    surge_mult: float = 3.0           # [A] per-UE file-rate x3 inside the disk, informed by an [S] AGGREGATE ratio
    #                                   (Shafiq et al. SIGMETRICS'13: DL volume and #users x3 over sectors within 1 mile)
    surge_radius_m: float = 250.0     # [A] venue-scale disk radius (= ISD/2); not sourced
    surge_loc: str = "band"           # [A] band: centre 150-250 m from a random macro site (as E6 hotspots) | vertex
    surge_onset_frac: float = 0.2     # [A] crowd starts arriving 20 % into the scored window
    surge_ramp_frac: float = 0.1      # [A] linear ramp up and down (compressed crowd arrival / egress)
    surge_hold_frac: float = 0.3      # [A] plateau (9 min of a 30 min scored window)
    surge_speed_mps: float = 0.0      # [A] 0 = static venue; HOLDOUT moving crowd at pedestrian 3 km/h
    # S2 mistune: HO-parameter mis-tuning on the cluster a high-speed corridor crosses (TR 36.902 cl. 4.5 MRO)
    corr_frac_ue: float = 0.10        # [A] share of UEs on the corridor (~ vehicular share of mobility=mixed, 12 %)
    corr_speed_kmh: float = 120.0     # [S] TR 36.839 Tab 5.2.4.1 highest UE speed; ITU-R M.2410 cl. 4.11 vehicular max
    corr_len_m: float = 1000.0        # [A] road section = 2 ISD, crosses several cell borders; U-turn at both ends
    corr_loc: str = "edge"            # [A] edge: centred midway between two adjacent macro sites | vertex
    mis_onset_frac: float = 0.1       # [A] parameter rollout 10 % into the scored window
    mis_hys_db: float = 3.0           # [A] mapping: TR 36.839 Tab 5.3.2.1 Set 1 A3 offset 3 dB [S] carried by Hys
    mis_ttt_ms: int = 480             # [S] TR 36.839 Tab 5.3.2.1 Set 1 TTT 480 ms (the numeric pair only; the
    #                                   rollout on this road is a too-late-HO hypothesis, not a documented incident)
    extra: dict = field(default_factory=dict)
    # E6-P extension (default OFF: E6PConfig() leaves E6-scn-v1 bit-identical)
    e6p: E6PConfig = field(default_factory=E6PConfig)

    def ue_count(self) -> int:
        return self.n_ue

    def lf(self) -> float:
        return self.load_factor or (1.8 if self.load == "medium" else 2.4)

    def validate_scenario(self) -> None:
        """Reject invalid stress-scenario parameters (called by the scenario builders; ``base`` ignores them)."""
        def need(ok, msg):
            if not ok:
                raise ValueError(f"E6 scenario {self.scenario!r}: {msg}")

        need(self.scenario in SCENARIO_NAMES, f"unknown scenario (known: {SCENARIO_NAMES})")
        need(self.warmup_s >= 0 and self.scored_s > 0, "need warmup_s >= 0 and scored_s > 0")
        if self.scenario == "surge":
            need(self.surge_mult >= 1.0, "surge_mult must be >= 1")
            need(self.surge_radius_m > 0, "surge_radius_m must be > 0")
            need(self.surge_speed_mps >= 0, "surge_speed_mps must be >= 0")
            need(self.surge_loc in ("band", "vertex"), "surge_loc must be 'band' or 'vertex'")
            for f in ("surge_onset_frac", "surge_ramp_frac", "surge_hold_frac"):
                need(0.0 <= getattr(self, f) <= 1.0, f"{f} must be in [0, 1]")
            need(self.surge_onset_frac + 2 * self.surge_ramp_frac + self.surge_hold_frac <= 1.0 + 1e-12,
                 "the surge must end by the end of the scored window (onset + 2 ramp + hold <= 1)")
        elif self.scenario == "mistune":
            need(0.0 < self.corr_frac_ue <= 1.0 and round(self.corr_frac_ue * self.n_ue) >= 1,
                 "corr_frac_ue must be in (0, 1] and put >= 1 UE on the corridor")
            need(self.corr_speed_kmh > 0, "corr_speed_kmh must be > 0")
            need(0 < self.corr_len_m <= 7 * ISD_M, "corr_len_m must be in (0, 7 ISD]")
            need(self.corr_loc in ("edge", "vertex"), "corr_loc must be 'edge' or 'vertex'")
            need(0.0 <= self.mis_onset_frac < 1.0, "mis_onset_frac must be in [0, 1)")
            need(HYS_RANGE[0] <= self.mis_hys_db <= HYS_RANGE[1] and abs(self.mis_hys_db * 2 - round(self.mis_hys_db * 2)) < 1e-9,
                 f"mis_hys_db must be on the 0.5 dB grid within {HYS_RANGE}")
            need(int(self.mis_ttt_ms) == self.mis_ttt_ms and int(self.mis_ttt_ms) in TTT_SET_MS,
                 f"mis_ttt_ms must be one of {TTT_SET_MS}")
