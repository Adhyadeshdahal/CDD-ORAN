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
    extra: dict = field(default_factory=dict)

    def ue_count(self) -> int:
        return self.n_ue

    def lf(self) -> float:
        return self.load_factor or (1.8 if self.load == "medium" else 2.4)
