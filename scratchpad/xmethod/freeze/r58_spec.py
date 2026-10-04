"""R-58 EVAL scope trim as pre-registered grid choices: rewrites the blocks, arms and not_in_grid rules of
specs/eval/full.json from the arm list (idempotent).

  uv run python scratchpad/xmethod/freeze/r58_spec.py [--e4r3-fallback]

R-58 (user GO 2026-10-04; CONTRACT.md):
  (1) cdl: R1 / R2 only, n <= 4000; not in E4 R3 / R4, not in the kappa sweep.
  (2) pmrt_nl_eq: R1 / R2 at all five n (C2a keeps its 50 cells).
  (3) E4 R3: C3 readers at S_E4. FALLBACK (``--e4r3-fallback``), only if the pmrt_nl_eq n 8000 / 24000 cost pilot
      shows E4 R3 n 24000 is expensive: drop n 24000 in E4 R3 for ALL C3 readers alike (never PMRT alone); then
      C3 F_max and T9 must be re-simulated on its 16 cells (fmax_calib.py se4 restricted to n <= 8000). Not
      triggered: the measured E4 R3 n 24000 cost is ~100 CPU-s per dataset (user 2026-10-04).
  (4) E4 R4: every arm at S (not S_E4); negative control, no claim.
  (5) mscr_eq_min dropped; mscr arms out of the kappa sweep (and at n <= 1000 only, R-54).
  (6) everything else unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = os.path.join(os.path.dirname(HERE), "specs", "eval", "full.json")
sys.path.insert(0, os.path.dirname(HERE))
import eval_analysis as E  # noqa: E402

W5, NS = ["E1", "E2", "E3", "E4", "E5"], [500, 1000, 4000, 8000, 24000]
DEV_TUNE, S_E4 = [3000000, 3000019], [3100000, 3100299]
MSCR, CDL = ["mscr_eq", "mscr_native"], ["cdl"]
NS_MSCR, NS_CDL = [500, 1000], [500, 1000, 4000]
GRID_NOTE = ("R-58 grid (user GO 2026-10-04, pre-registered grid choices; cells outside an arm's grid read 'not in "
             "grid', never missing / PARTIAL / NOT EVALUABLE): (1) cdl R1 / R2 n <= 4000 only, not in E4 R3 / R4 nor "
             "the kappa sweep; (2) pmrt_nl_eq R1 / R2 at all five n; (3) E4 R3: C3 readers at S_E4 (fallback: n "
             "24000 dropped in E4 R3 for all C3 readers alike, flag 'e4r3_fallback'); (4) E4 R4: every arm at S; (5) "
             "mscr_eq_min dropped, mscr arms out of the kappa sweep (n <= 1000, R-54)")


def blk(role, worlds, regimes, ns, kappas, seeds, arms):
    return {"role": role, "worlds": worlds, "regimes": regimes, "ns": ns, "kappas": kappas, "seeds": seeds,
            "arms": arms}


def build(spec: dict, fallback: bool) -> dict:
    arms = {a: d for a, d in spec["arms"].items() if a != "mscr_eq_min"}                     # R-58 (5)
    s = {**spec, "arms": arms}
    names = list(arms)
    readers = [a for a in names if E.arm_kind(arms[a]) in ("pmrt", "eq")]                # C3 readers (R-39)
    allx = [a for a in names if a not in MSCR + CDL]
    r3_ns = NS[:-1] if fallback else NS
    s["blocks"] = [
        blk("tune", W5, ["R1", "R2", "R3", "R4"], NS, [0.25], DEV_TUNE, allx),
        blk("tune", W5, ["R1", "R2", "R3", "R4"], NS_MSCR, [0.25], DEV_TUNE, MSCR),
        blk("tune", W5, ["R1", "R2"], NS_CDL, [0.25], DEV_TUNE, CDL),
        blk("tune", W5, ["R2"], [1000], [0.125, 0.5], DEV_TUNE, allx),
        blk("measure", W5, ["R1", "R2"], NS, [0.25], "TBD", allx),
        blk("measure", W5, ["R1", "R2"], NS_MSCR, [0.25], "TBD", MSCR),
        blk("measure", W5, ["R1", "R2"], NS_CDL, [0.25], "TBD", CDL),
        blk("measure", ["E4"], ["R3"], r3_ns, [0.25], S_E4, [a for a in readers if a not in MSCR]),
        blk("measure", ["E4"], ["R3"], NS_MSCR, [0.25], S_E4, [a for a in readers if a in MSCR]),
        blk("measure", ["E4"], ["R3"], NS, [0.25], "TBD", [a for a in allx if a not in readers]),
        blk("measure", ["E4"], ["R3"], NS_MSCR, [0.25], "TBD", [a for a in MSCR if a not in readers]),
        blk("measure", ["E4"], ["R4"], NS, [0.25], "TBD", allx),
        blk("measure", ["E4"], ["R4"], NS_MSCR, [0.25], "TBD", MSCR),
        blk("measure", W5, ["R2"], [1000], [0.125, 0.5], "TBD_HALF", allx),
    ]
    rules = [
        {"ruling": "R-58(1)", "arms": CDL, "regimes": ["R3", "R4"], "ns": None, "kappas": None,
         "reason": "cdl: training cost; feeds no claim (R1 / R2, n <= 4000 only)"},
        {"ruling": "R-58(1)", "arms": CDL, "regimes": None, "ns": [8000, 24000], "kappas": None,
         "reason": "cdl: training cost; feeds no claim (R1 / R2, n <= 4000 only)"},
        {"ruling": "R-58(1)", "arms": CDL, "regimes": None, "ns": None, "kappas": [0.125, 0.5],
         "reason": "cdl: not in the kappa sweep"},
        {"ruling": "R-54", "arms": MSCR, "regimes": None, "ns": [4000, 8000, 24000], "kappas": None,
         "reason": "mscr: reported-INVALID arm, EVAL n <= 1000 only"},
        {"ruling": "R-58(5)", "arms": MSCR, "regimes": None, "ns": None, "kappas": [0.125, 0.5],
         "reason": "mscr: not in the kappa sweep"},
    ]
    if fallback:
        rules.append({"ruling": "R-58(3) fallback", "arms": readers, "regimes": ["R3"], "ns": [24000], "kappas": None,
                      "reason": "E4 R3 n 24000 dropped for every C3 reader alike (pmrt_nl_eq cost pilot)"})
    s["not_in_grid"] = rules
    s["e4r3_fallback"] = bool(fallback)
    t = dict(spec["tbd"])
    t["grid_r58"] = GRID_NOTE
    t["mscr"] = ("SETTLED (R-54, R-58(5), PROTOCOL_A T11): mscr is kept as a reported-INVALID arm (R2 truth-null raw "
                 ".93-1.00 on DEV); mscr_eq and mscr_native run n 500 / 1000 only, not in the kappa sweep; "
                 "mscr_eq_min is dropped (R-58(5)); never in C2b (mscr_eq c2b false)")
    t["seeds"] = ("measure blocks: 'TBD' -> [3100000, 3100000 + S - 1] (S by PROTOCOL_A T1); 'TBD_HALF' -> [3100000, "
                  "3100000 + ceil(S/2) - 1]; the C3 readers' E4 R3 blocks (pmrt_nl_eq, pmrt_eq, pmrt_r3, eq arms; "
                  "R-39) are filled: [3100000, 3100299], S_E4 300 by PROTOCOL_A T9; E4 R4 at S for every arm "
                  "(R-58(4))")
    t["citests_arms"] = ("mscr (eq, native) / pcorr / rcot2 arms (eq, native, eq_min) admitted under PROTOCOL_A T4; "
                         "mscr_eq_min dropped (R-58(5))")
    s["tbd"] = t
    return s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e4r3-fallback", action="store_true", help="R-58(3) fallback: no E4 R3 n 24000 for C3 readers")
    ap.add_argument("--spec", default=SPEC)
    a = ap.parse_args(argv)
    spec = json.load(open(a.spec, encoding="utf-8"))
    out = build(spec, a.e4r3_fallback)
    open(a.spec, "w", encoding="utf-8", newline="\n").write(json.dumps(out, indent=1, ensure_ascii=True) + "\n")
    print(f"ok: {len(out['arms'])} arms, {len(out['blocks'])} blocks, e4r3_fallback {out['e4r3_fallback']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
