"""E2 causal-conflict spine — discovery mask → do-propagation → decision, in the live path.

Wires a DISCOVERED structure through the frozen E2 decision machinery and reports the decision
consequence. For each discovery method's mask, the masked world model
(``cdd_oran.benchmark.masked_world_model.MaskedE2WorldModel``) is rolled through the frozen H=1
selection oracle (``cdd_oran.analysis.v2_regret.score_grid``); the action it selects is scored under
the TRUE simulator via the frozen gate's decision-gap ``_gap_norm_for_state``. The reported
``mean_pos`` over the 32-positive bank is the normalized decision regret:

    0.0   the planner picks the oracle-optimal shared-knob action (the discovery recovered the
          harmful gated edge P0→K5, so do-propagation sees the conflict and avoids it);
    ~0.19 the planner walks into the shared-knob trap (the discovery MISSED P0→K5, so its world
          model is decoy-equivalent on the K5 term — the frozen decoy reference value).

This is the model-free live spine of ARCHITECTURE.md item 2 (structural do-propagation on the true
sim), now driven by a discovery mask rather than a hardcoded oracle/decoy. It consumes discovery
artifacts already on disk — no fresh discovery run.

Method masks read:
  - RCoT-v2  runs/e2slice-recovery/replicate-*/discovery_rcot_v2.json   (canonical E2 discovery)
  - RCoT-v1  runs/e2slice-recovery/replicate-*/discovery_rcot.json      (superseded)
  - M3       scratchpad/p0k5_fp_calibration/results.json                (stratified operating-point)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cdd_oran.analysis.v2_regret import P0_GRID, score_grid  # noqa: E402
from cdd_oran.benchmark.masked_world_model import (  # noqa: E402
    TRUE_PARAM_EDGES,
    MaskedE2WorldModel,
    param_edges_from_binary_mask,
)
from cdd_oran.envs.v2.e2 import E2V2Env  # noqa: E402
from scripts.e2_decision_gate import (  # noqa: E402
    FULL_PANEL_IDS,
    SHARED_PARAM,
    _gap_norm_for_state,
    build_bank,
    build_panel,
)

HARMFUL_EDGE = (5, 0)  # P0 → K5, the gated shared-knob edge that drives the trap


def spine_regret(param_edges, positives, panel):
    """Per-state decision regret (gap_norm) of a masked-structure planner vs the true oracle.

    Returns ``(gaps, n_action_mismatches)``: ``gaps[i]`` is the frozen decision-gap for bank state
    ``i`` when the planner believes ``param_edges``; ``n_action_mismatches`` counts states where the
    masked planner's argmax P0 differs from the true-SCM oracle's argmax.
    """
    edges = frozenset((int(k), int(p)) for (k, p) in param_edges)
    gaps: list[float] = []
    mism = 0
    for seed, snap in positives:
        true_env = E2V2Env(env_seed=seed)
        wm = MaskedE2WorldModel(edges, env_seed=seed)
        gaps.append(_gap_norm_for_state(true_env, wm, snap, panel)[0])
        s_true = score_grid(E2V2Env(env_seed=seed), snap, SHARED_PARAM, panel)
        s_wm = score_grid(MaskedE2WorldModel(edges, env_seed=seed), snap, SHARED_PARAM, panel)
        mism += int(np.argmax(s_wm) != np.argmax(s_true))
    return gaps, mism


def donothing_regret(positives, panel):
    """Do-nothing floor: normalized regret if the planner never moves the shared knob (holds the
    committed P0). ``gap = (max_v R_true − R_true(P0=committed)) / D(s)`` — the same normalization the
    frozen gate uses. This is the baseline every method must beat to justify acting at all.
    """
    gaps = []
    for seed, snap in positives:
        env = E2V2Env(env_seed=seed)
        s_true = score_grid(env, snap, SHARED_PARAM, panel)
        g_star = float(s_true.max())
        d = g_star - float(s_true.min())
        committed_p0 = float(snap[1][0])
        r_dn = float(score_grid(env, snap, SHARED_PARAM, panel, grid=np.array([committed_p0]))[0])
        gaps.append((g_star - r_dn) / d if d > 0 else 0.0)
    return gaps


def load_pdcor_masks(repo_root=_REPO_ROOT):
    """replicate -> live param→KPI edge set from pdCor (``discovery.json``) binary masks on disk."""
    return load_rcot_masks("discovery.json", repo_root)


def load_rcot_masks(filename, repo_root=_REPO_ROOT):
    """replicate_index -> live param→KPI edge set from RCoT ``binary_mask`` artifacts on disk."""
    out: dict[int, frozenset] = {}
    pattern = os.path.join(repo_root, "runs", "e2slice-recovery", "replicate-*", filename)
    for path in sorted(glob.glob(pattern)):
        rep = int(os.path.basename(os.path.dirname(path)).split("-")[1])
        rec = json.load(open(path))
        out[rep] = param_edges_from_binary_mask(rec["binary_mask"])
    return dict(sorted(out.items()))


def load_m3_masks(repo_root=_REPO_ROOT):
    """seed -> live param→KPI edge set from the M3 stratified-CI calibration records."""
    path = os.path.join(repo_root, "scratchpad", "p0k5_fp_calibration", "results.json")
    recs = json.load(open(path))["records"]
    out: dict[int, set] = {}
    for r in recs:
        if r["bh_declared"] and r["candidate"] < E2V2Env.num_params:
            out.setdefault(int(r["seed"]), set()).add((int(r["target"]), int(r["candidate"])))
    return {k: frozenset(v) for k, v in sorted(out.items())}


def summarize(name, masks, positives, panel):
    """Run every mask through the spine; print + return a per-method summary."""
    pooled: list[float] = []
    harmful_hits = 0
    total_mism = 0
    per_mask = {}
    for key, edges in masks.items():
        gaps, mism = spine_regret(edges, positives, panel)
        has_harmful = HARMFUL_EDGE in edges
        harmful_hits += int(has_harmful)
        total_mism += mism
        pooled.extend(gaps)
        per_mask[str(key)] = {
            "mean_pos": float(np.mean(gaps)),
            "has_harmful_edge": has_harmful,
            "action_mismatches": mism,
            "n_edges": len(edges),
        }
    n = len(masks)
    summary = {
        "n_masks": n,
        "harmful_recovered": harmful_hits,
        "harmful_recovery_rate": harmful_hits / n if n else float("nan"),
        "pooled_mean_pos": float(np.mean(pooled)) if pooled else float("nan"),
        "action_mismatches_total": total_mism,
        "per_mask": per_mask,
    }
    print(f"\n=== {name}  ({n} masks) ===")
    print(f"  P0->K5 recovered {harmful_hits}/{n}   pooled mean_pos(gap_norm)="
          f"{summary['pooled_mean_pos']:.6f}   action-mismatches(total)={total_mism}")
    for key, m in per_mask.items():
        flag = "" if m["has_harmful_edge"] else "  <-- MISSES P0->K5"
        print(f"    mask {key:>2}: mean_pos={m['mean_pos']:.6f}  edges={m['n_edges']:2d}  "
              f"mismatch={m['action_mismatches']}{flag}")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, help="optional path to write the JSON summary")
    args = ap.parse_args(argv)

    bank = build_bank(E2V2Env)
    if not bank.feasible:
        print("bank infeasible", file=sys.stderr)
        return 1
    positives = bank.positives
    panel = build_panel(E2V2Env(0), FULL_PANEL_IDS)
    print(f"Frozen bank: {len(positives)} positives; panel xApps {FULL_PANEL_IDS}; "
          f"shared knob P{SHARED_PARAM}")

    oracle_gaps, oracle_mism = spine_regret(TRUE_PARAM_EDGES, positives, panel)
    decoy_gaps, decoy_mism = spine_regret(TRUE_PARAM_EDGES - {HARMFUL_EDGE}, positives, panel)
    donothing_gaps = donothing_regret(positives, panel)
    print("\n=== reference arms ===")
    print(f"  oracle (16 true edges): mean_pos={np.mean(oracle_gaps):.6f}  mismatch={oracle_mism}")
    print(f"  do-nothing floor      : mean_pos={np.mean(donothing_gaps):.6f}"
          f"   (never move the shared knob)")
    print(f"  decoy  (omit P0->K5)  : mean_pos={np.mean(decoy_gaps):.6f}  mismatch={decoy_mism}"
          f"   (frozen trap ref 0.190)")

    out = {
        "reference": {
            "oracle_mean_pos": float(np.mean(oracle_gaps)),
            "donothing_mean_pos": float(np.mean(donothing_gaps)),
            "decoy_mean_pos": float(np.mean(decoy_gaps)),
        },
        "methods": {
            "M3": summarize("M3 (stratified operating-point)", load_m3_masks(), positives, panel),
            "RCoT_v2": summarize(
                "RCoT-v2 (canonical E2)", load_rcot_masks("discovery_rcot_v2.json"), positives, panel
            ),
            "RCoT_v1": summarize(
                "RCoT-v1 (superseded)", load_rcot_masks("discovery_rcot.json"), positives, panel
            ),
            "pdCor": summarize(
                "pdCor (partial distance corr, retired)", load_pdcor_masks(), positives, panel
            ),
        },
    }
    if args.out:
        json.dump(out, open(args.out, "w"), indent=2)
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
