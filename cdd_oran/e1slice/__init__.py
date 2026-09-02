"""E1 one-step-prediction vertical slice (Phase 1).

A deliberately MINIMAL, self-contained pipeline that exercises pipeline correctness
end-to-end on the E1 v2 SCM, separate from the discovery/planning stack:

    generate -> persisted rows + manifest
    split    -> episode-level train/test manifest (no transition-level leakage)
    train    -> oracle-graph and dense arms on IDENTICAL rows + split ids
    eval     -> held-out one-step prediction metrics (immutable)
    verify   -> reload artifact, re-predict, assert within a frozen tolerance

This milestone tests the plumbing, not a causal-superiority claim: E1 is the clean
recovery control (linear mechanism), so both arms are expected to fit. No causal
discovery, planning, OOD, or E5 work lives here.
"""

from __future__ import annotations

SCHEMA_VERSION = "e1slice.v1"
