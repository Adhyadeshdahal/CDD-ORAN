"""E2 nonlinear label-free discovery slice (Plan 007).

A self-contained, discovery-only pipeline sibling to ``e1slice``:

    generate -> persisted rows + manifest (E2V2Env TRUE SCM, decoy OFF, noiseless)
    discover -> U-centered partial distance correlation, per-candidate permutation null,
                per-target BH-FDR -> frozen discovery.json (reads NO env truth)
    recover  -> POST-FREEZE recovery scoring vs E2 truth -> recovery.json

This slice implements the frozen contract ``docs/benchmark/E2_DISCOVERY_PROTOCOL.md``
(``protocol_commit = 828e3458065bfe27ff7af07b1295b650f89e878e`` == ``828e345`` abbreviated).
Unlike E1 there is NO train/test split and NO model-training arm: discovery consumes the whole
frozen dataset and only the discovered graph's recovery vs truth is measured.
"""

from __future__ import annotations

SCHEMA_VERSION = "e2slice.v1"
