"""Pure precision/recall/F1 metric helper — NO environment / get_env dependency.

Extracted verbatim from ``threshold_sweep._prf`` so E-series consumers (recovery_metrics and the
e1slice/e2slice evaluators) can reach the precision/recall/F1 helper WITHOUT transitively importing
the retired legacy envs (``cdd_oran.envs`` eagerly loads ``env_i..iv``). ``threshold_sweep`` re-exports
``_prf`` from here so legacy callers of ``threshold_sweep._prf`` keep working unchanged.

This module imports only numpy; it must never import ``cdd_oran.envs`` or ``cdd_oran.config``.
"""

import numpy as np


def _prf(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    tp = int(np.sum((pred == 1) & (gt == 1)))
    fp = int(np.sum((pred == 1) & (gt == 0)))
    fn = int(np.sum((pred == 0) & (gt == 1)))
    tn = int(np.sum((pred == 0) & (gt == 0)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "edges": tp + fp,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }
