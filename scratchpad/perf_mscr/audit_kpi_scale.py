"""Do the shipped designs hit the audit's numeric hazards? KPI |mean|/sd, share of tied values, constant cols."""
from __future__ import annotations

import numpy as np

from cdd_oran.e2slice.dataset import E2DatasetConfig, generate_rows
from scripts.d1_decision_study import e3_corpus
from scripts.e5_baselines import corpus as e5_corpus


def describe(name, x, y):
    for lab, a in (("x", x), ("y", y)):
        m, s = np.abs(a.mean(0)), a.std(0)
        tied = [1.0 - len(np.unique(a[:, c])) / len(a) for c in range(a.shape[1])]
        print(f"{name} {lab}: max|mean|/sd={np.max(m / np.maximum(s, 1e-300)):.3g}  min sd={s.min():.3g}  "
              f"max tied share={max(tied):.3f}")


def main():
    r = generate_rows(E2DatasetConfig(n_rows_per_seed=2000, seed=0))
    describe("E2", np.concatenate([r.x_params, r.x_kpis], 1), r.y_kpis)
    describe("E5", *e5_corpus(n=2000, seed=0))
    describe("E3", *e3_corpus(2000, 0))


if __name__ == "__main__":
    main()
