"""Print knockout specs `gt:<seed>:<cell>:<t0>:reject` for finished placebo runs.

Usage: uv run python ns3/gt_specs.py RUNS_DIR PER_SEED
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    """Pick PER_SEED units per placebo seed whose 90 s windows fit, with a fixed RNG."""
    root, k = Path(sys.argv[1]), int(sys.argv[2])
    for d in sorted(root.glob("placebo_*")):
        if not (d / "units.csv").exists():
            continue
        seed = int(d.name.split("_")[1])
        u = pd.read_csv(d / "units.csv")
        u = u[(u.t0 >= 90) & (u.t0 <= 310)]
        rng = np.random.default_rng([seed, 6613])
        for i in sorted(rng.choice(len(u), min(k, len(u)), replace=False)):
            r = u.iloc[i]
            print(f"gt:{seed}:{int(r.cell)}:{round(r.t0)}:reject")


if __name__ == "__main__":
    main()
