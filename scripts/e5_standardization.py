"""E5 frozen standardization moments (E2 pattern: n=1e6 empirical, seed 0; E4 pattern: Z at its mean).

Reference distribution: params i.i.d. Uniform over E5V2Env.id_ranges; the lagged K_mid feeding the
gated chain term comes from an INDEPENDENT previous param draw (K_mid_prev = C_CHAIN * P0_prev);
composed default layers (subdom=0.2, chain_gamma=0.2), theta=0 (latent Z integrated to its mean,
the scoring convention), no noise. Prints per-KPI (mean, std) for pasting into GATE_CONTRACT_E5.md.
"""
import numpy as np

from cdd_oran.envs.v2.e5 import E5V2Env

N, SEED = 1_000_000, 0


def main():
    rng = np.random.default_rng(SEED)
    env = E5V2Env(env_seed=0, theta=0.0)
    lo, hi = np.array(E5V2Env.id_ranges, dtype=float).T
    params = rng.uniform(lo, hi, size=(N, E5V2Env.num_params))
    p0_prev = rng.uniform(lo[E5V2Env.P0], hi[E5V2Env.P0], size=N)
    k_prev = np.zeros(E5V2Env.num_kpis)
    out = np.empty((N, E5V2Env.num_kpis))
    for r in range(N):
        k_prev[E5V2Env.K_MID] = E5V2Env.C_CHAIN * p0_prev[r]
        out[r] = env._update_kpis(params[r], k_prev)
    names = ["K_ben", "K_harm", "K_mid", "K_dist"]
    for k in range(E5V2Env.num_kpis):
        print(f"K{k} {names[k]:6s}: ({out[:, k].mean():.6f}, {out[:, k].std():.6f})")


if __name__ == "__main__":
    main()
