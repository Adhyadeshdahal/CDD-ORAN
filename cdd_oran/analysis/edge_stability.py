"""Bootstrap edge aggregation / stability selection for a trained CDL causal graph.

No retraining: the trained predictor is FROZEN. A pool of transitions is collected
with a random policy. For each bootstrap resample of the pool the CMI matrix is
recomputed through the frozen model's existing mask-update path, thresholded, and
the per-edge selection frequency is aggregated over the B resamples. A frequency
cut then selects the stability graph, scored against ``env.true_adj_matrix``.

The selection frequency is a calibrated per-edge confidence: a strong true edge is
selected in almost every resample, pure noise in almost none, and weak true edges
sit in a mid-frequency band that a single hard CMI cut cannot separate.

Usage:
    uv run python -m cdd_oran.analysis.edge_stability --run RUN_DIR [--B 50] \\
        [--n-transitions 2048] [--pi 0.5] [--seed 0] [--device cuda] [--out OUT.json]
    uv run python -m cdd_oran.analysis.edge_stability --self-check
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from cdd_oran.analysis.threshold_sweep import _prf
from cdd_oran.config import load_config
from cdd_oran.conflicts import state_to_tensor
from cdd_oran.envs import get_env
from cdd_oran.models import get_model
from cdd_oran.policies import RandomPolicy
from cdd_oran.utils.seeding import seed_everything


def collect_transitions(env, policy, n):
    """Roll a random policy for ``n`` steps and return (s, a, s_next) tensors."""
    s_list, a_list, s_next_list = [], [], []
    obs = env.reset()
    while len(s_list) < n:
        action = policy.act()
        next_obs, _, done, _ = env.step(action)
        s_list.append(state_to_tensor(obs))
        a_list.append(torch.as_tensor(action, dtype=torch.float32))
        s_next_list.append(state_to_tensor(next_obs))
        obs = next_obs
        if done:
            obs = env.reset()
    return torch.stack(s_list), torch.stack(a_list), torch.stack(s_next_list)


def cmi_on_resample(model, s, a, s_next, idx, batch_size):
    """Recompute the CMI matrix on a bootstrap resample (frozen predictor).

    Uses the model's existing ``update_mask`` step-CMI path but disables the
    training-time EMA decay (``eval_tau``) so each resample yields a fresh,
    unscaled mean CMI over its batches. ``_eval_cmi_acc`` therefore holds the
    raw sum of step CMI, and ``mask_CMI`` is left untouched.
    """
    device = model.device
    s_pair = torch.stack([s[idx], s_next[idx]], dim=1).to(device)
    a_r = a[idx].to(device)
    n_batches = (len(idx) + batch_size - 1) // batch_size
    model._eval_cmi_acc.zero_()
    for i in range(n_batches):
        model._eval_step_count = 0  # keep the EMA from firing; accumulate step CMI only
        batch = slice(i * batch_size, (i + 1) * batch_size)
        model.update_mask(s_pair[batch], a_r[batch])
    cmi = (model._eval_cmi_acc / n_batches).cpu().detach().numpy()
    model._eval_cmi_acc.zero_()
    return cmi


def binary_graph_from_cmi(cmi, threshold):
    """Threshold a (fd, fd+1) CMI matrix into a (fd, fd) binary graph."""
    binary = cmi >= float(threshold)
    pred = binary[:, :-1].copy()
    np.fill_diagonal(pred, 0)
    return pred.astype(int)


def _snapshot_mask_state(model):
    return (
        model.mask_CMI.detach().clone(),
        model._eval_cmi_acc.detach().clone(),
        model._eval_step_count,
    )


def _restore_mask_state(model, snapshot):
    cmi, acc, count = snapshot
    model.mask_CMI.copy_(cmi)
    model._eval_cmi_acc.copy_(acc)
    model._eval_step_count = count


def edge_stability(run_dir, B=50, n_transitions=2048, pi=0.5, seed=0, device=None):
    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "config.yaml")
    if device:
        cfg = replace(cfg, device=device)
    if cfg.model_kind != "cdl":
        raise ValueError("edge_stability requires a CDL run (model_kind=cdl)")

    seed_everything(seed, deterministic=False)
    env = get_env(cfg)
    model = get_model(cfg, env)
    model.load_model(run_dir / "checkpoint.pt")

    state_dim = env.get_state_dim()
    policy = RandomPolicy(action_dim=env.action_dim, action_space=env.action_space)
    s, a, s_next = collect_transitions(env, policy, n_transitions)

    rng = np.random.default_rng(seed)
    batch_size = cfg.model.batch_size
    freq = np.zeros((state_dim, state_dim))
    snapshot = _snapshot_mask_state(model)
    try:
        for _ in range(B):
            idx = rng.integers(0, n_transitions, size=n_transitions)
            cmi = cmi_on_resample(model, s, a, s_next, idx, batch_size)
            freq += binary_graph_from_cmi(cmi, cfg.model.cmi_threshold)
    finally:
        _restore_mask_state(model, snapshot)
    freq = freq / B

    gt = env.true_adj_matrix
    table = []
    for cut in np.arange(0.1, 1.0 + 1e-9, 0.1):
        selected = (freq >= cut).astype(int)
        table.append({"cut": float(cut), **_prf(selected, gt)})

    baseline_cmi = model.mask_CMI.cpu().detach().numpy()
    baseline_pred = binary_graph_from_cmi(baseline_cmi, cfg.model.cmi_threshold)
    baseline = _prf(baseline_pred, gt)

    pi_selected = (freq >= pi).astype(int)
    return {
        "run_dir": str(run_dir),
        "environment": cfg.environment,
        "config_threshold": cfg.model.cmi_threshold,
        "B": B,
        "n_transitions": n_transitions,
        "batch_size": batch_size,
        "seed": seed,
        "device": cfg.device,
        "gt_edge_count": int(gt.sum()),
        "baseline_single_threshold": {"threshold": cfg.model.cmi_threshold, **baseline},
        "pi_selected": {"cut": pi, **_prf(pi_selected, gt)},
        "frequency_cut_table": table,
        "frequency_matrix": freq,
    }


def _self_check():
    """Deterministic stub: strong edge frequency ~1.0, pure noise stays low."""
    fd = 3
    pool = 256
    batch_size = 32
    n_strong = 204  # 80% of the pool

    s = torch.zeros(pool, fd)
    s[:n_strong, 0] = 1.0
    a = torch.zeros(pool, 3)
    s_next = torch.zeros(pool, fd)

    class StubModel:
        device = torch.device("cpu")
        eval_steps = 10
        eval_tau = 0.99

        def __init__(self):
            self.mask_CMI = torch.zeros(fd, fd + 1)
            self._eval_cmi_acc = torch.zeros(fd, fd + 1)
            self._eval_step_count = 0

        def update_mask(self, s_batch, a_batch):
            strong = s_batch[:, 0, 0].mean().item()
            self._eval_cmi_acc[0, 1] += strong
            self._eval_cmi_acc[0, 2] += (1.0 - strong) * 0.2
            self._eval_step_count += 1
            if self._eval_step_count >= self.eval_steps:
                avg = self._eval_cmi_acc / self.eval_steps
                self.mask_CMI = self.eval_tau * self.mask_CMI + (1 - self.eval_tau) * avg
                self._eval_cmi_acc.zero_()
                self._eval_step_count = 0

        def get_causal_graph(self):
            return self.mask_CMI

    def run(seed):
        rng = np.random.default_rng(seed)
        model = StubModel()
        freq = np.zeros((fd, fd))
        for _ in range(50):
            idx = rng.integers(0, pool, size=pool)
            cmi = cmi_on_resample(model, s, a, s_next, idx, batch_size)
            freq += binary_graph_from_cmi(cmi, 0.5)
        return freq / 50

    first = run(0)
    second = run(0)
    assert np.array_equal(first, second), "selection is not deterministic under a fixed seed"
    assert first[0, 1] > 0.95, f"strong edge frequency {first[0, 1]:.3f} should be ~1.0"
    assert first[0, 2] < 0.05, f"noise edge frequency {first[0, 2]:.3f} should stay low"
    print(
        f"edge_stability self-check passed: strong ~{first[0, 1]:.3f}, "
        f"noise {first[0, 2]:.3f}, deterministic"
    )


def _jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run")
    p.add_argument("--B", type=int, default=50)
    p.add_argument("--n-transitions", type=int, default=2048)
    p.add_argument("--pi", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out", help="Write the full result JSON here")
    p.add_argument("--self-check", action="store_true", help="Run the synthetic stub check and exit")
    args = p.parse_args(argv)
    if args.self_check:
        _self_check()
        return 0
    if not args.run:
        p.error("--run is required unless --self-check is given")
    result = edge_stability(
        args.run, args.B, args.n_transitions, args.pi, args.seed, args.device
    )
    print(json.dumps({k: v for k, v in result.items() if k != "frequency_matrix"}, indent=2, default=_jsonable))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, default=_jsonable))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
