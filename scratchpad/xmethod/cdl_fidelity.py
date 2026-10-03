"""CDL fidelity gates F1-F3 (brief cdl, ruling R-50): the port ``cdd_oran/xmethod/methods/_cdl_model.py`` against the
authors' conference code on branch ``refactor/codebase`` (commit 65b56f3).

Runs INSIDE a worktree of ``refactor/codebase`` (its ``cdd_oran`` package = the original model, envs and training
loop) with the port loaded by file path (it needs only torch):

  cd <refactor/codebase worktree>
  <xm-cdl venv python> <this file> --port <xm-cdl>/cdd_oran/xmethod/methods/_cdl_model.py --mode golden|equiv|f3 ...

Modes
  golden  F2: the golden regression fixtures tests/golden/Environment{I,II}_CDL.json pin an UNTRAINED CDL (seed 12345)
          through ``predict_next_state`` ("model_probe"); the port must reproduce them (fixture tolerance rtol 1e-5,
          atol 1e-8), and so must the original (sanity).
  equiv   F2: the ORIGINAL training loop ``cdd_oran.experiments.train.main`` (CPU, deterministic) run twice, once with
          the original CDL and once with the port (``train.get_model`` patched); final CMI matrix and every weight
          must be bitwise equal.
  f3      F3: the conference graph-recovery result (paper sec 5.1: Env I 20k steps and Env II 50k steps, seed 45,
          recall = F1 = 1.00) with the original loop and the PORT; records the final EMA CMI matrix, the learned graph
          at the repo threshold (.16, configs/env_i_cdl.yaml) and the paper's (.2), precision / recall / F1 vs the true
          adjacency, and the CMI margin (smallest true-edge CMI vs largest non-edge CMI).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.getcwd())


def load_port(path: str):
    spec = importlib.util.spec_from_file_location("cdl_port", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def golden(port) -> dict:
    import tests._harness as H
    out = {}
    for env_name in H.ENV_NAMES:
        g = json.loads((Path("tests/golden") / f"{env_name}_CDL.json").read_text())
        env = H.build_env(env_name)
        state = torch.tensor(g["state"], dtype=torch.float32)
        res = {}
        for label, cls in (("original", None), ("port", port.CDL)):
            if cls is None:
                model = H.build_model("CDL", env)
            else:
                H.seed_all(H.BASE_SEED)
                model = cls(state_dim=env.get_state_dim(), action_dim=env.get_action_dim(), device=H.DEVICE,
                            kpi_start=env.num_params, node_names=[f"n{i}" for i in range(env.get_state_dim())],
                            **H.MODEL_KWARGS)
            probe = H._model_probe(model, env, state)
            a, b = np.array(probe), np.array(g["model_probe"])
            res[label] = {"max_abs_diff": float(np.abs(a - b).max()),
                          "pass": bool(np.allclose(a, b, rtol=1e-5, atol=1e-8))}
        out[env_name] = res
    return out


def _cfg(env_yaml: str, steps: int | None, init: int | None, seed: int | None):
    from cdd_oran.config import load_config
    over = ["device=cpu", "deterministic=true"]
    if steps is not None:
        over.append(f"train.total_steps={steps}")
    if init is not None:
        over.append(f"train.init_steps={init}")
    if seed is not None:
        over.append(f"seed={seed}")
    return load_config(env_yaml, over)


def _train(cfg, port=None):
    """Run the original loop; returns the trained model (``train.get_model`` patched to the port if given)."""
    import cdd_oran.experiments.train as T
    import cdd_oran.models as M
    holder = {}
    orig = T.get_model

    def make(cfg_, env):
        if port is None:
            m = orig(cfg_, env)
        else:
            m = port.CDL(state_dim=env.get_state_dim(), action_dim=env.get_action_dim(), device=cfg_.device,
                         cmi_threshold=cfg_.model.cmi_threshold, eval_tau=cfg_.model.eval_tau,
                         eval_steps=cfg_.train.eval_steps, grad_clip=cfg_.model.grad_clip,
                         generative_fc_dims=cfg_.model.generative_fc_dims, feature_fc_dims=cfg_.model.feature_fc_dims,
                         lr=cfg_.model.lr, kpi_start=env.num_params, node_names=M.node_names(env))
        holder["model"] = m
        holder["env"] = env
        return m

    T.get_model = make
    try:
        with tempfile.TemporaryDirectory() as d:
            t0 = time.process_time()
            T.main(cfg, run_dir=Path(d))
            holder["cpu_s"] = time.process_time() - t0
    finally:
        T.get_model = orig
    return holder


def equiv(port, env_yaml: str, steps: int, init: int) -> dict:
    cfg = _cfg(env_yaml, steps, init, None)
    a = _train(cfg)
    b = _train(cfg, port)
    ca, cb = a["model"].mask_CMI.cpu().numpy(), b["model"].mask_CMI.cpu().numpy()
    sa, sb = a["model"].models.state_dict(), b["model"].models.state_dict()
    wdiff = max(float((sa[k] - sb[k]).abs().max()) for k in sa)
    return {"env": env_yaml, "total_steps": steps, "init_steps": init, "cmi_max_abs_diff": float(np.abs(ca - cb).max()),
            "weights_max_abs_diff": wdiff, "bitwise_equal": bool(np.array_equal(ca, cb) and wdiff == 0.0),
            "cmi_nonzero": bool(np.abs(ca).max() > 0), "cpu_s": [a["cpu_s"], b["cpu_s"]]}


def _graph_metrics(cmi: np.ndarray, gt: np.ndarray, thr: float) -> dict:
    pred = cmi[:, :-1] >= thr                     # as get_binary_graph()[:, :-1]
    fd = pred.shape[0]
    np.fill_diagonal(pred[:fd, :fd], False)
    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    return {"threshold": thr, "tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec + 1e-8), "edges": int(pred.sum())}


def f3(port, env_yaml: str, seed: int | None, which: str) -> dict:
    cfg = _cfg(env_yaml, None, None, seed)
    h = _train(cfg, None if which == "original" else port)
    cmi = h["model"].mask_CMI.cpu().numpy()
    gt = np.asarray(h["env"].true_adj_matrix)
    off = cmi[:, :-1].copy()
    np.fill_diagonal(off, np.nan)
    true_cmi = off[gt == 1]
    null_cmi = off[(gt == 0) & np.isfinite(off)]
    return {"env": env_yaml, "model": which, "seed": cfg.seed, "total_steps": cfg.train.total_steps,
            "cpu_s": h["cpu_s"], "threads": torch.get_num_threads(), "torch": torch.__version__,
            "metrics": [_graph_metrics(cmi, gt, t) for t in (0.16, 0.2)],
            "min_true_edge_cmi": float(true_cmi.min()), "max_null_edge_cmi": float(null_cmi.max()),
            "cmi": cmi.round(6).tolist(), "true_adj": gt.astype(int).tolist()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--mode", choices=("golden", "equiv", "f3"), required=True)
    ap.add_argument("--env", default="env_i_cdl.yaml")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--init", type=int, default=500)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--which", choices=("port", "original"), default="port")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    port = load_port(a.port)
    if a.mode == "golden":
        res = golden(port)
    elif a.mode == "equiv":
        res = equiv(port, a.env, a.steps, a.init)
    else:
        res = f3(port, a.env, a.seed, a.which)
    res = {"mode": a.mode, **res} if isinstance(res, dict) and "env" in res else {"mode": a.mode, "result": res}
    with open(a.out, "a") as f:
        f.write(json.dumps(res) + "\n")
    print(json.dumps(res)[:2000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
