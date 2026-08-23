"""Decoupled causal-structure DISCOVERY stage.

The conference hard-mask trainer was deleted; it was also the only place the causal-structure
signal (the CMI predictor head) was trained. The structure-conditioned world model CONSUMES a
calibrated posterior + a frozen enumeration graph but does not PRODUCE them. This stage closes
that gap so the full pipeline runs from scratch on ANY environment:

    discover (structure) -> calibrated posterior + frozen enumeration graph
        -> train (structure-conditioned dynamics) -> evaluate

What it does, decoupled from dynamics/residual/planner weights:
  1. Collects transitions from the env with a random policy (the generation the old trainer used).
  2. Trains ONLY the causal-structure signal: the per-source feature extractors + the max-pool
     predictor head, with the full + single-source-dropout NLL that makes the CMI meaningful.
     The structure-conditioned prediction head and the bounded residual are NEVER touched.
  3. Converges the step-CMI matrix (``mask_CMI``) directly to its steady-state mean, independent
     of the slow EMA / training budget, and freezes the crisp ENUMERATION graph from it.
  4. Bootstraps a per-edge inclusion posterior (P1 ``edge_stability``) from the frozen predictor,
     fits an isotonic calibration map, and writes the CALIBRATED posterior artifact.

Outputs (the exact formats ``train``/``evaluate`` already load and hash):
  * ``<out>/<environment>_posterior.json`` -- ``GraphPosterior.save`` (calibrated=True).
  * ``<out>/<environment>_enum.json``      -- ``freeze_enumeration_graph`` (nonzero state edges).

# ponytail: CMI-on-a-max-pool-predictor is the discovery SIGNAL for now. A more scalable /
# differentiable structure learner (NOTEARS / DiBS / a variational edge posterior) is a future
# drop-in swap -- keep it behind this entrypoint and preserve the posterior + enumeration OUTPUT
# contract (the two artifact formats above) so the swap does not touch train/evaluate.

Usage:
    uv run python -m cdd_oran.cli discover --config configs/env_i_cdl.yaml
    uv run python -m cdd_oran.cli discover --config configs/env_i_cdl.yaml \\
        --out artifacts --train-steps 4000 --B 50 --n-transitions 4096
"""

import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cdd_oran.analysis.edge_stability import collect_transitions
from cdd_oran.analysis.graph_posterior import GraphPosterior, IsotonicCalibrator
from cdd_oran.config import ExperimentConfig
from cdd_oran.envs import get_env
from cdd_oran.models import freeze_enumeration_graph, get_model, node_names
from cdd_oran.policies import RandomPolicy
from cdd_oran.utils.runs import create_run_dir
from cdd_oran.utils.seeding import seed_everything

logger = logging.getLogger(__name__)


def _cmi_head_grad_step(model, s_t, s_tp1, a_batch, optimizer):
    """One gradient step on the CMI/max-pool structure signal ONLY.

    Full-pool NLL + a single-source-dropout NLL (drop the intervention-informed source half the
    time, a random source otherwise) through the max-pool head. Optimizes the feature extractors
    + max-pool predictor; the structure-conditioned head and residual are not in this optimizer,
    so discovery never touches dynamics/planner weights.
    """
    fd = model.state_dim
    bs = s_t.shape[0]
    optimizer.zero_grad()

    changed = a_batch[:, 0].long().clamp_(0, fd)  # intervened source column (a valid source)
    use_informed = torch.rand(bs, device=model.device) > 0.5
    random_drop = torch.randint(fd + 1, (bs,), device=model.device)
    drop_idx = torch.where(use_informed, changed, random_drop)
    mask = F.one_hot(drop_idx, fd + 1).bool()  # (bs, fd+1)

    model.models.train()
    parameters = model._stacked_parameters()
    feats = model._batched_forward(
        parameters, s_t, a_batch, features=None, features_in_dim=None, return_features=True
    )  # (fd, bs, fd+1, feat)
    mu, std = model._batched_forward(parameters, features=feats, features_in_dim=0)
    targets = s_tp1.transpose(0, 1).unsqueeze(-1)  # (fd, bs, 1)
    full_loss = model._nll(mu, std, targets).mean()

    masked_feats = feats.masked_fill(mask.unsqueeze(0).unsqueeze(-1), float("-inf"))
    mu_m, std_m = model._batched_forward(parameters, features=masked_feats, features_in_dim=0)
    masked_loss = model._nll(mu_m, std_m, targets).mean()

    loss = full_loss + masked_loss
    loss.backward()
    nn.utils.clip_grad_norm_(model.models.parameters(), model.grad_clip)
    optimizer.step()
    return loss.detach()


def _converge_mask_cmi(model, s, a, s_next, batch_size, n_batches):
    """Set ``mask_CMI`` to the steady-state MEAN step CMI of the frozen predictor.

    ``update_mask`` accumulates step CMI into ``_eval_cmi_acc``; by resetting the step counter
    before each call the slow EMA never fires, so ``acc / count`` is the converged mean step CMI
    on the natural ``cmi_threshold`` scale -- independent of ``eval_tau`` and the training budget.
    This is the same accumulation the P1 bootstrap uses, minus the ``(1 - eval_tau)`` resample
    rescale, so a from-scratch tiny run still yields a non-empty enumeration graph.
    """
    model._eval_cmi_acc.zero_()
    count = 0
    n = s.shape[0]
    for i in range(n_batches):
        start = (i * batch_size) % n
        idx = slice(start, start + batch_size)
        sb, ab, snb = s[idx], a[idx], s_next[idx]
        if sb.shape[0] == 0:
            continue
        s_pair = torch.stack([sb, snb], dim=1).to(model.device)
        model._eval_step_count = 0  # keep the EMA from firing; accumulate step CMI only
        model.update_mask(s_pair, ab.to(model.device))
        count += 1
    model.mask_CMI = (model._eval_cmi_acc / max(count, 1)).detach().clone()
    model._eval_cmi_acc.zero_()
    model._eval_step_count = 0
    return model.mask_CMI


def _calibration_source(calibration_run, raw_marginals, env, *, B, n_transitions, seed, device):
    """Return ``(scores, labels, provenance)`` for the isotonic calibration fit.

    Held-out when ``--calibration-run`` is given (its own bootstrap frequencies vs its env
    ground truth); otherwise within-environment (this run's bootstrap frequencies vs this env's
    ground truth). Within-environment is score->label recalibration, not held-out structural
    coverage; it is recorded as such in the artifact metadata.
    """
    if calibration_run:
        reference = GraphPosterior.from_bootstrap(
            calibration_run, B=B, n_transitions=n_transitions, seed=seed, device=device
        )
        # Load the calibration run's own config/env for its held-out labels.
        from cdd_oran.config import load_config

        ref_cfg = load_config(Path(calibration_run) / "config.yaml")
        if device:
            ref_cfg = replace(ref_cfg, device=device)
        ref_env = get_env(ref_cfg)
        return (
            reference.marginals(),
            np.asarray(ref_env.true_adj_matrix, dtype=float),
            {"mode": "held_out", "run": str(calibration_run)},
        )
    return (
        np.asarray(raw_marginals, dtype=float),
        np.asarray(env.true_adj_matrix, dtype=float),
        {"mode": "within_environment", "run": None},
    )


def run_discovery(
    cfg: ExperimentConfig,
    *,
    out_dir="artifacts",
    train_steps=4000,
    pool_size=4096,
    cmi_batches=64,
    B=50,
    n_transitions=2048,
    calibration_run=None,
):
    """Run the full discovery stage and write the posterior + enumeration artifacts.

    Returns ``{"run_dir", "posterior", "enumeration_graph", "environment", "state_edges"}``.
    """
    if cfg.model_kind != "cdl":
        raise ValueError("discovery requires a CDL config (model_kind=cdl)")

    seed = cfg.seed
    device = cfg.device
    # Discovery is decoupled: it must NOT depend on pre-existing posterior/enumeration artifacts.
    cfg = replace(cfg, model=replace(cfg.model, posterior_artifact=None, enumeration_graph=None))

    seed_everything(seed, cfg.deterministic)
    env = get_env(cfg)
    model = get_model(cfg, env, sampler=None)  # no sampler, no enumeration graph: pure structure

    state_dim = env.get_state_dim()
    policy = RandomPolicy(action_dim=env.action_dim, action_space=env.action_space)
    pool_size = max(pool_size, cfg.model.batch_size)
    s, a, s_next = collect_transitions(env, policy, pool_size)
    logger.info(
        "Discovery: collected %d transitions on %s (state_dim=%d)", s.shape[0], cfg.environment, state_dim
    )

    # 1) Train ONLY the CMI/max-pool structure signal.
    optimizer = torch.optim.Adam(model.models.parameters(), lr=cfg.model.lr)
    rng = torch.Generator().manual_seed(seed)
    batch_size = cfg.model.batch_size
    last_loss = None
    for step in range(train_steps):
        idx = torch.randint(pool_size, (batch_size,), generator=rng)
        s_t = s[idx].to(device)
        s_tp1 = s_next[idx].to(device)
        a_batch = a[idx].to(device)
        last_loss = _cmi_head_grad_step(model, s_t, s_tp1, a_batch, optimizer)
        if step % max(train_steps // 5, 1) == 0:
            logger.info("Discovery train step %d/%d structure NLL=%.4f", step, train_steps, last_loss.item())

    # 2) Converge mask_CMI (crisp enumeration source) to steady state, then persist a run dir.
    _converge_mask_cmi(model, s, a, s_next, batch_size, cmi_batches)
    state_edges = int(model.get_binary_graph()[:, :state_dim].sum())
    logger.info("Discovery: converged mask_CMI -> %d enumeration state edges", state_edges)

    run_dir = create_run_dir(cfg)
    model.save_model(filepath=run_dir / "checkpoint.pt")
    logger.info("Discovery run saved to %s", run_dir)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    posterior_path = out_dir / f"{cfg.environment}_posterior.json"
    enum_path = out_dir / f"{cfg.environment}_enum.json"

    # 3) Frozen crisp enumeration graph from the trained CMI (fails loudly on zero state edges).
    freeze_enumeration_graph(run_dir, enum_path, device=device)

    # 4) Calibrated per-edge posterior from the frozen predictor's bootstrap frequencies.
    raw_post = GraphPosterior.from_bootstrap(
        run_dir, B=B, n_transitions=n_transitions, seed=seed, device=device
    )
    scores, labels, provenance = _calibration_source(
        calibration_run, raw_post.marginals(), env,
        B=B, n_transitions=n_transitions, seed=seed, device=device,
    )
    calibrator = IsotonicCalibrator().fit(scores, labels)
    calibrated = raw_post.apply_calibrator(calibrator)  # meta.calibrated = True
    calibrated.node_names = node_names(env)
    calibrated.meta = {
        **calibrated.meta,
        "environment": cfg.environment,
        "discovery": True,
        "calibration": provenance,
    }
    calibrated.save(posterior_path)
    logger.info(
        "Discovery artifacts: posterior=%s (calibration=%s) enum=%s (%d state edges)",
        posterior_path, provenance["mode"], enum_path, state_edges,
    )
    return {
        "run_dir": str(run_dir),
        "posterior": str(posterior_path),
        "enumeration_graph": str(enum_path),
        "environment": cfg.environment,
        "state_edges": state_edges,
    }


def main(
    cfg: ExperimentConfig,
    *,
    out_dir="artifacts",
    train_steps=4000,
    pool_size=4096,
    cmi_batches=64,
    B=50,
    n_transitions=2048,
    calibration_run=None,
):
    return run_discovery(
        cfg,
        out_dir=out_dir,
        train_steps=train_steps,
        pool_size=pool_size,
        cmi_batches=cmi_batches,
        B=B,
        n_transitions=n_transitions,
        calibration_run=calibration_run,
    )
