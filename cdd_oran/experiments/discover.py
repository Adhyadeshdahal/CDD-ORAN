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
  4. Bootstraps a per-edge inclusion posterior (P1 ``edge_stability``) from the frozen predictor.
     Calibrating that posterior into inclusion probabilities needs labels; those labels must be
     HELD OUT, never the target environment's own true adjacency (that is target leakage). So:
       * with ``--calibration-run`` (a DISTINCT held-out environment) it fits an isotonic map on
         that run's scores+labels and writes the CALIBRATED, downstream-sampleable posterior;
       * without it, it writes only a RAW diagnostic (calibrated=false) that robust structure
         sampling / staging reject -- the target's true adjacency is never read on this path.

Outputs (the exact formats ``train``/``evaluate`` already load and hash):
  * ``<out>/<environment>_posterior.json``     -- calibrated posterior (``--calibration-run`` mode).
  * ``<out>/<environment>_raw_posterior.json`` -- raw uncalibrated diagnostic (no calibration run).
  * ``<out>/<environment>_enum.json``          -- ``freeze_enumeration_graph`` (nonzero state edges).

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
from cdd_oran.models import freeze_enumeration_graph, get_model, node_names, sha256_file
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


def _load_reference_config(calibration_run, device):
    """Load a calibration run's own config (optionally overriding device)."""
    from cdd_oran.config import load_config

    ref_cfg = load_config(Path(calibration_run) / "config.yaml")
    if device:
        ref_cfg = replace(ref_cfg, device=device)
    return ref_cfg


def _check_held_out(ref_cfg, target_environment):
    """Refuse a calibration run that would leak the TARGET environment's ground truth.

    A calibration run in the same environment as the target shares its true adjacency, so
    fitting against its labels is indirect target leakage (self-calibration). Only a run from
    a DISTINCT held-out environment provides honest structural labels.
    """
    if ref_cfg.environment == target_environment:
        raise ValueError(
            f"--calibration-run environment {ref_cfg.environment!r} equals the target "
            f"environment {target_environment!r}: same-target (self) calibration would leak "
            "the target's true adjacency. Use a run from a DIFFERENT held-out environment."
        )


def _held_out_calibration_source(
    calibration_run, run_dir, ref_cfg, *, B, n_transitions, seed, device
):
    """Return ``(scores, labels, provenance)`` for the isotonic fit from a DISTINCT held-out run.

    Scores are the calibration run's OWN bootstrap inclusion frequencies; labels are that run's
    environment ground truth. The target run/environment's true adjacency is never read here, so
    the calibrated posterior has not (directly or indirectly) seen the answer it will be sampled
    against. Provenance records the calibration run identity + its checkpoint hash so held-out
    calibration is auditable in the persisted artifact.
    """
    if Path(calibration_run).resolve() == Path(run_dir).resolve():
        raise ValueError(
            "--calibration-run must differ from the discovery run just produced (self-calibration)"
        )
    reference = GraphPosterior.from_bootstrap(
        calibration_run, B=B, n_transitions=n_transitions, seed=seed, device=device
    )
    ref_env = get_env(ref_cfg)
    checkpoint = Path(calibration_run) / "checkpoint.pt"
    provenance = {
        "mode": "held_out",
        "run": str(calibration_run),
        "environment": ref_cfg.environment,
        "checkpoint_sha256": sha256_file(checkpoint) if checkpoint.exists() else None,
    }
    return (
        reference.marginals(),
        np.asarray(ref_env.true_adj_matrix, dtype=float),
        provenance,
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
    # Discovery is decoupled: it must NOT depend on pre-existing posterior/enumeration artifacts,
    # and it always learns the causal structure (never the oracle true-adjacency shortcut).
    cfg = replace(
        cfg,
        model=replace(
            cfg.model,
            posterior_artifact=None,
            enumeration_graph=None,
            structure_source="discovered",
        ),
    )

    # Validate a held-out calibration run BEFORE any training, so a same-target/self
    # calibration request fails fast instead of after a full discovery run.
    ref_cfg = None
    if calibration_run is not None:
        ref_cfg = _load_reference_config(calibration_run, device)
        _check_held_out(ref_cfg, cfg.environment)

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
    enum_path = out_dir / f"{cfg.environment}_enum.json"

    # 3) Frozen crisp enumeration graph from the trained CMI (label-free / preconfigured
    #    threshold; fails loudly on zero state edges). No ground truth is read here.
    freeze_enumeration_graph(run_dir, enum_path, device=device)

    # 4) Per-edge inclusion posterior from the frozen predictor's bootstrap frequencies.
    raw_post = GraphPosterior.from_bootstrap(
        run_dir, B=B, n_transitions=n_transitions, seed=seed, device=device
    )
    raw_post.node_names = node_names(env)
    common_meta = {"environment": cfg.environment, "discovery": True}

    if calibration_run is None:
        # NO-CALIBRATION MODE: persist a RAW diagnostic (calibrated=False). Without a held-out
        # calibration run there is no honest way to map scores to inclusion probabilities, so the
        # target environment's true adjacency is deliberately NOT read on this path. The artifact
        # is NOT downstream-sampleable -- from_artifact / stage_run_artifacts reject it.
        #
        # Corruption guard (review finding #8): a PRIOR calibrated run into this same --out dir
        # leaves a downstream-sampleable ``{env}_posterior.json``. This no-cal run refreshes only
        # the raw/enum pair and never rewrites that calibrated file, so leaving it in place would
        # let a config -- which consumes ``posterior_artifact`` and ``enumeration_graph`` as
        # INDEPENDENT paths with no cross-run binding -- silently pair this run's NEW enum graph
        # with the OLD calibrated posterior. Remove the stale calibrated artifact so a mismatched
        # cross-run pair can never be consumed.
        stale_calibrated = out_dir / f"{cfg.environment}_posterior.json"
        if stale_calibrated.exists():
            stale_calibrated.unlink()
            logger.warning(
                "Discovery no-cal run superseded a STALE calibrated posterior %s (removed): a "
                "raw/uncalibrated run does not refresh it, and leaving it beside this run's new "
                "enumeration graph %s would let downstream silently mix two runs",
                stale_calibrated, enum_path,
            )
        raw_post.meta = {
            **raw_post.meta,
            **common_meta,
            "calibrated": False,
            "calibration": {"mode": "uncalibrated", "run": None},
        }
        raw_path = out_dir / f"{cfg.environment}_raw_posterior.json"
        raw_post.save(raw_path)
        logger.info(
            "Discovery artifacts: raw_posterior=%s (UNCALIBRATED, not downstream-sampleable) "
            "enum=%s (%d state edges); pass --calibration-run for a calibrated posterior",
            raw_path, enum_path, state_edges,
        )
        return {
            "run_dir": str(run_dir),
            "raw_posterior": str(raw_path),
            "calibrated_posterior": None,
            "enumeration_graph": str(enum_path),
            "environment": cfg.environment,
            "state_edges": state_edges,
            "calibrated": False,
        }

    # CALIBRATION MODE: fit an isotonic map from a DISTINCT held-out run's scores + labels.
    scores, labels, provenance = _held_out_calibration_source(
        calibration_run, run_dir, ref_cfg,
        B=B, n_transitions=n_transitions, seed=seed, device=device,
    )
    calibrator = IsotonicCalibrator().fit(scores, labels)
    calibrated = raw_post.apply_calibrator(calibrator)  # meta.calibrated = True
    calibrated.node_names = node_names(env)
    calibrated.meta = {
        **calibrated.meta,
        **common_meta,
        "calibration": provenance,
        "calibration_runs": [provenance["run"]],
    }
    calibrated_path = out_dir / f"{cfg.environment}_posterior.json"
    calibrated.save(calibrated_path)
    logger.info(
        "Discovery artifacts: calibrated_posterior=%s (held-out run=%s) enum=%s (%d state edges)",
        calibrated_path, provenance["run"], enum_path, state_edges,
    )
    return {
        "run_dir": str(run_dir),
        "raw_posterior": None,
        "calibrated_posterior": str(calibrated_path),
        "enumeration_graph": str(enum_path),
        "environment": cfg.environment,
        "state_edges": state_edges,
        "calibrated": True,
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
