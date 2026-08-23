"""Prediction-level attribution harness (CPU-cheap, planner-free).

Answers the central open question -- does the causal STRUCTURE earn its keep, or is the bounded
dense RESIDUAL doing the work -- by scoring held-out ONE-STEP prediction quality per variant
BEFORE any planner eval. For each trained world model it reports, on a FIXED held-out transition
set (deterministic seed, identical for every variant):

  * held-out one-step prediction MSE (per-KPI + aggregate),
  * held-out one-step prediction NLL (per-KPI + aggregate),
  * the residual contribution fraction (0 for dense / oracle / structure_only),
  * the parameter count (so capacity is comparable across variants).

The output is a comparison table (markdown + JSON): rows = variants, cols = MSE, NLL, residual
fraction, param count. Runs in minutes on CPU for a tiny env and scales to the real held-out set
on GPU. No planning, no environment rollout beyond generating the held-out transitions.

The four named variants (see the brief):
  * full            -- structure-conditioned dynamics + bounded residual ON (today's default).
  * structure_only  -- discovered posterior, residual OFF (isolates the causal structure).
  * oracle          -- structure = env.true_adj_matrix, residual OFF (upper bound).
  * dense           -- the MLPInference world model, no structure (capacity-matched baseline).
"""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Normal

from cdd_oran.analysis.edge_stability import collect_transitions
from cdd_oran.policies import RandomPolicy
from cdd_oran.utils.seeding import seed_everything

# Canonical row order for the comparison table.
VARIANT_ORDER = ("full", "structure_only", "oracle", "dense")


def variant_label(cfg):
    """Name a run's variant from its config (the four names the brief defines)."""
    if cfg.model_kind == "mlp":
        return "dense"
    if cfg.model.structure_source == "oracle":
        return "oracle"
    if not cfg.model.residual_enabled:
        return "structure_only"
    return "full"


def held_out_transitions(env, n_transitions=512, seed=0):
    """A FIXED held-out one-step transition set, deterministic in ``seed`` so every variant is
    scored on the identical set. Rolls a random policy on ``env`` -- generate ONCE and reuse the
    returned dict for every variant."""
    seed_everything(seed)
    policy = RandomPolicy(action_dim=env.action_dim, action_space=env.action_space)
    s, a, s_next = collect_transitions(env, policy, n_transitions)
    return {"s": s, "a": a, "s_next": s_next}


def _model_device(model):
    return getattr(model, "device", torch.device("cpu"))


def prediction_metrics(model, s, a, s_next):
    """Held-out one-step MSE + NLL (per-KPI and aggregate) for one world model.

    The prediction mean is used (no sampling) so the score is deterministic. For a CDL model a
    single structure set is drawn and CACHED first so the score is fixed under the sampler seed;
    multi-member predictions are averaged over the member axis. Works for both CDL and MLP."""
    device = _model_device(model)
    s = s.to(device)
    a = a.reshape(s.shape[0], -1).to(device)
    s_next = s_next.to(device)
    target = s_next[:, model.kpi_start :]  # (batch, k)

    # Fix one structure draw for CDL so the metric is reproducible; MLP has no such hook.
    has_decision = hasattr(model, "sample_decision_structures")
    if has_decision:
        model.clear_decision_structures()
        model.sample_decision_structures()
    dist = model.predict_next_state(s, a)
    if has_decision:
        model.clear_decision_structures()

    mu = dist.mean
    std = dist.stddev
    if mu.dim() == 2:  # (batch, k) -> add a singleton member axis
        mu = mu.unsqueeze(0)
        std = std.unsqueeze(0)
    m = mu.shape[0]
    tgt = target.unsqueeze(0).expand(m, -1, -1)  # (m, batch, k)

    se = (mu - tgt) ** 2  # (m, batch, k)
    nll = -Normal(mu, std).log_prob(tgt)  # (m, batch, k)
    return {
        "mse_per_kpi": se.mean(dim=(0, 1)).detach().cpu().numpy(),
        "mse": float(se.mean().item()),
        "nll_per_kpi": nll.mean(dim=(0, 1)).detach().cpu().numpy(),
        "nll": float(nll.mean().item()),
        "n_members": int(m),
        "n_transitions": int(s.shape[0]),
    }


def residual_fraction(model, s, a):
    """Residual contribution fraction: exactly 0 for a model with no active residual
    (dense / oracle / structure_only), otherwise the aggregate anti-collapse fraction."""
    if not getattr(model, "residual_enabled", False):
        return 0.0
    if not hasattr(model, "residual_diagnostics"):
        return 0.0
    device = _model_device(model)
    a = a.reshape(s.shape[0], -1).to(device)
    diag = model.residual_diagnostics(s.to(device), a)
    return float(diag["fraction_aggregate"])


def parameter_count(model):
    """Trainable parameter count of a world model: structure predictors + active residual (CDL),
    or the dense MLP. Reported so the dense baseline's capacity is comparable to the CDL variants
    (a large mismatch is reported, not force-matched)."""
    total = 0
    models = getattr(model, "models", None)
    if models is not None:
        total += sum(p.numel() for p in models.parameters())
    residual = getattr(model, "residual", None)
    if residual is not None:
        total += sum(p.numel() for p in residual.parameters())
    return int(total)


def evaluate_variant(name, model, transitions):
    """Score one variant on the shared held-out set -> one comparison-table row."""
    s, a, s_next = transitions["s"], transitions["a"], transitions["s_next"]
    metrics = prediction_metrics(model, s, a, s_next)
    return {
        "variant": name,
        "mse": metrics["mse"],
        "nll": metrics["nll"],
        "residual_fraction": residual_fraction(model, s, a),
        "param_count": parameter_count(model),
        "mse_per_kpi": [float(x) for x in metrics["mse_per_kpi"]],
        "nll_per_kpi": [float(x) for x in metrics["nll_per_kpi"]],
        "n_members": metrics["n_members"],
        "n_transitions": metrics["n_transitions"],
    }


def _order_key(row):
    variant = row["variant"]
    return VARIANT_ORDER.index(variant) if variant in VARIANT_ORDER else len(VARIANT_ORDER)


def comparison_markdown(rows):
    """Render the comparison table as markdown (rows = variants)."""
    rows = sorted(rows, key=_order_key)
    lines = [
        "| variant | MSE | NLL | residual_frac | param_count |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['mse']:.6f} | {row['nll']:.6f} | "
            f"{row['residual_fraction']:.6f} | {row['param_count']} |"
        )
    return "\n".join(lines) + "\n"


def write_report(rows, out_md=None, out_json=None):
    """Write the comparison table (markdown) + full JSON; return ``(markdown, payload)``."""
    rows = sorted(rows, key=_order_key)
    markdown = comparison_markdown(rows)
    payload = {"variants": rows}
    if out_md is not None:
        Path(out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(out_md).write_text(markdown)
    if out_json is not None:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(payload, indent=2))
    return markdown, payload


def load_run(run_dir, device="cpu", config_path=None):
    """Load a trained run (any variant) as ``(cfg, env, model)`` for scoring.

    Mirrors ``experiments.evaluate.main``'s model construction: discovered CDL runs verify + load
    the staged posterior/enumeration artifacts and build the calibrated sampler; oracle CDL runs
    rebuild structure from ``env.true_adj_matrix`` inside ``get_model`` (no artifacts); MLP runs
    load the dense world model directly."""
    from cdd_oran.config import load_config
    from cdd_oran.envs import get_env
    from cdd_oran.models import (
        get_model,
        make_structure_sampler,
        verify_checkpoint_manifest,
        verify_run_artifacts,
    )

    run_dir = Path(run_dir)
    cfg = load_config(config_path or run_dir / "config.yaml")
    cfg = replace(cfg, device=device)
    env = get_env(cfg)
    checkpoint = run_dir / "checkpoint.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    sampler = None
    if (
        cfg.model_kind == "cdl"
        and cfg.model.structure_source != "oracle"
        and cfg.model.posterior_artifact
    ):
        manifest, resolved = verify_run_artifacts(run_dir, env=env, environment=cfg.environment)
        verify_checkpoint_manifest(checkpoint, manifest)
        cfg = replace(
            cfg,
            model=replace(
                cfg.model,
                posterior_artifact=resolved["posterior"],
                enumeration_graph=resolved["enumeration_graph"],
            ),
        )
        sampler = make_structure_sampler(cfg, seed=cfg.seed, env=env)
    model = get_model(cfg, env, sampler=sampler)
    model.load_model(checkpoint)
    return cfg, env, model


def run_attribution(run_dirs, *, names=None, device="cpu", n_transitions=512, seed=0,
                    out_md=None, out_json=None, config_path=None):
    """Load each run, build ONE held-out set (from the first run's env, deterministic in ``seed``),
    score every variant on it, and write the comparison table. Returns ``(rows, markdown)``."""
    names = list(names or [])
    loaded = []
    shared_env = None
    environment = None
    for index, run_dir in enumerate(run_dirs):
        cfg, env, model = load_run(run_dir, device=device, config_path=config_path)
        if shared_env is None:
            shared_env, environment = env, cfg.environment
        elif cfg.environment != environment:
            raise ValueError(
                f"all runs must share an environment for a common held-out set "
                f"({cfg.environment} != {environment})"
            )
        name = names[index] if index < len(names) else variant_label(cfg)
        loaded.append((name, model))

    transitions = held_out_transitions(shared_env, n_transitions=n_transitions, seed=seed)
    rows = [evaluate_variant(name, model, transitions) for name, model in loaded]
    markdown, _ = write_report(rows, out_md=out_md, out_json=out_json)
    return rows, markdown


def _self_check():
    """Tiny CPU check: MSE/NLL on a known constant-Normal stub match the closed form."""
    class _Stub:
        kpi_start = 1
        device = torch.device("cpu")
        residual_enabled = False

        def predict_next_state(self, s, a):
            return Normal(torch.zeros(s.shape[0], 2), torch.ones(s.shape[0], 2))

    s = torch.zeros(4, 3)
    a = torch.zeros(4, 3)
    s_next = torch.zeros(4, 3)
    s_next[:, 1:] = torch.tensor([1.0, 2.0])  # per-KPI targets 1, 2
    metrics = prediction_metrics(_Stub(), s, a, s_next)
    expected_mse = (1.0 + 4.0) / 2.0
    expected_nll = 0.5 * np.log(2 * np.pi) + 0.5 * (1.0 + 4.0) / 2.0
    assert abs(metrics["mse"] - expected_mse) < 1e-6, metrics["mse"]
    assert abs(metrics["nll"] - expected_nll) < 1e-6, metrics["nll"]
    print(f"attribution self-check passed: MSE={metrics['mse']:.4f} NLL={metrics['nll']:.4f}")


if __name__ == "__main__":
    _self_check()
