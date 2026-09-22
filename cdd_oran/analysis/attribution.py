"""Prediction-level attribution harness (CPU-cheap, planner-free).

Answers the central open question -- does the causal STRUCTURE earn its keep, or is the bounded
dense RESIDUAL doing the work -- by scoring held-out ONE-STEP prediction quality per variant
BEFORE any planner eval. Every variant is scored on ONE fixed held-out transition set (built from
an explicit held-out eval config, deterministic in ``--seed``) and, for CDL, ONE structure tensor
drawn from a dedicated ``--attribution-seed`` reused across CDL models.

Metrics (see the brief / review findings):
  * MSE  -- on the ENSEMBLE MEAN prediction: ``mu_bar = mu.mean(members)`` then MSE(mu_bar, target).
  * NLL  -- the equal-weight Normal-MIXTURE negative log-likelihood over members via ``logsumexp``:
            ``nll = -(logsumexp_m log N(target; mu_m, sigma_m) - log M)``. Single-member (dense /
            oracle / structure_only) collapses to the ordinary Normal NLL.
  * dMSE_residual / dNLL_residual -- the PRIMARY residual-contribution metric: a WITHIN-MODEL
            paired delta on the ``full`` model, same transitions + same sampled structures + same
            weights, ``MSE(mu_graph) - MSE(mu_total)`` (positive = residual helps). 0 by
            construction for structure_only / oracle / dense.
  * residual_magnitude_ratio -- the unsigned |delta|/|mu_graph| COLLAPSE DIAGNOSTIC only (NOT the
            attribution answer).
  * param_count -- structure predictors + active residual (CDL) or the dense MLP.

The four named variants:
  * full            -- structure-conditioned dynamics + bounded residual ON (today's default).
  * structure_only  -- discovered posterior, residual OFF (isolates the causal structure).
  * oracle          -- structure = env.true_adj_matrix, residual OFF (upper bound).
  * dense           -- the MLPInference world model, no structure (secondary baseline).
"""

import json
import math
from collections import namedtuple
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Normal

from cdd_oran.analysis.edge_stability import collect_transitions
from cdd_oran.policies import RandomPolicy
from cdd_oran.utils.seeding import seed_everything

VARIANT_ORDER = ("full", "structure_only", "oracle", "dense")

LoadedRun = namedtuple("LoadedRun", ["cfg", "env", "model", "posterior_sha", "variant"])


def variant_label(cfg):
    """Name a run's variant from its config (the four names the brief defines)."""
    if cfg.model_kind == "mlp":
        return "dense"
    if cfg.model.structure_source == "oracle":
        return "oracle"
    if not cfg.model.residual_enabled:
        return "structure_only"
    return "full"


def _model_device(model):
    return getattr(model, "device", torch.device("cpu"))


def _is_cdl(model):
    return hasattr(model, "predict_components")


def mixture_metrics(mu, std, target):
    """Ensemble-mean MSE + equal-weight Normal-mixture NLL (per-KPI + aggregate).

    ``mu``/``std``: ``(m, batch, k)``; ``target``: ``(batch, k)``. MSE is on the ensemble MEAN
    ``mu.mean(members)``; NLL is the mixture ``-(logsumexp_m log N(target; mu_m, sigma_m) - log
    M)``, NOT the average of component NLLs. Single-member (``m == 1``) collapses to the ordinary
    Normal NLL. Returns tensors."""
    m = mu.shape[0]
    mu_bar = mu.mean(dim=0)
    se = (mu_bar - target) ** 2
    tgt = target.unsqueeze(0).expand(m, -1, -1)
    comp_logprob = Normal(mu, std).log_prob(tgt)
    log_mix = torch.logsumexp(comp_logprob, dim=0) - math.log(m)
    nll = -log_mix
    return {
        "mse_per_kpi": se.mean(dim=0),
        "mse": se.mean(),
        "nll_per_kpi": nll.mean(dim=0),
        "nll": nll.mean(),
    }


def held_out_transitions(env, n_transitions=512, seed=0):
    """A FIXED held-out one-step transition set, deterministic in ``seed`` so every variant is
    scored on the identical set. Rolls a random policy on ``env`` -- generate ONCE and reuse the
    returned dict for every variant."""
    seed_everything(seed)
    policy = RandomPolicy(action_dim=env.action_dim, action_space=env.action_space)
    s, a, s_next = collect_transitions(env, policy, n_transitions)
    return {"s": s, "a": a, "s_next": s_next}


def draw_attribution_structures(model, attribution_seed):
    """Draw ONE structure tensor from a dedicated attribution seed, reusing that seed for every
    CDL model (review MAJOR #5 / BLOCKER #4). ``full`` & ``structure_only`` share posterior
    identity, so the same seed yields IDENTICAL structures; ``oracle``'s degenerate posterior
    yields the true adjacency regardless of the seed."""
    sampler = model.sampler
    sampler.rng = np.random.default_rng(attribution_seed)
    return sampler.sample_structures(model.predict_members, device=model.device)


def parameter_count(model):
    """Trainable parameter count: structure predictors + active residual (CDL), or the dense MLP.
    Reported so the dense baseline's capacity is comparable to the CDL variants (a large mismatch
    is reported, not force-matched)."""
    total = 0
    models = getattr(model, "models", None)
    if models is not None:
        total += sum(p.numel() for p in models.parameters())
    residual = getattr(model, "residual", None)
    if residual is not None:
        total += sum(p.numel() for p in residual.parameters())
    return int(total)


def score_variant(name, model, transitions, attribution_seed=0):
    """Score one variant on the shared held-out set with ONE structure draw used for BOTH the
    MSE/NLL prediction and the residual diagnostics (review MAJOR #5). Returns one table row.

    dMSE_residual / dNLL_residual are the within-model paired residual deltas on the SAME
    structures (0 for models with no active residual). residual_magnitude_ratio is the collapse
    diagnostic only."""
    device = _model_device(model)
    batch = transitions["s"].shape[0]
    s = transitions["s"].to(device)
    a = transitions["a"].reshape(batch, -1).to(device)
    s_next = transitions["s_next"].to(device)
    target = s_next[:, model.kpi_start :]
    k = target.shape[1]
    zeros_k = [0.0] * k

    has_decision = hasattr(model, "clear_decision_structures")
    try:
        if has_decision:
            model.clear_decision_structures()

        if _is_cdl(model):
            structures = draw_attribution_structures(model, attribution_seed)
            comps = model.predict_components(s, a, structures=structures)
            total = mixture_metrics(comps["mu_total"], comps["std"], target)
            n_members = int(comps["mu_total"].shape[0])

            if getattr(model, "residual", None) is not None:
                graph = mixture_metrics(comps["mu_graph"], comps["std"], target)
                dmse = float((graph["mse"] - total["mse"]).item())
                dnll = float((graph["nll"] - total["nll"]).item())
                dmse_pk = [float(x) for x in (graph["mse_per_kpi"] - total["mse_per_kpi"])]
                dnll_pk = [float(x) for x in (graph["nll_per_kpi"] - total["nll_per_kpi"])]
                diag = model.residual_diagnostics(s, a, structures=structures)
                mag_ratio = float(diag["residual_magnitude_ratio"])
            else:
                dmse = dnll = mag_ratio = 0.0
                dmse_pk = list(zeros_k)
                dnll_pk = list(zeros_k)
        else:  # dense MLP: no structure, no residual delta.
            dist = model.predict_next_state(s, a)
            mu = dist.mean.unsqueeze(0)
            std = dist.stddev.unsqueeze(0)
            total = mixture_metrics(mu, std, target)
            n_members = 1
            dmse = dnll = mag_ratio = 0.0
            dmse_pk = list(zeros_k)
            dnll_pk = list(zeros_k)
    finally:
        if has_decision:
            model.clear_decision_structures()

    return {
        "variant": name,
        "mse": float(total["mse"].item()),
        "nll": float(total["nll"].item()),
        "dmse_residual": dmse,
        "dnll_residual": dnll,
        "residual_magnitude_ratio": mag_ratio,
        "param_count": parameter_count(model),
        "mse_per_kpi": [float(x) for x in total["mse_per_kpi"]],
        "nll_per_kpi": [float(x) for x in total["nll_per_kpi"]],
        "dmse_residual_per_kpi": dmse_pk,
        "dnll_residual_per_kpi": dnll_pk,
        "n_members": n_members,
        "n_transitions": int(batch),
    }


def _order_key(row):
    variant = row["variant"]
    return VARIANT_ORDER.index(variant) if variant in VARIANT_ORDER else len(VARIANT_ORDER)


def comparison_markdown(rows):
    """Render the comparison table as markdown (rows = variants)."""
    rows = sorted(rows, key=_order_key)
    lines = [
        "| variant | MSE | NLL | dMSE_residual | dNLL_residual | residual_mag_ratio | param_count |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['mse']:.6f} | {row['nll']:.6f} | "
            f"{row['dmse_residual']:.6f} | {row['dnll_residual']:.6f} | "
            f"{row['residual_magnitude_ratio']:.6f} | {row['param_count']} |"
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
    """Load a trained run (any variant) as a ``LoadedRun`` for scoring.

    Mirrors ``experiments.evaluate.main``: discovered CDL runs verify + load the staged
    posterior/enumeration artifacts and build the calibrated sampler (recording the posterior
    sha256 for identity checks); oracle CDL runs rebuild structure from ``env.true_adj_matrix``
    inside ``get_model`` (no artifacts); MLP runs load the dense world model directly."""
    from cdd_oran.config import load_config
    from cdd_oran.envs.legacy import get_env
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
    posterior_sha = None
    if (
        cfg.model_kind == "cdl"
        and cfg.model.structure_source != "oracle"
        and cfg.model.posterior_artifact
    ):
        manifest, resolved = verify_run_artifacts(run_dir, env=env, environment=cfg.environment)
        verify_checkpoint_manifest(checkpoint, manifest)
        posterior_sha = manifest["posterior"]["sha256"]
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
    return LoadedRun(cfg=cfg, env=env, model=model, posterior_sha=posterior_sha,
                     variant=variant_label(cfg))


def _run_contract(cfg, env):
    """The held-out compatibility contract every variant must share (review MAJOR #6): env
    identity, held-out param ranges, dims, KPI split, and exact node order. Environment + ranges
    fix the normalization contract, so matching them matches normalization."""
    from cdd_oran.models import node_names

    return {
        "environment": cfg.environment,
        "param_ranges": cfg.param_ranges,
        "evaluation_param_ranges": cfg.evaluation_param_ranges,
        "state_dim": env.get_state_dim(),
        "action_dim": env.get_action_dim(),
        "kpi_start": env.num_params,
        "node_names": tuple(node_names(env)),
    }


def validate_comparison(loaded, *, require_full_set=False):
    """Validate a set of ``(name, LoadedRun)`` for a shared held-out comparison BEFORE scoring
    (order-independent). Rejects duplicate variant labels, incompatible run contracts, and
    mismatched discovered posterior identity (review BLOCKER #4 / MAJOR #6)."""
    names = [name for name, _ in loaded]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate variant label(s) in the comparison set: {duplicates}")
    if require_full_set:
        missing = [variant for variant in VARIANT_ORDER if variant not in names]
        if missing:
            raise ValueError(f"incomplete variant set; missing {missing}")

    reference = None
    reference_name = None
    for name, run in loaded:
        contract = _run_contract(run.cfg, run.env)
        if reference is None:
            reference, reference_name = contract, name
        elif contract != reference:
            diff = {
                key: (reference[key], contract[key])
                for key in contract
                if contract[key] != reference[key]
            }
            raise ValueError(
                f"run '{name}' is incompatible with '{reference_name}' for a shared held-out set: "
                f"{diff}"
            )

    discovered = [(name, run) for name, run in loaded if run.posterior_sha is not None]
    posterior_shas = {run.posterior_sha for _, run in discovered}
    if len(posterior_shas) > 1:
        raise ValueError(
            "discovered runs must share ONE staged posterior identity (full & structure_only must "
            f"come from the same posterior); got {len(posterior_shas)} distinct hashes"
        )
    # The discovered pair must draw the SAME structure tensor: shared attribution seed AND identical
    # predict_members (review BLOCKER #4). predict_members legitimately differs for dense/oracle, so
    # this is a paired check over discovered CDL runs only, not part of the global contract.
    member_counts = {(name, run.cfg.model.predict_members) for name, run in discovered}
    if len({count for _, count in member_counts}) > 1:
        raise ValueError(
            "discovered runs (full & structure_only) must use identical model.predict_members so "
            f"they draw the same structures; got {sorted(member_counts, key=lambda x: x[0])}"
        )
    return reference


def run_attribution(
    run_dirs,
    *,
    names=None,
    device="cpu",
    n_transitions=512,
    seed=0,
    attribution_seed=0,
    held_out_ranges=None,
    require_full_set=False,
    out_md=None,
    out_json=None,
    config_path=None,
):
    """Load every run, validate the comparison, build ONE explicit held-out set (default
    ``evaluation_param_ranges``), score each variant on it with a shared attribution structure
    seed, and write the comparison table. Returns ``(rows, markdown)``."""
    from cdd_oran.envs.legacy import get_env

    names = list(names or [])
    if not run_dirs:
        raise ValueError("run_attribution requires at least one run dir")

    loaded = []
    for index, run_dir in enumerate(run_dirs):
        run = load_run(run_dir, device=device, config_path=config_path)
        name = names[index] if index < len(names) else run.variant
        loaded.append((name, run))

    validate_comparison(loaded, require_full_set=require_full_set)

    # One explicit held-out eval config (normally evaluation_param_ranges), order-independent
    # because every run's contract has already been proven identical.
    base_cfg = loaded[0][1].cfg
    ranges = held_out_ranges or base_cfg.evaluation_param_ranges
    held_env = get_env(replace(base_cfg, param_ranges=ranges))
    transitions = held_out_transitions(held_env, n_transitions=n_transitions, seed=seed)

    rows = [
        score_variant(name, run.model, transitions, attribution_seed=attribution_seed)
        for name, run in loaded
    ]
    markdown, _ = write_report(rows, out_md=out_md, out_json=out_json)
    return rows, markdown


def _self_check():
    """Tiny CPU check: single-member mixture NLL collapses to the ordinary Normal NLL, and the
    two-member ensemble-mean MSE + mixture NLL match the closed form."""
    target = torch.tensor([[1.0, 2.0]])
    mu = torch.zeros(1, 1, 2)
    std = torch.ones(1, 1, 2)
    one = mixture_metrics(mu, std, target)
    assert abs(float(one["mse"]) - 2.5) < 1e-6
    expected_nll = 0.5 * math.log(2 * math.pi) + 0.5 * 2.5
    assert abs(float(one["nll"]) - expected_nll) < 1e-6

    mu2 = torch.tensor([[[0.0]], [[2.0]]])
    std2 = torch.tensor([[[1.0]], [[2.0]]])
    t2 = torch.tensor([[1.0]])
    two = mixture_metrics(mu2, std2, t2)
    mu_bar = (0.0 + 2.0) / 2.0
    exp_mse = (mu_bar - 1.0) ** 2
    n0 = math.exp(Normal(torch.tensor(0.0), torch.tensor(1.0)).log_prob(torch.tensor(1.0)))
    n1 = math.exp(Normal(torch.tensor(2.0), torch.tensor(2.0)).log_prob(torch.tensor(1.0)))
    exp_nll = -math.log(0.5 * (n0 + n1))
    assert abs(float(two["mse"]) - exp_mse) < 1e-6, two["mse"]
    assert abs(float(two["nll"]) - exp_nll) < 1e-6, two["nll"]
    print(
        f"attribution self-check passed: 1-member NLL={float(one['nll']):.4f}, "
        f"2-member mixture NLL={float(two['nll']):.4f} MSE={float(two['mse']):.4f}"
    )


if __name__ == "__main__":
    _self_check()
