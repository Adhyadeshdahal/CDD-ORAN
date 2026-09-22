import hashlib
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import torch

from cdd_oran.config import ExperimentConfig
from cdd_oran.models.cdl import CDL
from cdd_oran.models.mlp import MLPInference

logger = logging.getLogger(__name__)


def node_names(env):
    params = [f"param{i}" for i in range(env.num_params)]
    kpis = [kpi.name for kpi in env.kpis]
    return params + kpis


class PosteriorStructureSampler:
    """The ONE explicit P1<->P2 boundary adapter (spec risk 1).

    Phase 1's ``GraphPosterior.sample`` returns a ``(m, fd, fd)`` parent-block graph with
    NO action column; Phase 2's dynamics consume ``(m, fd, ns=fd+1)`` structures. This
    adapter appends the action source (always available) and hands back a bool tensor on
    the model device. It never infers or appends posterior edges inside the predictor.
    """

    def __init__(self, posterior, seed=0):
        self.posterior = posterior
        self.rng = np.random.default_rng(seed)

    def sample_structures(self, n_members, *, device):
        draws = np.asarray(self.posterior.sample(self.rng, size=n_members))
        m, fd, _ = draws.shape
        out = torch.zeros(m, fd, fd + 1, dtype=torch.bool, device=device)
        out[:, :, :fd] = torch.as_tensor(draws.astype(bool), device=device)
        out[:, :, -1] = True  # action source; downstream also force-keeps it on
        return out

    @classmethod
    def from_artifact(cls, path, *, seed=0, env=None, environment=None):
        """Load a precomputed calibrated posterior and optionally validate its target layout.

        Pure I/O -- no bootstrap, no global RNG mutation. Rejects any posterior whose metadata
        is not explicitly calibrated, so P2 can never sample raw inclusion frequencies. When
        ``env`` is supplied, dimensions, environment identity, and exact node ordering are
        checked before the sampler is constructed."""
        from cdd_oran.analysis.graph_posterior import GraphPosterior

        posterior = GraphPosterior.load(path)
        if not posterior.is_calibrated:
            raise ValueError(
                f"posterior artifact {path!r} is not calibrated (meta.calibrated != True); "
                "generate it offline with `python -m cdd_oran.analysis.graph_posterior "
                "--run ... --calibration-run ... --save-posterior <path>`"
            )
        if env is not None:
            validate_posterior_artifact(posterior, env, environment)
        return cls(posterior, seed=seed)

    @classmethod
    def from_run(cls, run_dir, *, seed=0, **bootstrap_kwargs):
        """OFFLINE ONLY: build a RAW (uncalibrated) sampler by bootstrapping a trained CDL run.
        Mutates global RNG via edge_stability -- never call this on the train-time path; use a
        precomputed calibrated artifact via ``from_artifact`` instead."""
        from cdd_oran.analysis.graph_posterior import GraphPosterior

        posterior = GraphPosterior.from_bootstrap(run_dir, **bootstrap_kwargs)
        return cls(posterior, seed=seed)


def build_oracle_enumeration_graph(env):
    """ORACLE variant enumeration graph, built directly from the env's TRUE adjacency (no
    discovery, no artifact). Returns a ``(fd, fd+1)`` bool tensor whose square state block is
    ``env.true_adj_matrix`` and whose action column is on (the action is always a source, and
    the action column is dropped for conflict enumeration anyway)."""
    true_adj = np.asarray(env.true_adj_matrix, dtype=bool)
    fd = env.get_state_dim()
    if true_adj.shape != (fd, fd):
        raise ValueError(f"env.true_adj_matrix shape {true_adj.shape} != (fd, fd)=({fd}, {fd})")
    graph = torch.zeros(fd, fd + 1, dtype=torch.bool)
    graph[:, :fd] = torch.as_tensor(true_adj)
    graph[:, -1] = True
    _validate_enum_structure(graph.numpy())  # reject a zero-edge true adjacency
    return graph


def build_oracle_posterior(env):
    """Degenerate one-hot GraphPosterior over the env's TRUE adjacency: per-edge probabilities
    are exactly 0/1, so every draw returns the true parent block. Flagged calibrated=True so the
    structure sampler accepts it with no discovery/calibration step."""
    from cdd_oran.analysis.graph_posterior import GraphPosterior

    true_adj = np.asarray(env.true_adj_matrix, dtype=float)
    return GraphPosterior(
        true_adj,
        node_names=node_names(env),
        meta={"environment": None, "calibrated": True, "source": "oracle_true_adjacency"},
    )


def build_oracle_sampler(env, seed=0):
    """ORACLE structure sampler: a PosteriorStructureSampler over the degenerate one-hot
    true-adjacency posterior. Needs no posterior artifact."""
    return PosteriorStructureSampler(build_oracle_posterior(env), seed=seed)


def make_structure_sampler(cfg: ExperimentConfig, seed=0, env=None):
    """Build the P1 structure sampler for the structure-conditioned world model.

    ``structure_source=oracle`` builds a degenerate one-hot sampler from ``env.true_adj_matrix``
    (no artifact needed). Otherwise (``discovered``) LOADS the configured precomputed calibrated
    posterior artifact (``model.posterior_artifact``): no global RNG mutation (review blocker #4);
    rejects uncalibrated artifacts (review blocker #2)."""
    if cfg.model.structure_source == "oracle":
        if env is None:
            from cdd_oran.envs.legacy import get_env

            env = get_env(cfg)
        return build_oracle_sampler(env, seed=seed)
    artifact = cfg.model.posterior_artifact
    if not artifact:
        raise ValueError(
            "the structure-conditioned world model requires model.posterior_artifact "
            "(a precomputed CALIBRATED posterior file; generate it offline with "
            "cdd_oran.analysis.graph_posterior --save-posterior)"
        )
    return PosteriorStructureSampler.from_artifact(
        artifact,
        seed=seed,
        env=env,
        environment=cfg.environment,
    )


def validate_posterior_artifact(posterior, env, environment):
    """Validate a calibrated posterior against the target environment's exact layout."""
    expected_fd = env.get_state_dim()
    actual_shape = tuple(posterior.marginals().shape)
    expected_shape = (expected_fd, expected_fd)
    if actual_shape != expected_shape:
        raise ValueError(
            f"posterior edge matrix shape {actual_shape} != target shape {expected_shape}"
        )
    actual_environment = posterior.meta.get("environment")
    if actual_environment != environment:
        raise ValueError(
            f"posterior environment {actual_environment!r} != target {environment!r}"
        )
    expected_nodes = node_names(env)
    if list(posterior.node_names or []) != expected_nodes:
        raise ValueError(
            "posterior node-name ordering does not match the target env "
            f"({posterior.node_names} != {expected_nodes})"
        )


def _validate_enum_structure(graph):
    """Structural checks independent of the target env (review v2 BLOCKER B): the graph is a
    ``(fd, fd+1)`` matrix whose SQUARE state-edge block has a NONZERO edge count. Returns
    ``(fd, state_edge_count)``. Raises on a zero-edge or wrong-layout artifact."""
    array = np.asarray(graph, dtype=bool)
    if array.ndim != 2 or array.shape[1] != array.shape[0] + 1:
        raise ValueError(
            f"enumeration graph must be (fd, fd+1); got shape {tuple(array.shape)}"
        )
    fd = array.shape[0]
    state_edges = int(array[:, :fd].sum())
    if state_edges == 0:
        raise ValueError(
            "enumeration graph has ZERO state edges -> it would enumerate no conflicts; "
            "the source run's learned graph must contain at least one state edge"
        )
    return fd, state_edges


def validate_enumeration_artifact(graph, meta, env, environment):
    """Validate a loaded enumeration artifact against the TARGET env (review v2 BLOCKER B):
    structural nonzero state edges + dimensions, matching environment identity, and EXACT
    node-name ordering. Raises on any mismatch."""
    fd, _ = _validate_enum_structure(graph)
    if fd != env.get_state_dim():
        raise ValueError(
            f"enumeration graph fd={fd} != target env state_dim={env.get_state_dim()}"
        )
    if meta.get("environment") != environment:
        raise ValueError(
            f"enumeration graph environment {meta.get('environment')!r} != target {environment!r}"
        )
    expected_nodes = node_names(env)
    if list(meta.get("node_names") or []) != expected_nodes:
        raise ValueError(
            "enumeration graph node-name ordering does not match the target env "
            f"({meta.get('node_names')} != {expected_nodes})"
        )
    if meta.get("model_kind") != "cdl":
        raise ValueError(
            f"enumeration graph must come from a CDL source (got model_kind={meta.get('model_kind')!r})"
        )


def freeze_enumeration_graph(run_dir, out_path, device=None):
    """OFFLINE: freeze the crisp hard ENUMERATION graph from a trained CDL run into its own
    artifact. The graph comes from the CMI-based ``get_binary_graph`` (the P1 bootstrap path);
    validates it has nonzero state edges and records env + node layout for load-time validation."""
    from dataclasses import replace

    from cdd_oran.config import load_config
    from cdd_oran.envs.legacy import get_env

    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "config.yaml")
    if device:
        cfg = replace(cfg, device=device)
    if cfg.model_kind != "cdl":
        raise ValueError(
            f"freeze_enumeration_graph requires a CDL source run (got model_kind={cfg.model_kind!r})"
        )
    env = get_env(cfg)
    model = get_model(cfg, env)
    model.load_model(run_dir / "checkpoint.pt")
    graph = model.get_binary_graph().cpu().numpy().astype(bool)
    _, state_edges = _validate_enum_structure(graph)  # reject a zero-edge source graph
    payload = {
        "graph": graph.tolist(),
        "shape": list(graph.shape),
        "source_run": str(run_dir),
        "environment": cfg.environment,
        "node_names": node_names(env),
        "state_edge_count": state_edges,
        "model_kind": cfg.model_kind,
    }
    Path(out_path).write_text(json.dumps(payload, indent=2))
    return out_path


def load_enumeration_graph(path):
    """Load a frozen enumeration-graph artifact -> ``(bool tensor (fd, fd+1), metadata)``.
    Applies the env-independent structural checks (nonzero state edges, layout)."""
    payload = json.loads(Path(path).read_text())
    graph = torch.as_tensor(payload["graph"], dtype=torch.bool)
    meta = {key: value for key, value in payload.items() if key != "graph"}
    _validate_enum_structure(graph)
    return graph, meta


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stage_run_artifacts(cfg: ExperimentConfig, run_dir, env=None):
    """Make a structure_conditioned run SELF-CONTAINED (review v2 MAJOR): copy the immutable
    calibrated posterior + frozen enumeration graph INTO ``run_dir/artifacts/``, record their
    sha256 hashes and the posterior identity in ``run_dir/artifacts.json``, and return the
    manifest (also persisted in the checkpoint). Evaluation reads the staged copies and
    verifies the hashes, so replacing a referenced source file afterward cannot change eval."""
    from cdd_oran.analysis.graph_posterior import GraphPosterior

    run_dir = Path(run_dir)
    if env is None:
        from cdd_oran.envs.legacy import get_env

        env = get_env(cfg)
    if not cfg.model.posterior_artifact:
        raise ValueError("structure_conditioned run requires model.posterior_artifact")
    if not cfg.model.enumeration_graph:
        raise ValueError("structure_conditioned run requires model.enumeration_graph")

    # Validate external sources BEFORE creating/staging any run artifacts.
    source_posterior = GraphPosterior.load(cfg.model.posterior_artifact)
    if not source_posterior.is_calibrated:
        raise ValueError("posterior artifact is not calibrated; refusing to stage")
    validate_posterior_artifact(source_posterior, env, cfg.environment)
    source_enum, source_enum_meta = load_enumeration_graph(cfg.model.enumeration_graph)
    validate_enumeration_artifact(source_enum, source_enum_meta, env, cfg.environment)

    art_dir = run_dir / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)
    post_dst = art_dir / "posterior.json"
    shutil.copyfile(cfg.model.posterior_artifact, post_dst)
    posterior = GraphPosterior.load(post_dst)
    validate_posterior_artifact(posterior, env, cfg.environment)

    enum_dst = art_dir / "enumeration_graph.json"
    shutil.copyfile(cfg.model.enumeration_graph, enum_dst)
    _graph, enum_meta = load_enumeration_graph(enum_dst)
    validate_enumeration_artifact(_graph, enum_meta, env, cfg.environment)

    manifest = {
        "posterior": {
            "filename": "posterior.json",
            "sha256": sha256_file(post_dst),
            "source": str(cfg.model.posterior_artifact),
            "calibrated": True,
            "node_names": posterior.node_names,
            "shape": list(posterior.marginals().shape),
            "environment": posterior.meta.get("environment"),
            "calibration_runs": posterior.meta.get("calibration_runs"),
        },
        "enumeration_graph": {
            "filename": "enumeration_graph.json",
            "sha256": sha256_file(enum_dst),
            "source": str(cfg.model.enumeration_graph),
            "environment": enum_meta.get("environment"),
            "node_names": enum_meta.get("node_names"),
            "shape": enum_meta.get("shape"),
            "model_kind": enum_meta.get("model_kind"),
            "dynamics_mode": enum_meta.get("dynamics_mode"),
        },
    }
    (run_dir / "artifacts.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def verify_run_artifacts(run_dir, env=None, environment=None):
    """Verify a self-contained run's staged artifacts against ``artifacts.json`` (review v2
    MAJOR). Fails loudly if the manifest or a staged file is missing or its hash changed.
    Returns ``(manifest, {key: resolved_in_run_path})``."""
    run_dir = Path(run_dir)
    manifest_path = run_dir / "artifacts.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"run artifact manifest missing: {manifest_path}; this structure_conditioned run "
            "is not self-contained (retrain with artifact staging)"
        )
    manifest = json.loads(manifest_path.read_text())
    expected_keys = {"posterior", "enumeration_graph"}
    if set(manifest) != expected_keys:
        raise ValueError(
            "run artifact manifest must contain exactly posterior and enumeration_graph entries"
        )
    resolved = {}
    expected_filenames = {"posterior": "posterior.json", "enumeration_graph": "enumeration_graph.json"}
    for key, entry in manifest.items():
        if not isinstance(entry, dict) or entry.get("filename") != expected_filenames[key]:
            raise ValueError(f"run artifact manifest has invalid {key!r} entry")
        if not isinstance(entry.get("sha256"), str) or len(entry["sha256"]) != 64:
            raise ValueError(f"run artifact manifest has invalid {key!r} sha256")
        staged = run_dir / "artifacts" / entry["filename"]
        if not staged.exists():
            raise FileNotFoundError(f"staged artifact missing: {staged}")
        digest = sha256_file(staged)
        if digest != entry["sha256"]:
            raise ValueError(
                f"staged artifact {staged} hash changed since training "
                f"({digest} != {entry['sha256']})"
            )
        resolved[key] = str(staged)
    if env is not None:
        from cdd_oran.analysis.graph_posterior import GraphPosterior

        posterior = GraphPosterior.load(resolved["posterior"])
        if not posterior.is_calibrated:
            raise ValueError("staged posterior artifact is not calibrated")
        validate_posterior_artifact(posterior, env, environment)
        graph, meta = load_enumeration_graph(resolved["enumeration_graph"])
        validate_enumeration_artifact(graph, meta, env, environment)
    return manifest, resolved


def verify_checkpoint_manifest(checkpoint_path, expected_manifest):
    """Require exact artifact-manifest equality without constructing a model or sampler."""
    state = torch.load(checkpoint_path, map_location="cpu")
    actual = state.get("artifact_manifest")
    if actual != expected_manifest:
        raise ValueError(
            "checkpoint artifact manifest does not exactly match the in-run artifact manifest"
        )
    return actual


def get_model(cfg: ExperimentConfig, env, sampler=None):
    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()
    model_kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "device": cfg.device,
        "cmi_threshold": cfg.model.cmi_threshold,
        "eval_tau": cfg.model.eval_tau,
        "eval_steps": cfg.train.eval_steps,
        "grad_clip": cfg.model.grad_clip,
        "generative_fc_dims": cfg.model.generative_fc_dims,
        "feature_fc_dims": cfg.model.feature_fc_dims,
        "lr": cfg.model.lr,
        "kpi_start": env.num_params,
        "node_names": node_names(env),
    }
    if cfg.model_kind == "cdl":
        # Structure source: oracle builds the enumeration graph (and, when no sampler is
        # injected, a degenerate one-hot sampler) straight from the env's TRUE adjacency, needing
        # no discovery artifacts. discovered loads + validates the frozen enumeration artifact.
        if cfg.model.structure_source == "oracle":
            enum_graph = build_oracle_enumeration_graph(env)
            # Oracle PREDICTION must use env.true_adj_matrix. Unconditionally rebuild the sampler
            # from the env and IGNORE any caller-supplied one, so a discovered/arbitrary posterior
            # can never leak into oracle prediction (review BLOCKER #3).
            if sampler is not None:
                logger.warning(
                    "structure_source=oracle ignores the caller-supplied sampler and rebuilds "
                    "the degenerate one-hot sampler from env.true_adj_matrix"
                )
            sampler = build_oracle_sampler(env, seed=cfg.seed)
        else:
            enum_graph = None
            if cfg.model.enumeration_graph:
                enum_graph, enum_meta = load_enumeration_graph(cfg.model.enumeration_graph)
                validate_enumeration_artifact(enum_graph, enum_meta, env, cfg.environment)
        return CDL(
            **model_kwargs,
            interv_weight=cfg.model.interv_weight,
            residual_bound=cfg.model.residual_bound,
            residual_l2=cfg.model.residual_l2,
            residual_l1=cfg.model.residual_l1,
            residual_hidden=cfg.model.residual_hidden,
            residual_alert_fraction=cfg.model.residual_alert_fraction,
            residual_enabled=cfg.model.residual_enabled,
            sampler=sampler,
            predict_members=cfg.model.predict_members,
            enumeration_graph=enum_graph,
        )
    if cfg.model_kind == "mlp":
        return MLPInference(**model_kwargs)
    raise ValueError(f"Unsupported model kind: {cfg.model_kind}")
