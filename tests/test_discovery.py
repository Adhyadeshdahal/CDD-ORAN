"""Tiny CPU end-to-end test for the decoupled discovery stage.

Runs discovery on an Env-I-tiny budget and proves it recovers a non-empty, right-shaped
structure and writes artifacts that ``train``/``evaluate`` can load and validate (the exact
posterior + enumeration formats those stages hash).
"""

from dataclasses import replace

import numpy as np

from cdd_oran.analysis.graph_posterior import GraphPosterior
from cdd_oran.config import load_config
from cdd_oran.envs import get_env
from cdd_oran.experiments.discover import run_discovery
from cdd_oran.models import (
    load_enumeration_graph,
    node_names,
    validate_enumeration_artifact,
    validate_posterior_artifact,
)


def _tiny_cfg():
    cfg = load_config("env_i_cdl.yaml")
    # Point at NON-EXISTENT posterior/enum artifacts: discovery must run from scratch and
    # ignore them (it is decoupled from the dynamics artifacts it produces).
    return replace(
        cfg,
        device="cpu",
        seed=0,
        model=replace(
            cfg.model,
            posterior_artifact="does/not/exist_posterior.json",
            enumeration_graph="does/not/exist_enum.json",
        ),
    )


def test_discovery_recovers_structure_and_writes_loadable_artifacts(tmp_path):
    cfg = _tiny_cfg()
    env = get_env(cfg)
    fd = env.get_state_dim()

    result = run_discovery(
        cfg,
        out_dir=tmp_path,
        train_steps=120,
        pool_size=512,
        cmi_batches=8,
        B=5,
        n_transitions=256,
    )

    # Artifacts written at the documented <env>_* paths.
    posterior_path = tmp_path / "EnvironmentI_posterior.json"
    enum_path = tmp_path / "EnvironmentI_enum.json"
    assert posterior_path.exists() and enum_path.exists()
    assert result["posterior"] == str(posterior_path)
    assert result["enumeration_graph"] == str(enum_path)

    # Enumeration graph: right-shaped (fd, fd+1), NON-EMPTY, and it recovers real structure.
    graph, meta = load_enumeration_graph(enum_path)
    assert tuple(graph.shape) == (fd, fd + 1)
    pred = graph[:, :fd].numpy().astype(int)
    gt = env.true_adj_matrix.astype(int)
    assert pred.sum() > 0, "enumeration graph must be non-empty"
    assert result["state_edges"] == int(pred.sum())
    tp = int(((pred == 1) & (gt == 1)).sum())
    assert tp > 0, "discovery should recover at least one true edge"
    validate_enumeration_artifact(graph, meta, env, cfg.environment)

    # Posterior: calibrated, right-shaped marginals in [0, 1], correct node layout.
    posterior = GraphPosterior.load(posterior_path)
    assert posterior.is_calibrated
    marg = posterior.marginals()
    assert marg.shape == (fd, fd)
    assert np.all((marg >= 0.0) & (marg <= 1.0))
    assert np.all(np.diag(marg) == 0.0)
    assert posterior.node_names == node_names(env)
    assert posterior.meta.get("environment") == cfg.environment
    validate_posterior_artifact(posterior, env, cfg.environment)
