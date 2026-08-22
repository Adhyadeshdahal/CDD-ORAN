"""Phase 0 corruption harness: CPU smoke test only (no GPU, tiny planner budget).

Covers the post-review v2 design:
  * the named edge is absent from the inference graph after corruption, and
    ``evaluate`` runs end to end on the corrupted graph;
  * Fix 1 -- the conflict population is IDENTICAL with and without corruption
    (enumeration stays on the base graph; only the world-model mask is corrupted);
  * Fix 2 -- two identical corrupted evaluates give bit-identical per-planner
    utilities (common random numbers);
  * Fix 3 -- the driver evaluates in a temp copy and leaves the banked run dir
    byte-identical.

The oracle adjacency is installed as the model's frozen graph so the removed edge is
genuinely present first.
"""

import importlib.util
import shutil
from dataclasses import replace
from pathlib import Path

import torch

from cdd_oran.analysis.graph_baselines import parse_edge_spec, remove_edges
from cdd_oran.config import load_config
from cdd_oran.envs import get_env
from cdd_oran.experiments.evaluate import main as evaluate_main
from cdd_oran.models import get_model
from cdd_oran.utils.runs import read_metrics, write_metadata

_SWEEP_PATH = Path(__file__).resolve().parent.parent / "scripts" / "corruption_sweep.py"


def _load_sweep_module():
    spec = importlib.util.spec_from_file_location("corruption_sweep", _SWEEP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tiny_cfg():
    """Env I CDL config on CPU with a minimal planner budget for a fast smoke run."""
    cfg = load_config("env_i_cdl.yaml")
    return replace(
        cfg,
        device="cpu",
        num_steps=1,
        deterministic=True,
        planner=replace(
            cfg.planner,
            cem=replace(cfg.planner.cem, n_candidate=4, n_top=2, n_iter=1),
            mppi=replace(cfg.planner.mppi, n_samples=4),
            mcts=replace(cfg.planner.mcts, n_simulations=2),
        ),
    )


def _install_oracle_graph(model, env):
    """Make ``get_binary_graph`` return the oracle adjacency (no training needed)."""
    oracle = torch.zeros_like(model.mask_CMI)
    oracle[:, : env.get_state_dim()] = torch.tensor(
        env.true_adj_matrix, dtype=oracle.dtype, device=oracle.device
    )
    model.mask_CMI = oracle


def _make_run(tmp_path, name="run"):
    """A run dir holding an oracle-graph checkpoint + config for the tiny cfg."""
    cfg = _tiny_cfg()
    env = get_env(cfg)
    model = get_model(cfg, env)
    _install_oracle_graph(model, env)
    run_dir = tmp_path / name
    run_dir.mkdir()
    model.save_model(run_dir / "checkpoint.pt")
    write_metadata(run_dir, cfg)
    return cfg, run_dir


def _read_utilities(run_dir):
    import json

    return json.loads((Path(run_dir) / "utilities.json").read_text())


def _panel_count(run_dir):
    steps = _read_utilities(run_dir)["steps"]
    return sum(len(step["panels"]) for step in steps)


def test_parse_edge_spec_orientation():
    env = get_env(_tiny_cfg())
    # KPI1<-P2 is the natural-failure edge: child row = num_params, parent col = 1 (P2).
    assert parse_edge_spec("KPI1<-P2", env) == (env.num_params, 1)
    # parent->child names the same cell.
    assert parse_edge_spec("P2->KPI1", env) == (env.num_params, 1)


def test_corruption_removes_edge_and_evaluate_runs(tmp_path):
    cfg = _tiny_cfg()
    env = get_env(cfg)
    model = get_model(cfg, env)
    _install_oracle_graph(model, env)

    child, parent = parse_edge_spec("KPI1<-P2", env)
    base = model.get_binary_graph()
    assert bool(base[child, parent]), "oracle graph must contain KPI1<-P2 before corruption"

    corrupted, removed = remove_edges(base, ["KPI1<-P2"], env)
    assert not bool(corrupted[child, parent]), "(i) KPI1<-P2 must be absent after corruption"
    assert bool(base[child, parent]), "remove_edges must not mutate the input graph"
    assert removed == [("KPI1<-P2", child, parent)]

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    model.save_model(run_dir / "checkpoint.pt")

    rc = evaluate_main(cfg, run_dir=str(run_dir), corrupt_edges=["KPI1<-P2"])
    assert rc == 0, "(ii) evaluate must run end to end on the corrupted graph"

    evaluation = read_metrics(run_dir)["evaluation"]
    assert "planner_mean_utilities" in evaluation


def test_conflict_population_fixed_under_corruption(tmp_path):
    """Fix 1: enumeration stays on the base graph, so the conflict count (panel count)
    is identical with and without corruption -- no conflict is dropped."""
    cfg, run_clean = _make_run(tmp_path, "clean")
    run_corrupt = tmp_path / "corrupt"
    shutil.copytree(run_clean, run_corrupt)

    assert evaluate_main(cfg, run_dir=str(run_clean), corrupt_edges=None) == 0
    assert evaluate_main(cfg, run_dir=str(run_corrupt), corrupt_edges=["KPI1<-P2"]) == 0

    clean_panels = _panel_count(run_clean)
    corrupt_panels = _panel_count(run_corrupt)
    assert clean_panels > 0, "smoke run must actually enumerate at least one conflict"
    assert corrupt_panels == clean_panels, (
        f"corruption must not change the conflict population: "
        f"{clean_panels} (clean) vs {corrupt_panels} (corrupt)"
    )


def test_common_random_numbers_are_deterministic(tmp_path):
    """Fix 2: two identical corrupted evaluates give bit-identical per-planner utilities."""
    cfg, run_a = _make_run(tmp_path, "crn_a")
    run_b = tmp_path / "crn_b"
    shutil.copytree(run_a, run_b)

    assert evaluate_main(cfg, run_dir=str(run_a), corrupt_edges=["KPI1<-P2"]) == 0
    assert evaluate_main(cfg, run_dir=str(run_b), corrupt_edges=["KPI1<-P2"]) == 0

    util_a = read_metrics(run_a)["evaluation"]["planner_mean_utilities"]
    util_b = read_metrics(run_b)["evaluation"]["planner_mean_utilities"]
    assert util_a == util_b, f"CRN broken: {util_a} != {util_b}"

    # Per-panel per-planner utilities must match bit-for-bit too.
    panels_a = _read_utilities(run_a)["steps"]
    panels_b = _read_utilities(run_b)["steps"]
    assert panels_a == panels_b, "per-panel planner utilities are not reproducible"


def test_driver_leaves_banked_run_unchanged(tmp_path):
    """Fix 3: the driver evaluates in a temp copy; the banked run dir is untouched."""
    sweep = _load_sweep_module()
    _, run_dir = _make_run(tmp_path, "banked")

    before = sweep._snapshot(run_dir)
    assert before, "banked run must have files to protect"

    args = sweep.build_parser().parse_args(
        [
            "--run",
            str(run_dir),
            "--edges",
            "KPI1<-P2",
            "--device",
            "cpu",
            "--out-table",
            str(tmp_path / "out.md"),
            "--out-json",
            str(tmp_path / "out.json"),
            "--out-edgedeltas",
            str(tmp_path / "edgedeltas.json"),
        ]
    )
    rc = sweep.run_sweep(args)
    assert rc == 0, "driver sweep must succeed on the tiny run"

    after = sweep._snapshot(run_dir)
    assert after == before, "driver must not mutate the banked run dir"

    # The edge-delta diagnostic (Fix 5) must have been emitted.
    import json

    payload = json.loads((tmp_path / "edgedeltas.json").read_text())
    assert payload["edges"][0]["edge"] == "KPI1<-P2"
    assert payload["edges"][0]["present_in_base_graph"] is True
