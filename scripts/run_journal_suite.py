"""One-command journal experiment runner.

Packages the deferred journal experiment queue into one reproducible entrypoint.
It shells out to the existing ``cdd_oran.cli`` (train / evaluate) and the existing
analysis modules (threshold_sweep, edge_stability, stats). It does NOT reimplement
training, evaluation, or statistics.

Two budget profiles:
- DEFAULT (light, safe): total_steps<=1500, 1 seed, small planner sim/sample
  counts, so a smoke of the whole pipeline finishes fast and proves the wiring.
- ``--full``: Env I 20000 / Env II 50000 / Env III 50000 steps, 3 seeds, default
  planner counts. This is the real journal run.

Usage:
    uv run python scripts/run_journal_suite.py --dry-run
    uv run python scripts/run_journal_suite.py
    uv run python scripts/run_journal_suite.py --full
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ENVS = ["EnvironmentI", "EnvironmentII", "EnvironmentIII"]
CONFIG_CDL = {
    "EnvironmentI": "configs/env_i_cdl.yaml",
    "EnvironmentII": "configs/env_ii_cdl.yaml",
    "EnvironmentIII": "configs/env_iii_cdl.yaml",
}
CONFIG_MLP = {
    "EnvironmentI": "configs/env_i_mlp.yaml",
    "EnvironmentII": "configs/env_ii_mlp.yaml",
    "EnvironmentIII": None,  # generated from the CDL config on first use
}
FULL_STEPS = {"EnvironmentI": 20000, "EnvironmentII": 50000, "EnvironmentIII": 50000}
DEFAULT_STEPS = 1500
DEFAULT_INIT = 400
DEFAULT_PLANNER_OVERRIDES = [
    "mcts.n_simulations=64",
    "mppi.n_samples=64",
    "cem.n_candidate=16",
    "cem.n_top=8",
    "cem.n_iter=5",
]
KAPPAS = [0.0, 0.25, 0.5, 1.0, 2.0]
GRAPH_OVERRIDES = ["causal", "full", "correlation"]
STAGE_NAMES = {
    1: "recovery",
    2: "world_models",
    3: "mitigation",
    4: "risk_sweep",
    5: "stats",
    6: "emit",
}

TEMP_CFG_DIR = Path(".temp/configs")
TEMP_DATA_DIR = Path(".temp/data/journal")


def _module_command(module, args):
    return ["uv", "run", "python", "-m", f"cdd_oran.analysis.{module}", *args]


def _cli_command(args):
    return ["uv", "run", "python", "-m", "cdd_oran.cli", *args]


def _train_args(config, seed, overrides, device=None):
    args = ["train", "--config", config, "--seed", str(seed)]
    for override in overrides:
        args += ["--set", override]
    if device:
        args += ["--set", f"device={device}"]
    return args


def _read_json(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else {}


def _set_path(data, key, value):
    *parents, leaf = key.split(".")
    target = data
    for parent in parents:
        target = target[parent]
    target[leaf] = value


def _dotted(data, key):
    target = data
    for part in key.split("."):
        target = target[part]
    return target


def _config_mlp(env):
    path = CONFIG_MLP[env]
    if path is not None and Path(path).exists():
        return path
    if env != "EnvironmentIII":
        raise FileNotFoundError(f"MLP config for {env} not found: {path}")
    source = CONFIG_CDL[env]
    data = yaml.safe_load(Path(source).read_text())
    data["model_kind"] = "mlp"
    generated = TEMP_CFG_DIR / Path(source).name.replace("_cdl", "_mlp")
    generated.parent.mkdir(parents=True, exist_ok=True)
    generated.write_text(yaml.safe_dump(data, sort_keys=False))
    return str(generated)


def _load_run_cfg(run_dir):
    return yaml.safe_load((Path(run_dir) / "config.yaml").read_text())


def _variant_config(run_dir, changes, name, device=None):
    data = _load_run_cfg(run_dir)
    for key, value in changes.items():
        _set_path(data, key, value)
    if device:
        data["device"] = device
    out = TEMP_CFG_DIR / f"{Path(run_dir).name}-{name}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(data, sort_keys=False))
    return str(out)


def discover_run(env, model_kind, seed, after=None):
    """Newest run under runs/<env>/<kind> matching env/kind/seed, optionally after mtime."""
    root = Path("runs") / env / model_kind
    if not root.is_dir():
        return None
    candidates = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        cfg_path = entry / "config.yaml"
        if not cfg_path.exists():
            continue
        try:
            cfg = yaml.safe_load(cfg_path.read_text())
        except yaml.YAMLError:
            continue
        if cfg.get("environment") != env or cfg.get("model_kind") != model_kind:
            continue
        if cfg.get("seed") != seed:
            continue
        if after is not None and entry.stat().st_mtime < after:
            continue
        candidates.append(entry)
    if not candidates:
        return None
    return max(candidates, key=lambda entry: entry.stat().st_mtime)


class Step:
    def __init__(self, stage, label, argv, out_json=None):
        self.stage = stage
        self.label = label
        self.argv = argv
        self.out_json = out_json
        self.ok = None
        self.returncode = None
        self.error = None

    def run(self):
        command = list(self.argv)
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=False)
        except FileNotFoundError as error:
            self.ok = False
            self.error = f"command not found: {error}"
            return False
        self.returncode = result.returncode
        self.ok = result.returncode == 0
        if not self.ok:
            tail = "\n".join((result.stderr or "").strip().splitlines()[-8:])
            self.error = tail or f"exit code {result.returncode}"
            return False
        return True

    def to_dict(self):
        return {
            "stage": self.stage,
            "label": self.label,
            "command": " ".join(self.argv),
            "ok": self.ok,
            "returncode": self.returncode,
            "error": self.error,
        }


class Suite:
    def __init__(self, args):
        self.args = args
        self.steps: list[Step] = []
        self.trained: dict[tuple, str] = {}
        self.recovery = {}
        self.world_models = {}
        self.mitigation = {}
        self.satisfaction = {}
        self.risk_sweep = {}
        self.stats_rows = []
        self.any_failed = False

    def _run_dir(self, env, model_kind, seed):
        key = (env, model_kind, seed)
        if key in self.trained:
            return self.trained[key]
        existing = discover_run(env, model_kind, seed)
        if existing is not None:
            self.trained[key] = str(existing)
            return str(existing)
        return f"<missing runs/{env}/{model_kind} seed={seed}>"

    def _do(self, stage, label, argv, out_json=None):
        step = Step(stage, label, argv, out_json)
        self.steps.append(step)
        if self.args.dry_run:
            step.ok = None
            return True
        if not step.run():
            self.any_failed = True
            print(f"[FAIL] {label}: {step.error}")
            return False
        return True

    def run_recovery(self):
        stage = 1
        for env in self.args.envs:
            self.recovery[env] = {}
            for seed in self.args.seeds:
                self.recovery[env][seed] = {}
                overrides = self._budget_overrides(env)
                train_argv = _cli_command(
                    _train_args(CONFIG_CDL[env], seed, overrides, self.args.device)
                )
                start = time.time()
                if not self._do(stage, f"train CDL {env} seed {seed}", train_argv):
                    continue
                run_dir = discover_run(env, "cdl", seed, after=start)
                if run_dir is None:
                    self._do(stage, f"discover CDL run {env} seed {seed}", [])
                    continue
                self.trained[(env, "cdl", seed)] = str(run_dir)
                entry = self.recovery[env][seed]
                entry["run_dir"] = str(run_dir)
                metrics = _read_json(run_dir / "metrics.json")
                entry["graph"] = metrics.get("graph")

                sweep_out = TEMP_DATA_DIR / f"{Path(run_dir).name}-sweep.json"
                sweep_argv = _module_command(
                    "threshold_sweep",
                    ["--run", str(run_dir), "--grid", self.args.grid, "--out", str(sweep_out)],
                )
                if self._do(stage, f"threshold sweep {env} seed {seed}", sweep_argv, sweep_out):
                    entry["threshold_sweep"] = _read_json(sweep_out)

                stab_out = TEMP_DATA_DIR / f"{Path(run_dir).name}-stability.json"
                stab_argv = _module_command(
                    "edge_stability",
                    [
                        "--run",
                        str(run_dir),
                        "--B",
                        str(self.args.B),
                        "--n-transitions",
                        str(self.args.n_transitions),
                        "--out",
                        str(stab_out),
                    ],
                )
                if self._do(stage, f"edge stability {env} seed {seed}", stab_argv, stab_out):
                    stab = _read_json(stab_out)
                    if "frequency_matrix" in stab:
                        stab.pop("frequency_matrix", None)
                    entry["edge_stability"] = stab

    def run_world_models(self):
        stage = 2
        for env in self.args.envs:
            self.world_models[env] = {}
            for seed in self.args.seeds:
                config = _config_mlp(env)
                overrides = self._budget_overrides(env)
                train_argv = _cli_command(_train_args(config, seed, overrides, self.args.device))
                start = time.time()
                if not self._do(stage, f"train MLP {env} seed {seed}", train_argv):
                    continue
                run_dir = discover_run(env, "mlp", seed, after=start)
                if run_dir is None:
                    self._do(stage, f"discover MLP run {env} seed {seed}", [])
                    continue
                self.trained[(env, "mlp", seed)] = str(run_dir)
                self.world_models[env][seed] = {
                    "run_dir": str(run_dir),
                    "graph": _read_json(run_dir / "metrics.json").get("graph"),
                }

    def run_mitigation(self):
        stage = 3
        for env in self.args.envs:
            self.mitigation[env] = {}
            self.satisfaction[env] = {}
            for seed in self.args.seeds:
                self.mitigation[env][seed] = {}
                self.satisfaction[env][seed] = {}
                cdl_run = self._run_dir(env, "cdl", seed)
                mlp_run = self._run_dir(env, "mlp", seed)
                for mseed in self.args.mitigation_seeds:
                    self.mitigation[env][seed][mseed] = {}
                    self.satisfaction[env][seed][mseed] = {}
                    for override in self.args.graph_overrides:
                        cdl_cfg = _variant_config(
                            cdl_run, {"mitigation_seed": mseed}, f"mit-{mseed}", self.args.device
                        )
                        cdl_argv = _cli_command(
                            [
                                "evaluate",
                                "--run",
                                cdl_run,
                                "--config",
                                cdl_cfg,
                                "--graph-override",
                                override,
                            ]
                        )
                        if self._do(
                            stage, f"evaluate CDL {env} s{seed} m{mseed} {override}", cdl_argv
                        ):
                            self._set_mitigation(
                                (env, seed, mseed, override, "cdl"),
                                self._planner_utilities(cdl_run),
                            )
                            self._set_satisfaction(
                                (env, seed, mseed, override, "cdl"),
                                self._run_satisfaction(cdl_run),
                            )
                        mlp_cfg = _variant_config(
                            mlp_run, {"mitigation_seed": mseed}, f"mit-{mseed}", self.args.device
                        )
                        mlp_argv = _cli_command(
                            [
                                "evaluate",
                                "--run",
                                mlp_run,
                                "--config",
                                mlp_cfg,
                                "--graph-run",
                                cdl_run,
                                "--graph-override",
                                override,
                            ]
                        )
                        if self._do(
                            stage, f"evaluate MLP {env} s{seed} m{mseed} {override}", mlp_argv
                        ):
                            self._set_mitigation(
                                (env, seed, mseed, override, "mlp"),
                                self._planner_utilities(mlp_run),
                            )
                            self._set_satisfaction(
                                (env, seed, mseed, override, "mlp"),
                                self._run_satisfaction(mlp_run),
                            )

    def _planner_utilities(self, run_dir):
        metrics = _read_json(Path(run_dir) / "metrics.json")
        utilities = metrics.get("evaluation", {}).get("planner_mean_utilities", {})
        return {name: value for name, value in utilities.items() if value is not None}

    def _set_mitigation(self, key, utilities):
        env, seed, mseed, override, model_kind = key
        self.mitigation[env][seed][mseed].setdefault(override, {})[model_kind] = utilities

    def _run_satisfaction(self, run_dir):
        """Per-planner satisfaction rate from the run's v2 utilities.json (no env needed).

        Calls stats.satisfaction_rate (do not reimplement). Returns {} for a v1 file or
        a missing utilities.json, so old runs degrade gracefully.
        """
        path = Path(run_dir) / "utilities.json"
        if not path.exists():
            return {}
        from cdd_oran.analysis import stats as stats_module

        result = stats_module.satisfaction_rate(str(path))
        return result.get("satisfaction", {}) if result.get("ok") else {}

    def _set_satisfaction(self, key, rates):
        env, seed, mseed, override, model_kind = key
        self.satisfaction[env][seed][mseed].setdefault(override, {})[model_kind] = rates

    def run_risk_sweep(self):
        stage = 4
        for env in self.args.envs:
            self.risk_sweep[env] = {}
            for seed in self.args.seeds:
                self.risk_sweep[env][seed] = {}
                cdl_run = self._run_dir(env, "cdl", seed)
                for kappa in self.args.kappas:
                    cfg = _variant_config(
                        cdl_run, {"planner.risk_kappa": kappa}, f"kappa-{kappa}", self.args.device
                    )
                    argv = _cli_command(
                        [
                            "evaluate",
                            "--run",
                            cdl_run,
                            "--config",
                            cfg,
                            "--graph-override",
                            "causal",
                        ]
                    )
                    if self._do(stage, f"risk kappa={kappa} {env} seed {seed}", argv):
                        self.risk_sweep[env][seed][kappa] = self._planner_utilities(cdl_run)

    def stats(self):
        import numpy as np

        from cdd_oran.analysis import stats as stats_module

        for env in self.args.envs:
            cdl_by_planner: dict[str, dict] = {}
            mlp_by_planner: dict[str, dict] = {}
            sat_cdl: dict[str, list] = {}
            sat_mlp: dict[str, list] = {}
            for seed in self.args.seeds:
                for mseed in self.args.mitigation_seeds:
                    causal = (
                        self.mitigation.get(env, {}).get(seed, {}).get(mseed, {}).get("causal", {})
                    )
                    cdl = causal.get("cdl", {})
                    mlp = causal.get("mlp", {})
                    for planner in sorted(set(cdl) & set(mlp)):
                        # One paired scalar per (seed, mseed): the run's mean utility.
                        cdl_by_planner.setdefault(planner, {})[(seed, mseed)] = cdl[planner]
                        mlp_by_planner.setdefault(planner, {})[(seed, mseed)] = mlp[planner]
                    for planner, rate in (
                        self.satisfaction.get(env, {})
                        .get(seed, {})
                        .get(mseed, {})
                        .get("causal", {})
                        .get("cdl", {})
                        .items()
                    ):
                        sat_cdl.setdefault(planner, []).append(rate)
                    for planner, rate in (
                        self.satisfaction.get(env, {})
                        .get(seed, {})
                        .get(mseed, {})
                        .get("causal", {})
                        .get("mlp", {})
                        .items()
                    ):
                        sat_mlp.setdefault(planner, []).append(rate)
            for planner, pairs in sorted(cdl_by_planner.items()):
                keys = sorted(pairs)
                cdl_vals = np.array([pairs[key] for key in keys], dtype=float)
                mlp_vals = np.array([mlp_by_planner[planner][key] for key in keys], dtype=float)
                row = {
                    "environment": env,
                    "planner": planner,
                    # Robust, seed-based recipe (report 23): paired diff + CI, Wilcoxon,
                    # Cliff's delta, P(CDL>MLP), plus Cohen's d kept as secondary.
                    **stats_module.compare_seeds(cdl_vals, mlp_vals, seed=self.args.seed),
                    "satisfaction_cdl": (
                        float(np.mean(sat_cdl[planner])) if sat_cdl.get(planner) else None
                    ),
                    "satisfaction_mlp": (
                        float(np.mean(sat_mlp[planner])) if sat_mlp.get(planner) else None
                    ),
                }
                self.stats_rows.append(row)

        # Family-wide multiple-comparison correction over every (env, planner) cell.
        pvals = [row["wilcoxon"]["p_value"] for row in self.stats_rows]
        if pvals:
            holm = stats_module.holm_correction(pvals)
            bh = stats_module.bh_fdr(pvals)  # kept available for the wider exploratory grid
            for row, p_holm, p_bh in zip(self.stats_rows, holm, bh, strict=True):
                row["wilcoxon_p_holm"] = float(p_holm)
                row["wilcoxon_p_bh"] = float(p_bh)

    def emit(self):
        out_dir = Path(self.args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "profile": "full" if self.args.full else "default",
            "seeds": self.args.seeds,
            "mitigation_seeds": self.args.mitigation_seeds,
            "envs": self.args.envs,
            "recovery": self.recovery,
            "world_models": self.world_models,
            "mitigation": self.mitigation,
            "satisfaction": self.satisfaction,
            "risk_sweep": self.risk_sweep,
            "stats": self.stats_rows,
            "steps": [step.to_dict() for step in self.steps],
            "any_failed": self.any_failed,
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
        (out_dir / "table.md").write_text(self._markdown_table())
        return out_dir

    def _markdown_table(self):
        lines = ["# Journal suite results", ""]
        lines.append(f"Profile: {'full' if self.args.full else 'default'}")
        lines.append(f"Seeds: {self.args.seeds}")
        lines.append(f"Mitigation seeds: {self.args.mitigation_seeds}")
        lines.append(f"Environments: {self.args.envs}")
        lines.append("")
        if self.stats_rows:
            lines.append("## CDL vs MLP (causal graph) - robust, seed-based recipe")
            lines.append("")
            lines.append(
                "Replication unit: independent seeds (paired per-seed mean utility). "
                "Lead with the paired diff + CI, Cliff's delta, P(CDL>MLP) and satisfaction; "
                "Cohen's d is secondary (assumption-bound: normality of paired diffs). "
                "Wilcoxon p is Holm-adjusted across all (env, planner) cells."
            )
            lines.append("")
            lines.append(
                "| env | planner | n | diff [95% CI] | Cliff's delta (label) | P(CDL>MLP) | "
                "sat CDL | sat MLP | Wilcoxon p (Holm) | Cohen's d (secondary) |"
            )
            lines.append(
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
            )
            for row in self.stats_rows:
                ci = row["ci_low"], row["ci_high"]
                p_holm = row.get("wilcoxon_p_holm", row["wilcoxon"]["p_value"])
                sat_cdl = row.get("satisfaction_cdl")
                sat_mlp = row.get("satisfaction_mlp")
                lines.append(
                    f"| {row['environment']} | {row['planner']} | {row['n_seeds']} | "
                    f"{row['mean_diff']:.4f} [{ci[0]:.4f}, {ci[1]:.4f}] | "
                    f"{row['cliffs_delta']:.3f} ({row['cliffs_label']}) | "
                    f"{row['prob_cdl_gt_mlp']:.3f} | "
                    f"{'-' if sat_cdl is None else f'{sat_cdl:.3f}'} | "
                    f"{'-' if sat_mlp is None else f'{sat_mlp:.3f}'} | "
                    f"{p_holm:.4f} | {row['cohens_d']:.4f} |"
                )
            lines.append("")
        if self.recovery:
            lines.append("## Causal recovery (CDL)")
            lines.append("")
            lines.append("| env | seed | run | F1 | sweep best F1 | stability best F1 |")
            lines.append("| --- | --- | --- | --- | --- | --- |")
            for env in self.args.envs:
                for seed, entry in self.recovery.get(env, {}).items():
                    graph = entry.get("graph") or {}
                    sweep = entry.get("threshold_sweep") or {}
                    stab = entry.get("edge_stability") or {}
                    sweep_f1 = (sweep.get("best") or {}).get("f1")
                    stab_f1 = None
                    if "frequency_cut_table" in stab:
                        stab_f1 = max(row["f1"] for row in stab["frequency_cut_table"])
                    lines.append(
                        f"| {env} | {seed} | {entry.get('run_dir', '-')} | "
                        f"{graph.get('f1') if graph.get('f1') is not None else '-'} | "
                        f"{sweep_f1 if sweep_f1 is not None else '-'} | "
                        f"{stab_f1 if stab_f1 is not None else '-'} |"
                    )
            lines.append("")
        if self.risk_sweep:
            lines.append("## Risk sweep (CDL, causal graph)")
            lines.append("")
            lines.append("| env | kappa | planner | mean utility |")
            lines.append("| --- | --- | --- | --- |")
            for env in self.args.envs:
                for _seed, kappas in self.risk_sweep.get(env, {}).items():
                    for kappa, utilities in kappas.items():
                        for planner, value in utilities.items():
                            lines.append(f"| {env} | {kappa} | {planner} | {value:.4f} |")
            lines.append("")
        lines.append("## Step status")
        lines.append("")
        lines.append("| stage | step | ok |")
        lines.append("| --- | --- | --- |")
        for step in self.steps:
            ok = "PASS" if step.ok else ("--" if step.ok is None else "FAIL")
            lines.append(f"| {step.stage} | {step.label} | {ok} |")
        lines.append("")
        return "\n".join(lines)

    def _budget_overrides(self, env):
        if self.args.full:
            return [f"train.total_steps={FULL_STEPS[env]}"]
        overrides = [f"train.total_steps={DEFAULT_STEPS}", f"train.init_steps={DEFAULT_INIT}"]
        return overrides + list(DEFAULT_PLANNER_OVERRIDES)

    def _counts(self):
        n_env, n_seed, n_mit, n_ov, n_kappa = (
            len(self.args.envs),
            len(self.args.seeds),
            len(self.args.mitigation_seeds),
            len(self.args.graph_overrides),
            len(self.args.kappas),
        )
        return {
            "trains": 2 * n_env * n_seed,
            "analyses": n_env * n_seed * 2,
            "mitigation_evals": n_env * n_seed * n_mit * n_ov * 2,
            "risk_evals": n_env * n_seed * n_kappa,
            "total_commands": 2 * n_env * n_seed
            + n_env * n_seed * 2
            + n_env * n_seed * n_mit * n_ov * 2
            + n_env * n_seed * n_kappa,
        }


def build_parser():
    parser = argparse.ArgumentParser(
        description="One-command journal experiment runner (shells out to cdd_oran.cli)"
    )
    parser.add_argument(
        "--full", action="store_true", help="Real journal profile (20000/50000 steps, 3 seeds)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the full command plan and exit"
    )
    parser.add_argument(
        "--emit-selftest",
        action="store_true",
        help="Feed synthetic seed arrays + a synthetic v2 utilities dict through the "
        "stats/emit code and print the resulting table.md (no training).",
    )
    parser.add_argument("--seeds", default=None, help="Comma-separated training seeds")
    parser.add_argument(
        "--mitigation-seeds",
        default=None,
        help="Comma-separated mitigation seeds (default: same as --seeds)",
    )
    parser.add_argument(
        "--envs", default=",".join(ENVS), help="Comma-separated environments (default: all three)"
    )
    parser.add_argument("--stages", default="1,2,3,4,5,6", help="Comma-separated stage numbers")
    parser.add_argument(
        "--graph-overrides",
        default=",".join(GRAPH_OVERRIDES),
        help="Comma-separated graph overrides (causal,full,correlation)",
    )
    parser.add_argument(
        "--kappa", default=",".join(str(k) for k in KAPPAS), help="Comma-separated risk kappas"
    )
    parser.add_argument(
        "--grid", default="0.02,0.40,20", help="threshold_sweep grid start,stop,num"
    )
    parser.add_argument("--B", type=int, default=50, help="edge_stability bootstrap resamples")
    parser.add_argument("--n-transitions", type=int, default=2048, help="edge_stability pool size")
    parser.add_argument("--out", default="results/journal", help="Output directory")
    parser.add_argument("--device", default=None, help="Force device (cuda/cpu) for all stages")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for stats bootstrap")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.seeds is None:
        args.seeds = [0, 1, 2] if args.full else [45]
    else:
        args.seeds = [int(value) for value in args.seeds.split(",")]
    args.mitigation_seeds = (
        list(args.seeds)
        if args.mitigation_seeds is None
        else [int(value) for value in args.mitigation_seeds.split(",")]
    )
    args.envs = [value.strip() for value in args.envs.split(",") if value.strip()]
    stages = {int(value) for value in args.stages.split(",")}
    if not stages <= set(range(1, 7)):
        raise SystemExit("--stages must be comma-separated numbers in 1..6")
    args.kappas = [float(value) for value in args.kappa.split(",")]
    args.graph_overrides = [value.strip() for value in args.graph_overrides.split(",")]

    if args.emit_selftest:
        return _emit_selftest(args)

    suite = Suite(args)
    counts = suite._counts()
    profile = "FULL" if args.full else "DEFAULT"
    print(f"Journal suite plan [{profile}]")
    print(f"  stages: {sorted(stages)}")
    print(f"  environments: {args.envs}")
    print(f"  seeds: {args.seeds}")
    print(f"  mitigation seeds: {args.mitigation_seeds}")
    print(f"  graph overrides: {args.graph_overrides}")
    print(f"  risk kappas: {args.kappas}")
    print("  estimated run count:")
    print(f"    trains:            {counts['trains']}")
    print(f"    analyses:          {counts['analyses']}")
    print(f"    mitigation evals:  {counts['mitigation_evals']}")
    print(f"    risk evals:        {counts['risk_evals']}")
    print(f"    total subprocess:  {counts['total_commands']}")

    if args.dry_run:
        _dry_run(stages, suite, args)
        return 0

    if 1 in stages:
        suite.run_recovery()
    if 2 in stages:
        suite.run_world_models()
    if 3 in stages:
        suite.run_mitigation()
    if 4 in stages:
        suite.run_risk_sweep()
    if 5 in stages:
        suite.stats()
    if 6 in stages:
        out_dir = suite.emit()
        print(f"Journal results written to {out_dir}")

    print("\nStep status:")
    for step in suite.steps:
        status = "PASS" if step.ok else "FAIL"
        if step.ok is None:
            status = "--"
        print(f"  [{status}] {step.stage:>2}  {step.label}")
        if step.error:
            print(f"         error: {step.error}")

    return 1 if suite.any_failed else 0


def _emit_selftest(args):
    """No-training check: synthetic seed arrays + a synthetic v2 utilities dict driven
    through the real stats functions and the emit table builder."""
    import os
    import tempfile

    import numpy as np

    from cdd_oran.analysis import stats as stats_module

    v2 = {
        "version": 2,
        "algorithm_names": ["QACM", "ModelBasedMPPI"],
        "steps": [
            {"step": 0, "panels": [
                {"conflict_xapp_ids": [0, 1],
                 "planner_satisfied": {"QACM": [1, 1], "ModelBasedMPPI": [1, 0]}},
            ]},
        ],
    }
    fd, tmp_path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(v2, handle)
        sat = stats_module.satisfaction_rate(tmp_path)
    finally:
        os.remove(tmp_path)
    assert sat["ok"] and sat["source"] == "planner_satisfied (v2)", sat

    cdl = np.array([0.90, 0.92, 0.88])
    mlp = np.array([0.61, 0.59, 0.63])
    rows: list[dict[str, Any]] = []
    for env, planner in [("EnvironmentI", "QACM"), ("EnvironmentI", "ModelBasedMPPI")]:
        row: dict[str, Any] = {
            "environment": env,
            "planner": planner,
            **stats_module.compare_seeds(cdl, mlp, seed=args.seed),
            "satisfaction_cdl": sat["satisfaction"].get(planner),
            "satisfaction_mlp": 0.5,
        }
        rows.append(row)
    holm = stats_module.holm_correction([r["wilcoxon"]["p_value"] for r in rows])
    bh = stats_module.bh_fdr([r["wilcoxon"]["p_value"] for r in rows])
    for row, p_holm, p_bh in zip(rows, holm, bh, strict=True):
        row["wilcoxon_p_holm"] = float(p_holm)
        row["wilcoxon_p_bh"] = float(p_bh)

    suite = Suite(args)
    suite.stats_rows = rows
    table = suite._markdown_table()

    for column in ("diff [95% CI]", "Cliff's delta (label)", "P(CDL>MLP)",
                   "sat CDL", "sat MLP", "Wilcoxon p (Holm)", "Cohen's d (secondary)"):
        assert column in table, f"missing recipe column: {column}"
    assert "0.500" in table  # satisfaction_mlp rendered
    print(table)
    print("emit self-test OK: recipe columns present, stats via stats.py (no training)")
    return 0


def _dry_cfg(run_dir, suffix):
    if str(run_dir).startswith("<"):
        return "<generated temp config>"
    return f"<temp-config {Path(str(run_dir)).name} {suffix}>"


def _dry_run(stages, suite, args):
    print("\nDry run: no commands are executed.")
    if 1 in stages:
        for env in args.envs:
            for seed in args.seeds:
                overrides = suite._budget_overrides(env)
                print(
                    " ".join(
                        _cli_command(_train_args(CONFIG_CDL[env], seed, overrides, args.device))
                    )
                )
                run_dir = suite._run_dir(env, "cdl", seed)
                print(
                    " ".join(
                        _module_command("threshold_sweep", ["--run", run_dir, "--grid", args.grid])
                    )
                )
                print(
                    " ".join(
                        _module_command(
                            "edge_stability",
                            [
                                "--run",
                                run_dir,
                                "--B",
                                str(args.B),
                                "--n-transitions",
                                str(args.n_transitions),
                            ],
                        )
                    )
                )
    if 2 in stages:
        for env in args.envs:
            for seed in args.seeds:
                config = _config_mlp(env)
                overrides = suite._budget_overrides(env)
                print(" ".join(_cli_command(_train_args(config, seed, overrides, args.device))))
    if 3 in stages:
        for env in args.envs:
            for seed in args.seeds:
                cdl_run = suite._run_dir(env, "cdl", seed)
                mlp_run = suite._run_dir(env, "mlp", seed)
                for mseed in args.mitigation_seeds:
                    for override in args.graph_overrides:
                        print(
                            " ".join(
                                _cli_command(
                                    [
                                        "evaluate",
                                        "--run",
                                        cdl_run,
                                        "--config",
                                        _dry_cfg(cdl_run, f"mit-{mseed}"),
                                        "--graph-override",
                                        override,
                                    ]
                                )
                            )
                        )
                        print(
                            " ".join(
                                _cli_command(
                                    [
                                        "evaluate",
                                        "--run",
                                        mlp_run,
                                        "--config",
                                        _dry_cfg(mlp_run, f"mit-{mseed}"),
                                        "--graph-run",
                                        cdl_run,
                                        "--graph-override",
                                        override,
                                    ]
                                )
                            )
                        )
    if 4 in stages:
        for env in args.envs:
            for seed in args.seeds:
                cdl_run = suite._run_dir(env, "cdl", seed)
                for kappa in args.kappas:
                    print(
                        " ".join(
                            _cli_command(
                                [
                                    "evaluate",
                                    "--run",
                                    cdl_run,
                                    "--config",
                                    _dry_cfg(cdl_run, f"kappa-{kappa}"),
                                    "--graph-override",
                                    "causal",
                                ]
                            )
                        )
                    )
    if 5 in stages:
        print(
            "stats: in-process cdd_oran.analysis.stats - compare_seeds per (env, planner) "
            "[paired diff+CI, Wilcoxon, Cliff's delta, P(CDL>MLP), Cohen's d], "
            "holm_correction over the family (bh_fdr kept), and satisfaction_rate per run "
            "(v2 planner_satisfied, averaged across seeds)"
        )
    if 6 in stages:
        print(
            f"emit: write {args.out}/summary.json and {args.out}/table.md "
            "(recipe columns: diff+CI, Cliff's delta, P(CDL>MLP), sat CDL/MLP, "
            "Wilcoxon p Holm-adjusted, Cohen's d secondary)"
        )


if __name__ == "__main__":
    raise SystemExit(main())
