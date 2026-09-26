# CDD-ORAN Codebase Remediation Plan

**Goal:** take the conference-paper codebase to a state where journal-scale experiments
(multi-seed statistics, $H>1$ planning, joint multi-conflict optimisation, online CDL,
new environments) can be run, validated, and defended without fighting the code.

**Status:** planning. Nothing implemented yet.

DO NOT COMMIT THIS FILE TO GIT.

---

## Context

The repo implements *CDD O-RAN* (Dahal, Dhakal, Panta, Dawadi, Gautam — conference 2026):
a Causal Discovery Layer learns a sparse NCP→KPI graph offline, the frozen graph drives
conflict enumeration and masks a world model for model-based planning by four planners
(QACM, CEM, MPPI, MCTS).

3,141 LOC across 22 files. It produced the paper's results, but it cannot currently
support the journal extension:

- **Config is a star-imported global module.** `from Parameters import *` in 9 files.
  Changing any setting means editing source. Running two configs in one process is
  impossible, so multi-seed sweeps — which the conclusion itself admits are needed
  ("broader validation across additional seeds") — cannot be scripted.
- **`import Parameters` runs 15M-sample Monte Carlo and prints to stdout**
  (`Parameters/__init__.py:77,94,116,131`). Every import of every module pays this.
- **No tests, no CI, no type checking, no linting.** Nothing protects the numbers.
- **~600 duplicated lines across the two environments**, `_weighted_distance` copied
  5× (one copy silently divergent), `_score_batch` 2×, `set_param1..8` 8×.
- **Paper↔code divergence** (see Phase 0) — the repo as published does not reproduce
  the published table.

The plan below is ordered so the code is never broken between steps, and so that the
paper's numbers are locked down *before* any restructuring begins.

---

# QUESTIONNAIRE — for the team

Everything below is blocked on a decision we cannot make from the code alone. Each item
gives the discrepancy, why it matters, and our recommendation. **Section A blocks
regenerating the results table; B and C do not.**

## A. Paper ↔ code mismatches (blocking)

The repo as committed does not reproduce the published table. Three concrete gaps:

**A1. MCTS simulation count.** Paper §IV says $N_{sim} = 3000$. Code has
`n_simulations: 3` (commit `b017895`, "changed mcts parameters for faster execution").
*Which produced Table `tab:mitigation`?*
→ **Recommendation:** assume the paper is right and the value was lowered afterwards for
speed. Restore 3000 and regenerate. Please confirm.

**A2. Evaluation step count.** Paper says $N_{eval} = 30$. Code has `num_steps: 10`.
*Which produced the table?*
→ **Recommendation:** restore 30 to match the paper.

**A3. Environment I OOD ranges — the serious one.** The paper lists distinct OOD ranges
($p_1,p_2 \in [-10,310]$, $p_3 \in [-5,8]$, $p_4 \in [-7,10]$, $p_5 \in [-10,13]$,
$p_7 \in [-3,6]$). In the code the "train" and "ood" branches are **byte-identical** —
both are the training ranges. So as committed, the Env I "OOD" column was evaluated on
*training* ranges, which would undercut the Env I generalisation claim ($d = 0.35$).
*Were the ranges reverted after the figures were made, or were the Env I results produced
on training ranges?*
→ **Recommendation:** if the paper's ranges are correct, restore them and regenerate Env I.
If not, the Env I OOD claim needs rewording. **This is the single most important answer.**

**A4. Do the trained checkpoints still exist?** No `.pt` files are in the repo and none
were archived. Without them we cannot reproduce the published numbers — only regenerate
them. *Does anyone have the original checkpoints or the exact configs used?*

**A5. Seeds.** Paper states seed 45 for discovery, 0 for mitigation. Configs currently use
seed 0 throughout. *Confirm the 45/0 split is intended?*

## B. Semantics we preserved but could not verify

We deliberately kept current behaviour in all four cases rather than "fixing" them.

**B1. The evaluation loop never applies the chosen action.** After each step,
`evaluate` advances the env with a zero action, so all $N_{eval}$ steps are measured from
one fixed trajectory. This is consistent with the paper's "each conflict resolved
independently" framing, but it is undocumented and reads like a bug.
→ **Recommendation:** if intentional, state it explicitly in the journal methodology.

**B2. Per-xApp weights $w_x$ are inert.** The code builds the weights array, multiplies by
1.0, then renormalises to all-ones — so $w_x$, a symbol in the paper's cost function
$\mathcal{F} = \beta\sum_x w_x d_x - (\sum_x s_x)^2$, has no effect.
*Were non-uniform weights ever intended?*
→ **Recommendation:** either implement real weights or drop $w_x$ from the paper's equation.
Leaving a live-looking symbol inert invites a reviewer question.

**B3. Two conflicting normalisations of the cost function.** Four copies of the
weighted-distance function used `xapp.mean/.std`; a fifth (in the now-deleted exploration
policy) used `env.kpis[xapp_idx].mean/.std`. These differ in Env II, where xApp 4
aggregates $K_{41}$ and $K_{42}$. We kept the `xapp` version (the one that produced the
results). *Confirm that is the intended normalisation?*

**B4. Env I variance floor differs between runtime and statistics estimator.** Runtime uses
`max(P, 1e-3)`; the estimator uses `where(P > 0, P, 1e-3)`, which leaves small positive
values unfloored. We measured the impact: 1,019 of 4,000,000 samples disagree (0.026%) and
the resulting mean/std are **identical to 10 decimal places** — so published statistics are
unaffected. Env II is already consistent.
→ **Recommendation:** align on `max(P, 1e-3)` for hygiene. No result changes. Low priority.

## C. Journal planning

**C1. Numeric parameter table.** Per-xApp thresholds $T_x$, weights $w_x$, directions
$\delta_x$ and the scale $\beta$ (hardcoded as `scaling_term = 10`) appear only as symbols
in the paper — the values live solely in code. A reviewer will flag this as a
reproducibility gap.
→ **Recommendation:** add a numeric table to the journal version. *Confirm $\beta = 10$.*

**C2. Seed count for the multi-seed sweep — now a compute question.** The current $\pm$ in
`tab:mitigation` is dispersion **within one run** (across steps, edges, planners); it has
no run-to-run component, so Cohen's $d = 1.79$ does not speak to reproducibility. The sweep
is built and varies the **discovery** seed (retraining per seed), since the claim is about
the learned model's quality.

**Measured cost (CPU — this machine has no usable GPU; `device: auto` → `cpu`):**
Env I CDL 10,000 steps took **62 min** (~0.37 s/step), dominated by `update_mask`
recomputing CMI every `eval_steps`. MLP is ~30× cheaper (no `update_mask`).

| Per paired seed (CDL+MLP) | 3 seeds | 5 seeds | 10 seeds |
|---|---|---|---|
| Env I ≈ 2 h 08 m | 6 h 23 m | 10 h 39 m | 21 h 18 m |
| Env II ≈ 6 h 47 m | 20 h 20 m | 33 h 54 m | 67 h 47 m |
| **Both ≈ 8 h 55 m** | **~27 h** | **~45 h** | **~89 h** |

→ **Recommendation:** run it on a GPU or lab machine if one exists — *what does the team
have access to?* On CPU, 5 seeds is ~2 days of continuous compute and 10 is infeasible.
Fallbacks, in order of preference: (a) GPU; (b) 3 seeds both environments (~27 h);
(c) multi-seed Env I only, single-seed Env II — though Env II is where the effect is
strongest ($d = 1.79$), so that weakens the headline claim; (d) reduce `update_mask`
frequency — cheapest by far, but it changes the CMI estimation methodology, so it is a
paper decision, not an engineering one.

**C3. Which Phase 7 extensions are actually planned?** Multi-step planning ($H>1$), joint
multi-conflict optimisation, online CDL under non-stationary xApps, contemporaneous causal
edges (relaxing A4), real O-RAN traces, per-planner latency benchmarking. This changes what
we build now versus defer.
→ **Recommendation:** confirm the top two or three; we will shape the architecture around
those and stop generalising for the rest.

## Known change already made (FYI, no decision needed)

Removing MPPI's dead warm-start deleted a `.sample()` call that was consuming torch RNG, so
MPPI's sampling stream shifted. **MPPI numbers will differ slightly from the conference
run.** The old behaviour was an unintended side effect of dead code, not a design choice —
but the MPPI column must be regenerated, not copied. Verified confined to MPPI: all other
planners, environments, the cost function and the model are bit-identical.

---

## Phase 0 — Safety net (do this first, nothing else until it's green)

Refactoring a research codebase without a golden baseline destroys results silently.

> **Tooling:** this project uses **uv exclusively**. No `pip`, no `python -m venv`,
> no `requirements.txt`. Dependencies live in `pyproject.toml`, are resolved by
> `uv lock`, installed by `uv sync`, and everything runs through `uv run`.

- [ ] **0.1 Resolve the paper↔code divergences.** *(DEFERRED — pending team confirmation.)*
      For each, decide: was the paper generated with a different value, or is the paper wrong?
  - `N_SIMULATIONS = 3` vs paper's $N_{sim}=3000$ — `Parameters/__init__.py:44`.
    Commit `b017895` ("changed mcts parameters for faster execution") is the culprit.
  - `NUM_STEPS = 10` vs paper's $N_{eval}=30$ — `Parameters/__init__.py:13`.
  - **Env I OOD ranges are identical to train ranges** — `Parameters/__init__.py:107-115`
    vs `122-130`. The paper lists distinct OOD ranges. This invalidates the Env I OOD
    claim unless the ranges are restored.
  - Record the resolution in `docs/paper_config.md` alongside the exact `Parameters`
    values used for each published figure/table.
- [x] **0.2 Make the repo runnable from a clean clone.** ✅ **DONE**
  - [x] `pyproject.toml` dependencies now match what is actually imported:
        `torch`, `numpy`, `gymnasium`, `networkx`, `matplotlib`, `tensorboard`.
        Dropped `torch-geometric` (imported nowhere).
  - [x] `gym` → `gymnasium` (`Environment_I.py:2`, `Environment_II.py:2`). Safe one-line
        change: `gym.Env` was a decorative base class only — `action_space` is a plain
        Python list and no gym space, wrapper, or `gym.make` is used anywhere.
        *(The envs' `reset()`/`step()` still use the legacy 4-tuple signature rather than
        Gymnasium's 5-tuple. Harmless at runtime since nothing consumes the gym API;
        conform them in Phase 2.1.)*
  - [x] Python floor raised to **3.12** (`.python-version`, `requires-python`), and the
        PEP 701 nested-quote f-string at `Parameters/__init__.py:57` rewritten to plain
        syntax anyway so the module no longer depends on that feature.
  - [x] Project renamed `causal-rl` → `cdd-oran`; `uv.lock` regenerated.
  - [x] Deleted `requirements.txt` — a second, inconsistent dependency source
        (listed `gym`/`matplotlib`/`tensorboard`, which `pyproject.toml` omitted).
  - [x] Fixed the clean-clone crash: `main.py:85` loaded a checkpoint unconditionally
        (because `IS_TRAIN` is shadowed at line 15 while `IS_TEST` is derived from the
        *Parameters* value), so a fresh clone died with `FileNotFoundError`. Now resumes
        if a checkpoint exists, else trains from scratch. The flag incoherence itself is
        Phase 1.2.
  - [x] Fixed `Tests/EnvironmentII_mean_std.py:50` — missing required `seed` argument
        made the module crash when run standalone.
  - **Deferred:** making the project an installable distribution. Adding a
    `[build-system]` and enumerating the six legacy top-level packages
    (`Environment/`, `Models/`, `Algorithms/`, `Policies/`, `Parameters/`, `Tests/`)
    is throwaway work that Phase 1 deletes. `[tool.uv] package = false` is set
    explicitly; flip to `true` when the `cdd_oran/` layout lands.

  **Verified:** both environments construct and step; both models train; all four
  planners build; short train (CMI + MLP) → evaluate pipeline completes with exit 0;
  `detect_conflict_edges` on the ground-truth graph yields **8 conflict edges for
  Env I**, matching the paper. Ground-truth adjacency edge counts also match the paper
  (Env I: 10 = 8 NCP→KPI + 2 implicit; Env II: 16).

  **Observed while verifying:** `import Parameters` took ~37 s on the very first run.
  **This was misattributed to the Monte Carlo — it is cold `torch` disk load.** Measured
  directly: the 1M-sample estimation costs **0.13 s (Env I) + 0.21 s (Env II)**. Warm
  `import Parameters` is ~1.8 s, essentially all torch. Phase 1.3 is therefore a
  correctness/architecture cleanup, not a performance win — re-ranked accordingly.
- [ ] **0.3 Regenerate and archive the paper's checkpoints.** No `.pt` files exist in the
      repo. Train Env I (20k) and Env II (50k) for both CMI and MLP, and store the four
      checkpoints + their exact configs somewhere durable (release asset / Drive / DVC —
      not git).
- [x] **0.4 Write a characterisation test.** ✅ **DONE** — `tests/`, 24 tests, ~75 s.
      Does **not** depend on the paper checkpoints (0.3): uses the env's *ground-truth*
      adjacency in place of a learned graph (8 edges Env I, 16 Env II) and a fixed-seed
      untrained world model. Covers both envs × both model kinds × 8/16 conflict edges ×
      4 planners = 192 planner results, plus continuous probes of the cost function and
      the model forward pass.
  - Planner hyperparameters are passed explicitly (not read from `Parameters`), so the
    goldens survive Phase 1's config rewrite.
  - Re-seeds before every `act()`, so results don't depend on planner ordering.
  - **Mutation-tested.** Forcing QACM's normalised threshold to a constant fails all 4
    fixtures. A *subtle* 1.05× scaling initially **survived** — planners argmin over a
    discrete grid, so small cost drift changes no action. Added `cost_probe` /
    `model_probe` continuous assertions; the 1.05× mutation now fails.
  - Blocked on nothing. Rename `Tests/` → `env_statistics/` was required first: on
    Windows the case-insensitive filesystem collides `Tests/` with `tests/`.

  **Every subsequent phase must keep `uv run pytest` green.**

- [x] **0.5 Purge committed artifacts.** ✅ **DONE** — `git rm -r --cached runs/`
      untracked all 63 TensorBoard event files (`.gitignore` already had `runs/*`; they
      predated the rule). Files remain on disk and in git history, so nothing is lost.

- [x] **0.6 Introduce `ruff` + baseline format.** ✅ **DONE** *(pulled forward from
      Phase 6.3 — formatting before the big refactors keeps their diffs readable.)*
  - [x] `ruff` added as a uv dev dependency (`[dependency-groups] dev`).
  - [x] `[tool.ruff]` config in `pyproject.toml`: `line-length = 100`,
        `target-version = "py312"`, lint rules `E,F,W,I,UP,B`.
  - [x] `uv run ruff format .` — **22 files reformatted, 2 unchanged.**
  - **Deliberately NOT run: `ruff check --fix`.** The codebase depends on
    `from Parameters import *` and on re-exporting `__init__.py` modules; autofixing
    "unused" imports (F401/F403) would break them. `F403`/`F405` are in the `ignore`
    list until Phase 1.4 removes the star-imports. Lint is **report-only** for now.

  **Lint baseline (56 findings, nothing fixed)** — the Phase 6.3 worklist:

  | Rule | N | What |
  |---|---|---|
  | `I001` | 16 | unsorted imports |
  | `F401` | 11 | unused imports (matches the audit: `numpy` in `CDL.py`/`MLP.py`, 3× matplotlib in `evaluate_algorithms.py`, `Callable` in `model_based.py`) |
  | `UP006`/`UP035` | 15 | `typing.List`/`Tuple`/`Callable` → builtin generics |
  | `B905` | 5 | `zip()` without `strict=` |
  | `F841` | 5 | unused locals |
  | `B007` | 2 | unused loop variables |
  | `F811`, `F541` | 2 | `np` re-imported in `CDL.py`; empty f-string |

  Ruff independently confirmed several audit findings — notably
  `Algorithms/model_based_mppi.py:71` `greedy_kpis` assigned and never used (the dead
  warm-start block, Phase 3.4), and `evaluate_algorithms.py:344` `algo_idx` unused,
  which is precisely the latent Phase 3.3 style-index bug: the index is unused *only*
  because `draw_panel` is dead.

  **Verified post-format:** smoke test and full train→evaluate pipeline both still pass;
  state dims, ground-truth edge counts (10 / 16) and conflict-edge count (8) unchanged.

  > Suggest committing the reformat as its own commit and adding it to
  > `.git-blame-ignore-revs` so it doesn't pollute `git blame`.

---

## Phase 1 — Configuration (highest leverage; unblocks everything downstream)

Replace `Parameters/__init__.py` with explicit, injectable, side-effect-free config.

- [ ] **1.1 Dataclass config.** `cdd_oran/config.py`:
      ```python
      @dataclass(frozen=True)
      class ModelConfig:   lr, batch_size, cmi_threshold, eval_tau, grad_clip, fc_dims...
      @dataclass(frozen=True)
      class TrainConfig:   total_steps, init_steps, eval_steps, plot_freq...
      @dataclass(frozen=True)
      class PlannerConfig: cem: CEMConfig; mppi: MPPIConfig; mcts: MCTSConfig
      @dataclass(frozen=True)
      class ExperimentConfig: seed, env, model_kind, train, model, planner, paths
      ```
      Loaded from YAML in `configs/` (`env_i_cdl.yaml`, `env_i_mlp.yaml`, `env_ii_*.yaml`)
      with CLI overrides. No module-level mutable state, no `datetime.now()` at import,
      no prints.
- [ ] **1.2 Kill the boolean-flag maze.** `IS_TRAIN` / `IS_TEST` / `USE_CMI` / `USE_MLP`
      are four booleans encoding two decisions, with `main.py:15` shadowing `IS_TRAIN`
      after the star-import (so `main.py` trains *and* takes the `IS_TEST` branch that
      loads a checkpoint — `main.py:85`). Replace with:
      `model_kind: Literal["cdl", "mlp"]` and `param_ranges: Literal["train", "ood"]`,
      and let the subcommand (`train` vs `evaluate`) carry the mode.
- [x] **1.3 Move the Monte-Carlo statistics off the import path.** ✅ **DONE**
      (delegated to opencode/terra). Lazy via PEP 562 `Parameters.__getattr__`, disk-cached
      in `.cache/stats/<sha256>.json` keyed by (environment, ranges, seed, num_samples);
      import-time prints removed. Values verified **bit-identical** to pre-refactor.
      `Tests/` → `env_statistics/` rename happened earlier in 0.4.
      **Note:** the original justification (a 37 s import) was wrong — see 0.2. Actual
      saving is ~0.34 s. Kept because lazy+cached is the right shape for 1.1, and it
      removes the import-time prints, not because it is fast.
- [x] **1.1 / 1.2 / 1.4** ✅ **DONE** (delegated to opencode/terra). `config.py` with frozen
      dataclasses + `load_config`; four YAMLs in `configs/` reproducing today's values;
      `IS_TRAIN`/`IS_TEST`/`USE_CMI`/`USE_MLP` replaced by `model_kind` +
      `param_ranges`, `main.py`'s shadowed flag gone (resume is now an explicit arg);
      all 7 star-imports removed; factories take `cfg`; model and planner
      hyperparameters are required args instead of globals-as-defaults.
      Goldens unmodified, 24 tests green.

  ⚠️ **Phase 1's headline benefit is NOT yet achieved.** The environments still read
  ranges and stats from `Parameters`, which derives from a module-level `DEFAULT_CONFIG`
  (hardcoded to `env_i_mlp.yaml`) — *not* from the `cfg` passed to `get_env(cfg)`. So
  `cfg.param_ranges` and `cfg.seed` are ignored by the environment, and two configs still
  cannot coexist in one process. This was a deliberate scope limit (envs are Phase 2),
  but it made `param_ranges: train` silently yield **ood** ranges. `get_env()` now raises
  `NotImplementedError` rather than returning a wrong env; **delete that guard in 2.4/2.7
  once envs take their ranges and stats from `cfg`.** Multi-seed sweeps (5.4) stay blocked
  until then.

- [ ] ~~**1.4 Delete `from Parameters import *` everywhere.**~~ Every factory
      (`Environment/__init__.py:5`, `Models/__init__.py:10`, `Algorithms/__init__.py:6`)
      takes an explicit config argument instead of reading globals. This is what makes
      the code testable — it is the precondition for Phase 6.

---

## Phase 2 — Environment layer

`Environment_I.py` (346) and `Environment_II.py` (527) independently redefine `XApp`,
`Param`, `KPI`, `reset`, `step`, `_get_state`, `action_to_param`, `get_utility_fns`,
`compute_utility_value`, `RewardFn` — with essentially identical bodies.

- [ ] **2.1 Extract `cdd_oran/envs/base.py`.** `BaseORANEnv(gym.Env)` holds all shared
      mechanics. A concrete environment then declares only its *specification*:
      KPI equations, parameter ranges, thresholds, xApp→KPI map, ground-truth adjacency.
      Target: each env file drops to ~120 lines of pure specification.
- [x] **2.1 / 2.3 / 2.5 / 2.6 / 2.7** ✅ **DONE** (delegated to opencode/terra).
      `Environment/base.py` (184 lines) holds all shared mechanics; the concrete envs are
      now pure specification — **Env I 370 → 52 lines, Env II 582 → 89**. Duplicated
      `set_param1..8`, the unreachable `reward()` block, dead methods and the ~19
      commented-out KPI variants are gone. `KPI_THRESHOLDS` / `MEAN_STD_KPIS` are instance
      state taken from `cfg`. **`Parameters/` deleted entirely.** `_check_env_globals`
      removed — `cfg.param_ranges="train"` now correctly yields Env II `(-100, 100)`.
      Goldens unmodified, 24 tests green. **The Phase 1 multi-config gap is closed.**

- [ ] **2.2 Single source of truth for the KPI physics.** *(Deliberately left undone —
      terra was instructed to stop if the implementations differed, and they do.)*

  **Finding: Env I's runtime and estimator use different variance floors.**
  - Runtime `Environment_I.py`: `max(P, 1e-3)` — floors all small values.
  - Estimator `env_statistics/EnvironmentI_mean_std.py`: `np.where(x > 0, x, 1e-3)` —
    only replaces **non-positive** values, so `P=0.0005` stays `0.0005`.
  - At `P1=-49.99, P2=0.0005` this gives `6.94e-12` vs `1.86e-44`.

  **Quantified impact: immaterial.** 1,019 of 4,000,000 samples disagree (0.0255%), and
  the resulting K1/K2 mean and std are **identical to 10 decimal places** — in the
  disagreeing region both forms drive the KPI to ~0. The published statistics are not
  affected.

  **Env II is already consistent** (`np.where(np.abs(x) > 1e-1, x, 1e-1)` is the vectorised
  form of the env's scalar `safe_exp`). So only Env I needs work, and only for hygiene.

  Remaining work is not a simple `safe_exp` swap: the Env I estimator uses a different
  *structure* (a chunked grid sweep for K3/K4, which depend on K1/K2). Unify carefully, or
  fold into 7.x when the env spec is finalised. The equations currently live
      twice — in the env (`Environment_II.py:145-193`) and in the statistics estimator
      (`Tests/EnvironmentII_mean_std.py:20-25`) — with *three* different `safe_exp`
      implementations. Any edit to one silently invalidates every normalised utility in
      the paper. The statistics estimator must call the env's own KPI functions.
- [ ] **2.3 Collapse `set_param1..8`** (`Environment_II.py:68-114`) — eight byte-identical
      3-line `np.clip` functions. Env I already solved this (`set_param_default`).
- [ ] **2.4 Inject randomness.** `Param.__init__` calls `np.random.uniform`
      (`Environment_I.py:54`), `__init__` calls `reset()`, and `_get_state` adds
      `np.random.normal(0, 0.01)` unconditionally (`Environment_I.py:296`). Give each env
      its own `np.random.Generator` seeded at construction, and make the observation noise
      a config field (`obs_noise_std`, settable to 0 for deterministic tests).
- [ ] **2.5 Promote `_get_state` to public `get_state()`.** `evaluate_algorithms.py:281`
      already reaches across the boundary. Delete the dead public API that nothing calls
      (`get_save_information`, `observation_spec`, `observation_dims`, `KPI.get_kpi`).
- [ ] **2.6 Delete the unreachable reward block** `Environment_II.py:371-390` (after the
      unconditional `return` at line 369), plus the now-dead `self.weights` / `self.zeta`.
- [ ] **2.7 Move `KPI_THRESHOLDS` / `MEAN_STD_KPIS` from module globals to instance state**
      (`Environment_I.py:6-7`, `Environment_II.py:6-7`). Currently two environments with
      different thresholds cannot coexist in one process — which blocks sweeps.

---

## Phase 3 — Planner layer

- [x] **3.1–3.6** ✅ **DONE** (delegated to opencode/terra, with one intervention).
      `Algorithms/cost.py` now holds the only `weighted_distance` / `score_batch`
      (was 5 and 2 copies); `Algorithms/base.py` adds a `Planner` ABC all four inherit;
      `Policies/Policy.py` and the unreachable `Policies/model_based.py` (230 lines)
      deleted along with `oran_reward`, the `mb_policy` branch and `model_based_start`;
      plot styles keyed by planner **name** not index (3.3 latent bug fixed);
      `# FIX N` changelog comments stripped. Lint 56 → 44 findings.
      Surviving cost implementation uses `xapp.mean`/`xapp.std` — the four-copy semantics.

  **Goldens were regenerated once here — the only time so far, and deliberately.**
  Deleting MPPI's dead warm-start (3.4) removed a `greedy_dist.sample()` call, which
  *consumed torch RNG*. Removing it shifts the subsequent `torch.randn` draw, so MPPI's
  samples legitimately change. Terra initially papered over this by inserting a dummy
  `torch.normal` purely to keep the RNG stream aligned — that was rejected: it encodes an
  accident of deleted code as a permanent requirement and would break mysteriously when a
  future maintainer removed it.

  Blast radius was measured before regenerating: `state`, `cost_probe` and `model_probe`
  are **byte-identical** (asserted), and exactly **27 of 48 `ModelBasedMPPI` results**
  changed — no other planner, no env, no cost, no model. Golden key-level diff shows only
  `raw_val` (plus bare `action`/`utilities` array elements).

  **Consequence for the paper:** MPPI numbers from this codebase will differ slightly from
  the conference run. The old behaviour was an unintended RNG side effect of dead code, not
  a designed choice — but Table `tab:mitigation`'s MPPI column must be regenerated, not
  copied, once 0.1/0.3 settle.

- [ ] ~~**3.1 One cost function.**~~ Create `cdd_oran/planners/cost.py` with a single
      `weighted_distance(...)` and `score_batch(...)`. Deletes 5 copies
      (`QACM.py:18`, `cem.py:115`, `mppi.py:134`, `mcts.py:167`, `Policies/model_based.py:209`)
      and 2 copies of `_score_batch`.
      **Note the divergence:** `Policies/model_based.py:221` indexes
      `env.kpis[xapp_idx].mean/.std` while the other four use `xapp.mean/.std`. In Env II
      the xApp→KPI map is not 1:1 (xApp 4 aggregates $K_{41},K_{42}$), so one of these is
      wrong. Decide which, and document it — the cost function $\mathcal{F}$ is the paper's
      Eq. for planning, so this is a correctness question, not a style one.
- [ ] **3.2 `Planner` ABC** with the existing `act(...)` signature, replacing the empty
      `Policies/Policy.py` stub that nothing inherits from. Registry maps name → class.
- [ ] **3.3 Fix the plotting index bug before re-enabling figures.**
      `Algorithms/__init__.py:11` returns `[QACM, CEM, MPPI, MCTS]` but
      `evaluate_algorithms.py:39-44` `ALGO_STYLES` is ordered `[QACM, MCTS, MPPI, CEM]`,
      index-matched at line 96. Currently latent only because `draw_panel` is dead;
      three of four planners get the wrong colour/marker/label the moment it's called.
      Key styles by planner *name*, not index.
- [ ] **3.4 Delete the dead warm-start** `model_based_mppi.py:68-71` — a full model
      forward pass over the batch whose result is never used (and whose slice
      `greedy_state[:, self.num_params:]` yields an empty tensor anyway).
- [ ] **3.5 Decide the fate of `Policies/model_based.py`** (230 LOC). It is unreachable
      (`MODEL_BASED_START == TOTAL_STEPS == 50000`) *and* would `TypeError` if reached
      (`main.py:147` passes `reward_fn=`, which `__init__` doesn't accept). Either wire it
      up properly as the exploration policy — it's the natural home for the journal's
      $H>1$ multi-step rollout work — or delete it. Do not leave it as-is.
- [ ] **3.6 Strip the `# FIX N` comment changelog** — ~30 sites across the four planners.
      That's what git log is for.

---

## Phase 4 — Model layer

- [x] **4.1 / 4.2 / 4.3 / 4.5** ✅ **DONE** (delegated to opencode/terra, clean run).
      `Models/base.py` defines `WorldModel` + a separate `CausalModel` ABC; `MLPInference`
      no longer claims graph capability (the three `pass` stubs returning `None` are gone),
      and call sites decide on `cfg.model_kind` explicitly. Visualisation moved to `viz/`
      (`graph.py`, `heatmap.py`) — **no matplotlib import remains under `Models/`** — and
      the blocking interactive slider is split from a plain `threshold=` parameter, so the
      analysis path no longer requires a human at a GUI. `evaluate_algorithms.py` now
      builds its CDL via the factory using `replace(cfg, model_kind="cdl")` instead of a
      duplicated arg list that omitted `node_names`. Lint 44 → 33. Goldens untouched.

- [ ] ~~**4.1 `WorldModel` Protocol**~~ in `cdd_oran/models/base.py`:
      `train_step`, `predict_next_state`, `evaluate_predictions`, `save`, `load`,
      and an *optional* `CausalModel` sub-protocol for `update_mask` / `get_binary_graph`.
- [ ] **4.2 Fix `MLPInference`'s silent stubs.** `Models/MLP.py:102-109` — `update_mask`,
      `get_causal_graph`, `get_binary_graph` are `pass` and return `None`, so callers get
      `TypeError: 'NoneType' is not subscriptable` far from the cause. Raise
      `NotImplementedError`, or better, remove them from the type entirely (per 4.1) and
      let the MLP path not claim to be a causal model.
- [ ] **4.3 Separate visualisation from the model.** `Models/CDL.py` imports pyplot at
      module level (line 8) for two methods, and `visualize_cmi_heatmap` **blocks on
      `plt.show()` with an interactive slider and returns the human-chosen threshold**
      (line 395), which `visualize.py:19-20` feeds into the graph plot. A human at a GUI
      is currently a required input to the analysis path. Move to `cdd_oran/viz/`, make
      the threshold a parameter, and add a non-interactive path for scripted figures.
- [ ] **4.4 Fix checkpoint naming.** `Parameters/__init__.py:57` —
      `MODEL_LOAD_NAME = MODEL_SAVE_NAME` are the same object, so every run overwrites the
      same `.pt` and you can never load A while saving B. Checkpoints belong under the
      run directory (Phase 5), not CWD.
- [ ] **4.5 Deduplicate model construction.** `evaluate_algorithms.py:239-250` hand-builds
      a `CDL` duplicating `Models/__init__.py:15-27`, omitting `node_names` (so any
      visualisation on that instance `TypeError`s at `CDL.py:290`). Use the factory.

---

## Phase 5 — Experiment runner and reproducibility

This is what turns "run the script and hope" into an experimental instrument.

- [x] **5.1 / 5.2 / 5.3 / 5.5 / 5.7 + 4.4** ✅ **DONE** (delegated to opencode/terra).
      `cli.py` with `train` / `evaluate` / `viz`; `--set` dotted overrides
      (`--set train.total_steps=60`); `utils/{seeding,logging,runs}.py`. Run dirs are
      `runs/<env>/<model_kind>/<timestamp>-<confighash>/` holding `checkpoint.pt`,
      **fully resolved** `config.yaml` (overrides applied, `device: auto` → `cpu`),
      `tensorboard/`, `metrics.json`, `git_sha.txt` (+ dirty flag). Checkpoints no longer
      overwrite a fixed CWD filename. `seed_everything` covers `random` (previously never
      seeded despite driving the train/test split), numpy, torch, cuda;
      `use_deterministic_algorithms` is **opt-in** via `deterministic: false`.
      Prints → `logging`; ANSI escapes gone. Legacy scripts forward to the CLI.
      Goldens untouched, 24 green, lint steady at 33.

  **Verified by driving the CLI directly**, not from the agent's report: short `train`
  produced all five artifacts; the snapshot correctly recorded `total_steps: 60`;
  `evaluate` on that run exits 0; a missing run exits 1. **MLP-mode graph dependency**
  handled via a required `--graph-run` — without it, `evaluate` fails with
  *"MLP evaluation requires --graph-run pointing to a trained CDL run"* rather than
  silently using a wrong graph. Good design.

  Minor wart: a nonexistent `--run` path is resolved relative to `configs/`, so the error
  reads `configs/runs/does-not-exist/config.yaml`. Exit code is correct; message is
  misleading. Also `--set` exists only on `train`, not `evaluate`.

- [ ] ~~**5.1 Real CLI.**~~ Single entry point:
      ```
      cdd-oran train    --config configs/env_i_cdl.yaml [--seed 45] [--override k=v]
      cdd-oran evaluate --run runs/<id> [--planners cem,mppi]
      cdd-oran sweep    --config configs/sweep_seeds.yaml
      cdd-oran viz      --run runs/<id> --figure causal-graph
      ```
      Replaces "edit `Parameters/__init__.py` and run one of three argument-free scripts".
- [ ] **5.2 Self-describing run directories.**
      `runs/<env>/<model>/<timestamp>-<confighash>/` containing `config.yaml` (full
      resolved snapshot), `checkpoint.pt`, `tensorboard/`, `metrics.json`, `git_sha.txt`.
      Today the 63 committed runs record only scalars — the `Parameters` values that
      produced them are unrecoverable except by reading `git log`. That is exactly the
      situation that produced the Phase 0.1 problem.
- [ ] **5.3 Centralised seeding.** `cdd_oran/utils/seeding.py::seed_everything(seed)`
      covering `random`, `numpy`, `torch`, `torch.cuda`, plus a `deterministic: bool`
      config flag. Today: `random` (stdlib) is **never seeded** yet `main.py:106` uses
      `random.random() > 0.2` for the train/test buffer split — every run gets a different
      partition. `main.py` seeds at import, `evaluate_algorithms.py` seeds inside `main()`,
      `visualize.py` never seeds.
- [x] **5.4 Multi-seed sweep + statistics.** ✅ **DONE** (delegated, with two fixes by me).
      `cli.py sweep --seeds 0,1,2` retrains per seed (varies the **discovery** seed);
      `cli.py aggregate --sweep <tag>` emits `summary.json` + `table.tex` shaped like
      `tab:mitigation`. Statistics verified **between-seed**: values keyed by seed, sample
      SD (n−1), pooled Cohen's $d$ on per-seed values, and `n` = seed count. It refuses to
      emit $d$ unless both arms have n > 1 rather than printing a garbage number.

  **Two problems found and fixed:**
  1. The first delivery produced a correctly-shaped but **entirely empty** table (`n = 0`)
     and was reported as complete. Cause: the 60-step smoke budget makes the CDL learn an
     empty graph → zero conflict edges → no utilities. **The flawed premise was mine** —
     "meaningless numbers are fine, we're validating machinery" is wrong when an empty
     graph means the machinery never runs. Wiring confirmed correct once a real model
     (F1 = 0.947) was trained: utilities populated
     `{CEM 0.306, MPPI 0.293, MCTS −0.067, QACM −0.037}`.
  2. CI used the normal approximation (1.96) — far too narrow at 3–5 seeds, where
     t(2) = 4.30. Replaced with a t-critical lookup. On the n = 2 fixture the interval
     correctly widened from `[0.043, 0.760]` to `[−1.924, 2.727]`.

  ⚠️ **Not fully validated:** the Cohen's $d$ path has never executed, because the CDL arm
  only ever had n = 1 (the 25-min training cap). The formula was reviewed by hand and is
  correct, but **exercise it on the first real sweep** before trusting the number.
- [ ] **5.5 Replace 40+ `print()` with `logging`.** Including the ANSI escapes hardcoded
      in `visualize.py:8,12` that corrupt piped output.
- [ ] **5.6 Fix the evaluation loop's two real bugs:**
  - `evaluate_algorithms.py:354` steps the env with `np.zeros(act_dim)` — the chosen
    actions are never applied, so all `NUM_STEPS` utilities are measured along a fixed
    trajectory. Confirm this is intentional (it is defensible as "independent per-step
    conflict resolution", matching the paper's framing) and **document it**, or fix it.
  - `evaluate_algorithms.py:310-314` builds `weights`, multiplies by `1.0`, then
    renormalises to all-ones. A no-op that hides the fact that per-xApp weights $w_x$
    — a symbol in the paper's cost function — are effectively unused.
- [ ] **5.7 Exit codes.** `evaluate_algorithms.py` returns `-1` on missing checkpoint but
      `__main__` discards it, so failures exit 0. Wrap in `sys.exit()`.

---

## Phase 6 — Tests, typing, CI

- [ ] **6.1 `pytest` suite.**
  - `test_envs.py` — KPI equations against hand-computed values; determinism under fixed
    seed; `action_to_param` round-trips; ground-truth adjacency matches the declared
    xApp→KPI map.
  - `test_cost.py` — the unified `weighted_distance` against analytically known cases;
    settles the Phase 3.1 divergence.
  - `test_conflicts.py` — `detect_conflict_edges` on hand-built adjacency matrices
    (direct / indirect / implicit).
  - `test_planners.py` — each planner returns a valid in-bounds action; on a trivial
    mock world model with a known optimum, each finds it.
  - `test_cdl.py` — CMI shape/threshold behaviour; a short training run on a 3-variable
    toy system recovers the true graph.
  - `test_regression.py` — the Phase 0.4 golden lock.
- [x] **6.2 / 6.3 / 6.5** ✅ **DONE** (delegated to opencode/**Luna**, clean run).
      `ruff check .` **33 → 0 findings**; `F403`/`F405` removed from the ignore list now
      that star-imports are gone. `mypy` added as a dev dependency with a **ratchet**
      config — strict globally (`disallow_untyped_defs`, `disallow_any_generics`,
      `check_untyped_defs`, `no_implicit_optional`), with a per-module relaxed list for the
      legacy planners; core modules (`config.py`, `utils/`, `Algorithms/base.py`,
      `Algorithms/cost.py`, `Models/base.py`, `Environment/base.py`) fully annotated.
      **`mypy .` passes on 39 files.** Stale `.gitignore` entries removed, README updated
      to the `cli.py` workflow, `__pycache__/` cleared.

  Only **three** suppressions repo-wide, all narrow and justified: two
  `# type: ignore[override]` in `Environment/base.py` for the gymnasium `reset`/`step`
  signature mismatch (the legacy 4-tuple issue from 0.2), and one `# noqa: E402` for an
  import after `sys.path` manipulation. No blanket sprays.

  **Nothing was deleted.** Luna correctly preserved everything on the do-not-delete list —
  `draw_panel` and the plotting helpers (paper figure generators, which any dead-code tool
  flags), all of `runs/`, the golden fixtures, the README image. It removed only untracked
  `__pycache__` and stale ignore entries.

- [ ] ~~**6.2 Type annotations + `mypy`/`pyright`.**~~ Start with `strict` on new modules and
      a per-module ignore list for legacy, ratcheting down. Fix `Policies/model_based.py:23`
      (`weights_per_xapps: list = None` → `Optional[list]`) and the `Any`-typed constructor.
- [ ] **6.3 `ruff` (lint + format) + `pre-commit`.** Will immediately catch the unused
      imports: `numpy` in `Models/CDL.py:5` and `Models/MLP.py:5` (zero `np.` uses),
      three matplotlib imports in `evaluate_algorithms.py:18-20`, `Callable` in
      `Policies/model_based.py:3`.
- [ ] **6.4 GitHub Actions.** lint → typecheck → pytest (excluding the slow regression
      job, which runs on a nightly/manual trigger).
- [ ] **6.5 Dead-code sweep** with `vulture`/`ruff`, then delete:
      `evaluate_algorithms.py:31-74,95-127,173-227` (`draw_panel` and its exclusive
      dependencies — 55 + ~60 lines, never called), `Policies/Policy.py`,
      `Environment_II.py:397-398`, `min_bin_length`, the 19 commented-out code sites in
      `Environment_II.py`, `Tests/EnvironmentI_mean_std.py:6` (`CHUNK`, unused, with a
      docstring describing a chunked sweep the code doesn't do).

---

## Phase 7 — Journal-extension enablers

Only after Phases 0–6. These are the paper's own stated future work, and the refactor
above is what makes each of them a contained change rather than a rewrite.

- [ ] **7.1 Multi-step planning $H>1$.** Requires the world model to expose clean batched
      rollout (Phase 4.1) and planners to share a rollout utility (Phase 3.1).
      `Policies/model_based.py::_rollout` is the existing prototype.
- [ ] **7.2 Joint multi-conflict optimisation.** Changes the action space from one NCP to
      a set. Needs the `Planner` ABC (3.2) so it's a new planner, not a fork of four.
- [ ] **7.3 Online / streaming CDL** under non-stationary xApp populations. Needs the
      frozen-graph assumption to become a config flag rather than a hardcoded workflow.
- [ ] **7.4 Multi-seed statistics with CIs** (5.4) — the conclusion already concedes this
      is missing.
- [ ] **7.5 Per-planner runtime/latency benchmarking.** Absent from the conference paper;
      a near-RT RIC venue will ask, given the 10 ms–1 s control loop budget.
- [ ] **7.6 Numeric table of xApp targets $T_x$, weights $w_x$, directions $\delta_x$, $\beta$.**
      Currently symbolic in the paper and buried in code — a reproducibility gap a
      reviewer will flag. (See also 5.6: $w_x$ is currently a no-op.)
- [ ] **7.7 Third environment / real O-RAN trace ingestion.** The `BaseORANEnv` from
      Phase 2.1 is precisely what makes this cheap.
- [ ] **7.8 Contemporaneous causal edges** (relaxing assumption A4). Changes the CMI test
      and adjacency semantics — the deepest change, do last.

---

## All figures save to disk ✅ DONE (commit `a33257b`)

`cli viz --run <dir>` now defaults to `--figure all` and writes every applicable figure to
the run directory in one command.

This closed a real gap: only `panels` had been made headless. `visualize_causal_graph` and
`visualize_cmi_heatmap` still called `plt.show()` with **no `savefig`** — they popped a
blocking GUI window and left nothing on disk, so they were unusable headless, in CI, or in
any scripted figure regeneration. *(An earlier claim that Phase 4.3 had removed the GUI
dependency was only half true: the threshold became a parameter, but rendering still only
displayed. Checking for the parameter and not for `savefig` is what missed it.)*

Now all three `savefig`, with `plt.show()` only under `--show`. `select_cmi_threshold`
deliberately still blocks — it is the opt-in interactive slider and that is its purpose.

Inapplicable figures skip with a reason instead of failing the command; exit is non-zero
only if nothing at all could be produced.

Verified by running it, not from a report:

```
CDL run  -> causal-graph.png (373,281)  cmi-heatmap.png (97,478)  panels.png (1,351,924)
            all valid PNGs (magic 89504e47), exit 0
MLP run  -> "Skipped causal-graph: run model_kind is mlp; requires a CDL model"
            "Skipped cmi-heatmap: ..."  panels.png written, exit 0
```

## Figures, variance floor, typing ✅ DONE (commit `fb1ac5d`, delegated to Luna)

**Paper figures regenerate again.** `viz/panels.py` was orphaned — nothing called it — and
the data it needs (per-step, per-planner utilities) was only ever logged to TensorBoard.
`evaluate` now persists `<run>/utilities.json`, and `cli viz --figure panels --run <dir>`
renders headless (no blocking `plt.show()`). Verified end to end: a real 1,351,924-byte
PNG with a valid header, from a real run directory.

**2.2 closed — Env I variance floor aligned.** The estimator now uses the runtime's
`max(P, 1e-3)` instead of `where(x > 0, x, 1e-3)`. The statistics cache keys on
`(environment, ranges, seed, num_samples)` and **not on the code**, so changing the formula
without invalidating would have silently returned stale values and looked successful — the
cache key is now versioned (`environment-i-max-floor-v2`).

**Goldens regenerated — the second deliberate regeneration.** My prediction that the change
would stay inside `rel=1e-5` was **wrong**: utilities are small (~0.019), so the relative
tolerance is a tight ±1.9e-7 window and the shift exceeded it. Luna hit the failure and
**reported it rather than regenerating** — the correct protocol.

Blast radius measured before regenerating:

| Fixture | max abs delta | planner actions changed |
|---|---|---|
| EnvironmentI-CDL | 1.62e-06 | **0** |
| EnvironmentI-MLP | 1.62e-06 | **0** |
| EnvironmentII-CDL | **0** (bit-identical) | 0 |
| EnvironmentII-MLP | **0** (bit-identical) | 0 |

Env II is untouched, confirming it was already consistent. In Env I nothing but the 6th
decimal moves and **not one planner decision changes**.

**Paper impact: none.** `tab:mitigation` reports to 3 decimals; a 1.6e-6 shift cannot move
a published figure.

**Typing:** `Any` replaced in `planners/cost.py` (`xapp` now properly typed), `envs/__init__.py`,
`envs/stats_cache.py` and the base setters. Deliberately left on the dynamic JSON/aggregation
structures in `utils/sweeps.py` and the legacy Gym surfaces — a wrong annotation is worse
than `Any`.

## Type checking: mypy → ty ✅ DONE (commit `c950bbe`, delegated to Luna)

`ty` replaces mypy as the project's type checker — the maintainer uses it in their IDE, so
CI and editor now agree. Pinned as a dev dependency (not `uvx`) so local and CI match.
**Zero per-module relaxations**, versus the relaxed legacy-planner list mypy needed.

`ty` found 28 diagnostics mypy's config had hidden. Two were real design defects:

1. **`CausalModel` ABC was incomplete.** `viz/graph.py` and `viz/heatmap.py` depend on
   `kpi_start`, `node_names` and `cmi_threshold`, none of which the interface declared —
   so implementing `CausalModel` from the ABC alone yields an object the viz layer crashes
   on. Attributes now declared.
2. **Possible `None` attribute access** in `experiments/evaluate.py`. Investigated:
   an annotation/control-flow gap, **not a reachable runtime bug**. Narrowed properly.

The rest were loose `TypedDict`-able test-harness kwargs and the known gym-legacy override
mismatch. Only **two** suppressions added, both narrow with stated reasons:
`# ty: ignore[invalid-method-override] -- Legacy Gym 4-tuple API is intentional.` on
`envs/base.py` `reset`/`step`. Changing those signatures would be a behaviour change, so
suppression is the honest option.

Verified: `ty check` clean, 24 tests green, `tests/golden/` unmodified, ruff clean.

**Follow-up (not blocking):** several `Any` annotations survive from the earlier mypy pass —
notably `xapp: Any` in `planners/cost.py`, which is core code where `XApp` is a known type.
These predate this migration (`cost.py` was untouched by it) and are worth tightening now
that relaxations are gone.

## Package restructure ✅ DONE (commit `1754651`, delegated to Luna)

Everything now lives under one PEP 8 package. Old top-level packages (`Algorithms/`,
`Environment/`, `Models/`, `Policies/`, `env_statistics/`, `utils/`, `viz/`) are gone, as
are the legacy root scripts (`main.py`, `evaluate_algorithms.py`, `visualize.py`) — the CLI
supersedes them. A thin root `cli.py` delegates to `cdd_oran.cli`.

`evaluate_algorithms.py` was split three ways: library helpers → `cdd_oran/conflicts.py`,
the evaluation loop → `cdd_oran/experiments/evaluate.py`, and the paper's plotting code →
`cdd_oran/viz/panels.py`.

Renames: `ORANEnvironment` → `ORANEnvironment1` (consistent with `ORANEnvironment2`);
`predictNextState` → `predict_next_state`; `evaluatePredictions` → `evaluate_predictions`;
`nodeNames` → `node_names`; `get_algorithms` → `get_planners`. No camelCase remains.

**`tests/conftest.py` was deleted entirely** — its `sys.path.insert` hack became
unnecessary once everything sat under one package, since pytest resolves `cdd_oran` from
the repo root on its own. That was the CWD-independence win, obtained **without** making
the project an installable distribution.

**`[tool.uv] package = false` is deliberate.** Installing was evaluated and rejected: the
restructure already delivers the import-resolution benefit, while a `[build-system]` adds a
build step and a stale-install failure mode for no practical gain here. Revisit only if the
journal wants an artifact badge or `pip install git+…` — a ~10 minute change, nothing
blocks it.

Verified: 24 tests green with `tests/golden/` unmodified (this was a pure move-and-rename —
no number was permitted to change), `ruff check` clean, `mypy` clean on 42 files, both
`python -m cdd_oran.cli` and `python cli.py` working. Git tracked **34 renames**, so
history follows the files.

## Original target layout (superseded by the above)

```
cdd_oran/
  config.py              # dataclasses + YAML; zero import side effects
  envs/
    base.py              # BaseORANEnv, XApp, Param, KPI  (~all shared mechanics)
    env_i.py env_ii.py   # specification only: equations, ranges, thresholds, adjacency
    statistics.py        # cached Monte-Carlo mean/std, calls env's own KPI fns
    registry.py
  models/
    base.py              # WorldModel / CausalModel protocols
    cdl.py mlp.py registry.py
  planners/
    base.py cost.py      # ONE weighted_distance, ONE score_batch
    qacm.py cem.py mppi.py mcts.py registry.py
  conflicts.py           # detect_conflict_edges
  viz/                   # graph.py heatmap.py panels.py  (no pyplot in models/)
  utils/                 # seeding.py logging.py io.py
  cli.py                 # train / evaluate / sweep / viz
configs/                 # env_i_cdl.yaml, env_ii_mlp.yaml, sweep_seeds.yaml, ...
tests/                   # incl. golden/ regression fixtures
docs/paper_config.md     # exact config behind every published figure/table
```

---

## Sequencing

Phase 0 gates everything. Phase 1 gates 2–5 (removing star-imports is what makes the
rest testable). Phases 2, 3, 4 are independent of each other and can be done in any
order or in parallel. Phase 5 depends on 1. Phase 6 runs continuously from Phase 1
onward — write the test with the refactor, not after. Phase 7 is post-refactor.

**Progress:** 0.2, 0.4, 0.5, 0.6 done. 0.1 deferred pending team confirmation; 0.3
deferred with it (retraining at `N_SIMULATIONS`/`NUM_STEPS`/Env I range values that differ
from the paper's produces checkpoints that cannot reproduce Table `tab:mitigation`).

**Next: Phase 1**, now covered by the golden lock.

What the lock does **not** cover, and must be kept in mind:

- It pins **current behaviour, not paper-correct behaviour.** It catches "the refactor
  changed the numbers"; it cannot catch "the numbers were already wrong" — e.g. the
  divergent `_weighted_distance` copy (Phase 3.1). That still needs 0.1 + 0.3.
- It uses the ground-truth graph, so **CDL's graph learning is untested**. `update_mask`
  / CMI estimation have no coverage until 0.3 provides trained checkpoints, or Phase 6.1
  adds the toy-system test.
- `main.py` and `evaluate_algorithms.main()` are untested end to end; only the numeric
  core they call is locked. Phase 5 rewrites both, so add CLI-level tests there.
