# Causal upgrade — systems / GPU / scale lane (2026-09-06)

**Status:** read-only research note (LANE 2 of 3). **Nothing here is applied.** The frozen E2
method (RCoT-v2, `discovery_rcot_v2.py`, `PROTOCOL_COMMIT_V2 = 8052e10…`) is untouched. This informs
a **future** fresh pre-registration, which is the user's call. Scope: scalable + GPU-accelerated
causal *tooling* — what to ADOPT vs LEARN-FROM for genuine structure discovery (PC/FCI with RCoT as
the CI oracle) plus a batched-GPU-RCoT CI engine.

---

## 0. TL;DR

- **Both primary links are the WRONG problem for us.** Tencent fast-causal-inference and the NVIDIA
  RAPIDS blog are **effect estimation** (treatment effects / DML), not **structure discovery**
  (learning the graph). Do not adopt either for the discovery pipeline.
- **The real GPU structure-discovery prior art is cuPC and gpucsl** — both are GPU **PC skeletons**,
  but both hard-code a **Gaussian partial-correlation CI test in CUDA**. Neither lets you plug in
  RCoT without writing kernels, and neither runs on Windows out of the box (Linux/CUDA-C/CuPy). →
  **LEARN-FROM** (their parallelization scheme), **don't adopt**.
- **The single most on-target project found is Bloomberg `causal-ts`** — a PyTorch library that
  already ships a **GPU-accelerated RCOT CI test** and drives a **PC / PCMCI+ skeleton** with it.
  This is almost exactly our "batched-GPU-RCoT-as-CI-oracle driving PC" plan, cross-platform (pure
  PyTorch → Windows-viable). Caveat: **GPL-3.0-or-later**. → **LEARN-FROM (donor reference), adopt
  only with a license decision.**
- **`causal-learn` (py-why)** is the mature CPU host for correctness prototyping: PC/FCI with
  **pluggable CI tests incl. RCOT/KCI** and p-value caching. → **ADOPT as the skeleton driver /
  ground-truth for a PC-with-RCoT prototype** before any GPU work.
- **Build call:** the object we need to batch on GPU *is our RCoT* (RFF residualization +
  block-permutation null), not a generic Gaussian test — so **build the batched GPU-RCoT CI engine
  ourselves in torch**, keep the PC skeleton loop on CPU (causal-learn or a light custom loop). This
  fits **4 GB local** comfortably; cloud only for the full frozen multi-seed sweep.

---

## 1. The distinction you must not let anyone blur: effect estimation ≠ structure discovery

| | **Causal EFFECT estimation** | **Causal STRUCTURE discovery** ← *us* |
|---|---|---|
| Question | *Given* the graph (or a treatment/outcome pair), how big is the effect? (ATE, CATE, uplift) | *Learn the graph*: which of the 84 candidate edges are real? |
| Inputs | Treatment T, outcome Y, covariates; a known/assumed adjustment set | Raw observational matrix; no known edges |
| Core math | Regression / DML / IV / matching / propensity | Conditional-independence tests + orientation (PC/FCI) or score search |
| Typical libs | DoWhy, EconML, DML, uplift, **Tencent fast-causal-inference**, **RAPIDS cuML DML blog** | **cuPC, gpucsl, causal-learn, gCastle, tigramite, causal-ts** |

Our E2 task (84 candidate edges → keep the real NCP→KPI / KPI→KPI ones) is squarely **structure
discovery**. The frozen RCoT-v2 is a per-edge CI test with per-target BH-FDR — a *proto*-skeleton;
the stated direction is to promote it to a real **PC/FCI skeleton with RCoT as the CI oracle**.

### Where each linked/found tool actually sits

| Tool | Effect est. or Structure? | Relevant to us? |
|---|---|---|
| **Tencent fast-causal-inference** | **Effect estimation** (ttest, OLS, Lasso, DID, DML, uplift, matching) over ClickHouse/StarRocks SQL | **No.** Wrong problem *and* wrong infra (OLAP/SQL at 100M-row scale; we have n=4000 dense numerics). BSD, v0.1.0, WeChat-internal. Ignore. |
| **NVIDIA RAPIDS causal blog** | **Effect estimation** — cuML RandomForest inside **DoubleML** (de-biased ATE). No PC, no CI test. | **No** for discovery. cuML is a generic GPU sklearn — could *incidentally* speed a regression inside RFF residualization, but marginal on 4 GB and not why the blog exists. |
| **RAPIDS cuGraph** | Graph *analytics* (traversal, centrality), **not** graph *learning* | **No.** cuGraph does not learn a DAG from data. Could only post-process an already-discovered graph — not a lever. |
| **cuPC** | **Structure discovery** (GPU PC skeleton) | **Yes** — learn-from. |
| **gpucsl** | **Structure discovery** (GPU PC skeleton) | **Yes** — learn-from. |
| **gpucmiknn** | **Structure discovery** (GPU CMIknn CI test for PC) | Yes — minor prior art. |
| **causal-learn** | **Structure discovery** (CPU PC/FCI, pluggable CI incl. RCOT) | **Yes** — adopt as host. |
| **Bloomberg causal-ts** | **Structure discovery** (GPU-RCOT + PC/PCMCI+) | **Yes** — closest match. |

---

## 2. GPU / scalable structure-discovery tooling — verdict on each

### 2a. cuPC — *LEARN-FROM (parallelization scheme), do not adopt*
- **What:** CUDA-C parallel PC skeleton (order-independent), R wrapper. Two variants: **cuPC-E**
  parallelizes CI tests across variable pairs (and within a pair); **cuPC-S** reuses per-conditioning-
  set computations across tests. Paper reports ~500×/1300× vs serial CPU; 11 h → ~4 s on a hard set.
- **CI test:** **Gaussian partial correlation only** (multivariate-normal Fisher-Z). This is the
  disqualifier — E2 is nonlinear/non-monotonic; Fisher-Z is exactly the blind spot RCoT exists to
  fix. Swapping in RCoT means rewriting the CUDA kernel.
- **Maturity/license:** research code, ~13 commits, stale since ~2020, **GPL-3.0**, Linux/nvcc.
- **Verdict:** **Learn-from** the *structure of the parallelism* — level-by-level skeleton, pairs ×
  conditioning-sets as the parallel axes. That mental model maps onto how we'd batch RCoT queries.
  Don't take the code (CUDA-C+R, Gaussian-only, GPL, Linux).
- Source: arXiv:1812.08491 (IEEE TPDS 2019); repo LIS-Laboratory/cupc.

### 2b. gpucsl — *LEARN-FROM (cleaner Python reference), do not adopt as-is*
- **What:** HPI-EPIC Python library, GPU PC via **CuPy** CUDA kernels. Cleaner packaging than cuPC
  (CLI, tests, multi-GPU experimental). ~9.5× vs pcalg, ~19.8× vs bnlearn.
- **CI test:** **Gaussian partial-correlation** + **discrete χ²** only, each a **hand-written CUDA
  kernel**. No documented plug-point for a custom (kernel-based) CI test — adding RCoT = writing a
  new CUDA kernel, not a Python subclass.
- **Maturity/license:** **MIT** (friendly), ~21 commits, CI pipelines present but small; **Linux-only,
  needs CUDA toolkit + matching CuPy**. No Windows support stated.
- **Verdict:** **Learn-from** — best readable reference for "PC loop orchestrating GPU CI kernels,"
  and MIT means we *may* borrow ideas/snippets. But it does not run on our Windows/torch stack and
  its CI menu excludes RCoT, so not a drop-in.
- Source: IEEE 2022 (GPUCSL); repo hpi-epic/gpucsl.

### 2c. gpucmiknn — *minor prior art, note only*
- GPU information-theoretic **CMIknn** CI test for constraint-based discovery (same HPI lineage).
  Proves GPU CI-oracle-for-PC is a trodden path. CMIknn itself was flagged risky at our 14-D
  conditioning width (see discovery-method-scout), so relevant as *pattern*, not *method*.
- Source: repo ChristopherSchmidt89/gpucmiknn.

### 2d. Bloomberg `causal-ts` — *LEARN-FROM / candidate-ADOPT (closest match), license gate*
- **What:** PyTorch time-series causal-discovery library. Ships **8 CI tests all GPU-accelerated via
  PyTorch — including `RCOT`** (also ParCorr, KCI, SplitKCI, DFCIT, LinSig, CMIknn, GCMI) behind a
  **unified CI-test layer**, and drives **CDNOTS (PC + temporal constraints) / CDNOTS+ (PCMCI+-style
  skeleton)**. This is, feature-for-feature, our target architecture: **GPU RCoT as the CI oracle for
  a temporal PC skeleton.**
- **Fit to us:** pure PyTorch → **Windows-viable on our torch 2.10/cu128 stack**; temporal constraint
  + t→t+1 orientation matches E2's known lag ordering; RCOT already implemented on GPU is a **donor
  reference** for the exact kernel we want to batch.
- **Reported speedups** are modest (KCI 2.9×, SplitKCI 2.0× on an L4, T=1000/d=20) — GPU wins grow
  with n and permutation count, which is precisely our regime (n=4000, 299 perms × 84 candidates).
- **Maturity/license:** pip-installable, Python 3.10+, active (Bloomberg OSS). **GPL-3.0-or-later** —
  a genuine gate: vendoring/importing it makes our tree GPL. For a research benchmark that may be
  fine; for anything redistributed it's a decision. **Reading it for design/validation is
  unrestricted.**
- **Verdict:** **Learn-from now** (validate our GPU-RCoT numerics against theirs; study their CI-layer
  API and PCMCI+ skeleton). **Adopt only** behind an explicit GPL decision; otherwise re-implement
  the small RCoT-GPU kernel ourselves (we own the frozen numerics anyway — median-heuristic
  bandwidth, RFF dims, T=n‖Cxy‖²_F, HBE/block-perm — so a clean-room torch port is cheap and keeps
  the freeze auditable).
- Source: arXiv:2607.24673; repo bloomberg/causal-ts.

### 2e. causal-learn (py-why) — *ADOPT as the CPU skeleton host / correctness oracle*
- **What:** mature Python constraint-based suite (PC, FCI, GES, …) with **pluggable CI tests incl.
  `rcit`/`rcot` and `kci`**, expert-constraint support, and **p-value checkpoint caching**.
- **Fit:** the honest, low-risk way to turn our per-edge RCoT into a real **PC/FCI skeleton** is to
  drive `causal-learn`'s PC with a **custom CI callable that wraps our frozen RCoT-v2**. Gives a
  correct, well-tested skeleton + orientation loop for free; GPU is confined to the CI callable.
- **Maturity/license:** widely used, documented, permissive (MIT-family). Cross-platform (pure
  Python). Its own RCOT is CPU — fine for a small-n correctness prototype; swap in our GPU-RCoT for
  the scaled run.
- **Verdict:** **Adopt** as the skeleton driver for prototyping PC-with-RCoT and as the reference the
  GPU pipeline must match. Source: arXiv:2307.16405; repo py-why/causal-learn.

### 2f. RAPIDS (cuML / cuGraph / cuDF) — *not for discovery; incidental utility only*
- **cuML** = GPU sklearn (used in the blog for DML **effect estimation**). No PC/CI/discovery. Could
  marginally accelerate a regression *inside* residualization, but our RFF residualization is already
  a couple of small dense solves — not the bottleneck (the **permutation null** is). Low leverage on
  4 GB. **cuGraph/cuDF**: analytics/dataframes, irrelevant to learning the graph.
- **Verdict:** **Don't adopt** for the discovery pipeline. Keep RAPIDS out of scope.

---

## 3. Adopt-vs-build for our stack (Windows / torch 2.10+cu128 / RTX 2050 4 GB + cloud)

**Build the CI engine, borrow the skeleton, learn from the GPU-PC prior art.** Rationale:

1. **The batched object is *our* RCoT, not a stock Gaussian test.** Every off-the-shelf GPU PC
   (cuPC, gpucsl) bakes Gaussian partial-correlation into a CUDA kernel; adopting them means
   rewriting a kernel *and* still leaving Linux-only, GPL/CuPy baggage. Since we already own the
   frozen RCoT numerics and can run torch today, a **clean-room batched GPU-RCoT in torch** is the
   shorter, more auditable path — and it keeps the pre-registration honest (we control every
   constant).

2. **Memory fits 4 GB with room to spare.** Per CI test the working tensors are tiny: RFF features
   are `n × dz` (4000×25) and `n × dxy` (4000×5); the statistic is a 5×5-ish cross-covariance. Even
   batching **all 299 block-perms × 84 candidates** at once is on the order of a few hundred MB in
   fp32 — the batch dimension, not any single matrix, is what you tile to stay under 4 GB. The
   permutation null (embarrassingly parallel, no cross-perm dependency) is the ideal GPU workload and
   the current CPU bottleneck. **Feasible locally.**

3. **Keep the PC skeleton loop on CPU.** The skeleton is O(vars²) bookkeeping with cheap set logic;
   its cost is dominated by the CI *tests*, which we offload. `causal-learn` (adopt) or a ~100-line
   custom level-by-level loop drives it and calls the GPU-RCoT oracle. No need for a GPU graph
   framework.

4. **Cloud GPU only where it pays:** the full frozen **multi-seed** sweep (the parked ~5–18-day
   pdCor-era cost) and any E4/E5 blow-up in n or vars. A single bigger card (A10/L4/A100) runs the
   whole sweep in hours; the *code* is identical to local (pure torch). Local 2050 is enough for
   development, single-seed validation, and E2-sized runs.

**What to actually take from each:** design/parallel-axis ideas from **cuPC**; Python-orchestration
pattern + MIT snippets from **gpucsl**; **GPU-RCoT numerics cross-check + CI-layer/PCMCI+ API** from
**causal-ts** (behind the GPL gate); **the PC/FCI skeleton itself** from **causal-learn** (adopt).
Ignore Tencent, the RAPIDS DML blog, cuGraph.

---

## 4. How it composes: batched GPU-RCoT as the CI oracle for a PC skeleton

```
                 CPU (causal-learn PC / light custom loop)
   level ℓ:  for each ordered pair (X,Y) still adjacent,
             for each conditioning set S ⊆ adj, |S| = ℓ:
                 enqueue CI query  X ⟂ Y | S
                        │  (batch many queries per PC level)
                        ▼
             ┌───────────────────────────────────────────────┐
             │  GPU-RCoT CI oracle (torch)                    │
             │  1. RFF-residualize X,Y on Z=S (frozen bw/dims)│
             │  2. statistic T = n·‖C_xy‖²_F                  │
             │  3. block-perm null: 299 perms  ← batch axis   │
             │  4. p-value  (block_perm floor 1/300)          │
             └───────────────────────────────────────────────┘
                        │  p-values back
                        ▼
             cut edge if p > α (+ BH-FDR / sepset store)  →  orient (t→t+1 known for E2)
```

Design notes / freeze cautions:

- **Two batch axes, both free wins:** (i) the **299 block-perms** for a single query (largest, most
  regular — do first); (ii) **multiple candidate edges at the same PC level** sharing the batch. The
  frozen RCoT-v2 already does per-edge perms on CPU — the GPU version must reproduce it
  bit-for-decision, not "improve" it.
- **p-value caching:** `causal-learn` caches CI results to checkpoint; PC re-queries the same
  (X,Y,S) across orientations, so caching compounds the GPU speedup. Free.
- **Determinism is a freeze requirement, not a nicety.** GPU atomics/reductions and the RFF RNG are
  the two nondeterminism sources; a frozen protocol needs pinned RNG streams, fixed reduction order
  (or fp64 accumulation for the statistic), and a documented device/driver, so a re-run reproduces
  the exact mask. **Validate the GPU oracle equals the frozen CPU RCoT-v2 decisions on E2 seed 0
  before it is allowed near a pre-registration.** This is the single biggest correctness risk in the
  whole plan.
- **PC vs the current per-target BH slice:** promoting to PC adds *conditioning-set search* (sepsets
  beyond "the other 13 candidates") — a real methodological change that must go through a fresh
  freeze + adversarial review, not slip in under "GPU acceleration." Keep the two changes separate:
  (a) GPU-accelerate the existing frozen test (behavior-identical, no re-freeze needed — pure
  systems); (b) adopt a PC skeleton (new science, new pre-reg). **Do (a) first; it de-risks (b).**

---

## 5. Sources

- Tencent fast-causal-inference — github.com/Tencent/fast-causal-inference (BSD, v0.1.0; **effect
  estimation**, ClickHouse/StarRocks SQL).
- NVIDIA RAPIDS causal blog — developer.nvidia.com/blog/faster-causal-inference-on-large-datasets-with-nvidia-rapids
  (cuML + DoubleML; **effect estimation**).
- cuPC — arXiv:1812.08491, IEEE TPDS 2019; repo LIS-Laboratory/cupc (CUDA-C+R, Gaussian, GPL-3, stale).
- gpucsl — GPUCSL (IEEE 2022); repo hpi-epic/gpucsl (CuPy, Gaussian/discrete kernels, MIT, Linux-only).
- gpucmiknn — repo ChristopherSchmidt89/gpucmiknn (GPU CMIknn CI test for PC).
- Bloomberg causal-ts — arXiv:2607.24673; repo bloomberg/causal-ts (**PyTorch GPU RCOT + PC/PCMCI+**,
  GPL-3.0-or-later, cross-platform).
- causal-learn — arXiv:2307.16405; repo py-why/causal-learn (CPU PC/FCI, **pluggable RCOT/KCI CI**,
  p-value caching).
- Scalable CI background — arXiv:1702.03877 (RCIT/RCoT), arXiv:2512.19510 (SCIT, learned spectral CI,
  GPU-accelerable, no code yet).
- Internal: `reports/2026-09-05-discovery-method-scout.md`; `cdd_oran/e2slice/discovery_rcot_v2.py`.
