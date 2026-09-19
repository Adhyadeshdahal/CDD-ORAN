# Causal-discovery upgrade — theory & FIT to our E2–E5 benchmark (2026-09-06)

**Lane 3 of 3 (theory + rigorous FIT).** Read-only research. Identifies the two supplied arXiv
papers, audits the theoretical foundations of kernel/CI- and functional-causal discovery against
**our** benchmark's peculiarities (noiseless deterministic Gaussian-bump KPIs, mutually-independent
exogenous params, **known** temporal order, KPI→KPI true-negatives), and gives a blunt verdict on the
"move toward causality." No code changed. Nothing frozen is touched; this informs a *future* fresh
pre-registration only.

**Grounding read:** `reports/2026-09-06-e2-rcot-lowrecall-diagnosis.md`,
`docs/benchmark/E2_RCOT_DISCOVERY_PROTOCOL_V2.md`, `cdd_oran/e2slice/discovery_rcot.py`.

---

## 0. The benchmark's causal structure, stated precisely (this is what everything is judged against)

- **8 exogenous params**, sampled **mutually independently** → they are the DAG **roots**; there are
  **no edges among them**.
- **6 KPIs**, each a **deterministic, noiseless** (`obs_noise_scale = 0`) nonlinear (Gaussian-bump)
  function of a **subset** of the params: `K_j(t+1) = f_j(P_{S_j}(t))`.
- **Known temporal order** `t → t+1`. Candidate edges are `param(t)→KPI(t+1)` (48) and lagged
  `KPI(t)→KPI(t+1)` (36).
- **All 36 lagged KPI→KPI candidates are true-negatives** — KPIs are pure functions of the *current*
  params, with no autoregression (per the E2 method redesign finding in project memory and confirmed in
  `discovery_rcot.py`'s docstring).

**Consequences that drive the entire FIT analysis:**

| Classical assumption | Who needs it | E2 status | Why |
|---|---|---|---|
| **Causal sufficiency** (no latent confounders) | PC, and the *reason* FCI exists | **SATISFIED** | params are independent exogenous roots; nothing hidden is a common cause |
| Acyclicity | all | **SATISFIED** | feed-forward `t→t+1` |
| **Orientation known a priori** | *no* method assumes this | **WE HAVE IT (a gift)** | temporal order fixes every edge direction |
| **Faithfulness** | PC / FCI (constraint-based) | **VIOLATED** | determinism induces extra CIs not from d-separation |
| Additive noise (ANM) / post-nonlinear (PNL) | ANM, PNL | **VIOLATED** | `obs_noise=0`, and a Gaussian bump is not additive-noise-shaped |
| Linear + non-Gaussian | LiNGAM | **VIOLATED** | mechanism is nonlinear |

The single most important reframing: **because direction is known and KPI→KPI are negatives, our task
is not "orient a graph." It is, per target KPI, "find the minimal subset of params `S_j` such that
`K_j = f_j(S_j)`" — a functional-support / Markov-boundary selection problem under determinism.** Almost
every theoretical advance in "causal discovery" is machinery for *orienting* edges we already have
oriented; that machinery is, for us, solving a solved problem.

---

## 1. What the two supplied papers actually ARE — and their (limited) relevance

Both were retrieved and identified precisely. **Neither is about determinism, and neither addresses our
blocker.** They are both about *practical extensions of the constraint-based (PC/FCI) family*.

### 1a. arXiv **1705.09031** — *"Fast Causal Inference with Non-Random Missingness by Test-Wise Deletion"*, Eric V. Strobl, Shyam Visweswaran, Peter L. Spirtes (2017)

- **What it is:** a fix for running **FCI/RFCI on data with values Missing-Not-At-Random (MNAR)**.
  Instead of list-wise deletion (drop any sample with any missing value), it does **test-wise deletion**
  (drop samples only among the variables entering each specific CI test), proven sound when missingness
  mechanisms don't cause each other. ([abs](https://arxiv.org/abs/1705.09031))
- **Relevance to E2–E5: essentially none, directly.** Our data is **fully synthetic and complete** —
  there is **no missingness**. The whole apparatus solves a problem we do not have.
- **The only genuine thread:** the author is **Eric Strobl**, who also authored **RCoT** (*Approximate
  Kernel-based Conditional Independence Tests…*, arXiv 1702.03877) — the exact test our frozen method
  uses. So this paper places us in Strobl's **constraint-based, kernel-CI discovery lineage**; it does
  not tell us anything about how to beat determinism. *Skeptical read: this "primary link" is lineage
  context, not a method we can apply.*

### 1b. arXiv **1502.02454** — *"A fast PC algorithm for high dimensional causal discovery with multi-core PCs"*, Thuc Duy Le, Tao Hoang, Jiuyong Li, Lin Liu, Huawen Liu (2015)

- **What it is:** **parallel-PC** — an engineering re-implementation of the PC algorithm that
  distributes CI tests across CPU cores, turning a >24 h run into ~6 h on 8 cores on a DREAM5-scale
  (thousands-of-variables) gene network. Pure **scalability**; the paper is explicit that it contributes
  *no* new theory on CI testing or discovery. ([abs](https://arxiv.org/abs/1502.02454))
- **Relevance to E2–E5: none scientifically.** Our graph is **14 nodes / 84 candidates**. Scalability is
  a non-problem; RCoT-v2 runs all 10 seeds in ~1.7 h single-core. Parallelism would only matter if we
  ever adopted a full PC/FCI sweep over a much larger O-RAN variable set (E5+), and even then it is an
  implementation detail, not a discovery idea.

**Verdict on the supplied links (blunt):** both anchor the *constraint-based PC/FCI + kernel-CI* family
that a "move toward causality" would draw on, but **neither touches the determinism/noiselessness crux
that is our actual difficulty.** They tell us *which family* people mean by "go causal," not *whether it
works on a noiseless deterministic benchmark*. That question is answered below from the primary
determinism literature, not from these two papers.

---

## 2. The real theory — determinism, faithfulness, and functional causal models — mapped onto E2

### 2a. Why plain PC/FCI is *actively harmed* here (faithfulness violation by determinism)

Constraint-based methods assume **faithfulness**: every conditional independence in the distribution is a
consequence of d-separation in the DAG. **Determinism breaks this.** If `Y = f(S)` exactly, then
`Y ⟂ W | S` for **every** `W` — including `W` that is *not* d-separated from `Y`. These "extra"
independences are not in the graph, so PC/FCI **delete true edges and mis-orient**. This is the textbook
failure and is confirmed by the on-point recent work **"On Causal Discovery in the Presence of
Deterministic Relations" (NeurIPS 2024)**, which states plainly that faithfulness is violated under
deterministic relations and that constraint-based methods are unreliable there; it proposes handling
deterministic variable *clusters* explicitly (and notes exact score-based methods as an alternative).
([NeurIPS 2024 PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/ec52572b9e16b91edff5dc70e2642240-Paper-Conference.pdf))
The classical partial remedy is to weaken faithfulness to **Adjacency-Faithfulness** and use
**Conservative PC (CPC)**, which refuses conclusions that the weaker assumption can't support
([Ramsey et al., UAI 2006](https://dl.acm.org/doi/10.5555/3020419.3020468)); see also the
[review of constraint-based discovery](https://arxiv.org/pdf/1611.03977).

**But — the crucial empirical nuance from our own diagnosis, which refines the naive story:** in E2 the
faithfulness violation did **not** manifest as vanishing power on the true `param→KPI` edges. The
diagnosis measured **raw per-cell conditional-test power ≈ 80%** (128/160 true edges rejected at p<0.05);
RCoT-v2 recovers ~0.7 recall. The reason: the candidate parent `P_i` being tested is **not** in the
conditioning set `Z` (Z = the other 13 candidates), so its deterministic contribution to `Y` **survives
residualization**. The "condition on everything ⇒ explain away the true parent" hypothesis was explicitly
**falsified** (power is flat across `Z_full`, `Z_params_only`, `Z_none`, `Z_coparents`).

Where determinism **did** bite in E2 is the **other** direction — the **KPI→KPI false positives**.
Sibling KPIs `K_a, K_b` share params; through those shared deterministic parents they are marginally (and,
if the null is mis-specified, apparently *conditionally*) dependent. The original marginal-permutation
pdCor null destroyed the `candidate|Z` coupling and over-rejected → **9× KPI→KPI over-selection**. RCoT's
*genuine* conditional test (residualize on RFF of Z) fixed that. **So in our benchmark, determinism shows
up as spurious dependence among co-children, not as lost power on true edges** — an important, non-obvious
distinction that any theory we adopt must respect.

### 2b. Functional Causal Models — mostly NON-STARTERS for us, and exactly why

| Method | Core mechanism | Why it fails / is redundant for E2 |
|---|---|---|
| **ANM** (Hoyer et al. 2009) | orient `X→Y` by finding the direction whose regression residual is independent of the cause | **Double non-starter.** (1) The residual *is the noise*; at `obs_noise=0` the residual is ~0 in **both** directions and the independence test is degenerate — literature is explicit that **"noise is essential for ANM identifiability"** ([JMLR 21/19-664](https://www.jmlr.org/papers/volume21/19-664/19-664.pdf)). (2) A Gaussian bump is not additive-noise-shaped. (3) Even if it worked it yields **direction**, which we already know. |
| **PNL** (Zhang & Hyvärinen 2009) | post-nonlinear generalization of ANM | Same as ANM: needs a noise term to test; degenerate at zero noise; produces orientation we don't need. |
| **LiNGAM** (Shimizu et al.) | linear + non-Gaussian ⇒ ICA recovers order | Mechanism is **nonlinear** ⇒ assumption violated at the root. Also produces order we already have. |
| **IGCI** (Daniusis/Janzing) | the **one** FCM built for **noiseless, deterministic, invertible** `Y=f(X)` — literally our regime — orients via independence of `P_X` and `log f'` (ICM) | **Still a non-starter for us**, three ways: (a) it is a **bivariate orientation** tool → answers **direction**, which temporal order already fixes; (b) it needs the **info-geometric ICM postulate** (`P_X ⟂ f'`), which we *engineer* both sides of (params **and** the bump), so it is **not guaranteed** and can be violated arbitrarily; (c) it assumes an **invertible** `f`, but our KPIs are **many-to-one** (several params → one KPI; bumps are non-injective), breaking the premise. ([IGCI justification, arXiv 1402.2499](https://arxiv.org/abs/1402.2499)) |

The pattern is stark: **every FCM method's product is edge *orientation*, obtained under a noise or
invertibility assumption we violate — and orientation is the one thing we already have for free.** The
FCM literature is aimed at a question E2 does not ask.

### 2c. What DOES apply — determinism-native functional-support / Markov-boundary discovery

Reframe the task as it actually is: for each target KPI `Y`, find the **minimal param subset `S` with
`Y = f(S)`.** Under **noiselessness this is razor-sharp**: the conditional entropy `H(Y | S) = 0`
**exactly** when `S ⊇` the true parents, and drops to 0 for **no** smaller set. Determinism is a
**friend** to a method that uses the right primitive, and a *foe* only to methods (CI/faithfulness) that
assume it away. Candidate primitives worth a prototype:

1. **Functional-dependence / predictability screening (strongest fit).** Param `i` is a parent of `Y`
   iff `Y` is **not** a function of `params∖{i}` — i.e., adding `i` strictly reduces achievable
   prediction error / conditional entropy. Fit a flexible noiseless-friendly regressor (GP, gradient
   boosting, random forest) `Y ~ params`; a param is a parent iff its removal makes the residual leave 0
   (equivalently, nonzero permutation/importance). Because the data is noiseless, the "residual hits
   exactly 0" signal is unusually clean and is **not** available in a noisy benchmark.
2. **Minimal-subset (Markov-boundary) search** with a **determinism-aware** independence oracle — e.g.
   the deterministic-cluster handling of the NeurIPS-2024 method, or a CI test that first quotients out
   exact functional relations before testing. This keeps a constraint-based flavor but repairs the
   faithfulness violation instead of ignoring it.
3. **Sensitivity / partial-derivative screening.** `∂Y/∂P_i ≠ 0` somewhere ⇒ `P_i` is a parent. Trivial
   to estimate on noiseless data; naturally handles the Gaussian-bump form (nonzero gradient inside the
   bump's support).

All three **exploit** noiselessness rather than fighting it, and all three **subsume** what RCoT is doing
(RCoT is a CI test straining against determinism; these test functional dependence directly). This is the
one direction I would actually prototype.

---

## 3. Is a determinism-aware / functional approach worth prototyping to beat RCoT's recall?

**Qualified yes — but calibrate the expectation against what RCoT-v2 already achieves.** The diagnosis
already established that RCoT's low *recall* in v1 was **not** determinism and **not** low power — it was a
**BH × permutation-resolution arithmetic artifact** (floor `1/(B+1)=0.010` couldn't clear the BH leading
threshold `0.05/14=0.0036`), fixed a-priori by `block_perm_reps 99→299` (floor `0.0033 < 0.0036`),
restoring recall to ~0.7 at KPI→KPI FP ~0.01. **So "beat RCoT's determinism-limited recovery" is partly a
false premise: RCoT-v2's recovery is not determinism-limited, it's already ~0.7.**

Therefore a functional-method prototype is worth it **only if** it clears a concrete bar, not merely
"because it's more causal":

- **Bar to beat:** RCoT-v2's **~0.7 NCP→KPI recall at ~0.01 KPI→KPI FP**, at comparable compute.
- **Where a functional method should genuinely win:** (a) the **remaining ~30% of true edges** RCoT
  misses — plausibly the low-amplitude / heteroscedastic width-param edges (the diagnosis flags these as
  the hard-to-detect denominator); a direct functional-support test with the exact-zero-residual signal
  may catch edges a kernel-CF-covariance statistic smears out. (b) **Robustness as E3–E5 add real
  KPI→KPI edges or confounding**, where the CI/faithfulness strain gets worse and the functional
  reframing degrades more gracefully.
- **Discipline:** prototype **truth-free on the byte-faithful test-bed** (`calib_study4_correct.kpi_fn`),
  score against the test-bed's *own* known support, and only if it clears the bar there, take it to a
  **fresh pre-registration** (never re-tune the frozen RCoT contracts). This mirrors exactly how RCoT-v2
  was de-risked.

**Do NOT prototype:** vanilla PC/FCI, ANM, PNL, LiNGAM, or IGCI — §2b shows each either violates a core
assumption or only produces orientation we already possess.

---

## 4. Blunt verdict — is "moving toward causality" genuinely promising here, or does noiselessness cap it?

**On the "move to PC/FCI + functional-causal methods" as usually advertised: LOW value for E2 as posed,
and partly a category error.** Reasons, in order of importance:

1. **We already have the output those methods fight to produce — orientation.** Temporal order fixes
   every edge direction. FCM methods (ANM/PNL/LiNGAM/IGCI) and half of PC's machinery exist to *orient*;
   for us that is a solved problem, so their marginal value is near zero.
2. **Causal sufficiency already holds**, so **FCI is overkill** and arXiv 1705.09031's missing-data
   apparatus is irrelevant (no latents, no missingness). arXiv 1502.02454 solves a **scale** problem we
   don't have (14 nodes). The two supplied "primary links" target problems E2 doesn't exhibit.
3. **Plain PC is actively *harmed* by determinism** (faithfulness violation) — adopting it naively would
   *lose* true edges the current CI approach already finds.
4. The **genuinely promising** upgrade is **not more causal-discovery apparatus** but a **determinism-
   native functional-support / Markov-boundary discovery per target**, which exploits noiselessness and
   subsumes RCoT. That is a modest, well-scoped prototype — not a wholesale "go causal."

**Does noiselessness cap it too? — Mostly NO, and this is the key link to the benchmark-noise question:**

- **For edge detection / parent selection (our actual task): noiselessness does NOT cap recovery — it
  *sharpens* it.** `H(Y|S)=0` is an exact, unambiguous parent signal. Our diagnosis confirms edges are
  recoverable (raw power ~80%; RCoT-v2 ~0.7). Noiselessness is a **strength** for a functional method and
  merely a **strain** (not a wall) for a CI method.
- **For orientation-by-FCM: noiselessness *does* cap it** (ANM/PNL degenerate at zero noise) — but we
  **don't need orientation**, so this cap is irrelevant to us.
- **The one real determinism cost** is the **co-parent spurious-dependence** that produced the KPI→KPI
  false positives — and a *proper conditional/functional* test already controls it (that is literally why
  RCoT replaced pdCor).

**Therefore, on the benchmark-noise question directly:** **adding noise is not needed and would be
counter-productive.** The diagnosis already found "no benchmark-noise change is needed" for E2's recall;
theory strengthens that — adding noise would **disable the sharpest determinism-native signal** (exact
zero residual) while only **enabling FCM orientation we don't use**. Near-determinism is arguably a
**realistic and desirable stressor** (real O-RAN KPIs are near-deterministic given RIC params), and the
literature's warning that synthetic-DAG benchmarks are gameable via variance/scale artifacts
([Reisach et al., "Beware of the Simulated DAG!", arXiv 2102.13647](https://arxiv.org/pdf/2102.13647))
argues *for* keeping a mechanism that doesn't reward scale-exploiting shortcuts. **Change the method
(functional-support discovery), not the benchmark.**

**Caveat for E3–E5 (where the verdict could flip):** if a later environment deliberately introduces
**(i) genuine stochastic noise** → ANM/PNL become applicable and CI-faithfulness eases (but it is then a
different benchmark); **(ii) latent confounders** → causal sufficiency breaks, and *then* FCI (and, only
if data were also missing, arXiv 1705.09031's test-wise deletion) become genuinely relevant; **(iii) real
KPI→KPI edges** → the determinism-faithfulness problem gets *harder* and the functional-support reframing
becomes *more* valuable, not less. The right posture is to keep the benchmark noiseless, upgrade the
method to a determinism-native functional test, and re-audit these assumptions **per environment** as E3–E5
are designed.

---

## 5. One-paragraph summary

The two supplied papers (Strobl et al. 2017, MNAR test-wise deletion for FCI; Le et al. 2015, parallel
PC) are **constraint-based-family plumbing** — missing-data and scalability fixes — and **neither
addresses our determinism/noiselessness crux**; they only confirm the lineage of the PC/FCI + kernel-CI
"go causal" direction. Auditing that direction against E2: **causal sufficiency and orientation are
already ours for free, so FCI and every FCM orientation method (ANM/PNL/LiNGAM/IGCI) are redundant or
degenerate under our zero-noise, nonlinear, non-invertible, known-order structure**, and **plain PC is
actively harmed by the determinism-induced faithfulness violation**. Our own diagnosis shows E2's edges
are already recoverable (RCoT-v2 ~0.7 recall at ~0.01 FP) and that determinism bit us as **co-parent
false positives**, not lost power. The **only** upgrade worth prototyping is a **determinism-native
functional-support / Markov-boundary discovery per KPI** (predictability / zero-conditional-entropy /
sensitivity screening) that **exploits** noiselessness — benchmarked truth-free against RCoT-v2's bar
before any fresh pre-registration. **Noiselessness does not cap our task; it sharpens it — so keep the
benchmark noiseless and change the method, not the noise.**

---

## Sources

- [arXiv 1705.09031 — Fast Causal Inference with Non-Random Missingness by Test-Wise Deletion (Strobl, Visweswaran, Spirtes, 2017)](https://arxiv.org/abs/1705.09031)
- [arXiv 1502.02454 — A fast PC algorithm for high dimensional causal discovery with multi-core PCs (Le, Hoang, Li, Liu, Liu, 2015)](https://arxiv.org/abs/1502.02454)
- [On Causal Discovery in the Presence of Deterministic Relations (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/ec52572b9e16b91edff5dc70e2642240-Paper-Conference.pdf)
- [Adjacency-faithfulness and conservative causal inference (Ramsey, Zhang, Spirtes, UAI 2006)](https://dl.acm.org/doi/10.5555/3020419.3020468)
- [A Review on Algorithms for Constraint-based Causal Discovery (arXiv 1611.03977)](https://arxiv.org/pdf/1611.03977)
- [Justifying Information-Geometric Causal Inference / IGCI (Janzing et al., arXiv 1402.2499)](https://arxiv.org/abs/1402.2499)
- [Identifiability of Additive Noise Models Using Conditional Variances (JMLR 21, 19-664)](https://www.jmlr.org/papers/volume21/19-664/19-664.pdf)
- [Beware of the Simulated DAG! Causal Discovery Benchmarks May Be Easy To Game (Reisach et al., arXiv 2102.13647)](https://arxiv.org/pdf/2102.13647)
- Internal: `reports/2026-09-06-e2-rcot-lowrecall-diagnosis.md`; `docs/benchmark/E2_RCOT_DISCOVERY_PROTOCOL_V2.md`; `cdd_oran/e2slice/discovery_rcot.py`
</content>
</invoke>
