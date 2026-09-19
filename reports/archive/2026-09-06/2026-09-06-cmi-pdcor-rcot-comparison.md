# Conditional-dependence measures for causal edge discovery: CMI vs pdCor vs RCoT

**Date:** 2026-09-06 · **Status:** methodological note (journal-carryover). Motivated by the E2 discovery
redesign (`reports/2026-09-06-e2-improvement-synthesis.md`). Records precise definitions and the
relationship between three edge-scoring statistics used or proposed across the project, so the comparison
can be lifted into the journal extension.

## Context

The conference method (CDL) infers a causal edge $s_t^i\to s_{t+1}^j$ from a **model-based conditional
mutual information** estimated with a learned masked predictor. The label-free E-series discovery slice
(Plan 007, E2) instead uses a **model-free partial distance correlation** (`|pdCor|`). Analysis of E2
showed the E2 `|pdCor|` **pipeline** was slow ($O(N^2)$ distance computation plus the implemented
permutation null — both properties of the E2 procedure, not intrinsic to the pdCor statistic) and that its
edge selection was miscalibrated (over-selecting edges — a *structural* consequence of pdCor lacking the
CI-equivalence, see §2). It proposed a **randomized kernel conditional-independence test (RCoT)** as its
successor. This note pins the
three objects side by side and states precisely why they differ — in particular, why the E-series
`|pdCor|` is *not* the conference CMI, and why RCoT is closer in spirit to the original CMI than `|pdCor|`
ever was.

Throughout, we test a candidate edge via the conditional-dependence of $X$ (candidate parent) and $Y$
(target) given a conditioning set $Z$ (the other candidates / the rest of the state). The target null
hypothesis is $H_0: X \perp Y \mid Z$. A *population dependence measure* is especially useful for CI
testing when its population value is zero **exactly** under conditional independence
($\text{value}=0 \iff X\perp Y\mid Z$); a finite-sample CI *test* additionally requires a calibrated null
distribution for its statistic. These are distinct properties, and we keep them separate below.

## 1. Model-based CMI (conference / CDL)

For a candidate edge $s_t^i \to s_{t+1}^j$, with action $a_t$ and state $s_t$:

$$
\mathrm{CMI}^{ij}
= \mathbb{E}\!\left[\log \frac{\hat p_\theta\!\left(s_{t+1}^j \mid a_t, s_t\right)}
{\hat p_\theta\!\left(s_{t+1}^j \mid a_t, s_t \setminus s_t^i\right)}\right],
$$

which is the conditional mutual information $I\!\left(s_{t+1}^j ; s_t^i \mid a_t, s_t^{\setminus i}\right)$:
the expected log-ratio of the target's predictive density with vs. without input $i$. The two conditionals
are represented by a single **learned neural predictor** $\hat p_\theta$ trained with input masking (the
two-term CDL loss), and the estimate is EMA-smoothed and thresholded at a fixed $\epsilon$.

- **Family:** information-theoretic (Kullback–Leibler).
- **Model-based:** requires a trained density model $\hat p_\theta$.
- **Population CI measure:** the *true* $I\!\left(s_{t+1}^j; s_t^i \mid \text{rest}\right)$ satisfies
  $I=0 \iff s_{t+1}^j \perp s_t^i \mid \text{rest}$. The implemented CDL score is a **model-based estimate**
  of this quantity through $\hat p_\theta$ and therefore inherits estimation and model-specification error;
  it does not itself carry the exact population equivalence.
- **Cost:** cheap at inference. Features are extracted once and shared; the full (with-input) prediction —
  the numerator of the log-ratio — is computed **once per batch** and amortized across all candidates; and
  every candidate's masked (without-input) denominator is produced in a **single batched/vectorized pass**
  reusing those features (implemented via `vmap` over the dropped-source axis in `CDL.update_mask`,
  `cdd_oran/models/cdl.py`). The two conditionals of the log-ratio therefore do *not* cost two passes per
  candidate — the marginal per-candidate cost is one vectorized masked prediction-head evaluation on top of
  a shared full pass. The dominant cost remains the offline training of $\hat p_\theta$.


## 2. Model-free partial distance correlation, `|pdCor|` (E-series / Plan 007)

Given $n$ samples of $X, Y, Z$, form pairwise Euclidean distance matrices $a_{kl}=\lVert X_k-X_l\rVert$
(and $b_{kl}$ for $Y$, $c_{kl}$ for $Z$). **U-center** each (Székely–Rizzo unbiased centering):

$$
\tilde A_{kl} = a_{kl} - \tfrac{1}{n-2}\!\sum_{m} a_{km} - \tfrac{1}{n-2}\!\sum_{m} a_{ml}
+ \tfrac{1}{(n-1)(n-2)}\!\sum_{m,q} a_{mq}\quad (k\neq l),\qquad \tilde A_{kk}=0,
$$

with the U-inner product
$\langle \tilde A, \tilde B\rangle = \tfrac{1}{n(n-3)}\sum_{k\neq l}\tilde A_{kl}\tilde B_{kl}$.
Condition on $Z$ by a **single rank-1 projection** (residualizing the $X$- and $Y$-distance matrices on the
$Z$-distance matrix):

$$
P_{\tilde A} = \tilde A - \frac{\langle \tilde A, \tilde C\rangle}{\langle \tilde C, \tilde C\rangle}\,\tilde C,
\qquad
P_{\tilde B} = \tilde B - \frac{\langle \tilde B, \tilde C\rangle}{\langle \tilde C, \tilde C\rangle}\,\tilde C,
$$

and take the signed cosine

$$
\mathrm{pdCor}(X,Y;Z) = \frac{\langle P_{\tilde A}, P_{\tilde B}\rangle}
{\lVert P_{\tilde A}\rVert\,\lVert P_{\tilde B}\rVert},
\qquad \text{edge score } = \lvert \mathrm{pdCor}\rvert .
$$

- **Family:** distance-covariance (a geometric dependence measure).
- **Model-free:** no learned model — computed directly from sample geometry.
- **NOT a CI test:** $\mathrm{pdCor}=0 \nRightarrow X\perp Y\mid Z$ (Székely–Rizzo 2014, explicit). It is a
  conditional **association** score. The "partial" is a single linear projection in U-space and cannot
  fully residualize a target that is a rich nonlinear/deterministic function of a high-dimensional $Z$.
- **Cost:** $O(N^2)$ distance matrices; thresholded by a per-candidate permutation null + BH-FDR
  ($\approx 84$ candidates $\times$ 999 permutations $\times$ $N{=}4000$ ⇒ multi-day per the E2 study).

## 3. Randomized conditional-independence test, RCoT (proposed successor)

Test $X \perp Y \mid Z$ via random Fourier features (RFF) approximating an RBF kernel. Map each variable:

$$
f_X(x) = \sqrt{\tfrac{2}{D}}\,\big[\cos(\omega_r^\top x + b_r)\big]_{r=1}^{D},
\qquad \omega_r \sim \mathcal N(0,\sigma^{-2} I),\ \ b_r\sim \mathrm{Unif}[0,2\pi],
$$

Writing $A, B, C$ for the RFF representations of $X, Y, Z$ (and per-block bandwidths $\sigma$ set by the
median heuristic), RCoT forms the **partial cross-covariance** of $A$ and $B$ given $C$,

$$
\widehat C_{AB\cdot C} = \widehat C_{AB} - \widehat C_{AC}\,\widehat C_{CC}^{-1}\,\widehat C_{CB},
$$

and the statistic

$$
S' = n\,\big\lVert \widehat C_{AB\cdot C}\big\rVert_F^2 .
$$

Equivalently, $\widehat C_{AB\cdot C}$ is the cross-covariance of the feature residuals after (ridge)
regressing $A$ and $B$ on $C$ — i.e. testing whether the residualized nonlinear feature representations of
$X$ and $Y$, after conditioning on $Z$, remain correlated; the $\widehat C_{CC}^{-1}$ is regularized in the
implementation. This is a randomized-feature approximation to KCIT. Under $H_0: X\perp Y\mid Z$, $S'$ has an
asymptotic **weighted-$\chi^2$** null distribution, which RCoT approximates analytically (Lindsay–Pilla–Basak
or Hall–Buckley–Eagleson) to yield a **p-value with no permutation**.

- **Family:** kernel-based (RFF approximation of KCIT).
- **Model-free:** random features + regression; no trained density model.
- **CI test with a calibrated null:** built around $H_0: X\perp Y\mid Z$ and consistent against general
  (nonlinear, non-monotonic) alternatives as $N, D\to\infty$. The finite-feature construction is not
  literally $S'=0\iff X\perp Y\mid Z$ for every distribution — it tests the partial cross-covariance null in
  the randomized feature space — but conditionally dependent variables are generically conditionally
  correlated after at least one nonlinear feature map, which is what gives RCoT its CI-testing power.
- **Cost:** $O(d^2 N)$ for $d$ random features, i.e. **linear in $N$** for fixed $d$, with no permutation
  loop ⇒ minutes for the whole E2 seed envelope.

## Side-by-side

| Property | CMI (conference) | `|pdCor|` (E-series) | RCoT (proposed) |
|---|---|---|---|
| Statistical family | Info-theoretic (KL) | Distance covariance | Kernel (RFF) |
| Requires a trained model? | **Yes** — neural $\hat p_\theta$ | No | No |
| CI-equivalence ($=0\iff$ CI)? | Yes at population level (estimate is model-limited) | **No** — association only | Yes asymptotically (finite-$D$ approximate) |
| Handles nonlinear/non-monotonic dep.? | Yes (model-limited) | Yes | Yes |
| Threshold / null | Fixed $\epsilon$ on EMA CMI | Permutation + BH-FDR | Analytic $\chi^2$-mixture p-value + BH-FDR |
| Complexity in $N$ | 1 shared full pass + 1 batched masked pass (cheap infer.) | $O(N^2)\times$ perms | $O(d^2 N)$, linear in $N$, no perms |
| Label/model-free discovery? | No (needs offline training) | Yes | Yes |

## The relationship — the point worth carrying to the journal

1. **The E-series `|pdCor|` is *not* the conference CMI.** They are different families entirely: CMI is a
   model-based information-theoretic quantity estimated through a learned predictor; `|pdCor|` is a
   model-free geometric statistic computed from sample distances. They coincide neither in definition nor
   in guarantees.
2. **CMI and RCoT are both built around the correct null** — "$X\perp Y \mid Z$?". At the population level
   CMI is zero exactly under conditional independence; RCoT tests the corresponding partial cross-covariance
   null in its randomized feature space and is a consistent CI test asymptotically (its finite-sample
   guarantees are approximate, and CMI's are limited by $\hat p_\theta$ — see §§1,3). **`|pdCor|` is the
   outlier**: even at the population level it only *approximates* the CI question with a
   conditional-*association* score that lacks the $=0\iff$CI equivalence entirely. This is the structural reason it
   over-selects (see the E2 synthesis: false positives concentrate on conditioning-set-dependent
   candidates, precisely where an association-not-CI score with a rank-1 partial is weakest).
3. **The reason the E-series adopted `|pdCor|` was deliberate but costly.** The discovery slice was
   designed to be **label-free and training-free** — recovering structure from raw observations *without*
   fitting a world model, unlike the conference CDL which trains $\hat p_\theta$. `|pdCor|` was the
   model-free stand-in; the trade dropped the genuine-CI property to avoid training a model.
4. **RCoT is the reconciliation.** It is **model-free like `|pdCor|`** (no predictor to train) yet a
   **CI test like the conference CMI**, and **in our E2 setting** it is expected to be substantially
   cheaper than the current pdCor-plus-permutation pipeline ($O(d^2 N)$ analytic null vs. $O(N^2)\times$
   permutations), while avoiding the offline training cost of CMI. In effect, RCoT recovers the CI-testing
   posture of the original conference CMI *without* paying for a trained density model — which is exactly
   why it is the natural successor for the label-free discovery slice.

**Journal framing.** This positions the project's discovery statistic on a clean axis — *model-based CI
(CMI)* → *model-free association (pdCor)* → *model-free CI (RCoT)* — and motivates the successor not as a
speed hack but as restoring the conditional-independence guarantee the label-free variant had given up.
The one caveat to state honestly in the journal is the noiseless-determinism regime (`Y=f(Z)` exactly),
which stresses *any* CI test and is being characterized empirically before RCoT is committed.

## References

- G. J. Székely, M. L. Rizzo. *Partial distance correlation with methods for dissimilarities.* Annals of
  Statistics, 42(6):2382–2412, 2014. (U-centered distance covariance; partial distance correlation;
  pdCor$=0 \nRightarrow$ conditional independence.)
- E. V. Strobl, K. Zhang, S. Visweswaran. *Approximate kernel-based conditional independence tests for
  fast non-parametric causal discovery.* Journal of Causal Inference, 2019. (RCIT / RCoT; RFF;
  analytic weighted-$\chi^2$ null.) arXiv:1702.03877.
- Z. Wang et al. *Causal Dynamics Learning for Task-Independent State Abstraction.* ICML 2022. (Masked
  predictor + CMI-based edge discovery adapted by CDL.)
