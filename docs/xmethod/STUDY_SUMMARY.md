# Cross-method study: end-of-study record (Study A = paper Study 4; Exp B; Exp C; R-60 X1-X3)

Written 2026-10-05 from the reports cited below. It adds no new result: every number is copied from the cited file and
section. Paper names follow `D:/academia/major-project/TERMINOLOGY.md`; code names are used here.

## 1. What each part asked

| part | question | status |
|---|---|---|
| Study A (paper: Study 4) | On action -> KPI data logged under a known randomization design, which edge tests hold their nominal false-positive level, and how much power do the valid ones have (`PROTOCOL_A.md` s.1)? Claims C1 (design-blind tests invalid on the R2 design), C2a (design-based test valid), C2b (design-covariate adjustment restores validity), C3 (valid under the logged confounded policy). | confirmatory, frozen |
| Exp B (paper: bridging analysis) | How do the Study A methods behave on the frozen E6-P v4 data (`EXP_B.md` s.0)? | post hoc, descriptive, no verdict |
| Exp C / X1 (runtime) | CPU-s per method and n from an uncontended timing run on the Kaggle reference host (`PROTOCOL_A.md` s.8, R-55); plus the R-55 host speed-factor calibration used by T3. | descriptive |
| R-60 X2 | Dither dose-response: how R2 rates change with the dither width and the number of setpoint blocks (`EXTRAS_PROTOCOL.md`). | exploratory, declared before launch |
| R-60 X3 | Design misspecification: rates when the design told to a method is wrong (`EXTRAS_PROTOCOL.md`). | exploratory, declared before launch |

## 2. Protocol, freeze, amendment, audits

- Protocol: `docs/xmethod/PROTOCOL_A.md`, FROZEN 2026-10-04 (LF sha256 c5f7a4fe...; S = 40; 24 arms; E1-E5, R1-R4;
  150 960 units, 11 000 datasets). Freeze record: `scratchpad/xmethod/freeze/FREEZE_NOTE.md` (+ FREEZE_MANIFEST /
  FREEZE_CALIB). Freeze commit **93856c2** (feat/v2).
- Campaign audit `results/audit/campaign_eval_audit.md`: FAIL -> CONTRACT R-59 fixes before the freeze. Re-audit
  `results/audit/campaign_eval_reaudit.md`: PASS-WITH-NOTES, commit **5a4ef0f**.
- Amendment **A-1** (`docs/xmethod/PROTOCOL_A_AMENDMENTS.md`, `specs/eval/amendments.json`): every EVAL record was
  produced at commit 5a95186 = freeze + a dispatcher-only change (`xm_dispatch.py`) + the run cost table. Written
  before the analysis. EVAL report: commit **6f04f02**, integrity FINAL.
- X2 / X3 declaration `scratchpad/xmethod/EXTRAS_PROTOCOL.md`: commit **86a9767**. Report: commit **8efa8fd**.
  Amendments: none (`EXTRAS_REPORT.md`, Notes).
- X1 paper timing run: commit **b278d71**. Exp B results and report: commit **6ee82f4**.

## 3. Headline results (copied; descriptive parts stay descriptive)

- **Study A** (`results/eval/REPORT.md`):
  - Claim SUPPORTED: C1, C2a, C2b and C3 all SUPPORTED (s.1). V11 integrity FINAL: 150 960 / 150 960 records, no
    failed check (V11).
  - C1: 5 of 6 assessable set-D arms are FAILURES; rcot2_native is INVALID IN R1. Without mscr_native: SUPPORTED,
    4 of 5 (s.2).
  - C2a: pmrt_nl_eq has 1 INVALID cell of 50 (F_max 3); pooled truth-null raw .045 and placebo raw .048, VALID (s.3).
  - C2b: granger_eq and pcorr_eq SUPPORTED; mscr_eq excluded (R-40). Named arm (R-56): rcot2_eq is INVALID in R2
    (.292) as already in R1 (.061) (s.4).
  - C3: pmrt_nl_eq has 0 INVALID cells of 20 (F_max 2). pcorr_eq also passes, so "the property is not specific to
    PMRT". pmrt_eq (secondary) is NOT SUPPORTED, with 3 INVALID cells vs F_max 2 (s.5).
- **Exp B** (`docs/xmethod/EXP_B.md` s.9; 1328 / 1328 units, 12 cdl infeasible):
  - The covariate-adjusted linear arms behave almost identically: BY recall 0.49-0.52 at 60 episodes, 0.69-0.72
    pooled. All stay below the frozen v4 primary's recall (0.56-0.79).
  - pmrt_nl_eq BY recall is 0.39-0.55.
  - corr and granger_native reject the P_placebo column at 0.59 / 0.40 (60 episodes).
  - Slice CIs rest on 2-10 clusters.
- **Exp C / X1** (`results/exp_c/paper/EXP_C_RUNTIME.md`, summary table; Kaggle reference CPU-s per unit, median):
  - pmrt_nl_eq 65.3 at n 1000 and 1,108 at n 24000; cdl 75.8 and 1,436; the largest median at n 1000 is mscr_eq, 131.
  - All arms are feasible at n <= 24000, except mscr (not in grid above n 1000).
  - Calibration factors: `results/exp_c/calib/factors.json` (Colab Xeon f 1.067, VPS EPYC-Rome f 2.562; FREEZE_NOTE).
- **X2** (`results/extras/EXTRAS_REPORT.md`, X2): corr's pooled truth-null declared rate falls with the dither delta,
  0.771 / 0.748 / 0.686 / 0.514 at delta .02 / .05 / .10 / .20. pmrt_nl_eq's raw rate stays at 0.043-0.049, and
  pcorr_eq's raw rate at 0.048-0.049.
- **X3** (`EXTRAS_REPORT.md`, X3): the first INVALID variant is:
  - pmrt_nl_eq: width x0.8, switch +5 %, lambda x2 (R3, P_placebo_conf);
  - pmrt_eq: width x0.8, switch +2 %;
  - pcorr_eq: invariant to width and lambda, INVALID at switch +5 %.

  Declared wording: "a degradation curve showing where the known-design assumption breaks; it is not evidence
  against C2a".

## 4. Where the artifacts live (paths under `scratchpad/xmethod/` unless absolute)

| item | location |
|---|---|
| DEV records + reports (T1 / T3 / T5) | `results/dev/{full,ci_c,pmrt_nl,cdl,pmrt_nl_cost,final}/` |
| EVAL records, report, tables | `results/eval/` (merged.jsonl.gz + summary, REPORT.md, report.json, csv/, n1_check.json) |
| Exp B | `results/exp_b/` (merged, EXP_B_TABLES.md, EXP_B_COSTS.md); write-up `docs/xmethod/EXP_B.md` |
| Exp C | `results/exp_c/calib/` (factors.json + records), `results/exp_c/paper/` (EXP_C_RUNTIME.md, raw/) |
| X2 / X3 | `results/extras/` (EXTRAS_REPORT.md, EXTRAS_TABLES.md, extras_tables.json, x2_*/, x3_told/) |
| audits, figure review | `results/audit/`, `reviews/figure_review_2026-10-05.md` |
| figure code (cddfig) | `D:/academia/major-project/figures/src/cddfig/figs/`: fig_study4_{recall, calibration, validity, e4_example, env_e1..e5}.py, tab_study4_{verdicts, env}.py, _study4.py, _env_page.py |
| journal figures | `xApp-Journal-Discover-Telecommunications/figures/fig_study4_{recall, calibration, validity, e4_example, env_e1..e5}.pdf` |
| journal tables | `xApp-Journal-Discover-Telecommunications/tables/tab_study4_env_e1..e5.{tex,csv}` |

## 5. Platforms and compute

Host CPU-h is the sum of record `cpu_s` (method CPU time per unit) in each merged file, on the host that ran it.
Kaggle-reference CPU-h is quoted where a cost table gives it.

| campaign | Kaggle | VPS | Colab | Lightning | total host CPU-h | Kaggle-ref CPU-h (source) |
|---|---|---|---|---|---|---|
| DEV (full, ci_c, pmrt_nl, cdl, T3 pilot) | 390.3 | - | 55.8 | 23.0 | 469.1 | - |
| EVAL (150 960 units) | 239.6 | 76.6 | - | - | 316.3 | projection 509 (`freeze/eval_projection.json`) |
| Exp B (1328 units) | 42.5 | 15.9 | - | - | 58.4 | 82.0 (`EXP_B_COSTS.md`) |
| Exp C calibration (255 records) | 1.7 | 1.3 | 3.1 | - | 6.1 | - |
| X1 paper timing (342 units) | 4.5 | - | - | - | 4.5 | 4.5 (reference host) |
| X2 / X3 (18 240 units) | 35.9 | 12.2 | - | - | 48.1 | 63.9 (`EXTRAS_REPORT.md`, Cost) |

EVAL ran on Kaggle (Intel Xeon 2.20GHz 72 519 units; AMD EPYC 7B12 12 680) and the VPS (65 761). Python 3.12.14,
fork + rlimit isolation, 7200 CPU-s budget, 0 infeasible units (REPORT V10, V11).

## 6. Seed blocks (`docs/benchmark/SEED_REGISTRY.json`, claimed_by)

- XMETHOD_DEV 3_000_000-3_000_199 (E1-E5): DEV, T1, the T3 pilot, Exp C, and the EVAL tune role.
- XMETHOD_EVAL 3_100_000-3_100_299 (E1-E5): measure seeds 3_100_000-3_100_039, E4 R3 C3 readers up to 3_100_299.
- XMETHOD_EXTRAS 3_200_000-3_200_199 (E1-E4):
  - X2 tune 3_200_000-019; X2 measure 020-059; X3 measure 060-119;
  - 120-199 unassigned.
- Exp B used no seed: it reads only the frozen E6-P v4 episodes.

## 7. Open items

- Deferred experiments (not run):
  - E6 effect injection;
  - a diagnostic of the PMRT-GBM (pmrt_nl_eq) recall gap on E6-P (`EXP_B.md` s.9);
  - Exp B seed replication.
- The VPS directory `/opt/cdd-xm` (3.1 GB) is kept.
- EVAL tune reproducibility vs DEV (REPORT V11) gives a dataset sha equal for 38 992 / 39 000 records. The 8 others
  are one DEV dataset (E2 R2 n 24000, seed 3_000_016, DEV commit a7b6e36); declarations are equal for all 39 000.
  This is descriptive, and not investigated.
- Windows does not reproduce some Linux-generated datasets bit for bit (`results/extras/provenance_platform.json`).
  Every record was generated on Linux.
