# Overnight 2026-10-02/03: what was decided for you (please review; all reversible before EVAL)

You said: "go through it all, don't await me, I review in the morning". Every decision below is in CONTRACT.md
section 8 (R-numbers). Independent reviews / audits are in scratchpad/xmethod/consult/ and audit/.

## Your two answers before sleep
- R-27 kappa = .25 (pooled), sweep .125 / .5.  DEV launch now + 40 extra seeds for validity cells.

## Decided overnight (orchestrator; for your approval)
1. NOTEARS kept, F3 19/20 disclosed.  two_tower labelled "adaptation".  Budget 2 CPU-h / dataset (nothing near it).
2. Protocol reviewed by 3 independent agents (statistics / adversarial / reproducibility) -> R-29..R-36:
   - R-29 score-only methods: threshold now holds 5 % false alarms (old rule gave 9.5 % in E4).
   - R-30 each method x cell = VALID / INVALID / INCONCLUSIVE, same burden of proof for PMRT and baselines.
   - R-31 claim C2 split: C2a PMRT; C2b every other test given the design covariates. Fallback wording fixed now.
   - R-32/R-38 new design-blind baseline robust to serial dependence: pcorr_hac (fixed-b HAC; chosen on synthetic
     data only). Without it a referee could call C1 a strawman.
   - R-33 minimal-design arm (only the focal setpoint added) so high-dimensional covariates don't penalise kNN/kernel tests.
   - R-34/R-39 seeds: E4 300 (rule: >= 90 % chance a correct PMRT passes C3); others from power calc, 40-100.
   - R-35 everything pinned (packages, Python 3.12, torch version), EVAL guard tied to the frozen protocol.
   - R-36 wording "validity does not depend on the outcome model"; estimand check PASSED (no null edge has a real effect).
3. R-37 the placebo-confounder column is never used as a conditioner (any method).
4. R-40 MSCR's "equal-information" arm is not truly equal information -> excluded from C2b, labelled.
   pdCor is a dependence test, not a CI test -> labelled, and not counted for C1.
5. R-41 CMI-kNN runs on GPU (bit-identical to the CPU reference); 2 h GPU wall budget per dataset.

## Findings worth knowing
- Design-blind methods ARE invalid in R2 (smoke): native pairwise Granger 57/160 null edges declared; eq arm 0.
- RCoT's own null is liberal (even the authors' R code: .075-.11); with the full design set E2 R2 null .16.
- SHAP-DAG / two-tower flag many null edges (score-only, no p-value).
- CMI-kNN and pcorr in the equal-information arm hold level in the smoke runs.

## Status at time of writing
- Merged into feat/v2: all method adapters (classic, CI tests, pcorr_hac), audits, protocol v3 (docs/xmethod/PROTOCOL_A.md,
  NOT frozen), eval_analysis.py, EVAL spec.
- Running on Kaggle: DEV campaign (10 arms) + CI-test DEV pilot. Next: CI DEV run, power calc -> fill TBDs -> freeze -> EVAL.

## 09:40 update: DEV full run done; IMPORTANT finding
- pmrt_eq is slightly liberal in E2 R2 at n 1000 (truth-null rate .070 at kappa .125, .084 at .5; .061 at .25),
  fine at n 500 / 4000 and on the placebo. A design-based test should be exact here (estimand check: zero effect),
  so this looks like an implementation flaw (suspect: covariates built from the action's own past dither held fixed
  while the dither is redrawn). Worker pmrt-diag is diagnosing on reserve DEV seeds; any fix must be justified by
  theory and disclosed in PROTOCOL_A s.0. Freeze waits for this.
- CMI-kNN GPU cost: ~40 min per dataset at n 4000, ~5 h at n 8000 (over budget). Full grid at n 4000 may need far
  more GPU-hours than Kaggle's ~30/week: decision pending the pilot projection.
- R-44 Lightning up to 3.77 credits; Colab-first retry policy (user).

## Night 2026-10-03 -> 04 (user asleep; user approved: freeze + EVAL launch tonight if green; cloud only, no laptop shards, no Lightning)
- Evening: CI DEV run done (58 480 / 58 480 ok; results/dev/ci_c/REPORT.md in wt xm-pmrt). R-54 (user): mscr kept as
  reported-INVALID arm, EVAL n <= 1000. rcot2_eq INVALID in most R2 cells (.067-.104) -> C2b fallback wording (R-31).
- Running: cdl DEV (dispatcher, Kaggle + Colab), pmrt_nl_eq T1 DEV (Kaggle xm-dev-pmrtnl-k1).
- Workers tonight: aud1 = finish cdl + pmrt_nl DEV, power calc; xm-harness = freeze package (T-register, EVAL spec,
  launch plan, checklist) on xm/freeze-prep; cdl = Experiment B plan/driver on xm/exp-b (no launch until slots free);
  xm-citests = Experiment C runtime-table script on xm/exp-c.
- Decisions overnight are logged below with the subagent panel's votes.
- 00:20 R-55 (Exp C / T3 timing; xm-citests Q1-Q4): panel of 3 subagents (generalist, skeptical referee, pragmatist)
  unanimous: trimmed paired calibration under controlled load before EVAL; host = platform + CPU model; paper runtime
  table from a dedicated uncontended Kaggle timing run (campaign CPU-s supplementary); pool rows, median/mean/min/max.
  Also: EVAL runs 1 process per vCPU (ci_c ran ~3.5x oversubscribed) and logs CPU model / load per unit.
- 00:45 R-56 (xm-harness freeze-prep Q1-Q3; panel: methodologist, skeptical referee, pragmatist):
  Q1 3/3 keep C2b rule + name rcot2_eq as failing eq arm + unfiltered eq-arm table. Q2 2/3 (referee dissent: set_D
  false, "padding") keep mscr_native in set D + C1 sensitivity with/without mscr. Q3 2/3 (methodologist dissent: keep
  100, cap unlikely to bind) T6 cap 60, S = clip(S_power, 40, 60). REVIEW: Q2 and Q3 were split votes.
  Also answered Q4 myself: calibration anchors at n 500/1000/4000, 3 repeats.
- 01:10 Exp B (cdl worker, xm/exp-b): plan + code ready (19 arms on frozen v4 E6 episodes, read from Kaggle outputs;
  GT as stored: TRUE 29 / NULL 21 / INDET 10; ~71 core-h, cdl ~55). Answered defaults: LOSO conformal tau, full
  context in eq + native, mscr n <= 1000, granger kept (labelled). GO for the pilot only, lowest priority (launch when
  < 4 of my Kaggle sessions run); P1 waits (EVAL first). Copied campaign.py / dev_power.py into its worktree.
- 02:20 Merged into feat/v2: e087160 freeze package (protocol v5, T-register filled except T1 S; final EVAL spec;
  launch plan S 40 ~1 240 CPU-h ~56 h wall on 4 Kaggle + 3 Colab; checklist), aa6d02e EVAL report generator (DEV
  rehearsal: C1 SUPPORTED 5/6 (4/5 without mscr), C2a SUPPORTED, C2b SUPPORTED, C3 NOT SUPPORTED = INCONCLUSIVE cells
  at DEV seed counts, 0 INVALID for pmrt; EVAL uses 300 E4 seeds). eval_report.py frozen with eval_analysis (T7).
- 05:20 NST (23:25 UTC) Colab reclaimed cdl DEV jobs c1 and c3 after ~7 h (272 / 264 of ~318 units pulled; the slow
  large-n units remained). aud1 asked to requeue the unfinished units via resume (preferably Kaggle), stop the orphan
  assignment, and make the dispatcher auto-requeue lost jobs. Watcher now also flags Colab job loss / failed launches.
  Exp B pilot xm-expb-p0b launched on Kaggle (allowed lowest-priority slot).
- 06:00 NST Colab refuses new runtimes tonight (capacity / usage limit); dispatcher (aud1 efbf9e2) now auto-requeues
  incomplete parts to Kaggle first (resume from done keys). Kaggle Exp C k3 done 612/612. pmrt_nl k1 overran (~7 h vs
  3.5 h est; aud1 checks on pull). Told xm-citests to launch the R-55 calibration on Kaggle's 5th slot (Kaggle-only
  factors tonight; Colab factors provisional until Colab accepts runtimes).
- 07:00 NST Exp B pilot done 144/144; P1 GO (1 Kaggle slot at a time, yields to EVAL); over-budget cdl units measured (R-13). Calibration k1 + pmrt_nl k1 COMPLETE (pull asked). Requeue launch hit a slug collision; aud1 fixing.
- 09:06 NST user: keep R-56 split votes as decided; Lightning GO for remaining credits (aud1: last cdl parts 60-63 on one cpu-4 studio). Paperspace probe proposed (key to be stored by user in env/config, not chat).
- 11:06 NST R-58 EVAL scope trim (user GO after 4-reviewer panel): ~1 243 -> ~360-450 CPU-h; freeze no longer waits for large-n cdl DEV.

## 2026-10-04 afternoon/evening: freeze + EVAL launch (user approved launch; cloud only)
- DEV done: final T1 S_power 14 -> S = 40 (41d6c37). Exp B done (6ee82f4). Exp C calib block merged (12748be).
- Freeze checklist failures F1 (calib block missing) -> merged; F2 (no campaign.py audit) -> independent audit by
  the exp-b worker: FAIL (spec-sha stamp never matched eval_analysis -> every EVAL record PROVISIONAL; 2x-cap breach
  recorded error vs protocol "infeasible"). Rulings R-59 (code follows pre-registered text; F3-F9 safeguards),
  fixed by aud1 (campaign/vps_run) + xm-harness (eval_analysis), re-audit PASS-WITH-NOTES (5a4ef0f).
- xm-harness notes ruled: pkgs_lock torch '2.10.0+cpu' (what records stamp); uncalibrated EPYC 7B12 hosts =
  provisional T3 read (margin >= 26.8x); _ref/ kept as DEV provenance, outside the manifest; PROTOCOL_A s.11 stamp
  field renamed (R-59). Re-audit N1: every EVAL relaunch uses --skip-complete-from; infeasible+ok key check at merge.
- Freeze merged: 93856c2 (tree = xm/freeze 22d85c3). EVAL launch GO to aud1 (Kaggle + VPS, Colab if granted).

## 2026-10-04 night: supplementary experiments (user: "run experiments that strengthen the paper")
- 4-reviewer panel (referee, paper editor, integrity auditor, compute economist) -> R-60 (2f01f1e).
  Consensus: X1 R-55 paper timing run first (all 4); X2 dither dose-response = C1 mechanism figure (referee #2,
  editor #1); X3 design-misspecification stress test (referee #1; where PMRT's known-design assumption breaks).
  Guardrails (auditor): declare before launch and before reading EVAL; fresh seeds 3_200_000+; own dirs; report all.
- Rejected: S 40 -> 100 seed extension (economist #3; auditor: S creep, worst harm); extra kappa sweep.
- Deferred to you: Exp B effect injection on E6 (referee #3), pmrt_nl recall-gap diagnostic (editor #3; reads as
  post-hoc tuning of the primary), Exp B method-RNG replication (auditor #2), zero-compute bridge figures (editor #2,
  paper phase: Study A recall-vs-n next to Exp B slices; placebo-p QQ adjusted vs unadjusted).
- Panel capacity estimate: EVAL ~1.75x faster than projected; Kaggle frees ~01:00 UTC, VPS ~02:10; ~160
  Kaggle-equivalent CPU-h usable before 09:00.

## 2026-10-05 early morning: EVAL done
- EVAL xm-eval-a: 96 parts (Kaggle k1-k14, VPS v1-v6) 15:10 -> 06:52 UTC; merged 150 960 / 150 960 ok, 0 missing /
  mixed / infeasible / N1 conflicts; single commit 5a95186, protocol c5f7a4fe (= frozen), spec file a02fd148
  (= feat/v2), python 3.12.14, one lock, fork+rlimit, budget 7200, one cost table (db644e0).
- Supplementary (R-60): X1 runtime table done on the reference Intel host (b278d71); X3 4560/4560 (VPS); X2 specs
  d05/d20 on VPS, d10/d02/b05/b80 on Kaggle (all launched; report pending). Freed VPS used for X2 (one platform
  per spec).
- Frozen analysis GO to xm-harness (eval_analysis + eval_report, FREEZE_NOTE sha check first).
- Ruling (no panel; the protocol prescribes the path): EVAL records carry commit 5a95186, not freeze 93856c2.
  Diff verified launcher-only (xm_dispatch.py job accounting + pull glob; run cost table; stray _ref .pyc);
  campaign / methods / analysis / spec byte-identical. PROTOCOL_A s.11 allows records "from the freeze commit or an
  amendment listing its key" -> Amendment A-1 (all 150 960 keys), disclosed in the report. Rejected: reporting
  PROVISIONAL for a launcher-only deviation; and passing --freeze-commit 5a95186 (would hide a manifest-file edit).

## 2026-10-05 morning: EVAL verdicts + paper files
- EVAL report FINAL, claim SUPPORTED (6f04f02): C1 5/6 design-blind fail (4/5 w/o MSCR); C2a PMRT-GBM null .045,
  placebo .048 (also Lin, R3 valid); C2b pcorr_eq .048, granger_eq .050 (+ R-56 sensitivity with rcot2_eq: SUPPORTED);
  C3 PMRT-GBM valid, pcorr_eq too (not specific to PMRT). Disclosed: rcot2_eq named; PMRT-Lin NOT SUPPORTED in C3
  (3 INVALID vs F_max 2); MSCR invalid; A-1.
- 3-reviewer naming panel -> PAPER_TERMS_DECISIONS.md D1-D7 (1f710b9). Applied by xm-harness to TERMINOLOGY.md
  (211 -> 296 lines) and WRITING_AGENT_PROMPT.md (51 -> 67), all tagged [PROVISIONAL 2026-10-05]; backups *.bak.
  For you: D5 CDL [DECIDE] (TERMINOLOGY s.10 conflict); names PMRT (frozen E6-P) / PMRT-GBM / PMRT-Lin; Study 4 main
  section + bridging analysis; pmrt_nl_eq low E6 recall reported openly.

## 2026-10-06 night: diagnostics, figures Q8, plan updates (user asleep; autonomous)
- Merged before sleep: X4 random-defer referee (0fe4eb3), X7 subset-equivalence (befcb82). Backups on C:
  (C:/Users/bishal/backups-cdd, SHA256SUMS.txt; user uploads to Drive). Worktrees cdl-ref, xm-audcit, xm-citests,
  xm-eval removed (D: 15 -> 31 GB free; local data backed up in backups-cdd/2026-10-06-worktrees; tag xm-eval-launch).
- Repo history cleanup (option 3): user deferred until all experiments are merged. No rewrite tonight.
- Running: X5 (B) VPS, X6 (C) Colab (cdl, diag.md).
- Started: Q8 global method-colour mapping (aud1, figures5.md; proposal -> panel review -> apply + rebuild Study 4).
- Started: PAPER_PLAN additions (A-D, X4, X7, ~9k word budget) + X4 cross-platform sentence fix (xm-harness).
- Cleanup trial on a mirror copy: pack 117.5 -> 18.2 MiB; command + steps in C:/Users/bishal/backups-cdd/CLEANUP_PLAN.md (not run on the real repo).
- PAPER_PLAN overnight additions done (xm-harness): A answered (matches advisor), B/C pending X5/X6, D -> X7; X4/X7 evidence rows; 9k word budget (flag: s.2 page targets sum ~9.6-9.9k words). 2 new [DECIDE]: name for random-deferral referee; X4/X7 tables in App B.4 vs App C. X4 cross-platform sentence softened (merged a23b8c8).
- Q8 colour mapping: aud1 proposal (figures5.md) -> 3-reviewer panel (design/accessibility, paper consistency,
  skeptical reader): 3x APPROVE-WITH-CHANGES. Key decisions (yours to review):
  - Frozen PMRT = navy hero in Studies 1-3 only; next to PMRT-GBM/Lin (Study 4 recall f, bridging) it is an INK
    reference line (2 of 3; guardrail: Study 4 must not look like it validates frozen PMRT). Dissent: navy dotted.
  - PMRT-GBM own filled navy glyph; PMRT-Lin light navy open; exactly one hero per figure, never "proposed" in Study 4.
  - Validity (a) points stay ink; DESIGN s.12 + captions: hue = "uses the design", not "valid".
  - Oracle open ink star 60 %, random map '+', CS vs MG non-hue cue, variants sharing glyph only in labelled-row
    figures (tested), clay/teal never co-plotted (tested), design-blind bars dashed edge.
  aud1 applying + rebuilding Study 4 (backups first).
- Q8 applied + verified (independent check PASS a-h; figures tests 293 passed / 7 skipped; DESIGN.md.bak2, figures/_bak_2026-10-06). Note: the 2 design-adjusted tests now share teal (markers separate them). Minor polish sent to aud1 (clipped markers, touching crosses, whisker-through-triangle, merged bands env_e4, placebo node vs sand).
- Q8 polish done (327 passed / 7 skipped; new checks for clipped markers etc.). aud1 compacted, idle. Study 4 figures final pending your visual review (figures/preview/).
- X6 (advisor C) DONE: PARTIAL. Cause confirmed = centring on the told law (told/told .079 INVALID; true centring
  .038 VALID; redraw law irrelevant). GBM U-shaped in told width (x.5 .156, x.8 .083, x1-1.5 valid, x2 .079);
  Lin monotone (narrow inflates, wide conservative). The declared first-order bias term failed (P4c), so the exact
  pathway stays open. High confidence for "centring", low for a quantitative mechanism.
- X5 (advisor B) primary DONE: CHANCE. Fresh 2000 datasets/cell: .052 / .054 / .051 (CP upper <= .065, Holm p .66).
  C3 verdict stands. Pending: x5_adj (descriptive, ETA ~01:00Z) + Linux dataset-hash provenance on VPS.
  Report: CDD-ORAN-wt/xm-classic scratchpad/xmethod/results/extras/DIAG_REPORT.md (merge when READY).
- 01:xxZ X5/X6 MERGED (0e8d4aa; EXTRAS_PROTOCOL conflict X7 vs X5/X6 sections resolved by keeping both).
  x5_adj: all 20 adjacent rates valid (.045-.058); Linux provenance 80/80 hashes equal; EVAL's heavy n24000 tail
  does not recur on fresh data. B answer = chance (high confidence: no excess > ~.015).
- ALL experiments now merged (X1-X7). Fresh post-experiment bundle: C:/Users/bishal/backups-cdd/2026-10-06-postexp/
  (upload to Drive too). History cleanup NOT run: waits for your go (CLEANUP_PLAN.md; you force-push).
- xm-harness adding final B/C wording to PAPER_PLAN (bak3).
- PAPER_PLAN: final B/C answers (X5 in 7.5, X6 Discussion P4a, App C.4/C.5), Q8 pointer to DESIGN s.12; status merged. Open [DECIDE] for you: random-deferral referee name; X4/X7 tables App B.4 vs C; xApp/KPI colour slots; word budget proposal; conf. BibTeX.
- 2026-10-06 day: xApp/KPI colour slots fixed (DESIGN s.12.2; panel 3x approve-with-changes, 11 changes applied; KPI row PROVISIONAL; Coverage out of system fig; TERMINOLOGY s.11 synced). Tests 419 passed / 7 skipped (verified); Study 4 figures byte-identical. Backups DESIGN.md.bak3, TERMINOLOGY.md.bak5, figures/_bak_2026-10-06b. Conf BibTeX added (dahal2026cddoran; Gautam name order corrected).
- 2026-10-06: USER: palette inconsistency (figures used muted METHODS set, specimens the bright ENTITIES set). Decided: ONE palette = specimen set (ENTITIES) for all figures; validity heat strip diverging centred at .05; rebuild everything in one go (aud1, figures7.md).
