# Study A freeze note (PROTOCOL_A freeze procedure step 4)

Freeze date: 2026-10-04. Base: feat/v2 5a4ef0f (final T1 41d6c37, R-55 calibration block 12748be, VPS smoke 9e91a4c,
R-59 audit fixes 0542a99, re-audit 5a4ef0f) + xm/vps-factors 229decf + xm/anchor-note 63e3c6e + the R-59 F1 fix
853392e / 581317a; branch xm/freeze (worker `protocol`). EVAL block 3_100_000-3_100_299 claimed for E1-E5 in
docs/benchmark/SEED_REGISTRY.json in this commit (orchestrator instruction). Every sha256 below is over the file with
LF line endings; all were recomputed after the R-59 fixes (C4-C6 re-run).

## Frozen files
| file | LF sha256 |
|---|---|
| docs/xmethod/PROTOCOL_A.md (= spec protocol_sha256) | c5f7a4fe4dff4bfdeef3ccebac2dcc3748dfa9c2d0c3d17f2ac9d6246379fa22 |
| scratchpad/xmethod/specs/eval/full.json | a02fd14833bb3a51bfb4136e78653b3b5c867bf0f494f344b3ad8abababdcc07 |
| scratchpad/xmethod/eval_analysis.py | 73685abb3e539a3125ead531ab63555c0292c6ef04bfd969ac9182035b8278df |
| scratchpad/xmethod/eval_report.py | 8099ca4a43afaa2791c275a45093f5c771124fdcbcb66c7cd3393b9fdd6cb185 |
| cdd_oran/xmethod/campaign.py | ab87a268ffbce0dc60c87b479ffec9e48dfa6604f5706695313b745189837450 |
| cdd_oran/xmethod/dev_power.py | 32ba4cd3da97d0f34d2e19f3d69dd4e75dca52830934c95d24a8fe8edabd2e8c |
| scratchpad/xmethod/results/fmax_sim/dependence.json (= spec fmax_dependence_sha256) | 402802b15c797f38cf1813aec1b4a912ffdbc3dba2ba3bfe94f1ee7a50d4f73e |
| uv.lock | c71e9a4e05377218c767fc3d13d212608032f61f7774e889fc81b07625fa7e79 |
| pyproject.toml | e7c8ca0748a2f944aa30cbb387970aaab50f5ef821323f951ab8ca488d2887fc |
| docs/benchmark/SEED_REGISTRY.json | 58ba33fc31aafefaef6bd49278a75036c795560780955e3a28b03867d3eb7c43 |

Adapters, runner / api / score / covariates / worlds and launchers (T7): every tracked .py / .json / .yaml / .toml
file under cdd_oran/, configs/ and scratchpad/e6_dev/ (301 files) with its LF sha256 in
`scratchpad/xmethod/freeze/FREEZE_MANIFEST.sha256`; in this commit, unchanged after it. `scratchpad/xmethod/_ref/`
(the eval_analysis / eval spec copies the DEV T1 run used; its provenance) is kept but is neither a frozen file nor a
freeze input (orchestrator, N3).

## Audit of campaign.py (T7 "audited"; PROTOCOL_A s.11)
| report | verdict | LF sha256 |
|---|---|---|
| results/audit/campaign_eval_audit.md (worker exp-b; campaign.py as of 9e91a4c) | FAIL (F1 blocker, F2 ruling, F3-F9 notes) -> CONTRACT R-59 (1524dc2): pre-freeze fixes | 3a7a2c9182538aa7b5483fbf60ae8e0183769210826e2f4a0038dd746b608909 |
| results/audit/campaign_eval_reaudit.md (xm/freeze 581317a: F1 853392e, F2-F9 0542a99) | PASS-WITH-NOTES (F1-F9 fixed, each with a test; notes N1-N6) | 4b477713040146098b11bf979cfd37990054555bd763bb86ed12d414d0da5ff0 |

- R-59 F1 changed PROTOCOL_A s.11 (records stamp `integrity.spec_file_sha256`; orchestrator CONFIRMED) and, after
  the re-audit's N4, the spec's descriptive `tbd.driver` text to match. No other frozen behaviour changed after the
  re-audit.
- Re-audit N1, operational rule (orchestrator): every EVAL relaunch passes `--skip-complete-from` over ALL prior EVAL
  shards of this run (so a cap-infeasible unit is never re-run on another host). Before eval_analysis, list the keys
  that have both an `infeasible` and an `ok` record across the raw shards, and report them (none expected).
- Re-audit N6 (test hygiene, not frozen behaviour): the two launch-command tests need the generated, gitignored lock
  export (`campaign lock`) in a fresh checkout.
- Re-audit N2, N3, N5: INFO. N5's hashes are superseded by the table above; only the spec differs, because of the N4
  text fix.

## T-register inputs
| item | input | LF sha256 |
|---|---|---|
| T1 | results/dev/full/merged.jsonl.gz | 6a74cb85e3ebb1f6e49c904a045c015e72b6f4c6deae66e79363345b2dc76f2a |
| T1 | results/dev/ci_c/merged.jsonl.gz | 1b0484798f5dfd2f765656c07efce384737e6c981957c9fcd918266352f4d350 |
| T1 | results/dev/pmrt_nl/merged.jsonl.gz | d8894fd3a1dee29f359a0ee3d16f653c8f1c6be28a8fb023e30f47a6c36f71d7 |
| T1 | results/dev/cdl/merged.jsonl.gz | be57d2421826185d43d126a6ee7b04b527e8414865090cac0655213f16f35630 |
| T1 | results/dev/final/t1.json | 079b18781829ad0c27c4a09a2b476b30e14e89f708920ea74df329ccdc7744e5 |
| T1 | results/dev/final/t1_pairs.json | 773a0882895715a341f1e4118d55e0a81f7f3907560223228b8d1f84757610c2 |
| T1 | results/dev/final/notes.txt | dcc506b42a1cbbebb551265ca21c52b71a028df531593d12064bba0e64ccd30c |
| T1 / T5 | results/dev/final/dev_cells.json | 6bbc16a5dd2d3c99e174766836427d89fb6e3b0ad44d4f87622037e7d6963e35 |
| T5 | results/dev/final/REPORT.md | 9e0c42bfefc66bdcadc46be03b5952ed82496698d41fe4fd1d2de2c66f09a16c |
| T5 | results/dev/full/REPORT.md | d56990e8f563f41e2d1b51e3a55edeece02bf892b5e6bca2fa04e696333bf216 |
| T5 | results/dev/ci_c/REPORT.md | 3ebb37500a8e349392d6bfc63254867653f83f9e1131cc7cc274de9148a9bc44 |
| T3 | results/dev/pmrt_nl_cost/merged_e2.jsonl.gz | 4300cd01dd8808a04ba92e209842a08df91fc3aef90a2ca293a296dfe6432abf |
| T3 | results/dev/pmrt_nl_cost/merged_e4r3.jsonl.gz | 6ff0389b97451d8da18d0e972b92c2ec31bc7201cef875409eba86d35e53b145 |
| T3 | freeze/pmrt_nl_cost_pilot.json | 4d8a36b138e1721da90ad9c322f2d4bae68349959887efe47ffc5f33a4363909 |
| T3 / T6 | freeze/eval_projection.json (S 40, R-55 factors) | 72a8ef01a5947bb3b6e298c5b212b13bf35e35f1f7850736ea284ac89efc8167 |
| T3 | R-55 calibration block: results/exp_c/calib/factors.json (xm-citests; feat/v2 12748be = xm/exp-c aee2fab, same sha) | f86cb61cf0b7fece4e666c02293d68425001e3415e4723b429c813b5e89518ce |
| T3 | its records (raw/xm-expc-cal-{k1, v2, c14, c16, v1}, kaggle_ref / vps / dev exp_c_runtime.json): 36 files with LF sha256 in freeze/FREEZE_CALIB.sha256 | per file |
| T9 | results/fmax_sim/s_e4_rule.json | ca9a28fc6189f37734849831dc6fe6ca5a464b2fe43587771abcdb247723b838 |
| T10 | results/pmrt_nl/SELECTION.md | 34dc469cb8bfab5d5cab5ebfd371faf68f2bc673daadce6f25ea30a4f7a4886b |
| estimand check (R-36) | results/estimand_check.json (re-run at this tree: all 18 cells primary PASS) | 03b03b1e9cbf7db76fe5a0c7c6616f1b9a598af7fce951c9d86ac605a827fb9d |

Paths without a prefix are under scratchpad/xmethod/.

## Values (copied from PROTOCOL_A section 13)
S = 40 (T1: S_power 14, floor 40; 115 pairs; min power .999); S cap 60 (T6, R-56); S_E4 300 (T9); T3 infeasible:
none (max converted pmrt_nl_eq 1148 CPU-s at n 24000; f / e: Colab Xeon 1.067 / .101, VPS EPYC-Rome 2.562 / .253;
uncalibrated Kaggle EPYC 7B12, Lightning 8488C, Colab EPYC 7B12 provisional, >= 26.8x margin); statistic gbm (T10);
mscr n <= 1000 (T11). pkgs_lock = uv.lock versions of campaign.PKGS, torch '2.10.0+cpu' (N1, orchestrator: the
locked public version 2.10.0 in the CPU build that every DEV and calibration record stamps on Kaggle, Colab and the
VPS (R-35 torch-build amendment; no GPU arm); eval_analysis requires the records' pkgs to equal pkgs_lock exactly,
so the build label is part of it).
Units: 150 960 (tune 40 520, measure 110 440), datasets 11 000 (`campaign list`, mode eval; eval_analysis.planned_units).

## Rulings at the freeze (orchestrator, 2026-10-04)
- N2 uncalibrated host types: DEV costs from Kaggle AMD EPYC 7B12 (max 268 CPU-s, cdl n 4000), Lightning Xeon
  Platinum 8488C (207) and Colab AMD EPYC 7B12 (121) are provisional T3 reads as PROTOCOL_A T3 says; each is over
  budget only if f > 26.8 / 34.7 / 59.4. Accepted. xm-citests may add an EPYC 7B12 factor later if Kaggle hands one
  out in EVAL, as descriptive only.
- Launch: the lane option (FREEZE_CHECKLIST A) is not needed; the fallback of 4 parts per Kaggle session is accepted.
- VPS: live smoke PASSED (feat/v2 9e91a4c: Python 3.12.14, cpu_quota 7, pulled, nothing left running).
