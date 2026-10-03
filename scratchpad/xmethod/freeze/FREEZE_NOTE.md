# Study A freeze note (TEMPLATE: filled in the freeze commit; PROTOCOL_A freeze procedure step 4)

Freeze date: <YYYY-MM-DD>. Base: feat/v2 <commit before the freeze>. EVAL block 3_100_000-3_100_299 (registered by
the orchestrator after this commit). Every sha256 below is over the file with LF line endings.

## Frozen files
| file | LF sha256 |
|---|---|
| docs/xmethod/PROTOCOL_A.md (= spec protocol_sha256) | <sha> |
| scratchpad/xmethod/specs/eval/full.json | <sha> |
| scratchpad/xmethod/eval_analysis.py | <sha> |
| cdd_oran/xmethod/campaign.py | <sha> |
| cdd_oran/xmethod/dev_power.py | <sha> |
| scratchpad/xmethod/results/fmax_sim/dependence.json (= spec fmax_dependence_sha256) | <sha> |
| uv.lock | <sha> |

Adapters (T7): cdd_oran/xmethod/methods/{pmrt_core, pmrt_nl, pc, granger, corr, notears, shap_dag, two_tower, cdl,
_cdl_model, mscr, pcorr, pcorr_hac, rcot2, shared modules}.py, runner / api / score / covariates / worlds: in this
commit, unchanged after it.

## T-register inputs
| item | input | LF sha256 |
|---|---|---|
| T1 | results/dev/full/merged.jsonl.gz, results/dev/ci_c/merged.jsonl.gz, pmrt_nl_eq DEV merged, cdl DEV merged; aud1 output | <sha each> |
| T3 | DEV cost maxima (eval_projection.json t3_max_cpu_s_by_arm_n, re-run), R-55 calibration block records (f, e by host type), pmrt_nl_eq cost pilot records | <sha each> |
| T5 | results/dev/{full, ci_c, cdl, pmrt_nl}/REPORT.md | <sha each> |
| T9 | results/fmax_sim/s_e4_rule.json | <sha> |
| T10 | results/pmrt_nl/SELECTION.md | <sha> |
| estimand check (R-36) | results/estimand_check.json (re-run at this tree) | <sha> |

## Values (copied from PROTOCOL_A section 13)
S = <T1>; S cap 60 (T6, R-56); S_E4 300 (T9); T3 infeasible: <none / list>; statistic gbm (T10); mscr n <= 1000 (T11).
Units: <n> (tune <n>), datasets <n> (spec, eval_analysis.planned_units).
