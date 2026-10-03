# Review C: PROTOCOL_A reproducibility / research-engineering (read-only)

Reviewed: docs/xmethod/PROTOCOL_A.md (DRAFT), status/protocol.md, CONTRACT.md, eval_analysis.py, its tests (re-run:
10 pass, 2 skip on feat/v2), specs/eval/full.json, SEED_REGISTRY.json, xm/dev-runs campaign.py, dev_power.py,
specs/dev/full.json, status/dev-runs.md. Verdict: the design is sound and eval_analysis follows sections 8-10 closely,
but the EVAL run CANNOT yet be executed exactly as pre-registered. Seven must-fix items (section 5).

## 1. Executability as pre-registered

Seeds. `campaign._seed_list` rejects everything outside runner.DEV_SEEDS (F4), and "TBD"/"TBD_HALF" crash `int()`.
The EVAL mode must accept DEV seeds in tune blocks and only the registered EVAL range in measure blocks. Range
notation `[lo, hi]` works only when hi > lo + 1, so a 2-seed range is read as a list. Harmless, but worth a test.

EVAL-mode guard (section 11). The protocol sha alone is not enough, and keying it to the "frozen file" clashes with
section 14: any amendment appended to PROTOCOL_A changes its sha, so later error re-runs and bug-fix re-runs would be
refused. Key the guard to a constant stored in the spec (`protocol_sha256` = sha of the git blob at the freeze
commit). Also check `FROZEN: yes`, and stamp protocol_sha256 + spec_sha256 + python version into every record. Use
the git blob content for the sha, not the working-tree bytes: Windows autocrlf makes "LF-normalised" ambiguous.

Code pinning. Records get `code` from XM_CODE_COMMIT (dirty None), else git, else the Kaggle MANIFEST. The manifest's
dirty check covers only `cdd_oran/` and `scratchpad/e6_dev/`, so a modified spec or eval script in the bundle is not
detected. eval_analysis treats dirty None as clean (`not dirty` lists only truthy values), but the protocol says
"dirty false". Also, the freeze commit must CONTAIN campaign.py + dev_power.py (today only on xm/dev-runs) and every
adapter branch (citests, classic, pmrt). The protocol never says this.

Package versions (largest gap). `_setup_cmd` pins only causal-learn and xgboost. `campaign kaggle` defaults to
`--pin off`, so numpy, scipy, statsmodels, shap, torch, scikit-learn come from whatever Kaggle image is live.
tigramite, dcor and momentchi2 (the CI-test arms) are not installed at all, which means error units or unknown
versions. R-16 allows pip in kernels only when pinned to uv.lock. The `pkgs` stamp is recorded but never checked.

Kaggle/Colab nondeterminism.
- Threads are 1 (OMP/OPENBLAS/MKL). Data sha agreement is checked, method outputs are not. BLAS dispatch
  (AVX2 vs AVX-512), torch and xgboost builds can move borderline p and BC stopping points.
- The budget is not host-invariant. The "Kaggle Xeon 2.2 GHz" reference CPU is not enforced, and merge ranks
  ok > infeasible. A unit that timed out on a slow host therefore becomes ok when re-run on a fast one.
- `run_units` propagates infeasibility to larger n only within one part. Parts are cost-balanced, so which units
  are "not run" depends on the partition.
- A child killed by SIGKILL (OOM killer) is recorded as "exceeded the budget".

Tune-seed re-run vs reuse. Re-run keys are identical to the DEV tune keys (same arm names and seeds). A merge glob
that picks up DEV shards silently replaces records ("later wins"). eval_analysis would then go PROVISIONAL on
commits, but only if the merge actually mixed commits.

Merge. Two ok records of one key from different commits are not flagged. Conflicts are flagged only when the
declarations differ on the same dataset hash.

## 2. eval_analysis.py vs sections 8-10 (spot checks)

Correct:
- BY step-up matches statsmodels fdr_by. Using the adapter's m (`by_family_m`) is right for F1.
- Flag = ABOVE if any of null_raw / plac_raw / null_decl / plac_decl has bootstrap LB > .05 (tau arms: decl only).
  few_seeds below 10.
- Clusters are sorted by seed. Pooled rates cluster on (world, seed).
- C1 set D = native arms declaring "by" = corr, granger_native, CI-test natives. pc_* are tau arms, so this matches
  the section 10 list. The failure rule (2*n2 >= |R2|, n1 <= F_max(|R1|)) and the SUPPORTED / PARTIAL / NOT rule
  match.
- C2/C3 (a) and (b) match. PARTIAL re-runs the full test with the n = 500 cells removed (state that reading in the
  text). C3 adds conf_raw.
- tau comes from tune-role records of the same (arm, cell) only. Untuned, infeasible and incomplete cells are
  excluded from verdicts.

Discrepancies and bugs:
- a) Section 10's "F_max = 95 % quantile of Binomial(N, .025)" understates chance flags. One flag is the union of
  2 roughly independent raw-rate tests, each with about .025 chance of LB > .05 at the exact level, so the per-cell
  chance is about .05. For C2 (N = 50): F_max(.025) = 3, while P(Bin(50, .05) > 3) is about .24. An exactly
  nominal pmrt would fail C2(a) roughly one time in four. Fix: use p = .05, or the per-cell flag rate pmrt_eq
  actually shows on DEV R1 (pre-specified). The section 8 text ".025" needs the same fix.
- b) Unexpected keys are not dropped. build_cells uses every record of a spec arm, so stray DEV measure records
  (3_000_100+) would enter the cells. FINAL ignores n_unexpected, and record role is never checked against the
  expected role.
- c) Errors vs FINAL. FINAL requires zero error records, but section 11 says a unit that fails every re-run is
  reported and its cell marked incomplete. Under that rule the study can never be FINAL. Section 14 bug-fix
  re-runs under a new commit fail `commits == [freeze]` the same way. Allow an amendment-listed commit set.
- d) Candidate completeness. Denominators count only the edges present in `r["edges"]`. An adapter that omits a
  candidate (score.py already computes `n_missing`) silently shrinks the rate. V11 should check
  edges == truth candidate set per record.
- e) `max_n` (T3) removes units from expand(). Those cells produce no rows at all, not "infeasible at this n
  (measured cost X)" as sections 4/7 promise, and they silently shrink the C1/C2 denominators. eval_analysis must
  synthesize infeasible rows from the spec's max_n plus the DEV cost.
- f) Not-applicable (R-25, pmrt R4, P_placebo_conf) candidates have p None. They leave the raw denominators but
  stay in the declaration denominators as "not declared". Section 8 only says that not-testable candidates leave
  the denominator. R4 is outside the verdicts, but V6 rates depend on this choice. State it.
- g) FINAL trusts `--freeze-commit` as typed. It does not check that the commit's PROTOCOL_A says FROZEN: yes or
  that the three recorded shas match. `pkgs` uniformity is not checked either.
- h) F3 is real. campaign.summarise_cell computes null_decl / plac_decl over n_null (not-testable candidates
  included), so DEV flags differ from EVAL flags whenever anything is not testable. T1 consumes the campaign's
  valid_flag, and T5 reports the campaign numbers. Either fix campaign.aggregate, or compute T1/T5 by running
  eval_analysis.build_cells on the DEV merged records, so that one definition exists.

Minor: V3 includes native CI arms and pmrt_r3 (beyond A2 "eq + native-only"). Label them or filter. In seed_row,
`by_edge[k]` raises KeyError if a true edge is missing (ties to d).

## 3. TBD register (section 13): mechanical?

- T1 is nearly mechanical. Still open:
  - which DEV aggregate (path + sha), and whether the CI-test arms' separate DEV run enters the focal-pair max
    (more arms means a larger S);
  - campaign-flag vs eval-flag validity (2h).
  "Minimum detectable gap at S = 100 per cell" is promised but `t1_seed_count` does not compute it. The max over
  many noisy sd_d estimates (20 seeds in R1, 60 in R2) biases S upward; acceptable but disclose.
- T3 is NOT mechanical:
  - "per dataset" max over which worlds and regimes? `max_n` is per arm, not per world.
  - It does not say what applies at an n where an arm (CI tests) has no DEV cost; extrapolate or pilot.
  - It does not say which host (DEV was Kaggle; cmi_knn may use a GPU / Colab).
  - It does not say how T3 interacts with EVAL-time RLIMIT kills.
- T4 is mostly mechanical. It needs the hand-back and audit file paths per arm, and an explicit ruling on whether
  rcot2's liberal F4 null (.084 / .065) counts as PASS-WITH-NOTES.
- T5 is fine (report only), once the aggregator definition is fixed (2h).
- Missing from the register, though also TBD before the freeze: the EVAL-mode code, the merged-branch list for the
  freeze commit, the pinned environment, and the S cap given CI-test cost.

## 4. Drafter's questions

- Q1 (exclude E4 from T1): AGREE. Per-seed recall is Bernoulli, and the paired-t power approximation is poor there.
  Also mark the E4 rows of V3 as descriptive, outside any power statement.
- Q2 (re-run vs reuse tune): RE-RUN, as drafted. Reasons:
  - FINAL requires a single commit;
  - DEV ran on an earlier commit (the pilot was pre-fix-classic2) with unpinned Kaggle packages;
  - tau for the primary tau arms must come from the same code as the measurements.
  Bonus: the re-run datasets have the same dataset sha as the DEV tune records, so comparing EVAL tune
  declarations with DEV is a free reproducibility check (report it in V11). If compute is tight, the p-arm tune
  units only serve B2: restrict B2 (e.g. n <= 4000) before the freeze rather than reuse.
- Q3 (who adds the EVAL mode): the campaign.py owner (dev-runs), under the orchestrator, merged INTO the freeze
  commit, with tests and an independent audit before the freeze. The spec: constant protocol sha in the spec plus
  a FROZEN check; measure seeds only in the registered EVAL block, tune seeds only DEV; stamp the shas into records;
  refuse when not clean; pin packages; never re-run an infeasible unit.
- Q4 (S floor 40 / cap 100): floor 40 is OK; it is justified by the null-rate precision. Do NOT confirm cap 100
  until `campaign project` is run with the CI-test pilot costs: pdcor/cmi_knn with B 9999 at n 24000 could
  dominate. Set the cap from the compute budget and record the projection in the freeze commit. Implement the
  minimum-detectable-gap output.
- Q5 (C1-C3 rules):
  - ">= half R2 ABOVE" and "PARTIAL if only n 500 fails" are reasonable.
  - Change F_max's p (2a).
  - "Any infeasible unit makes the cell infeasible" is OK only if infeasibility is host-invariant: define it at
    (arm, n) from T3, with EVAL RLIMIT kills reported but not re-run, and with the cells shown in V1 (2e).
  - Decide how persistent errors coexist with FINAL (2c).
  - C1 depends on T4's admitted set (rcot2_native's liberal null makes it ABOVE in R1 too, so it is "not a
    failure"). Say so.

## 5. Must-fix before freeze

1. Pinned environment. Install the full uv.lock export (hashes) in every Kaggle/Colab kernel, including tigramite,
   dcor, momentchi2, shap, torch, numpy, scipy. eval_analysis must require one `pkgs` set == lock for FINAL.
2. EVAL mode in campaign.py: constant protocol sha in the spec, a FROZEN check, the seed-block rules, and shas +
   python version stamped per record. Make the MANIFEST dirty check cover the whole repo, or refuse when the
   source is env/None. Merge it, plus all adapter branches and dev_power.py, into the freeze commit, and list
   them in section 11.
3. Fix F_max (p ~ .05, or a DEV-calibrated per-cell flag rate) in sections 8/10 and in eval_analysis.
4. eval_analysis integrity:
   - drop unexpected keys and require 0 for FINAL;
   - check role vs expected;
   - check per-record candidate completeness;
   - treat dirty None as not clean;
   - verify the freeze commit's FROZEN line and file shas;
   - emit T3 `max_n` cells as infeasible rows.
5. One validity definition (F3): fix campaign.aggregate's decl denominators, or compute T1/T5 through
   eval_analysis on the DEV records.
6. Infeasibility and error policy:
   - host-invariant budget (T3 at (arm, n); no re-run of infeasible units; disable cross-n propagation or make it
     partition-independent);
   - OOM kills recorded as error, not infeasible;
   - FINAL allows amendment-listed commits and "persistent error" cells per section 11 and 14.
7. Complete the T1/T3/T4 rules: name the DEV aggregate files and shas and the arm set for T1; fix T3's
   world/regime scope, extrapolation and host; give T4's verdict sources and the rcot2 ruling. Separate the EVAL
   output directory from DEV shards (key collision on the tune seeds).

Recommended, not blocking: re-run 1-2 % of EVAL units on a second host and report declaration agreement (merge
"conflicts" already detects it); record CPU flags; register the EVAL block at its exact S in SEED_REGISTRY.
