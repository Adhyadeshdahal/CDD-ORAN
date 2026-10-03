# Brief: protocol (freeze document + EVAL analysis pipeline for Study A)

Read: scratchpad/xmethod/CONTRACT.md (all rulings R-1..R-28), PROTOCOL_NOTES.md, docs/xmethod/*.md,
cdd_oran/xmethod/{api,runner,score,campaign,covariates}.py (campaign.py is on xm/dev-runs; read it with
git show xm/dev-runs:cdd_oran/xmethod/campaign.py), docs/benchmark/SEED_REGISTRY.json, and the frozen protocol of the
earlier v4 study for style (docs/benchmark/ or reports/2026-10-02-v4-pmrt-confounded-certsafe/; find it).

Deliverables (branch xm/protocol from feat/v2):
1. docs/xmethod/PROTOCOL_A.md: the pre-registration for the EVAL run of Study A. Every number fixed or marked
   "TBD-from-DEV" with the exact rule that fills it (seed count from R-12 power calc; validity flags R-20).
   Sections: question and claim (rescoped, CONTRACT end); worlds and regimes (E1-E5 = controlled diagnostic
   environments, each isolating one mechanism; list the mechanism of each); kappa (R-24/R-27); methods table
   (name, family, authors' null, arms, version, fidelity status incl. NOTEARS F3 19/20 disclosed, two_tower =
   adaptation, rcot2 liberal authors' null); arms (R-17/18/19/25/28); declaration rules (R-2, R-6, BY q .05, tau);
   metrics (validity: truth-null and placebo rates with seed-cluster CIs; recall among valid cells R-20; FDP; sign
   accuracy; not-testable counts R-22/23; cost); n grid; budget R-13; seeds (EVAL block 3_100_000+, count TBD by
   power calc; tune seeds = DEV tune seeds); primary vs secondary analyses; exact figures/tables to produce
   (list them; no plotting yet); what counts as support / non-support of the claim; deviations policy. < 250 lines.
2. scratchpad/xmethod/eval_analysis.py: reads merged campaign records (campaign.py format), outputs the exact tables
   listed in the protocol as JSON + markdown, deterministic; test it on the dev-runs pilot records
   (git show xm/dev-runs:scratchpad/xmethod/results/dev/pilot/merged.jsonl.gz or the worktree
   D:/academia/major-project/CDD-ORAN-wt/xm-pmrt). Tests for the analysis code.
3. A draft EVAL spec specs/eval/full.json (seed list placeholder "TBD") using campaign.py's format.
Do not claim or touch EVAL seeds; do not register anything in SEED_REGISTRY. Status scratchpad/xmethod/status/
protocol.md (line 1 READY-TO-MERGE when done; questions as "- Q<n> ..."). Commit locally, never push.
