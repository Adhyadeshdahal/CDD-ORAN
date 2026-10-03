# Brief: F7 independent fidelity audit (read-only review; you did NOT write the code you audit)
Read: scratchpad/xmethod/CONTRACT.md (rules, rulings sec 8), scratchpad/xmethod/PROTOCOL_NOTES.md, the method's
fidelity doc (docs/xmethod/FIDELITY_*.md or PMRT_CORE.md) and its code under cdd_oran/xmethod/methods/.
For EACH assigned method, compare the implementation with the ORIGINAL paper / reference implementation (papers in
D:/academia/major-project/archive/reference-papers/ where available; otherwise the authors' repository or package
docs; cite what you used). Check: (1) statistic and null match the source (or every deviation is listed in F6 with a
reason); (2) defaults are the authors' (R-13); (3) the contract rules hold (R-2 declarations, R-4 signs, R-6 family,
R-9 resolution, R-10 placebo use, tuning uses no truth); (4) the adapter cannot read Truth / privileged data;
(5) the fidelity-gate evidence (F2-F4) actually supports the claims (rerun a small piece yourself if cheap, locally
under 10 CPU-min, else Kaggle); (6) anything that makes the comparison unfair to this method or to PMRT.
Output: scratchpad/xmethod/audit/<method>.md per method: verdict PASS / PASS-WITH-NOTES / FAIL, findings ranked by
severity with file:line, and a recommended fix. Do not edit method code. Work in your worktree on a NEW branch
xm/audit-<your-name> created from feat/v2 (git switch -c xm/audit-<name> feat/v2 after committing nothing else).
Status: scratchpad/xmethod/status/audit-<your-name>.md (rule 7); end with ## HAND-BACK.
