# Reviewer C (journal reviewer / publication strategist)

## 1. Recommendation
Run one last disclosed, pre-registered follow-up: option 2b in a principled form, then write, whatever the result.
Call it a **power-aware referee**. MapGate allows an action only if, for every guard KPI (RLF), the map has a tested
edge whose bound clears the guard, or PMRT's own predicted minimum detectable effect (MDE) for that untested edge is
below the guard margin. Otherwise it defers. PMRT is a *predictable* matched filter, so it can report the MDE for each
edge. Correlation, Granger, SHAP and two-tower cannot do this. Apply the same rule to every map arm (GT, corr, granger,
granger_by, two_tower, shap) so the comparison stays fair. Reuse the frozen freeze-2 maps and discovery data. Run on
fresh eval seeds only. Do not run 2a or 2d. Rare-event neighbour RLF is limited by data, and a new statistic is
open-ended method work that reviewers will read as tuning on known failures.

Start writing now, in parallel. `sn-article.tex` is still a template skeleton (placeholder abstract, about 450
words). The manuscript is the critical path, not the experiments.

## 2. Skeptical reviewer view
- **Now (report as is):** "The pre-registered primary endpoint failed. The method buys SLA gains by raising radio-link
  failures 1.31x. The +0.47 R is irrelevant under your own guard. The discovery results are clean, but the paper's
  claim about preventing SLA violations is not supported." This is publishable in Discover as an honest negative, but
  it weakens the user's goal of showing a clear end-task edge.
- **After 2b (eligible, R > never_sleep):** "Good. The safety margin comes from the method's own uncertainty
  quantification, it was frozen before testing, and it was applied uniformly. Is the rule post hoc?" Answer: yes, it
  is disclosed as a follow-up, and option (a) is still reported in full. That is acceptable.
- **After 2b (eligible but collapses to about never_sleep):** "The referee just learned not to sleep." This is still a
  real claim: PMRT is the only map-based arm that is safe and not worse than the best static policy, while associational
  maps are -10.
- **After 2a/2d:** "You changed the discovery statistic after seeing which edges were missed." That is a
  forking-paths objection, it costs more, and the gain is uncertain.

## 3. Effort vs gain
| Option | Kaggle CPU-h | Calendar days | Gain |
|---|---|---|---|
| 1 report as is | 0 | 0 | baseline |
| 2b power-aware referee | ~20 (160 seeds, 3 kernels, ~7 h wall) | ~3 (1 code+freeze, 1 run, 1 analysis) | high: likely eligible, principled |
| 2a RLF filter | ~40-60 (rediscovery + eval) | 6-10 | uncertain, small power gain on rare events |
| 2c / 2d | 60-150 | 10+ | marginal; delays the paper |

## 4. Headline claims and structure
Claims, written in the conditional where a claim depends on the 2b outcome:
1. On randomized logs, PMRT recovers indirect conflicts (pooled F1 0.95, sign accuracy 1.00, 0 placebo edges).
   granger_by declares 13 placebo edges.
2. On confounded incumbent logs, PMRT gives 18/22 true edges, 0 wrong-sign and 0 placebo edges. Baselines give 6-29
   placebo or wrong-sign edges.
3. Referees built on associational maps are harmful (R -9.8 to -11.5). The PMRT referee has the highest R (+0.47).
4. Pre-registered result: NOT ELIGIBLE, because RLF rises 1.31x when power is low for rare neighbour RLF.
5. (2b) Gating on the method's own MDE restores eligibility while keeping an SLA gain over the best static policy (if
   confirmed).

Structure:
1. Intro
2. Related work (QACM, SHAP-DAG, Djidjev, PACIFISTA, GNN, two-tower)
3. PMRT method, including MDE
4. MapGate referee
5. E6-P simulator and frozen protocols
6. Discovery results (v4, option a)
7. End-task results (a)
8. Follow-up (2b), with the disclosure stated clearly
9. Limitations: rare-event power, simulator only
10. Conclusion

## 5. Risks
- 2b defers so much that R falls below never_sleep. Mitigation: pre-register in advance that this outcome is reported
  as is.
- The MDE computation has to be validated before the freeze, without peeking at eval seeds.
- A reviewer may call this outcome switching. Mitigation: report option (a) first and give 2b its own protocol hash.
- Scope creep. Set a hard stop: one run, then write.
