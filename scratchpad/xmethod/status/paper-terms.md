READY-TO-MERGE
# paper-terms (2026-10-05): PAPER_TERMS_DECISIONS D1-D7 (feat/v2 1f710b9) applied outside the repo, for user review
Files: D:/academia/major-project/TERMINOLOGY.md (211 -> 296 lines), WRITING_AGENT_PROMPT.md (51 -> 67); .bak untouched. Every new or changed entry is tagged [PROVISIONAL 2026-10-05]. Manuscript not touched.
Numbers only from results/eval/REPORT.md (s.1-5, V0, V1) and docs/xmethod/EXP_B.md s.9, cited by section.
## TERMINOLOGY.md
- Header: Study 4 / bridging sources of truth (PROTOCOL_A, EVAL REPORT, EXP_B s.9, the decisions file).
- s.2 (D7): E1-E5 row rewritten from PROTOCOL_A s.2 ("controlled diagnostic environments", Study 4; drops "E4
  scaffolded only" and "open decision"). New row: regimes R1-R4.
- s.6.1 (D1): "PMRT" alone = frozen E6-P test. New table: PMRT-GBM, PMRT-Lin, PMRT-Lin R3 covariates. Mandatory sentence.
- s.6.2: MSCR row says Study 4 design-blind, n <= 1000, reported invalid arm.
- New s.6.5 (D1, D4): families (design-based / design-adjusted / design-blind) with members; method set; dropped
  arms; never-in-prose list.
- s.8: rows for cell validity, F_max, Study 4 verdict labels, amendment A-1.
- s.9 (D2): Study 4 and bridging-analysis rows, main / Appendix A / B placement, never-write list extended.
- s.10 (D6): 8 do-not-use rows. The CDL row is marked [DECIDE] (D5).
- New s.12: D3 claim (exact) + disclosures with numbers, PMRT-GBM E6-P recall sentence (D2), bridging rules (D2),
  D5 CDL wording as [DECIDE].
## WRITING_AGENT_PROMPT.md
- READ FIRST item 7: REPORT, PROTOCOL_A, EXP_B and the TERMINOLOGY sections.
- Evidence boundary: Study 4 and bridging bullets (structure, mandatory sentence); the "do NOT claim" bullet now
  also lists D6; allowed number sources now include REPORT and EXP_B.
## For the user (morning review)
- D5 [DECIDE]: the panel's "rarely declared any edge at n <= 1000" is loose. REPORT V0 at the fixed threshold:
  recall .00 in 10/10 cells at n 500 and 7/10 at n 1000 (E1 R1 .75, E3 R1 .57, E5 R1 .27). Noted under s.12.
- "9 of 30 cells" = 9 INVALID truth-null declaration cells (REPORT V1); all other quoted numbers match.
