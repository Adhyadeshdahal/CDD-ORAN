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
## D5 + naming (user decision, 2026-10-05; tag [DECIDED 2026-10-05, user])
- TERMINOLOGY.md (296 -> 310 lines):
  - "How to use": CDD-ORAN is the repository's code name only.
  - New "Paper identity" table: journal framework name and title = [TBD: journal framework name and title, team
    decision], a new contribution that builds on and cites CDD O-RAN, never "CDD-ORAN" or an extension; conference
    paper and method = "CDD O-RAN" (title checked in archive/xApp-Nana-Conference-2026/main.tex).
  - s.2: "conference version" -> "conference paper (CDD O-RAN)".
  - s.6.5: CDL -> CDD O-RAN discovery with the first-use form; "CDL" / `cdl` added to never-in-prose.
  - s.10: [DECIDE] resolved. CDL row -> "the causal-discovery stage of CDD O-RAN [conf. ref], re-implemented here",
    then "CDD O-RAN discovery". MLP world model / CMI discovery only for the conference paper. New row bans
    "CDD-ORAN" as the paper name and "extension".
  - s.12: D5 wording uses the new names. The fixed-threshold sentence is replaced by the REPORT V0 facts: no edge at
    n 500, true edges in 3 of 10 cells at n 1000, no truth-null or placebo declaration. Never "CDD O-RAN is invalid".
- WRITING_AGENT_PROMPT.md (67 -> 72 lines): the journal-guide line no longer says "conference extension". New first
  evidence-boundary bullet: title / framework [TBD], new contribution citing CDD O-RAN, never "CDD-ORAN" /
  extension, never "CDL".
## Study 4 layout (user GO, 2026-10-05; tag [DECIDED 2026-10-05, user])
- TERMINOLOGY.md s.9: the D2 "~1 page" paragraph is replaced by the decided layout.
  - After Studies 1-3, framed "why the design-based test is the right instrument", ~1.5-2 pages.
  - Sections 4.1 (purpose + pre-registration, C1-C3 up front; "pre-registered" only with freeze hashes, s.8), 4.2
    (design), 4.3 (E4 worked example), 4.4 (results by claim), 4.5 (verdict grid + limits, the 3 named exceptions),
    4.6 (bridging; tables and cost in Appendix B).
  - Appendix A: one page per environment, fixed template; recall-vs-n, kappa sweep and R4 rows there.
- WRITING_AGENT_PROMPT.md: the Study 4 bullet's "~1 page" layout is replaced by a one-line pointer to TERMINOLOGY s.9.
## Journal title (user decision, 2026-10-05; tag [DECIDED 2026-10-05, user])
- TERMINOLOGY.md "Paper identity":
  - The [TBD] row is replaced by: title "Design-Based Causal Maps for xApp Conflict Mitigation in O-RAN".
  - It notes that the abstract's first sentences must define "design-based" as "valid because the actions in the logs
    were randomized by a known logging design (design-based inference)".
  - Running title: [DECIDE], proposal "Design-Based Causal Maps for xApp Conflicts" (not set).
  - New row: framework name = none (no acronym); "PMRT" remains the method name.
  - The s.10 "CDD-ORAN / extension" row now points to the title instead of [TBD]. No [TBD] remains in either file.
- WRITING_AGENT_PROMPT.md: title line set to the decided title, no framework name, pointer for the definition and the
  running title. Manuscript .tex not touched.
