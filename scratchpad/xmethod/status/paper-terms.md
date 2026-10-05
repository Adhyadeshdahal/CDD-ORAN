READY-TO-MERGE (overnight 2026-10-06: PAPER_PLAN post hoc evidence, X4 note)
# paper-terms (2026-10-06 overnight): PAPER_PLAN additions + X4_REPORT softening
Earlier change log (D1-D7, D5, layout, title, Q1-Q19): this file at ec1d79a (xm/freeze). Files outside the repo live
in D:/academia/major-project/ (no git there). .bak files untouched; TERMINOLOGY / WRITING_AGENT_PROMPT / .tex not edited.

## 1. PAPER_PLAN.md (backup PAPER_PLAN.md.bak2, sha 54fe89e4..., = pre-edit file)
- Patch: scratchpad n_patch1.py (exact-match inserts, counts asserted). 379 -> 486 lines, ASCII, LF.
- 108 lines added, 1 line changed (B.5 caveat cell, pointer appended). Decided content not rewritten.
- Every addition tagged [NEW 2026-10-06 overnight, review] (16 tags).
- a. Advisor section (s.6, "Science questions"):
  - Source note: the advisor output file (tasks/a8e519e3f6836435d.output) is EMPTY (0 bytes, checked 2026-10-06).
    No advisor wording quoted. The orchestrator session's transcript is not in ~/.claude/projects either.
  - A ANSWERED from the sources (worker reading, [VERIFY] vs the advisor): metric scope. Study 1's 0.90 = indirect
    recall over the 10 neighbour GT-TRUE edges; bridging recall = all 29 GT-TRUE edges, INDET unscored (EXPB s.4-5).
    FD:v4.pooled.loadsp_c+wby1s (= P3_values) ov_precision .92, ov_f1 .852 -> overall recall .79 = EXPB pooled 0.79.
    Appendix B sentence proposed. B.5 row points to it.
  - B "being tested by X5; final wording after DIAG_REPORT" (xm/diag 57cef93 / bc29a66; design summarised).
  - C "being tested by X6; final wording after DIAG_REPORT"; advisor hypothesis as declared in X6 (centring bias c_t).
  - D superseded by X7 (yes, descriptively, this scenario; numbers from X7_REPORT A2-A4).
- b. Claims-to-evidence: new subsection "Post hoc referee checks X4 and X7 (descriptive)" after the X rows:
  - X4.1-X4.4, X7.1-X7.5 with source paths (feat/v2 befcb82), placement (Discussion P2a / Appendix B.4), caveats.
  - Two proposed Discussion sentences (X7 translated from its suggested sentence; X4 one sentence).
  - Name translations: sub:ES+PowerES -> "the static ES + PowerES subset" (TERM s.7 static subsets); "Gate A anchor"
    dropped (TERM s.7/s.10); "E6-P plant" -> "the Study 3 seeds" / scenario; CS:<assoc> -> "certified-safe
    map-driven referees with associational maps"; MG:PMRT -> "the ungated PMRT referee".
  - No TERM term for X4:rand@p -> new s.6 item 20 [DECIDE] (proposal "random-deferral referee").
  - New Discussion bullet P2a (~100 words); new Appendix B.4 + tab_appB_x4 / tab_appB_x7 (TO BUILD; figure list).
  - New s.6 item 21 [DECIDE]: X4 / X7 tables in Appendix B.4 (as briefed) or Appendix C with X1-X3.
- c. Word budget: new table under s.2 "Word budget proposal (about 9 000 words of main text)":
  - Intro 850, Related 650, System 800, Method 1 300, Design 700, Results 2 250, Study 4 1 000, Discussion 900,
    Conclusion 200, reserve 350 = 9 000 (abstract, captions, tables, appendices not counted).
  - Moves / cuts named per section (e.g. 6.1 P4 rename paragraph cut per Q10; criteria to one table; parameters to
    a table / caption; 7.6 ~150 words; X3 detail stays in Appendix C).
  - Arithmetic check flagged: the s.2 page targets sum to 19.25-19.75 pages (~9 600-9 900 words), not 22 pages as
    the earlier note says; the earlier note was left unchanged (decided content).

## 2. CDD-ORAN, branch xm/paper-notes (from feat/v2 befcb82; not pushed)
- X4_REPORT.md, Provenance: "the comparison is not cross-platform" replaced by "share the plant's numerics on these
  provenance seeds" + a dated Note (2026-10-06): X7 found VPS vs Kaggle differ on about 15 % of seeds by a few
  violations, |dR| <= 4e-4 (X7_REPORT.md "Deviation"); comparison is cross-platform; no X4 number changes.
  No number changed.
- This status file. One commit, subject only.

## Open (for review, not blockers)
- Advisor's own report for A-D is missing (empty file); A rests on the worker's source reading, marked [VERIFY].
- s.6 items 20-21 [DECIDE]; word-budget table is a proposal; B / C wait for DIAG_REPORT.
- Still open from before: Q8 colour-slot list and xApp / KPI slots [DECIDE]; [conf. ref] BibTeX.
