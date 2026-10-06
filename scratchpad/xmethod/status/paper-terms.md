READY-TO-MERGE (overnight 2026-10-06: PAPER_PLAN incl. X5/X6, X4 note, Q8 pointer)
# paper-terms (2026-10-06 overnight): PAPER_PLAN additions + X4_REPORT softening
Earlier change log (D1-D7, D5, layout, title, Q1-Q19): this file at ec1d79a (xm/freeze). Files outside the repo live
in D:/academia/major-project/ (no git there). .bak files untouched; TERMINOLOGY / WRITING_AGENT_PROMPT / .tex not edited.

## 1. PAPER_PLAN.md (backup PAPER_PLAN.md.bak2, sha 54fe89e4..., = pre-edit file)
- n_patch1.py: 379 -> 486 lines; +108, 1 changed (B.5 cell appended); decided content not rewritten; 16 [NEW] tags.
- a. Advisor section (s.6, "Science questions"):
  - Source note: the advisor output file (tasks/a8e519e3f6836435d.output) is EMPTY (0 bytes, checked 2026-10-06).
    No advisor wording quoted.
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

## 3. Task 2: X5 / X6 final answers (DIAG_REPORT.md, feat/v2 0e8d4aa); backup PAPER_PLAN.md.bak3 (sha 76cdc82c...)
- Patch n_patch2.py: 486 -> 518 lines; removed only the two "being tested" blocks (task-1 text) + 1 source-note line;
  S4.8 caveat cell appended. 32 [NEW 2026-10-06 overnight, review] tags in total. ASCII, LF.
- B -> ANSWERED by X5: chance (.052 / .054 / .051, Holm p .657; adjacent 20 rates VALID .045-.058; 80/80 hashes;
  confidence high for no excess > ~.015, moderate-high for chance; C3 verdict stands).
- C -> ANSWERED by X6: PARTIAL (cause = told-law centring: .079 INVALID vs .038 with true centring; redraw law not the
  cause; GBM U-shaped vs told width, Lin monotone; P4c failed, quantitative mechanism open).
  Orchestrator's "redraw law irrelevant" written as "the redraw law is not the cause": true-law redraws (Tt .046)
  also restore validity, by CRT construction, so "irrelevant" alone would misread that cell.
- Claims rows X.5a-b, X.6a-c (after X.3); S4.8 pointer; source tag DIAG. Commits a23b8c8, then this one.
- Placement: X5 sentence in 7.5 (limits, next to the PMRT-Lin R3 exception), not 7.6 (= bridging analysis in this
  plan); X6 = Discussion P4a after the X3 limitation; tables Appendix C.4 / C.5 next to X2 / X3
  (tab_appC_x5, tab_appC_x6, fig_appC_x6_width TO BUILD, figure list updated).
- Sentences based on the DIAG suggestions, translated (pmrt_eq / pmrt_r3 / pmrt_nl_eq -> PMRT-Lin / PMRT-Lin R3 /
  PMRT-GBM; P_placebo -> placebo edge; z_bias -> first-order bias term; R-30 -> cell validity labels). No new TERM term.
- Word budget: X5 / X6 add ~130 words, taken from the reserve (350 -> 220).

## Open (for review, not blockers)
- Advisor's own report for A-D is missing (empty file); A rests on the worker's source reading, marked [VERIFY].
- s.6 items 20-21 [DECIDE]; word-budget table is a proposal.
- Still open: xApp / KPI entity slots [DECIDE] (DESIGN s.12); [conf. ref] BibTeX. Q8 done: PAPER_PLAN s.4 -> DESIGN s.12.
