READY-TO-MERGE (2026-10-06: user decisions 1-5 + conference citation recorded in TERM / PAPER_PLAN / WAP)
# paper-terms (2026-10-06): PAPER_PLAN post hoc evidence, X4 note, Q8 pointer, user decisions 1-5
Earlier change log (D1-D7, D5, layout, title, Q1-Q19): this file at ec1d79a (xm/freeze); overnight detail: this file at
45f8943 (xm/paper-notes). Files outside the repo: D:/academia/major-project/ (no git). Old .bak files untouched; .tex not edited.

## 1-3. Overnight 2026-10-06 (backups PAPER_PLAN.md.bak2 / .bak3; all additions [NEW 2026-10-06 overnight, review])
- Advisor A-D: the advisor output file was EMPTY (0 bytes); A answered from sources (metric scope: 29 vs 10 GT-TRUE
  edges; pooled overall recall .79 = bridging .79), [VERIFY]; B = X5 chance; C = X6 PARTIAL (told-law centring;
  mechanism open); D superseded by X7.
- Claims rows X4.1-X4.4, X7.1-X7.5, X.5a-b, X.6a-c with sources; Discussion P2a / P4a, 7.5 sentence; word budget table.
- CDD-ORAN xm/paper-notes a23b8c8: X4_REPORT "not cross-platform" softened + dated note (no number changed).
- Q8: PAPER_PLAN s.4 "Method slots" points to DESIGN s.12 / s.12.1 (45f8943).

## 4. User decisions 2026-10-06, tag [DECIDED 2026-10-06, user] (orchestrator brief)
Backups: TERMINOLOGY.md.bak4, PAPER_PLAN.md.bak4, WRITING_AGENT_PROMPT.md.bak4 (= pre-edit). Patch n_patch4.py
(exact-match, counts asserted, all three written only if all succeed). ASCII, LF. 23 DECIDED-2026-10-06 tags in PAPER_PLAN.
- 1. "random-deferral referee" for X4:rand@p:
  - TERMINOLOGY s.7 (referee side, next to never-sleep; s.6 is discovery methods, so the row went to s.7): definition
    from X4_REPORT Setup (each request deferred independently with probability p; p .25 / .352 / .50; from t = 0).
  - PAPER_PLAN s.6 item 20 RESOLVED; X4/X7 preamble and row X4.1 use the name; WRITING_AGENT_PROMPT bullet.
- 2. Appendix C = "Supplementary post hoc analyses", all X1-X7 in X order:
  - C.1 X1 runtime, C.2 X2, C.3 X3 (unchanged); C.4 X4 (moved from B.4); C.5 X5 (was C.4); C.6 X6 (was C.5);
    C.7 X7 (moved from B.4). B.4 removed; Appendix B note: bridging analysis only.
  - Tables renamed tab_appB_x4 / x7 -> tab_appC_x4 / x7; figure list rows moved into the appC block in X order.
  - Cross-references updated: P2a, rows X4.* (C.4), X7.* (C.7), X.5a-b (C.5), X.6a-c (C.6), advisor B / C answers.
    Check: no "Appendix B.4" or "tab_appB_x" left outside "Was:" / "moved from" text. The claims-row ID "B.4" (bridging
    PMRT-GBM recall) is a row ID, unchanged.
  - Added: X2 / X3 opening sentence stays; X4-X7 open with their own post hoc status.
  - Item 21 RESOLVED. TERMINOLOGY s.9 lists Appendix B / C (C.1-C.7); WRITING_AGENT_PROMPT bullet.
- 3. Word budget = rough guideline:
  - Heading "Word budget: rough guideline"; column "Rough guideline (words)"; DECIDED lines: writers do not check or
    flag word counts (s.2 header, budget table, item 17, WRITING_AGENT_PROMPT).
  - No existing instruction told writers to flag overruns (searched all three files); the s.2 "must shrink / must
    lose" sentences are marked planning arithmetic only, not a check for writers.
- 4. Data and code availability item: cleanup deferred to submission (Zenodo DOI, figures / code repointed, one
  history rewrite). Worker note added: a history rewrite changes commit hashes, so the freeze hashes cited in that
  statement must be taken or mapped after the rewrite.
- 5 + addition. Conference BibTeX is in xApp-Journal-Discover-Telecommunications/sn-bibliography.bib, key
  dahal2026cddoran (NaNA 2026, pp. 28-34, doi 10.23919/NaNACPS00070.2026.00013; entry checked). Every [conf. ref]
  replaced by \cite{dahal2026cddoran} (n_patch5.py): PAPER_PLAN 6, TERMINOLOGY 5, WRITING_AGENT_PROMPT 2; 0 left.
  The "leave [conf. ref] as is" notes became "cite as" notes; back matter BibTeX item and Q18 OPEN part RESOLVED.
- Diff vs .bak4: TERM +6 / 1 changed; WAP +4; PAPER_PLAN 520 -> 535 (changed lines only where retagged or moved).

## 5. CDD-ORAN
- Branch xm-paper-notes created from feat/v2 13408ae as briefed (hyphen; the older xm/paper-notes is merged into it).
- Only this status file changed; one commit, subject only, not pushed.

## Open (for review, not blockers)
- Advisor's own report for A-D is missing; A rests on the worker's source reading, marked [VERIFY].
- xApp / KPI entity slots [DECIDE] (DESIGN s.12). Conference BibTeX: closed (dahal2026cddoran).
- Freeze-hash citation vs the planned history rewrite (decision 4 note).
