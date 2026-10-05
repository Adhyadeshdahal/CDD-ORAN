READY-TO-MERGE
# paper-terms (2026-10-05): paper wording decisions recorded outside the repo (TERMINOLOGY, WRITING_AGENT_PROMPT, PAPER_PLAN)
Files in D:/academia/major-project/ (no git there). .bak files untouched; manuscript .tex never edited. ASCII, LF.
Numbers only from results/eval/REPORT.md and docs/xmethod/EXP_B.md s.9, cited by section.

## 1. D1-D7 panel decisions (feat/v2 1f710b9), tag [PROVISIONAL 2026-10-05]
- TERMINOLOGY:
  - header sources; s.2 E1-E5 row ("controlled diagnostic environments") + regimes R1-R4;
  - s.6.1 PMRT-GBM / PMRT-Lin / PMRT-Lin R3 + mandatory sentence; s.6.2 MSCR row; new s.6.5 families + method set;
  - s.8 validity / F_max / verdict labels / A-1; s.9 Study 4 + bridging rows; s.10 eight do-not-use rows;
  - new s.12 claim, disclosures, recall sentence, bridging rules.
- WRITING_AGENT_PROMPT: read-first item 7, Study 4 / bridging bullets, do-not-claim list, number sources.
- D5 check: REPORT V0 contradicts the panel's "rarely declared any edge at n <= 1000" (n 1000 recall > 0 in 3 / 10).

## 2. D5 naming (user), tag [DECIDED 2026-10-05, user]
- Conference method = "CDD O-RAN" (title checked in archive/xApp-Nana-Conference-2026/main.tex).
- Study 4 arm = "the causal-discovery stage of CDD O-RAN [conf. ref], re-implemented here", then "CDD O-RAN discovery".
- "CDL" = code name, never in prose; s.10 [DECIDE] resolved; s.12 wording keeps the recall correction.
- "CDD-ORAN" = repository code name only; the journal is a new contribution, never an extension.

## 3. Study 4 layout (user GO) and title (user), tag [DECIDED 2026-10-05, user]
- TERMINOLOGY s.9: 4.1-4.6 + Appendix A (one page per environment); WRITING_AGENT_PROMPT points to it.
- Title "Design-Based Causal Maps for xApp Conflict Mitigation in O-RAN"; no framework name; "design-based" defined
  in the abstract's first sentences.

## 4. PAPER_PLAN.md s.6 answers (user), tag [DECIDED 2026-10-05, user]
- PAPER_PLAN s.6: Q1-Q12 and Q14 marked RESOLVED with the answer (question text kept after "Was:"); Q13 PENDING
  (panel); Q15-Q19 open.
- Q1 running title "Design-Based Causal Maps for xApp Conflicts": TERMINOLOGY title row, WRITING_AGENT_PROMPT.
- Q2 "control parameter (NCP)" only, never "knob". Q3 "referee" (code: arbiter / UnitArbiter). TERMINOLOGY s.1
  [DECIDE] removed, s.10 rows added.
- Q4 "the system-level simulator", no "E6-P" in prose:
  - TERMINOLOGY s.2 row (E6-P now the internal label), s.3 heading, s.6.1, s.9, s.12 sentences, never-in-prose list.
  - WRITING_AGENT_PROMPT 5 places; PAPER_PLAN thesis, s.3 / s.6 titles, 7.6, P3, B.1, claims table, figure row,
    guardrails.
  - Mandatory sentence now ends "...weighted-BY layer of the PMRT test used in Studies 1-3." The recall sentence ends
    "...not on the simulator data of Studies 1-3." File names (fig_system_e6p) kept.
- Q5 "map-driven referee" / "certified-safe map-driven referee": TERMINOLOGY s.7 paper terms, PAPER_PLAN contribution 2.
- Q6:
  - TERMINOLOGY s.8 row: one Experimental-design sentence; commit hashes only in the data and code availability
    statement; no "pre-registered" otherwise.
  - Section titles changed accordingly: 4.1 / 7.1 "Purpose and frozen analysis plan". PAPER_PLAN 5.1 and 7.1 no
    longer cite hashes in prose. A-1 row: hashes in the availability statement only.
- Q7 TERMINOLOGY s.9 heading [DECIDE] removed.
- Q8 one global method-colour mapping (TERMINOLOGY s.11, PAPER_PLAN figure notes). Rebuild = aud1. The exact slot
  list and the xApp / KPI slots stay [DECIDE] (not answered).
- Q9 R-definition wording: WRITING_AGENT_PROMPT pointer, PAPER_PLAN S2.7 + guardrail, TERMINOLOGY s.10 row.
- Q10 never mention the rename or "MSCR+": TERMINOLOGY naming rule + s.10; PAPER_PLAN disclosures (rename item removed).
- Q11 Limitations sentence: PAPER_PLAN P5 tagged.
- Q12 primary z -0.70 in Results, z -2.21 one labelled Discussion sentence: PAPER_PLAN P3 mechanism + claims table S2.6.
- Q14 Appendix C: X2 / X3 figure + compact table each; PMRT time only if logged (FD:v4.timing_total_s).
## 5. Q13 / Q15-Q19 (user), tag [DECIDED 2026-10-05, user]
- Q13: contribution 4 removed; contribution 3 ends with the practical check + "frozen and hash-pinned"; the check is
  repeated in the abstract (plan item 8) or the Conclusion (PAPER_PLAN, TERM s.12, WRITING_AGENT_PROMPT).
- Q15: every [PROVISIONAL] tag becomes [DECIDED]. "outperforms" only for Study 1 (0.90 / 1.00 checked in FD:v4.P3_values).
  E4 disclosure added; its R3 clause is qualified to lambda > 0 (csv: 15 / 15 INVALID; at lambda 0 none).
- Q16 intro figure + kappa table in, thumbnail + e-process out. Q17 ~9 000 words (plan targets ~11 000: must shrink).
- Q18 PAPER_PLAN Back matter + s.6: funding / availability [TBD]; no competing interests; contributions = SN template;
  the [conf. ref] BibTeX is still to be supplied.
- Q19: Study 4 = Section 7 (TERM s.9 now 7.1-7.6). "pre-registered" left only in rules that forbid it.
