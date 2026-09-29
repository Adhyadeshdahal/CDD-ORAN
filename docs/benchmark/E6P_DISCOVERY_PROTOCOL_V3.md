# E6-P discovery protocol v3: MSCR-CRT v2 at a power-sized sample (step 1, second re-test)

FROZEN: yes (2026-09-30)

**Status: FROZEN 2026-09-30 03:40 NST, before any eval_v3 episode was simulated.** This is a disclosed amendment written
after the v2 verdict. It changes ONE thing, the EVAL sample. Everything else is protocol v2
(`E6P_DISCOVERY_PROTOCOL_V2.md`, frozen fea6488c): the method (MSCR-CRT v2, `crt_units_v2.py`, unchanged since
3919ac6), the analyzer (`e6p_disc_analyze_v2.py`), the GT rule, and the criteria K0 / K1 / G / P1v2 / P2v2 and
their precedence.

## 0. Disclosure: this is the third attempt
- **v1** (60 EVAL episodes, MSCR-CRT v1): KILL, with indirect recall 3/8 and a sign bug.
- **v2** (480 fresh episodes, MSCR-CRT v2): KILL, and the only failed part is the premise edge.
  - K0 rate .033 with 0 BY declarations; K1 and G pass.
  - 23/60 declared, overall precision .95, sign accuracy .95, chain 3/4, indirect recall 4/8.
  - sleep → nbr pv: β = +6.97, p = .007, z_approx = 2.67. The BY cutoff at that rank was about .004, so it was
    not declared, and P1v2 fails.
- **Why v2 was underpowered.** The v2 sample size came from a power analysis that used the v1 GT effect (+15.4). The
  fresh GT effect is +9.1 (CI 3.3-15.8), a winner's curse in the planning input.
- **v3 changes only n.** The new n is sized from the v2 observed z. The v2 records are NOT pooled into v3, which is
  a fresh, independent test.
- **Reporting rule.** Any claim must report all three attempts (v1 KILL, v2 KILL, v3 result). A v3 PASS is a
  third-attempt result on fresh seeds; it is not a confirmation.

## 1. Sample size
v2 has z = 2.67 at n = 480. The declaration threshold is about z 3.0-3.3 (BY at 20-25 declarations over 60
hypotheses). For about 80 % power the expected z must be about 4.1, which needs n ≈ 480 × (4.1 / 2.67)² ≈ 1130.
**n = 1200 episodes.**

## 2. Data

| stage (driver) | seeds | records | use |
|---|---|---|---|
| eval_v3 | 182000-183199 (1200 eps, fresh; unused part of the registered block, note added to SEED_REGISTRY) | stage "eval", sub "v3", fold = j // 300 (4 folds) | the discovery test |
| gt_ext (v2) | 183520-183539 | stage "gt", sub "ext" | GT (as v2; the method is unchanged, so the GT is not re-used for tuning) |
| placebo_ext (v2) | 183540-183559 | stage "placebo", sub "ext" | K0 (as v2) |
| dev (v1) | 183300-183319 | stage "dev" | baseline τ (as v2) |

- Collection: Kaggle, `e6p_discovery.py` stage eval_v3. The episode code is unchanged since v1.
- The v2 analyzer requires EVAL folds 0-3. eval_v3 uses folds 0-3, each of 300 episodes.

## 3. Criteria
Identical to protocol v2, section 4. P2v2 folds are the four eval_v3 folds (≥ 3 of 4). P2v2 needs the baseline pass,
which may finish after P1v2 is known. The verdict uses the full run.
