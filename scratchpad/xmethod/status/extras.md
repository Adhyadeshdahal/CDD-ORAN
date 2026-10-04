READY-TO-MERGE

# extras status (2026-10-05): R-60 X2 + X3 BUILT, NOT LAUNCHED (worker exp-b, branch xm/exp-b)

## Done
- Merged feat/v2 2f01f1e (51b2b1f). No EVAL output opened.
- `scratchpad/xmethod/EXTRAS_PROTOCOL.md`: the declaration (question, cells, arms, seeds, metrics per
  eval_analysis, exact report wording incl. "exploratory ... do not change C1-C3", Amendments section).
- `docs/benchmark/SEED_REGISTRY.json`: 3200000-3200199 claimed as XMETHOD_EXTRAS for E1-E4 (used + claimed_by).
  - X2 tune 3200000-019, X2 measure 3200020-059;
  - X3 measure 3200060-119, 3200120-199 unassigned.
- `cdd_oran/xmethod/extras.py` (new; no frozen file edited). In-process hooks around frozen campaign, restored on
  exit:
  - XMETHOD_EXTRAS seed guard + DEV mode;
  - X2: DITHER_DELTA / DITHER_BLOCKS at generation;
  - X3: told design via runner.run_one (width / switch shift / E4 R3 lambda);
  - `integrity.extras` stamp, a same-spec merge accept rule, and cloud sessions running `extras run`.
  - Also `specs`, `projection`, `analyse` (frozen eval_analysis screen / build_cells, pooled rates, X2 trend,
    X3 curve, provenance re-check against the frozen generator).
- Specs `scratchpad/xmethod/specs/extras/`: x2_d02 / d05 / d10 (= frozen design) / d20 / b05 / b80, x3_told.
  - Arms are copied from the frozen EVAL spec (test).
  - DEV cost table dev_cost_agg.json; PROJECTION.md / projection.json.
- Tests `tests/test_xmethod_extras.py`:
  - Windows 15 passed / 1 Linux-only skip; Linux (Docker, py 3.12.12) 16 passed;
  - covers: reproduction cell bit-for-bit vs frozen generator (6 world / regime cases), overrides, told variants,
    seed guard, hooks restored, records + stamps, fork children inherit the hooks, specs = protocol, analysis
    end-to-end, freeze manifest valid.
- Freeze manifest: all 301 files match under the LF rule (9 files committed with CRLF fail a plain
  `sha256sum -c`; as before).

## Cost (DEV cost table, Kaggle-ref CPU-h, upper estimate)
- each X2 spec: 2280 units, 9.6 CPU-h, ~2.4 h wall at 4 procs (Kaggle session);
- x3_told: 4560 units, 19.6 CPU-h, ~4.9 h wall at 4 procs;
- total 77.2 CPU-h = 7 Kaggle sessions.
- pmrt_nl_eq E4 R3 cost extrapolated (no DEV E4 cost).

## Launch (only after EVAL has launched all 96 parts and its last VPS job is pulled; keep 1 Kaggle slot free)
Order X2 (x2_d10 first: reproduction cell), then X3. One Kaggle session per spec, from the repo root:
```
for s in x2_d10 x2_d02 x2_d05 x2_d20 x2_b05 x2_b80 x3_told; do
  uv run python -m cdd_oran.xmethod.extras kaggle --spec scratchpad/xmethod/specs/extras/$s.json \
    --name xm-extras-${s//_/-} --parts 4 --cost-table scratchpad/xmethod/specs/extras/dev_cost_agg.json \
    --venv-python 3.12.14 --paths scratchpad/xmethod/EXTRAS_PROTOCOL.md
done        # (one at a time as Kaggle slots free up; add --dry-run to check)
```
- VPS alternative: `extras vps --spec <s> --name <n> --parts 7 --procs 7 --cost-table ...`. VPS CPU-s then
  convert by factors_vps_py3.12.13.json as in Exp B.
- Merge: `uv run python -m cdd_oran.xmethod.extras merge --spec <spec> --inputs '<pulled dir>/**/res_*.jsonl'
  --out scratchpad/xmethod/results/extras/<s>/merged.jsonl`.
- Analyse: `uv run python -m cdd_oran.xmethod.extras analyse --spec x2_d02 x2_d05 x2_d10 x2_d20 x2_b05 x2_b80
  x3_told --merged <7 merged files, same order> --out scratchpad/xmethod/results/extras/`.

## Notes
- X2 tune seeds run only the 3 tau arms (pc_eq, notears, shap_dag); p arms declare by BY. This saves ~1/3 of
  pmrt_nl_eq.
- X3 pcorr_eq told-width / lambda cells must equal its exact cells (checked on one dataset); shift changes it
  (told setpoints are its covariates).

## Questions: all ANSWERED
- (none open)
