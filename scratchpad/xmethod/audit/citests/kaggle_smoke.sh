# Kaggle job (audit-citests): validity smoke for mscr + pdcor (CPU, 4 cores); uv.lock-pinned extra deps (R-16/R-35).
set -x
OUT=${JOB_OUT:-out}; mkdir -p "$OUT"
python -m pip install -q tigramite==5.2.10.1 dcor==0.7 momentchi2==0.1.8 gymnasium==1.2.3 pyyaml==6.0.3 > "$OUT/pip_extra.log" 2>&1
nproc
SMOKE_OUT="$OUT/smoke_runs_kaggle.jsonl" python scratchpad/xmethod/audit/citests/smoke.py cpu mscr,pdcor 4 > "$OUT/smoke.log" 2>&1
tail -3 "$OUT/smoke.log"
