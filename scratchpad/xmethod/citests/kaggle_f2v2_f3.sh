# Kaggle job: (1) F2 rcot2 vs R RCIT v2 (exact momentchi2 py vs R; lpd4 num_f 100 vs RCIT defaults), perm skipped;
# (2) F3 rcot2 + pdcor on the stored frozen E2 replicate-00 artifacts (read-only copies in f3root/).
set -x
OUT=${JOB_OUT:-out}; mkdir -p "$OUT"
python -m pip install -q tigramite==5.2.10.1 dcor==0.7 momentchi2==0.1.8 gymnasium==1.2.3 pyyaml==6.0.3 numba==0.67.0 llvmlite==0.49.0 > "$OUT/pip_extra.log" 2>&1
(SKIP_PERM=1 REPS=200 bash scratchpad/xmethod/citests/f2_rcit_job.sh) > "$OUT/f2v2.log" 2>&1 &
python scratchpad/xmethod/citests/f3_frozen.py rcot2 scratchpad/xmethod/citests/f3root --out "$OUT" > "$OUT/f3_rcot2.log" 2>&1 &
python scratchpad/xmethod/citests/f3_frozen.py pdcor scratchpad/xmethod/citests/f3root --out "$OUT" > "$OUT/f3_pdcor.log" 2>&1 &
wait
tail -n 5 "$OUT"/f3_*.log
