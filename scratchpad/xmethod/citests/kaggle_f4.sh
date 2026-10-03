# Kaggle job (R-26; Colab unavailable): F4 heavy, all four parts in one CPU session (4 cores), arm native.
set -x
OUT=${JOB_OUT:-out}; mkdir -p "$OUT"
python -m pip install -q tigramite==5.2.10.1 dcor==0.7 momentchi2==0.1.8 gymnasium==1.2.3 pyyaml==6.0.3 numba==0.67.0 llvmlite==0.49.0 > "$OUT/pip_extra.log" 2>&1
nproc
for P in mscr pdcor_null pdcor_p500 pdcor_p1000; do
  PART=$P REPS_P1000=50 bash scratchpad/xmethod/citests/f4_heavy_job.sh
done
python scratchpad/xmethod/citests/f4.py summary "$OUT/f4_heavy.jsonl" --out "$OUT/F4_heavy_summary.json"
