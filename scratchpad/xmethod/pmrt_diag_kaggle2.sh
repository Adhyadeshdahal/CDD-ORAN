#!/usr/bin/env bash
# pmrt-diag Kaggle job 2: E2 R2 n 1000 kappa .5 dither-regenerated replicates (reps 0-9), 4 seed lanes.
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
O=${JOB_OUT:-out}; mkdir -p "$O"
for S in 3000100-3000114 3000115-3000129 3000130-3000144 3000145-3000159; do
  ( python scratchpad/xmethod/pmrt_diag.py --world E2 --regime R2 --n 1000 --kappa 0.5 --seeds $S --reps=0:10 \
      --variants prod,r3,noown,inv --out "$O/e2r2_n1000_k05_regen_$S.jsonl" > "$O/regen_$S.log" 2>&1; echo "done $S rc $?" ) &
done
wait
echo all done
