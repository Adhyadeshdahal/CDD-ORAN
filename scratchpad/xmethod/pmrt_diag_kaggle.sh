#!/usr/bin/env bash
# pmrt-diag Kaggle grid (4 lanes, 1 thread each). Cells: world regime n kappa reps variants -> $JOB_OUT/<cell>.jsonl
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
O=${JOB_OUT:-out}; mkdir -p "$O"
S=3000100-3000159
V=prod,r3,noown,inv
run() {  # world regime n kappa reps variants tag
  python scratchpad/xmethod/pmrt_diag.py --world $1 --regime $2 --n $3 --kappa $4 --seeds $S --reps=$5 \
    --variants $6 --out "$O/$7.jsonl" > "$O/$7.log" 2>&1
  echo "done $7 rc $?"
}
( run E2 R2 4000 0.5 -1:10 $V e2r2_n4000_k05 ) &
( run E2 R2 1000 0.5 -1:3 prod,prodB,inv,invB e2r2_n1000_k05_B; run E2 R2 500 0.5 -1:10 $V e2r2_n500_k05 ) &
( run E2 R2 1000 0.25 -1:10 $V e2r2_n1000_k025; run E2 R1 1000 0.5 -1:10 $V e2r1_n1000_k05; run E2 R2 1000 0.125 -1:10 $V e2r2_n1000_k0125 ) &
( run E3 R2 1000 0.25 -1:10 $V e3r2_n1000_k025; run E5 R2 1000 0.25 -1:10 $V e5r2_n1000_k025; run E1 R2 1000 0.25 -1:10 $V e1r2_n1000_k025 ) &
wait
echo all done
