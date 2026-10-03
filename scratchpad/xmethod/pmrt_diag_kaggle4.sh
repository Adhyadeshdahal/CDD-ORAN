#!/usr/bin/env bash
# pmrt-diag Kaggle job 4 (= the failed pcorr / reserve part of job 3; configs/ now bundled).
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
O=${JOB_OUT:-out}; mkdir -p "$O"
run() {
  python scratchpad/xmethod/pmrt_diag.py --world $1 --regime $2 --n $3 --kappa $4 --seeds $5 --reps=-1 \
    --variants $6 --out "$O/$7.jsonl" ${8:-} > "$O/$7.log" 2>&1
  echo "done $7 rc $?"
}
RES=3000160-3000189
( run E2 R2 1000 0.5 $RES prod,r3,inv,pcorr_eq res_e2r2_n1000_k05 ) &
( run E2 R2 1000 0.25 $RES prod,r3,inv,pcorr_eq res_e2r2_n1000_k025 ) &
( run E2 R2 1000 0.125 $RES prod,r3,inv,pcorr_eq res_e2r2_n1000_k0125 ) &
( run E2 R2 1000 0.5 3000100-3000159 pcorr_eq dev_e2r2_n1000_k05_pcorr ) &
wait
echo all done
