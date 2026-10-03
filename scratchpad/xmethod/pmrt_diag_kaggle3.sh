#!/usr/bin/env bash
# pmrt-diag Kaggle job 3: H6 confirmation on RESERVE DEV seeds 3_000_160-189 (DEV datasets, rep -1), same-data
# comparator pcorr_eq on the flagged DEV datasets, and power (every primary candidate) of prod vs the exact variant.
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
O=${JOB_OUT:-out}; mkdir -p "$O"
run() {  # world regime n kappa seeds variants tag [--power]
  python scratchpad/xmethod/pmrt_diag.py --world $1 --regime $2 --n $3 --kappa $4 --seeds $5 --reps=-1 \
    --variants $6 --out "$O/$7.jsonl" ${8:-} > "$O/$7.log" 2>&1
  echo "done $7 rc $?"
}
RES=3000160-3000189
( run E2 R2 1000 0.5 $RES prod,r3,inv,pcorr_eq res_e2r2_n1000_k05; run E2 R2 1000 0.25 $RES prod,r3,inv,pcorr_eq res_e2r2_n1000_k025;
  run E2 R2 1000 0.125 $RES prod,r3,inv,pcorr_eq res_e2r2_n1000_k0125 ) &
( run E2 R2 1000 0.5 3000100-3000159 pcorr_eq dev_e2r2_n1000_k05_pcorr; run E2 R2 1000 0.5 $RES prod,r3,noown,inv pow_res_e2r2_n1000_k05 --power ) &
( run E2 R2 1000 0.5 3000100-3000159 prod,r3,noown,inv pow_dev_e2r2_n1000_k05 --power ) &
( run E1 R2 1000 0.25 3000100-3000119 prod,noown,inv pow_dev_e1r2_n1000_k025 --power; run E3 R2 1000 0.25 3000100-3000119 prod,noown,inv pow_dev_e3r2_n1000_k025 --power;
  run E5 R2 1000 0.25 3000100-3000119 prod,noown,inv pow_dev_e5r2_n1000_k025 --power ) &
wait
echo all done
