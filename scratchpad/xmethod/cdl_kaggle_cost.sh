#!/usr/bin/env bash
# cdl Kaggle job (GPU T4 session, 4 vCPU): R-13 cost of ONE cdl run per dataset (E2 R2, DEV seed 3_000_000) at
# n 1000 / 4000 / 24000 on CPU (1 thread, fresh process each, 3 in parallel) and on the GPU; then the GPU vs CPU
# reproduction check (R-26) on synthetic data.
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
O=$(realpath -m "${JOB_OUT:-out}"); mkdir -p "$O"
python -c "import torch, numpy, scipy, sys; print(sys.version, torch.__version__, torch.version.cuda, numpy.__version__, scipy.__version__, torch.cuda.is_available() and torch.cuda.get_device_name(0))" > "$O/env.txt" 2>&1
cost() {  # n device tag
  timeout 7200 python scripts/xm_classic_fidelity.py cost --method cdl --n $1 --world E2 --regime R2 \
    --config "{\"device\": \"$2\"}" >> "$O/cost.jsonl" 2> "$O/$3.log"
  echo "done $3 rc $?"
}
cost 1000 cpu c1000 &
cost 4000 cpu c4000 &
cost 24000 cpu c24000 &
( cost 1000 cuda g1000; cost 4000 cuda g4000; cost 24000 cuda g24000
  python scratchpad/xmethod/cdl_gpu_check.py --n 1000 --seed 3000000 --out "$O/gpu_check.jsonl" > "$O/gpu_check.log" 2>&1
  echo "done gpu_check rc $?" ) &
wait
echo all done
