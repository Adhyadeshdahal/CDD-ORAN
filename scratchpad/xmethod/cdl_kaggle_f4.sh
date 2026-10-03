#!/usr/bin/env bash
# cdl Kaggle job F4: parts LO..HI (default 0-3) of K (default 4) of scratchpad/xmethod/cdl_f4.py (NS default
# 500,1000,4000; REPS default 50). torch pinned to uv.lock 2.10.0 (CPU wheel, R-35a).
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
O=$(realpath -m "${JOB_OUT:-out}"); mkdir -p "$O"
LO=${LO:-0}; HI=${HI:-3}; K=${K:-4}; NS=${NS:-500,1000,4000}; REPS=${REPS:-50}
python -m pip install -q torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu > "$O/install.log" 2>&1
python -c "import sys, torch, numpy, scipy; print(sys.version, torch.__version__, numpy.__version__, scipy.__version__)" > "$O/env.txt" 2>&1
for i in $(seq $LO $HI); do
  python -u scratchpad/xmethod/cdl_f4.py run --ns $NS --reps $REPS --part $i/$K --out "$O/f4_$i.jsonl" > "$O/log_$i.txt" 2>&1 &
done
wait
echo all done
