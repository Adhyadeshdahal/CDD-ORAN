#!/usr/bin/env bash
# cdl Kaggle job F3 seed check: Env I (20k steps) with the PORT in the original loop, seeds 45 (torch pinned 2.10.0,
# R-35a; cdl-f3-b ran the image's 2.11) and 46-49 (sensitivity of the weak p3 -> K2 edge). CPU deterministic, 1 thread.
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
W=$(pwd)
O=$(realpath -m "${JOB_OUT:-out}"); mkdir -p "$O"
R=/tmp/ref; rm -rf $R; mkdir -p $R
if [ -f scratchpad/xmethod/results/cdl/refcode.zip ]; then
  python -m zipfile -e scratchpad/xmethod/results/cdl/refcode.zip $R
else
  SRC=$(find /kaggle/input -type d -path '*results/cdl/refcode' | head -1); cp -r "$SRC"/. $R/
fi
python -m pip install -q torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu gymnasium==1.2.3 > "$O/install.log" 2>&1
python -c "import sys, torch, numpy, gymnasium; print(sys.version, torch.__version__, numpy.__version__, gymnasium.__version__)" > "$O/env.txt" 2>&1
run() {  # seed
  (cd $R && python $W/scratchpad/xmethod/cdl_fidelity.py --port $W/cdd_oran/xmethod/methods/_cdl_model.py \
     --mode f3 --env env_i_cdl.yaml --which port --seed $1 --out "$O/f3_envI_s$1.jsonl" > "$O/s$1.log" 2>&1)
  echo "done s$1 rc $?"
}
( run 45; run 49 ) &
run 46 &
run 47 &
run 48 &
wait
echo all done
