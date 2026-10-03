#!/usr/bin/env bash
# cdl Kaggle job F3 (CPU, 4 vCPU): the conference graph-recovery runs (seed 45; Env I 20k steps, Env II 50k steps)
# with the ORIGINAL training loop of refactor/codebase@65b56f3 (refcode.zip = git archive of cdd_oran configs tests),
# once with the original CDL and once with the port; CPU, deterministic, 1 thread each.
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
W=$(pwd)
O=$(realpath -m "${JOB_OUT:-out}"); mkdir -p "$O"
R=/tmp/ref; rm -rf $R; mkdir -p $R
if [ -f scratchpad/xmethod/results/cdl/refcode.zip ]; then
  python -m zipfile -e scratchpad/xmethod/results/cdl/refcode.zip $R
else  # Kaggle auto-unzips a nested zip into a directory of the mounted dataset
  SRC=$(find /kaggle/input -type d -path '*results/cdl/refcode' | head -1); cp -r "$SRC"/. $R/
fi
ls $R > "$O/ref_ls.txt"
python -c "import gymnasium" 2>/dev/null || pip install -q gymnasium==1.2.3 >> "$O/install.log" 2>&1
python -c "import torch, numpy, gymnasium, sys; print(sys.version, torch.__version__, numpy.__version__, gymnasium.__version__)" > "$O/env.txt" 2>&1
run() {  # env which tag
  (cd $R && python $W/scratchpad/xmethod/cdl_fidelity.py --port $W/cdd_oran/xmethod/methods/_cdl_model.py \
     --mode f3 --env $1 --which $2 --out "$O/f3_$3.jsonl" > "$O/$3.log" 2>&1)
  echo "done $3 rc $?"
}
run env_i_cdl.yaml port envI_port &
run env_i_cdl.yaml original envI_orig &
run env_ii_cdl.yaml port envII_port &
run env_ii_cdl.yaml original envII_orig &
wait
echo all done
