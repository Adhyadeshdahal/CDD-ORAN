#!/usr/bin/env bash
# cdl Kaggle job: integration (brief step 4) = scripts/xm_classic_integration.py --methods cdl, n 1000, kappa .25
# (R-27), tau on DEV seeds 3000001-3 per cell, run + score on DEV seed 3000000, one process per world.
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
O=$(realpath -m "${JOB_OUT:-out}"); mkdir -p "$O"
python -m pip install -q torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu > "$O/install.log" 2>&1
python -c "import sys, torch, numpy, scipy; print(sys.version, torch.__version__, numpy.__version__, scipy.__version__)" > "$O/env.txt" 2>&1
for w in E1 E2 E3 E4 E5; do
  ( python -u scripts/xm_classic_integration.py --methods cdl --worlds $w --n 1000 --kappa 0.25 --out "$O/$w" \
      > "$O/int_$w.log" 2>&1; echo "done $w rc $?" ) &
done
wait
echo all done
