# Kaggle job: R-13 cost campaign (cost.py), 4 single-thread chains in parallel, 2 CPU-h cap, 15-min probes.
set -x
OUT=${JOB_OUT:-out}; mkdir -p "$OUT"
python -m pip install -q tigramite==5.2.10.1 dcor==0.7 momentchi2==0.1.8 gymnasium==1.2.3 pyyaml==6.0.3 numba==0.67.0 llvmlite==0.49.0 > "$OUT/pip_extra.log" 2>&1
nproc; free -g
python scratchpad/xmethod/citests/cost.py all "$OUT/cost.jsonl" --budget 7200 --probe 900 --jobs 4 \
  --methods cmi_knn,pdcor,mscr,rcot2,pcorr
