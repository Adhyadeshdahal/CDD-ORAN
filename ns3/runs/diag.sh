#!/usr/bin/env bash
export LD_LIBRARY_PATH=$HOME/projects/ns-3-opt/build/lib
for s in 202 204 207; do
  d=$HOME/oran-ns3-runs/diag_$s; mkdir -p $d; cd $d
  [ -f units.csv ] && continue
  start=$(date +%s)
  $HOME/projects/ns-3-opt/build/scratch/ns3.48-oran-diag-optimized --bandwidth=5e6 --uePerCell=3 --rateMbps=1.5 --period=20 --warmup=30 --simTime=400 --placebo=1 --seed=$s --out=$d > run.log 2>&1
  echo "$(date +%T) done diag:$s rc=$? $(( $(date +%s) - start ))s"
done
