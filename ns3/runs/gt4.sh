#!/usr/bin/env bash
export LD_LIBRARY_PATH=$HOME/projects/ns-3-opt/build/lib
run() {
  IFS=: read -r _ seed cell t0 mode <<<"$1"
  dir=$HOME/oran-ns3-runs/gt_${seed}_${cell}_${t0}_$mode
  start=$(date +%s)
  $HOME/projects/ns-3-opt/build/scratch/ns3.48-oran-pmrt-scenario-optimized --bandwidth=5e6 --uePerCell=3 --rateMbps=1.5 --period=20 --warmup=30 --simTime=$((t0 + 91)) --gtCell=$cell --gtT0=$t0 --gtMode=$mode --seed=$seed --out=$dir > "$dir.log" 2>&1
  echo "$(date +%T) done $1 rc=$? $(( $(date +%s) - start ))s"
}
export -f run
printf '%s\n' gt:209:8:118:reject gt:209:0:300:reject gt:209:1:302:reject gt:210:7:236:reject gt:210:7:256:reject gt:212:2:304:reject | xargs -P 3 -I{} bash -c 'run {}'
