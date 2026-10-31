#!/usr/bin/env bash
# Usage: run_stage.sh RUNS_DIR JOBS SPEC...   SPEC = stage:seed[:cell:t0:mode]
set -u
root=$1; jobs=$2; shift 2
bin=${NS3_OPT:-$HOME/projects/ns-3-opt}/build/scratch/ns3.48-oran-pmrt-scenario-optimized
export LD_LIBRARY_PATH=${NS3_OPT:-$HOME/projects/ns-3-opt}/build/lib
common="--bandwidth=5e6 --uePerCell=3 --rateMbps=1.5 --period=20 --warmup=30"
one() {
  IFS=: read -r stage seed cell t0 mode <<<"$1"
  case $stage in
    dev|eval) dir=$root/${stage}_$seed; args="--simTime=400" ;;
    placebo) dir=$root/placebo_$seed; args="--simTime=400 --placebo=1" ;;
    gt) dir=$root/gt_${seed}_${cell}_${t0}_$mode; args="--simTime=$((t0 + 91)) --gtCell=$cell --gtT0=$t0 --gtMode=$mode" ;;
  esac
  [ -s "$dir/units.csv" ] && return 0
  start=$(date +%s)
  $bin $common $args --seed=$seed --out=$dir > "$dir.log" 2>&1
  echo "$(date +%T) done $1 rc=$? $(( $(date +%s) - start ))s"
}
export -f one; export root bin common
printf '%s\n' "$@" | xargs -P "$jobs" -I{} bash -c 'one {}'
