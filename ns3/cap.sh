#!/usr/bin/env bash
# Keep the CAP oldest ns-3 episode processes running; younger ones are paused (SIGSTOP), not killed.
cap=${1:-8}
names=ns3.48-oran-pmr,ns3.48-oran-dia
while pgrep -f "ns3.48-oran-(pmrt|diag)" >/dev/null || pgrep -f "xargs -P" >/dev/null; do
  procs=$(ps -C "$names" -o pid=,stat=,etimes= --sort=-etimes 2>/dev/null)
  run=$(echo "$procs" | head -n "$cap" | awk '$2 ~ /^T/ {print $1}')
  stop=$(echo "$procs" | tail -n +$((cap + 1)) | awk 'NF && $2 !~ /^T/ {print $1}')
  [ -n "$stop" ] && kill -STOP $stop
  [ -n "$run" ] && kill -CONT $run
  sleep 20
done
