# F4 for the resampling methods at their primary nulls (B 9999 Besag-Clifford), arm native (port fidelity gate, as
# F4 local), PRIMARY candidates only (action -> KPI: the null level is judged on the primary family; KPI -> KPI
# self edges would cost full B each). R-26: one Colab session per PART, each < 3 h on 2 vCPUs, resumable per rep
# (records already in out/f4_heavy.jsonl, or in the seed file f4_seed.jsonl copied there first, are skipped):
#   PART=mscr        mscr null / planted, n 500 / 1000, x100
#   PART=pdcor_null  pdcor null n 500 / 1000 x100
#   PART=pdcor_p500  pdcor planted n 500 x100
#   PART=pdcor_p1000 pdcor planted n 1000 x${REPS_P1000:-30}
#   (Kaggle: kaggle_f4.sh runs all four parts in one session; outputs -> $JOB_OUT)
# cmi_knn F4 runs on the GPU backend (separate job). Outputs -> out/.
set -x
OUT=${JOB_OUT:-out}; mkdir -p "$OUT"
O=$OUT/f4_heavy.jsonl
[ -f scratchpad/xmethod/citests/f4_seed.jsonl ] && [ ! -f "$O" ] && cp scratchpad/xmethod/citests/f4_seed.jsonl "$O"
J=$(nproc)
F="python scratchpad/xmethod/citests/f4.py run $O"
C='{"arm": "native"}'
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
case "$PART" in
  mscr)
    for n in 500 1000; do
      $F mscr null $n 100 --jobs "$J" --primary-only --cfg "$C"
      $F mscr planted $n 100 --jobs "$J" --primary-only --cfg "$C"
    done ;;
  pdcor_null)
    for n in 500 1000; do $F pdcor null $n 100 --jobs "$J" --primary-only --cfg "$C"; done ;;
  pdcor_p500)
    $F pdcor planted 500 100 --jobs "$J" --primary-only --cfg "$C" ;;
  pdcor_p1000)
    $F pdcor planted 1000 ${REPS_P1000:-30} --jobs "$J" --primary-only --cfg "$C" ;;
  *) echo "unknown PART $PART"; exit 2 ;;
esac
python scratchpad/xmethod/citests/f4.py summary $O --out $OUT/F4_heavy_summary_$PART.json
