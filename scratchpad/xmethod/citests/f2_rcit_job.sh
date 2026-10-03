# Kaggle/Colab job: install R + RCIT if missing, run the F2 rcot2-vs-RCIT comparison. Outputs -> $OUT.
set -x
OUT=${JOB_OUT:-out}; mkdir -p "$OUT"
which Rscript || (apt-get update -qq && apt-get install -y -qq r-base-core r-base-dev > "$OUT/apt.log" 2>&1)
Rscript --version 2>&1 | tee "$OUT/r_version.txt"
Rscript -e 'options(repos="https://cloud.r-project.org"); for (p in c("remotes","momentchi2","MASS")) if (!requireNamespace(p, quietly=TRUE)) install.packages(p); remotes::install_github("ericstrobl/RCIT", upgrade="never"); cat(as.character(packageVersion("RCIT")), "\n")' > "$OUT/r_install.log" 2>&1
tail -3 "$OUT/r_install.log"
git ls-remote https://github.com/ericstrobl/RCIT HEAD > "$OUT/rcit_commit.txt" 2>&1
python -m pip install -q momentchi2==0.1.8 > "$OUT/pip.log" 2>&1
export SKIP_PERM=${SKIP_PERM:-0}
python scratchpad/xmethod/citests/f2_rcit.py gen "$OUT/f2" --reps ${REPS:-200}
Rscript scratchpad/xmethod/citests/f2_rcit.R "$OUT/f2"
Rscript scratchpad/xmethod/citests/f2_m2.R "$OUT/f2"          # (a) exact momentchi2 py vs R -> m2_r.csv
python scratchpad/xmethod/citests/f2_rcit.py compare "$OUT/f2"
rm -f "$OUT"/f2/*_n*_z*.csv
