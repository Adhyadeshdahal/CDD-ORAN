# F2 v2 (a) EXACT: R momentchi2 (lpb4 / hbe / sw) on the (x, weights) cases written by f2_rcit.py gen
# (m2_cases.csv: case, x, py_lpb4, py_hbe, py_sw, w_1..w_k, empty padding). Writes m2_r.csv (case, lpb4, hbe, sw).
#   Rscript scratchpad/xmethod/citests/f2_m2.R DIR
args <- commandArgs(trailingOnly = TRUE)
d <- args[1]
suppressPackageStartupMessages(library(momentchi2))
a <- read.csv(file.path(d, "m2_cases.csv"), header = FALSE)
res <- t(sapply(seq_len(nrow(a)), function(i) {
  v <- as.numeric(a[i, ])
  w <- v[6:length(v)]; w <- w[!is.na(w)]
  c(case = v[1], lpb4 = lpb4(w, v[2]), hbe = hbe(w, v[2]), sw = sw(w, v[2]))
}))
write.csv(res, file.path(d, "m2_r.csv"), row.names = FALSE)
cat("momentchi2", as.character(packageVersion("momentchi2")), "cases", nrow(res), "\n")
