# F2: RCIT::RCoT p-values on the CSVs written by f2_rcit.py gen (see that file).
args <- commandArgs(trailingOnly = TRUE)
d <- args[1]
suppressPackageStartupMessages(library(RCIT))
out <- list()
for (f in list.files(d, pattern = "[.]csv$")) {
  if (f %in% c("rcit.csv", "m2_cases.csv", "m2_r.csv")) next
  a <- read.csv(file.path(d, f))
  for (r in sort(unique(a$rep))) {
    b <- a[a$rep == r, ]
    x <- b$x; y <- b$y; z <- as.matrix(b[, grep("^z", names(b))])
    ph <- RCoT(x, y, z, approx = "hbe", num_f = 25, num_f2 = 5, seed = r)$p
    pl <- RCoT(x, y, z, approx = "lpd4", num_f = 25, num_f2 = 5, seed = r)$p
    pp <- if (Sys.getenv("SKIP_PERM") == "1") NA else RCoT(x, y, z, approx = "perm", num_f = 25, num_f2 = 5, seed = r)$p
    pf <- RCoT(x, y, z, seed = r)$p          # RCIT defaults: approx lpd4, num_f 100, num_f2 5
    out[[length(out) + 1]] <- data.frame(file = f, rep = r, p_hbe = ph, p_lpd4 = pl, p_perm = pp, p_lpd4_f100 = pf)
  }
  cat(f, "done\n")
}
write.csv(do.call(rbind, out), file.path(d, "rcit.csv"), row.names = FALSE)
cat("RCIT", as.character(packageVersion("RCIT")), "\n")
