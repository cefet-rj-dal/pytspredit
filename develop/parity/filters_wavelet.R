#!/usr/bin/env Rscript
# Parity reference for the native MODWT wavelet filter vs the R `wavelets` package.
# MODWT is a deterministic linear transform, so the native port is compared VALUE-WISE.
# Mirrors ts_fil_wavelet.R (modwt, zero level-1 detail, imodwt) for each filter family
# on tsd, exporting develop/parity/filters_wavelet_ref.json. The Python guard
# (tests/test_quality_wavelet.py) compares the native filter output to this reference.
suppressWarnings(suppressMessages({
  if (!requireNamespace("wavelets", quietly = TRUE)) stop("wavelets not installed")
  if (!requireNamespace("jsonlite", quietly = TRUE)) stop("jsonlite not installed")
  library(wavelets); library(jsonlite)
}))

filters <- c("haar", "d4", "la8", "bl14", "c6")
y <- read.csv("src/pytspredit/data/tsd.csv")$y

denoise <- function(sig, f) {
  wt <- modwt(sig, filter = f, boundary = "periodic")
  wt@W[[1]] <- as.matrix(rep(0, length(wt@W[[1]])), ncol = 1)
  as.numeric(imodwt(wt))
}

ref <- list(input = y, filters = filters, denoised = list())
for (f in filters) ref$denoised[[f]] <- denoise(y, f)

writeLines(toJSON(ref, digits = NA, auto_unbox = TRUE),
           "develop/parity/filters_wavelet_ref.json")
cat(sprintf("wrote develop/parity/filters_wavelet_ref.json (%d filters)\n", length(filters)))
