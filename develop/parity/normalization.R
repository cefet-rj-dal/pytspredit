#!/usr/bin/env Rscript
# Parity reference for the TsNorm* family (retroactive M3 backfill, 0.1 theme).
# Normalizations are DETERMINISTIC transforms, so the full transformed window matrix
# is compared VALUE-WISE (tol <= 1e-9). Mirrors ts_norm_{none,gminmax,swminmax,diff,an}.R
# on ts_data(tsd$y, 10), exporting develop/parity/normalization_ref.json. The Python
# guard (tests/test_quality_normalization.py) compares TsNorm* output to this reference.
# See develop/parity.md. Run from the repo root with tspredit installed:
#   /c/R/R-4.5.0/bin/Rscript.exe develop/parity/normalization.R
suppressWarnings(suppressMessages({
  library(daltoolbox)
  library(tspredit)
  if (!requireNamespace("jsonlite", quietly = TRUE)) stop("jsonlite not installed")
  library(jsonlite)
}))

y <- read.csv("src/pytspredit/data/tsd.csv")$y
sw <- 10
ts <- ts_data(y, sw)

# One builder per named config (name -> fitted ts_norm_* object).
builders <- list(
  none               = function() ts_norm_none(),
  gminmax            = function() ts_norm_gminmax(),
  swminmax           = function() ts_norm_swminmax(),
  diff               = function() ts_norm_diff(),
  an_default         = function() ts_norm_an(),
  an_subtract        = function() ts_norm_an(operation = "subtract"),
  an_ema_softdiv_mad = function() ts_norm_an(average = "ema", operation = "softdivide", scale = "mad"),
  an_asinh_sd        = function() ts_norm_an(operation = "asinh", scale = "sd")
)

as_rows <- function(m) {
  m <- as.matrix(m)
  lapply(seq_len(nrow(m)), function(i) as.numeric(m[i, ]))
}

ref <- list(input = y, sw = sw, configs = names(builders), transformed = list())
for (nm in names(builders)) {
  obj <- daltoolbox::fit(builders[[nm]](), ts)
  tr <- transform(obj, ts)
  ref$transformed[[nm]] <- as_rows(tr)
}

writeLines(toJSON(ref, digits = NA, auto_unbox = TRUE),
           "develop/parity/normalization_ref.json")
cat(sprintf("wrote develop/parity/normalization_ref.json (%d configs)\n", length(builders)))
