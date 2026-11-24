#!/usr/bin/env Rscript
# Parity reference for the TsFil* family (retroactive M3 backfill, 0.4 theme; EMD + wavelet
# already have their own guards). Invariant = the full filtered series on the shared tsd
# with one injected outlier (y[9] doubled, == Python 0-based index 8). Most filters are
# DETERMINISTIC and compared value-wise; qes/seas_adj/spline use different backends
# (HoltWinters / stats::stl / smooth.spline) and get a loose guard. Mirrors ts_fil_*.R,
# exporting develop/parity/filters_ref.json. Guard: tests/test_quality_filters.py.
suppressWarnings(suppressMessages({
  library(daltoolbox)
  library(tspredit)
  if (!requireNamespace("jsonlite", quietly = TRUE)) stop("jsonlite not installed")
  library(jsonlite)
}))

y <- read.csv("src/pytspredit/data/tsd.csv")$y
y[9] <- 2 * y[9]   # inject outlier (R 1-based index 9 == Python 0-based 8)

builders <- list(
  none      = function() ts_fil_none(),
  ma        = function() ts_fil_ma(3),
  ema       = function() ts_fil_ema(3),
  winsor    = function() ts_fil_winsor(),
  recursive = function() ts_fil_recursive(0.05),
  fft       = function() ts_fil_fft(),
  smooth    = function() ts_fil_smooth(),
  hp        = function() ts_fil_hp(100),
  gabor     = function() ts_fil_gabor(3, 0.1),
  lowess    = function() ts_fil_lowess(0.2),
  ses       = function() ts_fil_ses(),
  qes       = function() ts_fil_qes(),
  seas_adj  = function() ts_fil_seas_adj(12),
  spline    = function() ts_fil_spline(0.5),
  kalman    = function() ts_fil_kalman()
)

ref <- list(input = y, filters = names(builders), filtered = list())
for (nm in names(builders)) {
  obj <- daltoolbox::fit(builders[[nm]](), y)
  ref$filtered[[nm]] <- as.numeric(transform(obj, y))
}

writeLines(toJSON(ref, digits = NA, auto_unbox = TRUE, na = "null"),
           "develop/parity/filters_ref.json")
cat(sprintf("wrote develop/parity/filters_ref.json (%d filters)\n", length(builders)))
