#!/usr/bin/env Rscript
# Parity reference for the 0.2 model zoo (retroactive M3 backfill).
# DETERMINISTIC, value-wise exact both for the rule-based trio (periodic/persist) and for
# the window-linear models (darima/warma), whose _RLinear reproduces R `lm` column-dropping
# EXACTLY -- including the degenerate rank-3 pure-sine (tsd) case. Invariant = the one-step
# per-row forecast + the 5-step-ahead forecast. Mirrors ts_{deterministic,darima,warma}.R,
# exporting develop/parity/modelzoo_ref.json. Guard: tests/test_quality_modelzoo.py.
# Inputs: frozen develop/parity/logistic.csv (well-conditioned) + src/.../tsd.csv (degenerate).
suppressWarnings(suppressMessages({
  library(daltoolbox)
  library(tspredit)
  if (!requireNamespace("jsonlite", quietly = TRUE)) stop("jsonlite not installed")
  library(jsonlite)
}))

logistic <- read.csv("develop/parity/logistic.csv")$y
tsd <- read.csv("src/pytspredit/data/tsd.csv")$y
sw <- 8
test_size <- 5

split_io <- function(series) {
  samp <- ts_sample(ts_data(series, sw), test_size = test_size)
  list(tr = ts_projection(samp$train), te = ts_projection(samp$test))
}

fit_one_multi <- function(model, s) {
  io <- split_io(s)
  model <- daltoolbox::fit(model, x = io$tr$input, y = io$tr$output)
  one <- as.numeric(predict(model, io$te$input))
  multi <- as.numeric(predict(model, io$te$input[1, ], steps_ahead = test_size))
  list(one = one, multi = multi, steps = if (is.null(model$steps)) NA else model$steps)
}

ref <- list(sw = sw, test_size = test_size, models = list())

# --- rule-based deterministic (explicit tiny series == the unit-test cases) ---
pm <- daltoolbox::fit(ts_deterministic("periodic", period = 7), c(4, 5, 6, 7, 1, 2, 3))
ref$models$periodic <- list(series = c(4, 5, 6, 7, 1, 2, 3), h = 5,
                            pred = as.numeric(predict(pm, steps_ahead = 5)))
ps <- daltoolbox::fit(ts_deterministic("persist"), c(10, 11, 11, 12))
ref$models$persist <- list(series = c(10, 11, 11, 12), h = 3,
                           pred = as.numeric(predict(ps, steps_ahead = 3)))

# --- window-linear (darima / warma) ---
ref$models$darima_diff_logistic <- fit_one_multi(ts_darima(ts_norm_diff(), input_size = 5), logistic)
ref$models$darima_none_logistic <- fit_one_multi(ts_darima(ts_norm_none(), input_size = 5), logistic)
ref$models$darima_none_tsd      <- fit_one_multi(ts_darima(ts_norm_none(), input_size = 5), tsd)
ref$models$warma_steps1_logistic <- fit_one_multi(ts_warma(input_size = 5, steps = 1), logistic)
ref$models$warma_auto_logistic   <- fit_one_multi(ts_warma(input_size = 5, steps = NA), logistic)

writeLines(toJSON(ref, digits = NA, auto_unbox = TRUE),
           "develop/parity/modelzoo_ref.json")
cat(sprintf("wrote develop/parity/modelzoo_ref.json (%d models)\n", length(ref$models)))
