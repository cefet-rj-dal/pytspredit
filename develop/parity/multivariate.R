#!/usr/bin/env Rscript
# Parity reference for the 0.5 multivariate family (retroactive M3 backfill).
# Invariant = the target (y) 5-step-ahead forecast on the canonical tsd-based MV system
# (x1 = shifted y; x2 = 3-point moving average with NA->y), mirroring the R examples.
# Fidelity: TsVar (native OLS) and TsLmMv/TsRegswMv (orchestration + _RLinear/knn) are
# value-wise exact; TsArimax is backend (stats::arima+xreg vs statsmodels). Mirrors
# ts_{var,lm_mv,regsw_mv,arimax}.R, exporting develop/parity/multivariate_ref.json.
# Guard: tests/test_quality_multivariate.py. See develop/parity.md.
suppressWarnings(suppressMessages({
  library(daltoolbox)
  library(tspredit)
  if (!requireNamespace("jsonlite", quietly = TRUE)) stop("jsonlite not installed")
  library(jsonlite)
}))

y <- read.csv("src/pytspredit/data/tsd.csv")$y
x1 <- c(y[-1], tail(y, 1))
x2 <- stats::filter(y, rep(1 / 3, 3), sides = 1)
x2[is.na(x2)] <- y[is.na(x2)]
mv <- ts_data_mv(data.frame(y = y, x1 = x1, x2 = as.numeric(x2)), y = "y")
samp <- ts_sample(mv, test_size = 5)
h <- 5

ref <- list(test_size = h, pred = list())

# VAR (native OLS + AICc order search); no known future
vm <- daltoolbox::fit(ts_var(p_max = 3), samp$train)
ref$pred$var <- as.numeric(predict(vm, steps_ahead = h))

# LmMv with the known future covariates (y ~ x1 + x2)
lm <- daltoolbox::fit(ts_lm_mv(formula = "y ~ x1 + x2"), samp$train)
ref$pred$lm_mv <- as.numeric(predict(lm, samp$test, steps_ahead = h))

# RegswMv: KNN target + periodic auxiliaries (deterministic orchestration)
rm <- ts_regsw_mv(
  model_y = ts_mv_spec(ts_knn(ts_norm_gminmax(), k = 3), variables = c("y", "x1", "x2")),
  models_x = list(x1 = ts_mv_spec(ts_deterministic("periodic", period = 7)),
                  x2 = ts_mv_spec(ts_deterministic("periodic", period = 7))),
  window_size = 5)
rm <- daltoolbox::fit(rm, samp$train)
ref$pred$regsw_mv <- as.numeric(predict(rm, steps_ahead = h))

# ArimaX with known future covariates (backend: stats::arima + xreg)
ax <- daltoolbox::fit(ts_arimax(p = 2, d = 0, q = 0), samp$train)
ref$pred$arimax <- as.numeric(predict(ax, samp$test, steps_ahead = h))

writeLines(toJSON(ref, digits = NA, auto_unbox = TRUE),
           "develop/parity/multivariate_ref.json")
cat(sprintf("wrote develop/parity/multivariate_ref.json (%d models)\n", length(ref$pred)))
