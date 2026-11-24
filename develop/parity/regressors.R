#!/usr/bin/env Rscript
# Parity reference for the forecast regressors (retroactive M3 backfill, 0.3 theme).
# Invariant = the 5-step-ahead predicted vector on the shared tsd split. Fidelity classes:
#   deterministic (knn; elm purelin with nhid==n_lags == OLS)      -> value-wise ~1e-6
#   backend (svm = e1071 SVR; arima = stats::arima MLE)            -> near-exact tolerance
#   stochastic (rf = randomForest; mlp = nnet)                     -> loose guard only
# Mirrors ts_{knn,elm,svm,arima,rf,mlp}.R on ts_data(tsd$y,10), ts_sample(test_size=5),
# exporting develop/parity/regressors_ref.json. Guard: tests/test_quality_regressors.py.
# See develop/parity.md.
suppressWarnings(suppressMessages({
  library(daltoolbox)
  library(tspredit)
  if (!requireNamespace("jsonlite", quietly = TRUE)) stop("jsonlite not installed")
  library(jsonlite)
}))

y <- read.csv("src/pytspredit/data/tsd.csv")$y
sw <- 10
test_size <- 5
ts <- ts_data(y, sw)
samp <- ts_sample(ts, test_size = test_size)
tr <- ts_projection(samp$train)
te <- ts_projection(samp$test)
actual <- as.numeric(te$output)

predict5 <- function(model) {
  model <- daltoolbox::fit(model, x = tr$input, y = tr$output)
  as.numeric(predict(model, x = te$input[1, ], steps_ahead = test_size))
}

pred <- list()
# deterministic
pred$knn <- predict5(ts_knn(ts_norm_gminmax(), 4, ts_lagmap("pacf"), k = 3))
pred$elm_ols <- predict5(ts_elm(ts_norm_gminmax(), 4, ts_lagmap("recent"),
                                nhid = 4, actfun = "purelin"))
# backend
pred$svm <- predict5(ts_svm(ts_norm_gminmax(), 4, ts_lagmap("seasonal", seasonality = 4)))
# arima works on the RAW series (ts_reg, sw = 1) -- NOT the sliding-window ts_data above.
rts <- ts_data(y, 1)
rsamp <- ts_sample(rts, test_size = test_size)
am <- daltoolbox::fit(ts_arima(p = 5, d = 0, q = 0), x = rsamp$train)
pred$arima <- as.numeric(predict(am, x = rsamp$test[1, ], steps_ahead = test_size))
# stochastic (seed only makes the REFERENCE reproducible; Python RNG differs -> loose guard)
set.seed(1); pred$rf <- predict5(ts_rf(ts_norm_gminmax(), 9))
set.seed(1); pred$mlp <- predict5(ts_mlp(ts_norm_gminmax(), 4, ts_lagmap("even"),
                                         size = 4, decay = 0))

ref <- list(input = y, sw = sw, test_size = test_size, actual = actual,
            models = names(pred), pred = pred)
writeLines(toJSON(ref, digits = NA, auto_unbox = TRUE),
           "develop/parity/regressors_ref.json")
cat(sprintf("wrote develop/parity/regressors_ref.json (%d models)\n", length(pred)))
