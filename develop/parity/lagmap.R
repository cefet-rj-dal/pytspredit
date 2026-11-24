#!/usr/bin/env Rscript
# Parity reference for the TsLagmap family (retroactive M3 backfill, 0.1 theme).
# Lag selection is a DISCRETE decision, so the invariant is the SET of selected lags
# (convention-independent; 0/1-based position offsets are not compared). Mirrors
# ts_lagmap.R for all 12 strategies on ts_data(tsd$y, 10) with input_size=4,
# seasonality=4, exporting develop/parity/lagmap_ref.json. The Python guard
# (tests/test_quality_lagmap.py) asserts TsLagmap selects the identical lag sets.
# See develop/parity.md.
suppressWarnings(suppressMessages({
  library(daltoolbox)
  library(tspredit)
  if (!requireNamespace("jsonlite", quietly = TRUE)) stop("jsonlite not installed")
  library(jsonlite)
}))

y <- read.csv("src/pytspredit/data/tsd.csv")$y
sw <- 10
input_size <- 4
seasonality <- 4
io <- ts_projection(ts_data(y, sw))

methods <- c("recent", "even", "geom", "acf", "pacf", "peaks",
             "seasonal", "acf_seasonal", "pacf_seasonal", "blocks", "mi", "mrmr")

ref <- list(input = y, sw = sw, input_size = input_size, seasonality = seasonality,
            methods = methods, lags = list(), positions = list())
for (m in methods) {
  mapper <- ts_lagmap(method = m, seasonality = seasonality)
  mapper <- daltoolbox::fit(mapper, io$input, io$output, input_size = input_size)
  ref$lags[[m]] <- sort(as.integer(mapper$lags))
  ref$positions[[m]] <- sort(as.integer(mapper$positions))
}

writeLines(toJSON(ref, digits = NA, auto_unbox = TRUE),
           "develop/parity/lagmap_ref.json")
cat(sprintf("wrote develop/parity/lagmap_ref.json (%d methods)\n", length(methods)))
