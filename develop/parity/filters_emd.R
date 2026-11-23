#!/usr/bin/env Rscript
# Parity reference for the native EMD filters vs the R hht engine.
# Mirrors ts_fil_emd.R / ts_fil_remd.R (hht::CEEMD) on the tsd+outlier example and
# exports the filtered series + 1st IMF to develop/parity/filters_emd_ref.json.
# The Python guard (tests/test_quality_emd.py) compares the native filter output to
# this reference with a LOOSE tolerance (EMD is ensemble/stochastic + different backend).
suppressWarnings(suppressMessages({
  if (!requireNamespace("hht", quietly=TRUE)) stop("hht not installed")
  if (!requireNamespace("jsonlite", quietly=TRUE)) stop("jsonlite not installed")
  library(hht); library(jsonlite)
}))

y <- read.csv("src/pytspredit/data/tsd.csv")$y
yo <- y; yo[9] <- 2 * yo[9]          # outlier at R index 9 (Python index 8)

set.seed(1)
id <- seq_along(yo)
ce <- suppressWarnings(hht::CEEMD(yo, id, noise.amp = 0.1, trials = 50, verbose = FALSE))
imf1 <- as.numeric(ce$imf[, 1])
emd_filtered <- yo - imf1            # ts_fil_emd: remove 1st IMF

ref <- list(
  input = yo,
  emd_imf1 = imf1,
  emd_filtered = emd_filtered,
  n = length(yo)
)
writeLines(toJSON(ref, pretty = TRUE, digits = NA, auto_unbox = TRUE),
           "develop/parity/filters_emd_ref.json")
cat(sprintf("wrote develop/parity/filters_emd_ref.json (n=%d, nimf=%d)\n",
            length(yo), ncol(ce$imf)))
