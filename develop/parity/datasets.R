#!/usr/bin/env Rscript
# Parity reference exporter for the Eidolon dataset miniatures.
# Reads the R miniatures (.RData) and writes develop/parity/datasets_ref.json with
# the full numeric content (precision NA) + string labels + schema, in a canonical
# flatten order. The Python guard (tests/test_quality_datasets.py) replays the same
# flatten on the shipped CSV/JSON miniatures and asserts equality (atol 1e-9), so
# parity is checked WITHOUT needing R at test time.
suppressWarnings(suppressMessages(library(jsonlite)))
mini_dir <- "C:/Git/dal/tspredbench/data"           # the R miniatures (source of truth)
out <- "C:/Git/dal/pytspredit/develop/parity/datasets_ref.json"

num_cols <- function(df) names(df)[vapply(df, is.numeric, logical(1))]
chr_cols <- function(df) names(df)[!vapply(df, is.numeric, logical(1))]
flat_df_num <- function(df) as.numeric(unlist(lapply(df[num_cols(df)], as.numeric), use.names = FALSE))
flat_df_lab <- function(df) as.character(unlist(lapply(df[chr_cols(df)], as.character), use.names = FALSE))

fingerprint <- function(obj) {
  if (is.data.frame(obj)) {
    return(list(kind = "dataframe", nrow = nrow(obj), ncol = ncol(obj),
                columns = names(obj), numeric = flat_df_num(obj), labels = flat_df_lab(obj)))
  }
  el <- obj[[1]]
  if (is.data.frame(el)) {
    num <- c(); lab <- c()
    for (k in names(obj)) { num <- c(num, flat_df_num(obj[[k]])); lab <- c(lab, flat_df_lab(obj[[k]])) }
    return(list(kind = "df_list", keys = names(obj), numeric = num, labels = lab))
  }
  if (is.list(el)) {
    num <- c()
    for (k in names(obj)) for (sk in names(obj[[k]])) num <- c(num, as.numeric(obj[[k]][[sk]]))
    return(list(kind = "nested_list", keys = names(obj), numeric = num, labels = character(0)))
  }
  num <- as.numeric(unlist(obj, use.names = FALSE))
  list(kind = "series_list", keys = names(obj), numeric = num, labels = character(0))
}

files <- sort(list.files(mini_dir, pattern = "[.]RData$", full.names = FALSE))
ref <- list()
for (f in files) {
  e <- new.env(parent = emptyenv()); nm <- load(file.path(mini_dir, f), envir = e)
  ref[[nm[1]]] <- fingerprint(get(nm[1], envir = e))
}
writeLines(toJSON(ref, pretty = TRUE, auto_unbox = TRUE, na = "null", null = "null", digits = NA), out)
cat(sprintf("wrote %s (%d datasets)\n", out, length(ref)))
