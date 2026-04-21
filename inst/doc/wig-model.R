## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ----setup--------------------------------------------------------------------
library(rwig) |> suppressPackageStartupMessages()

## -----------------------------------------------------------------------------
# create a small dataset
wigdf <- data.frame(
  ref_date = as.Date(c("2012-01-01", "2012-02-01")),
  docs = c("this is a sentence", "this is another sentence")
)

wigfit <- wig(wigdf, ref_date, docs, specs = wig_specs(
  wdl_control = list(num_topics = 2),
  word2vec_control = list(min_count = 1)
))
wigfit

