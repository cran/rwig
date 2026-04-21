## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ----setup--------------------------------------------------------------------
library(rwig) |> suppressPackageStartupMessages()

## -----------------------------------------------------------------------------
# a very simple example
sentences <- c("this is a sentence", "this is another one", "yet another sentence")
wdl_fit <- wdl(sentences, specs = wdl_specs(
  wdl_control = list(num_topics = 2),
  word2vec_control = list(min_count = 1)
))

wdl_fit

## -----------------------------------------------------------------------------
wdl_fit$topics

## -----------------------------------------------------------------------------
wdl_fit$weights

