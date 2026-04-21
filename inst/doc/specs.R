## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ----setup--------------------------------------------------------------------
library(rwig) |> suppressPackageStartupMessages()

## ----eval=FALSE---------------------------------------------------------------
# wig_control = list(
#   group_unit = "month",
#   svd_method = "docs",
#   standardize = TRUE
# )

## ----eval=FALSE---------------------------------------------------------------
# type = "cbow"
# dim = 10
# min_count = 1

## ----eval=FALSE---------------------------------------------------------------
# with_grad = TRUE

## ----eval=FALSE---------------------------------------------------------------
# optimizer_control = list(
#   optimizer = "adamw",
#   lr = .005,
#   decay = .01,
#   beta1 = .9,
#   beta2 = .999,
#   eps = 1e-8
# )

