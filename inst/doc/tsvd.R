## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ----setup--------------------------------------------------------------------
library(rwig) |> suppressPackageStartupMessages()

## -----------------------------------------------------------------------------
A <- rbind(c(1,3), c(2,-4))

tsvd(A, 1, "auto")
tsvd(A, 1, "sklearn")
tsvd(A, 1, "none")

