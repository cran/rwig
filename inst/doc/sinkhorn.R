## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ----setup--------------------------------------------------------------------
library(rwig) |> suppressPackageStartupMessages()

## -----------------------------------------------------------------------------
a <- c(.3, .4, .1, .1, .1)
b <- c(.4, .5, .1)

## -----------------------------------------------------------------------------
C <- rbind(
  c(.1, .2, .3),
  c(.2, .3, .4),
  c(.4, .3, .2),
  c(.3, .2, .1),
  c(.5, .5, .4)
)
reg <- .1

## -----------------------------------------------------------------------------
sinkhorn(
  a,
  b,
  C,
  sinkhorn_control = list(
    reg = reg,
    verbose = 0
  )
)

## -----------------------------------------------------------------------------
sol_vanilla <- sinkhorn(
  a,
  b,
  C,
  sinkhorn_control = list(
    reg = reg,
    method = "vanilla",
    verbose = 0
  )
)
sol_log <- sinkhorn(
  a,
  b,
  C,
  sinkhorn_control = list(
    reg = reg,
    method = "log",
    verbose = 0
  )
)

## -----------------------------------------------------------------------------
sol_thread <- sinkhorn(
  a,
  b,
  C,
  sinkhorn_control = list(
    reg = reg,
    method = "log",
    n_threads = 4,
    verbose = 0
  )
)

## -----------------------------------------------------------------------------
sol_auto <- sinkhorn(
  a,
  b,
  C,
  sinkhorn_control = list(
    reg = reg,
    threshold = .1,
    verbose = 0
  )
)

## -----------------------------------------------------------------------------
sol_grad <- sinkhorn(
  a,
  b,
  C,
  sinkhorn_control = list(
    reg = reg,
    with_grad = TRUE,
    verbose = 0
  )
)

sol_grad$grad_a

## ----eval=FALSE---------------------------------------------------------------
# sinkhorn_control = list(
#   reg = .1,
#   with_grad = FALSE,
#   method = "auto",
#   threshold = .1,
#   max_iter = 1000L,
#   zero_tol = 1e-6,
#   verbose = 0L
# )

## -----------------------------------------------------------------------------
sol_verbose <- sinkhorn(
  a,
  b,
  C,
  sinkhorn_control = list(
    reg = reg,
    verbose = 10
  )
)

