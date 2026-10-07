library(tinytest)

# Regression tests for the performance work: the fast paths must give the
# same numbers as the reference paths.

set.seed(11)
tol <- 1e-10

# synthetic WDL problem (vocab N, docs M), C unnormalised as in wdl()
N <- 30; M <- 50; S <- 3; reg <- .1
emb <- matrix(rnorm(N * 5), N, 5)
C <- as.matrix(dist(emb))
Y <- matrix(runif(N * M), N, M); Y <- sweep(Y, 2, colSums(Y), "/")

############################################################
# 1. batched WDL inference equals the per-document parallel barycenter
#    (zero_tol tiny so both run exactly max_iter iterations)
############################################################
set.seed(1)
fit <- rwig:::wdl_cpp(Y, C, reg, S, 16L, 1L, FALSE, 20L, 1e-12,
                      2L, .005, .01, .9, .999, 1e-8, FALSE, 42L)
for (m in c(1, 7, M)) {
  sol <- barycenter(fit$A, C, fit$W[, m], barycenter_control = list(
    reg = reg, method = "parallel", use_cuda = FALSE, max_iter = 20, zero_tol = 1e-12))
  expect_equal(sol$b, fit$Yhat[, m], tolerance = tol)
}

############################################################
# 2. threaded log solvers equal the serial ones
############################################################
a <- runif(N); a <- a / sum(a); b <- runif(M); b <- b / sum(b)
Cr <- C[, 1] %o% rep(1, M) + abs(outer(seq_len(N), seq_len(M), "-")) / N
ctl <- list(reg = .05, method = "log", with_grad = TRUE, max_iter = 200, zero_tol = 1e-9)
s0 <- sinkhorn(a, b, Cr, modifyList(ctl, list(n_threads = 0)))
s3 <- sinkhorn(a, b, Cr, modifyList(ctl, list(n_threads = 3)))
expect_equal(s0$P, s3$P, tolerance = tol)
expect_equal(s0$grad_a, s3$grad_a, tolerance = tol)
expect_equal(s0$iter, s3$iter)
A <- matrix(runif(N * S), N, S); A <- sweep(A, 2, colSums(A), "/")
w <- c(.2, .3, .5); bN <- runif(N); bN <- bN / sum(bN)
b0 <- barycenter(A, C / max(C), w, bN, modifyList(ctl, list(n_threads = 0)))
b3 <- barycenter(A, C / max(C), w, bN, modifyList(ctl, list(n_threads = 3)))
expect_equal(b0$b, b3$b, tolerance = tol)
expect_equal(b0$grad_A, b3$grad_A, tolerance = tol)
expect_equal(b0$grad_w, b3$grad_w, tolerance = tol)

############################################################
# 3. log and vanilla Sinkhorn agree on a well-conditioned problem
############################################################
ctl2 <- list(reg = .2, with_grad = TRUE, max_iter = 2000, zero_tol = 1e-12, use_cuda = FALSE)
sv <- sinkhorn(a, b, Cr, modifyList(ctl2, list(method = "vanilla")))
sl <- sinkhorn(a, b, Cr, modifyList(ctl2, list(method = "log")))
expect_equal(sv$P, sl$P, tolerance = 1e-8)
expect_equal(sv$loss, sl$loss, tolerance = 1e-8)
expect_equal(sv$grad_a, sl$grad_a, tolerance = 1e-6)
