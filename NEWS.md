# rwig 0.3.0

Performance work driven by `perf` profiles; all solver outputs are checked
against frozen outputs of version 0.2.0 (relative differences below 1e-11)
and the Julia-derived reference solutions in the tests.

- `wdl()`/`wig()` on the CPU now predict the document barycenters with the
  same batched Gibbs-kernel iteration the training uses, on a whole batch of
  documents at once. Before, inference solved one barycenter per document
  and, with the default `method = "auto"`, did so in the log domain, which
  cost about 20 times the whole training (2000 NYT headlines, 1 epoch:
  33 minutes, now 33 seconds). `method`, `threshold` and `n_threads` of the
  barycenter control are documented as unused by WDL.
- The log-stabilized `sinkhorn()` and `barycenter()` exponentiate with a
  vectorised rational approximation (1-2 ulp of `exp()`) instead of one
  libm call per matrix entry, and log `sinkhorn()` no longer computes a
  column soft-min per iteration whose only use, the column term of the
  error, is identically zero. 1000 x 1000 log Sinkhorn, 1000 iterations:
  19 s to 8 s on one thread.
- Worker threads (`n_threads`) spin briefly between the short parallel
  sections of an iteration instead of sleeping; 12 threads now give about
  5x on the forward passes where they gave 3x.
- The backward passes of the log-stabilized `sinkhorn()` and `barycenter()`
  (`with_grad = TRUE`) no longer materialize the M x N softmax matrices: the
  forward pass keeps its row and column soft-mins (M + N numbers per
  iteration), from which each adjoint product is one fused sweep over the
  cost matrix, split over the threads by column blocks. 1000 x 1000 log
  Sinkhorn with gradient, 1000 iterations: 15.6 s to 13.2 s on one thread,
  3.8 s to 2.1 s on 12; 800 x 800 log barycenter with gradients: 37.9 s to
  33.9 s and 10.9 s to 5.2 s.
- Kernel products with up to 4 right-hand columns (the "parallel"
  `barycenter()`) use one `dsymv`/`dgemv` per column rather than a level-3
  call that repacks the N x N kernel every time: 800 x 800 barycenter with
  4 sources, 0.79 s to 0.34 s.
- CUDA `wdl()`: the per-document `cublasDgemv` launches of the weight
  adjoint are one strided-batched GEMM (2000 headlines: 5.6 s to 4.5 s).

# rwig 0.2.0

- Dropped the dependencies on Rcpp and RcppArmadillo. The C++ code talks to
  R through the native C API (`.Call` with registered routines) and calls the
  BLAS/LAPACK libraries shipped with R directly through a small internal
  matrix layer, which also makes the installed package much smaller. The
  package now has no compiled-code dependencies at all.
- Interrupting a long computation (Ctrl-C) now releases worker threads and
  buffers before returning to R, and surfaces as an R error.
- Faster log-stabilized `sinkhorn()` and `barycenter()`: the soft-min
  kernels no longer materialize the M x N matrix `C - f 1' - 1 g'` on every
  iteration, and worker threads (`n_threads`) are created once per call
  instead of once per iteration.
- Faster `wdl()` on CPU: the softmax Jacobian is applied in O(N) instead of
  O(N^2) per column, an unused matrix product per backward step was removed,
  and the per-batch history buffers use about half the memory.
- `wdl()` now draws its random initialization from R's RNG (`rnorm()`), so
  `set.seed()` reproduces the documented R sequence. Fits with the same seed
  therefore differ from version 0.1.0, which used Armadillo's own generator.
- Fixed: `wdl()` returned `weights`, `docs_pred` and `docs_dist` in the
  shuffled training order (the default `shuffle = TRUE`) next to `docs` in
  the input order, so `wig()` summed document scores into the wrong
  periods. Per-document outputs are now put back into the input order.
- Fixed: `wig()` failed when `wig_control` was given without `group_unit`.
- `wdl_specs()`/`wig_specs()` now honor the values passed in
  `barycenter_control` (`method`, `max_iter`, ...) instead of silently
  overriding them; the `wdl_control` seed entry is named `seed` (it was
  documented as `rng_seed` but read as `seed`). `verbose` defaults to 0
  when omitted from a partial `sinkhorn_control`/`barycenter_control`.
- Dropped the dependency on RhpcBLASctl: `rwig` no longer sets the BLAS
  thread count to 1 for the whole session when attached. The startup message
  now explains the recommendation and, if RhpcBLASctl is installed (it is
  now only suggested), reports the current thread count and the call to
  change it.
- Dropped the dependency on lubridate: `group_unit` is now passed to
  `cut()` (`?cut.Date`), so it accepts "day", "week", "month", "quarter",
  "year" and multiples such as "2 months". Weeks start on Monday.
- Fixed: threaded (`n_threads > 0`) log barycenter crashed when the cost
  matrix had more columns than rows.
- Fixed: the CUDA build passes an explicit GPU architecture to `nvcc`
  (recent CUDA toolkits default to `sm_52`, which lacks double-precision
  `atomicAdd`). Set `RWIG_CUDA_ARCH` to override the detected flag, or
  `RWIG_NO_CUDA=1` to build without GPU support even when CUDA is installed.

# rwig 0.1.0

- Initial CRAN submission.
- Efficient implementation of several Optimal Transport algorithms in  
   Fangzhou Xie (2025) <doi:10.48550/arXiv.2504.08722> and
  the Wasserstein Index Generation (WIG) model in
  Fangzhou Xie (2020) <doi:10.1016/j.econlet.2019.108874>.
