// Fused kernels for the log-domain (log-stabilized) Sinkhorn / barycenter
// algorithms.
//
// All kernels work on the implicit matrix
//     R_ij = C_ij - f_i - g_j          (C is M x N, column-major, ld = M)
// without ever forming it. The "soft-min" of a row is
//     min_j R_ij - reg * log sum_j exp(-(R_ij - min_j R_ij) / reg)
// (Xie 2025); likewise for columns. The backward passes need the row- and
// column-softmax matrices X and W of -R / reg applied to a vector. With the
// row / column soft-mins s of the same R (the forward pass has them), the
// softmax entries are exp((s_i - R_ij) / reg) directly, so the products are
// one fused sweep over C per call: no M x N buffer is formed, and every
// thread streams its own contiguous column block of C.

#ifndef RWIG_LOGDOMAIN_H
#define RWIG_LOGDOMAIN_H

#include <cmath>
#include <limits>

#include "linalg.hpp"
#include "thread_pool.hpp"
#include "vexp.hpp"

namespace logdom {

// problem description shared by all kernels
struct Problem {
  const double *C;
  int M, N;
  double reg;
};

/*
  serial kernels on a row range [i0, i1) or column range [j0, j1)

  The exp-heavy inner loops go through la::vexp_block on fixed-size blocks:
  the arguments (m - R_ij) / reg of a block are gathered into a small local
  array, exponentiated in one vectorised sweep, then stored / accumulated.
*/

namespace detail {

constexpr int B = la::VEXP_BLOCK;

// Runs use(i, exp(arg(i))) for i in [i0, i1): the arguments of a block of
// B lanes are gathered into a local array, exponentiated in one vectorised
// sweep, then consumed. The last partial block is padded with zeros and
// only its first lanes are consumed, so callers never see the padding.
template <class Arg, class Use>
inline void exp_blocks(int i0, int i1, Arg arg, Use use) {
  double t[B], e[B];
  int i = i0;
  for (; i + B <= i1; i += B) {
    for (int b = 0; b < B; ++b) t[b] = arg(i + b);
    la::vexp_block(t, e);
    for (int b = 0; b < B; ++b) use(i + b, e[b]);
  }
  if (i < i1) {
    const int n = i1 - i;
    for (int b = 0; b < B; ++b) t[b] = b < n ? arg(i + b) : 0.0;
    la::vexp_block(t, e);
    for (int b = 0; b < n; ++b) use(i + b, e[b]);
  }
}

// The kernels below work on one column j of R: Cj points at column j of C,
// so R_ij = Cj[i] - f[i] - gj, and every exponent is (shift - R_ij) / reg
// with a shift that keeps it <= 0.

// sum_i exp((m - R_ij) / reg) over i in [0, M)
inline double col_exp_sum(const double *Cj, const double *f, double gj,
                          double m, double invreg, int M) {
  double s = 0.0;
  exp_blocks(
      0, M, [=](int i) { return (m - (Cj[i] - f[i] - gj)) * invreg; },
      [&](int, double e) { s += e; });
  return s;
}

// rowsum_i += exp((rowmin_i - R_ij) / reg) for i in [i0, i1)
inline void row_exp_acc(const double *Cj, const double *f, double gj,
                        const double *rowmin, double invreg, int i0, int i1,
                        double *rowsum) {
  exp_blocks(
      i0, i1, [=](int i) { return (rowmin[i] - (Cj[i] - f[i] - gj)) * invreg; },
      [=](int i, double e) { rowsum[i] += e; });
}

// sum_i exp((s_i - R_ij) / reg) * x_i over i in [0, M), s the row soft-min
inline double col_exp_dot(const double *Cj, const double *f, double gj,
                          const double *s, const double *x, double invreg,
                          int M) {
  double acc = 0.0;
  exp_blocks(
      0, M, [=](int i) { return (s[i] - (Cj[i] - f[i] - gj)) * invreg; },
      [&](int i, double e) { acc += e * x[i]; });
  return acc;
}

// acc_i += exp((tj - R_ij) / reg) * yj for i in [0, M), tj the column soft-min
inline void row_exp_axpy(const double *Cj, const double *f, double gj,
                         double tj, double yj, double invreg, int M,
                         double *acc) {
  exp_blocks(
      0, M, [=](int i) { return (tj - (Cj[i] - f[i] - gj)) * invreg; },
      [=](int i, double e) { acc[i] += e * yj; });
}

// min_i R_ij over column j
inline double col_min(const double *Cj, const double *f, double gj, int M) {
  double m = std::numeric_limits<double>::infinity();
  for (int i = 0; i < M; ++i) {
    const double r = Cj[i] - f[i] - gj;
    if (r < m) m = r;
  }
  return m;
}

} // namespace detail

// row minima of R(f, g)
inline void row_min(const Problem &p, const double *f, const double *g, int i0,
                    int i1, double *rowmin) {
  const double inf = std::numeric_limits<double>::infinity();
  for (int i = i0; i < i1; ++i) rowmin[i] = inf;
  for (int j = 0; j < p.N; ++j) {
    const double *Cj = p.C + (std::size_t)j * p.M;
    const double gj = g[j];
    for (int i = i0; i < i1; ++i) {
      const double r = Cj[i] - f[i] - gj;
      if (r < rowmin[i]) rowmin[i] = r;
    }
  }
}

// row soft-min: out[i] = rowmin_i - reg * log(rowsum_i); rowmin / rowsum are
// caller-provided scratch of length M and hold the raw min / sum on return
inline void row_lse(const Problem &p, const double *f, const double *g, int i0,
                    int i1, double *out, double *rowmin, double *rowsum) {
  row_min(p, f, g, i0, i1, rowmin);
  for (int i = i0; i < i1; ++i) rowsum[i] = 0.0;
  const double invreg = 1.0 / p.reg;
  for (int j = 0; j < p.N; ++j) {
    const double *Cj = p.C + (std::size_t)j * p.M;
    detail::row_exp_acc(Cj, f, g[j], rowmin, invreg, i0, i1, rowsum);
  }
  for (int i = i0; i < i1; ++i)
    out[i] = rowmin[i] - p.reg * std::log(rowsum[i]);
}

// column soft-min: out[j] = colmin_j - reg * log(colsum_j)
inline void col_lse(const Problem &p, const double *f, const double *g, int j0,
                    int j1, double *out) {
  const double invreg = 1.0 / p.reg;
  for (int j = j0; j < j1; ++j) {
    const double *Cj = p.C + (std::size_t)j * p.M;
    const double m = detail::col_min(Cj, f, g[j], p.M);
    const double s = detail::col_exp_sum(Cj, f, g[j], m, invreg, p.M);
    out[j] = m - p.reg * std::log(s);
  }
}

/*
  pooled operations (rows or columns split across the pool's threads)
*/

// scratch of the pooled operations
struct Scratch {
  la::Vec rowmin, rowsum; // raw row minima / sums of the row soft-min (M)
  la::Mat part;           // per-chunk partial sums of apply_W, sized on use
  void resize(int M) {
    rowmin.resize(M);
    rowsum.resize(M);
  }
};

// out (M) = row soft-min of R(f, g)
inline void soft_min_rows(ThreadPool &pool, const Problem &p, const double *f,
                          const double *g, double *out, Scratch &s) {
  double *rowmin = s.rowmin.data(), *rowsum = s.rowsum.data();
  pool.parallel_for(p.M, [=](int i0, int i1, int) {
    row_lse(p, f, g, i0, i1, out, rowmin, rowsum);
  });
}

// out (N) = column soft-min of R(f, g)
inline void soft_min_cols(ThreadPool &pool, const Problem &p, const double *f,
                          const double *g, double *out) {
  pool.parallel_for(p.N, [=](int j0, int j1, int) { col_lse(p, f, g, j0, j1, out); });
}

// y (N) = alpha * X^T x with X the row-wise softmax of -R(f, g) / reg,
// given rlse (M), the row soft-min of R(f, g):
//   y_j = alpha * sum_i exp((rlse_i - R_ij) / reg) x_i
// Each thread owns a column block of C and writes only its part of y.
inline void apply_XT(ThreadPool &pool, const Problem &p, const double *f,
                     const double *g, const double *rlse, double alpha,
                     const double *x, double *y) {
  const double invreg = 1.0 / p.reg;
  pool.parallel_for(p.N, [=](int j0, int j1, int) {
    for (int j = j0; j < j1; ++j) {
      const double *Cj = p.C + (std::size_t)j * p.M;
      y[j] = alpha * detail::col_exp_dot(Cj, f, g[j], rlse, x, invreg, p.M);
    }
  });
}

// x (M) = alpha * W y with W the column-wise softmax of -R(f, g) / reg,
// given clse (N), the column soft-min of R(f, g):
//   x_i = alpha * sum_j exp((clse_j - R_ij) / reg) y_j
// Each chunk accumulates its column block of C into its own partial sum
// (a column of s.part), then the partial sums are reduced over row blocks.
// A row-block split would need no reduction but touches a short strided
// slab of every column, which prefetches badly: 3.7x on 12 threads against
// 10x here.
inline void apply_W(ThreadPool &pool, const Problem &p, const double *f,
                    const double *g, const double *clse, double alpha,
                    const double *y, double *x, Scratch &s) {
  const double invreg = 1.0 / p.reg;
  const int M = p.M, N = p.N;
  const int T = pool.chunks(N);
  if (s.part.nrow() != (la::idx)M || s.part.ncol() < (la::idx)T)
    s.part.resize(M, T);
  double *part = s.part.data();
  pool.parallel_for(N, [=](int j0, int j1, int t) {
    double *acc = part + (std::size_t)t * M;
    for (int i = 0; i < M; ++i) acc[i] = 0.0;
    for (int j = j0; j < j1; ++j) {
      const double *Cj = p.C + (std::size_t)j * M;
      detail::row_exp_axpy(Cj, f, g[j], clse[j], alpha * y[j], invreg, M, acc);
    }
  });
  pool.parallel_for(M, [=](int i0, int i1, int) {
    for (int i = i0; i < i1; ++i) {
      double a = 0.0;
      for (int t = 0; t < T; ++t) a += part[(std::size_t)t * M + i];
      x[i] = a;
    }
  });
}

} // namespace logdom

#endif // RWIG_LOGDOMAIN_H
