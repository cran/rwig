
// implement the WDL algorithm in this header file
// https://arxiv.org/abs/2504.08722
// Section 7.2 and 7.3

#ifndef RWIG_WDL_H
#define RWIG_WDL_H

#include <cmath>

#include "common.hpp"

// WIG-related headers
#include "barycenter_impl.hpp" // using the barycenter algorithms
#include "optimizer.hpp"       // using the optimizers: SGD, Adam, AdamW
#include "timer.hpp"           // TicToc timer class

class WassersteinDictionaryLearning {

private:
  // data
  la::Mat _Y;  // document matrix (N x M)
  la::Mat _C;  // distance matrix (N x N)
  double _reg; // regularization

  // model dimensions
  la::idx _N; // number of tokens
  la::idx _M; // number of docs
  la::idx _S; // number of topics

  // model related
  int _B; // number of docs per batch
  int _E; // number of epochs

  // batched Gibbs-kernel barycenter (training and inference)
  int _maxiter;    // max iterations per batch
  double _zerotol; // convergence tolerance

  // optimizer related
  int _opt; // optimizer mode (SGD, Adam, AdamW)
  double _eta, _gamma, _beta1, _beta2, _eps; // AdamW params

  // logging
  bool _verbose;

  // latent vars
  la::Mat _Alpha;  // N * S
  la::Mat _Lambda; // S * M

  // precomputed Gibbs kernel exp(-C / reg), N x N
  la::KernelOp _K;

  // TicToc timer class
  TicToc _timer; // used for logging each iteration time

  // gradients for latent vars: Alpha, Lambda
  la::Mat _g_Alpha;  // gradient of Alpha (for A)
  la::Mat _g_Lambda; // gradient of Lambda (for W)
  la::Vec _g_lambda; // batch-averaged gradient of one column of Lambda

  // optimizers for latent vars
  Optimizer _opt_Alpha, _opt_Lambda;

  // batched barycenter scratch, sized once for B docs (N x S*B unless noted)
  la::Mat _UB, _VB, _KVB, _KTUB, _UBbar, _VBbar, _ABbar, _tmpB;
  la::Mat _bB, _bBbar; // N x B
  la::Mat _wBbar;      // S x B
  // history buffers, flat: slot l occupies [l * N*S*B, (l+1) * N*S*B)
  la::Vec _KVB_hist;  // L slots
  la::Vec _KTUB_hist; // L + 1 slots
  la::Vec _bB_hist;   // L + 1 slots of N*B

  // column-wise softmax: out[:, j] = softmax(in[:, j])
  static void _softmax_cols(const la::Mat &in, la::Mat &out) {
    const la::idx nr = in.nrow(), nc = in.ncol();
    for (la::idx j = 0; j < nc; ++j) {
      const double *x = in.col(j);
      double *y = out.col(j);
      double m = x[0];
      for (la::idx i = 1; i < nr; ++i) m = std::max(m, x[i]);
      double s = 0.0;
      for (la::idx i = 0; i < nr; ++i) {
        y[i] = std::exp(x[i] - m);
        s += y[i];
      }
      for (la::idx i = 0; i < nr; ++i) y[i] /= s;
    }
  }

  // softmax for both latent vars (A and W)
  void _softmax() {
    // A = softmax(Alpha), W = softmax(Lambda), column-wise
    _softmax_cols(_Alpha, this->A);
    _softmax_cols(_Lambda, this->W);
  }

  // multiply the softmax Jacobian: g <- (diag(a) - a a^T) g = a % g - a (a . g)
  static void _softmax_jac(const double *a, double *g, la::idx n) {
    double d = 0.0;
    for (la::idx i = 0; i < n; ++i) d += a[i] * g[i];
    for (la::idx i = 0; i < n; ++i) g[i] = a[i] * g[i] - a[i] * d;
  }

  ///////////////////////////////////////////
  // training
  ///////////////////////////////////////////

  void _compute_serial();
  // documents in batch `batch_id` (the last batch takes the remainder)
  int _batch_docs(int batch_id) const {
    return batch_id == (int)(_M / _B) ? (int)(_M % _B) : _B;
  }
  void _forward_step_batched(int D, const double *wB);
  void _train_batch_batched(int batch_id);
  void _infer_batch_batched(int batch_id);
  // optimizer step: update _Alpha, _Lambda in place
  void _optimize() {
    _opt_Alpha.step(_opt, _Alpha, _g_Alpha);
    _opt_Lambda.step(_opt, _Lambda, _g_Lambda);
  }

public:
  // output vars
  la::Mat A;    // topics, N * S
  la::Mat W;    // weight, S * M
  la::Mat Yhat; // predicted barycenters, N * M

  // init the class
  WassersteinDictionaryLearning(const int batch_size, const int epochs,
                                const int max_iter = 1000,
                                const double zero_tol = 1e-6,
                                const int optimizer = 2,
                                const double eta = .001,
                                const double gamma = .01,
                                const double beta1 = .9,
                                const double beta2 = .999,
                                const double eps = 1e-8,
                                const bool verbose = false) {
    _B = batch_size;
    _E = epochs;
    _maxiter = max_iter;
    _zerotol = zero_tol;
    _opt = optimizer;
    _eta = eta;
    _gamma = gamma;
    _beta1 = beta1;
    _beta2 = beta2;
    _eps = eps;
    _verbose = verbose;
  }
  // destructor
  ~WassersteinDictionaryLearning() {}

  // init the data
  void init_data(const la::Mat &Y, const la::Mat &C, double reg, int S) {
    // load the data
    _Y = Y;
    _C = C;
    _reg = reg;

    // dimension
    _M = _Y.ncol();
    _N = _Y.nrow();
    _S = S;
  }

  // actual compute method for WDL
  void compute() { _compute_serial(); }
};

#endif // RWIG_WDL_H
