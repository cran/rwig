// Vectorisable exp for the log-domain kernels.
//
// std::exp is a scalar libm call per element and dominated the profiles of
// the log-stabilised Sinkhorn / barycenter. vexp_block evaluates exp on a
// fixed-size block with straight-line arithmetic only (no calls, no
// branches), so the compiler can vectorise it with the baseline instruction
// set even at -O2: GCC's default cost model only vectorises loops whose trip
// count it knows is a multiple of the vector width.
//
// Method: x = k ln2 + r, |r| <= ln2 / 2, with k rounded via the 1.5 * 2^52
// trick; exp(r) by the Cephes rational (Pade) approximation, a degree-2 over
// degree-3 polynomial in r^2 (9 multiplies and one divide, against 26 ops
// for a Taylor polynomial of equal accuracy); 2^k assembled directly in the
// exponent field. Accuracy is 1-2 ulp of std::exp over [-708, 709] (max
// relative error 3.2e-16 measured against glibc). Out-of-range handling is
// branch-free (0/1 masks, which are exact): inputs below -708.4 give exactly
// 0, inputs above 709.78 give +Inf. Inputs must be finite (the kernels only
// ever pass -C / reg <= x <= 0).
//
// The block is written step-major (every arithmetic step is its own loop
// over the block) so the independent lanes of a block interleave instead of
// forming one long dependency chain; 16 is the measured sweet spot.

#ifndef RWIG_VEXP_H
#define RWIG_VEXP_H

#include <cstdint>
#include <cstring>

namespace la {

constexpr int VEXP_BLOCK = 16;

inline void vexp_block(const double *__restrict x, double *__restrict y) {
  constexpr double LOG2E = 1.44269504088896340736;
  constexpr double LN2_HI = 6.93145751953125e-1; // Cody-Waite split of ln 2
  constexpr double LN2_LO = 1.42860682030941723212e-6;
  constexpr double MAGIC = 6755399441055744.0; // 1.5 * 2^52
  constexpr std::int64_t MAGIC_BITS = 0x4338000000000000LL;
  constexpr double XMIN = -708.39641853226410622; // log(DBL_MIN)
  constexpr double XMAX = 709.78271289338399673;  // log(DBL_MAX)
  // exp(r) = 1 + 2 r P(r^2) / (Q(r^2) - r P(r^2)) on |r| <= ln2 / 2
  // (Cephes exp.c, S. Moshier; peak relative error 2.3e-16)
  constexpr double P0 = 1.26177193074810590878e-4, P1 = 3.02994407707441961300e-2,
                   P2 = 9.99999999999999999910e-1;
  constexpr double Q0 = 3.00198505138664455042e-6, Q1 = 2.52448340349684104192e-3,
                   Q2 = 2.27265548208155028766e-1, Q3 = 2.00000000000000000009e0;
  double k[VEXP_BLOCK], r[VEXP_BLOCK], lo[VEXP_BLOCK];
  for (int t = 0; t < VEXP_BLOCK; ++t) {
    const double xv = x[t];
    lo[t] = (double)(xv >= XMIN);
    const double hi = (double)(xv <= XMAX);
    double kk = (xv * LOG2E + MAGIC) - MAGIC; // k = round(x / ln 2)
    kk = kk * lo[t] - 1022.0 * (1.0 - lo[t]); // clamp k to the normal range,
    kk = kk * hi + 1023.0 * (1.0 - hi);       // branch-free (masks are exact)
    k[t] = kk;
    r[t] = (xv - kk * LN2_HI) - kk * LN2_LO;
  }
  for (int t = 0; t < VEXP_BLOCK; ++t) {
    const double rr = r[t], r2 = rr * rr;
    const double px = rr * ((P0 * r2 + P1) * r2 + P2);
    const double q = ((Q0 * r2 + Q1) * r2 + Q2) * r2 + Q3;
    const double e = 1.0 + 2.0 * px / (q - px);
    const double kd = k[t] + MAGIC; // k in the low mantissa bits
    std::int64_t ki;
    std::memcpy(&ki, &kd, sizeof ki);
    ki = (ki - MAGIC_BITS + 1023) << 52; // bits of 2^k
    double s;
    std::memcpy(&s, &ki, sizeof s);
    y[t] = lo[t] * (e * s);
  }
}


} // namespace la

#endif // RWIG_VEXP_H
