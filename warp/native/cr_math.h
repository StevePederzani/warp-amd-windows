// Correctly rounded exp and pow, enabled with the module option correctly_rounded.
// Each function has three parts.
//   The fast path runs on the GPU's full-rate FP32 and produces the result in float-float, two floats with about 48 bits. exp
//     uses a Cody-Waite reduction made exact by FMA, 2^(j/128) stored as two floats, and a degree-5 polynomial. pow computes
//     log2 x from a 256-entry reciprocal table, then t = y log2 x as an exact product, then 2^t with the exp path. The table's
//     first and last entries are exactly 1 and 1/2, and the entries past sqrt 2 carry the exponent's 1, so a logarithm near 0
//     keeps every bit.
//   The rounding test checks whether a midpoint between floats lies within the fast result's own error bound, which is 2^-46
//     for exp and 2^-46 (1 + |t|) for pow. A subnormal result is checked on the subnormal grid.
//   The accurate path handles the calls the rounding test passes on. It uses Feynman's digit recurrence for the logarithm in
//     128-bit fixed point (62 factors (1 + 2^-k), with a table exact to 2^-124), computes t as a double-double, and rounds the
//     result in integers, ties to even. Before both, pow checks for exact results, where x^y is dyadic. Those are computed in
//     128 bits and rounded once, and every midpoint, where a tie can really occur, is one of them.
// Verified against CORE-MATH's crf_expf and crf_powf. exp was checked on all 4,294,967,296 float inputs, and pow on CORE-MATH's
// 129,187 hard-to-round pairs and on families of random, near-1, integer, n/2^k, and bit-pattern pairs. Every error-free step sits
// under '#pragma clang fp contract(off)', because a product fused into a later sum counts its rounding twice. hipRTC honors the
// pragma (-ffp-contract=fast-honor-pragmas), and so does the CPU's clang, which only contracts within a statement. NVRTC does not
// read clang's pragmas and fuses by default, so this header is used by the HIP and CPU builds only, never by a CUDA build.
#pragma once

#if defined(__CUDACC__) && !defined(__HIPCC__)
#error "correctly_rounded: the HIP port's AMD and processor builds only (NVRTC would fuse the error-free steps)"
#endif

#if defined(__HIPCC__)
#define WP_CR_TABLE static __constant__ const
#else
#define WP_CR_TABLE static const
#endif

#include "cr_math_tables.h"

namespace wp_cr {
typedef unsigned __int128 u128;
typedef __int128 i128;

inline CUDA_CALLABLE unsigned int bits_of(float f)
{
    union { float f; unsigned int u; } c;
    c.f = f;
    return c.u;
}
inline CUDA_CALLABLE float float_of(unsigned int u)
{
    union { float f; unsigned int u; } c;
    c.u = u;
    return c.f;
}

// ---- Float-Float Steps
inline CUDA_CALLABLE void two_sum(float a, float b, float& s, float& e)
{
#pragma clang fp contract(off)
    s = a + b;
    const float bb = s - a;
    e = (a - (s - bb)) + (b - bb);
}
inline CUDA_CALLABLE void fast_two_sum(float a, float b, float& s, float& e)
{
#pragma clang fp contract(off)
    s = a + b;
    e = b - (s - a);
}
inline CUDA_CALLABLE void two_prod(float a, float b, float& p, float& e)
{
#pragma clang fp contract(off)
    p = a * b;
    e = __builtin_fmaf(a, b, -p);
}

// ====================================================================== The Accurate Path, With Feynman's Logarithm in 128 Bits
inline CUDA_CALLABLE u128 tab(int k) { return ((u128)WP_CR_F128_L[k][0] << 64) | (u128)WP_CR_F128_L[k][1]; }

inline CUDA_CALLABLE int clz128(u128 v)
{
    const unsigned long long h = (unsigned long long)(v >> 64), l = (unsigned long long)v;
    return h ? __builtin_clzll(h) : (l ? 64 + __builtin_clzll(l) : 128);
}

// Rounds p 2^E (p > 0) once to the float grid, ties to even, keeping the subnormal spacing of 2^-149, and returns infinity past
// the largest float.
inline CUDA_CALLABLE float round_to_float(u128 p, int E)
{
    const int bits = 128 - clz128(p);
    int ex = E + bits - 1;
    if (ex > 127)
        return __builtin_huge_valf();
    const int keep_lsb = ex >= -126 ? ex - 23 : -149;
    const int sh = keep_lsb - E;
    unsigned long long M;
    if (sh <= 0) {
        M = (unsigned long long)(p << (-sh));
    } else if (sh > bits) {
        M = 0;
    } else if (sh >= 128) {
        M = (p == ((u128)1 << 127)) ? 0 : 1;
    } else {
        const u128 one = 1;
        M = (unsigned long long)(p >> sh);
        const u128 rem = p & ((one << sh) - 1), half = one << (sh - 1);
        if (rem > half || (rem == half && (M & 1)))
            ++M;
    }
    if (ex >= -126) {
        if (M == (1ULL << 24)) { M >>= 1; ++ex; }
        if (ex > 127) return __builtin_huge_valf();
        return float_of(((unsigned int)(ex + 127) << 23) | ((unsigned int)M & 0x7FFFFF));
    }
    return float_of((unsigned int)M);
}

// Converts an unsigned 128-bit integer to a double-double. hi holds its top 53 bits exactly, and lo holds the rest, rounded once.
inline CUDA_CALLABLE void dd_of(u128 u, double* hi, double* lo)
{
#pragma clang fp contract(off)
    const int bits = 128 - clz128(u);
    if (bits <= 53) {
        *hi = (double)(unsigned long long)u;
        *lo = 0.0;
        return;
    }
    const int sh = bits - 53;
    const unsigned long long top = (unsigned long long)(u >> sh);
    const u128 rest = u - ((u128)top << sh);
    *hi = __builtin_ldexp((double)top, sh);
    *lo = __builtin_ldexp((double)(unsigned long long)(rest >> 64), 64) + (double)(unsigned long long)rest;
}

// Returns log2(m) 2^124 for m = mant 2^-23 in [1, 2).
inline CUDA_CALLABLE u128 log2m(unsigned int mant)
{
#pragma clang fp contract(off)
    const u128 M = (u128)mant << 101;
    u128 P = (u128)1 << 124, acc = 0;
    for (int k = 1; k <= 62; ++k) {
        const u128 d = P >> k;
        if (d <= M - P) {
            P += d;
            acc += tab(k);
        }
    }
    const double eps = (double)(unsigned long long)(M - P) / __builtin_ldexp((double)(unsigned long long)(P >> 61), 61);
    const double l = eps * (1.0 - 0.5 * eps) * 1.4426950408889634073599246810019;
    return acc + (u128)(unsigned long long)(l * 0x1p124);
}

// ---- Double-Double Steps for the Accurate Path's Logarithm Near 1
inline CUDA_CALLABLE void two_sum_d(double a, double b, double& s, double& e)
{
#pragma clang fp contract(off)
    s = a + b;
    const double bb = s - a;
    e = (a - (s - bb)) + (b - bb);
}
inline CUDA_CALLABLE void fast_two_sum_d(double a, double b, double& s, double& e)
{
#pragma clang fp contract(off)
    s = a + b;
    e = b - (s - a);
}
inline CUDA_CALLABLE void dd_mul(double ah, double al, double bh, double bl, double& ph, double& pl)
{
#pragma clang fp contract(off)
    ph = ah * bh;
    double e = __builtin_fma(ah, bh, -ph);
    e = __builtin_fma(ah, bl, __builtin_fma(al, bh, e));
    fast_two_sum_d(ph, e, ph, pl);
}

// log2 e as a double-double, taken from the accurate path's 128-bit constant, whose high word is log2 e 2^60 and whose low word
// holds the rest at 2^-124.
inline CUDA_CALLABLE void log2e_dd(double* h, double* l)
{
#pragma clang fp contract(off)
    const unsigned long long hw = WP_CR_F128_LOG2E[0];
    const double hd = (double)hw;
    const long long left = (long long)(hw - (unsigned long long)hd);
    *h = __builtin_ldexp(hd, -60);
    *l = __builtin_ldexp((double)left, -60) + __builtin_ldexp((double)WP_CR_F128_LOG2E[1], -124);
}

// Computes log2(1 + d) for |d| < 2^-7 (d exact) as a double-double with a relative error near 2^-103. It sums the series
// (-d)^(k-1) / k by Horner's method in double-double from k = 20, where the first term left out is under 2^-140 relative, with
// each 1/k as a double-double built from its exact remainder. The recurrence's absolute error of 2^-113 is only about 2^-91
// relative at x = 1 + 2^-23, and |y| up to 2^32 multiplies it there.
inline CUDA_CALLABLE void log2_near1(double d, double* Lh, double* Ll)
{
#pragma clang fp contract(off)
    double sh = 1.0 / 20.0;
    double sl = __builtin_fma(-sh, 20.0, 1.0) / 20.0;
    for (int k = 19; k >= 1; --k) {
        double ph, pl;
        dd_mul(d, 0.0, sh, sl, ph, pl);                 // d S
        const double ih = 1.0 / (double)k;
        const double il = __builtin_fma(-ih, (double)k, 1.0) / (double)k;
        double s, e;
        two_sum_d(ih, -ph, s, e);                       // 1/k - d S
        e += il - pl;
        fast_two_sum_d(s, e, sh, sl);
    }
    double qh, ql, gh, gl;
    dd_mul(d, 0.0, sh, sl, qh, ql);                     // log(1 + d)
    log2e_dd(&gh, &gl);
    dd_mul(qh, ql, gh, gl, *Lh, *Ll);
}

// Computes log2(|x|) for a finite x != 0 as a double-double and stores a bound on its absolute error in *errL. Within 2^-7 of 1
// it uses the series above, which claims a relative error of 2^-100, and the largest error measured against mpmath on 190,385
// floats there was 2^-104.33. Elsewhere it joins e + log2 m in integers, with the recurrence's absolute error of 2^-113 (from the
// double tail after 62 factors and the shift to 2^-116), and the largest relative error measured was 2^-105.7.
inline CUDA_CALLABLE void log2dd(float x, double* Lh, double* Ll, double* errL)
{
#pragma clang fp contract(off)
    const unsigned int u = bits_of(x);
    int e = (int)((u >> 23) & 0xFF);
    unsigned int mant = u & 0x7FFFFF;
    if (e == 0) {
        int shift = 0;
        while (!(mant & 0x800000)) { mant <<= 1; ++shift; }
        e = 1 - shift;
    } else {
        mant |= 0x800000;
    }
    e -= 127;
    if ((e == 0 && mant < 0x810000u) || (e == -1 && mant > 0xFE0000u)) {
        const double d = e == 0 ? __builtin_ldexp((double)((int)mant - 0x800000), -23) : __builtin_ldexp((double)((int)mant - 0x1000000), -24);
        log2_near1(d, Lh, Ll);
        *errL = __builtin_fabs(*Lh) * 0x1p-100;
        return;
    }
    const i128 L = (i128)e * ((i128)1 << 116) + (i128)(log2m(mant) >> 8);
    double h, l;
    dd_of(L < 0 ? (u128)(-L) : (u128)L, &h, &l);
    if (L < 0) { h = -h; l = -l; }
    *Lh = __builtin_ldexp(h, -116);
    *Ll = __builtin_ldexp(l, -116);
    // The integers contribute 2^-113, and the double-double's own 106 bits matter once |L| is large (2^-99.3 absolute at
    // |L| = 127, measured against mpmath).
    *errL = 0x1p-112 + __builtin_fabs(*Lh) * 0x1p-104;
}

// Converts v in [0, 1) to a 2^124 fixed-point value exactly, down to its last bit at or above 2^-124, by shifting its 53-bit
// integer mantissa into place.
inline CUDA_CALLABLE u128 fix124(double v)
{
    if (!(v > 0.0)) return 0;
    int ex;
    const double mant = __builtin_frexp(v, &ex);
    const unsigned long long m = (unsigned long long)__builtin_ldexp(mant, 53);
    const int sh = 124 + ex - 53;
    if (sh >= 0) return (u128)m << sh;
    return -sh >= 64 ? 0 : (u128)(m >> (-sh));
}

// Computes 2^t for t = th + tl (|t| < 160), rounded once to float. Sets *undecided when the remainder was within this call's own
// error of a midpoint. err_t is the caller's bound on the absolute error of t.
inline CUDA_CALLABLE float exp2dd(double th, double tl, double err_t, bool* undecided)
{
#pragma clang fp contract(off)
    *undecided = false;
    double n = __builtin_floor(th);
    double fh, fl;
    {
        const double a = th - n, ab = a - th;
        const double ae = (th - (a - ab)) + (-n - ab);
        const double s = a + tl, sb = s - a;
        const double se = (a - (s - sb)) + (tl - sb);
        fh = s;
        fl = ae + se;
    }
    if (fh < 0.0) { const double v = 1.0 + fh; fl += fh - (v - 1.0); fh = v; n -= 1.0; }
    if (fh >= 1.0) { const double v = fh - 1.0; fl += (fh - 1.0) - v; fh = v; n += 1.0; }
    const double fl100 = fl * 0x1p100;
    const long long lo = (long long)fl100;
    const long long lo2 = (long long)((fl100 - (double)lo) * 0x1p24);
    // Keep the fraction as one signed integer at 2^124 and carry it into n either way. An integral th with tl just under 0 gives
    // fh = 0 and fl < 0, which wrapped around when it was treated as unsigned.
    const i128 ONE = (i128)1 << 124;
    i128 Rs = (i128)fix124(fh) + (i128)lo * ((i128)1 << 24) + (i128)lo2;
    if (Rs < 0) { Rs += ONE; n -= 1.0; }
    else if (Rs >= ONE) { Rs -= ONE; n += 1.0; }
    if (n >= 128.0) return __builtin_huge_valf();
    if (n < -151.0) return 0.0f;
    u128 R = (u128)Rs;
    u128 Q = (u128)1 << 124;
    for (int k = 1; k <= 62; ++k) {
        const u128 t = tab(k);
        if (t <= R) {
            R -= t;
            Q += Q >> k;
        }
    }
    const double q = __builtin_ldexp((double)(unsigned long long)(Q >> 61), -63);
    Q += (u128)(unsigned long long)(q * 0.6931471805599453094172321214581766 * (double)(unsigned long long)R);
    const int ni = (int)n;
    const int sh = ni >= -126 ? 101 : -ni - 25;
    const u128 one = 1;
    unsigned long long M = (unsigned long long)(Q >> sh);
    const u128 rem = Q & ((one << sh) - 1), half = one << (sh - 1);
    const u128 err = (u128)(unsigned long long)((err_t * 0.6931471805599453 + 0x1p-111) * 0x1p125) + 64;
    if ((rem > half ? rem - half : half - rem) <= err)
        *undecided = true;
    if (rem > half || (rem == half && (M & 1)))
        ++M;
    if (ni >= -126) {
        int ex = ni;
        if (M == (1ULL << 24)) { M >>= 1; ++ex; }
        if (ex >= 128) return __builtin_huge_valf();
        return float_of(((unsigned int)(ex + 127) << 23) | ((unsigned int)M & 0x7FFFFF));
    }
    return float_of((unsigned int)M);
}

// When undecided_out is given, it reports whether the accurate path's own error bound reached a midpoint. A probe counts these as
// the cases this path cannot settle on its own, and none of CORE-MATH's hard-to-round pairs should be among them.
inline CUDA_CALLABLE float expf_arb(float x, bool* undecided_out = nullptr)
{
#pragma clang fp contract(off)
    double LH, LL;
    log2e_dd(&LH, &LL);
    const double th = (double)x * LH;
    const double tl = __builtin_fma((double)x, LH, -th) + (double)x * LL;
    bool undecided;
    const float r = exp2dd(th, tl, __builtin_fabs(th) * 0x1p-103, &undecided);
    if (undecided_out) *undecided_out = undecided;
    return r;
}

inline CUDA_CALLABLE float powf_arb(float ax, float y, bool* undecided_out = nullptr)
{
#pragma clang fp contract(off)
    double Lh, Ll, errL;
    log2dd(ax, &Lh, &Ll, &errL);
    const double th = (double)y * Lh;
    const double tl = __builtin_fma((double)y, Lh, -th) + (double)y * Ll;
    if (undecided_out) *undecided_out = false;
    if (th > 200.0) return __builtin_huge_valf();
    if (th < -200.0) return 0.0f;
    bool undecided;
    // The error of t is |y| times the logarithm's error plus the product's own rounding (|t| 2^-106, bounded at 2^-103).
    const float r = exp2dd(th, tl, __builtin_fabs((double)y) * errL + __builtin_fabs(th) * 0x1p-103, &undecided);
    if (undecided_out) *undecided_out = undecided;
    return r;
}

// ======================================================================================================================= exp
inline CUDA_CALLABLE float exp_fast(float x, bool* handed)
{
#pragma clang fp contract(off)
    *handed = false;
    // This covers the whole range expf_cr leaves to it, [-104, 89]. The reduction stays exact there (|k| <= 19,206, and
    // k 0x1.63p-8 has at most 24 bits), a result past the largest float overflows in the final scaling exactly when it should,
    // and a subnormal result is checked on the 2^-149 grid below. Sending everything under -87.33 to the accurate path instead
    // made exp over x spread across [-104, 89] run at 1.77 G/s against AMD's 23.72 G/s, because each wave of 32 with one such
    // lane paid for the accurate path.
    if (!(x >= -104.0f && x <= 89.0f)) { *handed = true; return 0.0f; }
    const float kf = __builtin_rintf(x * 0x1.715476p+7f);
    const int k = (int)kf;
    const float r1 = __builtin_fmaf(-kf, 0x1.63p-8f, x);
    float p2h, p2l;
    two_prod(kf, -0x1.bd0106p-20f, p2h, p2l);
    float rh, re;
    two_sum(r1, -p2h, rh, re);
    const float rl = re - p2l - kf * 0x1.cf79acp-47f;
    const float q = rh * rh * rh * (0x1.555556p-3f + rh * (0x1.555556p-5f + rh * 0x1.111112p-7f));
    float sh, sl;
    two_prod(rh, rh, sh, sl);
    float u, ue;
    fast_two_sum(rh, 0.5f * sh, u, ue);
    float ph, pl;
    fast_two_sum(u, ue + (rl + 0.5f * sl + rh * rl + q), ph, pl);
    const int j = k & 127, e = k >> 7;
    const float th = WP_CR_EXP_HI[j], tl = WP_CR_EXP_LO[j];
    float mh, ml;
    two_prod(th, ph, mh, ml);
    float yh, ye;
    fast_two_sum(th, mh, yh, ye);
    float yl = ye + (ml + th * pl + tl + tl * ph);
    fast_two_sum(yh, yl, yh, yl);
    const float delta = yh * 0x1p-46f;
    if (e <= -127 || (e == -126 && yh < 1.0f)) {
        // A subnormal result is checked on the 2^-149 grid, as pow_fast does, with the fraction kept exact as dh + dl.
        const float inv = __builtin_ldexpf(1.0f, 149 + e);
        const float A = yh * inv, B = yl * inv;
        const float Ai = __builtin_rintf(A);
        float dh, dl;
        two_sum(A - Ai, B, dh, dl);
        const float dS = delta * inv;
        const float above = (dh - 0.5f) + dl, below = (dh + 0.5f) + dl;
        if (__builtin_fabsf(above) <= dS || __builtin_fabsf(below) <= dS) { *handed = true; return 0.0f; }
        return __builtin_ldexpf(Ai + (above > 0.0f ? 1.0f : (below < 0.0f ? -1.0f : 0.0f)), -149);
    }
    const unsigned int bits = bits_of(yh);
    const float half = float_of((bits & 0x7F800000u) - (24u << 23));
    const float low = (bits & 0x7FFFFFu) ? half : 0.5f * half;
    if (__builtin_fabsf(yl - half) <= delta || __builtin_fabsf(yl + low) <= delta) { *handed = true; return 0.0f; }
    return __builtin_ldexpf(yh + yl, e);
}

inline CUDA_CALLABLE float expf_cr(float x)
{
    if (x != x) return x;
    if (x > 89.0f) return __builtin_huge_valf();
    if (x < -104.0f) return 0.0f;
    bool handed;
    const float y = exp_fast(x, &handed);
    return handed ? expf_arb(x) : y;
}

// ======================================================================================================================= pow
// Exact results. x^y is dyadic when it is a power of two at any y with ex y whole, or when y = n / 2^k > 0 (n < 128, k <= 3)
// on a perfect 2^k-th power. These are computed in 128 bits and rounded once, and every midpoint is one of them.
inline CUDA_CALLABLE bool pow_exact(float ax, float y, float* out)
{
    const unsigned int uy = bits_of(y), ux = bits_of(ax);
    const int by = (int)((uy >> 23) & 0xFF);
    if (by == 0) return false;
    unsigned int my = (uy & 0x7FFFFF) | 0x800000;
    int ey = by - 150;
    const int tzy = __builtin_ctz(my);
    my >>= tzy;
    ey += tzy;
    const int bx = (int)((ux >> 23) & 0xFF);
    unsigned long long a = bx == 0 ? (ux & 0x7FFFFF) : ((ux & 0x7FFFFF) | 0x800000);
    long long ex = bx == 0 ? -149 : bx - 150;
    const int tzx = __builtin_ctzll(a);
    a >>= tzx;
    ex += tzx;
    if (a == 1) {
        if (ex == 0) { *out = 1.0f; return true; }      // |x| = 1 gives 1 for every finite y. The caller gives (-1)^odd its sign.
        long long R;
        if (ey > 30) {
            R = ((ex > 0) == (y > 0.0f)) ? 1000 : -1000;
        } else if (ey >= 0) {
            R = ex * (long long)my * (1LL << ey);
        } else {
            if (ey < -40) return false;
            const long long d = 1LL << (-ey);
            if (ex % d != 0) return false;
            R = (ex / d) * (long long)my;
        }
        if (ey <= 30 && y < 0.0f) R = -R;
        if (R > 200) { *out = __builtin_huge_valf(); return true; }
        if (R < -200) { *out = 0.0f; return true; }
        *out = round_to_float(1, (int)R);
        return true;
    }
    if (!(y > 0.0f) || (uy & 0x3FF)) return false;
    const int k = ey >= 0 ? 0 : -ey;
    if (k > 3 || (ey >= 0 && ey > 7)) return false;
    const unsigned long long n = ey >= 0 ? ((unsigned long long)my << ey) : (unsigned long long)my;
    if (n > 127) return false;
    for (int i = 0; i < k; ++i) {
        if (ex & 1) return false;
        unsigned long long r = (unsigned long long)__builtin_sqrtf((float)a);
        while (r * r > a) --r;
        while ((r + 1) * (r + 1) <= a) ++r;
        if (r * r != a) return false;
        a = r;
        ex /= 2;
    }
    u128 p = 1;
    const int abits = 64 - __builtin_clzll(a);
    if ((unsigned long long)(abits - 1) * n + 1 > 126) return false;   // a^n has at least (bits - 1) n + 1 bits, so past 126 there is no loop.
    for (unsigned long long i = 0; i < n; ++i) {
        if ((128 - clz128(p)) + abits > 126) return false;
        p *= a;
    }
    const long long E = ex * (long long)n;
    if (E > 2000 || E < -2000) return false;
    *out = round_to_float(p, (int)E);
    return true;
}

inline CUDA_CALLABLE void log2_ff(float x, float& Lh, float& Ll)
{
#pragma clang fp contract(off)
    const unsigned int b = bits_of(x);
    int e = (int)(b >> 23) - 127;
    const float m = float_of((b & 0x7FFFFF) | 0x3F800000u);
    const int j = (b >> 15) & 0xFF;
    if (j >= WP_CR_POW_SPLIT) e += 1;
    // r = m c - 1 is exact in one FMA. Every c has 9 significant bits (checked when the table is generated), so m c is a multiple
    // of 2^-32, and |r| < 2^-8 leaves r with at most 24 bits (the same form LLVM-libc uses). No two_prod, two_sum, or r_lo is
    // needed below.
    const float rh = __builtin_fmaf(m, WP_CR_POW_C[j], -1.0f);
    float sh, sl;
    two_prod(rh, rh, sh, sl);
    float ch, cl;
    two_prod(sh, rh, ch, cl);
    // r^3 = (sh + sl) rh, with sh rh computed exactly above and sl rh here. That term is about 2^-50 at the top of an entry, and
    // dropping it made a relative error of 2^-42.6 in the logarithm, multiplied by |t| in the result. 131 of 67 million pairs came
    // out wrong where log2 x was small and t was large, confirmed against CORE-MATH.
    cl = __builtin_fmaf(sl, rh, cl);
    float th3, tl3;
    two_prod(ch, WP_CR_POW_THIRD_H, th3, tl3);
    tl3 = __builtin_fmaf(ch, WP_CR_POW_THIRD_L, __builtin_fmaf(cl, WP_CR_POW_THIRD_H, tl3));
    const float tail = -sh * sh * (0.25f - rh * (0x1.99999ap-3f - rh * (0x1.555556p-3f - rh * 0x1.249250p-3f)));
    // fast_two_sum is safe because |r| >= r^2 / 2 >= r^3 / 3 for |r| < 2^-8, so the order is known.
    float u, ue;
    fast_two_sum(rh, -0.5f * sh, u, ue);
    float v, ve;
    fast_two_sum(u, th3, v, ve);
    float lh, ll;
    fast_two_sum(v, ue + ve + (tl3 - 0.5f * sl + tail), lh, ll);
    float qh, ql;
    two_prod(lh, WP_CR_POW_INVLN2_H, qh, ql);
    ql = __builtin_fmaf(lh, WP_CR_POW_INVLN2_L, __builtin_fmaf(ll, WP_CR_POW_INVLN2_H, ql));
    fast_two_sum(qh, ql, qh, ql);
    float wh, we;
    two_sum(WP_CR_POW_EH[j], qh, wh, we);
    fast_two_sum(wh, we + (WP_CR_POW_EL[j] + ql), wh, we);
    if (e == 0) { Lh = wh; Ll = we; return; }
    float zh, ze;
    fast_two_sum((float)e, wh, zh, ze);   // Here |e| >= 1 > |wh|.
    fast_two_sum(zh, ze + we, Lh, Ll);
}

inline CUDA_CALLABLE void exp2_ff(float th, float tl, float& yh, float& yl, int& e2)
{
#pragma clang fp contract(off)
    const float kf = __builtin_rintf(th * 128.0f);
    const int k = (int)kf;
    float rh, rl;
    // th - k/128 is exact and is either 0 or a multiple of ulp(th) >= 2 |tl|, so fast_two_sum holds.
    fast_two_sum(th - kf * 0.0078125f, tl, rh, rl);
    float zh, zl;
    two_prod(rh, WP_CR_POW_LN2_H, zh, zl);
    zl = __builtin_fmaf(rh, WP_CR_POW_LN2_L, __builtin_fmaf(rl, WP_CR_POW_LN2_H, zl));
    fast_two_sum(zh, zl, zh, zl);
    const float q = zh * zh * zh * (0x1.555556p-3f + zh * (0x1.555556p-5f + zh * 0x1.111112p-7f));
    float sh, sl;
    two_prod(zh, zh, sh, sl);
    float u, ue;
    fast_two_sum(zh, 0.5f * sh, u, ue);
    float ph, pl;
    // With z = zh + zl, z^2 / 2 is sh / 2 + sl / 2 + zh zl. The cross term of z^3 / 6, zh^2 zl / 2 = sh zl / 2, is kept too,
    // since it can reach 2^-49.6.
    fast_two_sum(u, ue + (zl + 0.5f * sl + zh * zl + 0.5f * sh * zl + q), ph, pl);
    const int j = k & 127;
    e2 = k >> 7;
    const float t_h = WP_CR_EXP_HI[j], t_l = WP_CR_EXP_LO[j];
    // T (1 + p) = t_h + t_l + t_h ph + (t_h pl + t_l ph). The three parts near 2^-23 (the table's t_l, the rounding left by
    // t_h + t_h ph, and the next one) are added exactly before anything rounds, so the only rounding left is of a part under
    // ulp(yh) / 2, at most 2^-48 of the result. Summing all of them as floats rounded at 2^-47 three times.
    float mh, ml;
    two_prod(t_h, ph, mh, ml);
    float y0, ye;
    fast_two_sum(t_h, mh, y0, ye);
    float s1, e1;
    two_sum(ye, t_l, s1, e1);
    const float rest = e1 + (ml + (t_h * pl + t_l * ph));   // Not called "small", because Windows' rpcndr.h defines that as char.
    float w;
    fast_two_sum(y0, s1, yh, w);
    yl = w + rest;
    fast_two_sum(yh, yl, yh, yl);
}

inline CUDA_CALLABLE float pow_fast(float ax, float y, bool* handed)
{
#pragma clang fp contract(off)
    *handed = false;
    if (__builtin_fabsf(y) >= 0x1p32f) return ((y > 0.0f) == (ax > 1.0f)) ? __builtin_huge_valf() : 0.0f;
    float sc = 0.0f;
    if (ax < 0x1p-126f) {
        ax *= 0x1p24f;
        sc = -24.0f;
    }
    float Lh, Ll;
    log2_ff(ax, Lh, Ll);
    if (sc != 0.0f) {
        float a, b;
        two_sum(Lh, sc, a, b);
        fast_two_sum(a, b + Ll, Lh, Ll);
    }
    float th, tl;
    two_prod(y, Lh, th, tl);
    tl = __builtin_fmaf(y, Ll, tl);
    fast_two_sum(th, tl, th, tl);
    if (th >= 129.0f) return __builtin_huge_valf();
    if (th <= -151.0f) return 0.0f;
    // The overflow edge needs no accurate path. With the mantissa rounded correctly, scaling it by 2^e2 overflows to infinity
    // exactly when it should.
    float yh, yl;
    int e2;
    exp2_ff(th, tl, yh, yl, e2);
    // The rounding test's window is 1.1875 2^-46 (1 + |t|), so the proven error bound stays at least a quarter bit inside it at
    // every |t|. At 2^-46 the margin was only 0.034 bits once log2_ff switched to its one-FMA r (2^-46.13).
    const float delta = yh * 0x1.3p-46f * (1.0f + __builtin_fabsf(th));
    if (e2 <= -127 || (e2 == -126 && yh < 1.0f)) {
        const float inv = __builtin_ldexpf(1.0f, 149 + e2);
        const float A = yh * inv, B = yl * inv;
        const float Ai = __builtin_rintf(A);
        // d = (A - Ai) + B is kept exact as dh + dl. As a single float it rounded at 2^-25 of the subnormal spacing, which is
        // wider than dS for results near 2^-136, so a value that close to a midpoint could be read on the wrong side. dh -/+ 0.5
        // is exact (Sterbenz), so each distance below rounds far below dS.
        float dh, dl;
        two_sum(A - Ai, B, dh, dl);
        const float dS = delta * inv;
        const float above = (dh - 0.5f) + dl, below = (dh + 0.5f) + dl;
        if (__builtin_fabsf(above) <= dS || __builtin_fabsf(below) <= dS) { *handed = true; return 0.0f; }
        return __builtin_ldexpf(Ai + (above > 0.0f ? 1.0f : (below < 0.0f ? -1.0f : 0.0f)), -149);
    }
    const unsigned int bits = bits_of(yh);
    const float half = float_of((bits & 0x7F800000u) - (24u << 23));
    const float low = (bits & 0x7FFFFFu) ? half : 0.5f * half;
    if (__builtin_fabsf(yl - half) <= delta || __builtin_fabsf(yl + low) <= delta) { *handed = true; return 0.0f; }
    return __builtin_ldexpf(yh + yl, e2);
}

// Handle IEEE 754 and C special values first. A signaling NaN gives NaN even where a quiet NaN gives 1, as in CORE-MATH.
inline CUDA_CALLABLE float powf_cr(float x, float y)
{
    const float inf = __builtin_huge_valf();
    const unsigned int sx = bits_of(x), sy = bits_of(y);
    if (((sx & 0x7F800000u) == 0x7F800000u && (sx & 0x7FFFFFu) && !(sx & 0x400000u)) ||
        ((sy & 0x7F800000u) == 0x7F800000u && (sy & 0x7FFFFFu) && !(sy & 0x400000u)))
        return x + y;
    if (y == 0.0f || x == 1.0f) return 1.0f;
    if (x != x || y != y) return x + y;
    const float ax = __builtin_fabsf(x);
    if (y == inf || y == -inf) {
        if (ax == 1.0f) return 1.0f;
        return (y > 0.0f) == (ax > 1.0f) ? inf : 0.0f;
    }
    const bool yint = __builtin_truncf(y) == y;
    const bool yodd = yint && __builtin_fabsf(y) < 16777216.0f && ((long long)y & 1);
    if (x == 0.0f) {
        if (y < 0.0f) return yodd ? __builtin_copysignf(inf, x) : inf;
        return yodd ? x : 0.0f;
    }
    if (x == inf) return y > 0.0f ? inf : 0.0f;
    if (x == -inf) {
        if (y > 0.0f) return yodd ? -inf : inf;
        return yodd ? -0.0f : 0.0f;
    }
    if (x < 0.0f && !yint) return __builtin_nanf("");
    const bool flip = x < 0.0f && yodd;
    // |x| = 1 gives 1 for every finite y, with the sign set above. This comes before the fast path, because its |y| >= 2^32 exit
    // treats 1 as being on neither side of 1.
    if (ax == 1.0f) return flip ? -1.0f : 1.0f;
    // Only try the exact path after a hand-off, the order CORE-MATH and LLVM-libc use. An exact midpoint always falls inside the
    // rounding test's window, since it is a midpoint, and an exact result that is a float, or farther from a midpoint than the
    // window, is rounded correctly by the fast path, because the window's soundness is proven. Before this, every call paid for
    // pow_exact's 25 or so integer steps.
    bool handed;
    float r = pow_fast(ax, y, &handed);
    if (handed && !pow_exact(ax, y, &r)) r = powf_arb(ax, y);
    return flip ? -r : r;
}
}  // namespace wp_cr
