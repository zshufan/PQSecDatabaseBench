/*
 * Minimal BFV encryption path for the aggregation microbenchmark.
 *
 * Only what additive aggregation needs: negacyclic NTT, public-key
 * encryption, and ciphertext addition, over an RNS ciphertext modulus.
 * The NTT is the Harvey/Longa-Naehrig merged-twist transform with Shoup
 * precomputation -- the same scalar algorithm Microsoft SEAL uses when it
 * is built without Intel HEXL. Parameters are chosen by the Python driver
 * to match SEAL's BFVDefault coefficient moduli.
 *
 * Built by aggregation_microbench.py with: gcc -O3 -march=native -shared -fPIC
 */
#include <stddef.h>
#include <stdint.h>
#include <math.h>

typedef unsigned __int128 u128;

/* Branch-free conditional subtraction: x in [0, 2p), p < 2^62 -> x mod p.
 * Written with a mask so that the compiler cannot emit a data-dependent
 * branch; with branches, random data made the forward NTT ~3x slower. */
static inline uint64_t csub(uint64_t x, uint64_t p) {
    uint64_t y = x - p;
    return y + (p & (0 - (y >> 63)));
}
static inline uint64_t mul_shoup(uint64_t x, uint64_t w, uint64_t wp, uint64_t p) {
    uint64_t q = (uint64_t)(((u128)x * wp) >> 64);
    uint64_t r = x * w - q * p;          /* in [0, 2p) for p < 2^63 */
    return csub(r, p);
}
static inline uint64_t add_mod(uint64_t a, uint64_t b, uint64_t p) {
    return csub(a + b, p);
}
static inline uint64_t sub_mod(uint64_t a, uint64_t b, uint64_t p) {
    uint64_t d = a - b;
    return d + (p & (0 - (d >> 63)));
}
static inline uint64_t lift(int64_t v, uint64_t p) {   /* small signed -> [0,p) */
    return v < 0 ? p - (uint64_t)(-v) : (uint64_t)v;
}

/* Forward: natural order in, bit-reversed out. psi[k] = psi^{bitrev(k)}. */
void ntt_fwd(uint64_t *a, size_t n, uint64_t p, const uint64_t *psi, const uint64_t *psis) {
    size_t t = n;
    for (size_t m = 1; m < n; m <<= 1) {
        t >>= 1;
        for (size_t i = 0; i < m; i++) {
            const uint64_t w = psi[m + i], wp = psis[m + i];
            uint64_t *x = a + 2 * i * t, *y = x + t;
            for (size_t j = 0; j < t; j++) {
                uint64_t u = x[j];
                uint64_t v = mul_shoup(y[j], w, wp, p);
                x[j] = add_mod(u, v, p);
                y[j] = sub_mod(u, v, p);
            }
        }
    }
}

/* Inverse: bit-reversed in, natural out, scaled by n^{-1}. ipsi[k] = psi^{-bitrev(k)}. */
void ntt_inv(uint64_t *a, size_t n, uint64_t p, const uint64_t *ipsi, const uint64_t *ipsis,
             uint64_t ninv, uint64_t ninvs) {
    size_t t = 1;
    for (size_t m = n; m > 1; m >>= 1) {
        size_t h = m >> 1;
        for (size_t i = 0; i < h; i++) {
            const uint64_t w = ipsi[h + i], wp = ipsis[h + i];
            uint64_t *x = a + 2 * i * t, *y = x + t;
            for (size_t j = 0; j < t; j++) {
                uint64_t u = x[j], v = y[j];
                x[j] = add_mod(u, v, p);
                y[j] = mul_shoup(sub_mod(u, v, p), w, wp, p);
            }
        }
        t <<= 1;
    }
    for (size_t j = 0; j < n; j++) a[j] = mul_shoup(a[j], ninv, ninvs, p);
}

/* out[j] = a[j] * b[j] mod p for arbitrary a, b < p (key generation, decryption). */
void pointwise_mul(size_t n, uint64_t p, const uint64_t *a, const uint64_t *b, uint64_t *out) {
    for (size_t j = 0; j < n; j++) out[j] = (uint64_t)(((u128)a[j] * b[j]) % p);
}

/*
 * Public-key BFV encryption, all limbs:
 *   c0 = INTT(pk0 * NTT(u)) + e1 + Delta*m,   c1 = INTT(pk1 * NTT(u)) + e2.
 * pk0/pk1 are in NTT form with Shoup companions; u is ternary, e1/e2 are
 * small signed errors, m < t is the (already encoded) plaintext polynomial.
 */
void bfv_encrypt(size_t L, size_t n, const uint64_t *primes,
                 const uint64_t *psi, const uint64_t *psis,
                 const uint64_t *ipsi, const uint64_t *ipsis,
                 const uint64_t *ninv, const uint64_t *ninvs,
                 const uint64_t *pk0, const uint64_t *pk0s,
                 const uint64_t *pk1, const uint64_t *pk1s,
                 const uint64_t *delta, const uint64_t *deltas,
                 const int64_t *u, const int64_t *e1, const int64_t *e2,
                 const uint64_t *m, uint64_t *tmp, uint64_t *c0, uint64_t *c1) {
    for (size_t l = 0; l < L; l++) {
        const uint64_t p = primes[l];
        const size_t o = l * n;
        for (size_t j = 0; j < n; j++) tmp[j] = lift(u[j], p);
        ntt_fwd(tmp, n, p, psi + o, psis + o);
        uint64_t *x0 = c0 + o, *x1 = c1 + o;
        for (size_t j = 0; j < n; j++) {
            x0[j] = mul_shoup(tmp[j], pk0[o + j], pk0s[o + j], p);
            x1[j] = mul_shoup(tmp[j], pk1[o + j], pk1s[o + j], p);
        }
        ntt_inv(x0, n, p, ipsi + o, ipsis + o, ninv[l], ninvs[l]);
        ntt_inv(x1, n, p, ipsi + o, ipsis + o, ninv[l], ninvs[l]);
        for (size_t j = 0; j < n; j++) {
            uint64_t dm = mul_shoup(m[j], delta[l], deltas[l], p);
            x0[j] = add_mod(add_mod(x0[j], lift(e1[j], p), p), dm, p);
            x1[j] = add_mod(x1[j], lift(e2[j], p), p);
        }
    }
}

/* acc += c, coefficient-wise, both components, all limbs. */
void bfv_add_inplace(size_t L, size_t n, const uint64_t *primes,
                     uint64_t *acc0, uint64_t *acc1, const uint64_t *c0, const uint64_t *c1) {
    for (size_t l = 0; l < L; l++) {
        const uint64_t p = primes[l];
        const size_t o = l * n;
        for (size_t j = 0; j < n; j++) {
            acc0[o + j] = add_mod(acc0[o + j], c0[o + j], p);
            acc1[o + j] = add_mod(acc1[o + j], c1[o + j], p);
        }
    }
}

/* ===================================================================== */
/* Additions for the database-level benchmarks (ciphertexts kept in NTT   */
/* form, plaintext-mask multiplication, and fast decryption).             */
/* ===================================================================== */

/* SEAL's base-2^64 Barrett reduction of a 128-bit value; cr = floor(2^128/p). */
static inline uint64_t barrett_reduce_128(uint64_t lo, uint64_t hi, uint64_t p,
                                          uint64_t cr0, uint64_t cr1) {
    u128 t;
    uint64_t carry, tmp1, tmp3, t2lo, t2hi;
    t = (u128)lo * cr0; carry = (uint64_t)(t >> 64);
    t = (u128)lo * cr1; t2lo = (uint64_t)t; t2hi = (uint64_t)(t >> 64);
    t = (u128)t2lo + carry; tmp1 = (uint64_t)t; tmp3 = t2hi + (uint64_t)(t >> 64);
    t = (u128)hi * cr0; t2lo = (uint64_t)t; t2hi = (uint64_t)(t >> 64);
    t = (u128)tmp1 + t2lo; tmp1 = (uint64_t)t; carry = t2hi + (uint64_t)(t >> 64);
    tmp1 = hi * cr1 + tmp3 + carry;
    uint64_t r = lo - tmp1 * p;
    return csub(r, p);
}
static inline uint64_t mul_barrett(uint64_t a, uint64_t b, uint64_t p, uint64_t cr0, uint64_t cr1) {
    u128 x = (u128)a * b;
    return barrett_reduce_128((uint64_t)x, (uint64_t)(x >> 64), p, cr0, cr1);
}

void pointwise_mul_barrett(size_t n, uint64_t p, uint64_t cr0, uint64_t cr1,
                           const uint64_t *a, const uint64_t *b, uint64_t *out) {
    for (size_t j = 0; j < n; j++) out[j] = mul_barrett(a[j], b[j], p, cr0, cr1);
}

/* Public-key encryption straight into NTT form (3 forward NTTs per limb):
 *   c0 = pk0 * NTT(u) + NTT(e1 + Delta*m),   c1 = pk1 * NTT(u) + NTT(e2). */
void bfv_encrypt_ntt(size_t L, size_t n, const uint64_t *primes,
                     const uint64_t *psi, const uint64_t *psis,
                     const uint64_t *pk0, const uint64_t *pk0s,
                     const uint64_t *pk1, const uint64_t *pk1s,
                     const uint64_t *delta, const uint64_t *deltas,
                     const int64_t *u, const int64_t *e1, const int64_t *e2,
                     const uint64_t *m, uint64_t *tmp, uint64_t *tmp2,
                     uint64_t *c0, uint64_t *c1) {
    for (size_t l = 0; l < L; l++) {
        const uint64_t p = primes[l];
        const size_t o = l * n;
        for (size_t j = 0; j < n; j++) tmp[j] = lift(u[j], p);
        ntt_fwd(tmp, n, p, psi + o, psis + o);
        for (size_t j = 0; j < n; j++) {
            c0[o + j] = mul_shoup(tmp[j], pk0[o + j], pk0s[o + j], p);
            c1[o + j] = mul_shoup(tmp[j], pk1[o + j], pk1s[o + j], p);
        }
        for (size_t j = 0; j < n; j++)
            tmp2[j] = add_mod(lift(e1[j], p), mul_shoup(m[j], delta[l], deltas[l], p), p);
        ntt_fwd(tmp2, n, p, psi + o, psis + o);
        for (size_t j = 0; j < n; j++) c0[o + j] = add_mod(c0[o + j], tmp2[j], p);
        for (size_t j = 0; j < n; j++) tmp2[j] = lift(e2[j], p);
        ntt_fwd(tmp2, n, p, psi + o, psis + o);
        for (size_t j = 0; j < n; j++) c1[o + j] = add_mod(c1[o + j], tmp2[j], p);
    }
}

/* Plaintext polynomial (coefficients in [0,t)) -> centred lift into each limb, then NTT. */
void plain_to_ntt(size_t L, size_t n, const uint64_t *primes, uint64_t t, const uint64_t *m,
                  const uint64_t *psi, const uint64_t *psis, uint64_t *out) {
    const uint64_t half = t / 2;
    for (size_t l = 0; l < L; l++) {
        const uint64_t p = primes[l];
        const size_t o = l * n;
        for (size_t j = 0; j < n; j++) {
            uint64_t v = m[j];
            out[o + j] = v > half ? (p - (t - v) % p) % p : v % p;
        }
        ntt_fwd(out + o, n, p, psi + o, psis + o);
    }
}

/* acc += c * M (all in NTT form): ciphertext times plaintext, accumulated. */
void bfv_mul_plain_acc(size_t L, size_t n, const uint64_t *primes,
                       const uint64_t *cr0, const uint64_t *cr1,
                       uint64_t *acc0, uint64_t *acc1,
                       const uint64_t *c0, const uint64_t *c1, const uint64_t *M) {
    for (size_t l = 0; l < L; l++) {
        const uint64_t p = primes[l], r0 = cr0[l], r1 = cr1[l];
        const size_t o = l * n;
        for (size_t j = 0; j < n; j++) {
            acc0[o + j] = add_mod(acc0[o + j], mul_barrett(c0[o + j], M[o + j], p, r0, r1), p);
            acc1[o + j] = add_mod(acc1[o + j], mul_barrett(c1[o + j], M[o + j], p, r0, r1), p);
        }
    }
}

/* Decryption of an NTT-form ciphertext: x = INTT(c0 + c1*s) per limb, then
 * m = round(t*x/q) mod t computed in RNS (Halevi-Polyakov-Shoup scaling):
 *   m = sum_i x_i*I_i + round(sum_i x_i*F_i)  (mod t),
 * with q~_i = (q/p_i)^{-1} mod p_i, I_i = floor(q~_i*t/p_i) mod t, F_i = frac(q~_i*t/p_i). */
void bfv_decrypt_ntt(size_t L, size_t n, const uint64_t *primes,
                     const uint64_t *cr0, const uint64_t *cr1,
                     const uint64_t *ipsi, const uint64_t *ipsis,
                     const uint64_t *ninv, const uint64_t *ninvs,
                     const uint64_t *s_ntt, const uint64_t *c0, const uint64_t *c1,
                     uint64_t *tmp, uint64_t t, const uint64_t *I, const long double *F,
                     uint64_t *m) {
    for (size_t l = 0; l < L; l++) {
        const uint64_t p = primes[l];
        const size_t o = l * n;
        for (size_t j = 0; j < n; j++)
            tmp[o + j] = add_mod(c0[o + j], mul_barrett(c1[o + j], s_ntt[o + j], p, cr0[l], cr1[l]), p);
        ntt_inv(tmp + o, n, p, ipsi + o, ipsis + o, ninv[l], ninvs[l]);
    }
    for (size_t j = 0; j < n; j++) {
        u128 acc = 0;
        long double fr = 0.0L;
        for (size_t l = 0; l < L; l++) {
            uint64_t x = tmp[l * n + j];
            acc += (u128)x * I[l];
            fr += (long double)x * F[l];
        }
        uint64_t a = (uint64_t)(acc % t);
        uint64_t b = ((uint64_t)llroundl(fr)) % t;
        m[j] = (a + b) % t;
    }
}
