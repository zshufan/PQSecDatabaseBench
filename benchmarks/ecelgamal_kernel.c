/* Additively homomorphic (exponential) ElGamal on P-256, on OpenSSL's libcrypto.
 *
 *   Enc(m) = (rG, mG + rH),  H = xG          ciphertext: two compressed points, 66 bytes
 *   Add    = pointwise point addition
 *   Dec    = mG = C2 - x*C1, then m by baby-step giant-step over a table of T = 2^t
 *            baby steps (x-coordinate fingerprints of jG, j < T); cost grows with m / T.
 *
 * The aggregation loop mirrors paillier_kernel.c: a pool of genuine ciphertexts is
 * indexed by row, and rows are added into per-group accumulators.
 *
 *   gcc -O3 -shared -fPIC ecelgamal_kernel.c -o ecelgamal_kernel.so -lcrypto
 */
#include <openssl/bn.h>
#include <openssl/ec.h>
#include <openssl/obj_mac.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define PT 33  /* compressed point */
#define CT 66  /* ciphertext */

typedef struct {
    EC_GROUP *g;
    BN_CTX *bn;
    BIGNUM *order, *x, *xx, *yy;
    EC_POINT *H, *giant;          /* giant = -T*G */
    uint64_t T, hmask;
    uint64_t *keys;               /* open addressing, 0 = empty */
    uint32_t *vals;
} eg_t;

typedef struct {
    size_t n;
    EC_POINT **c1, **c2;
} egpool_t;

static uint64_t fp(eg_t *e, const EC_POINT *P) {
    unsigned char b[32];
    EC_POINT_get_affine_coordinates(e->g, P, e->xx, e->yy, e->bn);
    BN_bn2binpad(e->xx, b, 32);
    uint64_t k;
    memcpy(&k, b + 24, 8);
    return k ? k : 1;
}

static void put(eg_t *e, uint64_t k, uint32_t v) {
    uint64_t h = (k * 0x9E3779B97F4A7C15ULL) & e->hmask;
    while (e->keys[h]) h = (h + 1) & e->hmask;
    e->keys[h] = k;
    e->vals[h] = v;
}

static int64_t get(eg_t *e, uint64_t k) {
    uint64_t h = (k * 0x9E3779B97F4A7C15ULL) & e->hmask;
    while (e->keys[h]) {
        if (e->keys[h] == k) return e->vals[h];
        h = (h + 1) & e->hmask;
    }
    return -1;
}

static void ser(eg_t *e, const EC_POINT *P, unsigned char *out) {
    if (EC_POINT_is_at_infinity(e->g, P)) { memset(out, 0, PT); return; }
    EC_POINT_point2oct(e->g, P, POINT_CONVERSION_COMPRESSED, out, PT, e->bn);
}

static void des(eg_t *e, const unsigned char *in, EC_POINT *P) {
    static const unsigned char z[PT] = {0};
    if (!memcmp(in, z, PT)) { EC_POINT_set_to_infinity(e->g, P); return; }
    EC_POINT_oct2point(e->g, P, in, PT, e->bn);
}

eg_t *eg_new(int log_table) {
    eg_t *e = calloc(1, sizeof *e);
    e->g = EC_GROUP_new_by_curve_name(NID_X9_62_prime256v1);
    e->bn = BN_CTX_new();
    e->order = BN_new(); e->x = BN_new(); e->xx = BN_new(); e->yy = BN_new();
    EC_GROUP_get_order(e->g, e->order, e->bn);
    BN_rand_range(e->x, e->order);
    e->H = EC_POINT_new(e->g);
    EC_POINT_mul(e->g, e->H, e->x, NULL, NULL, e->bn);
    e->T = 1ULL << log_table;
    e->hmask = (e->T << 1) - 1;
    e->keys = calloc(e->T << 1, sizeof(uint64_t));
    e->vals = calloc(e->T << 1, sizeof(uint32_t));
    /* baby steps: jG for 1 <= j < T, made affine in batches */
    const size_t B = 4096;
    EC_POINT **pts = malloc(B * sizeof *pts);
    for (size_t i = 0; i < B; i++) pts[i] = EC_POINT_new(e->g);
    EC_POINT *acc = EC_POINT_new(e->g);
    const EC_POINT *G = EC_GROUP_get0_generator(e->g);
    EC_POINT_set_to_infinity(e->g, acc);
    for (uint64_t j0 = 0; j0 < e->T; j0 += B) {
        size_t m = 0;
        for (uint64_t j = j0; j < j0 + B && j < e->T; j++, m++) {
            EC_POINT_copy(pts[m], acc);
            EC_POINT_add(e->g, acc, acc, G, e->bn);
        }
        if (j0 == 0) { /* skip the point at infinity (j = 0) */
            EC_POINTs_make_affine(e->g, m - 1, pts + 1, e->bn);
            for (size_t i = 1; i < m; i++) put(e, fp(e, pts[i]), (uint32_t)(j0 + i));
        } else {
            EC_POINTs_make_affine(e->g, m, pts, e->bn);
            for (size_t i = 0; i < m; i++) put(e, fp(e, pts[i]), (uint32_t)(j0 + i));
        }
    }
    e->giant = EC_POINT_new(e->g);
    EC_POINT_copy(e->giant, acc);            /* acc = T*G */
    EC_POINT_invert(e->g, e->giant, e->bn);
    for (size_t i = 0; i < B; i++) EC_POINT_free(pts[i]);
    free(pts);
    EC_POINT_free(acc);
    return e;
}

void eg_encrypt_many(eg_t *e, const uint64_t *m, size_t n, unsigned char *out) {
    BIGNUM *r = BN_new(), *bm = BN_new();
    EC_POINT *c1 = EC_POINT_new(e->g), *c2 = EC_POINT_new(e->g);
    for (size_t i = 0; i < n; i++) {
        BN_rand_range(r, e->order);
        BN_set_word(bm, m[i]);
        EC_POINT_mul(e->g, c1, r, NULL, NULL, e->bn);
        EC_POINT_mul(e->g, c2, bm, e->H, r, e->bn);          /* mG + rH */
        ser(e, c1, out + i * CT);
        ser(e, c2, out + i * CT + PT);
    }
    BN_free(r); BN_free(bm); EC_POINT_free(c1); EC_POINT_free(c2);
}

/* returns m, or -1 if m > bound */
int64_t eg_decrypt(eg_t *e, const unsigned char *ct, uint64_t bound) {
    EC_POINT *c1 = EC_POINT_new(e->g), *M = EC_POINT_new(e->g);
    des(e, ct, c1);
    des(e, ct + PT, M);
    EC_POINT_mul(e->g, c1, NULL, c1, e->x, e->bn);       /* x*C1 */
    EC_POINT_invert(e->g, c1, e->bn);
    EC_POINT_add(e->g, M, M, c1, e->bn);                 /* mG */
    int64_t res = -1;
    for (uint64_t i = 0; i * e->T <= bound; i++) {
        if (EC_POINT_is_at_infinity(e->g, M)) { res = (int64_t)(i * e->T); break; }
        int64_t j = get(e, fp(e, M));
        if (j >= 0) { res = (int64_t)(i * e->T + (uint64_t)j); break; }
        EC_POINT_add(e->g, M, M, e->giant, e->bn);
    }
    EC_POINT_free(c1); EC_POINT_free(M);
    return res;
}

/* homomorphic sum of two serialized ciphertexts (for per-operation rates) */
void eg_add(eg_t *e, const unsigned char *a, const unsigned char *b, unsigned char *out) {
    EC_POINT *p = EC_POINT_new(e->g), *q = EC_POINT_new(e->g);
    for (int h = 0; h < 2; h++) {
        des(e, a + h * PT, p);
        des(e, b + h * PT, q);
        EC_POINT_add(e->g, p, p, q, e->bn);
        ser(e, p, out + h * PT);
    }
    EC_POINT_free(p); EC_POINT_free(q);
}

egpool_t *egpool_new(eg_t *e, const unsigned char *cts, size_t n) {
    egpool_t *p = malloc(sizeof *p);
    p->n = n;
    p->c1 = malloc(n * sizeof(EC_POINT *));
    p->c2 = malloc(n * sizeof(EC_POINT *));
    for (size_t i = 0; i < n; i++) {
        p->c1[i] = EC_POINT_new(e->g); p->c2[i] = EC_POINT_new(e->g);
        des(e, cts + i * CT, p->c1[i]);
        des(e, cts + i * CT + PT, p->c2[i]);
    }
    return p;
}

void egpool_free(egpool_t *p) {
    for (size_t i = 0; i < p->n; i++) { EC_POINT_free(p->c1[i]); EC_POINT_free(p->c2[i]); }
    free(p->c1); free(p->c2); free(p);
}

/* out[g] = sum of pool[idx[r]] over rows r with gid[r] == g (gid < 0: row not selected) */
void eg_groupsum(eg_t *e, egpool_t *p, size_t nrows, const int32_t *idx, const int32_t *gid,
                 size_t G, unsigned char *out) {
    EC_POINT **a1 = malloc(G * sizeof *a1), **a2 = malloc(G * sizeof *a2);
    for (size_t g = 0; g < G; g++) {
        a1[g] = EC_POINT_new(e->g); a2[g] = EC_POINT_new(e->g);
        EC_POINT_set_to_infinity(e->g, a1[g]); EC_POINT_set_to_infinity(e->g, a2[g]);
    }
    for (size_t r = 0; r < nrows; r++) {
        int32_t g = gid[r];
        if (g < 0) continue;
        EC_POINT_add(e->g, a1[g], a1[g], p->c1[idx[r]], e->bn);
        EC_POINT_add(e->g, a2[g], a2[g], p->c2[idx[r]], e->bn);
    }
    for (size_t g = 0; g < G; g++) {
        ser(e, a1[g], out + g * CT);
        ser(e, a2[g], out + g * CT + PT);
        EC_POINT_free(a1[g]); EC_POINT_free(a2[g]);
    }
    free(a1); free(a2);
}
