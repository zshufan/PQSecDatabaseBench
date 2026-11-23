/* CryptDB's adjustable join (JOIN-ADJ) on P-256, on OpenSSL's libcrypto.
 *
 *   JOIN-ADJ_K(v) = (K * PRF_K0(v)) * G      one compressed point, 33 bytes
 *   adjust(c, D)  = D * c, with D = K' / K   turns column key K into K'
 *
 * Popa et al. (SOSP 2011) use NIST P-192; we use P-256 for 128-bit classical
 * security. Both operations are scalar multiplications; encryption uses the
 * fixed generator (precomputed tables), adjustment a variable base.
 *
 *   gcc -O3 -shared -fPIC joinadj_kernel.c -o joinadj_kernel.so -lcrypto
 */
#include <openssl/bn.h>
#include <openssl/ec.h>
#include <openssl/obj_mac.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define PT 33

typedef struct {
    EC_GROUP *g;
    BN_CTX *bn;
    BIGNUM *order;
} ja_t;

ja_t *ja_new(void) {
    ja_t *j = calloc(1, sizeof *j);
    j->g = EC_GROUP_new_by_curve_name(NID_X9_62_prime256v1);
    j->bn = BN_CTX_new();
    j->order = BN_new();
    EC_GROUP_get_order(j->g, j->order, j->bn);
    return j;
}

/* out[i] = ((k * s_i) mod n) * G, s_i a 32-byte big-endian PRF output */
void ja_encrypt_many(ja_t *j, const unsigned char *k32, const unsigned char *s, size_t n, unsigned char *out) {
    BIGNUM *k = BN_bin2bn(k32, 32, NULL), *x = BN_new(), *e = BN_new();
    EC_POINT *P = EC_POINT_new(j->g);
    for (size_t i = 0; i < n; i++) {
        BN_bin2bn(s + 32 * i, 32, x);
        BN_mod_mul(e, k, x, j->order, j->bn);
        EC_POINT_mul(j->g, P, e, NULL, NULL, j->bn);
        EC_POINT_point2oct(j->g, P, POINT_CONVERSION_COMPRESSED, out + PT * i, PT, j->bn);
    }
    BN_free(k); BN_free(x); BN_free(e); EC_POINT_free(P);
}

/* out[i] = delta * pts[i] */
void ja_adjust_many(ja_t *j, const unsigned char *d32, const unsigned char *pts, size_t n, unsigned char *out) {
    BIGNUM *d = BN_bin2bn(d32, 32, NULL);
    EC_POINT *P = EC_POINT_new(j->g);
    for (size_t i = 0; i < n; i++) {
        EC_POINT_oct2point(j->g, P, pts + PT * i, PT, j->bn);
        EC_POINT_mul(j->g, P, NULL, P, d, j->bn);
        EC_POINT_point2oct(j->g, P, POINT_CONVERSION_COMPRESSED, out + PT * i, PT, j->bn);
    }
    BN_free(d); EC_POINT_free(P);
}

/* Commutative blinding as in Private Join and Compute (Ion et al.): hash an
 * identifier to a P-256 point by try-and-increment over SHA-256, then multiply
 * by the secret exponent. in: n identifiers of inlen bytes each. */
#include <openssl/sha.h>
void ec_hash_mul_many(ja_t *j, const unsigned char *k32, const unsigned char *in, size_t inlen, size_t n,
                      unsigned char *out) {
    BIGNUM *k = BN_bin2bn(k32, 32, NULL), *x = BN_new(), *p = BN_new();
    EC_GROUP_get_curve(j->g, p, NULL, NULL, j->bn);
    EC_POINT *P = EC_POINT_new(j->g);
    unsigned char buf[256], h[32];
    for (size_t i = 0; i < n; i++) {
        memcpy(buf, in + i * inlen, inlen);
        for (uint32_t ctr = 0;; ctr++) {
            memcpy(buf + inlen, &ctr, 4);
            SHA256(buf, inlen + 4, h);
            BN_bin2bn(h, 32, x);
            BN_nnmod(x, x, p, j->bn);
            if (EC_POINT_set_compressed_coordinates(j->g, P, x, 0, j->bn)) break;
        }
        EC_POINT_mul(j->g, P, NULL, P, k, j->bn);
        EC_POINT_point2oct(j->g, P, POINT_CONVERSION_COMPRESSED, out + PT * i, PT, j->bn);
    }
    BN_free(k); BN_free(x); BN_free(p); EC_POINT_free(P);
}
