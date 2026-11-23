/*
 * Paillier homomorphic sums with GMP, in C, so that the classical baseline is
 * not slowed down by Python call overhead. GMP's header is not installed in
 * the benchmark environment, so the few prototypes needed are declared here;
 * link against libgmp.so.10.
 */
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct { int _mp_alloc; int _mp_size; void *_mp_d; } mpz_struct;
extern void __gmpz_init(mpz_struct *);
extern void __gmpz_clear(mpz_struct *);
extern void __gmpz_set_ui(mpz_struct *, unsigned long);
extern void __gmpz_mul(mpz_struct *, const mpz_struct *, const mpz_struct *);
extern void __gmpz_mod(mpz_struct *, const mpz_struct *, const mpz_struct *);
extern void __gmpz_import(mpz_struct *, size_t, int, size_t, int, size_t, const void *);
extern void *__gmpz_export(void *, size_t *, int, size_t, int, size_t, const mpz_struct *);

typedef struct { size_t count, ctlen; mpz_struct *ct; mpz_struct n2; } pool_t;

/* count ciphertexts of ctlen big-endian bytes each, and the modulus n^2. */
pool_t *pool_new(size_t count, size_t ctlen, const unsigned char *bytes,
                 const unsigned char *n2bytes, size_t n2len) {
    pool_t *p = malloc(sizeof *p);
    p->count = count; p->ctlen = ctlen;
    p->ct = malloc(count * sizeof(mpz_struct));
    for (size_t i = 0; i < count; i++) {
        __gmpz_init(&p->ct[i]);
        __gmpz_import(&p->ct[i], ctlen, 1, 1, 0, 0, bytes + i * ctlen);
    }
    __gmpz_init(&p->n2);
    __gmpz_import(&p->n2, n2len, 1, 1, 0, 0, n2bytes);
    return p;
}

void pool_free(pool_t *p) {
    for (size_t i = 0; i < p->count; i++) __gmpz_clear(&p->ct[i]);
    __gmpz_clear(&p->n2);
    free(p->ct); free(p);
}

/* For each row r with gid[r] >= 0: acc[gid[r]] *= ct[idx[r]] (mod n^2).
 * Writes the G products as ctlen-byte big-endian values into out. */
void pool_groupsum(pool_t *p, size_t nrows, const int32_t *idx, const int32_t *gid,
                   size_t G, unsigned char *out) {
    mpz_struct *acc = malloc(G * sizeof(mpz_struct));
    for (size_t g = 0; g < G; g++) { __gmpz_init(&acc[g]); __gmpz_set_ui(&acc[g], 1); }
    for (size_t r = 0; r < nrows; r++) {
        int32_t g = gid[r];
        if (g < 0) continue;
        __gmpz_mul(&acc[g], &acc[g], &p->ct[idx[r]]);
        __gmpz_mod(&acc[g], &acc[g], &p->n2);
    }
    unsigned char *buf = malloc(p->ctlen + 8);
    for (size_t g = 0; g < G; g++) {
        size_t cnt = 0;
        __gmpz_export(buf, &cnt, 1, 1, 0, 0, &acc[g]);
        memset(out + g * p->ctlen, 0, p->ctlen);
        memcpy(out + g * p->ctlen + (p->ctlen - cnt), buf, cnt);
        __gmpz_clear(&acc[g]);
    }
    free(buf); free(acc);
}
