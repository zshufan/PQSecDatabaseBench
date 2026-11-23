#!/usr/bin/env python3
"""Primitive costs: classical primitives at the parameters in common use today
against post-quantum primitives at their recommended parameters (Fig. 2), and
the cost of raising classical parameters to higher security levels (Fig. 3).

Common classical parameters: X25519 and ECDH P-256 for key exchange, RSA-2048
for key wrapping, ECDSA P-256 for signatures, Paillier-2048 and EC-ElGamal on
P-256 for additive homomorphic encryption, AES-128. Post-quantum: ML-KEM-768
(the default of NIST and of the TLS hybrid X25519MLKEM768), ML-DSA-44 and -65,
AES-256, BFV with N = 8192 and a 202-bit modulus, and ASHE over AES-256.

Key exchange is timed as a whole handshake, key generation included on both
sides, since TLS uses fresh keys for every handshake: two X25519 or ECDH key
generations and derivations against one ML-KEM key generation, encapsulation,
and decapsulation. Key wrapping (a long-term recipient key) is timed per wrap
and unwrap.

Scaling sweep (classical security level in bits, NIST SP 800-57):
    RSA and Paillier 2048 (112), 3072 (128), 7680 (192)
    ECDH P-256 (128), P-384 (192), P-521 (256), X25519 (128), X448 (224)
    ECDSA P-256 (128), P-384 (192), P-521 (256)
with ML-KEM-768/1024 and ML-DSA-44/65/87 as references. RSA-15360 is left
out because generating its key takes too long here.

    python3 benchmarks/primitives_bench.py        # ~6 minutes

Writes benchmarks/results/primitives.json.
"""
import json
import os
import platform

import numpy as np
import cryptography
from cryptography.hazmat.backends.openssl import backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, mldsa, mlkem, padding, rsa, x25519, x448
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from pqdb import ASHE, BFV, ECElGamal, Paillier, cpu_model, rate

HERE = os.path.dirname(os.path.abspath(__file__))
MSG = os.urandom(64)
KEY = os.urandom(32)


def rsa_oaep(bits):
    sk = rsa.generate_private_key(public_exponent=65537, key_size=bits)
    pk = sk.public_key()
    pad = padding.OAEP(mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(), label=None)
    ct = pk.encrypt(KEY, pad)
    return dict(wrap=rate(lambda: pk.encrypt(KEY, pad)), unwrap=rate(lambda: sk.decrypt(ct, pad)),
                pk=bits // 8, ct=len(ct))


def ecdsa(curve, h):
    sk = ec.generate_private_key(curve)
    pk = sk.public_key()
    alg = ec.ECDSA(h)
    sig = sk.sign(MSG, alg)
    return dict(sign=rate(lambda: sk.sign(MSG, alg)), verify=rate(lambda: pk.verify(sig, MSG, alg)),
                pk=2 * curve.key_size // 8, sig=2 * curve.key_size // 8)


def mldsa_rates(cls):
    sk = cls.generate()
    pk = sk.public_key()
    sig = sk.sign(MSG)
    return dict(sign=rate(lambda: sk.sign(MSG)), verify=rate(lambda: pk.verify(sig, MSG)),
                pk=len(pk.public_bytes_raw()), sig=len(sig))


def aes(bits, size):
    g = AESGCM(AESGCM.generate_key(bit_length=bits))
    nonce, msg = os.urandom(12), os.urandom(size)
    r = rate(lambda: g.encrypt(nonce, msg, None))
    return dict(ops=r, bytes_per_s=r * size, ct=len(g.encrypt(nonce, msg, None)), key=bits // 8)


def additive_he_more():
    """EC-ElGamal (non-PQ) and ASHE with AES-256 (PQ, owner-decrypts setting only)."""
    eg = ECElGamal(20)
    c = eg.encrypt(123)
    pool = eg.make_pool(np.arange(4096, dtype=np.uint64))
    idx = np.random.default_rng(0).integers(0, 4096, 200000).astype(np.int32)
    gid = np.zeros(200000, dtype=np.int32)
    t_add = 1 / rate(lambda: eg.groupsum(pool, idx, gid, 1)) / 200000
    dec = {}
    for bits in (16, 24, 32):
        cm = eg.encrypt((1 << bits) - 1)
        dec[str(bits)] = rate(lambda: eg.decrypt(cm, 1 << 40), seconds=0.5)
    elgamal = dict(encrypt=rate(lambda: eg.encrypt(123)), add=1 / t_add, decrypt_by_bits=dec,
                   table_bits=eg.log_table, table_seconds=eg.setup_seconds, ct=eg.CT, values_per_ct=1)
    a = ASHE(256)
    v = np.random.default_rng(1).integers(0, 1 << 32, 1_000_000)
    ct = a.encrypt(v)
    st, en = np.array([1]), np.array([len(v)])
    many = np.arange(1, 2_000_001, 2)                  # 1M runs of length one
    t_run = 1 / rate(lambda: a.decrypt_sum(1, many, many), seconds=0.5) / len(many)
    ashe = dict(encrypt=1e6 * rate(lambda: a.encrypt(v), seconds=0.5),
                add=1e6 * rate(lambda: ct.sum(dtype=np.uint64), seconds=0.5),
                decrypt=rate(lambda: a.decrypt_sum(1, st, en), seconds=0.5), decrypt_per_run=1 / t_run,
                ct=8, values_per_ct=1, prf="AES-256")
    return elgamal, ashe



def xdh_handshake(cls, size):
    peer = cls.generate().public_key()
    a = cls.generate()

    def side():
        k = cls.generate()
        k.public_key()
        k.exchange(peer)
    one = rate(side)
    return dict(side=one, handshake=one / 2, derive=rate(lambda: a.exchange(peer)), msg=size)


def ecdh_handshake(curve):
    peer = ec.generate_private_key(curve).public_key()
    a = ec.generate_private_key(curve)

    def side():
        ec.generate_private_key(curve).exchange(ec.ECDH(), peer)
    one = rate(side)
    return dict(side=one, handshake=one / 2, derive=rate(lambda: a.exchange(ec.ECDH(), peer)),
                msg=2 * ((curve.key_size + 7) // 8))


def mlkem_handshake(cls):
    sk = cls.generate()
    pk = sk.public_key()
    _, ct = pk.encapsulate()
    kg = rate(lambda: cls.generate().public_key())
    en = rate(pk.encapsulate)
    de = rate(lambda: sk.decapsulate(ct))
    return dict(keygen=kg, encaps=en, decaps=de, handshake=1 / (1 / kg + 1 / en + 1 / de),
                pk=len(pk.public_bytes_raw()), ct=len(ct))


def paillier_rates(bits):
    pa = Paillier(bits)
    c = pa.encrypt_bytes(123)
    pool = pa.make_pool([1, 2])
    idx = np.zeros(20000, dtype=np.int32)
    gid = np.zeros(20000, dtype=np.int32)
    t_add = 1 / rate(lambda: pa.groupsum(pool, idx, gid, 1)) / 20000
    return dict(encrypt=rate(lambda: pa.encrypt(123)), add=1 / t_add,
                decrypt=rate(lambda: pa.decrypt(c)), ct=pa.ctlen, values_per_ct=1, bits=bits)


def bfv_rates():
    rng = np.random.default_rng(int.from_bytes(os.urandom(8), "little"))
    b = BFV(8192, [40, 40, 40, 41, 41], 50, rng)
    m = b.encode(rng.integers(0, 1000, size=b.n))
    ct = b.encrypt(m)
    acc = b.zero()
    M = b.mask_ntt(rng.integers(0, 2, size=b.n))
    return dict(encrypt=rate(lambda: b.encrypt(m)), add=rate(lambda: b.add_inplace(acc, ct)),
                mask_prepare=rate(lambda: b.mask_ntt(rng.integers(0, 2, size=b.n))),
                mul_plain=rate(lambda: b.mul_plain_acc(acc, ct, M)),
                decrypt=rate(lambda: b.decrypt_slots(ct)), ct=b.ciphertext_bytes(), values_per_ct=b.n,
                n=b.n, log2_q=b.q.bit_length(), t_bits=b.t.bit_length())


def main():
    out = dict(machine=dict(cpu=cpu_model(), logical_cpus=os.cpu_count(), python=platform.python_version(),
                            cryptography=cryptography.__version__, openssl=backend.openssl_version_text()),
               method="single thread; median of 3 one-second windows; operations per second")
    print("key exchange ...")
    out["kex"] = {
        "X25519": dict(**xdh_handshake(x25519.X25519PrivateKey, 32), level=128, common=True),
        "ECDH P-256": dict(**ecdh_handshake(ec.SECP256R1()), level=128, common=True),
        "X448": dict(**xdh_handshake(x448.X448PrivateKey, 56), level=224),
        "ECDH P-384": dict(**ecdh_handshake(ec.SECP384R1()), level=192),
        "ECDH P-521": dict(**ecdh_handshake(ec.SECP521R1()), level=256),
        "ML-KEM-768": dict(**mlkem_handshake(mlkem.MLKEM768PrivateKey), level="cat. 3", common=True),
        "ML-KEM-1024": dict(**mlkem_handshake(mlkem.MLKEM1024PrivateKey), level="cat. 5"),
    }
    print("key wrapping ...")
    out["wrap"] = {f"RSA-{b}-OAEP": dict(**rsa_oaep(b), level=lvl, common=(b == 2048))
                   for b, lvl in ((2048, 112), (3072, 128), (7680, 192))}
    print("signatures ...")
    out["sig"] = {
        "ECDSA P-256": dict(**ecdsa(ec.SECP256R1(), hashes.SHA256()), level=128, common=True),
        "ECDSA P-384": dict(**ecdsa(ec.SECP384R1(), hashes.SHA384()), level=192),
        "ECDSA P-521": dict(**ecdsa(ec.SECP521R1(), hashes.SHA512()), level=256),
        "ML-DSA-44": dict(**mldsa_rates(mldsa.MLDSA44PrivateKey), level="cat. 2", common=True),
        "ML-DSA-65": dict(**mldsa_rates(mldsa.MLDSA65PrivateKey), level="cat. 3", common=True),
        "ML-DSA-87": dict(**mldsa_rates(mldsa.MLDSA87PrivateKey), level="cat. 5"),
    }
    out["sig"]["ECDSA P-521"]["pk"] = out["sig"]["ECDSA P-521"]["sig"] = 132
    print("symmetric ...")
    out["aes"] = {f"AES-{b}-GCM": {str(s): aes(b, s) for s in (32, 4096, 1 << 20)} for b in (128, 256)}
    print("additive HE ...")
    out["additive_he"] = {f"Paillier-{b}": dict(**paillier_rates(b), level=lvl, common=(b == 2048))
                          for b, lvl in ((2048, 112), (3072, 128), (7680, 192))}
    g, a = additive_he_more()
    out["additive_he"]["EC-ElGamal P-256"] = dict(**g, level=128, common=True)
    out["additive_he"]["BFV-8192"] = dict(**bfv_rates(), level="128 (quantum)", common=True)
    out["additive_he"]["ASHE AES-256"] = dict(**a, level="256", common=True)
    path = os.path.join(HERE, "results", "primitives.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k != "aes"}, indent=1, default=lambda x: round(float(x), 1)))
    print("wrote", path)


if __name__ == "__main__":
    main()
