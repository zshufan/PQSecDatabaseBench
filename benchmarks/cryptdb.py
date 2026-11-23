"""CryptDB's onion layers (Popa et al., SOSP 2011), at 128-bit classical security,
with the key size as a parameter so the same code gives the post-quantum variant.

  RND       AES-CBC with a random IV per value (one block)      symmetric
  DET       AES on one 16-byte block (values of up to 16 bytes)  symmetric
  OPE       Boldyreva et al. (Eurocrypt 2009), coins from AES    symmetric
  JOIN-ADJ  (K * PRF(v)) * G on P-256, adjusted by scalar mult   Q1: broken by Shor
  HOM       Paillier, in pqdb.py                                 Q1: broken by Shor

The symmetric onions take key_bits=128 (classical, as in CryptDB) or 256 (the
post-quantum choice). CryptDB used Blowfish for 64-bit DET and RND and NIST
P-192 for JOIN-ADJ; both are below 128-bit security, so we use AES and P-256.

OPE follows Boldyreva et al.'s lazy sampling: each step splits the ciphertext
range at its midpoint and draws, with coins from a PRF of the current state,
how many plaintexts fall below it (a hypergeometric sample). NumPy's sampler
is exact only for populations below 10^9, so the domain is 16 bits and the
range 29 bits (CryptDB doubles the bit length; the cost per encryption is set
by the domain size, about one step per domain bit). Dates, quantities and
discounts in TPC-H fit the domain.

Deterministic onions (DET, OPE, JOIN-ADJ) are computed once per distinct value
and mapped to rows, which is what a caching proxy would do; JOIN-ADJ
adjustment likewise transforms each distinct ciphertext once.
"""
import ctypes
import os
import secrets

import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from pqdb import _compile

P256_N = 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551


def blocks_of(values):
    """16-byte blocks (n x 2 uint64) for int or short-string values."""
    v = np.asarray(values)
    out = np.zeros((len(v), 2), dtype=np.uint64)
    if v.dtype.kind in "iu":
        out[:, 0] = v.astype(np.int64).view(np.uint64)
    else:
        s = np.char.encode(v.astype(str), "utf-8").astype("S16")
        out[:] = np.frombuffer(s.tobytes(), dtype=np.uint64).reshape(-1, 2)
    return out


class AESBlock:
    def __init__(self, key_bits):
        self.c = Cipher(algorithms.AES(os.urandom(key_bits // 8)), modes.ECB())

    def enc(self, b):
        return np.frombuffer(self.c.encryptor().update(np.ascontiguousarray(b).tobytes()), dtype=np.uint64).reshape(-1, 2)

    def dec(self, b):
        return np.frombuffer(self.c.decryptor().update(np.ascontiguousarray(b).tobytes()), dtype=np.uint64).reshape(-1, 2)


class DET:
    """Deterministic encryption of values up to 16 bytes: a PRP on one block."""
    bytes_per_value = 16

    def __init__(self, key_bits):
        self.aes = AESBlock(key_bits)

    def enc(self, values):
        return self.aes.enc(blocks_of(values))

    def dec(self, ct):
        return self.aes.dec(ct)

    @staticmethod
    def server_key(ct):
        """What the server hashes and compares: the first 64 bits of the PRP output.
        Collisions among 10^6 distinct values have probability ~3e-8; a real
        executor would compare the remaining bits on a hash match."""
        return ct[:, 0]


class RND:
    """Randomized outer layer: AES-CBC over one block with a fresh IV (32 bytes stored)."""
    bytes_per_value = 32

    def __init__(self, key_bits):
        self.aes = AESBlock(key_bits)

    def enc(self, inner):
        iv = np.frombuffer(os.urandom(16 * len(inner)), dtype=np.uint64).reshape(-1, 2)
        return iv, self.aes.enc(inner ^ iv)

    def dec(self, iv, ct):
        return self.aes.dec(ct) ^ iv


class OPE:
    """Boldyreva et al.'s order-preserving encryption with AES as the coin PRF."""
    bytes_per_value = 8

    def __init__(self, key_bits, dom_bits=16, rng_bits=29):
        self.aes = Cipher(algorithms.AES(os.urandom(key_bits // 8)), modes.ECB())
        self.D, self.R = dom_bits, rng_bits
        self.cache = {}

    def _rng(self, dlo, dhi, rlo, rhi, y):
        state = dlo | (dhi << 16) | (rlo << 32) | (rhi << 61) | (y << 90)
        k = int.from_bytes(self.aes.encryptor().update(state.to_bytes(16, "little")), "little")
        return np.random.Generator(np.random.Philox(key=k))

    def _walk(self, m=None, c=None):
        dlo, dhi, rlo, rhi = 0, (1 << self.D) - 1, 0, (1 << self.R) - 1
        while True:
            M, N = dhi - dlo + 1, rhi - rlo + 1
            if M == 1:
                if c is not None:
                    return dlo
                return rlo + int(self._rng(dlo, dhi, rlo, rhi, (1 << 29) - 1 if N > 1 else 0).integers(0, N))
            y = rlo + (N + 1) // 2 - 1
            x = dlo - 1 + int(self._rng(dlo, dhi, rlo, rhi, y).hypergeometric(M, N - M, y - rlo + 1))
            left = (m <= x) if c is None else (c <= y)
            if left:
                dhi, rhi = x, y
            else:
                dlo, rlo = x + 1, y + 1

    def encrypt(self, m):
        m = int(m)
        if m not in self.cache:
            self.cache[m] = self._walk(m=m)
        return self.cache[m]

    def decrypt(self, c):
        return self._walk(c=int(c))

    def encrypt_many(self, values):
        u, inv = np.unique(np.asarray(values, dtype=np.int64), return_inverse=True)
        return np.array([self.encrypt(x) for x in u], dtype=np.uint64)[inv], len(u)


class JoinAdj:
    """CryptDB's adjustable join on P-256. PRF_K0 is AES (K0 shared by all columns)."""
    PT = 33

    def __init__(self, prf_key_bits=128):
        k = _compile("joinadj_kernel.c", "joinadj_kernel.so", ("-lcrypto", "-Wno-deprecated-declarations"))
        k.ja_new.restype = ctypes.c_void_p
        k.ja_encrypt_many.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t,
                                      ctypes.c_char_p]
        k.ja_encrypt_many.restype = None
        k.ja_adjust_many.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t,
                                     ctypes.c_char_p]
        k.ja_adjust_many.restype = None
        self.k, self.j = k, k.ja_new()
        self.prf = AESBlock(prf_key_bits)

    @staticmethod
    def new_key():
        return secrets.randbelow(P256_N - 1) + 1

    def encrypt_distinct(self, key, distinct):
        """Ciphertexts (n x 33 bytes) of distinct values under column key `key`."""
        b = blocks_of(distinct)
        two = np.repeat(b, 2, axis=0)
        two[1::2, 1] ^= np.uint64(1)                      # (v, 0) and (v, 1): 32 PRF bytes per value
        s = self.prf.enc(two).tobytes()
        out = ctypes.create_string_buffer(len(b) * self.PT)
        self.k.ja_encrypt_many(self.j, key.to_bytes(32, "big"), s, len(b), out)
        return out.raw

    def adjust(self, pts, k_old, k_new):
        d = (k_new * pow(k_old, -1, P256_N)) % P256_N
        n = len(pts) // self.PT
        out = ctypes.create_string_buffer(len(pts))
        self.k.ja_adjust_many(self.j, d.to_bytes(32, "big"), pts, n, out)
        return out.raw

    @staticmethod
    def server_key(pts):
        """64 bits of the x-coordinate (bytes 1-8 of the compressed point)."""
        a = np.frombuffer(pts, dtype=np.uint8).reshape(-1, 33)[:, 1:9]
        return np.ascontiguousarray(a).view(np.uint64).ravel()
