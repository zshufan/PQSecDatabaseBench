"""Shared crypto back ends for the database-level benchmarks.

* ``BFV``      -- packed BFV with ciphertexts kept in NTT form: encryption,
                  addition, multiplication by a plaintext mask, and fast RNS
                  decryption. Kernel: bfv_kernel.c (Harvey NTT, Shoup and
                  Barrett multiplication as in SEAL without HEXL).
* ``Paillier`` -- Paillier on GMP: owner-side (CRT) encryption, CRT decryption,
                  and homomorphic sums / grouped sums in C (paillier_kernel.c).
* ``ECElGamal``-- additive EC-ElGamal on P-256 with baby-step giant-step
                  decryption (ecelgamal_kernel.c, libcrypto).
* ``ASHE``     -- additively symmetric homomorphic encryption (Seabed, OSDI'16)
                  with an AES-256 PRF.
* ``rate``     -- median-of-windows throughput measurement.

The C kernels are compiled with gcc on first use into benchmarks/build/.
"""
import ctypes
import ctypes.util
import os
import platform
import statistics
import subprocess
import time

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, "build")
LIBGMP = ctypes.util.find_library("gmp") or "libgmp.so.10"


def rate(fn, seconds=1.0, repeats=3, min_calls=3):
    fn()
    out = []
    for _ in range(repeats):
        n, t0 = 0, time.perf_counter()
        while True:
            fn()
            n += 1
            el = time.perf_counter() - t0
            if el >= seconds and n >= min_calls:
                break
        out.append(n / el)
    return statistics.median(out)


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def _compile(src, so, extra=()):
    os.makedirs(BUILD, exist_ok=True)
    src, so = os.path.join(HERE, src), os.path.join(BUILD, so)
    if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src):
        subprocess.run(["gcc", "-O3", "-march=native", "-shared", "-fPIC", src, "-o", so, *extra, "-lm"],
                       check=True)
    return ctypes.CDLL(so)


# --------------------------------------------------------------- number theory
def is_prime(n):
    if n < 2:
        return False
    for sp in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % sp == 0:
            return n == sp
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):  # deterministic < 2^64
        x = pow(a % n, d, n)
        if x in (0, 1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def ntt_primes(bit_sizes, n):
    """Largest distinct primes below 2^b with p = 1 mod 2n, one per entry."""
    out, used = [], set()
    for b in bit_sizes:
        p = ((1 << b) - 1) // (2 * n) * (2 * n) + 1
        while p in used or not is_prime(p):
            p -= 2 * n
        out.append(p)
        used.add(p)
    return out


def bitrev(k, bits):
    return int(format(k, f"0{bits}b")[::-1], 2)


def shoup(w, p):
    return (w << 64) // p


class NTTTables:
    def __init__(self, n, p):
        self.n, self.p = n, p
        bits = n.bit_length() - 1
        g = 2
        while True:
            psi = pow(g, (p - 1) // (2 * n), p)
            if pow(psi, n, p) == p - 1:
                break
            g += 1
        ipsi = pow(psi, -1, p)
        pw, ipw = [1] * n, [1] * n
        for k in range(1, n):
            pw[k] = pw[k - 1] * psi % p
            ipw[k] = ipw[k - 1] * ipsi % p
        rev = [pw[bitrev(k, bits)] for k in range(n)]
        irev = [ipw[bitrev(k, bits)] for k in range(n)]
        self.psi = np.array(rev, dtype=np.uint64)
        self.psis = np.array([shoup(w, p) for w in rev], dtype=np.uint64)
        self.ipsi = np.array(irev, dtype=np.uint64)
        self.ipsis = np.array([shoup(w, p) for w in irev], dtype=np.uint64)
        self.ninv = pow(n, -1, p)
        self.ninvs = shoup(self.ninv, p)


# ------------------------------------------------------------------------ GMP
class MPZ(ctypes.Structure):
    _fields_ = [("alloc", ctypes.c_int), ("size", ctypes.c_int), ("d", ctypes.c_void_p)]


class GMP:
    """Thin ctypes binding. Symbols are looked up with getattr because names
    starting with two underscores would be mangled inside a class body."""

    def __init__(self):
        path = ctypes.util.find_library("gmp") or "libgmp.so.10"
        g = ctypes.CDLL(path)
        Z = ctypes.POINTER(MPZ)
        f = lambda name: getattr(g, "__gmpz_" + name)
        self.init, self.set_str, self.get_str = f("init"), f("set_str"), f("get_str")
        self.sizeinbase, self.import_ = f("sizeinbase"), f("import")
        self.powm, self.mul, self.mod, self.sub, self.add = (f(x) for x in ("powm", "mul", "mod", "sub", "add"))
        self.mul_ui, self.add_ui = f("mul_ui"), f("add_ui")
        self.init.argtypes = [Z]
        self.set_str.argtypes = [Z, ctypes.c_char_p, ctypes.c_int]
        self.get_str.argtypes = [ctypes.c_char_p, ctypes.c_int, Z]
        self.get_str.restype = ctypes.c_char_p
        self.sizeinbase.argtypes = [Z, ctypes.c_int]
        self.sizeinbase.restype = ctypes.c_size_t
        self.import_.argtypes = [Z, ctypes.c_size_t, ctypes.c_int, ctypes.c_size_t,
                                 ctypes.c_int, ctypes.c_size_t, ctypes.c_char_p]
        self.powm.argtypes = [Z] * 4
        for fn in (self.mul, self.mod, self.sub, self.add):
            fn.argtypes = [Z] * 3
        self.mul_ui.argtypes = [Z, Z, ctypes.c_ulong]
        self.add_ui.argtypes = [Z, Z, ctypes.c_ulong]
        for fn in (self.init, self.powm, self.mul, self.mod, self.sub, self.add, self.mul_ui,
                   self.add_ui, self.import_):
            fn.restype = None
        self.set_str.restype = ctypes.c_int
        self.version = ctypes.c_char_p.in_dll(g, "__gmp_version").value.decode()

    def new(self, v=0):
        z = MPZ()
        self.init(ctypes.byref(z))
        self.set_str(ctypes.byref(z), format(v, "x").encode(), 16)
        return z

    def to_int(self, z):
        size = self.sizeinbase(ctypes.byref(z), 16) + 2
        buf = ctypes.create_string_buffer(size)
        return int(self.get_str(buf, 16, ctypes.byref(z)), 16)

    def set_random_bytes(self, z, nbytes):
        self.import_(ctypes.byref(z), nbytes, 1, 1, 0, 0, os.urandom(nbytes))


def cpu_model():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


# ------------------------------------------------------------------------ BFV
def _bfv_lib():
    lib = _compile("bfv_kernel.c", "bfv_kernel.so")
    P = np.ctypeslib.ndpointer(dtype=np.uint64, flags="C_CONTIGUOUS")
    I = np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")
    LD = np.ctypeslib.ndpointer(dtype=np.longdouble, flags="C_CONTIGUOUS")
    S, U = ctypes.c_size_t, ctypes.c_uint64
    sig = {
        "ntt_fwd": [P, S, U, P, P],
        "ntt_inv": [P, S, U, P, P, U, U],
        "pointwise_mul_barrett": [S, U, U, U, P, P, P],
        "bfv_encrypt_ntt": [S, S, P, P, P, P, P, P, P, P, P, I, I, I, P, P, P, P, P],
        "plain_to_ntt": [S, S, P, U, P, P, P, P],
        "bfv_mul_plain_acc": [S, S, P, P, P, P, P, P, P, P],
        "bfv_decrypt_ntt": [S, S, P, P, P, P, P, P, P, P, P, P, P, U, P, LD, P],
        "bfv_add_inplace": [S, S, P, P, P, P, P],
    }
    for name, args in sig.items():
        f = getattr(lib, name)
        f.argtypes, f.restype = args, None
    return lib


class BFV:
    SIGMA, BOUND = 3.2, 19

    def __init__(self, n, bit_sizes, t_bits, rng):
        self.lib = lib = _bfv_lib()
        self.n, self.rng = n, rng
        self.primes = ntt_primes(bit_sizes, n)
        self.L = len(self.primes)
        self.q = 1
        for p in self.primes:
            self.q *= p
        self.t = ntt_primes([t_bits], n)[0]           # batching prime, = 1 mod 2n
        assert self.t not in self.primes
        self.tabs = [NTTTables(n, p) for p in self.primes]
        self.ttab = NTTTables(n, self.t)
        cat = lambda a: np.ascontiguousarray(np.concatenate([getattr(x, a) for x in self.tabs]))
        self.psi, self.psis, self.ipsi, self.ipsis = cat("psi"), cat("psis"), cat("ipsi"), cat("ipsis")
        u64 = lambda xs: np.array(xs, dtype=np.uint64)
        self.ninv, self.ninvs = u64([x.ninv for x in self.tabs]), u64([x.ninvs for x in self.tabs])
        self.P = u64(self.primes)
        cr = [(1 << 128) // p for p in self.primes]
        self.cr0, self.cr1 = u64([c & ((1 << 64) - 1) for c in cr]), u64([c >> 64 for c in cr])
        delta = self.q // self.t
        self.delta = u64([delta % p for p in self.primes])
        self.deltas = u64([shoup(delta % p, p) for p in self.primes])
        I, F = [], []
        for p in self.primes:
            qt = pow(self.q // p, -1, p)
            num = qt * self.t
            I.append((num // p) % self.t)
            F.append(np.longdouble(num % p) / np.longdouble(p))
        self.I = u64(I)
        self.F = np.array(F, dtype=np.longdouble)
        self.tmp = np.zeros(n, dtype=np.uint64)
        self.tmp2 = np.zeros(n, dtype=np.uint64)
        self.tmpL = np.zeros(self.L * n, dtype=np.uint64)
        self._keygen()

    # sampling
    def ternary(self):
        return self.rng.integers(-1, 2, size=self.n, dtype=np.int64)

    def gaussian(self):
        e = np.rint(self.rng.normal(0.0, self.SIGMA, size=self.n)).astype(np.int64)
        return np.clip(e, -self.BOUND, self.BOUND)

    def _small_ntt(self, v):
        out = np.empty(self.L * self.n, dtype=np.uint64)
        for l, p in enumerate(self.primes):
            seg = np.ascontiguousarray((v % np.int64(p)).astype(np.uint64))
            x = self.tabs[l]
            self.lib.ntt_fwd(seg, self.n, p, x.psi, x.psis)
            out[l * self.n:(l + 1) * self.n] = seg
        return out

    def _keygen(self):
        n = self.n
        self.s_ntt = self._small_ntt(self.ternary())
        e_ntt = self._small_ntt(self.gaussian())
        a = np.concatenate([self.rng.integers(0, p, size=n, dtype=np.uint64) for p in self.primes])
        pk0 = np.empty_like(a)
        for l, p in enumerate(self.primes):
            sl = slice(l * n, (l + 1) * n)
            prod = np.empty(n, dtype=np.uint64)
            self.lib.pointwise_mul_barrett(n, p, int(self.cr0[l]), int(self.cr1[l]),
                                           np.ascontiguousarray(a[sl]), np.ascontiguousarray(self.s_ntt[sl]), prod)
            pk0[sl] = (np.uint64(p) - ((prod + e_ntt[sl]) % np.uint64(p))) % np.uint64(p)
        self.pk0, self.pk1 = np.ascontiguousarray(pk0), np.ascontiguousarray(a)
        sh = lambda arr: np.array([shoup(int(w), self.primes[k // n]) for k, w in enumerate(arr)], dtype=np.uint64)
        self.pk0s, self.pk1s = sh(self.pk0), sh(self.pk1)

    # batching
    def encode(self, slots):
        m = np.ascontiguousarray(np.asarray(slots, dtype=np.int64) % np.int64(self.t)).astype(np.uint64)
        x = self.ttab
        self.lib.ntt_inv(m, self.n, self.t, x.ipsi, x.ipsis, x.ninv, x.ninvs)
        return m

    def decode(self, m):
        m = np.ascontiguousarray(m.copy())
        x = self.ttab
        self.lib.ntt_fwd(m, self.n, self.t, x.psi, x.psis)
        return m

    # ciphertext operations (NTT form)
    def encrypt(self, m):
        c0 = np.empty(self.L * self.n, dtype=np.uint64)
        c1 = np.empty(self.L * self.n, dtype=np.uint64)
        u, e1, e2 = self.ternary(), self.gaussian(), self.gaussian()
        self.lib.bfv_encrypt_ntt(self.L, self.n, self.P, self.psi, self.psis, self.pk0, self.pk0s,
                                 self.pk1, self.pk1s, self.delta, self.deltas, u, e1, e2, m,
                                 self.tmp, self.tmp2, c0, c1)
        return c0, c1

    def zero(self):
        return np.zeros(self.L * self.n, dtype=np.uint64), np.zeros(self.L * self.n, dtype=np.uint64)

    def add_inplace(self, acc, c):
        self.lib.bfv_add_inplace(self.L, self.n, self.P, acc[0], acc[1], c[0], c[1])

    def plain_ntt(self, m):
        out = np.empty(self.L * self.n, dtype=np.uint64)
        self.lib.plain_to_ntt(self.L, self.n, self.P, self.t, m, self.psi, self.psis, out)
        return out

    def mask_ntt(self, mask):
        """Slot mask (0/1 per slot) -> plaintext in NTT form, ready for mul_plain_acc."""
        return self.plain_ntt(self.encode(mask))

    def mul_plain_acc(self, acc, c, M):
        self.lib.bfv_mul_plain_acc(self.L, self.n, self.P, self.cr0, self.cr1, acc[0], acc[1], c[0], c[1], M)

    def decrypt(self, c):
        m = np.empty(self.n, dtype=np.uint64)
        self.lib.bfv_decrypt_ntt(self.L, self.n, self.P, self.cr0, self.cr1, self.ipsi, self.ipsis,
                                 self.ninv, self.ninvs, self.s_ntt, c[0], c[1], self.tmpL,
                                 self.t, self.I, self.F, m)
        return m

    def decrypt_slots(self, c):
        return self.decode(self.decrypt(c))

    def ciphertext_bytes(self):
        return 2 * self.n * self.q.bit_length() // 8      # bit-packed


# ------------------------------------------------------------------- Paillier
class Paillier:
    """Paillier on GMP (modulus size is a parameter). Hot loops (sums) run in C;
    encryption and decryption are single GMP calls driven from Python."""

    def __init__(self, bits=3072):
        from cryptography.hazmat.primitives.asymmetric import rsa
        nums = rsa.generate_private_key(public_exponent=65537, key_size=bits).private_numbers()
        self.p, self.q = nums.p, nums.q
        self.n = self.p * self.q
        self.n2 = self.n * self.n
        self.ctlen = (self.n2.bit_length() + 7) // 8
        self.g = g = GMP()
        Z = ctypes.POINTER(MPZ)
        lib = ctypes.CDLL(LIBGMP)
        self.tdiv_q = getattr(lib, "__gmpz_tdiv_q"); self.tdiv_q.argtypes = [Z] * 3; self.tdiv_q.restype = None
        self.sub_ui = getattr(lib, "__gmpz_sub_ui"); self.sub_ui.argtypes = [Z, Z, ctypes.c_ulong]
        self.sub_ui.restype = None
        B = ctypes.byref
        self.B = B
        p, q, n = self.p, self.q, self.n
        self.N, self.N2 = g.new(n), g.new(self.n2)
        self.Pz, self.Qz = g.new(p), g.new(q)
        self.P2, self.Q2 = g.new(p * p), g.new(q * q)
        self.EP, self.EQ = g.new(n % (p * (p - 1))), g.new(n % (q * (q - 1)))
        self.P2INV = g.new(pow(p * p, -1, q * q))
        self.PM1, self.QM1 = g.new(p - 1), g.new(q - 1)
        # h_p = L_p(g^(p-1) mod p^2)^(-1) mod p with g = n + 1
        Lp = ((pow(n + 1, p - 1, p * p) - 1) // p) % p
        Lq = ((pow(n + 1, q - 1, q * q) - 1) // q) % q
        self.HP, self.HQ = g.new(pow(Lp, -1, p)), g.new(pow(Lq, -1, q))
        self.PINV = g.new(pow(p, -1, q))
        self.rbytes = (bits - 8) // 8
        self.r, self.rn, self.gm, self.c = g.new(), g.new(), g.new(), g.new()
        self.a, self.b, self.mp, self.mq = g.new(), g.new(), g.new(), g.new()
        # C kernel
        self.k = _compile("paillier_kernel.c", "paillier_kernel.so", ("-l:libgmp.so.10",))
        self.k.pool_new.restype = ctypes.c_void_p
        self.k.pool_new.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
        self.k.pool_free.argtypes = [ctypes.c_void_p]
        I32 = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
        self.k.pool_groupsum.argtypes = [ctypes.c_void_p, ctypes.c_size_t, I32, I32, ctypes.c_size_t, ctypes.c_char_p]
        self.k.pool_groupsum.restype = None

    def encrypt(self, m):
        """Owner-side encryption: r^n mod n^2 by CRT. Returns the ciphertext as bytes."""
        g, B = self.g, self.B
        g.set_random_bytes(self.r, self.rbytes)
        g.powm(B(self.a), B(self.r), B(self.EP), B(self.P2))
        g.powm(B(self.b), B(self.r), B(self.EQ), B(self.Q2))
        g.sub(B(self.b), B(self.b), B(self.a))
        g.mul(B(self.b), B(self.b), B(self.P2INV))
        g.mod(B(self.b), B(self.b), B(self.Q2))
        g.mul(B(self.b), B(self.b), B(self.P2))
        g.add(B(self.rn), B(self.b), B(self.a))
        g.mul_ui(B(self.gm), B(self.N), m)
        g.add_ui(B(self.gm), B(self.gm), 1)
        g.mul(B(self.c), B(self.gm), B(self.rn))
        g.mod(B(self.c), B(self.c), B(self.N2))
        return self.c

    def encrypt_bytes(self, m):
        return self.g.to_int(self.encrypt(m)).to_bytes(self.ctlen, "big")

    def encrypt_public(self, m):
        """Writer-side encryption with the public key only: r^n mod n^2 without CRT."""
        g, B = self.g, self.B
        g.set_random_bytes(self.r, self.rbytes)
        g.powm(B(self.rn), B(self.r), B(self.N), B(self.N2))
        g.mul_ui(B(self.gm), B(self.N), m)
        g.add_ui(B(self.gm), B(self.gm), 1)
        g.mul(B(self.c), B(self.gm), B(self.rn))
        g.mod(B(self.c), B(self.c), B(self.N2))
        return self.c

    def decrypt(self, cbytes):
        """CRT decryption with GMP. Input: ctlen-byte big-endian ciphertext."""
        g, B = self.g, self.B
        g.import_(B(self.c), len(cbytes), 1, 1, 0, 0, cbytes)
        for (E, M2, M, H, out) in ((self.PM1, self.P2, self.Pz, self.HP, self.mp),
                                   (self.QM1, self.Q2, self.Qz, self.HQ, self.mq)):
            g.powm(B(self.a), B(self.c), B(E), B(M2))
            self.sub_ui(B(self.a), B(self.a), 1)
            self.tdiv_q(B(self.a), B(self.a), B(M))
            g.mul(B(self.a), B(self.a), B(H))
            g.mod(B(out), B(self.a), B(M))
        g.sub(B(self.b), B(self.mq), B(self.mp))
        g.mul(B(self.b), B(self.b), B(self.PINV))
        g.mod(B(self.b), B(self.b), B(self.Qz))
        g.mul(B(self.b), B(self.b), B(self.Pz))
        g.add(B(self.b), B(self.b), B(self.mp))
        return self.b

    def decrypt_int(self, cbytes):
        return self.g.to_int(self.decrypt(cbytes))

    # pools of genuine ciphertexts, reused by index to build large columns
    def make_pool(self, values):
        cts = b"".join(self.encrypt_bytes(int(v)) for v in values)
        n2b = self.n2.to_bytes(self.ctlen, "big")
        return self.k.pool_new(len(values), self.ctlen, cts, n2b, len(n2b))

    def groupsum(self, pool, idx, gid, G):
        out = ctypes.create_string_buffer(G * self.ctlen)
        self.k.pool_groupsum(pool, len(idx), np.ascontiguousarray(idx, dtype=np.int32),
                             np.ascontiguousarray(gid, dtype=np.int32), G, out)
        raw = out.raw
        return [raw[i * self.ctlen:(i + 1) * self.ctlen] for i in range(G)]


# ----------------------------------------------------------------- EC-ElGamal
class ECElGamal:
    """Additive (exponential) ElGamal on P-256 via libcrypto (ecelgamal_kernel.c).

    Ciphertexts are two compressed points (66 bytes). Decryption recovers mG and
    then m by baby-step giant-step over a 2^log_table table, so its cost grows
    with the decrypted value; sums wider than the table allows are handled by
    splitting values into limbs (``limbs``), each encrypted separately."""

    CT = 66

    def __init__(self, log_table=20):
        k = _compile("ecelgamal_kernel.c", "ecelgamal_kernel.so", ("-lcrypto", "-Wno-deprecated-declarations"))
        k.eg_new.restype = ctypes.c_void_p
        k.eg_new.argtypes = [ctypes.c_int]
        U64 = np.ctypeslib.ndpointer(dtype=np.uint64, flags="C_CONTIGUOUS")
        I32 = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
        k.eg_encrypt_many.argtypes = [ctypes.c_void_p, U64, ctypes.c_size_t, ctypes.c_char_p]
        k.eg_encrypt_many.restype = None
        k.eg_decrypt.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint64]
        k.eg_decrypt.restype = ctypes.c_int64
        k.eg_add.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
        k.eg_add.restype = None
        k.egpool_new.restype = ctypes.c_void_p
        k.egpool_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
        k.egpool_free.argtypes = [ctypes.c_void_p]
        k.eg_groupsum.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, I32, I32, ctypes.c_size_t,
                                  ctypes.c_char_p]
        k.eg_groupsum.restype = None
        self.k = k
        self.log_table = log_table
        self.e, self.setup_seconds = timed(lambda: k.eg_new(log_table))

    def encrypt_many(self, values):
        v = np.ascontiguousarray(values, dtype=np.uint64)
        out = ctypes.create_string_buffer(len(v) * self.CT)
        self.k.eg_encrypt_many(self.e, v, len(v), out)
        return out.raw

    def encrypt(self, m):
        return self.encrypt_many([m])

    def add(self, a, b):
        out = ctypes.create_string_buffer(self.CT)
        self.k.eg_add(self.e, a, b, out)
        return out.raw

    def decrypt(self, ct, bound):
        m = self.k.eg_decrypt(self.e, ct, int(bound))
        if m < 0:
            raise ValueError("plaintext above the decryption bound")
        return m

    def make_pool(self, values):
        return self.k.egpool_new(self.e, self.encrypt_many(values), len(values))

    def groupsum(self, pool, idx, gid, G):
        out = ctypes.create_string_buffer(G * self.CT)
        self.k.eg_groupsum(self.e, pool, len(idx), np.ascontiguousarray(idx, dtype=np.int32),
                           np.ascontiguousarray(gid, dtype=np.int32), G, out)
        raw = out.raw
        return [raw[i * self.CT:(i + 1) * self.CT] for i in range(G)]

    @staticmethod
    def limbs(max_value, width=16):
        """(number of limbs, bit width of the top limb) for values up to max_value."""
        bits = max(int(max_value).bit_length(), 1)
        n = -(-bits // width)
        return n, bits - width * (n - 1)


# ----------------------------------------------------------------------- ASHE
class ASHE:
    """Seabed's additively symmetric homomorphic encryption (Papadimitriou et al.,
    OSDI 2016): Enc_k(m, i) = m + F_k(i) - F_k(i-1) mod 2^64 for row id i, with
    F = AES (here AES-256) used as a PRF. A sum over ids i in [a, b] telescopes to
    sum(m) + F_k(b) - F_k(a-1), so the decryptor needs the ids that were summed,
    as runs, and pays two PRF calls per run. Row ids are row positions, so a
    stored value costs 8 bytes. It needs the key to encrypt and decrypt and
    reveals the summed ids to the decryptor: it fits the owner-decrypts setting
    only. In-place updates must take a fresh id (reusing one reveals the
    difference of the two values), so updates append."""

    def __init__(self, key_bits=256):
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        self.cipher = Cipher(algorithms.AES(os.urandom(key_bits // 8)), modes.ECB())
        self.key_bits = key_bits

    def prf(self, ids):
        blocks = np.zeros((len(ids), 2), dtype=np.uint64)
        blocks[:, 0] = ids
        enc = self.cipher.encryptor().update(blocks.tobytes())
        return np.frombuffer(enc, dtype=np.uint64)[0::2].copy()

    def encrypt(self, values, first_id=1):
        """Encrypt values[j] under id first_id + j."""
        f = self.prf(np.arange(first_id - 1, first_id + len(values), dtype=np.int64))
        return np.asarray(values).astype(np.uint64) + f[1:] - f[:-1]

    @staticmethod
    def runs(ids):
        """Maximal runs of consecutive ids in a sorted id array, as (starts, ends)."""
        if len(ids) == 0:
            return ids[:0], ids[:0]
        br = np.flatnonzero(np.diff(ids) != 1)
        return ids[np.r_[0, br + 1]], ids[np.r_[br, len(ids) - 1]]

    @staticmethod
    def idlist_bytes(starts, ends, n_ids):
        """Bytes to ship an id set: runs as varint(gap) + varint(length), or a bitmap
        over all n_ids ids, whichever is smaller."""
        if len(starts) == 0:
            return 0
        gaps = starts - np.r_[1, ends[:-1] + 1]
        lens = ends - starts + 1
        vl = lambda x: 1 + np.floor(np.log2(np.maximum(x, 1)) / 7).astype(np.int64)
        return int(min(vl(gaps).sum() + vl(lens).sum(), -(-n_ids // 8)))

    def decrypt_sum(self, s, starts, ends):
        f = self.prf(np.concatenate([ends, starts - 1]))
        n = len(starts)
        return (int(s) - int(f[:n].sum(dtype=np.uint64)) + int(f[n:].sum(dtype=np.uint64))) % (1 << 64)

    @staticmethod
    def server_groupsum(cts, ids, gid, G):
        """Server side: per-group sums (mod 2^64) of each ciphertext column in cts
        (one array, or a list of arrays sharing the row ids) and the id runs per
        group. ids are 1-based row ids, ascending; gid[r] < 0 means not selected.
        Returns (sums, run_group, run_start, run_end); sums is a list if cts is."""
        many = isinstance(cts, (list, tuple))
        cols = list(cts) if many else [cts]
        sel = gid >= 0
        ids, g = ids[sel], gid[sel]
        order = np.argsort(g, kind="stable")
        ids, g = ids[order], g[order]
        starts_g = np.r_[0, np.flatnonzero(np.diff(g)) + 1] if len(ids) else np.zeros(0, dtype=np.int64)
        sums = []
        for ct in cols:
            s_ = np.zeros(G, dtype=np.uint64)
            if len(ids):
                s_[g[starts_g]] = np.add.reduceat(ct[ids - 1], starts_g)
            sums.append(s_)
        if len(ids):
            brk = np.flatnonzero((np.diff(ids) != 1) | (np.diff(g) != 0))
            rs, re_ = np.r_[0, brk + 1], np.r_[brk, len(ids) - 1]
            out = (g[rs], ids[rs], ids[re_])
        else:
            out = (g, ids, ids)
        return (sums if many else sums[0],) + out

    def decrypt_groups(self, sums, rg, rs, re_, G):
        """Client side: all G group sums at once; two PRF calls per run."""
        n = len(rs)
        f = self.prf(np.concatenate([re_, rs - 1]))
        delta = f[:n] - f[n:]                                  # wraps mod 2^64
        mask = np.zeros(G, dtype=np.uint64)
        if n:
            first = np.r_[0, np.flatnonzero(np.diff(rg)) + 1]
            mask[rg[first]] = np.add.reduceat(delta, first)
        return [int(x) for x in (np.asarray(sums, dtype=np.uint64) - mask)]

    @staticmethod
    def idlist_bytes_groups(rg, rs, re_, G, n_ids):
        """Total bytes to ship every group's id set (see idlist_bytes)."""
        if len(rs) == 0:
            return 0
        first = np.r_[0, np.flatnonzero(np.diff(rg)) + 1]
        last = np.r_[first[1:], len(rs)]
        return sum(ASHE.idlist_bytes(rs[a:b], re_[a:b], n_ids) for a, b in zip(first, last))
