#!/usr/bin/env python3
"""Harvest-now-decrypt-later attack on a DH-based private join-and-aggregate.

This script backs Sec. II-A of the paper. It

  1. runs the DDH-based private intersection-sum protocol of Ion et al.
     (EuroS&P 2020; the design behind Google's Private Join and Compute)
     between Hospital A (identifier/value pairs, owner of the additively
     homomorphic key) and Hospital B (identifiers only);
  2. records B's view of the session -- exactly what an honest-but-curious
     counterparty, or anyone who later obtains its logs, holds;
  3. plays the adversary *after Q-day* and recovers A's entire input table,
     although the protocol was meant to reveal to B only the intersection size.

The adversary needs two quantum computations: one elliptic-curve discrete
logarithm (A's blinding exponent) and one factorisation (A's Paillier modulus).
We cannot run Shor's algorithm, so the script *simulates those two steps by
handing the adversary their outputs*. Everything else is the adversary's
classical work, and that is what the script executes and times: a dictionary
walk over the identifier domain. The measured rate is extrapolated to the
10^9 possible nine-digit identifiers.

Blinding follows Private Join and Compute: identifiers are hashed to P-256
points (try-and-increment over SHA-256) and multiplied by secret exponents,
in C on OpenSSL's libcrypto (joinadj_kernel.c). Because the points are full
group elements, one discrete logarithm on the sum of B's blinded elements
against the sum of A's re-blinded ones yields A's exponent even though A
shuffles them.

    python3 benchmarks/hndl_join_aggregate_attack.py           # ~1-2 minutes
"""
import argparse
import hashlib
import json
import os
import platform
import random
import secrets
import time

import cryptography
from cryptography.hazmat.backends.openssl import backend
from cryptography.hazmat.primitives.asymmetric import rsa


# ----------------------------------------------------------------- blinding
import ctypes
import secrets as _secrets

from cryptdb import P256_N
from pqdb import _compile

_K = _compile("joinadj_kernel.c", "joinadj_kernel.so", ("-lcrypto", "-Wno-deprecated-declarations"))
_K.ja_new.restype = ctypes.c_void_p
_K.ec_hash_mul_many.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t,
                                ctypes.c_size_t, ctypes.c_char_p]
_K.ec_hash_mul_many.restype = None
_K.ja_adjust_many.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p]
_K.ja_adjust_many.restype = None
_J = _K.ja_new()
ID_LEN = 9


def new_exponent() -> bytes:
    return (_secrets.randbelow(P256_N - 1) + 1).to_bytes(32, "big")


def hash_blind(k: bytes, identifiers):
    """[k * H(u)] for nine-character identifiers u, as 33-byte compressed points."""
    buf = "".join(identifiers).encode()
    out = ctypes.create_string_buffer(33 * len(identifiers))
    _K.ec_hash_mul_many(_J, k, buf, ID_LEN, len(identifiers), out)
    raw = out.raw
    return [raw[33 * i:33 * (i + 1)] for i in range(len(identifiers))]


def reblind(k: bytes, points):
    """[k * P] for compressed points P."""
    out = ctypes.create_string_buffer(33 * len(points))
    _K.ja_adjust_many(_J, k, b"".join(points), len(points), out)
    raw = out.raw
    return [raw[33 * i:33 * (i + 1)] for i in range(len(points))]


# ------------------------------------------------------------------ Paillier
class Paillier:
    """Textbook Paillier with g = n + 1 (pure-Python big integers)."""

    def __init__(self, bits: int):
        nums = rsa.generate_private_key(public_exponent=65537, key_size=bits).private_numbers()
        self.p, self.q = nums.p, nums.q
        self.n = self.p * self.q
        self.n2 = self.n * self.n

    def enc(self, m: int) -> int:
        r = secrets.randbelow(self.n - 1) + 1
        return (1 + m * self.n) * pow(r, self.n, self.n2) % self.n2

    def add(self, c1: int, c2: int) -> int:
        return c1 * c2 % self.n2

    @staticmethod
    def dec_with_factors(c: int, p: int, q: int) -> int:
        n = p * q
        lam = (p - 1) * (q - 1)
        return (pow(c, lam, n * n) - 1) // n * pow(lam, -1, n) % n


def fmt_id(i: int) -> str:
    return f"{i:09d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", type=int, default=1_000_000,
                    help="identifier domain walked by the adversary (real SSN domain: 10^9)")
    ap.add_argument("--rows-a", type=int, default=300, help="A's (id, value) pairs")
    ap.add_argument("--rows-b", type=int, default=300, help="B's ids")
    ap.add_argument("--overlap", type=int, default=100)
    ap.add_argument("--paillier-bits", type=int, default=3072)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results",
                                                  "hndl_join_aggregate_attack.json"))
    args = ap.parse_args()
    rng = random.Random(args.seed)

    # ------------------------------------------------------------ the inputs
    ids = rng.sample(range(args.domain), args.rows_a + args.rows_b - args.overlap)
    table_a = {fmt_id(i): rng.randint(120, 320) for i in ids[:args.rows_a]}       # ssn -> cholesterol
    set_b = [fmt_id(i) for i in ids[args.rows_a - args.overlap:]]                 # ssn of bypass patients
    truth_ids = set(table_a) & set(set_b)
    truth_sum = sum(table_a[i] for i in truth_ids)

    # ------------------------------------------------- the protocol (honest run)
    k_a, k_b = new_exponent(), new_exponent()
    ahe = Paillier(args.paillier_bits)                       # A's key pair

    # Round 1, B -> A: B's identifiers blinded with k_b.
    msg1 = hash_blind(k_b, set_b)
    rng.shuffle(msg1)
    # Round 2, A -> B: (i) B's elements blinded again with k_a;
    #                  (ii) A's identifiers blinded with k_a, with encrypted values.
    msg2_double = reblind(k_a, msg1)
    rng.shuffle(msg2_double)
    msg2_pairs = list(zip(hash_blind(k_a, list(table_a)), [ahe.enc(t) for t in table_a.values()]))
    rng.shuffle(msg2_pairs)
    # Round 3, B -> A: homomorphic sum over the intersection.
    double_set = set(msg2_double)
    acc, card = ahe.enc(0), 0
    for z2, (z, c) in zip(reblind(k_b, [z for z, _ in msg2_pairs]), msg2_pairs):
        if z2 in double_set:
            acc, card = ahe.add(acc, c), card + 1
    # Output: A decrypts the sum; B learns only the cardinality.
    out_sum = Paillier.dec_with_factors(acc, ahe.p, ahe.q)
    assert (out_sum, card) == (truth_sum, len(truth_ids)), "protocol run is incorrect"

    # What B (or whoever obtains B's logs) holds after the session:
    view_b = dict(own_blinded=msg1, double_blinded=msg2_double, pairs=msg2_pairs, n=ahe.n)

    # ------------------------------------------------------- after Q-day
    # Quantum step 1 (simulated): one EC discrete logarithm yields A's exponent.
    k_a_recovered = k_a
    # Quantum step 2 (simulated): factoring view_b["n"] yields A's Paillier key.
    p_recovered, q_recovered = ahe.p, ahe.q

    # Classical step: walk the identifier domain.
    lookup = {z: c for z, c in view_b["pairs"]}
    recovered = {}
    t0 = time.perf_counter()
    CH = 50_000
    for lo in range(0, args.domain, CH):
        cand = [fmt_id(i) for i in range(lo, min(lo + CH, args.domain))]
        for s, z in zip(cand, hash_blind(k_a_recovered, cand)):
            c = lookup.get(z)
            if c is not None:
                recovered[s] = c
    walk_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    recovered = {s: Paillier.dec_with_factors(c, p_recovered, q_recovered)
                 for s, c in recovered.items()}
    dec_seconds = time.perf_counter() - t0

    assert recovered == table_a, "attack did not recover A's full table"
    rate = args.domain / walk_seconds
    result = dict(
        machine=dict(cpu=open("/proc/cpuinfo").read().split("model name")[1].split(":")[1].split("\n")[0].strip()
                     if os.path.exists("/proc/cpuinfo") else platform.processor(),
                     python=platform.python_version(), cryptography=cryptography.__version__,
                     openssl=backend.openssl_version_text()),
        protocol="DDH-based private intersection-sum (Ion et al.), P-256 blinding as in Private Join and Compute, "
                 f"Paillier-{args.paillier_bits}",
        intended_output=dict(to_b="intersection cardinality", to_a="sum over the intersection",
                             cardinality=card, sum=out_sum),
        simulated_quantum_steps=["one elliptic-curve discrete logarithm (A's blinding exponent)",
                                 "one factorisation (A's Paillier modulus)"],
        rows_a=args.rows_a, rows_b=args.rows_b, true_intersection=len(truth_ids),
        recovered_pairs=len(recovered),
        recovered_pairs_outside_intersection=len(set(recovered) - truth_ids),
        recovered_equals_a_input=True,
        domain_walked=args.domain, walk_seconds=round(walk_seconds, 2),
        candidates_per_second_one_core=round(rate),
        extrapolated_core_hours_for_1e9_identifiers=round(1e9 / rate / 3600, 1),
        paillier_decryptions_per_second_one_core=round(len(recovered) / dec_seconds, 1),
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
