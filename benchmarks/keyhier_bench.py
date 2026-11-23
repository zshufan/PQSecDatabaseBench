#!/usr/bin/env python3
"""Key hierarchies for encrypted storage, classical vs post-quantum (Fig. 6c).

Encrypted databases keep data keys (one per column, partition, tenant, or user)
wrapped under an asymmetric master key, as Always Encrypted does with its
column master key and cloud key management services do with their key
encryption keys. Ingest and backup jobs then wrap keys with the public half,
and only the holder of the private half (often a hardware module) unwraps them.

For N = 100,000 random 256-bit data keys, each scheme wraps every key, unwraps
every key (checked against the original), and rotates the master key, which
unwraps every key under the old private key and wraps it under a new public
key (as Always Encrypted does when a column master key is rotated). The stored
size of a wrapped key is recorded.

    RSA-2048-OAEP     non-PQ   RSA-OAEP with SHA-256 encrypts the data key directly
                               (Always Encrypted's only accepted algorithm; cloud KMS)
    X25519            non-PQ   ephemeral-static X25519, HKDF-SHA256, AES-256 key wrap
                               (the shape of HPKE's DHKEM(X25519), RFC 9180)
    ML-KEM-768        PQ       encapsulation, HKDF-SHA256, AES-256 key wrap
    X25519+ML-KEM-768 hybrid   both shared secrets into one HKDF, AES-256 key wrap

    python3 benchmarks/keyhier_bench.py       # ~3 minutes

Writes benchmarks/results/key_hierarchy.json.
"""
import json
import os

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import mlkem, padding, rsa, x25519
from cryptography.hazmat.primitives.keywrap import aes_key_unwrap, aes_key_wrap
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from pqdb import cpu_model, timed

HERE = os.path.dirname(os.path.abspath(__file__))
N = 100_000
OAEP = padding.OAEP(mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(), label=None)
RAW = dict(encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)


def kek(secret, info):
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=info).derive(secret)


class RSAWrap:
    name = "RSA-2048-OAEP"

    def __init__(self):
        self.sk = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.pk = self.sk.public_key()

    def wrap(self, dek):
        return self.pk.encrypt(dek, OAEP)

    def unwrap(self, w):
        return self.sk.decrypt(w, OAEP)


class X25519Wrap:
    name = "X25519"

    def __init__(self):
        self.sk = x25519.X25519PrivateKey.generate()
        self.pk = self.sk.public_key()
        self.pkb = self.pk.public_bytes(**RAW)

    def wrap(self, dek):
        e = x25519.X25519PrivateKey.generate()
        epk = e.public_key().public_bytes(**RAW)
        return epk + aes_key_wrap(kek(e.exchange(self.pk), epk + self.pkb), dek)

    def unwrap(self, w):
        epk = w[:32]
        ss = self.sk.exchange(x25519.X25519PublicKey.from_public_bytes(epk))
        return aes_key_unwrap(kek(ss, epk + self.pkb), w[32:])


class MLKEMWrap:
    name = "ML-KEM-768"
    CT = 1088

    def __init__(self):
        self.sk = mlkem.MLKEM768PrivateKey.generate()
        self.pk = self.sk.public_key()

    def wrap(self, dek):
        ss, ct = self.pk.encapsulate()
        return ct + aes_key_wrap(kek(ss, ct), dek)

    def unwrap(self, w):
        ct = w[:self.CT]
        return aes_key_unwrap(kek(self.sk.decapsulate(ct), ct), w[self.CT:])


class HybridWrap:
    name = "X25519+ML-KEM-768"
    CT = 1088

    def __init__(self):
        self.x, self.m = X25519Wrap(), MLKEMWrap()

    def wrap(self, dek):
        e = x25519.X25519PrivateKey.generate()
        epk = e.public_key().public_bytes(**RAW)
        ss, ct = self.m.pk.encapsulate()
        k = kek(e.exchange(self.x.pk) + ss, epk + ct + self.x.pkb)
        return epk + ct + aes_key_wrap(k, dek)

    def unwrap(self, w):
        epk, ct = w[:32], w[32:32 + self.CT]
        ss1 = self.x.sk.exchange(x25519.X25519PublicKey.from_public_bytes(epk))
        ss2 = self.m.sk.decapsulate(ct)
        return aes_key_unwrap(kek(ss1 + ss2, epk + ct + self.x.pkb), w[32 + self.CT:])


def run(cls, deks):
    old = cls()
    wrapped, t_wrap = timed(lambda: [old.wrap(d) for d in deks])
    back, t_unwrap = timed(lambda: [old.unwrap(w) for w in wrapped])
    assert back == deks, cls.name
    new = cls()
    rotated, t_rotate = timed(lambda: [new.wrap(old.unwrap(w)) for w in wrapped])
    assert [new.unwrap(w) for w in rotated[:1000]] == deks[:1000], cls.name
    size = len(wrapped[0])
    assert all(len(w) == size for w in wrapped)
    return dict(keys=len(deks), wrap_seconds=t_wrap, unwrap_seconds=t_unwrap, rotate_seconds=t_rotate,
                wrapped_bytes=size, total_bytes=size * len(deks))


def main():
    deks = [os.urandom(32) for _ in range(N)]
    out = dict(machine=dict(cpu=cpu_model(), logical_cpus=os.cpu_count()), keys=N, data_key_bits=256,
               method="single thread; every key wrapped, unwrapped and checked, then re-wrapped under a new master key",
               schemes={})
    for cls in (RSAWrap, X25519Wrap, MLKEMWrap, HybridWrap):
        r = run(cls, deks)
        out["schemes"][cls.name] = r
        print(f"{cls.name:18s} wrap {r['wrap_seconds']:6.2f}s  unwrap {r['unwrap_seconds']:6.2f}s  "
              f"rotate {r['rotate_seconds']:6.2f}s  {r['wrapped_bytes']} B/key")
    path = os.path.join(HERE, "results", "key_hierarchy.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("wrote", path)


if __name__ == "__main__":
    main()
