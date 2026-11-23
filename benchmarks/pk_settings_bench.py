#!/usr/bin/env python3
"""Settings that require public-key structure, classical vs post-quantum (Fig. 6a-b).

CryptDB's proxy holds every key, so most of its onions can be symmetric. Two
common settings rule that out, and this script measures both.

1. Many writers, one aggregator (telemetry, smart meters, IoT; Talos, SenSys'15).
   Writers hold only a public key; a server aggregates; the owner reads the total.
     non-PQ   EC-ElGamal on P-256 (Talos), Paillier-2048 (CryptDB's HOM at today's
              common size, public key only), and Paillier packed with 32 readings of 64 bits per
              plaintext (MONOMI-style packing) when a writer sends a batch
     PQ       BFV-8192 public-key encryption (one ciphertext per batch of up to
              8,192 readings), or ML-KEM + AES-GCM (KEM-DEM), which hides the
              readings but cannot be aggregated, so every reading goes to the owner
   Swept over the batch size b a writer sends at once: writer time and bytes per
   reading, server time per reading, and what the owner downloads for 10^6
   readings. ML-KEM-768, the recommended parameter set, is used throughout.

2. Tuples that third parties must verify (ledger and verifiable databases).
   A symmetric MAC suffices only if every verifier may hold the key; otherwise
   each row carries a signature.
     non-PQ   ECDSA P-256 (HMAC-SHA256 shown for the shared-key case)
     PQ       ML-DSA-44, ML-DSA-65
   50,000 rows of 100-byte payload in SQLite (4 KB pages) and, if a PostgreSQL 16
   server is installed, in PostgreSQL (8 KB pages, TOAST above ~2 KB): sign,
   load, table size (heap and TOAST), a scan that reads only the payload, a scan
   that fetches every signature, verification, and whether a B-tree index on the
   signature column can be built. All signatures are genuine.

    python3 benchmarks/pk_settings_bench.py      # ~5 minutes
"""
import hashlib
import hmac
import json
import os
import sqlite3
import tempfile

import numpy as np
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, mldsa, mlkem
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from pqdb import BFV, ECElGamal, Paillier, cpu_model, rate, timed

HERE = os.path.dirname(os.path.abspath(__file__))


def _build_dir():
    d = os.path.join(HERE, "build")
    os.makedirs(d, exist_ok=True)
    return d
BATCHES = [1, 4, 16, 64, 256, 1024, 4096, 8192]
PACK = 32                     # 64-bit slots in a 2048-bit Paillier plaintext
N_READINGS = 1_000_000
ROWS = 50_000
KEM_CT = 1088                 # ML-KEM-768 ciphertext, FIPS 203
GCM_OVERHEAD = 12 + 16        # nonce + tag


def writers(rng):
    pa, eg = Paillier(2048), ECElGamal(16)
    b = BFV(8192, [40, 40, 40, 41, 41], 50, rng)
    CT = b.ciphertext_bytes()
    # writer side
    t_pa = 1 / rate(lambda: pa.encrypt_public(1234))
    t_eg = 1 / rate(lambda: eg.encrypt(1234))
    full = b.encode(rng.integers(0, 1 << 16, b.n))
    t_bfv = 1 / rate(lambda: b.encrypt(full))                 # one ciphertext, any b <= n
    kem_pk = mlkem.MLKEM768PrivateKey.generate().public_key()
    t_kem = 1 / rate(lambda: kem_pk.encapsulate())
    aes = AESGCM(AESGCM.generate_key(bit_length=256))
    nonce = os.urandom(12)
    t_aes = {bb: 1 / rate(lambda: aes.encrypt(nonce, os.urandom(8 * bb), None), seconds=0.3) for bb in BATCHES}
    # server side, per reading
    pool = pa.make_pool(np.arange(PACK))
    idx = rng.integers(0, PACK, 100_000).astype(np.int32)
    gid = np.zeros(100_000, dtype=np.int32)
    s_pa = 1 / rate(lambda: pa.groupsum(pool, idx, gid, 1)) / 100_000
    gpool = eg.make_pool(np.arange(64, dtype=np.uint64))
    s_eg = 1 / rate(lambda: eg.groupsum(gpool, idx % 64, gid, 1)) / 100_000
    acc = b.zero()
    c = b.encrypt(full)
    s_bfv = 1 / rate(lambda: b.add_inplace(acc, c))         # per ciphertext
    # owner side
    pct = pa.encrypt_bytes(5)
    o_pa = 1 / rate(lambda: pa.decrypt_int(pct))
    o_bfv = 1 / rate(lambda: b.decrypt_slots(c))
    big = eg.encrypt((1 << 24) - 1)                          # decryption time grows with the sum;
    o_eg = 1 / rate(lambda: eg.decrypt(big, 1 << 25))       # timed for a sum just below 2^24
    rows = []
    for bb in BATCHES:
        rows.append(dict(
            b=bb,
            bytes=dict(paillier=pa.ctlen, paillier_packed=pa.ctlen / min(bb, PACK), ecelgamal=eg.CT,
                       bfv=CT / bb, kemdem=(KEM_CT + GCM_OVERHEAD) / bb + 8),
            writer=dict(paillier=t_pa, paillier_packed=t_pa / min(bb, PACK), ecelgamal=t_eg,
                        bfv=t_bfv / bb, kemdem=(t_kem + t_aes[bb]) / bb),
            server=dict(paillier=s_pa, paillier_packed=s_pa / min(bb, PACK), ecelgamal=s_eg,
                        bfv=s_bfv / bb, kemdem=0.0),
            owner_bytes=dict(paillier=pa.ctlen, paillier_packed=pa.ctlen, ecelgamal=eg.CT, bfv=CT,
                             kemdem=N_READINGS * ((KEM_CT + GCM_OVERHEAD) / bb + 8))))
    return dict(rows=rows, unit=dict(paillier_public_encrypt=t_pa, ecelgamal_encrypt=t_eg, bfv_encrypt=t_bfv,
                                     mlkem768_encaps=t_kem, paillier_add=s_pa, ecelgamal_add=s_eg,
                                     bfv_add=s_bfv, paillier_decrypt=o_pa, bfv_decrypt=o_bfv,
                                     ecelgamal_decrypt_2_24=o_eg),
                sizes=dict(paillier=pa.ctlen, ecelgamal=eg.CT, bfv=CT, mlkem768_ct=KEM_CT))


# PostgreSQL 16 for the signed-tuple measurements; skipped if PG_BIN has no server.
#   PQDB_PG_BIN   directory with initdb, pg_ctl, postgres and psql
#   PQDB_PG_DIR   scratch directory for the throwaway cluster (deleted and recreated)
#   PQDB_PG_PORT  port of the throwaway cluster
#   PQDB_PG_USER  unprivileged user the server runs as when this script runs as root
#                 (created if missing), since PostgreSQL refuses to run as root
PG_BIN = os.environ.get("PQDB_PG_BIN", "/usr/lib/postgresql/16/bin")
PG_DIR = os.environ.get("PQDB_PG_DIR", os.path.join(tempfile.gettempdir(), "pqdb_pg"))
PG_PORT = os.environ.get("PQDB_PG_PORT", "54329")
PG_USER = os.environ.get("PQDB_PG_USER", "pqdbpg")


class Postgres:
    """A throwaway PostgreSQL 16 cluster (default 8 KB pages, pglz TOAST compression)."""

    def __init__(self):
        import shutil
        import subprocess
        self.sp = subprocess
        self.root = os.geteuid() == 0
        if self.root and subprocess.run(["id", PG_USER], capture_output=True).returncode:
            subprocess.run(["useradd", "-M", PG_USER], check=True)
        shutil.rmtree(PG_DIR, ignore_errors=True)
        os.makedirs(PG_DIR)
        if self.root:
            shutil.chown(PG_DIR, PG_USER)
        self.run_as([f"{PG_BIN}/initdb", "-D", f"{PG_DIR}/data", "-A", "trust", "-U", "postgres"])
        self.run_as([f"{PG_BIN}/pg_ctl", "-D", f"{PG_DIR}/data", "-o", f"-p {PG_PORT} -k {PG_DIR}",
                     "-l", f"{PG_DIR}/log", "-w", "start"])

    def run_as(self, cmd):
        prefix = ["runuser", "-u", PG_USER, "--"] if self.root else []
        self.sp.run(prefix + cmd, check=True, capture_output=True)

    def psql(self, script):
        psql = f"{PG_BIN}/psql" if os.path.exists(f"{PG_BIN}/psql") else "psql"
        r = self.sp.run([psql, "-h", PG_DIR, "-p", PG_PORT, "-U", "postgres", "-X", "-q", "-A", "-t",
                         "-v", "ON_ERROR_STOP=0", "-c", "\\timing on"] + sum((["-c", c] for c in script), []),
                        capture_output=True, text=True)
        return r.stdout, r.stderr

    def timed_sql(self, sql, repeat=1):
        out, err = self.psql([sql] * repeat)
        ms = [float(l.split()[1]) for l in out.splitlines() if l.startswith("Time:")]
        return min(ms) / 1e3 if ms else None, out, err

    def stop(self):
        self.run_as([f"{PG_BIN}/pg_ctl", "-D", f"{PG_DIR}/data", "-m", "fast", "stop"])


def ledger_pg(pg, payload, sigs):
    """Load, size, scan and fetch a signed table in PostgreSQL; try a B-tree on the signature."""
    path = os.path.join(PG_DIR, "rows.tsv")
    with open(path, "w") as f:
        for i, (m, sg) in enumerate(zip(payload, sigs)):
            f.write(f"{i}\t\\\\x{m.hex()}\t\\\\x{sg.hex()}\n")
    os.chmod(path, 0o644)
    pg.psql(["DROP TABLE IF EXISTS t", "CREATE TABLE t (id int PRIMARY KEY, payload bytea, sig bytea)"])
    t_copy, _, err = pg.timed_sql(f"COPY t FROM '{path}'")
    assert not err.strip(), err
    pg.psql(["VACUUM ANALYZE t"])
    out, _ = pg.psql(["SELECT pg_relation_size('t'), pg_total_relation_size('t'), "
                      "pg_relation_size(reltoastrelid) FROM pg_class WHERE relname = 't'"])
    heap, total, toast = map(int, [l for l in out.splitlines() if "|" in l][0].split("|"))
    t_scan, _, _ = pg.timed_sql("SELECT sum(length(payload)) FROM t", repeat=3)
    t_fetch, _, _ = pg.timed_sql("SELECT sum(get_byte(sig, 0)) FROM t", repeat=3)   # forces detoasting
    _, _, ierr = pg.timed_sql("CREATE INDEX t_sig ON t (sig)")
    os.remove(path)
    return dict(copy_seconds=t_copy, heap_bytes=heap, toast_bytes=toast, total_bytes=total,
                scan_payload_seconds=t_scan, fetch_signatures_seconds=t_fetch,
                btree_on_signature="ok" if not ierr.strip() else ierr.strip().splitlines()[0])


def ledger(rng):
    payload = [os.urandom(100) for _ in range(ROWS)]
    key = os.urandom(32)
    p256 = ec.generate_private_key(ec.SECP256R1())
    schemes = {
        "HMAC-SHA256": (lambda m: hmac.new(key, m, hashlib.sha256).digest(),
                        lambda m, s: hmac.compare_digest(hmac.new(key, m, hashlib.sha256).digest(), s)),
        "ECDSA P-256": (lambda m: p256.sign(m, ec.ECDSA(hashes.SHA256())),
                        lambda m, s: p256.public_key().verify(s, m, ec.ECDSA(hashes.SHA256())) is None),
    }
    for name, cls in (("ML-DSA-44", mldsa.MLDSA44PrivateKey), ("ML-DSA-65", mldsa.MLDSA65PrivateKey)):
        sk = cls.generate()
        schemes[name] = (sk.sign, lambda m, s, pk=sk.public_key(): pk.verify(s, m) is None)
    out = {}
    pg = Postgres() if os.path.exists(f"{PG_BIN}/postgres") else None
    tmp = tempfile.mkdtemp(dir=_build_dir())
    for name, (sign, verify) in schemes.items():
        path = os.path.join(tmp, "ledger.db")
        db = sqlite3.connect(path)
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, payload BLOB, sig BLOB)")
        sigs, t_sign = timed(lambda: [sign(m) for m in payload])

        def insert():
            with db:
                db.executemany("INSERT INTO t VALUES (?, ?, ?)", zip(range(ROWS), payload, sigs))
        _, t_ins = timed(insert)
        pages = db.execute("PRAGMA page_count").fetchone()[0] * db.execute("PRAGMA page_size").fetchone()[0]
        db.close()
        db = sqlite3.connect(path)
        _, t_scan = timed(lambda: db.execute("SELECT sum(length(payload)) FROM t").fetchone())

        def verify_all():
            return all(verify(m, s) for m, s in db.execute("SELECT payload, sig FROM t"))
        ok, t_ver = timed(verify_all)
        assert ok
        db.close()
        os.remove(path)
        out[name] = dict(sig_bytes=len(sigs[0]), sign_seconds=t_sign, insert_seconds=t_ins, db_bytes=pages,
                         scan_seconds=t_scan, verify_seconds=t_ver)
        print(name, json.dumps({k: round(v, 4) for k, v in out[name].items()}))
        if pg:
            out[name]["postgres"] = ledger_pg(pg, payload, sigs)
            print("   postgres", json.dumps(out[name]["postgres"], default=lambda x: round(x, 4)))
    os.rmdir(tmp)
    if pg:
        pg.stop()
    return out


def main():
    rng = np.random.default_rng(5)
    res = dict(machine=dict(cpu=cpu_model()), rows=ROWS, readings=N_READINGS)
    res["writers"] = writers(rng)
    for r in res["writers"]["rows"]:
        print("b=%-5d" % r["b"], {k: round(v, 1) for k, v in r["bytes"].items()})
    res["ledger"] = ledger(rng)
    path = os.path.join(HERE, "results", "pk_settings.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1, default=float)
    print("wrote", path)


if __name__ == "__main__":
    main()
