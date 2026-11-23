#!/usr/bin/env python3
"""What larger signatures do to storage and I/O (Sec. III-E, tuples that third parties verify).

The signed table of pk_settings_bench.py (50,000 rows, a 100-byte payload and
one signature each) is rebuilt with genuine signatures and measured in pages
rather than seconds, so the result does not depend on this machine's disks:

  SQLite (4 KB pages)   pages in the table and rows per page; a scan of any
                        column reads every page, since signatures are stored
                        in line.
  PostgreSQL 16         heap and TOAST pages, and the buffers a query touches
  (8 KB pages,          (EXPLAIN (ANALYZE, BUFFERS)) for a scan of the payload
  shared_buffers        and for a scan that fetches every signature. Each query
  128 MB, default)      runs twice; the second run shows how many pages still
                        have to be read because they do not stay in the
                        buffer pool.

Also reported: signature bytes as a multiple of the payload, and rows that fit
in 1 GB of buffer pool (table bytes per row).

    python3 benchmarks/storage_io_bench.py     # ~4 minutes, needs PostgreSQL 16 for the second part

Writes benchmarks/results/storage_io.json.
"""
import hashlib
import hmac
import json
import os
import sqlite3
import tempfile

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, mldsa

from pk_settings_bench import PG_BIN, PG_DIR, ROWS, Postgres
from pqdb import cpu_model

HERE = os.path.dirname(os.path.abspath(__file__))


def _build_dir():
    d = os.path.join(HERE, "build")
    os.makedirs(d, exist_ok=True)
    return d
PAYLOAD = 100


def explain(pg, sql):
    out, err = pg.psql(["EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + sql])
    assert not err.strip(), err
    plan = json.loads(out[out.index("["):out.rindex("]") + 1])[0]["Plan"]
    return dict(hit=plan.get("Shared Hit Blocks", 0), read=plan.get("Shared Read Blocks", 0))


def pg_pages(pg, payload, sigs):
    path = os.path.join(PG_DIR, "rows.tsv")
    with open(path, "w") as f:
        for i, (m, sg) in enumerate(zip(payload, sigs)):
            f.write(f"{i}\t\\\\x{m.hex()}\t\\\\x{sg.hex()}\n")
    os.chmod(path, 0o644)
    pg.psql(["DROP TABLE IF EXISTS t", "CREATE TABLE t (id int PRIMARY KEY, payload bytea, sig bytea)",
             f"COPY t FROM '{path}'", "VACUUM ANALYZE t"])
    os.remove(path)
    out, _ = pg.psql(["SELECT current_setting('block_size'), pg_relation_size('t'), "
                      "coalesce(pg_relation_size(reltoastrelid), 0), pg_total_relation_size('t'), "
                      "current_setting('shared_buffers') FROM pg_class WHERE relname = 't'"])
    bs, heap, toast, total, sb = [l for l in out.splitlines() if "|" in l][0].split("|")
    bs = int(bs)
    res = dict(block_size=bs, heap_pages=int(heap) // bs, toast_pages=int(toast) // bs, total_bytes=int(total),
               shared_buffers=sb, queries={})
    for name, sql in (("scan_payload", "SELECT sum(length(payload)) FROM t"),
                      ("fetch_signatures", "SELECT sum(get_byte(sig, 0)) FROM t")):
        first, second = explain(pg, sql), explain(pg, sql)
        res["queries"][name] = dict(first=first, second=second, pages_touched=second["hit"] + second["read"])
    return res


def main():
    payload = [os.urandom(PAYLOAD) for _ in range(ROWS)]
    key = os.urandom(32)
    p256 = ec.generate_private_key(ec.SECP256R1())
    schemes = {"HMAC-SHA256": lambda m: hmac.new(key, m, hashlib.sha256).digest(),
               "ECDSA P-256": lambda m: p256.sign(m, ec.ECDSA(hashes.SHA256())),
               "ML-DSA-44": mldsa.MLDSA44PrivateKey.generate().sign,
               "ML-DSA-65": mldsa.MLDSA65PrivateKey.generate().sign}
    pg = Postgres() if os.path.exists(f"{PG_BIN}/postgres") else None
    out = dict(machine=dict(cpu=cpu_model()), rows=ROWS, payload_bytes=PAYLOAD, schemes={})
    tmp = tempfile.mkdtemp(dir=_build_dir())
    try:
        for name, sign in schemes.items():
            sigs = [sign(m) for m in payload]
            path = os.path.join(tmp, "t.db")
            db = sqlite3.connect(path)
            db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, payload BLOB, sig BLOB)")
            with db:
                db.executemany("INSERT INTO t VALUES (?, ?, ?)", zip(range(ROWS), payload, sigs))
            ps, pc = (db.execute(f"PRAGMA {p}").fetchone()[0] for p in ("page_size", "page_count"))
            db.close()
            os.remove(path)
            sig = len(sigs[0])
            r = dict(sig_bytes=sig, sig_over_payload=sig / PAYLOAD,
                     sqlite=dict(page_size=ps, pages=pc, rows_per_page=ROWS / pc, scan_pages=pc,
                                 rows_per_gb=ROWS * 1e9 / (ps * pc)))
            if pg:
                r["postgres"] = pg_pages(pg, payload, sigs)
                r["postgres"]["rows_per_gb"] = ROWS * 1e9 / r["postgres"]["total_bytes"]
            out["schemes"][name] = r
            print(name, json.dumps(r))
    finally:
        os.rmdir(tmp)
        if pg:
            pg.stop()
    path = os.path.join(HERE, "results", "storage_io.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
