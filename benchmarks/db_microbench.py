#!/usr/bin/env python3
"""Operator level: CryptDB (non-PQ) vs its post-quantum counterpart (Fig. 4).

The non-PQ side is CryptDB's onion design (Popa et al., SOSP 2011) at 128-bit
classical security; the PQ side replaces each layer with its post-quantum
counterpart:

  onion / operator          CryptDB (non-PQ)            post-quantum
  Eq:  DET under RND        AES-128                     AES-256
  Ord: OPE under RND        Boldyreva OPE, AES-128 PRF  same, AES-256 PRF
  Join                      JOIN-ADJ on P-256           none at the same leakage; we time
                                                        a DET key shared by the join columns
                                                        from ingest, which reveals their
                                                        equalities before any join is issued
  Add: HOM                  Paillier-2048               BFV-8192 (server- or client-side
                                                        grouping); ASHE with AES-256 (the
                                                        data owner decrypts, as CryptDB's
                                                        proxy does)

Over a 1,000,000-row table: ingest of each onion, equality and range selection
(client encrypts the constants, server scans), the one-time JOIN-ADJ
adjustment and an equi-join with a 250,000-row table, and for HOM: ingest, SUM,
1%-selective SUM, 64-group GROUP BY with rows in random order, and a one-row
update. Per-tuple signatures (ECDSA P-256 vs ML-DSA-44) and a 32 MB AES chunk
are kept for the text.

Every aggregate is checked against the plaintext answer. Paillier columns reuse
a pool of 4,096 genuine ciphertexts by index, so Paillier ingest is projected
from the measured per-value rate (as are the signature rows); everything else
is timed end to end. Writes benchmarks/results/db_microbench.json.

    python3 benchmarks/db_microbench.py      # ~4 minutes; needs primitives.json
"""
import json
import os

import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

import pandas as pd

from cryptdb import DET, OPE, RND, JoinAdj
from pqdb import ASHE, BFV, Paillier, cpu_model, rate, timed

HERE = os.path.dirname(os.path.abspath(__file__))
ROWS, POOL, GROUPS, SEL = 1_000_000, 4096, 64, 0.01
VMAX = 1000
KEYS, DAYS, PK = 100_000, 3650, 250_000


def best(fn, n=3, before=None):
    """Result and the minimum time over n runs (for short symmetric-key operations)."""
    ts = []
    for _ in range(n):
        if before:
            before()
        out, t = timed(fn)
        ts.append(t)
    return out, min(ts)


def main():
    rng = np.random.default_rng(7)
    L1 = json.load(open(os.path.join(HERE, "results", "primitives.json")))
    res = dict(rows=ROWS, machine=dict(cpu=cpu_model()), ops={})

    def put(op, scheme, **kw):
        res["ops"].setdefault(op, {})[scheme] = kw

    # ------------------------------------------- Eq, Ord and Join onions
    kcol = rng.integers(0, KEYS, ROWS)                 # equality column
    dcol = rng.integers(0, DAYS, ROWS)                 # date column (day numbers)
    fk = rng.integers(1, PK + 1, ROWS)                 # foreign key into a PK-row table
    pk = np.arange(1, PK + 1)
    eq_c, lo_d, hi_d = 4242, 1000, 1036                # equality constant; a ~1% date range
    ja = JoinAdj(128)
    for scheme, bits in (("cryptdb", 128), ("pq", 256)):
        det, rnd = DET(bits), RND(bits)

        def eq_ingest():
            return rnd.enc(det.enc(kcol))
        (iv, ct), t = best(eq_ingest)
        put("eq_ingest", scheme, seconds=t, bytes=ROWS * RND.bytes_per_value)
        inner, t = best(lambda: rnd.dec(iv, ct))       # one-time peel of the RND layer
        put("eq_peel", scheme, seconds=t, bytes=0)
        skey = DET.server_key(inner)

        def eq_select():
            c = DET.server_key(det.enc(np.array([eq_c])))[0]   # client: encrypt the constant
            return np.flatnonzero(skey == c)                   # server: scan
        rows, t = best(eq_select)
        assert len(rows) == int((kcol == eq_c).sum())
        put("eq_select", scheme, seconds=t, bytes=len(rows) * RND.bytes_per_value)

        ope = OPE(bits)

        def ord_ingest():
            o, n = ope.encrypt_many(dcol)
            return o, n, rnd.enc(np.c_[o, np.zeros_like(o)])
        (ocol, ndist, _), t = best(ord_ingest, before=ope.cache.clear)
        put("ord_ingest", scheme, seconds=t, bytes=ROWS * RND.bytes_per_value, distinct=ndist)
        ope.cache.clear()                              # query constants are encrypted afresh

        def ord_select():
            lo, hi = ope.encrypt(lo_d), ope.encrypt(hi_d)       # client
            return np.flatnonzero((ocol >= lo) & (ocol <= hi))   # server
        rows, t = best(ord_select, before=ope.cache.clear)
        assert len(rows) == int(((dcol >= lo_d) & (dcol <= hi_d)).sum())
        put("ord_select", scheme, seconds=t, bytes=len(rows) * RND.bytes_per_value)

        if scheme == "cryptdb":
            k_fk, k_pk = ja.new_key(), ja.new_key()
            ufk, inv = np.unique(fk, return_inverse=True)

            def join_ingest():
                return ja.encrypt_distinct(k_fk, ufk), ja.encrypt_distinct(k_pk, pk)
            (cfk, cpk), t = timed(join_ingest)
            put("join_ingest", scheme, seconds=t, bytes=ROWS * (JoinAdj.PT + DET.bytes_per_value),
                note="JOIN-ADJ || DET per row; distinct values encrypted once")
            adj, t = timed(lambda: ja.adjust(cfk, k_fk, k_pk))   # first join query: one-time
            put("join_adjust", scheme, seconds=t, bytes=0, distinct=len(ufk))
            jfk, jpk = JoinAdj.server_key(adj)[inv], JoinAdj.server_key(cpk)
        else:
            shared = DET(bits)                          # one key for the whole join group

            def join_ingest():
                return shared.enc(fk), shared.enc(pk)
            (dfk, dpk), t = best(join_ingest)
            put("join_ingest", scheme, seconds=t, bytes=ROWS * DET.bytes_per_value, leaks_more_today=True,
                note="DET under a key shared by the join columns")
            put("join_adjust", scheme, seconds=0.0, bytes=0, not_applicable=True)
            jfk, jpk = DET.server_key(dfk), DET.server_key(dpk)

        def join_run():
            return pd.DataFrame({"j": jfk}).merge(pd.DataFrame({"j": jpk, "r": np.arange(PK)}), on="j")
        out, t = best(join_run)
        assert len(out) == ROWS
        put("join_run", scheme, seconds=t, bytes=0)

    # ------------------------------------------------------- HOM: setup
    pa = Paillier(2048)
    ah = ASHE(256)
    pool_vals = rng.integers(0, VMAX, POOL)
    ppool = pa.make_pool(pool_vals)
    idx = rng.integers(0, POOL, ROWS).astype(np.int32)
    vals = pool_vals[idx]                          # the column every scheme holds
    ids = np.arange(1, ROWS + 1, dtype=np.int64)
    b = BFV(8192, [40, 40, 40, 41, 41], 50, rng)
    N, CT = b.n, b.ciphertext_bytes()
    K = -(-ROWS // N)
    padded = np.zeros(K * N, dtype=np.int64)
    padded[:ROWS] = vals
    chunks = padded.reshape(K, N)
    real = (np.arange(K * N) < ROWS).reshape(K, N)

    # ---------------------------------------------------------------- ingest
    p_enc = rate(lambda: pa.encrypt(123))
    put("ingest", "paillier", seconds=ROWS / p_enc, projected=True, bytes=ROWS * pa.ctlen)
    cts, t = timed(lambda: [b.encrypt(b.encode(chunks[k])) for k in range(K)])
    put("ingest", "bfv", seconds=t, bytes=K * CT)
    act, t = timed(lambda: ah.encrypt(vals))
    put("ingest", "ashe", seconds=t, bytes=ROWS * 8)

    # ------------------------------------------- SUM, filtered SUM, GROUP BY
    sel = rng.random(ROWS) < SEL
    grp = rng.integers(0, GROUPS, ROWS).astype(np.int32)
    cases = {"sum": (np.zeros(ROWS, dtype=np.int32), 1),
             "filter": (np.where(sel, 0, -1).astype(np.int32), 1),
             "groupby": (grp, GROUPS)}
    for op, (gid, G) in cases.items():
        mask = gid >= 0
        truth = [int(vals[mask & (gid == g)].sum()) for g in range(G)]

        out, ts = timed(lambda: pa.groupsum(ppool, idx, gid, G))
        dec, tc = timed(lambda: [pa.decrypt_int(c) for c in out])
        assert dec == truth, (op, "paillier")
        put(op, "paillier", seconds=ts + tc, server=ts, client=tc, bytes=G * pa.ctlen)

        (sums, rg, rs, re_), ts = timed(lambda: ASHE.server_groupsum(act, ids, gid, G))
        idb = ASHE.idlist_bytes_groups(rg, rs, re_, G, ROWS)
        dec, tc = timed(lambda: ah.decrypt_groups(sums, rg, rs, re_, G))
        assert dec == truth, (op, "ashe")
        put(op, "ashe", seconds=ts + tc, server=ts, client=tc, bytes=G * 8 + idb, id_bytes=idb, runs=len(rs))

        gp = np.full(K * N, -1, dtype=np.int64)
        gp[:ROWS] = gid
        gp = gp.reshape(K, N)

        def bfv_server():
            accs = [b.zero() for _ in range(G)]
            for k in range(K):
                if G == 1:                                # one group: add, or mask once
                    selk = gp[k] >= 0
                    if selk[real[k]].all():
                        b.add_inplace(accs[0], cts[k])    # every row selected (padding is zero)
                    elif selk.any():
                        b.mul_plain_acc(accs[0], cts[k], b.mask_ntt(selk.astype(np.int64)))
                    continue
                for g in np.unique(gp[k][gp[k] >= 0]):
                    b.mul_plain_acc(accs[g], cts[k], b.mask_ntt((gp[k] == g).astype(np.int64)))
            return accs
        accs, ts = timed(bfv_server)
        dec, tc = timed(lambda: [int(b.decrypt_slots(a).astype(np.int64).sum()) for a in accs])
        assert dec == truth, (op, "bfv")
        put(op, "bfv", seconds=ts + tc, server=ts, client=tc, bytes=G * CT)

        if op == "groupby":
            def bfv_client():
                sums = np.zeros(G, dtype=np.int64)
                for k in range(K):
                    slots = b.decrypt_slots(cts[k]).astype(np.int64)
                    ok = gp[k] >= 0
                    sums += np.bincount(gp[k][ok], weights=slots[ok], minlength=G).astype(np.int64)
                return sums
            sums_c, tc = timed(bfv_client)
            assert list(map(int, sums_c)) == truth
            put(op, "bfv_client", seconds=tc, server=0.0, client=tc, bytes=K * CT)

    # ------------------------------------------------------------ row update
    put("update", "paillier", seconds=1 / p_enc, bytes=pa.ctlen)
    onehot = np.zeros(N, dtype=np.int64)
    onehot[17] = 5
    target = (cts[0][0].copy(), cts[0][1].copy())
    put("update", "bfv", seconds=1 / rate(lambda: b.add_inplace(target, b.encrypt(b.encode(onehot)))), bytes=CT)
    one = np.array([5])
    put("update", "ashe", seconds=1 / rate(lambda: ah.encrypt(one, first_id=ROWS + 1)), bytes=8,
        note="appends under a fresh id; the old id is tombstoned")

    # ------------------------------------------------------------ signatures
    s = L1["sig"]
    for scheme, name in (("classical", "ECDSA P-256"), ("pq", "ML-DSA-44")):
        put("sign", scheme, seconds=ROWS / s[name]["sign"], projected=True, bytes=ROWS * s[name]["sig"])
        put("verify", scheme, seconds=ROWS / s[name]["verify"], projected=True, bytes=ROWS * s[name]["sig"])

    # ---------------------------------------------------------- column chunk
    chunk = os.urandom(32 << 20)
    nonce = os.urandom(12)
    for scheme, bits in (("classical", 128), ("pq", 256)):
        g = AESGCM(AESGCM.generate_key(bit_length=bits))
        put("aes_chunk", scheme, seconds=1 / rate(lambda: g.encrypt(nonce, chunk, None), seconds=2.0),
            bytes=len(chunk) + 16, note=f"AES-{bits}-GCM")

    res["params"] = dict(paillier_bits=2048, bfv_n=N, bfv_log2_q=b.q.bit_length(), bfv_t_bits=b.t.bit_length(),
                         bfv_ciphertexts=K, bfv_ct_bytes=CT, paillier_ct_bytes=pa.ctlen,
                         keys=KEYS, days=DAYS, pk_rows=PK,
                         ashe_prf="AES-256", groups=GROUPS, selectivity=SEL, pool=POOL, value_max=VMAX)
    path = os.path.join(HERE, "results", "db_microbench.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1, default=float)
    for op, d in res["ops"].items():
        print(f"{op:10s}", "  ".join(f"{k}: {v['seconds']:.4g}s {v['bytes']:.3g}B" for k, v in d.items()))
    print("wrote", path)


if __name__ == "__main__":
    main()
