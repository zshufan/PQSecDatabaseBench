#!/usr/bin/env python3
"""Query level: TPC-H end to end over CryptDB-encrypted columns, non-PQ vs PQ (Fig. 5, top).

Queries Q1, Q3, Q5, Q6 and Q10 run in the split-execution style of CryptDB and
MONOMI: the server evaluates filters, joins and grouping over the Eq (DET), Ord
(OPE) and Join onions and aggregates the Add (HOM) onion; the client (CryptDB's
proxy, which holds the keys) encrypts the query constants, decrypts the
aggregates and the group keys it needs, and computes AVG, ORDER BY and LIMIT.
Aggregated expressions such as l_extendedprice*(1-l_discount) are precomputed
at ingest, as MONOMI does.

Two worlds, built from the same data:

  CryptDB (non-PQ)  DET, RND and the OPE PRF on AES-128; JOIN-ADJ on P-256,
                    adjusted once per join column before the first join;
                    HOM = Paillier-2048
  post-quantum      the same onions on AES-256; join columns share a DET key
                    from ingest (no PQ adjustable join exists; this reveals the
                    columns' equalities before any join is issued); HOM =
                    BFV-8192 with server- or client-side grouping, or ASHE with
                    AES-256

Per query and configuration: the relational part on encrypted data (server
scans and joins, client constants and group-key decryption), the HOM part
(server aggregation, client decryption), bytes shipped to the client, and the
storage of every onion. The encrypted plan is checked row by row against the
plaintext plan of tpch_gen.queries(), and every aggregate against its plaintext
answer. Paillier columns reuse a pool of 4,096 genuine ciphertexts, so Paillier
ingest is projected from the measured per-value rate; server-side BFV grouping
with more than 64 result ciphertexts (Q3, Q10) is projected from measured unit
costs, since no planner would choose it.

    python3 benchmarks/tpch_bench.py --sf 0.1     # ~6 minutes
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from cryptdb import DET, OPE, RND, JoinAdj
from pqdb import ASHE, BFV, Paillier, cpu_model, rate, timed
from tpch_gen import REGIONS, SEGMENTS, START, generate, queries

HERE = os.path.dirname(os.path.abspath(__file__))
POOL = 4096
SERVER_SIDE_MAX_RESULTS = 64

DAY = lambda s: int((np.datetime64(s) - START) / np.timedelta64(1, "D"))
days = lambda col: ((col.values - START) / np.timedelta64(1, "D")).astype(np.int64)

# (table, column) -> plaintext values used for each onion
EQ_COLS = {("lineitem", "l_returnflag"), ("lineitem", "l_linestatus"), ("customer", "c_mktsegment"),
           ("region", "r_name"), ("nation", "n_name"), ("orders", "o_shippriority")}
ORD_COLS = {("lineitem", "l_shipdate"), ("orders", "o_orderdate"), ("lineitem", "l_discount"),
            ("lineitem", "l_quantity")}
JOIN_GROUPS = [  # first column is the anchor whose key the others are adjusted to
    [("customer", "c_custkey"), ("orders", "o_custkey")],
    [("orders", "o_orderkey"), ("lineitem", "l_orderkey")],
    [("supplier", "s_suppkey"), ("lineitem", "l_suppkey")],
    [("nation", "n_nationkey"), ("customer", "c_nationkey"), ("supplier", "s_nationkey")],
    [("region", "r_regionkey"), ("nation", "n_regionkey")]]
GROUP_KEYS = {"Q1": 2, "Q3": 3, "Q5": 1, "Q6": 0, "Q10": 1}   # result key columns shipped per group
TOP = {"Q3": 10, "Q10": 20}


def plain(db, t, c):
    v = db[t][c]
    if c in ("l_shipdate", "o_orderdate"):
        return days(v)
    if c == "c_mktsegment":
        return np.array(SEGMENTS)[v.values]
    return v.values


def exact_groupsum(gid, vals, G):
    """Exact int64 per-group sums (float64 bincount is inexact above 2^53)."""
    v = vals.astype(np.int64)
    lo = np.bincount(gid, weights=v & ((1 << 26) - 1), minlength=G).astype(np.int64)
    hi = np.bincount(gid, weights=v >> 26, minlength=G).astype(np.int64)
    return lo + (hi << 26)


class World:
    """Encrypted copies of the relational columns, as the server sees them."""

    def __init__(self, db, bits, cryptdb):
        self.bits, self.cryptdb = bits, cryptdb
        self.det, self.ope, self.srv = {}, {}, {}
        self.ingest, self.one_time, self.storage = 0.0, 0.0, 0
        rnd = RND(bits)
        for t, c in sorted(EQ_COLS):
            d = DET(bits)
            v = plain(db, t, c)
            (ct, _), dt = timed(lambda: (d.enc(v), None))
            (iv, outer), rt = timed(lambda: rnd.enc(ct))
            _, pt = timed(lambda: rnd.dec(iv, outer))
            self.det[c], self.srv[c] = d, DET.server_key(ct)
            self.ingest += dt + rt
            self.one_time += pt
            self.storage += len(v) * RND.bytes_per_value
        for t, c in sorted(ORD_COLS):
            o = OPE(bits)
            v = plain(db, t, c)
            (oc, _), et = timed(lambda: o.encrypt_many(v))
            (iv, outer), rt = timed(lambda: rnd.enc(np.c_[oc, np.zeros_like(oc)]))
            _, pt = timed(lambda: rnd.dec(iv, outer))
            o.cache.clear()                            # query constants are encrypted afresh
            self.ope[c], self.srv[c] = o, oc
            self.ingest += et + rt
            self.one_time += pt
            self.storage += len(v) * RND.bytes_per_value
        self.adjust = 0.0
        ja = JoinAdj(128) if cryptdb else None
        for group in JOIN_GROUPS:
            if cryptdb:
                k_anchor = ja.new_key()
                for i, (t, c) in enumerate(group):
                    v = plain(db, t, c)
                    u, inv = np.unique(v, return_inverse=True)
                    k = k_anchor if i == 0 else ja.new_key()
                    pts, et = timed(lambda: ja.encrypt_distinct(k, u))
                    self.ingest += et
                    if i:
                        pts, at = timed(lambda: ja.adjust(pts, k, k_anchor))
                        self.adjust += at
                    self.srv[c] = JoinAdj.server_key(pts)[inv]
                    self.storage += len(v) * 80        # RND(DET(JOIN-ADJ || DET)): 49 bytes padded to 64, plus IV
            else:
                shared = DET(bits)
                for t, c in group:
                    v = plain(db, t, c)
                    ct, et = timed(lambda: shared.enc(v))
                    self.ingest += et
                    self.srv[c] = DET.server_key(ct)
                    self.storage += len(v) * RND.bytes_per_value
            for t, c in group:
                self.det.setdefault(c, None)
        self.one_time += self.adjust
        self.n = {t: len(db[t]) for t in db}

    # ------------------------------------------------------------ queries
    def const_det(self, col, value):
        return DET.server_key(self.det[col].enc(np.array([value])))[0]

    def const_ope(self, col, value):
        return np.uint64(self.ope[col].encrypt(value))

    def run(self, name):
        """Server plan over encrypted columns. Returns (gid over lineitem rows, G, client-constant seconds)."""
        S = self.srv
        n_li = self.n["lineitem"]
        tc = 0.0
        if name == "Q1":
            c, tc = timed(lambda: self.const_ope("l_shipdate", DAY("1998-09-02")))
            m = S["l_shipdate"] <= c
            keys = [S["l_returnflag"][m], S["l_linestatus"][m]]
        elif name == "Q3":
            (seg, d), tc = timed(lambda: (self.const_det("c_mktsegment", "BUILDING"),
                                          (self.const_ope("o_orderdate", DAY("1995-03-15")),
                                           self.const_ope("l_shipdate", DAY("1995-03-15")))))
            cust = S["c_custkey"][S["c_mktsegment"] == seg]
            om = (S["o_orderdate"] < d[0]) & np.isin(S["o_custkey"], cust)
            m = (S["l_shipdate"] > d[1]) & np.isin(S["l_orderkey"], S["o_orderkey"][om])
            keys = [S["l_orderkey"][m]]
        elif name == "Q5":
            (asia, lo, hi), tc = timed(lambda: (self.const_det("r_name", "ASIA"),
                                                self.const_ope("o_orderdate", DAY("1994-01-01")),
                                                self.const_ope("o_orderdate", DAY("1995-01-01"))))
            regs = S["r_regionkey"][S["r_name"] == asia]
            nats = S["n_nationkey"][np.isin(S["n_regionkey"], regs)]
            cm = np.isin(S["c_nationkey"], nats)
            cust = pd.DataFrame({"ck": S["c_custkey"][cm], "cn": S["c_nationkey"][cm]})
            om = (S["o_orderdate"] >= lo) & (S["o_orderdate"] < hi)
            ords = pd.DataFrame({"ok": S["o_orderkey"][om], "ck": S["o_custkey"][om]}).merge(cust, on="ck")
            li = pd.DataFrame({"ok": S["l_orderkey"], "sk": S["l_suppkey"], "row": np.arange(n_li)})
            li = li.merge(ords[["ok", "cn"]], on="ok").merge(
                pd.DataFrame({"sk": S["s_suppkey"], "sn": S["s_nationkey"]}), on="sk")
            li = li[li.cn == li.sn]
            m = np.zeros(n_li, dtype=bool)
            m[li.row.values] = True
            sn = np.zeros(n_li, dtype=np.uint64)
            sn[li.row.values] = li.sn.values
            keys = [sn[m]]
        elif name == "Q6":
            c, tc = timed(lambda: [self.const_ope("l_shipdate", DAY("1994-01-01")),
                                   self.const_ope("l_shipdate", DAY("1995-01-01")),
                                   self.const_ope("l_discount", 5), self.const_ope("l_discount", 7),
                                   self.const_ope("l_quantity", 24)])
            m = ((S["l_shipdate"] >= c[0]) & (S["l_shipdate"] < c[1]) & (S["l_discount"] >= c[2]) &
                 (S["l_discount"] <= c[3]) & (S["l_quantity"] < c[4]))
            keys = [np.zeros(int(m.sum()), dtype=np.uint64)]
        elif name == "Q10":
            (lo, hi, r), tc = timed(lambda: (self.const_ope("o_orderdate", DAY("1993-10-01")),
                                             self.const_ope("o_orderdate", DAY("1994-01-01")),
                                             self.const_det("l_returnflag", "R")))
            om = (S["o_orderdate"] >= lo) & (S["o_orderdate"] < hi)
            ords = pd.DataFrame({"ok": S["o_orderkey"][om], "ck": S["o_custkey"][om]})
            li = pd.DataFrame({"ok": S["l_orderkey"], "row": np.arange(n_li)})[S["l_returnflag"] == r]
            li = li.merge(ords, on="ok")
            m = np.zeros(n_li, dtype=bool)
            m[li.row.values] = True
            ck = np.zeros(n_li, dtype=np.uint64)
            ck[li.row.values] = li.ck.values
            keys = [ck[m]]
        gid = np.full(n_li, -1, dtype=np.int32)
        idx = pd.MultiIndex.from_arrays(keys) if len(keys) > 1 else pd.Index(keys[0])
        codes, uniq = pd.factorize(idx)
        gid[np.flatnonzero(m)] = codes
        return gid, len(uniq), tc

    def client_keys(self, name, G):
        """Decrypt the DET group keys the client needs (all groups, or the top k after ranking)."""
        k = min(G, TOP.get(name, G)) * GROUP_KEYS[name]
        d = DET(self.bits)
        blocks = d.enc(np.arange(k))
        return timed(lambda: d.dec(blocks))[1]


def same_partition(g1, g2):
    """True if two group-id vectors select the same rows and group them the same way."""
    if not np.array_equal(g1 >= 0, g2 >= 0):
        return False
    s = g1 >= 0
    pairs = pd.DataFrame({"a": g1[s], "b": g2[s]}).drop_duplicates()
    return pairs.a.is_unique and pairs.b.is_unique


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sf", type=float, default=0.1)
    args = ap.parse_args()
    rng = np.random.default_rng(11)

    db = generate(args.sf)
    Qp = queries(db)                          # plaintext plans, for checking
    L = db["lineitem"]
    n_li = len(L)
    cols = sorted({c for q in Qp.values() for c in q["cols"]})

    worlds = {}
    for w, (bits, cdb) in {"cryptdb": (128, True), "pq": (256, False)}.items():
        worlds[w], t = timed(lambda: World(db, bits, cdb))
        print(w, "built in", round(t, 1), "s; ingest", round(worlds[w].ingest, 2), "s; one-time",
              round(worlds[w].one_time, 2), "s (JOIN-ADJ adjust", round(worlds[w].adjust, 2), "s)")

    # HOM back ends
    pa = Paillier(2048)
    pool_vals = rng.integers(0, 1 << 20, POOL)
    pool = pa.make_pool(pool_vals)
    pidx = {c: rng.integers(0, POOL, n_li).astype(np.int32) for c in cols}
    p_enc = rate(lambda: pa.encrypt(123))

    ah = {c: ASHE(256) for c in cols}
    ids = np.arange(1, n_li + 1, dtype=np.int64)
    act, ashe_ingest = {}, 0.0
    for c in cols:
        act[c], t = timed(lambda: ah[c].encrypt(L[c].values))
        ashe_ingest += t

    b = BFV(8192, [40, 40, 40, 41, 41], 50, rng)
    N, CT = b.n, b.ciphertext_bytes()
    K = -(-n_li // N)
    data, cts, bfv_ingest = {}, {}, 0.0
    for c in cols:
        v = np.zeros(K * N, dtype=np.int64)
        v[:n_li] = L[c].values
        assert v.max() < b.t
        data[c] = v.reshape(K, N)
        cts[c], t = timed(lambda: [b.encrypt(b.encode(data[c][k])) for k in range(K)])
        bfv_ingest += t
    M = b.mask_ntt(rng.integers(0, 2, N))
    acc = b.zero()
    t_mask = 1 / rate(lambda: b.mask_ntt(rng.integers(0, 2, N)))
    t_mul = 1 / rate(lambda: b.mul_plain_acc(acc, cts[cols[0]][0], M))
    t_dec = 1 / rate(lambda: b.decrypt_slots(cts[cols[0]][0]))

    nc = len(cols)
    out = dict(sf=args.sf, lineitem_rows=n_li, machine=dict(cpu=cpu_model()),
               params=dict(paillier_bits=2048, paillier_ct_bytes=pa.ctlen, bfv_n=N, bfv_log2_q=b.q.bit_length(),
                           bfv_t_bits=b.t.bit_length(), bfv_ct_bytes=CT, bfv_ciphertexts_per_column=K,
                           hom_columns=cols),
               worlds={w: dict(ingest_relational=W.ingest, one_time=W.one_time, join_adjust=W.adjust,
                               storage_relational=W.storage) for w, W in worlds.items()},
               hom=dict(paillier=dict(ingest=nc * n_li / p_enc, ingest_projected=True, storage=nc * n_li * pa.ctlen),
                        bfv=dict(ingest=bfv_ingest, storage=nc * K * CT),
                        ashe=dict(ingest=ashe_ingest, storage=nc * n_li * 8)),
               unit_costs=dict(bfv_mask_prepare=t_mask, bfv_mul_plain=t_mul, bfv_decrypt=t_dec,
                               paillier_encrypt=1 / p_enc),
               queries={})

    for name in ("Q1", "Q3", "Q5", "Q6", "Q10"):
        qp = Qp[name]
        qc, A = qp["cols"], len(qp["cols"])
        r = dict(aggregates=A)
        rel = {}
        for w, W in worlds.items():
            (gid, G, tconst), ts = timed(lambda: W.run(name))
            assert same_partition(gid, qp["gid"]), (name, w)
            assert G == qp["G"], (name, w, G, qp["G"])
            tk = W.client_keys(name, G)
            rel[w] = dict(server=ts - tconst, client=tconst + tk, gid=gid)
        gid = rel["cryptdb"]["gid"]
        G = qp["G"]
        sel = gid >= 0
        r.update(groups=G, selected_rows=int(sel.sum()))
        keyb = min(G, TOP.get(name, G)) * GROUP_KEYS[name] * DET.bytes_per_value
        truth = {c: exact_groupsum(gid[sel], L[c].values[sel], G) for c in qc}

        # CryptDB HOM: Paillier
        res, ts = timed(lambda: {c: pa.groupsum(pool, pidx[c], gid, G) for c in qc})
        dec, tc = timed(lambda: {c: [pa.decrypt_int(x) for x in res[c]] for c in qc})
        for c in qc:
            assert dec[c] == list(map(int, exact_groupsum(gid[sel], pool_vals[pidx[c][sel]], G))), (name, c)
        r["cryptdb"] = dict(hom_server=ts, hom_client=tc, shipped=G * A * pa.ctlen + keyb)

        gp = np.full(K * N, -1, dtype=np.int64)
        gp[:n_li] = gid
        gp = gp.reshape(K, N)
        pairs = sum(len(np.unique(gp[k][gp[k] >= 0])) for k in range(K))
        touched = [k for k in range(K) if (gp[k] >= 0).any()]

        # PQ HOM: BFV, server-side grouping
        if G * A <= SERVER_SIDE_MAX_RESULTS:
            def server_side():
                accs = {c: [b.zero() for _ in range(G)] for c in qc}
                for k in touched:
                    for g in np.unique(gp[k][gp[k] >= 0]):
                        Mk = b.mask_ntt((gp[k] == g).astype(np.int64))
                        for c in qc:
                            b.mul_plain_acc(accs[c][g], cts[c][k], Mk)
                return accs
            accs, ts = timed(server_side)
            dec, tc = timed(lambda: {c: [int(b.decrypt_slots(a).astype(np.int64).sum()) for a in accs[c]]
                                     for c in qc})
            for c in qc:
                assert dec[c] == list(map(int, truth[c])), (name, c)
            r["pq_bfv"] = dict(hom_server=ts, hom_client=tc, shipped=G * A * CT + keyb, projected=False)
        else:
            r["pq_bfv"] = dict(hom_server=pairs * (t_mask + A * t_mul), hom_client=G * A * t_dec,
                               shipped=G * A * CT + keyb, projected=True)
        r["pq_bfv"]["mask_ops"] = pairs

        # PQ HOM: BFV, client-side grouping (ships every ciphertext holding a selected row)
        def client_side():
            sums = {c: np.zeros(G, dtype=np.int64) for c in qc}
            for c in qc:
                for k in touched:
                    slots = b.decrypt_slots(cts[c][k]).astype(np.int64)
                    ok = gp[k] >= 0
                    sums[c] += exact_groupsum(gp[k][ok], slots[ok], G)
            return sums
        sums, tc = timed(client_side)
        for c in qc:
            assert np.array_equal(sums[c], truth[c]), (name, c)
        r["pq_bfv_client"] = dict(hom_server=0.0, hom_client=tc, shipped=len(touched) * A * CT + keyb)

        # PQ HOM: ASHE
        (sums, rg, rs, re_), ts = timed(lambda: ASHE.server_groupsum([act[c] for c in qc], ids, gid, G))
        dec, tc = timed(lambda: {c: ah[c].decrypt_groups(s_, rg, rs, re_, G) for c, s_ in zip(qc, sums)})
        for c in qc:
            assert dec[c] == [int(x) % (1 << 64) for x in truth[c]], (name, c)
        idb = ASHE.idlist_bytes_groups(rg, rs, re_, G, n_li)
        r["pq_ashe"] = dict(hom_server=ts, hom_client=tc, shipped=G * A * 8 + idb + keyb, id_bytes=idb)

        for cfg in ("cryptdb", "pq_bfv", "pq_bfv_client", "pq_ashe"):
            w = "cryptdb" if cfg == "cryptdb" else "pq"
            d = r[cfg]
            d.update(rel_server=rel[w]["server"], rel_client=rel[w]["client"])
            d["total"] = d["rel_server"] + d["rel_client"] + d["hom_server"] + d["hom_client"]
        out["queries"][name] = r
        print(name, json.dumps({k: (round(v["total"], 4), v["shipped"]) if isinstance(v, dict) else v
                                for k, v in r.items()}))

    path = os.path.join(HERE, "results", f"tpch_sf{args.sf:g}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("wrote", path)


if __name__ == "__main__":
    main()
