#!/usr/bin/env python3
"""BFV with one value per ciphertext, at the operator level (Fig. 4) and the query level (Fig. 5).

Packing 8,192 values into one BFV ciphertext pays off only when values arrive
together. This script measures BFV without packing: every value is its own
ciphertext (a constant polynomial), sums are plain ciphertext additions with no
masks, and every result is one decryption.

Two parts:

1. Validation, run on the current machine. Over the 1,000,000-row column of
   db_microbench.py, SUM, the 1%-selective SUM and the 64-group GROUP BY are
   executed end to end over a pool of genuine ciphertexts reused by index (as
   the Paillier columns are) and checked against the plaintext answer. The
   measured time is compared with a model that multiplies operation counts by
   per-ciphertext unit costs measured in the same run.

2. Projections for the figures. The same model, with the unit costs of
   results/primitives.json (the run every other bar comes from
   on), gives the operator costs of Fig. 4 and, with the per-query counts in
   results/tpch_sf0.1.json, the query costs of Fig. 5. Ingest is projected in
   both, since a million 414 KB ciphertexts do not fit on disk here.

Writes benchmarks/results/unpacked_bfv.json.

    python3 benchmarks/unpacked_bfv_bench.py      # ~3 minutes
"""
import json
import os

import numpy as np

from pqdb import BFV, cpu_model, rate, timed

HERE = os.path.dirname(os.path.abspath(__file__))
ROWS, POOL, GROUPS, SEL, VMAX = 1_000_000, 64, 64, 0.01, 1000
R = lambda name: json.load(open(os.path.join(HERE, "results", name)))


def model(counts, u):
    """counts: encrypts, adds, decrypts, result_cts; u: seconds per operation; returns server, client, bytes."""
    server = counts["adds"] * u["add"]
    client = counts["encrypts"] * u["encrypt"] + counts["decrypts"] * u["decrypt"]
    return server, client


def main():
    rng = np.random.default_rng(11)
    b = BFV(8192, [40, 40, 40, 41, 41], 50, rng)
    CT = b.ciphertext_bytes()

    def enc1(v):
        m = np.zeros(b.n, dtype=np.uint64)
        m[0] = int(v) % b.t
        return b.encrypt(m)

    # unit costs on this machine
    one = enc1(7)
    acc = b.zero()
    here = dict(encrypt=1 / rate(lambda: enc1(7)), add=1 / rate(lambda: b.add_inplace(acc, one)),
                decrypt=1 / rate(lambda: b.decrypt(one)))

    # validation: full-scale execution over a pool of genuine ciphertexts
    pool_vals = rng.integers(0, VMAX, POOL)
    pool = [enc1(v) for v in pool_vals]
    idx = rng.integers(0, POOL, ROWS)
    vals = pool_vals[idx]
    sel = rng.random(ROWS) < SEL
    grp = rng.integers(0, GROUPS, ROWS)
    cases = {"sum": (np.zeros(ROWS, dtype=np.int64), 1),
             "filter": (np.where(sel, 0, -1), 1),
             "groupby": (grp, GROUPS)}
    validation = {}
    for op, (gid, G) in cases.items():
        rows = np.nonzero(gid >= 0)[0]
        truth = [int(vals[rows][gid[rows] == g].sum()) for g in range(G)]

        def server():
            accs = [b.zero() for _ in range(G)]
            for r in rows:
                b.add_inplace(accs[gid[r]], pool[idx[r]])
            return accs
        accs, ts = timed(server)
        dec, tc = timed(lambda: [int(b.decrypt(a)[0]) for a in accs])
        assert dec == truth, op
        ms, mc = model(dict(encrypts=0, adds=len(rows), decrypts=G), here)
        validation[op] = dict(adds=int(len(rows)), decrypts=G, measured=ts + tc, model=ms + mc,
                              measured_over_model=(ts + tc) / (ms + mc))

    # projections with the unit costs of the machine behind every other bar
    he = R("primitives.json")["additive_he"]["BFV-8192"]
    u = dict(encrypt=1 / he["encrypt"], add=1 / he["add"], decrypt=1 / he["decrypt"])
    ops = {}
    n_sel = int(round(ROWS * SEL))
    for op, counts, nbytes in (
            ("ingest", dict(encrypts=ROWS, adds=0, decrypts=0), ROWS * CT),
            ("sum", dict(encrypts=0, adds=ROWS, decrypts=1), CT),
            ("filter", dict(encrypts=0, adds=n_sel, decrypts=1), CT),
            ("groupby", dict(encrypts=0, adds=ROWS, decrypts=GROUPS), GROUPS * CT),
            ("update", dict(encrypts=1, adds=0, decrypts=0), CT)):
        s, c = model(counts, u)
        ops[op] = dict(seconds=s + c, server=s, client=c, bytes=nbytes, projected=True, **counts)

    tp = R("tpch_sf0.1.json")
    n_li, n_cols = tp["lineitem_rows"], len(tp["params"]["hom_columns"])
    queries = {}
    for q, d in tp["queries"].items():
        A, G, S = d["aggregates"], d["groups"], d["selected_rows"]
        keyb = d["cryptdb"]["shipped"] - G * A * tp["params"]["paillier_ct_bytes"]
        s, c = model(dict(encrypts=0, adds=S * A, decrypts=G * A), u)
        rel = d["cryptdb"]["rel_server"] + d["cryptdb"]["rel_client"]
        queries[q] = dict(hom_server=s, hom_client=c, rel=rel, total=s + c + rel,
                          shipped=G * A * CT + keyb, adds=S * A, decrypts=G * A, projected=True)
    ingest = dict(seconds=n_li * n_cols * u["encrypt"], storage=n_li * n_cols * CT, projected=True)

    out = dict(machine_validation=cpu_model(), unit_costs_here=here, validation=validation,
               unit_costs_projection=u, unit_source="results/primitives.json",
               ops=ops, tpch=dict(queries=queries, ingest=ingest), ct_bytes=CT,
               rows=ROWS, groups=GROUPS, selectivity=SEL, pool=POOL)
    path = os.path.join(HERE, "results", "unpacked_bfv.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1, default=float)
    for op, v in validation.items():
        print(f"validate {op:8s} measured {v['measured']:.3f}s  model {v['model']:.3f}s  "
              f"ratio {v['measured_over_model']:.3f}")
    for op, v in ops.items():
        print(f"op {op:8s} {v['seconds']:.4g}s {v['bytes']:.3g}B")
    for q, v in queries.items():
        print(f"{q:4s} total {v['total']:.3f}s shipped {v['shipped']:.3g}B")
    print("tpch ingest", ingest)
    print("wrote", path)


if __name__ == "__main__":
    main()
