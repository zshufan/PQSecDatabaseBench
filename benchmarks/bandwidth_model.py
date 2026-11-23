#!/usr/bin/env python3
"""TPC-H end to end with the transfer to the client included (Fig. 5, bottom).

The measured runs of tpch_bench.py and unpacked_bfv_bench.py report compute
time (server and client) and the bytes shipped to the client separately. Here
the two are combined as

    time at bandwidth B = compute + 8 * shipped / B

for B = 10 Gbps, 1 Gbps, 100 Mbps, and 10 Mbps, ignoring round trips and
protocol overhead (both favour the plans that ship more). For every
post-quantum plan and query, the break-even bandwidth is the bandwidth below
which the plan becomes slower than CryptDB: (8 * (shipped_pq - shipped_cdb)) /
(compute_cdb - compute_pq), or "always slower" / "never slower".

    python3 benchmarks/bandwidth_model.py      # instant

Writes benchmarks/results/tpch_bandwidth.json.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda name: json.load(open(os.path.join(HERE, "results", name)))
BANDWIDTHS = {"10 Gbps": 10e9, "1 Gbps": 1e9, "100 Mbps": 1e8, "10 Mbps": 1e7}
PLANS = ["cryptdb", "pq_bfv_one", "pq_bfv", "pq_bfv_client", "pq_ashe"]


def break_even(c, p):
    """Bandwidth (bit/s) below which plan p is slower than CryptDB c."""
    dt, db = c["total"] - p["total"], 8 * (p["shipped"] - c["shipped"])
    if dt <= 0:
        return "always slower" if db >= 0 else 8 * (c["shipped"] - p["shipped"]) / (p["total"] - c["total"])
    return db / dt if db > 0 else "never slower"


def main():
    t = R("tpch_sf0.1.json")["queries"]
    for q, v in R("unpacked_bfv.json")["tpch"]["queries"].items():
        t[q]["pq_bfv_one"] = v
    out = dict(model="compute + 8 * shipped / bandwidth; round trips ignored", bandwidths=BANDWIDTHS, queries={})
    for q, d in t.items():
        row = {}
        for plan in PLANS:
            p = d[plan]
            row[plan] = dict(compute=p["total"], shipped=p["shipped"],
                             at={b: p["total"] + 8 * p["shipped"] / bw for b, bw in BANDWIDTHS.items()})
            if plan != "cryptdb":
                be = break_even(d["cryptdb"], p)
                # a number below which the plan is slower; for a plan slower at compute that ships less,
                # the number is the bandwidth *below* which it becomes faster
                row[plan]["break_even_bps"] = be
                row[plan]["faster_at_compute"] = p["total"] < d["cryptdb"]["total"]
        out["queries"][q] = row
    path = os.path.join(HERE, "results", "tpch_bandwidth.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    for q, row in out["queries"].items():
        print(q, "  ".join(f"{p}: {r['at']['100 Mbps']:.2f}s" for p, r in row.items()))
        for p, r in row.items():
            if p != "cryptdb":
                be = r["break_even_bps"]
                print(f"    {p:14s} faster at compute: {r['faster_at_compute']!s:5s} break-even: "
                      + (f"{be / 1e6:.3g} Mbps" if isinstance(be, float) else be))
    print("wrote", path)


if __name__ == "__main__":
    main()
