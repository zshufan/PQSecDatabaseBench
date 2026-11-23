#!/usr/bin/env python3
"""Analytic sizes: ciphertext sizes for additive encryption and the TPC-H estimate of Sec. IV-C.

Nothing here is measured: the numbers follow from parameter sizes and from
the TPC-H specification. The script exists so that every number quoted in the
paper can be regenerated and audited.

    python3 benchmarks/sizing_model.py
"""
import json
import os

# ---------------------------------------------------------------- aggregation
# Paillier with a 3072-bit modulus (128-bit classical security): a ciphertext
# is an element of Z_{n^2}, i.e. 6144 bits, and holds one value.
PAILLIER_CT_BYTES = 2 * 3072 // 8

# BFV/BGV at 128-bit security, largest ciphertext modulus allowed by the
# Homomorphic Encryption Standard (ePrint 2019/939, Table 1, ternary secret).
# A fresh ciphertext is two ring elements of N coefficients modulo q. Sizes are
# bit-packed lower bounds; libraries that store 64-bit RNS limbs use more
# (e.g. 3 limbs for 109 bits -> 1.76x; 5 limbs for 218 bits -> 1.47x).
HE_STANDARD = {4096: 109, 8192: 218, 16384: 438}


def bfv_ct_bytes(n, logq):
    return 2 * n * logq // 8


aggregation = dict(paillier_3072_bytes_per_value=PAILLIER_CT_BYTES, bfv=[])
for n, logq in HE_STANDARD.items():
    ct = bfv_ct_bytes(n, logq)
    aggregation["bfv"].append(dict(
        ring_dimension=n, log2_q=logq, ciphertext_bytes=ct, slots=n,
        bytes_per_value_full=round(ct / n, 2),
        # packed BFV is smaller than Paillier only if a ciphertext carries at
        # least this many live values:
        crossover_values_per_ciphertext=round(ct / PAILLIER_CT_BYTES, 1),
        single_value_blowup_vs_paillier=round(ct / PAILLIER_CT_BYTES, 1),
        full_packing_gain_vs_paillier=round(PAILLIER_CT_BYTES / (ct / n), 1),
    ))

# ------------------------------------------------------------------- TPC-H
# Illustrative horizon labelling: personally identifying columns are
# long-horizon, everything else (commercial facts) is short-horizon.
# Column widths per the TPC-H specification (average for variable-length).
LONG_HORIZON_COLUMNS = {
    "customer": dict(rows_per_sf=150_000, cols=dict(c_name=18, c_address=25, c_phone=15)),
    "supplier": dict(rows_per_sf=10_000, cols=dict(s_name=18, s_address=25, s_phone=15)),
}
RAW_BYTES_PER_SF = 1.0e9  # dbgen output is ~1 GB per unit of scale factor

long_bytes = sum(t["rows_per_sf"] * sum(t["cols"].values())
                 for t in LONG_HORIZON_COLUMNS.values())

# Queries that reference a long-horizon column, and how.
QUERIES_TOUCHING_LONG = {
    "Q2":  "s_name, s_address, s_phone in the output; ORDER BY s_name",
    "Q10": "c_name, c_address, c_phone in the output and GROUP BY",
    "Q15": "s_name, s_address, s_phone in the output",
    "Q18": "c_name in the output and GROUP BY",
    "Q20": "s_name, s_address in the output; ORDER BY s_name",
    "Q21": "s_name in the output, GROUP BY and ORDER BY",
    "Q22": "substring(c_phone) in a predicate and GROUP BY",
}

tpch = dict(
    labelling="PII columns of CUSTOMER and SUPPLIER are long-horizon; all else short-horizon",
    long_horizon_bytes_per_sf=long_bytes,
    long_horizon_fraction_of_raw_bytes=round(long_bytes / RAW_BYTES_PER_SF, 4),
    queries_total=22,
    queries_touching_long_horizon=len(QUERIES_TOUCHING_LONG),
    queries_computing_on_long_horizon=["Q22 (predicate)", "Q2/Q20/Q21 (ORDER BY)",
                                       "Q10/Q18/Q21 (GROUP BY, functionally determined by a key)"],
    detail=QUERIES_TOUCHING_LONG,
)

result = dict(aggregation=aggregation, tpch=tpch)
out = os.path.join(os.path.dirname(__file__), "results", "sizing_model.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
