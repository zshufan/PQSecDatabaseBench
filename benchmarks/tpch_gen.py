"""TPC-H data for the columns used by Q1, Q3, Q5, Q6 and Q10.

The official dbgen is not available in the benchmark environment, so this
generator follows the TPC-H specification (v3, Sec. 4.2.3) for the columns these
queries touch: table cardinalities, key ranges, the customer/order skew (no
orders for custkeys divisible by 3), 1-7 lineitems per order, date arithmetic,
the return-flag and line-status rules, P_RETAILPRICE, the supplier choice for a
part, and the uniform distributions of quantity, discount and tax. Text columns
and comments are omitted. Lineitems are emitted in order-key order, as dbgen does.
"""
import numpy as np
import pandas as pd

START = np.datetime64("1992-01-01")
CURRENT = np.datetime64("1995-06-17")
END = np.datetime64("1998-12-31")

NATIONS = [  # (nationkey, name, regionkey)
    (0, "ALGERIA", 0), (1, "ARGENTINA", 1), (2, "BRAZIL", 1), (3, "CANADA", 1), (4, "EGYPT", 4),
    (5, "ETHIOPIA", 0), (6, "FRANCE", 3), (7, "GERMANY", 3), (8, "INDIA", 2), (9, "INDONESIA", 2),
    (10, "IRAN", 4), (11, "IRAQ", 4), (12, "JAPAN", 2), (13, "JORDAN", 4), (14, "KENYA", 0),
    (15, "MOROCCO", 0), (16, "MOZAMBIQUE", 0), (17, "PERU", 1), (18, "CHINA", 2), (19, "ROMANIA", 3),
    (20, "SAUDI ARABIA", 4), (21, "VIETNAM", 2), (22, "RUSSIA", 3), (23, "UNITED KINGDOM", 3),
    (24, "UNITED STATES", 1)]
REGIONS = ["AFRICA", "AMERICA", "ASIA", "EUROPE", "MIDDLE EAST"]
SEGMENTS = ["AUTOMOBILE", "BUILDING", "FURNITURE", "HOUSEHOLD", "MACHINERY"]


def generate(sf=0.1, seed=42):
    rng = np.random.default_rng(seed)
    n_cust, n_supp, n_part, n_ord = int(sf * 150_000), int(sf * 10_000), int(sf * 200_000), int(sf * 1_500_000)

    nation = pd.DataFrame(NATIONS, columns=["n_nationkey", "n_name", "n_regionkey"])
    region = pd.DataFrame({"r_regionkey": range(5), "r_name": REGIONS})
    supplier = pd.DataFrame({"s_suppkey": np.arange(1, n_supp + 1),
                             "s_nationkey": rng.integers(0, 25, n_supp)})
    customer = pd.DataFrame({"c_custkey": np.arange(1, n_cust + 1),
                             "c_nationkey": rng.integers(0, 25, n_cust),
                             "c_mktsegment": rng.integers(0, 5, n_cust)})   # index into SEGMENTS

    eligible = np.arange(1, n_cust + 1)
    eligible = eligible[eligible % 3 != 0]
    span = int((END - np.timedelta64(151, "D") - START) / np.timedelta64(1, "D"))
    orders = pd.DataFrame({"o_orderkey": np.arange(1, n_ord + 1),
                           "o_custkey": rng.choice(eligible, n_ord),
                           "o_orderdate": START + rng.integers(0, span + 1, n_ord).astype("timedelta64[D]"),
                           "o_shippriority": 0})

    per = rng.integers(1, 8, n_ord)
    okey = np.repeat(orders.o_orderkey.values, per)
    odate = np.repeat(orders.o_orderdate.values, per)
    n_li = len(okey)
    partkey = rng.integers(1, n_part + 1, n_li)
    retail_cents = 90000 + (partkey // 10) % 20001 + 100 * (partkey % 1000)       # P_RETAILPRICE * 100
    S = n_supp
    i = rng.integers(0, 4, n_li)
    suppkey = (partkey + i * (S // 4 + (partkey - 1) // S)) % S + 1
    qty = rng.integers(1, 51, n_li)
    disc = rng.integers(0, 11, n_li)          # hundredths
    tax = rng.integers(0, 9, n_li)            # hundredths
    ship = odate + rng.integers(1, 122, n_li).astype("timedelta64[D]")
    receipt = ship + rng.integers(1, 31, n_li).astype("timedelta64[D]")
    rflag = np.where(receipt <= CURRENT, np.where(rng.integers(0, 2, n_li) == 0, "R", "A"), "N")
    lstatus = np.where(ship > CURRENT, "O", "F")
    extprice = qty * retail_cents                                                   # cents
    lineitem = pd.DataFrame({
        "l_orderkey": okey, "l_suppkey": suppkey, "l_quantity": qty, "l_extendedprice": extprice,
        "l_discount": disc, "l_tax": tax, "l_shipdate": ship, "l_returnflag": rflag, "l_linestatus": lstatus,
    })
    # integer encodings of the aggregated expressions (exact; scale noted per column)
    lineitem["disc_price"] = extprice * (100 - disc)          # cents x 100
    lineitem["charge"] = lineitem["disc_price"] * (100 + tax)  # cents x 10^4
    lineitem["price_x_disc"] = extprice * disc               # cents x 100
    return dict(region=region, nation=nation, supplier=supplier, customer=customer,
                orders=orders, lineitem=lineitem)


def _gid(mask, keys):
    """Group ids over lineitem rows: -1 if not selected, else dense id of the key tuple."""
    gid = np.full(len(mask), -1, dtype=np.int32)
    if mask.any():
        codes, uniq = pd.factorize(pd.MultiIndex.from_arrays([k[mask] for k in keys]) if len(keys) > 1
                                   else pd.Index(keys[0][mask]))
        gid[mask] = codes
        return gid, len(uniq)
    return gid, 0


def queries(db):
    """For each query: aggregated lineitem columns, group id per lineitem row, number of groups.
    Filters, joins and grouping run in plaintext here; in the encrypted system the server runs
    the same plan over deterministically and order-revealingly encrypted columns."""
    L, O, C, S, N, R = (db[k] for k in ("lineitem", "orders", "customer", "supplier", "nation", "region"))
    d = np.datetime64
    out = {}

    m = (L.l_shipdate <= d("1998-09-02")).values
    gid, G = _gid(m, [L.l_returnflag.values, L.l_linestatus.values])
    out["Q1"] = dict(cols=["l_quantity", "l_extendedprice", "l_discount", "disc_price", "charge"], gid=gid, G=G)

    o = O[O.o_orderdate < d("1995-03-15")].merge(C[C.c_mktsegment == SEGMENTS.index("BUILDING")],
                                                  left_on="o_custkey", right_on="c_custkey")
    m = (L.l_orderkey.isin(o.o_orderkey) & (L.l_shipdate > d("1995-03-15"))).values
    gid, G = _gid(m, [L.l_orderkey.values])
    out["Q3"] = dict(cols=["disc_price"], gid=gid, G=G)

    asia = N[N.n_regionkey == REGIONS.index("ASIA")].n_nationkey
    o = O[(O.o_orderdate >= d("1994-01-01")) & (O.o_orderdate < d("1995-01-01"))]
    oc = o.merge(C[C.c_nationkey.isin(asia)], left_on="o_custkey", right_on="c_custkey")[["o_orderkey", "c_nationkey"]]
    li = L.reset_index().merge(oc, left_on="l_orderkey", right_on="o_orderkey") \
          .merge(S, left_on="l_suppkey", right_on="s_suppkey")
    li = li[li.c_nationkey == li.s_nationkey]
    m = np.zeros(len(L), dtype=bool)
    m[li["index"].values] = True
    nat = np.full(len(L), -1)
    nat[li["index"].values] = li.s_nationkey.values
    gid, G = _gid(m, [nat])
    out["Q5"] = dict(cols=["disc_price"], gid=gid, G=G)

    m = ((L.l_shipdate >= d("1994-01-01")) & (L.l_shipdate < d("1995-01-01")) &
         (L.l_discount >= 5) & (L.l_discount <= 7) & (L.l_quantity < 24)).values
    gid, G = _gid(m, [np.zeros(len(L), dtype=np.int8)])
    out["Q6"] = dict(cols=["price_x_disc"], gid=gid, G=G)

    o = O[(O.o_orderdate >= d("1993-10-01")) & (O.o_orderdate < d("1994-01-01"))]
    cust = L[["l_orderkey"]].merge(o[["o_orderkey", "o_custkey"]], left_on="l_orderkey",
                                   right_on="o_orderkey", how="left").o_custkey.values
    m = (~np.isnan(cust)) & (L.l_returnflag == "R").values
    gid, G = _gid(m, [np.nan_to_num(cust).astype(np.int64)])
    out["Q10"] = dict(cols=["disc_price"], gid=gid, G=G)
    return out


if __name__ == "__main__":
    db = generate(0.1)
    print({k: len(v) for k, v in db.items()})
    for name, q in queries(db).items():
        print(name, "groups", q["G"], "selected rows", int((q["gid"] >= 0).sum()))
