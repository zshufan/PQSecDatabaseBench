#!/usr/bin/env python3
"""Turn the benchmark results into the LaTeX the paper \\inputs:

  figures/fig_primitives.tex     Fig. 2, primitive cost ratios (primitives.json)
  figures/fig_scaling.tex        Fig. 3, raising classical parameters, key unwrap and signing (primitives.json)
  figures/fig_micro.tex          Fig. 4, operator level, three panels for a figure*  (db_microbench.json)
  figures/fig_tpch.tex           Fig. 5, query level, TPC-H bars      (tpch_sf0.1.json)
  figures/fig_pk.tex             Fig. 6, use-case level, three panels (pk_settings.json, key_hierarchy.json)

    python3 benchmarks/make_figures.py
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "figures")
R = lambda name: json.load(open(os.path.join(HERE, "results", name)))

LW = ", line width=0.2pt"
PAILLIER = "fill=black!60, draw=black!80" + LW          # non-PQ
ELGAMAL = "fill=black!20, draw=black!60" + LW           # non-PQ
BFV = "fill=blue!65!black, draw=blue!30!black" + LW     # PQ, one value per ciphertext (or a PQ default)
BFV_PACKED = "fill=blue!10, draw=blue!60!black, densely dashed" + LW              # PQ, 8,192 values packed
BFV_CLIENT = "fill=blue!32, draw=blue!50!black, densely dashed" + LW              # PQ, packed, client-side grouping
LIGHT_BLUE = "fill=blue!22, draw=blue!50!black" + LW    # PQ, a second post-quantum series (ML-DSA-65)
ASHE = "fill=teal!55, draw=teal!40!black" + LW          # PQ
CLASSICAL, PQ = PAILLIER, BFV


def k(x):
    """Compact rate: 51.4k, 4.3M, 82."""
    if x is None:
        return "--"
    if x >= 1e9:
        return f"{x / 1e9:.1f}G"
    if x >= 1e6:
        return f"{x / 1e6:.1f}M" if x < 1e8 else f"{x / 1e6:.0f}M"
    if x >= 1e4:
        return f"{x / 1e3:.0f}k"
    if x >= 1e3:
        s = f"{x / 1e3:.1f}"
        return ("10" if s == "10.0" else s) + "k"
    return f"{x:.0f}"


def b(x):
    return f"{int(x):,}".replace(",", "{,}")


def arrow(a, c):
    return f"${a} \\rightarrow {c}$"

def coords(pairs):
    return " ".join(f"({x},{y:.6g})" for x, y in pairs if y is not None and y > 0)


def bounds(series):
    import math
    ys = [y for _, pts, _ in series for _, y in pts if y]
    return 10 ** math.floor(math.log10(min(ys))), 10 ** math.ceil(math.log10(max(ys)))


def axis(ylabel, xcoords, xlabels, series, width=r"\columnwidth", height="3.5cm", bar="3pt", legend=None,
         legend_cols=-1, axis_only=False, extra=None, opts="", more_legend=(), ybounds=None):
    """extra: (plot command, legend name, points) drawn after the bars, e.g. a marker series.
    more_legend: (legend image options, name) pairs appended to the legend without a plot."""
    ymin, ymax = ybounds or bounds(series + ([(None, extra[2], None)] if extra else []))
    s = [r"\begin{axis}[",
         r"  ybar=0.4pt, ymode=log, log origin=infty, width=" + width + ", height=" + height + ",",
         r"  scale only axis," if axis_only else "",
         r"  bar width=" + bar + ", enlarge x limits=0.12, ymin=" + f"{ymin:g}" + ", ymax=" + f"{ymax:g}" + ",",
         r"  symbolic x coords={" + ",".join(xcoords) + "}, xtick=data,",
         r"  xticklabels={" + ",".join(xlabels) + "},",
         r"  tick label style={font=\scriptsize}, label style={font=\scriptsize},",
         r"  xticklabel style={align=center, inner sep=1pt},",
         r"  ylabel={" + ylabel + "}, ylabel shift=-3pt, ymajorgrids, grid style={gray!25}, axis x line*=bottom,",
         r"  legend style={font=\scriptsize, draw=none, fill=none, legend columns=" + str(legend_cols) + ","
         r" /tikz/every even column/.append style={column sep=4pt}},",
         r"  legend image code/.code={\draw[#1] (0cm,-0.08cm) rectangle (0.22cm,0.12cm);}, legend cell align=left,",
         (r"  legend to name=" + legend + ",") if legend else "",
         ("  " + opts + ",") if opts else "",
         r"]"]
    for style, pts, _ in series:
        s.append(r"\addplot[" + style + "] coordinates {" + coords(pts) + "};")
    names = [name for _, _, name in series]
    if extra:
        s.append(extra[0])
        if extra[1]:
            names.append(extra[1])
    for img, name in more_legend:
        s.append(r"\addlegendimage{" + img + "}")
        names.append(name)
    if legend:
        s.append(r"\legend{" + ",".join("{" + name + "}" for name in names) + "}")
    s.append(r"\end{axis}")
    return "\n".join(x for x in s if x)


PLAIN_MARK = ("\\addplot[only marks, mark=-, mark size=6pt, line width=1.4pt, black, bar shift=0pt, forget plot]")
PLAIN_LEGEND = "legend image code/.code={\\draw[black, line width=1.4pt] (0cm,0.02cm) -- (0.22cm,0.02cm);}"
HATCH = {PAILLIER: "white", BFV: "white", BFV_PACKED: "blue!60!black", BFV_CLIENT: "blue!60!black", ASHE: "white"}
MBPS = "100 Mbps"


HOM = [("paillier", PAILLIER, "Paillier, CryptDB (non-PQ)"), ("bfv_one", BFV, "BFV, one value per ct.$^\\dagger$ (PQ)"),
       ("bfv", BFV_PACKED, "BFV, packed (PQ)"), ("bfv_client", BFV_CLIENT, "BFV, packed, client grouping (PQ)"),
       ("ashe", ASHE, "ASHE, AES-256 (PQ)")]


def micro():
    j = R("db_microbench.json")["ops"]
    for op, v in R("unpacked_bfv.json")["ops"].items():        # BFV, one value per ciphertext
        j[op]["bfv_one"] = v
    on = ["eq_ingest", "eq_select", "ord_ingest", "ord_select", "join_ingest", "join_adjust", "join_run"]
    on_lbl = ["DET\\\\ingest", "Eq.\\\\select", "OPE\\\\ingest", "Range\\\\select",
              "Join\\\\ingest$^\\ddagger$", "Adjust", "Join$^\\ddagger$"]
    onion = [(CLASSICAL, [(o, j[o]["cryptdb"]["seconds"]) for o in on], "CryptDB (non-PQ)"),
             (PQ, [(o, j[o]["pq"]["seconds"]) for o in on], "Post-quantum")]
    ops = ["ingest", "sum", "filter", "groupby", "update"]
    get = lambda op, sch, f: j[op][sch][f] if sch in j[op] else None
    t = [(st, [(o, get(o, sch, "seconds")) for o in ops], nm) for sch, st, nm in HOM]
    by = [(st, [(o, get(o, sch, "bytes")) for o in ops], nm) for sch, st, nm in HOM]
    rows, groups = R("db_microbench.json")["rows"], R("unpacked_bfv.json")["groups"]
    plain = [("ingest", 8 * rows), ("sum", 8), ("filter", 8), ("groupby", 8 * groups), ("update", 8)]   # 8-byte integers
    plain_mark = (PLAIN_MARK + " coordinates {" + coords(plain) + "};", None, plain)
    tex = ("% generated by benchmarks/make_figures.py from results/db_microbench.json\n"
           "\\begin{tikzpicture}\n" +
           axis("Time (s)", on, on_lbl, onion, width=r"0.33\textwidth", height="2.6cm", bar="3.5pt",
                legend="onionleg", axis_only=True) +
           "\n\\end{tikzpicture}\\hfill\n\\begin{tikzpicture}\n" +
           axis("Time (s)", ops, ["Ingest$^\\dagger$", "SUM", "Filter\\\\1\\%", "GROUP\\\\BY", "Update"], t,
                width=r"0.23\textwidth", height="2.6cm", bar="2.6pt", legend="homleg", legend_cols=3,
                axis_only=True, more_legend=[(PLAIN_LEGEND, "Plaintext (bytes)")]) +
           "\n\\end{tikzpicture}\\hfill\n\\begin{tikzpicture}\n" +
           axis("Bytes", ops, ["Stored\\\\column", "SUM", "Filter\\\\1\\%", "GROUP\\\\BY",
                               "Update"], by, width=r"0.23\textwidth", height="2.6cm", bar="2.6pt",
                axis_only=True, extra=plain_mark) +
           "\n\\end{tikzpicture}\n\n\\smallskip\n"
           "\\pgfplotslegendfromname{onionleg}\\hfill\\pgfplotslegendfromname{homleg}\n")
    open(os.path.join(OUT, "fig_micro.tex"), "w").write(tex)
    return j


def tpch():
    """Fig. 5: TPC-H without (top) and with (bottom) the transfer to the client at 100 Mbps."""
    j = R("tpch_sf0.1.json")
    bw = R("tpch_bandwidth.json")["queries"]
    qs = ["Q1", "Q3", "Q5", "Q6", "Q10"]
    for q, v in R("unpacked_bfv.json")["tpch"]["queries"].items():   # BFV, one value per ciphertext
        j["queries"][q]["pq_bfv_one"] = v
    cfg = [("cryptdb", PAILLIER, "CryptDB, Paillier (non-PQ)"), ("pq_bfv_one", BFV, "BFV, one value per ct.$^\\ddagger$ (PQ)"),
           ("pq_bfv", BFV_PACKED, "BFV, packed (PQ)"), ("pq_bfv_client", BFV_CLIENT, "BFV, packed, client grouping (PQ)"),
           ("pq_ashe", ASHE, "ASHE (PQ)")]
    BAR, GAP = 3.0, 0.4
    shift = lambda i: f"bar shift={(i - (len(cfg) - 1) / 2) * (BAR + GAP):.2f}pt"
    labels = [q + ("$^\\ddagger$" if j["queries"][q]["pq_bfv"].get("projected") else "") for q in qs]
    rel = lambda d: d["rel_server"] + d["rel_client"]
    pts = [(q, rel(j["queries"][q]["cryptdb"])) for q in qs]
    mark = ("\\addplot[only marks, mark=-, mark size=6pt, line width=1.4pt, black, bar shift=0pt, "
            "legend image code/.code={\\draw[black, line width=1.4pt] (0cm,0.02cm) -- (0.22cm,0.02cm);}] "
            "coordinates {" + coords(pts) + "};")
    compute = [(st + ", " + shift(i), [(q, j["queries"][q][k]["total"]) for q in qs], nm)
               for i, (k, st, nm) in enumerate(cfg)]
    with_net = [(st + ", " + shift(i) + ", forget plot, postaction={pattern=north east lines, pattern color="
                 + HATCH[st] + "}", [(q, bw[q][k]["at"][MBPS]) for q in qs], None) for i, (k, st, nm) in enumerate(cfg)]
    lo, hi = bounds(compute + with_net)
    hatch_img = ("fill=white, draw=black!60, line width=0.2pt, postaction={pattern=north east lines, "
                 "pattern color=black!60}")
    top = axis("Without transfer (s)", qs, labels, compute, legend="tpchleg", legend_cols=2, ybounds=(lo, hi),
               extra=(mark, "Relational part on DET/OPE (all)", pts),
               more_legend=[(hatch_img, "Transfer to client at 100\\,Mbps")])
    # bottom: the total at 100 Mbps, hatched, with the compute part drawn over it, so the hatch is the transfer
    bottom = axis("With transfer (s)", qs, labels, with_net + compute, height="3.1cm", ybounds=(lo, hi))
    tex = ("% generated by benchmarks/make_figures.py from results/tpch_sf0.1.json and tpch_bandwidth.json\n"
           "\\begin{tikzpicture}\n" + top +
           "\n\\end{tikzpicture}\n\n\\begin{tikzpicture}\n" + bottom +
           "\n\\end{tikzpicture}\n\n{\\setlength{\\tabcolsep}{2pt}\\pgfplotslegendfromname{tpchleg}}\n")
    open(os.path.join(OUT, "fig_tpch.tex"), "w").write(tex)
    return j


def pk():
    """Fig. 6, use cases that require public-key structure, one panel each (a figure*)."""
    j = R("pk_settings.json")
    W, H = r"0.74\linewidth", "2.5cm"     # \linewidth is the minipage width
    WA, UPLINK = r"0.6\linewidth", 1e6     # panel (a) also has a right axis; uplink in bit/s
    title = lambda t: r"title={" + t + r"}, title style={font=\scriptsize\bfseries, yshift=-4pt}"
    baseline = (r"extra y ticks={1}, extra y tick labels={}, "
                r"extra y tick style={grid=major, major grid style={black, densely dashed}}")
    # (a) many writers, one aggregator
    rows = j["writers"]["rows"]
    lines = [("paillier", "black!60, mark=square*", "Paillier"), ("paillier_packed", "black!60, densely dotted, mark=square",
              "Paillier, packed"), ("ecelgamal", "black!40, mark=triangle*", "EC-ElGamal (Talos)"),
             ("bfv", "blue!65!black, thick, mark=*", "BFV (PQ)"), ("kemdem", "teal!70!black, thick, dashed, mark=diamond*",
                                                                  "ML-KEM+AES (PQ)$^\\S$")]
    a = [r"\begin{axis}[",
         r"  xmode=log, ymode=log, log basis x=2, scale only axis, width=" + W + ", height=" + H + ",",
         r"  xmin=0.8, xmax=10000, ymin=4, ymax=1e6, xtick={1,16,256,4096}, xticklabels={1,16,256,4096},",
         r"  tick label style={font=\scriptsize}, label style={font=\scriptsize},",
         r"  xlabel={Readings per writer message}, ylabel={Bytes per reading}, ylabel shift=-3pt,",
         r"  xlabel shift=-2pt, ymajorgrids, grid style={gray!25}, mark size=1.3pt,",
         r"  legend style={font=\scriptsize, draw=none, fill=none, legend columns=2,",
         r"    /tikz/every even column/.append style={column sep=3pt}}, legend cell align=left,",
         r"  legend to name=writersleg, " + title("(a) Many writers, one aggregator") + ",",
         r"]"]
    a[1] = a[1].replace("width=" + W, "width=" + WA)
    for key, style, name in lines:
        a.append(r"\addplot[" + style + "] coordinates {" +
                 " ".join(f"({r['b']},{r['bytes'][key]:.6g})" for r in rows) + "};")
    a.append(r"\legend{" + ",".join("{" + n + "}" for _, _, n in lines) + "}")
    a.append(r"\end{axis}")
    # right axis: the same bytes as seconds on a 1 Mbps uplink (LTE-M class)
    up = 8 / UPLINK
    a += [r"\begin{axis}[",
          r"  scale only axis, width=" + WA + ", height=" + H + ", xmode=log, ymode=log, xmin=0.8, xmax=10000,",
          f"  ymin={4 * up:g}, ymax={1e6 * up:g}, axis x line=none, axis y line*=right,",
          r"  ytick={1e-4,1e-2,1}, yticklabels={$10^{-4}$,$10^{-2}$,1},",
          r"  tick label style={font=\scriptsize}, label style={font=\scriptsize},",
          r"  ylabel={Seconds at 1\,Mbps}, ylabel shift=-3pt,",
          r"]",
          r"\addplot[draw=none] coordinates {(1,1)};",
          r"\end{axis}"]
    # (b) tuples that third parties verify
    L = j["ledger"]
    base = L["ECDSA P-256"]
    S = R("storage_io.json")["schemes"]
    for k in L:
        L[k]["pg_pages"] = S[k]["postgres"]["queries"]["fetch_signatures"]["pages_touched"]
    mets = [("sign_seconds", "Sign"), ("verify_seconds", "Verify"), ("insert_seconds", "Load"),
            ("db_bytes", "Size"), ("scan_seconds", "Scan"),
            ("pg_total", "Size"), ("pg_pages", "Pages"), ("pg_fetch", "Fetch")]
    getm = lambda d, m: (d["postgres"]["total_bytes"] if m == "pg_total" else
                         d["postgres"]["fetch_signatures_seconds"] if m == "pg_fetch" else d[m])
    n = len(mets)
    rx = lambda x: (x + 0.12 * (n - 1)) / ((n - 1) * 1.24)          # symbolic x -> rel axis x (enlarge 0.12)
    ser = [(ELGAMAL, "HMAC-SHA256", "HMAC (shared key)"), (BFV, "ML-DSA-44", "ML-DSA-44 (PQ)"),
           (LIGHT_BLUE, "ML-DSA-65", "ML-DSA-65 (PQ)")]
    series = [(st, [(m, getm(L[k], m) / getm(base, m)) for m, _ in mets], nm) for st, k, nm in ser]
    b = axis(r"Ratio to ECDSA P-256", [m for m, _ in mets], [l for _, l in mets], series, width=W, height=H,
             bar="2.8pt", legend="ledgerleg", legend_cols=2, axis_only=True,
             opts=baseline + ", " + title("(b) Signed tuples") + r", xticklabel style={font=\tiny}, "
                  r"extra description/.code={\node[font=\tiny, anchor=north] at (rel axis cs:" + f"{rx(2):.3f}" +
                  r",-0.2) {SQLite};\node[font=\tiny, anchor=north] at (rel axis cs:" + f"{rx(6):.3f}" +
                  r",-0.2) {PostgreSQL};\draw[gray] (rel axis cs:" + f"{rx(4.5):.3f}" + r",0) -- (rel axis cs:" +
                  f"{rx(4.5):.3f}" + r",1);}")
    # (c) key hierarchies
    K = R("key_hierarchy.json")["schemes"]
    kb = K["RSA-2048-OAEP"]
    kmets = [("wrap_seconds", "Wrap"), ("unwrap_seconds", "Unwrap"), ("rotate_seconds", "Rotate"),
             ("wrapped_bytes", "Size")]
    kser = [(ELGAMAL, "X25519", "X25519"), (BFV, "ML-KEM-768", "ML-KEM-768 (PQ)"),
            (LIGHT_BLUE, "X25519+ML-KEM-768", "Hybrid (PQ)")]
    kseries = [(st, [(m, K[k][m] / kb[m]) for m, _ in kmets], nm) for st, k, nm in kser]
    c = axis(r"Ratio to RSA-2048", [m for m, _ in kmets], [l for _, l in kmets], kseries, width=W, height=H,
             bar="4pt", legend="keyleg", legend_cols=2, axis_only=True,
             opts=baseline + ", " + title("(c) Key hierarchy") + ", enlarge x limits=0.2")
    panel = lambda body, leg, w: ("\\begin{minipage}[t]{" + w + "\\textwidth}\\centering\n"
                                  "\\begin{tikzpicture}[baseline=(current axis.south)]\n" + body +
                               "\n\\end{tikzpicture}\n\n\\smallskip\n{\\setlength{\\tabcolsep}{2pt}"
                               "\\pgfplotslegendfromname{" + leg + "}}\n\\end{minipage}")
    tex = ("% generated by benchmarks/make_figures.py from results/pk_settings.json and key_hierarchy.json\n" +
           panel("\n".join(a), "writersleg", "0.31") + "\\hfill\n" + panel(b, "ledgerleg", "0.39") + "\\hfill\n" +
           panel(c, "keyleg", "0.28") + "\n")
    open(os.path.join(OUT, "fig_pk.tex"), "w").write(tex)
    return j


# ---------------------------------------------------------------- Fig. (primitives), ratios
CHEAPER = "fill=teal!55, draw=teal!40!black" + LW
COSTLIER = "fill=orange!75!black, draw=orange!40!black" + LW


def nbytes(n):
    """64 -> '64\\,B', 2272 -> '2.3\\,KB', 413696 -> '414\\,KB' (decimal units, as in the text)."""
    if n < 1000:
        return f"{n:.0f}\\,B"
    if n < 1e6:
        return (f"{n / 1e3:.1f}" if n < 1e4 else f"{n / 1e3:.0f}") + "\\,KB"
    return (f"{n / 1e6:.1f}" if n < 1e7 else f"{n / 1e6:.0f}") + "\\,MB"


def factor(r):
    x = r if r >= 1 else 1 / r
    if x >= 100:                      # two significant digits, as in the text
        from math import floor, log10
        x = round(x, -int(floor(log10(x))) + 1)
    s = f"{x:.1f}" if x < 9.95 else f"{x:,.0f}".replace(",", "{,}")
    return s + "$\\times$"


def ratio_panel(groups, xlabel, xmin, xmax, xticks, width, height_cm):
    """Rows are (label, ratio) or (label, ratio, ratio_unpacked); a header row has ratio None.
    Bars run from 1 on a log axis. With two ratios, the unpacked one is the filled bar and the packed
    one (the first ratio) a dashed box below it."""
    rows = []
    for head, items in groups:
        rows.append((r"\textbf{" + head + "}", None, None, None))
        rows += [(it[0], it[1], it[2] if len(it) > 2 else None, it[3] if len(it) > 3 else None) for it in items]
    n = len(rows)
    row_cm = height_cm / (n + 0.2)
    hh = 0.095 / row_cm                     # half bar height, in rows, so bars look alike in both panels
    tl = ",".join(f"$10^{{{e}}}$" if e else "1" for e in xticks)
    s = [r"\begin{axis}[",
         r"  xmode=log, y dir=reverse, scale only axis,",
         f"  width={width}, height={height_cm}cm, xmin={xmin:g}, xmax={xmax:g},",
         f"  ymin=-0.6, ymax={n - 0.4}, ytick={{0,...,{n - 1}}}, yticklabels={{" + ",".join("{" + l + "}" for l, _, _, _ in rows) + "},",
         "  xtick={" + ",".join(f"1e{e}" for e in xticks) + "}, xticklabels={" + tl + "},",
         r"  tick label style={font=\scriptsize}, label style={font=\scriptsize}, yticklabel style={inner sep=2pt},",
         r"  major y tick style={draw=none}, xlabel={" + xlabel + r"}, xmajorgrids, grid style={gray!25}, clip=false,",
         r"  extra x ticks={1}, extra x tick labels={}, extra x tick style={grid=major, major grid style={black, line width=0.5pt}},",
         r"]",
         r"\addplot[draw=none] coordinates {(1,0)};"]

    def bar(r, y, h, dashed, note=None):
        good = r < 1
        if dashed:
            st = ("draw=teal!45!black, fill=teal!12" if good else "draw=orange!45!black, fill=orange!14") + ", densely dashed, line width=0.45pt"
        else:
            st = CHEAPER if good else COSTLIER
        out = [f"\\filldraw[{st}] (axis cs:1,{y - h:.4f}) rectangle (axis cs:{r:.6g},{y + h:.4f});"]
        anchor = "west" if r >= 1 else "east"
        text = factor(r) + (f" ({note})" if note else "")
        out.append(f"\\node[font=\\tiny, anchor={anchor}, inner sep=2.5pt] at (axis cs:{r:.6g},{y:.4f}) {{{text}}};")
        return out

    for i, (lbl, r, r1, note) in enumerate(rows):
        if r is None:
            if i:
                s.append(f"\\draw[gray!50] ({{rel axis cs:0,0}} |- {{axis cs:1,{i - 0.5}}}) -- ({{rel axis cs:1,0}} |- {{axis cs:1,{i - 0.5}}});")
            continue
        if r1 is None:
            s += bar(r, i, hh, False, note)
        else:                               # two thinner bars, 0.16 cm apart
            d, h2 = 0.085 / row_cm, 0.068 / row_cm
            n_r, n_r1 = note or (None, None)
            s += bar(r1, i - d, h2, False, n_r1)  # one value per ciphertext, filled
            s += bar(r, i + d, h2, True, n_r)     # 8,192 values packed, dashed
    s.append(r"\end{axis}")
    return "\n".join(s)


def primitives_fig():
    """Fig. 2: common classical parameters against recommended post-quantum ones."""
    j = R("primitives.json")
    kex, wrap, sig, aes, he = j["kex"], j["wrap"], j["sig"], j["aes"], j["additive_he"]
    p, f, e = he["Paillier-2048"], he["BFV-8192"], he["EC-ElGamal P-256"]
    v = f["values_per_ct"]
    enc, add = f["encrypt"] * v, f["add"] * v
    a128, a256 = aes["AES-128-GCM"], aes["AES-256-GCM"]
    t = lambda c, q: c / q          # post-quantum time / classical time, from rates
    km, rsa = kex["ML-KEM-768"], wrap["RSA-2048-OAEP"]
    time = [
        ("ML-KEM-768, key exchange", [("Handshake vs X25519", t(kex["X25519"]["handshake"], km["handshake"])),
                                      ("Handshake vs ECDH P-256", t(kex["ECDH P-256"]["handshake"], km["handshake"]))]),
        ("ML-KEM-768 vs RSA-2048", [("Wrap a key", t(rsa["wrap"], km["encaps"])),
                                    ("Unwrap a key", t(rsa["unwrap"], km["decaps"]))]),
        ("ML-DSA vs ECDSA P-256", [("ML-DSA-44, sign", t(sig["ECDSA P-256"]["sign"], sig["ML-DSA-44"]["sign"])),
                                   ("ML-DSA-44, verify", t(sig["ECDSA P-256"]["verify"], sig["ML-DSA-44"]["verify"])),
                                   ("ML-DSA-65, sign", t(sig["ECDSA P-256"]["sign"], sig["ML-DSA-65"]["sign"])),
                                   ("ML-DSA-65, verify", t(sig["ECDSA P-256"]["verify"], sig["ML-DSA-65"]["verify"]))]),
        ("AES-256 vs AES-128", [("32\\,B message", t(a128["32"]["bytes_per_s"], a256["32"]["bytes_per_s"])),
                                ("1\\,MB message", t(a128[str(1 << 20)]["bytes_per_s"], a256[str(1 << 20)]["bytes_per_s"]))]),
        ("BFV vs Paillier-2048", [("Encrypt", t(p["encrypt"], enc), t(p["encrypt"], f["encrypt"])),
                                  ("Add", t(p["add"], add), t(p["add"], f["add"])),
                                  ("Decrypt", t(p["decrypt"], f["decrypt"]))]),
        ("BFV vs EC-ElGamal", [("Encrypt", t(e["encrypt"], enc), t(e["encrypt"], f["encrypt"])),
                               ("Add", t(e["add"], add), t(e["add"], f["add"])),
                               ("Decrypt, sum $<2^{16}$", t(e["decrypt_by_bits"]["16"], f["decrypt"])),
                               ("Decrypt, sum $<2^{32}$", t(e["decrypt_by_bits"]["32"], f["decrypt"]))]),
    ]
    per = f["ct"] / v
    msgs = km["pk"] + km["ct"]
    sz = lambda a, b: f"{nbytes(a)}$\\to${nbytes(b)}"          # classical size -> post-quantum size
    x2, p2 = 2 * kex["X25519"]["msg"], 2 * kex["ECDH P-256"]["msg"]
    es = sig["ECDSA P-256"]["sig"]
    size = [
        ("ML-KEM-768", [("Handshake vs X25519", msgs / x2, None, sz(x2, msgs)),
                        ("Handshake vs ECDH P-256", msgs / p2, None, sz(p2, msgs)),
                        ("Wrapped key vs RSA-2048", km["ct"] / rsa["ct"], None, sz(rsa["ct"], km["ct"]))]),
        ("ML-DSA vs ECDSA P-256", [("ML-DSA-44 signature", sig["ML-DSA-44"]["sig"] / es, None,
                                    sz(es, sig["ML-DSA-44"]["sig"])),
                                   ("ML-DSA-65 signature", sig["ML-DSA-65"]["sig"] / es, None,
                                    sz(es, sig["ML-DSA-65"]["sig"]))]),
        ("AES-256 vs AES-128", [("Ciphertext", 1.0)]),
        ("BFV, per value", [("vs Paillier-2048", per / p["ct"], f["ct"] / p["ct"],
                             ("$\\to$" + nbytes(per), sz(p["ct"], f["ct"]))),
                            ("vs EC-ElGamal", per / e["ct"], f["ct"] / e["ct"],
                             ("$\\to$" + nbytes(per), sz(e["ct"], f["ct"])))]),
    ]
    tex = ("% generated by benchmarks/make_figures.py from results/primitives.json\n"
           "\\begin{tikzpicture}[baseline=(current bounding box.north)]\n" +
           ratio_panel(time, r"Time, post-quantum $\div$ classical", 1e-6, 1e3, [-6, -3, 0, 3],
                       r"0.25\textwidth", 8.0) +
           "\n\\end{tikzpicture}\\hspace{2em}\n\\begin{tikzpicture}[baseline=(current bounding box.north)]\n" +
           ratio_panel(size, r"Bytes, post-quantum $\div$ classical", 1e-4, 1e7, [-3, 0, 3, 6],
                       r"0.22\textwidth", 8.0) +
           "\n\\end{tikzpicture}\n")
    open(os.path.join(OUT, "fig_primitives.tex"), "w").write(tex)
    return j


# ------------------------------------------------- Fig. 3, raising classical parameters
def scaling_fig():
    """Time per operation relative to the common classical parameter, across security levels.
    Classical schemes are lines; post-quantum parameter sets are placed at the classical
    level of their NIST category (1/2 -> 128, 3 -> 192, 5 -> 256)."""
    j = R("primitives.json")
    kex, wrap, sig, he = j["kex"], j["wrap"], j["sig"], j["additive_he"]
    rel = lambda base, r: base / r          # time relative to the base, from rates
    x25519 = kex["X25519"]["handshake"]
    rsa2048 = wrap["RSA-2048-OAEP"]["unwrap"]
    p256 = sig["ECDSA P-256"]["sign"]
    pa = he["Paillier-2048"]
    panels = [
        ("Key exchange (handshake)", "relative to X25519", "south east", [
            ("X25519, X448", "black, mark=*", [(128, rel(x25519, kex["X25519"]["handshake"])),
                                                (224, rel(x25519, kex["X448"]["handshake"]))]),
            ("ECDH, NIST curves", "black!50, mark=square*", [(128, rel(x25519, kex["ECDH P-256"]["handshake"])),
                                                              (192, rel(x25519, kex["ECDH P-384"]["handshake"])),
                                                              (256, rel(x25519, kex["ECDH P-521"]["handshake"]))]),
            ("ML-KEM", "blue!65!black, dashed, mark=diamond*", [(192, rel(x25519, kex["ML-KEM-768"]["handshake"])),
                                                                         (256, rel(x25519, kex["ML-KEM-1024"]["handshake"]))])]),
        ("Key unwrap", "relative to RSA-2048", "north west", [
            ("RSA", "black, mark=*", [(112, 1.0), (128, rel(rsa2048, wrap["RSA-3072-OAEP"]["unwrap"])),
                                                     (192, rel(rsa2048, wrap["RSA-7680-OAEP"]["unwrap"]))]),
            ("ML-KEM", "blue!65!black, dashed, mark=diamond*", [(192, rel(rsa2048, kex["ML-KEM-768"]["decaps"])),
                                                                         (256, rel(rsa2048, kex["ML-KEM-1024"]["decaps"]))])]),
        ("Signing", "relative to ECDSA P-256", "south east", [
            ("ECDSA", "black, mark=*", [(128, 1.0), (192, rel(p256, sig["ECDSA P-384"]["sign"])),
                                                      (256, rel(p256, sig["ECDSA P-521"]["sign"]))]),
            ("ML-DSA", "blue!65!black, dashed, mark=diamond*", [(128, rel(p256, sig["ML-DSA-44"]["sign"])),
                                                                         (192, rel(p256, sig["ML-DSA-65"]["sign"])),
                                                                         (256, rel(p256, sig["ML-DSA-87"]["sign"]))])]),
        ("Paillier", "relative to Paillier-2048", "south east", [
            ("Encrypt", "black, mark=*", [(112, 1.0), (128, rel(pa["encrypt"], he["Paillier-3072"]["encrypt"])),
                                          (192, rel(pa["encrypt"], he["Paillier-7680"]["encrypt"]))]),
            ("Add", "black!50, mark=square*", [(112, 1.0), (128, rel(pa["add"], he["Paillier-3072"]["add"])),
                                               (192, rel(pa["add"], he["Paillier-7680"]["add"]))]),
            ("Decrypt", "black!35, mark=triangle*", [(112, 1.0), (128, rel(pa["decrypt"], he["Paillier-3072"]["decrypt"])),
                                                    (192, rel(pa["decrypt"], he["Paillier-7680"]["decrypt"]))])]),
    ]
    # The paper keeps the two panels with a post-quantum counterpart that differ most (one column).
    panels = [pn for pn in panels if pn[0] in ("Key unwrap", "Signing")]
    X = {112: 1, 128: 2, 192: 3, 224: 3.5, 256: 4}     # ordinal positions, so 112 and 128 do not collide
    out = ["% generated by benchmarks/make_figures.py from results/primitives.json"]
    for k, (title, ylab, lpos, lines) in enumerate(panels):
        ys = [y for _, _, pts in lines for _, y in pts]
        lo, hi = 10 ** math.floor(math.log10(min(ys) / 1.5)), 10 ** math.ceil(math.log10(max(ys) * 1.5))
        a = [r"\begin{tikzpicture}[baseline=(current bounding box.north)]", r"\begin{axis}[",
             r"  scale only axis, width=0.33\columnwidth, height=2.6cm, ymode=log,",
             f"  ymin={lo:g}, ymax={hi:g}, xmin=0.6, xmax=4.4, xtick={{1,2,3,4}}, xticklabels={{112,128,192,256}},",
             r"  tick label style={font=\scriptsize}, label style={font=\scriptsize}, title style={font=\scriptsize\bfseries},",
             f"  title={{{title}}}, ylabel={{{ylab}}}, ylabel shift=-3pt,",
             r"  xlabel={Classical security (bits)}, ymajorgrids, grid style={gray!25},",
             r"  legend style={font=\tiny, draw=none, fill=white, fill opacity=0.85, text opacity=1, inner sep=1pt, row sep=-2pt," +
             {"south east": "at={(0.98,0.03)}, anchor=south east},", "east": "at={(0.98,0.5)}, anchor=east},",
              "north west": "at={(0.03,0.97)}, anchor=north west},",
              "south west": "at={(0.03,0.03)}, anchor=south west},"}[lpos],
             r"  legend cell align=left, x tick label style={rotate=0},",
             r"]"]
        for name, style, pts in lines:
            a.append(f"\\addplot[{style}, line width=0.7pt, mark size=1.6pt] coordinates {{" +
                     " ".join(f"({X[x]},{y:.6g})" for x, y in pts) + "};")
            a.append(f"\\addlegendentry{{{name}}}")
        a += [r"\end{axis}", r"\end{tikzpicture}"]
        out.append("\n".join(a) + ("\\hfill" if k < len(panels) - 1 else ""))
    open(os.path.join(OUT, "fig_scaling.tex"), "w").write("\n".join(out) + "\n")
    return panels

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    primitives_fig(); scaling_fig(); micro(); tpch(); pk()
    print("wrote", sorted(os.listdir(OUT)))
