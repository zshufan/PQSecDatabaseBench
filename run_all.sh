#!/usr/bin/env bash
# Run every benchmark in dependency order, then regenerate the figures.
# About 30 minutes on one core. Results go to benchmarks/results/, figures to figures/.
set -euo pipefail
cd "$(dirname "$0")/benchmarks"

python3 primitives_bench.py            # primitives.json        (Figs. 2, 3)
python3 db_microbench.py               # db_microbench.json     (Fig. 4; reads primitives.json)
python3 tpch_bench.py --sf 0.1         # tpch_sf0.1.json        (Fig. 5)
python3 unpacked_bfv_bench.py          # unpacked_bfv.json      (Figs. 4, 5; reads primitives.json, tpch_sf0.1.json)
python3 bandwidth_model.py             # tpch_bandwidth.json    (Fig. 5 bottom; reads tpch_sf0.1.json, unpacked_bfv.json)
python3 pk_settings_bench.py           # pk_settings.json       (Fig. 6a-b; PostgreSQL 16 optional)
python3 storage_io_bench.py            # storage_io.json        (Fig. 6b pages; PostgreSQL 16 optional)
python3 keyhier_bench.py               # key_hierarchy.json     (Fig. 6c)
python3 hndl_join_aggregate_attack.py  # hndl_join_aggregate_attack.json (Sec. II-A)
python3 sizing_model.py                # sizing_model.json      (Sec. IV-C, analytic)
python3 make_figures.py                # ../figures/*.tex
