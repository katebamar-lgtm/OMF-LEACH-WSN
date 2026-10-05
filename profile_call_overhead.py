"""
profile_call_overhead.py
------------------------
Supplementary check for Section V-C: how much of one CH-selection call is the objective
evaluation shared by the three optimizers, and how much is optimizer/framework overhead?

For each network state N_a, it times (i) 600 raw objective evaluations of random CH sets and
(ii) one full OMF call and one full NSGA-II call (pymoo), interleaved, on the same topology.

Usage (run alone on an idle machine; put it next to MOomf.py / nsga2.py):
    python profile_call_overhead.py --seed 3 --reps 8 --out results_profile
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ImprovedLEACH import LeachParams                                   # noqa: E402
from CustomLEACH import generate_topology                               # noqa: E402
from MOomf import OMFParams, optimize_omf_pareto_front                  # noqa: E402
from nsga2 import (NSGA2Params, optimize_nsga2_pareto_front,            # noqa: E402
                   evaluate_nsga_objectives)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--states", type=int, nargs="+", default=[80, 60, 40, 20])
    ap.add_argument("--out", default="results_profile")
    a = ap.parse_args()

    p = LeachParams()
    topo = generate_topology(p, a.seed)
    n = topo["n"]
    omf, ns = OMFParams(base=p), NSGA2Params(base=p)
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for na in a.states:
        alive = np.zeros(n, dtype=bool)
        alive[np.random.default_rng(0).choice(n, size=na, replace=False)] = True
        E = np.where(alive, p.e0, 0.0)
        ai = np.where(alive)[0]
        k = max(1, int(round(p.p * na)))
        for rep in range(a.reps):
            rng = np.random.default_rng(1000 + rep)
            cands = [rng.choice(ai, size=k, replace=False) for _ in range(600)]
            t = time.perf_counter()
            for c in cands:
                evaluate_nsga_objectives(ns, topo, E, c)
            t_eval = (time.perf_counter() - t) * 1e3
            t = time.perf_counter()
            optimize_omf_pareto_front(omf, topo, E, alive, np.random.default_rng(2000 + rep))
            t_omf = (time.perf_counter() - t) * 1e3
            t = time.perf_counter()
            optimize_nsga2_pareto_front(ns, topo, E, alive, np.random.default_rng(3000 + rep))
            t_ns = (time.perf_counter() - t) * 1e3
            rows.append([na, rep, t_eval, t_omf, t_ns])
    path = os.path.join(a.out, "profile_call_overhead.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["N_a", "rep", "eval600_ms", "omf_call_ms", "nsga2_call_ms"])
        w.writerows(rows)
    arr = np.asarray(rows, dtype=float)
    for na in a.states:
        m = arr[arr[:, 0] == na]
        e, o, s = m[:, 2].mean(), m[:, 3].mean(), m[:, 4].mean()
        print(f"N_a={na:3d}  600 evals {e:7.1f} ms | OMF {o:7.1f} ms ({100*e/o:4.0f}% evals)"
              f" | NSGA-II {s:7.1f} ms ({100*e/s:4.0f}% evals) | NSGA-II/OMF = {s/o:5.2f}")
    print("saved", path)


if __name__ == "__main__":
    main()
