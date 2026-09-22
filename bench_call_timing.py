import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import MOomf                                                            # noqa: E402
import nsga2                                                            # noqa: E402
from ImprovedLEACH import LeachParams, generate_topology               # noqa: E402
from MOomf import OMFParams, optimize_omf_pareto_front                 # noqa: E402
from nsga2 import NSGA2Params, optimize_nsga2_pareto_front             # noqa: E402
from pso_leach import PSOParams, _MOPSOProblem                         # noqa: E402
from pymoo.algorithms.moo.mopso_cd import MOPSO_CD                      # noqa: E402
from pymoo.optimize import minimize                                     # noqa: E402

COUNT = {"n": 0}


def _wrap(module):
    orig = module.full_round_objectives

    def counted(*a, **k):
        COUNT["n"] += 1
        return orig(*a, **k)

    module.full_round_objectives = counted


_wrap(MOomf)
_wrap(nsga2)          # MOPSO objectives are evaluated through nsga2 as well


def call_mopso(pso, topo, E, alive, rng):
    """Same construction as run_pso_leach (pso_leach.py), for one round."""
    alive_idx = np.where(alive)[0]
    n_alive = len(alive_idx)
    K = min(int(max(1, round(pso.base.p * n_alive))), n_alive)
    problem = _MOPSOProblem(pso=pso, topo=topo, E=E, alive=alive, alive_idx=alive_idx,
                            n_nodes=topo["n"], k=K, rng=rng)
    algorithm = MOPSO_CD(pop_size=int(pso.swarm_size), w=float(pso.w_inertia),
                         c1=float(pso.c1), c2=float(pso.c2),
                         archive_size=max(int(pso.swarm_size) * 5, 50))
    minimize(problem=problem, algorithm=algorithm,
             termination=("n_gen", max(1, int(pso.iters))),
             seed=int(rng.integers(0, 2_147_483_647)), verbose=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-nodes", type=int, default=80)
    ap.add_argument("--area-w", type=float, default=100.0)
    ap.add_argument("--area-h", type=float, default=100.0)
    ap.add_argument("--bs-x", type=float, default=None)
    ap.add_argument("--bs-y", type=float, default=None)
    ap.add_argument("--states", type=int, nargs="+", default=[80, 60, 40, 20],
                    help="numbers of alive nodes at which the call is timed")
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=3, help="topology seed")
    ap.add_argument("--out", default="results_timing")
    a = ap.parse_args()

    bs_x = a.bs_x if a.bs_x is not None else a.area_w / 2
    bs_y = a.bs_y if a.bs_y is not None else 1.5 * a.area_h
    p = LeachParams(n_nodes=a.n_nodes, area_w=a.area_w, area_h=a.area_h, bs_x=bs_x, bs_y=bs_y)
    topo = generate_topology(p, a.seed)
    n = topo["n"]
    states = [s for s in a.states if s <= n]
    os.makedirs(a.out, exist_ok=True)

    omf, nsga, pso = OMFParams(base=p), NSGA2Params(base=p), PSOParams(base=p)
    runners = {
        "OMF": lambda E, al, rng: optimize_omf_pareto_front(omf, topo, E, al, rng),
        "NSGA-II": lambda E, al, rng: optimize_nsga2_pareto_front(nsga, topo, E, al, rng),
        "MOPSO": lambda E, al, rng: call_mopso(pso, topo, E, al, rng),
    }

    # warm-up (imports, caches) -- not recorded
    al = np.ones(n, dtype=bool)
    for f in runners.values():
        f(np.full(n, p.e0), al, np.random.default_rng(0))

    rows = []
    for rep in range(a.reps):
        for n_alive in states:
            alive = np.zeros(n, dtype=bool)
            alive[:n_alive] = True
            E = np.where(alive, p.e0, 0.0)
            for name, f in runners.items():          # interleaved order
                rng = np.random.default_rng(1000 + rep)
                COUNT["n"] = 0
                t0 = time.perf_counter()
                f(E, alive, rng)
                dt = time.perf_counter() - t0
                rows.append(dict(algorithm=name, n_alive=n_alive, rep=rep,
                                 seconds=dt, evaluations=COUNT["n"]))

    raw = os.path.join(a.out, "call_timing_raw.csv")
    with open(raw, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    mean_ms = {}
    print(f"\nScenario: {n} nodes, {a.area_w:g}x{a.area_h:g} m | states N_a = {states} | reps = {a.reps}")
    print(f"{'algorithm':<9} " + " ".join(f"{'Na='+str(s):>9}" for s in states) +
          f" {'mean (ms)':>11} {'evals/call':>11}")
    for name in runners:
        sub = [r for r in rows if r["algorithm"] == name]
        per_state = [1000 * np.mean([r["seconds"] for r in sub if r["n_alive"] == s]) for s in states]
        m = 1000 * np.mean([r["seconds"] for r in sub])
        mean_ms[name] = m
        evals = sorted({r["evaluations"] for r in sub})
        ev = f"{evals[0]}" if len(evals) == 1 else f"{evals[0]}-{evals[-1]}"
        print(f"{name:<9} " + " ".join(f"{v:>9.1f}" for v in per_state) + f" {m:>11.1f} {ev:>11}")

    x, y, z = mean_ms["OMF"], mean_ms["NSGA-II"], mean_ms["MOPSO"]
    print("\nValues for Section V-C:")
    print(f"  X = {x:.0f} ms (OMF)   Y = {y:.0f} ms (NSGA-II, +{100*(y/x-1):.0f}%)   "
          f"Z = {z:.0f} ms (MOPSO, +{100*(z/x-1):.0f}%)")
    print(f"\nRaw data: {raw}")


if __name__ == "__main__":
    main()
