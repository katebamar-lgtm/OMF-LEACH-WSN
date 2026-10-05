"""
ablation_selection_pool.py
--------------------------
Robustness check requested by the weaknesses analysis (NOT used for the paper's tables).

Question: is OMF's FND/energy advantage over NSGA-II partly due to the *size of the pool*
from which the compromise (knee-point) solution is selected?
    * OMF  : selects from its full non-dominated archive (~10^3 points per round)
    * NSGA-II: selects from the final population's non-dominated set (<= 30 points)
and to the fact that NSGA2/MOPSO decode a real vector into a CH set with a *random* repair
(CustomLEACH.decode_particle), so the CH set deployed may differ from the one evaluated.

Variants of NSGA-II (same pymoo algorithm, same parameters, same budget = 600 evaluations):
    exact    : final non-dominated set, but each solution is deployed EXACTLY as it was
               evaluated (decode cached per individual)
    archive  : non-dominated archive of ALL evaluated candidates (like OMF), deployed
               exactly as evaluated

Usage:
    python ablation_selection_pool.py --variant exact   --seeds 1 2 3 --n-rounds 700
    python ablation_selection_pool.py --variant archive --seeds 1 2 3 --n-rounds 700
(use --n-rounds 2500 for full-lifetime runs; FND needs only ~700 rounds)
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import nsga2                                                   # noqa: E402
from nsga2 import NSGA2Params, run_nsga2_leach                  # noqa: E402
from CustomLEACH import decode_particle                         # noqa: E402
from ImprovedLEACH import LeachParams                           # noqa: E402

from pymoo.algorithms.moo.nsga2 import NSGA2                    # noqa: E402
from pymoo.operators.crossover.sbx import SBX                   # noqa: E402
from pymoo.operators.mutation.pm import PM                      # noqa: E402
from pymoo.optimize import minimize                             # noqa: E402

VARIANT = "exact"


class _RecordingProblem(nsga2._NSGA2Problem):
    """Same problem as the paper, but remembers the CH set actually evaluated."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.cache = {}      # bytes(x) -> (ch_idx, F)

    def _evaluate(self, x, out, *args, **kwargs):
        x = np.asarray(x, dtype=float)
        ch_idx = decode_particle(x, self.alive_idx, self.k, self.rng)
        F = nsga2.evaluate_nsga_objectives(self.nsga, self.topo, self.E, ch_idx)
        out["F"] = F
        self.cache[x.tobytes()] = (np.asarray(ch_idx, dtype=int).copy(),
                                   np.asarray(F, dtype=float).copy())


def _nondominated(F):
    F = np.asarray(F, dtype=float)
    keep = np.ones(len(F), dtype=bool)
    for i in range(len(F)):
        if not keep[i]:
            continue
        dom = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        if dom.any():
            keep[i] = False
    return np.where(keep)[0]


def patched_optimize(nsga, topo, E, alive, rng):
    alive_idx = np.where(alive)[0]
    n_alive = len(alive_idx)
    if n_alive == 0:
        return [], np.empty((0, 3), dtype=float)
    k = min(int(max(1, round(nsga.base.p * n_alive))), n_alive)
    problem = _RecordingProblem(nsga=nsga, topo=topo, E=E, alive_idx=alive_idx, k=k, rng=rng)
    algorithm = NSGA2(pop_size=int(nsga.pop_size),
                      crossover=SBX(prob=float(nsga.crossover_prob), eta=15.0),
                      mutation=PM(prob=float(nsga.mutation_prob), eta=20.0),
                      eliminate_duplicates=False)
    result = minimize(problem=problem, algorithm=algorithm,
                      termination=("n_gen", max(1, int(nsga.generations))),
                      seed=int(rng.integers(0, 2_147_483_647)), verbose=False)
    if result is None or result.X is None:
        return [], np.empty((0, 3), dtype=float)

    if VARIANT == "exact":
        X = np.atleast_2d(np.asarray(result.X, dtype=float))
        sols, objs = [], []
        for x in X:
            ch, F = problem.cache[x.tobytes()]
            sols.append(ch)
            objs.append(F)
        return sols, np.asarray(objs)

    # VARIANT == "archive": non-dominated set of every evaluated candidate
    chs = [v[0] for v in problem.cache.values()]
    Fs = np.asarray([v[1] for v in problem.cache.values()])
    idx = _nondominated(Fs)
    return [chs[i] for i in idx], Fs[idx]


def main():
    global VARIANT
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["baseline", "exact", "archive"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--n-rounds", type=int, default=700)
    ap.add_argument("--out", default="results_ablation")
    a = ap.parse_args()
    VARIANT = a.variant
    if a.variant != "baseline":
        nsga2.optimize_nsga2_pareto_front = patched_optimize

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"ablation_{a.variant}.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["variant", "seed", "n_rounds", "FND", "HND", "LND", "seconds"])
        for s in a.seeds:
            p = LeachParams(n_rounds=a.n_rounds)
            t = time.time()
            res = run_nsga2_leach(NSGA2Params(base=p), seed=s)
            w.writerow([a.variant, s, a.n_rounds, res.get("FND"), res.get("HND"),
                        res.get("LND"), round(time.time() - t, 1)])
            fh.flush()
            print(a.variant, s, res.get("FND"), res.get("HND"), res.get("LND"), flush=True)


if __name__ == "__main__":
    main()
