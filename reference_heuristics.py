"""
reference_heuristics.py
-----------------------
Two trivial CH-selection heuristics run in the SAME simulator as OMF / NSGA-II /
MOPSO (CustomLEACH engine: Dijkstra min-energy routing, Ds = d0,
Ncl_max = round(N_alive / K)), with the same topologies (seed -> generate_topology)
and the same K rule K = max(1, round(P * N_alive)).

    random      : K alive nodes drawn uniformly at random each round
    topk_energy : the K alive nodes with the highest residual energy each round

"""
import argparse
import csv
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ImprovedLEACH import LeachParams, generate_topology, _d0          # noqa: E402
from CustomLEACH import _simulate_one_round                             # noqa: E402


def run_reference(mode: str, seed: int, params: LeachParams) -> dict:
    topo = generate_topology(params, seed)
    n = topo["n"]
    rng = np.random.default_rng(seed + 5000)
    E = np.full(n, params.e0, dtype=float)
    alive = np.ones(n, dtype=bool)
    half = n // 2
    FND = HND = LND = None
    delivered = lost = 0
    consumed = []

    for r in range(params.n_rounds):
        idx = np.where(alive)[0]
        n_alive = len(idx)
        if n_alive == 0:
            break
        k = min(max(1, int(round(params.p * n_alive))), n_alive)
        if mode == "random":
            ch = rng.choice(idx, size=k, replace=False)
        elif mode == "topk_energy":
            ch = idx[np.argsort(-E[idx], kind="stable")[:k]]
        else:
            raise ValueError(mode)
        ncl_max = max(1, int(round(n_alive / max(k, 1))))

        e_before = float(E.sum())
        E, alive, data_delivered, stats_round = _simulate_one_round(
            params, topo, E, alive, ch, Ds=_d0(params), Ncl_max=ncl_max
        )
        delivered += int(data_delivered)
        lost += int(stats_round["packet_loss"])
        consumed.append(max(0.0, e_before - float(E.sum())))

        n_after = int(np.count_nonzero(alive))
        if FND is None and n_after < n:
            FND = r + 1
        if HND is None and n_after <= half:
            HND = r + 1
        if n_after == 0:
            LND = r + 1
            break

    n_rounds = params.n_rounds
    return dict(
        algorithm=mode, seed=seed,
        FND=FND or n_rounds, HND=HND or n_rounds, LND=LND or n_rounds,
        total_packets=delivered, total_packet_loss=lost,
        packet_loss_ratio=lost / max(delivered + lost, 1),
        avg_energy_consumed_per_round=float(np.mean(consumed)) if consumed else 0.0,
    )


def mean_ci(x):
    x = np.asarray(x, dtype=float)
    m = x.mean()
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
    return m, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-nodes", type=int, default=80)
    ap.add_argument("--area-w", type=float, default=100.0)
    ap.add_argument("--area-h", type=float, default=100.0)
    ap.add_argument("--bs-x", type=float, default=None, help="default: area_w / 2")
    ap.add_argument("--bs-y", type=float, default=None, help="default: 1.5 * area_h")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--n-rounds", type=int, default=2500)
    ap.add_argument("--out", default="results_reference")
    a = ap.parse_args()

    bs_x = a.bs_x if a.bs_x is not None else a.area_w / 2
    bs_y = a.bs_y if a.bs_y is not None else 1.5 * a.area_h
    params = LeachParams(n_nodes=a.n_nodes, area_w=a.area_w, area_h=a.area_h,
                         bs_x=bs_x, bs_y=bs_y, n_rounds=a.n_rounds)
    os.makedirs(a.out, exist_ok=True)

    rows = []
    for mode in ("random", "topk_energy"):
        for s in range(1, a.seeds + 1):
            rows.append(run_reference(mode, s, params))

    raw_path = os.path.join(a.out, "reference_heuristics_raw.csv")
    with open(raw_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    metrics = ["FND", "HND", "LND", "avg_energy_consumed_per_round", "packet_loss_ratio"]
    summary = []
    print(f"\nScenario: {a.n_nodes} nodes, {a.area_w:g}x{a.area_h:g} m, BS=({bs_x:g},{bs_y:g}), "
          f"{a.seeds} seeds (mean +/- 95% t-CI)")
    print(f"{'heuristic':<12} {'FND':>16} {'HND':>16} {'LND':>16} {'E/round (J)':>18} {'loss (%)':>14}")
    for mode in ("random", "topk_energy"):
        sub = [r for r in rows if r["algorithm"] == mode]
        rec = {"algorithm": mode}
        cells = []
        for m in metrics:
            mu, h = mean_ci([r[m] for r in sub])
            rec[m + "_mean"], rec[m + "_ci95"] = mu, h
            cells.append((mu, h))
        summary.append(rec)
        print(f"{mode:<12} "
              f"{cells[0][0]:>9.1f}+/-{cells[0][1]:<5.1f} "
              f"{cells[1][0]:>9.1f}+/-{cells[1][1]:<5.1f} "
              f"{cells[2][0]:>9.1f}+/-{cells[2][1]:<5.1f} "
              f"{cells[3][0]:>10.4f}+/-{cells[3][1]:<7.4f} "
              f"{100*cells[4][0]:>7.2f}+/-{100*cells[4][1]:<5.2f}")

    sum_path = os.path.join(a.out, "reference_heuristics_summary.csv")
    with open(sum_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"\nSaved: {raw_path}\n       {sum_path}")


if __name__ == "__main__":
    main()
