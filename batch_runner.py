"""
batch_runner.py  (v2 — multi-objective indicators)
====================================================
Parallel multi-seed experiment runner for the OMF-LEACH / Improved LEACH /
NSGA-II / MOPSO comparison.

WHAT CHANGED vs v1
-------------------
1. **Three objectives** (f1 energy, f2 max-cluster-distance, f3 packet-loss)
   are now evaluated and collected at every round by MOomf.py, nsga2.py, and
   pso_leach.py.  Improved LEACH has no Pareto search, so it has no front.

2. **Multi-objective quality indicators** are computed here, after each run,
   using pymoo's built-in HV and IGD classes:

   * Hypervolume (HV)  — measures the volume of objective space dominated by
     the Pareto front.  Higher is better.  A common reference point is the
     nadir vector + 10%, estimated from all runs jointly after all seeds are
     complete (two-pass approach).  For the per-seed records written during
     the run we store the raw front arrays and compute HV/IGD in a second
     pass once the global nadir / reference front are known.

   * IGD (Inverted Generational Distance)  — average distance from each point
     on the approximated Pareto reference front to the nearest point on the
     algorithm's front.  Lower is better.  The reference front (PF_ref) is
     the non-dominated set built from the union of all algorithm fronts across
     all seeds.

   * Spread (Δ-indicator)  — measures how evenly solutions are distributed
     along the front.  Lower is better.  Defined as the mean distance between
     consecutive solutions in objective space after sorting.

3. **New output files**
   results/
     raw/<algo>_seed<seed>.json        (unchanged + "pareto_fronts" field)
     pareto/<algo>_seed<seed>_pf.npy   (stacked Pareto objectives per run)
     mo_indicators.csv                 (HV, IGD, Spread per algo, mean±std)
     summary.csv                       (unchanged network metrics)

USAGE
-----
    python batch_runner.py --algorithms omf nsga pso improved_leach \\
                            --n-seeds 20 --base-seed 1 --n-rounds 2500 \\
                            --workers 8 --out results

Already-completed runs are skipped automatically (incremental restart).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ImprovedLEACH import LeachParams, run_improved_leach
from MOomf import OMFParams, run_omf_leach
from nsga2 import NSGA2Params, run_nsga2_leach
from pso_leach import PSOParams, run_pso_leach

# ---------------------------------------------------------------------------
# Algorithm registry
# ---------------------------------------------------------------------------

ALGO_RUNNERS = {
    "improved_leach": (run_improved_leach, LeachParams),
    "omf":            (run_omf_leach,       OMFParams),
    "nsga":           (run_nsga2_leach,     NSGA2Params),
    "pso":            (run_pso_leach,       PSOParams),
}

# Algorithms that produce a Pareto front at each round.
MO_ALGOS = {"omf", "nsga", "pso"}

# Network-level metrics summarised in summary.csv (unchanged from v1).
SUMMARY_METRICS = [
    "FND", "HND", "LND",
    "total_packets", "total_packet_loss",
    "packet_loss_ratio", "avg_energy_consumed_per_round",
    "elapsed_seconds",
]

# Number of objectives (f1 energy, f2 cluster-distance, f3 packet-loss).
N_OBJ = 3


# ---------------------------------------------------------------------------
# Pareto utilities
# ---------------------------------------------------------------------------

def _is_dominated(point: np.ndarray, front: np.ndarray) -> bool:
    """Return True if *point* is dominated by any row in *front*."""
    return bool(np.any(np.all(front <= point, axis=1) & np.any(front < point, axis=1)))


def _non_dominated_front(points: np.ndarray) -> np.ndarray:
    """Return the non-dominated subset of *points* (shape N×M)."""
    if len(points) == 0:
        return points
    nd_mask = np.ones(len(points), dtype=bool)
    for i, p in enumerate(points):
        if not nd_mask[i]:
            continue
        others = points[nd_mask]
        others_no_i = others[others is not p]   # quick ref check may fail
        # Proper mask approach:
        idx = np.where(nd_mask)[0]
        for j_pos, j in enumerate(idx):
            if j == i:
                continue
            if np.all(points[j] <= p) and np.any(points[j] < p):
                nd_mask[i] = False
                break
    return points[nd_mask]


def _aggregate_pareto_front(fronts_per_round: List[np.ndarray]) -> np.ndarray:
    """Stack all per-round fronts and return the global non-dominated set."""
    non_empty = [f for f in fronts_per_round if len(f) > 0]
    if not non_empty:
        return np.empty((0, N_OBJ), dtype=float)
    stacked = np.vstack(non_empty)
    return _non_dominated_front(stacked)


# ---------------------------------------------------------------------------
# Hypervolume (using pymoo)
# ---------------------------------------------------------------------------

def _compute_hv(front: np.ndarray, ref_point: np.ndarray) -> float:
    """Compute hypervolume of *front* w.r.t. *ref_point* using pymoo."""
    try:
        from pymoo.indicators.hv import HV
    except ImportError:
        return float("nan")

    if len(front) == 0:
        return 0.0

    # pymoo HV expects minimisation and ref_point to be strictly dominated.
    # Filter out any point that is not dominated by ref_point component-wise.
    dominated = np.all(front < ref_point, axis=1)
    front_filt = front[dominated]
    if len(front_filt) == 0:
        return 0.0

    ind = HV(ref_point=ref_point)
    return float(ind(front_filt))


# ---------------------------------------------------------------------------
# IGD  (using pymoo)
# ---------------------------------------------------------------------------

def _compute_igd(front: np.ndarray, pf_ref: np.ndarray) -> float:
    """Compute IGD of *front* against reference front *pf_ref*."""
    try:
        from pymoo.indicators.igd import IGD
    except ImportError:
        return float("nan")

    if len(front) == 0 or len(pf_ref) == 0:
        return float("nan")

    ind = IGD(pf=pf_ref)
    return float(ind(front))


# ---------------------------------------------------------------------------
# Spread / Δ-indicator
# ---------------------------------------------------------------------------

def _compute_spread(front: np.ndarray) -> float:
    """Compute Spread (Δ) for a 2-D or 3-D Pareto front.

    For a set of N non-dominated solutions, Spread is defined as:

        Δ = (d_f + d_l + Σ|d_i - d̄|) / (d_f + d_l + (N-1)·d̄)

    where d_i is the Euclidean distance between consecutive solutions (after
    sorting by the first objective), d̄ is the mean of those distances, and
    d_f, d_l are the Euclidean distances from the extreme solutions to the
    boundary solutions of the front.  Δ = 0 means perfect uniform spread.

    For fronts with fewer than 2 solutions the indicator is undefined (NaN).
    """
    if len(front) < 2:
        return float("nan")

    # Sort by first objective for a reproducible ordering.
    order = np.argsort(front[:, 0])
    sorted_f = front[order]

    dists = np.linalg.norm(np.diff(sorted_f, axis=0), axis=1)  # shape (N-1,)
    d_mean = float(np.mean(dists))

    if d_mean < 1e-15:
        return 0.0

    # Extreme distances: distance from boundary to first/last solution.
    # Here we use the boundary defined by the min/max of each objective.
    ideal = np.min(front, axis=0)
    nadir = np.max(front, axis=0)
    d_f = float(np.linalg.norm(sorted_f[0]  - ideal))
    d_l = float(np.linalg.norm(sorted_f[-1] - nadir))

    numerator   = d_f + d_l + float(np.sum(np.abs(dists - d_mean)))
    denominator = d_f + d_l + (len(front) - 1) * d_mean

    return float(numerator / max(denominator, 1e-15))


# ---------------------------------------------------------------------------
# GD (Generational Distance)
# ---------------------------------------------------------------------------

def _compute_gd(front: np.ndarray, pf_ref: np.ndarray) -> float:
    """Compute GD: mean distance from each point in *front* to *pf_ref*."""
    if len(front) == 0 or len(pf_ref) == 0:
        return float("nan")

    dists = np.array([
        np.min(np.linalg.norm(pf_ref - p, axis=1))
        for p in front
    ])
    return float(np.mean(dists))


# ---------------------------------------------------------------------------
# Worker: single (algorithm, seed) run
# ---------------------------------------------------------------------------

def _run_one(task: Tuple[str, int, int, Path]) -> Dict:
    """Run one (algorithm, seed) combination, persist JSON, return record.

    Executed in a worker process.  Must be a top-level function for pickling.
    """
    algo, seed, n_rounds, out_dir = task

    out_path    = out_dir / "raw"    / f"{algo}_seed{seed}.json"
    pareto_path = out_dir / "pareto" / f"{algo}_seed{seed}_pf.npy"

    # ── Skip if already computed ──────────────────────────────────────────
    if out_path.exists() and pareto_path.exists():
        with open(out_path, "r") as fh:
            return json.load(fh)

    # ── Run the simulation ────────────────────────────────────────────────
    runner, params_cls = ALGO_RUNNERS[algo]
    base_params = LeachParams(n_rounds=n_rounds)

    t0 = time.perf_counter()
    if algo == "improved_leach":
        sim_result = runner(params=base_params, seed=seed)
    else:
        sim_result = runner(params_cls(base=base_params), seed=seed)
    elapsed = time.perf_counter() - t0

    # ── Network-level metrics (unchanged from v1) ─────────────────────────
    total_packets = int(sim_result.get("total_packets", 0))
    total_loss    = int(sim_result.get("total_packet_loss", 0))
    generated     = total_packets + total_loss

    record: Dict = {
        "algorithm":  algo,
        "seed":       seed,
        "elapsed_seconds": elapsed,
        "FND":  sim_result.get("FND"),
        "HND":  sim_result.get("HND"),
        "LND":  sim_result.get("LND"),
        "total_packets":      total_packets,
        "total_packet_loss":  total_loss,
        "packet_loss_ratio":  (total_loss / generated) if generated > 0 else 0.0,
        "avg_energy_consumed_per_round": float(
            np.mean(sim_result.get("energy_consumed_per_round", [0.0]))
        ),
    }

    # ── Collect and persist the aggregate Pareto front ────────────────────
    # For MO algorithms: stack all per-round non-dominated sets and extract
    # the global non-dominated front for this seed.
    # For Improved LEACH: no Pareto search → save an empty array.
    pareto_path.parent.mkdir(parents=True, exist_ok=True)

    if algo in MO_ALGOS:
        fronts_per_round: List[np.ndarray] = sim_result.get(
            "pareto_fronts_per_round", []
        )
        agg_front = _aggregate_pareto_front(fronts_per_round)
    else:
        agg_front = np.empty((0, N_OBJ), dtype=float)

    np.save(str(pareto_path), agg_front)

    # Store basic shape info in the JSON record for quick inspection.
    record["pareto_front_size"] = int(len(agg_front))

    # ── Persist JSON record ───────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(record, fh, indent=2)

    return record


# ---------------------------------------------------------------------------
# Statistics helper
# ---------------------------------------------------------------------------

def _mean_std_ci95(values: List[float]) -> Tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(arr))
    std  = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    n    = len(arr)
    try:
        from scipy import stats as scipy_stats
        tcrit = float(scipy_stats.t.ppf(0.975, df=max(n - 1, 1)))
    except ImportError:
        tcrit = 1.96
    half_width = tcrit * std / math.sqrt(n) if n > 1 else 0.0
    return mean, std, half_width


# ---------------------------------------------------------------------------
# Build reference front (PF_ref) from all MO algorithm runs
# ---------------------------------------------------------------------------

def _build_reference_front(out_dir: Path, algorithms: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Union of all seed-level aggregate fronts → global non-dominated set,
    plus the component-wise max over the RAW (unfiltered) point pool.

    Returns (pf_ref, raw_max). `raw_max` is computed BEFORE non-domination
    filtering, over every point from every seed/algorithm -- this is what
    the HV reference point should be built from, not `pf_ref` alone. Using
    only the filtered pf_ref risks a reference point that does not actually
    dominate every point in every individual seed's front: a point removed
    during filtering (because some other seed/algorithm's point dominated
    it) can still have a worse raw coordinate than anything left in pf_ref,
    silently causing that point to be dropped from its own seed's HV
    computation later on.

    This is the standard approach when the true Pareto front is not known
    analytically.  Only MO algorithms contribute to PF_ref.
    """
    all_points: List[np.ndarray] = []
    for algo in algorithms:
        if algo not in MO_ALGOS:
            continue
        for pf_file in sorted((out_dir / "pareto").glob(f"{algo}_seed*_pf.npy")):
            arr = np.load(str(pf_file))
            if len(arr) > 0:
                all_points.append(arr)

    if not all_points:
        return np.empty((0, N_OBJ), dtype=float), np.empty((0, N_OBJ), dtype=float)

    stacked = np.vstack(all_points)
    raw_max = np.max(stacked, axis=0)
    return _non_dominated_front(stacked), raw_max


# ---------------------------------------------------------------------------
# Compute multi-objective indicators for every (algo, seed)
# ---------------------------------------------------------------------------

def _compute_mo_indicators(
    out_dir: Path,
    algorithms: List[str],
    pf_ref: np.ndarray,
    ref_point: np.ndarray,
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """Return nested dict  algo → seed → {hv, igd, spread, gd}."""
    results: Dict[str, Dict[int, Dict[str, float]]] = {}

    for algo in algorithms:
        results[algo] = {}
        if algo not in MO_ALGOS:
            continue

        for pf_file in sorted((out_dir / "pareto").glob(f"{algo}_seed*_pf.npy")):
            # Extract seed from filename  <algo>_seed<N>_pf.npy
            stem  = pf_file.stem                   # e.g. "omf_seed3_pf"
            parts = stem.split("_seed")
            if len(parts) < 2:
                continue
            seed_str = parts[-1].replace("_pf", "")
            try:
                seed = int(seed_str)
            except ValueError:
                continue

            front = np.load(str(pf_file))

            hv     = _compute_hv(front, ref_point)
            igd    = _compute_igd(front, pf_ref)
            spread = _compute_spread(front)
            gd     = _compute_gd(front, pf_ref)

            results[algo][seed] = {
                "hv":     hv,
                "igd":    igd,
                "spread": spread,
                "gd":     gd,
            }

    return results


# ---------------------------------------------------------------------------
# Write summary CSVs
# ---------------------------------------------------------------------------

def _write_summary_csv(records: List[Dict], out_dir: Path) -> None:
    """Write summary.csv (network metrics, unchanged from v1)."""
    import csv

    by_algo: Dict[str, List[Dict]] = {}
    for r in records:
        by_algo.setdefault(r["algorithm"], []).append(r)

    summary_path = out_dir / "summary.csv"
    with open(summary_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["algorithm", "metric", "n_seeds", "mean", "std", "ci95_halfwidth"])
        for algo, recs in sorted(by_algo.items()):
            n_seeds = len(recs)
            for metric in SUMMARY_METRICS:
                values = [r[metric] for r in recs if r.get(metric) is not None]
                if not values:
                    continue
                mean, std, ci = _mean_std_ci95(values)
                writer.writerow(
                    [algo, metric, n_seeds, f"{mean:.6f}", f"{std:.6f}", f"{ci:.6f}"]
                )
    print(f"  Network summary  → {summary_path}")


def _write_mo_indicators_csv(
    mo_results: Dict[str, Dict[int, Dict[str, float]]],
    out_dir: Path,
    pf_ref_size: int,
) -> None:
    """Write mo_indicators.csv (HV, IGD, Spread, GD per algo)."""
    import csv

    csv_path = out_dir / "mo_indicators.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "algorithm", "metric", "n_seeds",
            "mean", "std", "ci95_halfwidth",
        ])
        for algo in sorted(mo_results.keys()):
            seed_data = mo_results[algo]
            if not seed_data:
                continue
            n_seeds = len(seed_data)
            for indicator in ("hv", "igd", "spread", "gd"):
                values = [v[indicator] for v in seed_data.values()]
                mean, std, ci = _mean_std_ci95(values)
                writer.writerow([
                    algo, indicator, n_seeds,
                    f"{mean:.6f}", f"{std:.6f}", f"{ci:.6f}",
                ])

    print(f"  MO indicators    → {csv_path}")
    print(f"  (PF_ref size = {pf_ref_size} non-dominated points)")


def _write_mo_indicators_per_seed_csv(
    mo_results: Dict[str, Dict[int, Dict[str, float]]],
    out_dir: Path,
) -> None:
    """Write mo_indicators_per_seed.csv (raw values for every algo×seed)."""
    import csv

    csv_path = out_dir / "mo_indicators_per_seed.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["algorithm", "seed", "hv", "igd", "spread", "gd"])
        for algo in sorted(mo_results.keys()):
            for seed, vals in sorted(mo_results[algo].items()):
                writer.writerow([
                    algo, seed,
                    f"{vals['hv']:.6f}",
                    f"{vals['igd']:.6f}",
                    f"{vals['spread']:.6f}",
                    f"{vals['gd']:.6f}",
                ])
    print(f"  Per-seed MO data → {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--algorithms", nargs="+",
        default=["improved_leach", "omf", "nsga", "pso"],
        choices=list(ALGO_RUNNERS.keys()),
    )
    parser.add_argument("--n-seeds",   type=int, default=20,
                        help="Number of independent seeds per algorithm.")
    parser.add_argument("--base-seed", type=int, default=1)
    parser.add_argument("--n-rounds",  type=int, default=2500)
    parser.add_argument(
        "--workers", type=int, default=max(1, cpu_count() - 1),
        help="Parallel worker processes.  Default: all cores minus one.",
    )
    parser.add_argument("--out", type=str, default="results")
    # Reference-point multiplier for HV (nadir × factor).
    parser.add_argument(
        "--hv-ref-factor", type=float, default=1.1,
        help="Nadir-point multiplier for the HV reference point (default 1.1).",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    (out_dir / "pareto").mkdir(parents=True, exist_ok=True)

    # ── Build task list ───────────────────────────────────────────────────
    tasks: List[Tuple[str, int, int, Path]] = [
        (algo, args.base_seed + i, args.n_rounds, out_dir)
        for algo in args.algorithms
        for i in range(args.n_seeds)
    ]

    print(
        f"\nLaunching {len(tasks)} runs "
        f"({len(args.algorithms)} algorithms × {args.n_seeds} seeds) "
        f"across {args.workers} worker processes..."
    )
    print("(Completed runs found on disk are skipped automatically.)\n")

    # ── Phase 1: run all (algo, seed) combinations ────────────────────────
    t0 = time.perf_counter()
    records: List[Dict] = []

    with Pool(processes=args.workers) as pool:
        for i, record in enumerate(pool.imap_unordered(_run_one, tasks), start=1):
            elapsed = record.get("elapsed_seconds", 0.0)
            pf_size = record.get("pareto_front_size", "-")
            print(
                f"  [{i:3d}/{len(tasks)}] {record['algorithm']:15s} "
                f"seed={record['seed']:3d}  ({elapsed:7.1f}s)  "
                f"FND={record['FND']}  |PF|={pf_size}"
            )
            records.append(record)

    total_time = time.perf_counter() - t0
    print(f"\nAll runs completed in {total_time / 60:.1f} min "
          f"(workers={args.workers}).")

    # ── Phase 2: build reference front and HV reference point ─────────────
    print("\nBuilding PF_ref (union of all seed-level Pareto fronts)...")
    pf_ref, raw_max = _build_reference_front(out_dir, args.algorithms)

    if len(pf_ref) == 0:
        print("  WARNING: PF_ref is empty.  No MO algorithm produced a front.")
        nadir_ref = np.ones(N_OBJ, dtype=float) * args.hv_ref_factor
    else:
        # Use raw_max (pre-filtering), not pf_ref's own max, so the
        # reference point is guaranteed to dominate every point from every
        # seed -- see _build_reference_front docstring.
        nadir_ref = raw_max * args.hv_ref_factor

    print(f"  PF_ref size = {len(pf_ref)} points")
    print(f"  HV ref point = {nadir_ref.tolist()}")

    # Save PF_ref for reproducibility.
    np.save(str(out_dir / "pf_ref.npy"), pf_ref)

    # ── Phase 3: compute HV, IGD, Spread, GD for every (algo, seed) ───────
    print("\nComputing multi-objective indicators (HV, IGD, Spread, GD)...")
    mo_results = _compute_mo_indicators(
        out_dir, args.algorithms, pf_ref, nadir_ref
    )

    # ── Phase 4: write output files ────────────────────────────────────────
    print("\nWriting output files:")
    _write_summary_csv(records, out_dir)
    _write_mo_indicators_csv(mo_results, out_dir, len(pf_ref))
    _write_mo_indicators_per_seed_csv(mo_results, out_dir)

    # ── Phase 5: print a quick comparison table ────────────────────────────
    print("\n" + "=" * 72)
    print(f"{'Algorithm':<18} {'HV ↑':>12} {'IGD ↓':>12} {'Spread ↓':>12} {'GD ↓':>12}")
    print("-" * 72)
    for algo in sorted(mo_results.keys()):
        seed_data = mo_results[algo]
        if not seed_data:
            print(f"  {algo:<16} {'(no Pareto front)':>50}")
            continue
        hv_vals  = [v["hv"]     for v in seed_data.values()]
        igd_vals = [v["igd"]    for v in seed_data.values()]
        spr_vals = [v["spread"] for v in seed_data.values()]
        gd_vals  = [v["gd"]     for v in seed_data.values()]

        def fmt(vals: List[float]) -> str:
            clean = [v for v in vals if not math.isnan(v)]
            if not clean:
                return "  nan  "
            return f"{np.mean(clean):.4f}±{np.std(clean, ddof=1):.4f}"

        print(f"  {algo:<16} {fmt(hv_vals):>14} {fmt(igd_vals):>14} "
              f"{fmt(spr_vals):>14} {fmt(gd_vals):>14}")
    print("=" * 72)
    print("\nDone.\n")


if __name__ == "__main__":
    main()
