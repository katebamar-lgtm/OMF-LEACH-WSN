from __future__ import annotations

"""Scalability campaign for OMF-LEACH.

Typical workflow:
  1) Reuse seeds 1..10 from the original 80-node Results.
  2) Run only 100 and 150 nodes.
  3) Build one combined CSV for 80/100/150.
"""

import argparse
import csv
import json
import math
import multiprocessing as mp
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from ImprovedLEACH import LeachParams, generate_topology
from MOomf import OMFParams, run_omf_leach
from nsga2 import NSGA2Params, run_nsga2_leach
from pso_leach import PSOParams, run_pso_leach

ALGORITHMS = ("omf", "nsga", "pso")
ALGORITHM_LABELS = {"omf": "OMF", "nsga": "NSGA-II", "pso": "MOPSO"}


def mean_std_ci95(values: Sequence[float]) -> Tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    if len(arr) <= 1:
        return mean, std, 0.0
    try:
        from scipy import stats
        tcrit = float(stats.t.ppf(0.975, df=len(arr) - 1))
    except ImportError:
        tcrit = 1.96
    return mean, std, float(tcrit * std / math.sqrt(len(arr)))


def _run_algorithm(algorithm: str, base_params: LeachParams, topo: Dict, seed: int):
    if algorithm == "omf":
        return run_omf_leach(omf=OMFParams(base=base_params), topo=topo, seed=seed)
    if algorithm == "nsga":
        return run_nsga2_leach(nsga=NSGA2Params(base=base_params), topo=topo, seed=seed)
    if algorithm == "pso":
        return run_pso_leach(pso=PSOParams(base=base_params), topo=topo, seed=seed)
    raise ValueError(f"Unknown algorithm: {algorithm}")


def _run_one(task: Tuple[int, str, int, int, str]) -> Dict:
    n_nodes, algorithm, seed, n_rounds, out_root = task
    scenario_dir = Path(out_root) / "raw" / f"{n_nodes}nodes"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    out_path = scenario_dir / f"{algorithm}_seed{seed}.json"

    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    base_params = LeachParams(
        n_nodes=n_nodes,
        area_w=100.0,
        area_h=100.0,
        bs_x=50.0,
        bs_y=150.0,
        p=0.05,
        n_rounds=n_rounds,
    )
    topo = generate_topology(base_params, seed)

    t0 = time.perf_counter()
    result = _run_algorithm(algorithm, base_params, topo, seed)
    elapsed = time.perf_counter() - t0

    total_packets = int(result.get("total_packets", 0))
    total_loss = int(result.get("total_packet_loss", 0))
    generated = total_packets + total_loss
    energy_values = np.asarray(result.get("energy_consumed_per_round", []), dtype=float)

    record = {
        "n_nodes": int(n_nodes),
        "algorithm": algorithm,
        "algorithm_label": ALGORITHM_LABELS[algorithm],
        "seed": int(seed),
        "n_rounds_cap": int(n_rounds),
        "FND": result.get("FND"),
        "HND": result.get("HND"),
        "LND": result.get("LND"),
        "total_packets": total_packets,
        "total_packet_loss": total_loss,
        "packet_loss_ratio": (total_loss / generated) if generated else 0.0,
        "avg_energy_consumed_per_round": float(np.mean(energy_values)) if energy_values.size else 0.0,
        "elapsed_seconds": float(elapsed),
        "lnd_reached_cap": bool(result.get("LND") == n_rounds),
    }
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
    return record


def normalize_existing_record(data: Dict, n_nodes: int, algorithm: str, seed: int) -> Dict:
    """Adapt an original results/raw JSON to the scalability schema."""
    total_packets = int(data.get("total_packets", 0))
    total_loss = int(data.get("total_packet_loss", 0))
    generated = total_packets + total_loss
    return {
        "n_nodes": int(n_nodes),
        "algorithm": algorithm,
        "algorithm_label": ALGORITHM_LABELS[algorithm],
        "seed": int(seed),
        "n_rounds_cap": int(data.get("n_rounds_cap", 2500)),
        "FND": data.get("FND"),
        "HND": data.get("HND"),
        "LND": data.get("LND"),
        "total_packets": total_packets,
        "total_packet_loss": total_loss,
        "packet_loss_ratio": float(data.get("packet_loss_ratio", total_loss / generated if generated else 0.0)),
        "avg_energy_consumed_per_round": float(data.get("avg_energy_consumed_per_round", 0.0)),
        "elapsed_seconds": float(data.get("elapsed_seconds", 0.0)),
        "lnd_reached_cap": bool(data.get("LND") == data.get("n_rounds_cap", 2500)),
        "source": "original_results",
    }


def reuse_original_results(existing_root: Path, out_root: Path, n_nodes: int, seeds: Sequence[int]) -> List[Dict]:
    """Copy/reformat selected original raw results into results_scalability/raw."""
    records = []
    for algorithm in ALGORITHMS:
        for seed in seeds:
            src = existing_root / "raw" / f"{algorithm}_seed{seed}.json"
            if not src.exists():
                raise FileNotFoundError(f"Missing original result: {src}")
            with src.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            record = normalize_existing_record(data, n_nodes, algorithm, seed)
            dst_dir = out_root / "raw" / f"{n_nodes}nodes"
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / f"{algorithm}_seed{seed}.json"
            with dst.open("w", encoding="utf-8") as fh:
                json.dump(record, fh, indent=2)
            records.append(record)
    return records


def write_per_run(records: Iterable[Dict], out_path: Path) -> None:
    rows = sorted(records, key=lambda r: (int(r["n_nodes"]), str(r["algorithm"]), int(r["seed"])))
    fields = [
        "n_nodes", "algorithm", "algorithm_label", "seed", "n_rounds_cap",
        "FND", "HND", "LND", "total_packets", "total_packet_loss",
        "packet_loss_ratio", "avg_energy_consumed_per_round", "elapsed_seconds",
        "lnd_reached_cap", "source",
    ]
    for row in rows:
        row.setdefault("source", "new_campaign")
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(records: Sequence[Dict], out_path: Path) -> None:
    metrics = ["FND", "HND", "LND", "avg_energy_consumed_per_round", "packet_loss_ratio", "elapsed_seconds"]
    grouped: Dict[Tuple[int, str], List[Dict]] = {}
    for r in records:
        grouped.setdefault((int(r["n_nodes"]), str(r["algorithm"])), []).append(r)

    rows = []
    for (n_nodes, algorithm), group in sorted(grouped.items()):
        row = {
            "n_nodes": n_nodes,
            "algorithm": algorithm,
            "algorithm_label": ALGORITHM_LABELS[algorithm],
            "n_seeds": len(group),
            "lnd_capped_count": sum(bool(r["lnd_reached_cap"]) for r in group),
        }
        for metric in metrics:
            vals = [float(r[metric]) for r in group if r.get(metric) is not None]
            mean, std, ci = mean_std_ci95(vals)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95"] = ci
        rows.append(row)

    fields = ["n_nodes", "algorithm", "algorithm_label", "n_seeds", "lnd_capped_count"]
    for metric in metrics:
        fields += [f"{metric}_mean", f"{metric}_std", f"{metric}_ci95"]

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_pivot(records: Sequence[Dict], out_path: Path) -> None:
    metrics = ["FND", "HND", "LND", "avg_energy_consumed_per_round", "elapsed_seconds"]
    grouped: Dict[Tuple[int, str], List[Dict]] = {}
    for r in records:
        grouped.setdefault((int(r["n_nodes"]), str(r["algorithm"])), []).append(r)

    fields = ["n_nodes", "algorithm"]
    for metric in metrics:
        fields += [f"{metric}_mean", f"{metric}_std"]

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for n_nodes in sorted({int(r["n_nodes"]) for r in records}):
            for algorithm in ALGORITHMS:
                group = grouped.get((n_nodes, algorithm), [])
                if not group:
                    continue
                row = {"n_nodes": n_nodes, "algorithm": algorithm}
                for metric in metrics:
                    vals = [float(r[metric]) for r in group if r.get(metric) is not None]
                    mean, std, _ = mean_std_ci95(vals)
                    row[f"{metric}_mean"] = mean
                    row[f"{metric}_std"] = std
                writer.writerow(row)


def load_all_records(out_root: Path) -> List[Dict]:
    records = []
    for p in sorted(out_root.glob("raw/*nodes/*.json")):
        with p.open("r", encoding="utf-8") as fh:
            records.append(json.load(fh))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Run/reuse the OMF-LEACH scalability campaign.")
    parser.add_argument("--nodes", nargs="+", type=int, default=[80, 100, 150])
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=1)
    parser.add_argument("--n-rounds", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--out", type=str, default="results_scalability")
    parser.add_argument("--reuse-80", action="store_true",
                        help="Reuse seeds from original Results/raw for the 80-node case instead of rerunning them.")
    parser.add_argument("--existing-results", type=str, default="results",
                        help="Original Results directory containing raw/*.json.")
    args = parser.parse_args()

    nodes = sorted(set(args.nodes))
    if any(n < 2 for n in nodes):
        raise ValueError("Each network size must be at least 2 nodes.")
    if args.n_seeds < 1 or args.workers < 1:
        raise ValueError("--n-seeds and --workers must be >= 1")

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    imported = []

    if args.reuse_80 and 80 in nodes:
        imported = reuse_original_results(Path(args.existing_results), out_root, 80,
                                          range(args.base_seed, args.base_seed + args.n_seeds))
        nodes = [n for n in nodes if n != 80]
        print(f"Reused {len(imported)} existing 80-node records from {args.existing_results!r}.")

    tasks = [
        (n_nodes, algorithm, args.base_seed + i, args.n_rounds, str(out_root))
        for n_nodes in nodes
        for algorithm in ALGORITHMS
        for i in range(args.n_seeds)
    ]

    print("=" * 78)
    print("OMF-LEACH scalability campaign")
    print(f"New nodes   : {nodes if nodes else 'none'}")
    print("Final set   : 80, 100, 150 (when 80 is reused)")
    print(f"Algorithms  : {', '.join(ALGORITHMS)}")
    print(f"Seeds       : {args.n_seeds} ({args.base_seed}..{args.base_seed + args.n_seeds - 1})")
    print(f"Round cap   : {args.n_rounds}")
    print(f"Workers     : {args.workers}")
    print("Area        : 100 m × 100 m")
    print("BS          : (50, 150)")
    print("P           : 0.05")
    print(f"New runs    : {len(tasks)}")
    print("=" * 78)

    t0 = time.perf_counter()
    if tasks:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.workers) as pool:
            for index, record in enumerate(pool.imap_unordered(_run_one, tasks), start=1):
                print(
                    f"[{index:3d}/{len(tasks)}] N={record['n_nodes']:3d} "
                    f"{record['algorithm_label']:<7s} seed={record['seed']:2d} "
                    f"FND={str(record['FND']):>5s} HND={str(record['HND']):>5s} "
                    f"LND={str(record['LND']):>5s} time={record['elapsed_seconds']:8.1f}s"
                )

    records = load_all_records(out_root)
    write_per_run(records, out_root / "scalability_per_run.csv")
    write_summary(records, out_root / "scalability_summary.csv")
    write_pivot(records, out_root / "scalability_summary_pivot.csv")

    expected = set((n, a, s) for n in (80, 100, 150) for a in ALGORITHMS
                   for s in range(args.base_seed, args.base_seed + args.n_seeds))
    actual = set((int(r["n_nodes"]), str(r["algorithm"]), int(r["seed"])) for r in records)
    missing = sorted(expected - actual)
    capped = [r for r in records if r.get("lnd_reached_cap")]

    print("\n" + "=" * 78)
    print(f"Wall time this invocation : {(time.perf_counter() - t0) / 60:.1f} min")
    print(f"Records collected        : {len(records)} / {len(expected)}")
    print(f"Per-run CSV              : {out_root / 'scalability_per_run.csv'}")
    print(f"Summary CSV              : {out_root / 'scalability_summary.csv'}")
    print(f"Manuscript pivot         : {out_root / 'scalability_summary_pivot.csv'}")
    if missing:
        print("WARNING missing records:")
        for item in missing:
            print("  ", item)
    if capped:
        print(f"WARNING: {len(capped)} run(s) reached the round cap. Inspect the per-run CSV.")
    else:
        print("No run reached the round cap among collected records.")
    print("=" * 78)


if __name__ == "__main__":
    main()
