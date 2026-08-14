"""
MOomf.py
========
The genuine Optimization by Morphological Filters (OMF) metaheuristic,
adapted to WSN cluster-head selection, as described in Section 3.6 of the
article (filters / neighborhood-in-width / neighborhood-in-depth / random
walk). This is NOT NSGA-II under a different name.

HISTORY / IMPORTANT NOTE
-------------------------
A previous revision of this file accidentally replaced the OMF search
procedure with a second, differently-configured instance of NSGA-II (via
pymoo), while keeping the "OMF" name and parameter class. That version must
never be used to produce results reported as "OMF" in the article, since it
was not testing the algorithm the article describes. This file restores the
original filter/neighborhood-based OMF logic (validated in the prior
performance audit, including the Fi_obj caching optimization), and adds:

1. Support for the 3-objective formulation (energy, max intra-cluster
   distance, packet loss) now returned by `full_round_objectives` in
   CustomLEACH.py (Eq. 4-6 in the article) -- previously only 2 objectives
   were wired through.
2. Per-round Pareto front collection (`pareto_fronts_per_round`), reusing the
   non-dominated set (ND / ND_obj) that OMF already maintains internally at
   every round -- needed downstream to compute HV / IGD / Spread / GD.
3. An OPTIONAL, OFF-BY-DEFAULT parallel evaluation mode for the
   neighborhood-in-width/depth candidates, built on pymoo's
   `StarmapParallelization` runner. See the "PARALLEL EVALUATION" section
   below for an honest discussion of when this is (and is not) expected to
   help.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from CustomLEACH import (
    _simulate_one_round,
    decode_particle,
    full_round_objectives,
    merge_packet_stats,
    new_packet_stats,
)
from ImprovedLEACH import LeachParams, generate_topology, _d0
from solution_selection import select_solution_index

N_OBJ = 3  # energy, max intra-cluster distance, packet loss (Eq. 4-6)


@dataclass
class OMFParams:
    """Parameters of the real OMF algorithm (filters / neighbors / random walk)."""
    base: LeachParams = field(default_factory=LeachParams)

    NF: int = 10       # number of filters
    NN: int = 6        # number of neighbors evaluated per filter, per iteration
    IT: int = 10       # outer iterations
    R: float = 0.7     # probability of a neighborhood-in-width move (vs in-depth)

    FS_init: float = 0.5
    FS_decay: float = 0.5
    selection_mode: str = "knee_point"

    # ------------------------------------------------------------------
    # OPTIONAL / EXPERIMENTAL: parallel evaluation of NN neighbor candidates
    # within a filter, using pymoo's StarmapParallelization runner.
    #
    # HONEST CAVEAT (read before enabling): after the CustomLEACH.py
    # performance patch, a single candidate evaluation (full_round_objectives)
    # is already fast (sub-millisecond to a few ms). Process-based
    # parallelism has per-task dispatch/serialization overhead of a similar
    # order of magnitude, so the expected net benefit at this granularity is
    # small and possibly negative -- especially when batch_runner.py is
    # already parallelizing across seeds at the outer level (nested
    # parallelism can cause oversubscription/contention rather than speedup).
    # This flag is OFF by default. If you want to try it, benchmark a single
    # run with and without it on your own machine before trusting the result
    # -- do not assume it helps without measuring.
    # ------------------------------------------------------------------
    # EMPIRICAL RESULT (measured during integration testing, single-core
    # sandbox, 30 rounds): parallel_workers=2 was SLOWER than sequential
    # (16.6s vs 9.3s) -- confirming the caveat above. Process-pool dispatch
    # overhead outweighed the per-task compute time at this granularity.
    # Benchmark on your own (multi-core) machine before enabling this for a
    # real run; do not assume it will help just because more cores are
    # available -- the bottleneck here is task granularity, not core count.
    parallel_workers: int = 0  # 0 = disabled (sequential, default, recommended)


def evaluate_omf_objectives(omf: OMFParams, topo: Dict, E: np.ndarray, ch_idx: np.ndarray) -> np.ndarray:
    if len(ch_idx) == 0:
        return np.full(N_OBJ, 1e9, dtype=float)

    return full_round_objectives(
        params=omf.base,
        topo=topo,
        E=E,
        ch_idx=ch_idx,
        ds=getattr(omf, "Ds", None),
    )


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def update_pareto_front(ND: List[np.ndarray], ND_obj: List[np.ndarray], sol: np.ndarray, obj: np.ndarray):
    keep = []
    for s, o in zip(ND, ND_obj):
        if dominates(obj, o):
            continue
        if dominates(o, obj):
            return ND, ND_obj
        keep.append((s, o))

    keep.append((sol, obj))
    sols, objs = zip(*keep)
    return list(sols), list(objs)


def neighbor_width(alive_idx: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(alive_idx, size=k, replace=False)


def neighbor_depth(current: np.ndarray, alive_idx: np.ndarray, k: int, FS: float, rng: np.random.Generator) -> np.ndarray:
    new = current.copy()
    n_changes = max(1, int(FS * len(current)))

    for _ in range(n_changes):
        idx = rng.integers(0, len(new))
        new[idx] = rng.choice(alive_idx)

    if len(np.unique(new)) < k:
        missing = np.setdiff1d(alive_idx, np.unique(new), assume_unique=False)
        if len(missing) > 0:
            fill = rng.choice(missing, size=min(k - len(np.unique(new)), len(missing)), replace=False)
            new = np.concatenate([np.unique(new), fill])

    return np.unique(new)[:k]


# ============================================================================
# OPTIONAL parallel batch evaluation (pymoo StarmapParallelization runner)
# ============================================================================

def _evaluate_batch_sequential(omf: OMFParams, topo: Dict, E: np.ndarray, candidates: List[np.ndarray]) -> List[np.ndarray]:
    return [evaluate_omf_objectives(omf, topo, E, c) for c in candidates]


def _make_parallel_runner(n_workers: int):
    """Build a pymoo StarmapParallelization runner backed by a process pool.

    Returned alongside the pool itself so the caller can close it explicitly
    (the pool must be created once per run and reused across rounds/
    iterations, never recreated per evaluation -- recreating a process pool
    thousands of times per run would itself dominate runtime).
    """
    from multiprocessing import Pool
    from pymoo.core.problem import StarmapParallelization

    pool = Pool(processes=n_workers)
    runner = StarmapParallelization(pool.starmap)
    return runner, pool


def _eval_single_arg(args_tuple):
    """Adapter for pymoo's StarmapParallelization, which calls f(x) with a
    single positional argument per task (it wraps each item of the task list
    as [x] internally) rather than unpacking a tuple as *args. Must be a
    top-level function so it can be pickled for multiprocessing."""
    omf, topo, E, ch_idx = args_tuple
    return evaluate_omf_objectives(omf, topo, E, ch_idx)


def _evaluate_batch(
    omf: OMFParams,
    topo: Dict,
    E: np.ndarray,
    candidates: List[np.ndarray],
    runner=None,
) -> List[np.ndarray]:
    """Evaluate a batch of candidate CH-sets, in parallel if a runner is given."""
    if runner is None or len(candidates) <= 1:
        return _evaluate_batch_sequential(omf, topo, E, candidates)

    results = runner(_eval_single_arg, [(omf, topo, E, c) for c in candidates])
    return list(results)


# ============================================================================
# CORE OMF SEARCH (unchanged algorithmic logic; validated Fi_obj-caching
# performance patch preserved)
# ============================================================================

def optimize_omf_pareto_front(
    omf: OMFParams,
    topo: Dict,
    E: np.ndarray,
    alive: np.ndarray,
    rng: np.random.Generator,
    runner=None,
) -> Tuple[List[np.ndarray], np.ndarray]:
    alive_idx = np.where(alive)[0]
    n_alive = len(alive_idx)

    if n_alive == 0:
        return [], np.empty((0, N_OBJ))

    k = min(int(max(1, round(omf.base.p * n_alive))), n_alive)

    filters = [
        rng.choice(alive_idx, size=k, replace=False)
        for _ in range(omf.NF)
    ]
    filter_sizes = [omf.FS_init for _ in range(omf.NF)]

    ND, ND_obj = [], []

    # Initial evaluation of the NF filters (optionally parallel).
    init_objs = _evaluate_batch(omf, topo, E, filters, runner)
    filter_objs: List[np.ndarray] = []
    for f, obj in zip(filters, init_objs):
        filter_objs.append(obj)
        ND, ND_obj = update_pareto_front(ND, ND_obj, f, obj)

    for _ in range(omf.IT):
        for i in range(len(filters)):
            Fi = filters[i]
            FS = filter_sizes[i]
            Fi_obj = filter_objs[i]  # cached: not recomputed (see class docstring)
            improved = False

            # Generate all NN neighbor candidates for this filter up front,
            # then evaluate them as a batch (sequential or parallel).
            candidates = []
            for _ in range(omf.NN):
                if rng.random() < omf.R:
                    Vij = neighbor_width(alive_idx, k, rng)
                else:
                    Vij = neighbor_depth(Fi, alive_idx, k, FS, rng)
                candidates.append(Vij)

            cand_objs = _evaluate_batch(omf, topo, E, candidates, runner)

            for Vij, Vij_obj in zip(candidates, cand_objs):
                if dominates(Vij_obj, Fi_obj):
                    Fi = Vij
                    Fi_obj = Vij_obj
                    improved = True
                ND, ND_obj = update_pareto_front(ND, ND_obj, Vij, Vij_obj)

            if not improved:
                filter_sizes[i] *= omf.FS_decay

            filters[i] = Fi
            filter_objs[i] = Fi_obj

        if all(fs < 1e-3 for fs in filter_sizes):
            break

    return ND, np.array(ND_obj) if len(ND_obj) > 0 else np.empty((0, N_OBJ))


def run_omf_one_round(omf, topo, E, alive, rng, runner=None):
    pareto_solutions, pareto_objs = optimize_omf_pareto_front(
        omf, topo, E, alive, rng, runner=runner
    )

    if len(pareto_solutions) == 0:
        return None, pareto_objs

    best_idx = select_solution_index(pareto_objs, omf.selection_mode)
    return pareto_solutions[best_idx], pareto_objs


def run_omf_leach(
    omf: Optional[OMFParams] = None,
    topo: Optional[Dict] = None,
    seed: int = 42,
    stop_event: Optional[object] = None,
    collect_history: bool = False,
):
    if omf is None:
        omf = OMFParams()

    params = omf.base
    if topo is None:
        topo = generate_topology(params, seed)

    rng = np.random.default_rng(seed)

    n = topo["n"]
    E = np.full(n, params.e0)
    alive = np.ones(n, dtype=bool)

    alive_per_round = []
    total_energy_per_round = []
    energy_consumed_per_round = []
    energy_consumption_ratio_per_round = []
    data_generated_per_round = []
    data_delivered_to_bs_per_round = []
    packet_loss_per_round = []
    packet_loss_ratio_per_round = []
    FND = HND = LND = None
    half = n // 2

    packet_stats = new_packet_stats()
    packet_stats_per_round = {key: [] for key in packet_stats}
    history_per_round = []

    # Pareto fronts collected at each round for HV / IGD / Spread / GD.
    # OMF already maintains a non-dominated set (ND_obj) internally at every
    # round -- we simply persist it here instead of discarding it.
    pareto_fronts_per_round: List[np.ndarray] = []

    # Optional parallel evaluation runner (off by default -- see OMFParams).
    runner = None
    pool = None
    if omf.parallel_workers and omf.parallel_workers > 0:
        runner, pool = _make_parallel_runner(omf.parallel_workers)

    try:
        for r in range(params.n_rounds):
            if stop_event is not None and stop_event.is_set():
                raise RuntimeError("Simulation cancelled.")

            alive_idx = np.where(alive)[0]
            n_alive = len(alive_idx)

            alive_per_round.append(int(n_alive))
            total_energy_per_round.append(float(np.sum(E)))

            if n_alive == 0:
                if LND is None:
                    LND = r
                break

            best_ch, pareto_objs = run_omf_one_round(omf, topo, E, alive, rng, runner=runner)
            pareto_fronts_per_round.append(
                np.asarray(pareto_objs, dtype=float).copy() if len(pareto_objs) > 0
                else np.empty((0, N_OBJ), dtype=float)
            )

            if best_ch is None or len(best_ch) == 0:
                best_ch = rng.choice(alive_idx, size=1)

            K = len(best_ch)
            Ncl_max = max(1, int(round(n_alive / max(K, 1))))

            alive_before = alive.copy()
            energies_before = E.copy()

            round_result = _simulate_one_round(
                params,
                topo,
                E,
                alive,
                best_ch,
                Ds=_d0(params),
                Ncl_max=Ncl_max,
                collect_history=collect_history,
            )

            if collect_history:
                E, alive, data_delivered, round_packet_stats, round_details = round_result
            else:
                E, alive, data_delivered, round_packet_stats = round_result

            merge_packet_stats(packet_stats, round_packet_stats)

            for key, value in round_packet_stats.items():
                packet_stats_per_round[key].append(int(value))

            energy_before_total = float(np.sum(energies_before))
            round_energy_consumed = max(0.0, float(energy_before_total - np.sum(E)))
            round_energy_ratio = float(round_energy_consumed / max(energy_before_total, 1e-12))
            energy_consumed_per_round.append(round_energy_consumed)
            energy_consumption_ratio_per_round.append(round_energy_ratio)
            data_generated = int(round_packet_stats.get("data_generated", n_alive))
            packet_loss = int(round_packet_stats.get("packet_loss", max(0, data_generated - int(data_delivered))))
            packet_loss_ratio = float(packet_loss / max(data_generated, 1))
            data_generated_per_round.append(data_generated)
            data_delivered_to_bs_per_round.append(int(data_delivered))
            packet_loss_per_round.append(packet_loss)
            packet_loss_ratio_per_round.append(packet_loss_ratio)

            n_alive_after = int(np.count_nonzero(alive))

            if FND is None and n_alive_after < n:
                FND = r + 1
            if HND is None and n_alive_after <= half:
                HND = r + 1

            if collect_history:
                history_per_round.append({
                    "round": r + 1,
                    "ch_idx": np.asarray(best_ch, dtype=int).copy(),
                    "cluster_members": copy.deepcopy(round_details["cluster_members"]),
                    "abandoned_nodes": np.asarray(round_details["abandoned_nodes"], dtype=int).copy(),
                    "abandoned_paths": copy.deepcopy(round_details.get("abandoned_paths", {})),
                    "paths": copy.deepcopy(round_details["paths"]),
                    "alive_before": alive_before.copy(),
                    "energies_before": energies_before.copy(),
                    "alive": alive.copy(),
                    "energies": E.copy(),
                    "energy_consumed": round_energy_consumed,
                    "energy_consumption_ratio": round_energy_ratio,
                    "data_generated": data_generated,
                    "data_delivered_to_bs": int(data_delivered),
                    "packet_loss": packet_loss,
                    "packet_loss_ratio": packet_loss_ratio,
                    "packets": int(data_delivered),
                    "dead_this_round": np.asarray(round_details.get(
                        "dead_this_round",
                        np.where(alive_before & ~alive)[0],
                    ), dtype=int).copy(),
                    "packet_stats": dict(round_packet_stats),
                })

            if n_alive_after == 0:
                if LND is None:
                    LND = r + 1
                break
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    if FND is None:
        FND = params.n_rounds
    if HND is None:
        HND = params.n_rounds
    if LND is None:
        LND = params.n_rounds

    result = {
        "alive_per_round": alive_per_round,
        "total_energy_per_round": total_energy_per_round,
        "energy_consumed_per_round": energy_consumed_per_round,
        "energy_consumption_ratio_per_round": energy_consumption_ratio_per_round,
        "data_generated_per_round": data_generated_per_round,
        "data_delivered_to_bs_per_round": data_delivered_to_bs_per_round,
        "packet_loss_per_round": packet_loss_per_round,
        "packet_loss_ratio_per_round": packet_loss_ratio_per_round,
        "packets_delivered_per_round": data_delivered_to_bs_per_round,
        "FND": FND,
        "HND": HND,
        "LND": LND,
        "total_packets": sum(data_delivered_to_bs_per_round),
        "total_packet_loss": sum(packet_loss_per_round),
        "packet_stats": packet_stats,
        "packet_stats_per_round": packet_stats_per_round,
        "selection_mode": omf.selection_mode,
        "pareto_fronts_per_round": pareto_fronts_per_round,
    }

    if collect_history:
        result["history"] = {
            "algo": "omf",
            "seed": seed,
            "topo": copy.deepcopy(topo),
            "params": params,
            "selection_mode": omf.selection_mode,
            "rounds": history_per_round,
        }

    return result


if __name__ == "__main__":
    print("Running OMF (genuine filter/neighborhood search, 3 objectives)...")
    start_time = time.perf_counter()

    params = OMFParams()
    result = run_omf_leach(params, seed=42)

    end_time = time.perf_counter()
    runtime = end_time - start_time

    print("\n===== RESULTS =====")
    print("FND:", result["FND"])
    print("HND:", result["HND"])
    print("LND:", result["LND"])
    print("Total packets:", result["total_packets"])
    print("Total packet loss:", result["total_packet_loss"])
    print(f"\nRuntime: {runtime:.4f} seconds")
