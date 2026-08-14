from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ImprovedLEACH import LeachParams, generate_topology, run_improved_leach
from MOomf import OMFParams, run_omf_leach
from nsga2 import NSGA2Params, run_nsga2_leach
from pso_leach import PSOParams, run_pso_leach
from solution_selection import normalize_selection_mode


class SimulationCancelledError(RuntimeError):
    pass


def _pad(arr: List[float], n: int, pad: float = 0.0) -> np.ndarray:
    out = np.full(n, pad, dtype=float)
    m = min(len(arr), n)
    if m > 0:
        out[:m] = np.asarray(arr[:m], dtype=float)
    return out


def _average_results(results: List[Dict], n_rounds: int) -> Dict:
    alive_mat, energy_mat, consumed_mat, consumption_ratio_mat = [], [], [], []
    generated_mat, delivered_mat, loss_mat, loss_ratio_mat = [], [], [], []
    fnds, hnds, lnds = [], [], []
    packet_stats_totals = {}
    packet_stats_per_round = {}

    for r in results:
        alive_mat.append(_pad(r["alive_per_round"], n_rounds, 0.0))
        energy_mat.append(_pad(r["total_energy_per_round"], n_rounds, 0.0))
        consumed_mat.append(_pad(r.get("energy_consumed_per_round", []), n_rounds, 0.0))
        consumption_ratio_mat.append(_pad(r.get("energy_consumption_ratio_per_round", []), n_rounds, 0.0))
        generated_mat.append(_pad(r.get("data_generated_per_round", []), n_rounds, 0.0))
        delivered_mat.append(_pad(r.get("data_delivered_to_bs_per_round", r.get("packets_delivered_per_round", [])), n_rounds, 0.0))
        loss_mat.append(_pad(r.get("packet_loss_per_round", []), n_rounds, 0.0))
        loss_ratio_mat.append(_pad(r.get("packet_loss_ratio_per_round", []), n_rounds, 0.0))
        if r["FND"] is not None:
            fnds.append(r["FND"])
        if r["HND"] is not None:
            hnds.append(r["HND"])
        if r["LND"] is not None:
            lnds.append(r["LND"])
        stats = r.get("packet_stats", {})
        for key, value in stats.items():
            packet_stats_totals.setdefault(key, []).append(float(value))
        stats_round = r.get("packet_stats_per_round", {})
        for key, values in stats_round.items():
            packet_stats_per_round.setdefault(key, []).append(
                _pad(values, n_rounds, 0.0)
            )

    alive_mean = np.mean(np.vstack(alive_mat), axis=0)
    energy_mean = np.mean(np.vstack(energy_mat), axis=0)
    consumed_mean = np.mean(np.vstack(consumed_mat), axis=0)
    consumption_ratio_mean = np.mean(np.vstack(consumption_ratio_mat), axis=0)
    generated_mean = np.mean(np.vstack(generated_mat), axis=0)
    delivered_mean = np.mean(np.vstack(delivered_mat), axis=0)
    loss_mean = np.mean(np.vstack(loss_mat), axis=0)
    loss_ratio_mean = np.mean(np.vstack(loss_ratio_mat), axis=0)

    avg_fnd = int(round(float(np.mean(fnds)))) if fnds else None
    avg_hnd = int(round(float(np.mean(hnds)))) if hnds else None
    avg_lnd = int(round(float(np.mean(lnds)))) if lnds else None

    def at(arr, idx):
        if idx is None:
            return None
        return float(arr[min(idx, len(arr) - 1)])

    return {
        "alive_per_round": alive_mean.tolist(),
        "total_energy_per_round": energy_mean.tolist(),
        "energy_consumed_per_round": consumed_mean.tolist(),
        "energy_consumption_ratio_per_round": consumption_ratio_mean.tolist(),
        "data_generated_per_round": generated_mean.tolist(),
        "data_delivered_to_bs_per_round": delivered_mean.tolist(),
        "packet_loss_per_round": loss_mean.tolist(),
        "packet_loss_ratio_per_round": loss_ratio_mean.tolist(),
        "packets_delivered_per_round": delivered_mean.tolist(),
        "FND": avg_fnd,
        "HND": avg_hnd,
        "LND": avg_lnd,
        "alive_at_FND": int(at(alive_mean, avg_fnd)) if avg_fnd is not None else None,
        "alive_at_HND": int(at(alive_mean, avg_hnd)) if avg_hnd is not None else None,
        "alive_at_LND": int(at(alive_mean, avg_lnd)) if avg_lnd is not None else None,
        "energy_at_FND": at(energy_mean, avg_fnd),
        "energy_at_HND": at(energy_mean, avg_hnd),
        "energy_at_LND": at(energy_mean, avg_lnd),
        "avg_packet_loss_ratio": float(np.mean(loss_ratio_mean[generated_mean > 0])) if np.any(generated_mean > 0) else 0.0,
        "total_packets_delivered": int(np.sum(delivered_mean)),
        "total_packet_loss": int(np.sum(loss_mean)),
        "dead_per_round": [],
        "packet_stats": {
            key: int(round(float(np.mean(values))))
            for key, values in packet_stats_totals.items()
        },
        "packet_stats_per_round": {
            key: np.mean(np.vstack(values), axis=0).tolist()
            for key, values in packet_stats_per_round.items()
        },
    }


def run_averaged_leach_vs_pso(
    n_runs: int = 20,
    base_seed: int = 42,
    n_rounds_cap: int = 2500,
) -> Tuple[Dict, Dict]:
    leach_runs: List[Dict] = []
    pso_runs: List[Dict] = []

    base_params = LeachParams(n_rounds=n_rounds_cap)
    pso_params = PSOParams(base=LeachParams(n_rounds=n_rounds_cap))

    for i in range(n_runs):
        seed = base_seed + i
        topo = generate_topology(base_params, seed)

        leach_runs.append(run_improved_leach(
            params=base_params,
            topo=topo,
            seed=seed,
        ))

        pso_runs.append(run_pso_leach(
            pso=pso_params,
            topo=topo,
            seed=seed,
        ))

    avg_leach = _average_results(leach_runs, n_rounds_cap)
    avg_pso = _average_results(pso_runs, n_rounds_cap)

    return avg_leach, avg_pso


def run_averaged_leach_vs_pso_vs_nsga(
    n_runs: int = 20,
    base_seed: int = 1,
    n_rounds_cap: int = 2500,
) -> Tuple[Dict, Dict, Dict]:
    leach_runs: List[Dict] = []
    pso_runs: List[Dict] = []
    nsga_runs: List[Dict] = []

    base_params = LeachParams(n_rounds=n_rounds_cap)
    pso_params = PSOParams(base=LeachParams(n_rounds=n_rounds_cap))
    nsga_params = NSGA2Params(base=LeachParams(n_rounds=n_rounds_cap))

    for i in range(n_runs):
        seed = base_seed + i
        topo = generate_topology(base_params, seed)

        leach_runs.append(run_improved_leach(
            params=base_params,
            topo=topo,
            seed=seed,
        ))

        pso_runs.append(run_pso_leach(
            pso=pso_params,
            topo=topo,
            seed=seed,
        ))

        nsga_runs.append(run_nsga2_leach(
            nsga=nsga_params,
            topo=topo,
            seed=seed,
        ))

    avg_leach = _average_results(leach_runs, n_rounds_cap)
    avg_pso = _average_results(pso_runs, n_rounds_cap)
    avg_nsga = _average_results(nsga_runs, n_rounds_cap)

    return avg_leach, avg_pso, avg_nsga


def run_averaged_algorithm(
    algorithm: str,
    n_runs: int = 20,
    base_seed: int = 1,
    n_rounds_cap: int = 2500,
    selection_mode: str = "knee_point",
    cancel_event: Optional[object] = None,
    history_ready_callback: Optional[Callable[[Dict], None]] = None,
) -> Dict:
    algo = algorithm.strip().lower()
    selection_mode = normalize_selection_mode(selection_mode)
    if algo not in {"improved_leach", "pso", "nsga", "omf"}:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    runs: List[Dict] = []
    first_seed_history = None
    base_params = LeachParams(n_rounds=n_rounds_cap)
    pso_params = PSOParams(base=LeachParams(n_rounds=n_rounds_cap), selection_mode=selection_mode)
    nsga_params = NSGA2Params(base=LeachParams(n_rounds=n_rounds_cap), selection_mode=selection_mode)
    omf_params = OMFParams(base=LeachParams(n_rounds=n_rounds_cap), selection_mode=selection_mode)

    for i in range(n_runs):
        if cancel_event is not None and cancel_event.is_set():
            raise SimulationCancelledError("Simulation cancelled by user.")

        seed = base_seed + i
        topo = generate_topology(base_params, seed)

        if algo == "improved_leach":
            result = run_improved_leach(
                params=base_params,
                topo=topo,
                seed=seed,
                stop_event=cancel_event,
                collect_history=(i == 0),
            )
        elif algo == "pso":
            result = run_pso_leach(
                pso=pso_params,
                topo=topo,
                seed=seed,
                stop_event=cancel_event,
                collect_history=(i == 0),
            )
        elif algo == "omf":
            result = run_omf_leach(
                omf=omf_params,
                topo=topo,
                seed=seed,
                stop_event=cancel_event,
                collect_history=(i == 0),
            )
        else:
            result = run_nsga2_leach(
                nsga=nsga_params,
                topo=topo,
                seed=seed,
                stop_event=cancel_event,
                collect_history=(i == 0),
            )

        if i == 0:
            first_seed_history = result.get("history")
            if first_seed_history is not None and history_ready_callback is not None:
                history_ready_callback(first_seed_history)
        runs.append(result)

    averaged = _average_results(runs, n_rounds_cap)
    averaged["selection_mode"] = selection_mode
    if first_seed_history is not None:
        averaged["history"] = first_seed_history
    return averaged