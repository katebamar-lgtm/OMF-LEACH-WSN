
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from pymoo.algorithms.moo.mopso_cd import MOPSO_CD
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize

from ImprovedLEACH import LeachParams, generate_topology, _d0
from CustomLEACH import (
    decode_particle,
    merge_packet_stats,
    new_packet_stats,
    simulate_custom_leach_round,
)
from nsga2 import evaluate_nsga_objectives
from solution_selection import select_solution_index


@dataclass
class PSOParams:
    base: LeachParams = field(default_factory=LeachParams)

    swarm_size: int = 30
    iters: int = 20

    w_inertia: float = 0.6
    c1: float = 1.6
    c2: float = 1.6
    selection_mode: str = "knee_point"

    Ncl_max: int = 20
    Ds: float | None = None


_decode_particle = decode_particle
_simulate_one_round = simulate_custom_leach_round


class _MOPSOProblem(ElementwiseProblem):
    def __init__(
        self,
        pso: PSOParams,
        topo: Dict,
        E: np.ndarray,
        alive: np.ndarray,
        alive_idx: np.ndarray,
        n_nodes: int,
        k: int,
        rng: np.random.Generator,
    ) -> None:
        super().__init__(
            n_var=k,
            n_obj=3,   # f1: energy, f2: max intra-cluster distance, f3: packet-loss
            xl=np.zeros(k, dtype=float),
            xu=np.full(k, max(len(alive_idx) - 1, 0), dtype=float),
        )
        self.pso = pso
        self.topo = topo
        self.E = E
        self.alive = alive
        self.alive_idx = alive_idx
        self.n_nodes = n_nodes
        self.k = k
        self.rng = rng

    def _evaluate(self, x, out, *args, **kwargs):
        ch_idx = _decode_particle(np.asarray(x, dtype=float), self.alive_idx, self.k, self.rng)
        out["F"] = _objectives(self.pso, self.topo, self.E, ch_idx)


def _objectives(
    pso: PSOParams,
    topo: Dict,
    E: np.ndarray,
    ch_idx: np.ndarray,
) -> np.ndarray:
    # Reuse exactly the NSGA-II objectives for MOPSO.
    return evaluate_nsga_objectives(pso, topo, E, ch_idx)


def run_pso_leach(
    pso: Optional[PSOParams] = None,
    topo: Optional[Dict] = None,
    seed: Optional[int] = None,
    stop_event: Optional[object] = None,
    collect_history: bool = False,
) -> Dict:
    if pso is None:
        pso = PSOParams()

    params = pso.base
    sim_seed = seed if seed is not None else int(getattr(params, "seed", 0))
    n_rounds = int(getattr(params, "n_rounds", 2500))

    if topo is None:
        topo = generate_topology(params, sim_seed)

    rng = np.random.default_rng(sim_seed + 2000)

    n = topo["n"]
    e0 = float(getattr(params, "e0", 0.5))
    p = float(getattr(params, "p", 0.05))
    Ds = float(pso.Ds) if pso.Ds is not None else _d0(params)

    E = np.full(n, e0, dtype=float)
    alive = np.ones(n, dtype=bool)

    alive_per_round: List[int] = []
    total_energy_per_round: List[float] = []
    energy_consumed_per_round: List[float] = []
    energy_consumption_ratio_per_round: List[float] = []
    data_generated_per_round: List[int] = []
    data_delivered_to_bs_per_round: List[int] = []
    packet_loss_per_round: List[int] = []
    packet_loss_ratio_per_round: List[float] = []
    packets_delivered_per_round: List[int] = []
    packet_stats = new_packet_stats()
    packet_stats_per_round = {key: [] for key in packet_stats}
    history_per_round = []

    # Pareto front collected at each round for HV / IGD computation.
    pareto_fronts_per_round: List[np.ndarray] = []

    FND = HND = LND = None
    half = n // 2

    for r in range(n_rounds):
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("Simulation cancelled by user.")

        alive_idx = np.where(alive)[0]
        n_alive = int(len(alive_idx))

        alive_per_round.append(n_alive)
        total_energy_per_round.append(float(np.sum(E)))

        if n_alive == 0:
            if LND is None:
                LND = r
            break

        K = min(int(max(1, round(p * n_alive))), n_alive)
        swarm_size = int(pso.swarm_size)
        iters = int(pso.iters)

        problem = _MOPSOProblem(
            pso=pso,
            topo=topo,
            E=E,
            alive=alive,
            alive_idx=alive_idx,
            n_nodes=n,
            k=K,
            rng=rng,
        )
        algorithm = MOPSO_CD(
            pop_size=swarm_size,
            w=float(pso.w_inertia),
            c1=float(pso.c1),
            c2=float(pso.c2),
            archive_size=max(swarm_size * 5, 50),
        )
        result = minimize(
            problem=problem,
            algorithm=algorithm,
            termination=("n_gen", max(1, iters)),
            seed=int(rng.integers(0, 2_147_483_647)),
            verbose=False,
        )

        pareto_x = None
        pareto_f = None
        if hasattr(algorithm, "archive") and algorithm.archive is not None and len(algorithm.archive) > 0:
            pareto_x = algorithm.archive.get("X")
            pareto_f = algorithm.archive.get("F")
        elif result is not None and result.X is not None and result.F is not None:
            pareto_x = result.X
            pareto_f = result.F

        # ── Capture the Pareto front for HV / IGD computation ─────────────
        if pareto_f is not None:
            pf_arr = np.asarray(pareto_f, dtype=float)
            if pf_arr.ndim == 1:
                pf_arr = pf_arr[None, :]
            pareto_fronts_per_round.append(pf_arr.copy())
        else:
            pareto_fronts_per_round.append(np.empty((0, 3), dtype=float))

        if pareto_x is None or pareto_f is None:
            best_ch = rng.choice(alive_idx, size=1)
        else:
            pareto_x = np.asarray(pareto_x, dtype=float)
            pareto_f = np.asarray(pareto_f, dtype=float)
            if pareto_x.ndim == 1:
                pareto_x = pareto_x[None, :]
            if pareto_f.ndim == 1:
                pareto_f = pareto_f[None, :]

            best_idx = select_solution_index(pareto_f, pso.selection_mode)
            best_x = pareto_x[best_idx]
            best_ch = _decode_particle(best_x, alive_idx, K, rng)
            if len(best_ch) == 0:
                best_ch = rng.choice(alive_idx, size=1)
        Ncl_max = max(1, int(round(len(alive_idx) / max(len(best_ch), 1))))

        alive_before = alive.copy()
        energies_before = E.copy()
        round_result = _simulate_one_round(
            params, topo, E, alive, best_ch, Ds, Ncl_max,
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

        packets_delivered_per_round.append(int(data_delivered))
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

    if FND is None:
        FND = n_rounds
    if HND is None:
        HND = n_rounds
    if LND is None:
        LND = n_rounds

    result = {
        "alive_per_round": alive_per_round,
        "total_energy_per_round": total_energy_per_round,
        "energy_consumed_per_round": energy_consumed_per_round,
        "energy_consumption_ratio_per_round": energy_consumption_ratio_per_round,
        "data_generated_per_round": data_generated_per_round,
        "data_delivered_to_bs_per_round": data_delivered_to_bs_per_round,
        "packet_loss_per_round": packet_loss_per_round,
        "packet_loss_ratio_per_round": packet_loss_ratio_per_round,
        "packets_delivered_per_round": packets_delivered_per_round,
        "FND": FND,
        "HND": HND,
        "LND": LND,
        "total_packets": sum(packets_delivered_per_round),
        "total_packet_loss": sum(packet_loss_per_round),
        "packet_stats": packet_stats,
        "packet_stats_per_round": packet_stats_per_round,
        "selection_mode": pso.selection_mode,
        "pareto_fronts_per_round": pareto_fronts_per_round,
    }
    if collect_history:
        result["history"] = {
            "algo": "pso",
            "seed": sim_seed,
            "topo": copy.deepcopy(topo),
            "params": params,
            "selection_mode": pso.selection_mode,
            "rounds": history_per_round,
        }
    return result