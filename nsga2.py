
from __future__ import annotations
import copy
from dataclasses import dataclass, field
import numpy as np

from typing import Dict, List, Optional, Tuple
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize

from ImprovedLEACH import LeachParams, generate_topology, _d0
from CustomLEACH import (
    _simulate_one_round,
    decode_particle,
    full_round_objectives,
    merge_packet_stats,
    new_packet_stats,
)
from solution_selection import select_solution_index


# =========================
# PARAMETERS
# =========================

@dataclass
class NSGA2Params:
    base: LeachParams = field(default_factory=LeachParams)

    pop_size: int = 30
    generations: int = 20

    crossover_prob: float = 0.9
    mutation_prob: float = 0.2
    selection_mode: str = "knee_point"


class _NSGA2Problem(ElementwiseProblem):
    def __init__(
        self,
        nsga: NSGA2Params,
        topo: Dict,
        E: np.ndarray,
        alive_idx: np.ndarray,
        k: int,
        rng: np.random.Generator,
    ) -> None:
        super().__init__(
            n_var=k,
            n_obj=3,   # f1: energy, f2: max intra-cluster distance, f3: packet-loss
            xl=np.zeros(k, dtype=float),
            xu=np.full(k, max(len(alive_idx) - 1, 0), dtype=float),
        )
        self.nsga = nsga
        self.topo = topo
        self.E = E
        self.alive_idx = alive_idx
        self.k = k
        self.rng = rng

    def _evaluate(self, x, out, *args, **kwargs):
        ch_idx = decode_particle(np.asarray(x, dtype=float), self.alive_idx, self.k, self.rng)
        out["F"] = evaluate_nsga_objectives(self.nsga, self.topo, self.E, ch_idx)


def evaluate_nsga_objectives(nsga, topo, E, ch_idx):
    if len(ch_idx) == 0:
        return np.array([1e9, 1e9, 1e9], dtype=float)

    return full_round_objectives(
        params=nsga.base,
        topo=topo,
        E=E,
        ch_idx=ch_idx,
        ds=getattr(nsga, "Ds", None),
    )


def optimize_nsga2_pareto_front(
    nsga: NSGA2Params,
    topo: Dict,
    E: np.ndarray,
    alive: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[List[np.ndarray], np.ndarray]:
    alive_idx = np.where(alive)[0]
    n_alive = len(alive_idx)
    if n_alive == 0:
        return [], np.empty((0, 2), dtype=float)

    k = min(int(max(1, round(nsga.base.p * n_alive))), n_alive)
    problem = _NSGA2Problem(
        nsga=nsga,
        topo=topo,
        E=E,
        alive_idx=alive_idx,
        k=k,
        rng=rng,
    )
    algorithm = NSGA2(
        pop_size=int(nsga.pop_size),
        crossover=SBX(prob=float(nsga.crossover_prob), eta=15.0),
        mutation=PM(prob=float(nsga.mutation_prob), eta=20.0),
        eliminate_duplicates=False,
    )
    result = minimize(
        problem=problem,
        algorithm=algorithm,
        termination=("n_gen", max(1, int(nsga.generations))),
        seed=int(rng.integers(0, 2_147_483_647)),
        verbose=False,
    )

    if result is None or result.X is None or result.F is None:
        return [], np.empty((0, 2), dtype=float)

    pareto_x = np.asarray(result.X, dtype=float)
    pareto_f = np.asarray(result.F, dtype=float)

    if pareto_x.ndim == 1:
        pareto_x = pareto_x[None, :]
    if pareto_f.ndim == 1:
        pareto_f = pareto_f[None, :]

    pareto_solutions = [
        decode_particle(ind, alive_idx, k, rng)
        for ind in pareto_x
    ]
    return pareto_solutions, pareto_f


# =========================
# NSGA-II PER ROUND (PARETO)
# =========================

def run_nsga2_one_round(nsga, topo, E, alive, rng):
    pareto_solutions, pareto_objs = optimize_nsga2_pareto_front(nsga, topo, E, alive, rng)
    if len(pareto_solutions) == 0:
        return None

    best_idx = select_solution_index(pareto_objs, nsga.selection_mode)
    return pareto_solutions[best_idx]


# =========================
# FULL SIMULATION
# =========================

def run_nsga2_leach(
    nsga: Optional[NSGA2Params] = None,
    topo: Optional[Dict] = None,
    seed: int = 42,
    stop_event: Optional[object] = None,
    collect_history: bool = False,
):

    if nsga is None:
        nsga = NSGA2Params()

    params = nsga.base
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

    # Pareto front collected at each round for HV / IGD computation.
    pareto_fronts_per_round: List[np.ndarray] = []

    for r in range(params.n_rounds):
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("Simulation cancelled by user.")

        alive_idx = np.where(alive)[0]
        n_alive = len(alive_idx)

        alive_per_round.append(int(n_alive))
        total_energy_per_round.append(float(np.sum(E)))

        if n_alive == 0:
            if LND is None:
                LND = r
            break

        # ── Run NSGA-II and capture the Pareto front ──────────────────────
        pareto_solutions, pareto_objs = optimize_nsga2_pareto_front(
            nsga, topo, E, alive, rng
        )
        if len(pareto_objs) > 0:
            pareto_fronts_per_round.append(np.asarray(pareto_objs, dtype=float).copy())
        else:
            pareto_fronts_per_round.append(np.empty((0, 3), dtype=float))

        if len(pareto_solutions) == 0:
            best_ch = None
        else:
            best_idx = select_solution_index(pareto_objs, nsga.selection_mode)
            best_ch = pareto_solutions[best_idx]

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
        "selection_mode": nsga.selection_mode,
        "pareto_fronts_per_round": pareto_fronts_per_round,
    }
    if collect_history:
        result["history"] = {
            "algo": "nsga",
            "seed": seed,
            "topo": copy.deepcopy(topo),
            "params": params,
            "selection_mode": nsga.selection_mode,
            "rounds": history_per_round,
        }
    return result


# =========================
# MAIN
# =========================

if __name__ == "__main__":

    print("Running NSGA-II (Pareto Energy-Packet Loss)...")

    params = NSGA2Params()
    result = run_nsga2_leach(params, seed=42)

    print("\n===== RESULTS =====")
    print("FND:", result["FND"])
    print("HND:", result["HND"])
    print("LND:", result["LND"])
    print("Total packets:", result["total_packets"])
    print("Total packet loss:", result["total_packet_loss"])