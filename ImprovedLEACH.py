
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class LeachParams:
    # Network parameters from Daanoune et al. (2021)
    n_nodes: int = 80
    area_w: float = 100.0
    area_h: float = 100.0
    bs_x: float = 50.0
    bs_y: float = 150.0
    p: float = 0.05

    # Radio energy model from the article
    e0: float = 0.5
    k_bits: int = 4000
    e_elec: float = 50e-9
    e_fs: float = 10e-12
    e_mp: float = 0.0013e-12
    e_da: float = 5e-9

    # Paper-faithful cluster-size limit:
    # Ncl = total_nodes / desired_CHs = 80 / (0.05 * 80) = 20
    ncl: int = 20

    # Simulation
    n_rounds: int = 2500
    seed: int = 21


def _d0(params: LeachParams) -> float:
    return float(np.sqrt(params.e_fs / params.e_mp))


def _tx_energy(params: LeachParams, d: float) -> float:
    if d < _d0(params):
        return params.e_elec * params.k_bits + params.e_fs * params.k_bits * (d ** 2)
    return params.e_elec * params.k_bits + params.e_mp * params.k_bits * (d ** 4)


def _rx_energy(params: LeachParams) -> float:
    return params.e_elec * params.k_bits


def _agg_energy(params: LeachParams, n_pkts: int) -> float:
    return params.e_da * params.k_bits * n_pkts


def _new_packet_stats() -> Dict[str, int]:
    return {
        "data_generated": 0,
        "data_delivered_to_bs": 0,
        "data_lost": 0,
        "packet_loss": 0,
        "member_to_ch_generated": 0,
        "member_to_ch_success": 0,
        "abandoned_to_bs_generated": 0,
        "abandoned_to_bs_success": 0,
        "abandon_no_ch_available": 0,
        "abandon_all_ch_too_far": 0,
        "abandon_all_in_range_full": 0,
        "ch_to_bs_generated": 0,
        "ch_to_bs_success": 0,
        "ch_to_ch_forward_success": 0,
        "cluster_head_count_total": 0,
        "cluster_member_assignments": 0,
        "cluster_non_ch_nodes": 0,
        "drops_sender_energy_fail": 0,
        "drops_receiver_unavailable": 0,
        "delivered_to_bs": 0,
    }


def _accumulate_packet_stats(total: Dict[str, int], delta: Dict[str, int]) -> None:
    for key, value in delta.items():
        total[key] += int(value)


def _record_packet_stats(history: Dict[str, List[int]], delta: Dict[str, int]) -> None:
    for key, value in delta.items():
        history[key].append(int(value))


def _mark_dead(E: np.ndarray, alive: np.ndarray, idx: int) -> None:
    alive[idx] = False
    E[idx] = 0.0


def _try_single_hop(
    E: np.ndarray,
    alive: np.ndarray,
    sender: int,
    tx_cost: float,
    receiver: Optional[int] = None,
    rx_cost: float = 0.0,
) -> bool:
    if not alive[sender]:
        return False

    if E[sender] < tx_cost:
        _mark_dead(E, alive, sender)
        return False

    if receiver is not None:
        if not alive[receiver]:
            return False
        if E[receiver] < rx_cost:
            _mark_dead(E, alive, receiver)
            return False

    E[sender] -= tx_cost
    if E[sender] <= 0.0:
        _mark_dead(E, alive, sender)

    if receiver is not None:
        E[receiver] -= rx_cost
        if E[receiver] <= 0.0:
            _mark_dead(E, alive, receiver)

    return True


def generate_topology(params: LeachParams, seed: int) -> Dict:
    rng = np.random.default_rng(seed)
    n = params.n_nodes
    pos = rng.uniform([0.0, 0.0], [params.area_w, params.area_h], size=(n, 2))
    sink = np.array([params.bs_x, params.bs_y], dtype=float)

    diff = pos[:, None, :] - pos[None, :, :]
    dist_nn = np.sqrt(np.sum(diff * diff, axis=2))
    dist_bs = np.sqrt(np.sum((pos - sink) ** 2, axis=1))

    return {
        "pos": pos,
        "sink": sink,
        "dist_nn": dist_nn,
        "dist_bs": dist_bs,
        "n": n,
    }


def run_improved_leach(
    params: Optional[LeachParams] = None,
    topo: Optional[Dict] = None,
    seed: Optional[int] = None,
    stop_event: Optional[object] = None,
    collect_history: bool = False,
) -> Dict:

    if params is None:
        params = LeachParams()

    sim_seed = seed if seed is not None else params.seed

    if topo is None:
        topo = generate_topology(params, sim_seed)

    rng = np.random.default_rng(sim_seed + 1000)

    dist_nn = topo["dist_nn"]
    dist_bs = topo["dist_bs"]
    n = topo["n"]

    E = np.full(n, params.e0, dtype=float)
    alive = np.ones(n, dtype=bool)

    epoch = max(1, int(round(1.0 / params.p)))
    eligible = np.ones(n, dtype=bool)

    alive_per_round: List[int] = []
    total_energy_per_round: List[float] = []
    energy_consumed_per_round: List[float] = []
    energy_consumption_ratio_per_round: List[float] = []
    data_generated_per_round: List[int] = []
    data_delivered_to_bs_per_round: List[int] = []
    packet_loss_per_round: List[int] = []
    packet_loss_ratio_per_round: List[float] = []
    packets_delivered_per_round: List[int] = []
    packet_stats = _new_packet_stats()
    packet_stats_per_round = {key: [] for key in packet_stats}
    history_per_round = []

    FND = HND = LND = None
    half = n // 2

    for r in range(params.n_rounds):
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("Simulation cancelled by user.")

        if r % epoch == 0:
            eligible[:] = True

        alive_idx = np.where(alive)[0]
        n_alive = int(len(alive_idx))

        alive_per_round.append(n_alive)
        total_energy_per_round.append(float(np.sum(E)))

        if n_alive == 0:
            if LND is None:
                LND = r
            break

        # =========================
        # IMPROVED CH SELECTION
        # =========================
        denom = 1.0 - params.p * (r % epoch)
        base_T = (params.p / denom) if denom > 0 else 1.0

        ch = np.zeros(n, dtype=bool)
        rand_vals = rng.random(n)

        for i in alive_idx:
            if eligible[i]:
                T_i = base_T * (E[i] / params.e0)
                if rand_vals[i] < T_i:
                    ch[i] = True

        if ch.sum() == 0:
            best = alive_idx[np.argmax(E[alive_idx])]
            ch[best] = True

        eligible[ch] = False
        ch_idx = np.where(ch)[0]
        round_packet_stats = _new_packet_stats()
        round_packet_stats["data_generated"] = n_alive
        round_packet_stats["cluster_head_count_total"] = int(len(ch_idx))

        # =========================
        # IMPROVED CLUSTERING
        # =========================
        cluster_members: Dict[int, List[int]] = {int(c): [] for c in ch_idx}
        cluster_payload_counts: Dict[int, int] = {int(c): 1 for c in ch_idx}
        Ncl_max = params.ncl
        d0 = _d0(params)

        abandoned_nodes = []

        for i in alive_idx:
            if ch[i]:
                continue

            round_packet_stats["cluster_non_ch_nodes"] += 1
            dists = dist_nn[i, ch_idx]
            in_range_idx = np.where(dists < d0)[0]

            joined = False
            if len(ch_idx) == 0:
                round_packet_stats["abandon_no_ch_available"] += 1
            elif len(in_range_idx) == 0:
                round_packet_stats["abandon_all_ch_too_far"] += 1
            else:
                sorted_idx = in_range_idx[np.argsort(dists[in_range_idx])]
                for idx in sorted_idx:
                    c = int(ch_idx[idx])
                    if len(cluster_members[c]) < Ncl_max:
                        cluster_members[c].append(i)
                        joined = True
                        round_packet_stats["cluster_member_assignments"] += 1
                        break
                if not joined:
                    round_packet_stats["abandon_all_in_range_full"] += 1

            if not joined:
                abandoned_nodes.append(i)

        # =========================
        # ENERGY + DATA DELIVERY
        # =========================
        alive_before = alive.copy()
        energies_before = E.copy()
        data_delivered_to_bs = 0
        abandoned_paths: Dict[int, List[int]] = {}

        # Abandoned nodes → BS
        for m in abandoned_nodes:
            if not alive[m]:
                continue

            round_packet_stats["abandoned_to_bs_generated"] += 1
            d_bs = float(dist_bs[m])
            tx_cost = _tx_energy(params, d_bs)
            if E[m] < tx_cost:
                round_packet_stats["drops_sender_energy_fail"] += 1
                _mark_dead(E, alive, m)
                continue
            if _try_single_hop(E, alive, m, tx_cost):
                data_delivered_to_bs += 1
                abandoned_paths[int(m)] = [int(m), n]
                round_packet_stats["abandoned_to_bs_success"] += 1
                round_packet_stats["delivered_to_bs"] += 1

        for c, members in cluster_members.items():
            if not alive[c]:
                continue

            n_received = 0
            for m in members:
                if not alive[m] or not alive[c]:
                    continue

                round_packet_stats["member_to_ch_generated"] += 1
                d_m_ch = float(dist_nn[m, c])
                tx_cost = _tx_energy(params, d_m_ch)
                rx_cost = _rx_energy(params)
                if E[m] < tx_cost:
                    round_packet_stats["drops_sender_energy_fail"] += 1
                    _mark_dead(E, alive, m)
                    continue
                if not alive[c] or E[c] < rx_cost:
                    round_packet_stats["drops_receiver_unavailable"] += 1
                    if alive[c] and E[c] < rx_cost:
                        _mark_dead(E, alive, c)
                    continue
                if _try_single_hop(E, alive, m, tx_cost, receiver=c, rx_cost=rx_cost):
                    n_received += 1
                    cluster_payload_counts[int(c)] = cluster_payload_counts.get(int(c), 1) + 1
                    round_packet_stats["member_to_ch_success"] += 1

            if n_received > 0 and alive[c]:
                E[c] -= _agg_energy(params, n_received)
                if E[c] <= 0.0:
                    _mark_dead(E, alive, c)

        # =========================
        # GREEDY MULTI-HOP
        # =========================
        paths: Dict[int, List[int]] = {}
        for c in ch_idx:
            if not alive[c]:
                continue

            round_packet_stats["ch_to_bs_generated"] += 1
            current = c
            path_succeeded = False
            path = [int(c)]

            while True:
                d_current_bs = float(dist_bs[current])

                candidates = [
                    j for j in ch_idx
                    if alive[j] and dist_bs[j] < d_current_bs
                ]

                if not candidates:
                    tx_cost = _tx_energy(params, d_current_bs)
                    if E[current] < tx_cost:
                        round_packet_stats["drops_sender_energy_fail"] += 1
                        _mark_dead(E, alive, current)
                        break
                    if _try_single_hop(E, alive, current, tx_cost):
                        path.append(n)
                        path_succeeded = True
                        round_packet_stats["ch_to_bs_success"] += 1
                        round_packet_stats["delivered_to_bs"] += 1
                    break

                dists = [dist_nn[current, j] for j in candidates]
                next_ch = int(candidates[np.argmin(dists)])
                d_next = float(np.min(dists))

                tx_cost = _tx_energy(params, d_next)
                rx_cost = _rx_energy(params)
                if E[current] < tx_cost:
                    round_packet_stats["drops_sender_energy_fail"] += 1
                    _mark_dead(E, alive, current)
                    break
                if not alive[next_ch] or E[next_ch] < rx_cost:
                    round_packet_stats["drops_receiver_unavailable"] += 1
                    if alive[next_ch] and E[next_ch] < rx_cost:
                        _mark_dead(E, alive, next_ch)
                    break
                if not _try_single_hop(E, alive, current, tx_cost, receiver=next_ch, rx_cost=rx_cost):
                    break

                current = next_ch
                path.append(int(current))
                round_packet_stats["ch_to_ch_forward_success"] += 1

            if path_succeeded:
                paths[int(c)] = path
                data_delivered_to_bs += int(cluster_payload_counts.get(int(c), 1))

        alive = E > 0
        E[~alive] = 0.0
        n_alive_after = int(np.count_nonzero(alive))
        energy_before_total = float(np.sum(energies_before))
        round_energy_consumed = max(0.0, float(energy_before_total - np.sum(E)))
        round_energy_ratio = float(round_energy_consumed / max(energy_before_total, 1e-12))
        packet_loss = max(0, n_alive - int(data_delivered_to_bs))
        packet_loss_ratio = float(packet_loss / max(n_alive, 1))
        round_packet_stats["data_delivered_to_bs"] = int(data_delivered_to_bs)
        round_packet_stats["data_lost"] = int(packet_loss)
        round_packet_stats["packet_loss"] = int(packet_loss)

        if FND is None and n_alive_after < n:
            FND = r + 1
        if HND is None and n_alive_after <= half:
            HND = r + 1
        if n_alive_after == 0:
            if LND is None:
                LND = r + 1
            energy_consumed_per_round.append(round_energy_consumed)
            energy_consumption_ratio_per_round.append(round_energy_ratio)
            data_generated_per_round.append(n_alive)
            data_delivered_to_bs_per_round.append(int(data_delivered_to_bs))
            packet_loss_per_round.append(int(packet_loss))
            packet_loss_ratio_per_round.append(packet_loss_ratio)
            packets_delivered_per_round.append(int(data_delivered_to_bs))
            _accumulate_packet_stats(packet_stats, round_packet_stats)
            _record_packet_stats(packet_stats_per_round, round_packet_stats)
            if collect_history:
                history_per_round.append({
                    "round": r + 1,
                    "ch_idx": ch_idx.copy(),
                    "cluster_members": copy.deepcopy(cluster_members),
                    "abandoned_nodes": np.asarray(abandoned_nodes, dtype=int).copy(),
                    "abandoned_paths": copy.deepcopy(abandoned_paths),
                    "paths": copy.deepcopy(paths),
                    "alive_before": alive_before.copy(),
                    "energies_before": energies_before.copy(),
                    "alive": alive.copy(),
                    "energies": E.copy(),
                    "energy_consumed": round_energy_consumed,
                    "energy_consumption_ratio": round_energy_ratio,
                    "data_generated": n_alive,
                    "data_delivered_to_bs": int(data_delivered_to_bs),
                    "packet_loss": int(packet_loss),
                    "packet_loss_ratio": packet_loss_ratio,
                    "packets": int(data_delivered_to_bs),
                    "dead_this_round": np.where(alive_before & ~alive)[0].astype(int).copy(),
                    "packet_stats": dict(round_packet_stats),
                })
            break

        energy_consumed_per_round.append(round_energy_consumed)
        energy_consumption_ratio_per_round.append(round_energy_ratio)
        data_generated_per_round.append(n_alive)
        data_delivered_to_bs_per_round.append(int(data_delivered_to_bs))
        packet_loss_per_round.append(int(packet_loss))
        packet_loss_ratio_per_round.append(packet_loss_ratio)
        packets_delivered_per_round.append(int(data_delivered_to_bs))
        _accumulate_packet_stats(packet_stats, round_packet_stats)
        _record_packet_stats(packet_stats_per_round, round_packet_stats)
        if collect_history:
            history_per_round.append({
                "round": r + 1,
                "ch_idx": ch_idx.copy(),
                "cluster_members": copy.deepcopy(cluster_members),
                "abandoned_nodes": np.asarray(abandoned_nodes, dtype=int).copy(),
                "abandoned_paths": copy.deepcopy(abandoned_paths),
                "paths": copy.deepcopy(paths),
                "alive_before": alive_before.copy(),
                "energies_before": energies_before.copy(),
                "alive": alive.copy(),
                "energies": E.copy(),
                "energy_consumed": round_energy_consumed,
                "energy_consumption_ratio": round_energy_ratio,
                "data_generated": n_alive,
                "data_delivered_to_bs": int(data_delivered_to_bs),
                "packet_loss": int(packet_loss),
                "packet_loss_ratio": packet_loss_ratio,
                "packets": int(data_delivered_to_bs),
                "dead_this_round": np.where(alive_before & ~alive)[0].astype(int).copy(),
                "packet_stats": dict(round_packet_stats),
            })

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
        "packets_delivered_per_round": packets_delivered_per_round,
        "FND": FND,
        "HND": HND,
        "LND": LND,
        "total_packets": sum(packets_delivered_per_round),
        "total_packet_loss": sum(packet_loss_per_round),
        "packet_stats": packet_stats,
        "packet_stats_per_round": packet_stats_per_round,
    }
    if collect_history:
        result["history"] = {
            "algo": "improved_leach",
            "seed": sim_seed,
            "topo": copy.deepcopy(topo),
            "params": params,
            "rounds": history_per_round,
        }
    return result


run_leach = run_improved_leach


if __name__ == "__main__":
    res = run_improved_leach()
    print("FND:", res["FND"], "HND:", res["HND"], "LND:", res["LND"])