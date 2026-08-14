from __future__ import annotations

import heapq
import copy
from typing import Dict, List, Tuple

import numpy as np

from ImprovedLEACH import LeachParams, generate_topology, _d0


def radio_params(params: LeachParams) -> Tuple[float, float, float, float, float]:
    e_elec = float(getattr(params, "e_elec", 50e-9))
    e_fs = float(getattr(params, "e_fs", 10e-12))
    e_mp = float(getattr(params, "e_mp", 0.0013e-12))
    e_da = float(getattr(params, "e_da", 5e-9))
    d0 = float(np.sqrt(e_fs / e_mp))
    return e_elec, e_fs, e_mp, e_da, d0


def tx_energy(k_bits: int, d: float, e_elec: float, e_fs: float, e_mp: float, d0: float) -> float:
    if d < d0:
        return (e_elec * k_bits) + (e_fs * k_bits * (d ** 2))
    return (e_elec * k_bits) + (e_mp * k_bits * (d ** 4))


def rx_energy(k_bits: int, e_elec: float) -> float:
    return e_elec * k_bits


def agg_energy(k_bits: int, e_da: float, n_pkts: int) -> float:
    return e_da * k_bits * n_pkts


def new_packet_stats() -> Dict[str, int]:
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


def merge_packet_stats(total: Dict[str, int], delta: Dict[str, int]) -> None:
    for key, value in delta.items():
        total[key] = total.get(key, 0) + int(value)


def mark_dead(e: np.ndarray, alive: np.ndarray, idx: int) -> None:
    alive[idx] = False
    e[idx] = 0.0


def try_single_hop(
    e: np.ndarray,
    alive: np.ndarray,
    sender: int,
    tx_cost: float,
    receiver: int | None = None,
    rx_cost: float = 0.0,
) -> bool:
    if not alive[sender]:
        return False

    if e[sender] < tx_cost:
        mark_dead(e, alive, sender)
        return False

    if receiver is not None:
        if not alive[receiver]:
            return False
        if e[receiver] < rx_cost:
            mark_dead(e, alive, receiver)
            return False

    e[sender] -= tx_cost
    if e[sender] <= 0.0:
        mark_dead(e, alive, sender)

    if receiver is not None:
        e[receiver] -= rx_cost
        if e[receiver] <= 0.0:
            mark_dead(e, alive, receiver)

    return True


def dijkstra_min_energy(
    ch_idx: np.ndarray,
    dist_nn: np.ndarray,
    dist_bs: np.ndarray,
    k_bits: int,
    e_elec: float,
    e_fs: float,
    e_mp: float,
    d0: float,
    n_nodes: int,
) -> Tuple[Dict[int, List[int]], Dict[int, float], Dict[int, int]]:
    bs_idx = n_nodes
    ch_set = set(ch_idx.tolist())

    def edge_energy(i: int, j: int, d: float) -> float:
        return tx_energy(k_bits, d, e_elec, e_fs, e_mp, d0) + rx_energy(k_bits, e_elec)

    paths: Dict[int, List[int]] = {}
    path_energies: Dict[int, float] = {}
    path_hops: Dict[int, int] = {}

    for source_ch in ch_idx:
        inf = float("inf")
        dist_cost = {bs_idx: inf}
        for c in ch_idx:
            dist_cost[int(c)] = inf
        dist_cost[int(source_ch)] = 0.0

        prev: Dict[int, int] = {}
        visited = set()
        pq = [(0.0, int(source_ch))]

        while pq:
            cost, u = heapq.heappop(pq)

            if u in visited:
                continue
            visited.add(u)

            if u == bs_idx:
                break

            if u in ch_set:
                for v in ch_idx:
                    v = int(v)
                    if v == u or v in visited:
                        continue
                    d = float(dist_nn[u, v])
                    new_cost = cost + edge_energy(u, v, d)
                    if new_cost < dist_cost.get(v, inf):
                        dist_cost[v] = new_cost
                        prev[v] = u
                        heapq.heappush(pq, (new_cost, v))

                if bs_idx not in visited:
                    d_bs = float(dist_bs[u])
                    new_cost = cost + tx_energy(k_bits, d_bs, e_elec, e_fs, e_mp, d0)
                    if new_cost < dist_cost.get(bs_idx, inf):
                        dist_cost[bs_idx] = new_cost
                        prev[bs_idx] = u
                        heapq.heappush(pq, (new_cost, bs_idx))

        path: List[int] = []
        node = bs_idx
        if node in prev or node == int(source_ch):
            while node != int(source_ch):
                path.append(node)
                node = prev.get(node, int(source_ch))
            path.append(int(source_ch))
            path.reverse()
        else:
            path = [int(source_ch), bs_idx]

        paths[int(source_ch)] = path
        path_energies[int(source_ch)] = dist_cost.get(bs_idx, inf)
        path_hops[int(source_ch)] = len(path) - 1

    return paths, path_energies, path_hops


def full_round_objectives(
    params: LeachParams,
    topo: Dict,
    E: np.ndarray,
    ch_idx: np.ndarray,
    ds: float | None = None,
) -> np.ndarray:
    """Evaluate three conflicting objectives for a candidate CH configuration.

    Returns
    -------
    np.ndarray, shape (3,)
        f1 : normalised round energy consumption           [0, 1]  (minimise)
        f2 : normalised maximum intra-cluster distance     [0, 1]  (minimise)
        f3 : per-round packet-loss ratio                   [0, 1]  (minimise)

    The three objectives correspond exactly to Eq. (4)–(6) in the article.
    Extending from 2 to 3 objectives required adding f2 here; all callers
    (OMF, NSGA-II via _NSGA2Problem, MOPSO via _MOPSOProblem) receive and
    forward the updated (3,) array transparently because they pass the return
    value of this function directly to their respective Pareto-update logic.
    The only downstream change is n_obj=3 in the pymoo Problem subclasses
    (see nsga2.py and pso_leach.py).
    """
    alive_tmp = E > 0
    E_before = E.copy()

    ch_idx = np.asarray(
        [int(ch) for ch in ch_idx if 0 <= int(ch) < len(E_before) and alive_tmp[int(ch)]],
        dtype=int,
    )
    if len(ch_idx) == 0:
        return np.array([1.0, 1.0, 1.0], dtype=float)

    n_alive = int(np.count_nonzero(alive_tmp))
    Ncl_max = max(1, int(round(n_alive / max(len(ch_idx), 1))))

    round_result = _simulate_one_round(
        params=params,
        topo=topo,
        E=E_before.copy(),
        alive=alive_tmp.copy(),
        ch_idx=ch_idx,
        Ds=float(ds) if ds is not None else _d0(params),
        Ncl_max=Ncl_max,
        collect_history=False,
    )

    E_after = round_result[0]
    round_packet_stats = round_result[3]

    # ── f1 : normalised communication energy consumed this round ──────────
    total_energy_before = float(np.sum(E_before))
    round_energy = float(total_energy_before - np.sum(E_after))
    energy_obj = float(np.clip(round_energy / max(total_energy_before, 1e-12), 0.0, 1.0))

    # ── f2 : normalised maximum intra-cluster distance (Eq. 5) ───────────
    # D_max(CH) = max over all clusters of the max member→CH distance.
    # Fully vectorised: extract the submatrix dist_nn[non_ch, :][:, ch_idx],
    # assign each non-CH to its nearest CH, then take the global max.
    # Normalised by the sensing-area diagonal so that f2 ∈ [0, 1].
    dist_nn   = topo["dist_nn"]              # shape (n, n), precomputed
    area_diag = float(np.sqrt(
        params.area_w ** 2 + params.area_h ** 2
    ))
    alive_idx_all = np.where(alive_tmp)[0]
    non_ch_mask   = np.ones(len(alive_idx_all), dtype=bool)
    ch_set_local  = set(ch_idx.tolist())
    for k_pos, k_node in enumerate(alive_idx_all):
        if k_node in ch_set_local:
            non_ch_mask[k_pos] = False

    non_ch_idx = alive_idx_all[non_ch_mask]   # global node indices of non-CHs

    if len(non_ch_idx) == 0 or len(ch_idx) == 0:
        max_intra_dist = 0.0
    else:
        # sub-matrix: distances from each non-CH to each CH  (shape: n_nonch × n_ch)
        d_sub  = dist_nn[np.ix_(non_ch_idx, ch_idx)]   # vectorised slice
        # For each non-CH: distance to its nearest CH
        nearest_dist = d_sub.min(axis=1)                # shape (n_nonch,)
        max_intra_dist = float(nearest_dist.max())

    dist_obj = float(np.clip(max_intra_dist / max(area_diag, 1e-12), 0.0, 1.0))

    # ── f3 : per-round packet-loss ratio (Eq. 6) ─────────────────────────
    data_generated = int(round_packet_stats.get("data_generated", n_alive))
    data_lost = int(round_packet_stats.get("data_lost", data_generated))
    packet_loss_obj = float(np.clip(data_lost / max(data_generated, 1), 0.0, 1.0))

    return np.array([energy_obj, dist_obj, packet_loss_obj], dtype=float)


def decode_particle(p: np.ndarray, alive_idx: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    if k <= 0:
        return np.array([], dtype=int)
    m = len(alive_idx)
    if m == 0:
        return np.array([], dtype=int)

    idx = np.clip(np.rint(p).astype(int), 0, m - 1)
    ch = alive_idx[idx]
    ch_unique = np.unique(ch)

    if len(ch_unique) == k:
        return ch_unique

    needed = k - len(ch_unique)
    remaining = np.setdiff1d(alive_idx, ch_unique, assume_unique=False)
    if len(remaining) > 0:
        if needed >= len(remaining):
            fill = remaining
        else:
            fill = rng.choice(remaining, size=needed, replace=False)
        ch_unique = np.concatenate([ch_unique, np.array(fill, dtype=int)])

    return ch_unique[:k]


def simulate_custom_leach_round(
    params: LeachParams,
    topo: Dict,
    e: np.ndarray,
    alive: np.ndarray,
    ch_idx: np.ndarray,
    ds: float,
    ncl_max: int,
    collect_history: bool = False,
) -> Tuple[np.ndarray, np.ndarray, int, Dict[str, int]]:
    dist_nn = topo["dist_nn"]
    dist_bs = topo["dist_bs"]
    n = topo["n"]

    k_bits = int(getattr(params, "k_bits", 4000))
    e_elec, e_fs, e_mp, e_da, d0 = radio_params(params)

    e2 = e.copy()
    alive2 = alive.copy()
    alive_before = alive2.copy()

    alive_idx = np.where(alive2)[0]
    ch_idx = np.array([c for c in ch_idx if alive2[c]], dtype=int)

    data_generated = int(len(alive_idx))
    data_delivered_to_bs = 0
    abandoned_paths: Dict[int, List[int]] = {}
    round_packet_stats = new_packet_stats()
    round_packet_stats["data_generated"] = data_generated

    def finalize_packet_stats() -> None:
        data_lost = max(0, data_generated - data_delivered_to_bs)
        round_packet_stats["data_delivered_to_bs"] = int(data_delivered_to_bs)
        round_packet_stats["data_lost"] = int(data_lost)
        round_packet_stats["packet_loss"] = int(data_lost)

    if len(alive_idx) == 0:
        finalize_packet_stats()
        if collect_history:
            details = {
                "cluster_members": {},
                "abandoned_nodes": [],
                "abandoned_paths": {},
                "paths": {},
                "dead_this_round": [],
                "data_generated": 0,
                "data_delivered_to_bs": 0,
                "packet_loss": 0,
                "packet_loss_ratio": 0.0,
            }
            return e2, alive2, 0, round_packet_stats, details
        return e2, alive2, 0, round_packet_stats

    if len(ch_idx) == 0:
        abandoned = [int(i) for i in alive_idx]
        for i in alive_idx:
            round_packet_stats["abandoned_to_bs_generated"] += 1
            round_packet_stats["abandon_no_ch_available"] += 1
            d = float(dist_bs[i])
            tx_cost = tx_energy(k_bits, d, e_elec, e_fs, e_mp, d0)
            if e2[i] < tx_cost:
                round_packet_stats["drops_sender_energy_fail"] += 1
                mark_dead(e2, alive2, i)
                continue
            if try_single_hop(e2, alive2, i, tx_cost):
                data_delivered_to_bs += 1
                abandoned_paths[int(i)] = [int(i), n]
                round_packet_stats["abandoned_to_bs_success"] += 1
                round_packet_stats["delivered_to_bs"] += 1
        finalize_packet_stats()
        packet_loss_ratio = round_packet_stats["packet_loss"] / max(data_generated, 1)
        if collect_history:
            details = {
                "cluster_members": {},
                "abandoned_nodes": abandoned,
                "abandoned_paths": copy.deepcopy(abandoned_paths),
                "paths": {},
                "dead_this_round": np.where(alive_before & ~alive2)[0].astype(int).tolist(),
                "data_generated": data_generated,
                "data_delivered_to_bs": int(data_delivered_to_bs),
                "packet_loss": int(round_packet_stats["packet_loss"]),
                "packet_loss_ratio": float(packet_loss_ratio),
            }
            return e2, alive2, int(data_delivered_to_bs), round_packet_stats, details
        return e2, alive2, int(data_delivered_to_bs), round_packet_stats

    ch_idx_list: List[int] = [int(c) for c in ch_idx]
    ch_set = set(ch_idx_list)
    cluster_members: Dict[int, List[int]] = {c: [] for c in ch_idx_list}
    cluster_payload_counts: Dict[int, int] = {c: 1 for c in ch_idx_list}
    round_packet_stats["cluster_head_count_total"] = int(len(ch_idx_list))
    abandoned: List[int] = []

    # NOTE (performance patch): the original implementation rebuilt small NumPy
    # arrays and called np.where/np.argsort for every single non-CH node, on
    # every one of the ~500+ candidate evaluations per round performed by the
    # OMF/NSGA-II/MOPSO optimizers. Profiling showed this pattern alone
    # accounted for the majority of total runtime (np.argsort called ~928k
    # times for just 20 rounds), because NumPy's per-call overhead dominates
    # when operating on arrays of only a handful of elements (here, the number
    # of cluster heads, typically ~4). Replacing it with plain Python list
    # operations preserves the exact same behavior (ascending distance order,
    # identical tie-breaking via Python's stable sort) while removing this
    # overhead entirely. No simulation logic or results are altered.
    for i in alive_idx:
        i = int(i)
        if i in ch_set:
            continue
        round_packet_stats["cluster_non_ch_nodes"] += 1
        row = dist_nn[i]
        candidates = [(float(row[c]), c) for c in ch_idx_list if row[c] < ds]

        if not candidates:
            if len(ch_idx_list) == 0:
                round_packet_stats["abandon_no_ch_available"] += 1
            else:
                round_packet_stats["abandon_all_ch_too_far"] += 1
            abandoned.append(i)
            continue

        candidates.sort(key=lambda t: t[0])
        joined = False
        for _, ch in candidates:
            if len(cluster_members[ch]) < ncl_max:
                cluster_members[ch].append(i)
                joined = True
                round_packet_stats["cluster_member_assignments"] += 1
                break
        if not joined:
            round_packet_stats["abandon_all_in_range_full"] += 1
            abandoned.append(i)

    for ch, members in cluster_members.items():
        if not alive2[ch]:
            continue
        n_received = 0
        for m in members:
            if not alive2[m] or not alive2[ch]:
                continue
            round_packet_stats["member_to_ch_generated"] += 1
            d_m_ch = float(dist_nn[m, ch])
            tx_cost = tx_energy(k_bits, d_m_ch, e_elec, e_fs, e_mp, d0)
            rx_cost = rx_energy(k_bits, e_elec)
            if e2[m] < tx_cost:
                round_packet_stats["drops_sender_energy_fail"] += 1
                mark_dead(e2, alive2, m)
                continue
            if not alive2[ch] or e2[ch] < rx_cost:
                round_packet_stats["drops_receiver_unavailable"] += 1
                if alive2[ch] and e2[ch] < rx_cost:
                    mark_dead(e2, alive2, ch)
                continue
            if try_single_hop(e2, alive2, m, tx_cost, receiver=ch, rx_cost=rx_cost):
                n_received += 1
                cluster_payload_counts[int(ch)] = cluster_payload_counts.get(int(ch), 1) + 1
                round_packet_stats["member_to_ch_success"] += 1

        if n_received > 0 and alive2[ch]:
            e2[ch] -= agg_energy(k_bits, e_da, n_received)
            if e2[ch] <= 0.0:
                mark_dead(e2, alive2, ch)

    for i in abandoned:
        if not alive2[i]:
            continue
        round_packet_stats["abandoned_to_bs_generated"] += 1
        d = float(dist_bs[i])
        tx_cost = tx_energy(k_bits, d, e_elec, e_fs, e_mp, d0)
        if e2[i] < tx_cost:
            round_packet_stats["drops_sender_energy_fail"] += 1
            mark_dead(e2, alive2, i)
            continue
        if try_single_hop(e2, alive2, i, tx_cost):
            data_delivered_to_bs += 1
            abandoned_paths[int(i)] = [int(i), n]
            round_packet_stats["abandoned_to_bs_success"] += 1
            round_packet_stats["delivered_to_bs"] += 1

    paths, _, _ = dijkstra_min_energy(
        ch_idx, dist_nn, dist_bs, k_bits, e_elec, e_fs, e_mp, d0, n
    )

    bs_idx = n
    for source_ch, path in paths.items():
        if not alive2[source_ch]:
            continue

        round_packet_stats["ch_to_bs_generated"] += 1
        path_succeeded = True
        for step in range(len(path) - 1):
            sender = path[step]
            receiver = path[step + 1]

            if sender == bs_idx:
                break

            if receiver == bs_idx:
                d = float(dist_bs[sender])
                tx_cost = tx_energy(k_bits, d, e_elec, e_fs, e_mp, d0)
                if e2[sender] < tx_cost:
                    round_packet_stats["drops_sender_energy_fail"] += 1
                    mark_dead(e2, alive2, sender)
                    path_succeeded = False
                    break
                if try_single_hop(e2, alive2, sender, tx_cost):
                    round_packet_stats["ch_to_bs_success"] += 1
                    round_packet_stats["delivered_to_bs"] += 1
                else:
                    path_succeeded = False
                    break
            else:
                d = float(dist_nn[sender, receiver])
                tx_cost = tx_energy(k_bits, d, e_elec, e_fs, e_mp, d0)
                rx_cost = rx_energy(k_bits, e_elec)
                if e2[sender] < tx_cost:
                    round_packet_stats["drops_sender_energy_fail"] += 1
                    mark_dead(e2, alive2, sender)
                    path_succeeded = False
                    break
                if not alive2[receiver] or e2[receiver] < rx_cost:
                    round_packet_stats["drops_receiver_unavailable"] += 1
                    if alive2[receiver] and e2[receiver] < rx_cost:
                        mark_dead(e2, alive2, receiver)
                    path_succeeded = False
                    break
                if not try_single_hop(e2, alive2, sender, tx_cost, receiver=receiver, rx_cost=rx_cost):
                    path_succeeded = False
                    break
                round_packet_stats["ch_to_ch_forward_success"] += 1

        if path_succeeded:
            data_delivered_to_bs += int(cluster_payload_counts.get(int(source_ch), 1))

    finalize_packet_stats()
    packet_loss_ratio = round_packet_stats["packet_loss"] / max(data_generated, 1)
    if collect_history:
        details = {
            "cluster_members": copy.deepcopy(cluster_members),
            "abandoned_nodes": list(abandoned),
            "abandoned_paths": copy.deepcopy(abandoned_paths),
            "paths": copy.deepcopy(paths),
            "dead_this_round": np.where(alive_before & ~alive2)[0].astype(int).tolist(),
            "data_generated": data_generated,
            "data_delivered_to_bs": int(data_delivered_to_bs),
            "packet_loss": int(round_packet_stats["packet_loss"]),
            "packet_loss_ratio": float(packet_loss_ratio),
        }
        return e2, alive2, int(data_delivered_to_bs), round_packet_stats, details
    return e2, alive2, int(data_delivered_to_bs), round_packet_stats


def _simulate_one_round(
    params: LeachParams,
    topo: Dict,
    E: np.ndarray,
    alive: np.ndarray,
    ch_idx: np.ndarray,
    Ds: float,
    Ncl_max: int,
    collect_history: bool = False,
) -> Tuple[np.ndarray, np.ndarray, int, Dict[str, int]]:
    return simulate_custom_leach_round(
        params=params,
        topo=topo,
        e=E,
        alive=alive,
        ch_idx=ch_idx,
        ds=Ds,
        ncl_max=Ncl_max,
        collect_history=collect_history,
    )