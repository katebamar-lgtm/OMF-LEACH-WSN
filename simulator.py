from __future__ import annotations

import copy
from typing import Any, Dict, Optional

import numpy as np


class MyWSNSimulator:
    """Pure replay simulator for one saved algorithm history."""

    def __init__(self, history: Dict[str, Any]):
        self.history = history
        self.topo = history["topo"]
        self.params = history.get("params")
        self.rounds = history.get("rounds", [])
        self.current_round = 0

    def has_next(self) -> bool:
        return self.current_round < len(self.rounds)

    def step(self) -> Optional[Dict[str, Any]]:
        if not self.has_next():
            return None

        data = self.rounds[self.current_round]
        self.current_round += 1
        return self._copy_round(data)

    def get_round(self, round_index: int) -> Optional[Dict[str, Any]]:
        if round_index < 0 or round_index >= len(self.rounds):
            return None
        return self._copy_round(self.rounds[round_index])

    def reset(self) -> None:
        self.current_round = 0

    def _copy_round(self, data: Dict[str, Any]) -> Dict[str, Any]:
        alive = np.asarray(data.get("alive", []), dtype=bool).copy()
        energies = np.asarray(data.get("energies", []), dtype=float).copy()
        alive_before = np.asarray(data.get("alive_before", alive), dtype=bool).copy()
        energies_before = np.asarray(data.get("energies_before", energies), dtype=float).copy()

        return {
            "round": int(data.get("round", self.current_round)),
            "ch_idx": np.asarray(data.get("ch_idx", []), dtype=int).copy(),
            "cluster_members": copy.deepcopy(data.get("cluster_members", {})),
            "abandoned_nodes": np.asarray(data.get("abandoned_nodes", []), dtype=int).copy(),
            "abandoned_paths": copy.deepcopy(data.get("abandoned_paths", {})),
            "paths": copy.deepcopy(data.get("paths", {})),
            "alive_before": alive_before,
            "energies_before": energies_before,
            "alive": alive,
            "energies": energies,
            "packets": int(data.get("packets", 0)),
            "data_generated": int(data.get("data_generated", 0)),
            "data_delivered_to_bs": int(data.get("data_delivered_to_bs", data.get("packets", 0))),
            "packet_loss": int(data.get("packet_loss", 0)),
            "packet_loss_ratio": float(data.get("packet_loss_ratio", 0.0)),
            "dead_this_round": np.asarray(data.get("dead_this_round", []), dtype=int).copy(),
            "packet_stats": dict(data.get("packet_stats", {})),
        }