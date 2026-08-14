from __future__ import annotations

import numpy as np


SELECTION_MODES = {
    "knee_point": "Knee point selection",
    "energy_consumption": "Energy consumption",
    "distance": "Cluster communication distance",
    "packet_loss": "Packet loss",
}


def normalize_selection_mode(selection_mode: str | None) -> str:
    mode = (selection_mode or "knee_point").strip().lower()
    if mode not in SELECTION_MODES:
        raise ValueError(f"Unknown solution selection mode: {selection_mode}")
    return mode


def select_solution_index(objectives, selection_mode: str | None = "knee_point") -> int:
    objs = np.asarray(objectives, dtype=float)
    if objs.ndim == 1:
        objs = objs[None, :]
    if len(objs) == 0:
        raise ValueError("Cannot select from an empty objective matrix.")

    mode = normalize_selection_mode(selection_mode)
    # Objective order (Eq. 4-6 in the article): [0]=energy, [1]=distance, [2]=packet_loss.
    if mode == "energy_consumption":
        return int(np.argmin(objs[:, 0]))
    if mode == "distance":
        return int(np.argmin(objs[:, 1]))
    if mode == "packet_loss":
        return int(np.argmin(objs[:, 2]))

    min_vals = np.min(objs, axis=0)
    range_vals = np.ptp(objs, axis=0)
    range_vals[range_vals == 0] = 1e-12
    norm = (objs - min_vals) / range_vals
    return int(np.argmin(np.linalg.norm(norm, axis=1)))