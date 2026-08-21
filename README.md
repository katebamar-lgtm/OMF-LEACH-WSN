# OMF-LEACH — Simulation Code

Code and data supporting:

> Kateb Hachemi Amar, A., Tahraoui, M. A., & Belmadani, A.
> *Optimization by Morphological Filters for Multi-Objective Cluster-Head Selection in
> Wireless Sensor Networks.*

This repository contains the Python simulator, the four cluster-head selection methods
compared in the paper (Improved LEACH, OMF-LEACH, NSGA-II, MOPSO), and the batch runner
used to produce all results reported in Sections 5.1–5.4 of the article.

---

## 1. Repository structure

```
WSN_SIM/
├── ImprovedLEACH.py       # Baseline protocol (Section 3.1) + shared network/energy/
│                          # routing model (Section 3.2, Eqs. 3–4, d0, radio parameters)
├── CustomLEACH.py         # Shared round simulation engine: cluster formation, multi-hop
│                          # routing, objective evaluation f1/f2/f3 (Eqs. 4–6)
├── MOomf.py                # Proposed OMF-LEACH algorithm (Section 3.4, Algorithm 1)
├── nsga2.py                # NSGA-II baseline via pymoo (Table 2 parameters)
├── pso_leach.py             # MOPSO baseline via pymoo (Table 2 parameters)
├── solution_selection.py    # Ideal-point / knee-point compromise selection (Eq. 8)
├── simulator.py              # Lightweight replay simulator for a saved run history
├── batch_runner.py           # Main entry point: runs all seeds × algorithms, computes
│                          # network-level metrics and Pareto quality indicators
│                          # (HV, IGD, Spread, GD — Section 4.3)
└── experiments.py            # Experiment orchestration helpers
```

## 2. Requirements

- Python ≥ 3.10
- `numpy`
- `pymoo` (NSGA-II, MOPSO, and Hypervolume/IGD indicators)

```bash
pip install numpy pymoo
```

No other third-party dependency is required; the OMF implementation (`MOomf.py`) is
native Python/NumPy, consistent with the implementation-asymmetry note in Section 4.1
of the paper (pymoo's vectorized NSGA-II/MOPSO vs. OMF's non-vectorized native code).

## 3. Reproducing the paper's results

The full experimental campaign (4 algorithms × 20 matched seeds = 80 full-depletion
simulations, Section 5.1) is launched with:

```bash
python batch_runner.py \
    --algorithms improved_leach omf nsga pso \
    --n-seeds 20 \
    --base-seed 1 \
    --n-rounds 2500 \
    --hv-ref-factor 1.1 \
    --out results
```

| Argument | Default | Meaning |
|---|---|---|
| `--algorithms` | `improved_leach omf nsga pso` | Which methods to run |
| `--n-seeds` | `20` | Number of matched seeds (Section 5.1) |
| `--base-seed` | `1` | First seed; seeds used are `base_seed, ..., base_seed + n_seeds - 1` |
| `--n-rounds` | `2500` | Round cap per run (network lifetime averages ≈1030–1673 rounds, well under this cap — see Table 5) |
| `--workers` | all cores − 1 | Parallel worker processes (one per (algorithm, seed) task) |
| `--hv-ref-factor` | `1.1` | Nadir-point multiplier for the Hypervolume reference point (Section 4.3: "inflated by 10% per dimension") |
| `--out` | `results` | Output directory |

All algorithm-specific parameters (NF/NN/IT/R for OMF; population/generations/Pc/Pm for
NSGA-II; swarm/iterations/ω/c1/c2/repository for MOPSO — Table 2) are set as dataclass
defaults in `MOomf.py`, `nsga2.py`, and `pso_leach.py` respectively, and match the paper
exactly. Radio/network parameters (Table 1) are set as defaults in `ImprovedLEACH.py`'s
`LeachParams` dataclass.

Runtime note: the full campaign takes roughly 1 hour on a modern multi-core machine
(dominated by NSGA-II/MOPSO's per-round re-optimization, consistent with Table 8's
per-run execution times of ~13–27 minutes per seed, ×20 seeds, ×3 metaheuristics).
Results are cached: re-running `batch_runner.py` skips any `(algorithm, seed)` pair
whose output files already exist in `--out`.

## 4. Output files

Running `batch_runner.py` populates `--out` with:

```
results/
├── raw/<algorithm>_seed<N>.json         # Per-run scalar metrics (FND, HND, LND,
│                                        # packets, energy, elapsed_seconds, ...)
├── pareto/<algorithm>_seed<N>_pf.npy     # Aggregate non-dominated front for that
│                                        # run: array of shape (n_points, 3) with
│                                        # columns [f1_energy, f2_distance, f3_loss]
├── pf_ref.npy                            # Global reference front: non-dominated
│                                        # union of all OMF/NSGA-II/MOPSO solutions
│                                        # across all seeds (Section 4.3)
├── summary.csv                           # Network-level metrics, mean ± 95% CI
│                                        # per algorithm (feeds Tables 5–8)
├── mo_indicators.csv                     # HV/IGD/Spread/GD, mean ± 95% CI per
│                                        # algorithm (feeds Table 10)
└── mo_indicators_per_seed.csv            # Same, per seed (used for the matched-seed
                                         # Friedman/Wilcoxon tests, Tables 3, 4, 11)
```

`raw/*.json` and `mo_indicators_per_seed.csv` are the ground truth for every
statistical test in the paper; `summary.csv` and `mo_indicators.csv` are convenience
aggregates derived from them.

## 5. Mapping code → paper sections

| Code | Paper |
|---|---|
| `ImprovedLEACH.py` (`LeachParams`, `_tx_energy`, `_rx_energy`, `_agg_energy`, `_d0`) | Section 3.2, Eqs. 3, Table 1 |
| `CustomLEACH.py` (objective evaluation) | Section 3.3, Eqs. 4–6 |
| `MOomf.py` (`OMFParams`, main loop) | Section 3.4, Algorithm 1, Table 2 |
| `solution_selection.py` (`select_solution_index`, mode `"knee_point"`) | Eq. 8 |
| `nsga2.py` | Section 3.5 (T_NSGA-II), Table 2 |
| `pso_leach.py` | Section 3.5 (T_MOPSO), Table 2 |
| `batch_runner.py` (`_compute_hv`, `_compute_igd`, `_compute_spread`, `_compute_gd`) | Section 4.3, Table 10 |

Statistical analysis (Friedman/Wilcoxon tests, Tables 3, 4, 11; Benjamini–Hochberg
correction) is not included in this repository as executable code — it was performed
in a separate analysis script from `raw/*.json` and `mo_indicators_per_seed.csv` using
SciPy (`scipy.stats.friedmanchisquare`, `scipy.stats.wilcoxon`, `zero_method='wilcox'`,
no continuity correction; exact distribution used for p < 0.0001, per Table 4's note).

## 6. Data availability

The raw simulation outputs (80 runs) and aggregate CSVs used to produce all tables and
figures in the paper are archived alongside this code at https://doi.org/10.5281/zenodo.21933475.

## 7. Authors

- Kateb Hachemi Amar Amar — Dept. Computer Science, Faculty of Exact Sciences and Computer Science, University of Chlef -Hassiba Benbouali- Chlef, Algeria
- Benbrik Nihad
- Bouazdia Rania
- Tahraoui Mohamed Amine — Dept. Computer Science, Faculty of Exact Sciences and Computer Science, University of Chlef -Hassiba Benbouali- Chlef, Algeria
- Belmadani Abderrahim — Dept. Computer Science, Faculty of Mathematics and Computer Science, University of Science and Technology of Oran -Mohamed Boudiaf- Oran, Algeria

## 8. Citation

If you use this code, please cite:

```
[Full citation — to insert once accepted/published]
```

## 9. License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
