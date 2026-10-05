# OMF-LEACH — Simulation code and data

Code and data accompanying the paper **"Optimization by Morphological Filters for Multi-Objective
Cluster-Head Selection in Wireless Sensor Networks"** (Advances in Electrical and Computer
Engineering). All numerical results of the paper are **simulation outputs** of the simulator in this
repository; no field measurements were performed.

## 1. Methods compared

| Label in the paper | Description | Main file |
|---|---|---|
| Improved LEACH | Closed-form energy-aware threshold, fixed N_cl = 20, greedy multi-hop routing | `ImprovedLEACH.py` |
| OMF (proposed) | Multi-objective OMF, 3 objectives (energy, max CH-to-member distance, packet loss) | `MOomf.py` |
| NSGA-II | pymoo NSGA-II, same objectives, same budget (600 evaluations/call) | `nsga2.py` |
| MOPSO | pymoo MOPSO-CD, same objectives, same budget | `pso_leach.py` |
| Random CH (ref.) | Random selection, same simulator, K rule, Dijkstra routing | `reference_heuristics.py` |
| Top-K energy (ref.) | Top-K residual-energy selection, same simulator | `reference_heuristics.py` |

The three metaheuristics and the two reference heuristics share the simulator, the K rule
(K = round(P·N_a)), minimum-energy Dijkstra inter-cluster routing and N_cl = [N_a/K].
Improved LEACH differs in routing and cluster size (comparisons with it are indicative only).

## 2. Campaigns and mapping to the paper

| Campaign | Command | Output | Paper |
|---|---|---|---|
| Main campaign: Improved LEACH, NSGA-II, MOPSO, OMF — 20 seeds, 80 nodes, run until depletion | `python batch_runner.py --n-seeds 20 --out results` | `results/` | Tables VI–VIII, X (V-B, V-D) |
| Reference heuristics: Random CH, Top-K — same 20 seeds | `python reference_heuristics.py --seeds 20 --out results_reference` | `reference_heuristics_raw.csv`, `reference_heuristics_summary.csv` | Rows "ref." of Tables VI and VIII, Section VI |
| Statistics (Friedman, BH-FDR, Wilcoxon, TOST, indicators) | `python stats_analysis.py` | console / `--save` | Tables IV, V, XI; Sections V-A, V-B |
| Per-call timing, 80 nodes, N_a ∈ {80, 60, 40, 20}, 10 repetitions, interleaved | `python bench_call_timing.py --reps 10 --out results_timing_80` | `call_timing_raw.csv` | Table IX, Fig. 4 |
| Scalability (100 and 150 nodes, seeds 1–10) | `python scalability_runner.py --nodes 80 100 150 --n-seeds 10 --out results_scalability` | `scalability_per_run.csv`, `scalability_summary.csv` | Table XII, Section V-E |
| Timing at larger sizes | `python bench_call_timing.py --n-nodes 100 --out results_timing_100` (and 150 nodes with `--area-w 200 --area-h 200 --bs-x 100 --bs-y 300`) | `results_timing_100/`, `results_timing_150/` | Section V-E |
| Proof of Table VIII | `TableVIII_per_seed_proof.csv` | — | Section V-B |

Run `python check_environment.py` first (requires Python 3, numpy, scipy, pymoo).

## 3. Network and parameters

80 nodes (100 × 100 m), base station at (50, 150), initial energy 0.5 J per node, 4000-bit packets,
E_elec = 50 nJ/bit, ε_fs = 10 pJ/bit/m², ε_mp = 0.0013 pJ/bit/m⁴, E_DA = 5 nJ/bit/signal,
d₀ = 87.7 m, P = 0.05. NSGA-II and MOPSO: population/swarm 30 × 20 iterations (600 evaluations);
OMF: NF = 10, NN = 6, IT = 10, R = 0.7, FS_init = 0.5, FS_decay = 0.5, ε = 10⁻³
(10 initial + 600 = 610 evaluations per call). See Tables II–III of the paper.

## 4. Data dictionary (important)

* `FND`, `HND`, `LND`: rounds at which the first / half / last node dies.
* `total_packet_loss`: packets lost over the whole run.
* **`total_packets`: packets successfully delivered to the base station.**
  *Packets generated* (Table VII) = `total_packets` + `total_packet_loss`.
* `packet_loss_ratio`: `total_packet_loss` / (`total_packets` + `total_packet_loss`).
* `avg_energy_consumed_per_round`: total consumed energy / number of rounds. Because every run is
  continued until full depletion, `avg_energy_consumed_per_round × LND = N × 0.5 J` (40 J at 80 nodes,
  50 J at 100 nodes, 75 J at 150 nodes) — see `TableVIII_per_seed_proof.csv`.
* `elapsed_seconds`: wall-clock time of a complete run (separate, non-interleaved campaigns; not used
  for ranking — see Table IX / `call_timing_raw.csv` for the controlled per-call benchmark).

## 5. Reproducibility notes

* Seeds 1–20 define node placement and channel realization and are shared by all methods
  (matched design). Scalability uses seeds 1–10.
* Per-call evaluations: OMF 610, NSGA-II 600, MOPSO 630 (21 evaluations of the 30-particle swarm).
* The Pareto fronts used for HV / IGD / GD / Spread are stored per seed in `mo_indicators_per_seed.csv`.
  OMF's archive keeps all non-dominated candidates evaluated during the search, whereas NSGA-II and
  MOPSO report their final population / repository (see Section IV-C of the paper).

## 6. Supplementary robustness check (not used in the paper's tables)

`ablation_selection_pool.py` re-runs NSGA-II with (i) the CH set deployed exactly as evaluated
(`--variant exact`) and (ii) a non-dominated archive of all evaluated candidates (`--variant archive`),
to test whether the selection pool size or the random repair in `decode_particle` influences FND.
Results are in `results_ablation/`.

## 7. Citation / license

Please cite the paper and the Zenodo record (doi: 10.5281/zenodo.21933476).
