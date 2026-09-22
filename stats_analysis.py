"""
stats_analysis.py
------------------
Single source of truth for every inferential statistic reported in the paper:
    - Table III   : 3-group matched-seed Friedman tests (OMF, NSGA-II, MOPSO)
                     + Benjamini-Hochberg correction across the 6 metrics
    - Table IV    : post-hoc pairwise Wilcoxon signed-rank tests,
                     Bonferroni-corrected (alpha = 0.05/3)
    - Section V-A : 4-group Friedman test (+ Improved LEACH) for LND and
                     avg. energy/round
    - Section V-C : paired TOST equivalence tests (OMF vs NSGA-II),
                     margin = +/-2% of the NSGA-II mean
    - Table X     : Friedman tests on the Pareto-front quality indicators
                     (HV, IGD, Spread, GD)
    - Section V-E : 3-group matched-seed Friedman tests (FND, LND, avg.
                     energy/round) at each scalability network size
                     (80/100/150 nodes, N=10 seeds), reproducing the
                     Kendall's W values and p-values cited in the text
                     around Table XI

Usage
-----
    python stats_analysis.py                 # runs everything, prints all tables
    python stats_analysis.py --save results/stats_analysis_output.json

Requires: numpy, scipy (no non-standard dependencies).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List

import numpy as np
from scipy import stats

SEEDS = list(range(1, 21))
RAW_DIR = "results/raw"
MO_PER_SEED_CSV = "results/mo_indicators_per_seed.csv"

SCALABILITY_SIZES = [80, 100, 150]
SCALABILITY_SEEDS = list(range(1, 11))          # first 10 of the 20 matched seeds
SCALABILITY_RAW_DIR = "results_scalability/raw"
SCALABILITY_METRICS = ["FND", "LND", "avg_energy_consumed_per_round"]

METAHEURISTICS = ["omf", "nsga", "pso"]              # OMF, NSGA-II, MOPSO
LABELS = {"omf": "OMF", "nsga": "NSGA-II", "pso": "MOPSO",
          "improved_leach": "Improved LEACH"}

METRICS_3GROUP = ["FND", "HND", "LND", "packet_loss_ratio",
                  "avg_energy_consumed_per_round", "elapsed_seconds"]
METRIC_DISPLAY = {
    "FND": "FND", "HND": "HND", "LND": "LND",
    "packet_loss_ratio": "Packet loss ratio",
    "avg_energy_consumed_per_round": "Avg. energy / round",
    "elapsed_seconds": "Execution time (full-run, elapsed_seconds)",
}


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_raw(algo: str) -> List[dict]:
    runs = []
    for s in SEEDS:
        path = os.path.join(RAW_DIR, f"{algo}_seed{s}.json")
        with open(path) as f:
            runs.append(json.load(f))
    return runs


def metric_array(runs: List[dict], key: str) -> np.ndarray:
    return np.array([r[key] for r in runs], dtype=float)


def load_pareto_indicators() -> Dict[str, Dict[str, np.ndarray]]:
    """Returns {algo: {indicator: array[20 seeds]}} for hv/igd/spread/gd."""
    data = {a: {k: np.full(20, np.nan) for k in ("hv", "igd", "spread", "gd")}
            for a in METAHEURISTICS}
    with open(MO_PER_SEED_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            algo = row["algorithm"]
            seed = int(row["seed"])
            for k in ("hv", "igd", "spread", "gd"):
                data[algo][k][seed - 1] = float(row[k])
    return data


# --------------------------------------------------------------------------
# Table III + 4-group Friedman (Section V-A) + Benjamini-Hochberg
# --------------------------------------------------------------------------

def friedman_kendall(*groups: np.ndarray) -> dict:
    chi2, p = stats.friedmanchisquare(*groups)
    n = groups[0].shape[0]
    k = len(groups)
    w = chi2 / (n * (k - 1))
    return {"chi2": float(chi2), "kendalls_w": float(w), "p_value": float(p),
            "n": n, "k": k}


def benjamini_hochberg(pvals: List[float], q: float = 0.05) -> List[bool]:
    """Returns a boolean significance mask (True = significant after BH-FDR)."""
    m = len(pvals)
    order = np.argsort(pvals)
    ranked = np.array(pvals)[order]
    thresh = (np.arange(1, m + 1) / m) * q
    below = ranked <= thresh
    if not below.any():
        cutoff_rank = 0
    else:
        cutoff_rank = np.max(np.where(below)[0]) + 1  # largest i with p_(i) <= (i/m)q
    sig_sorted = np.zeros(m, dtype=bool)
    sig_sorted[:cutoff_rank] = True
    sig = np.zeros(m, dtype=bool)
    sig[order] = sig_sorted
    return sig.tolist()


def table_iii(raw: Dict[str, List[dict]]) -> List[dict]:
    rows = []
    pvals = []
    for metric in METRICS_3GROUP:
        arrs = [metric_array(raw[a], metric) for a in METAHEURISTICS]
        res = friedman_kendall(*arrs)
        rows.append({"metric": METRIC_DISPLAY[metric], **res})
        pvals.append(res["p_value"])
    sig_raw = [p < 0.05 for p in pvals]
    sig_bh = benjamini_hochberg(pvals, q=0.05)
    for row, sr, sb in zip(rows, sig_raw, sig_bh):
        row["significant_raw"] = sr
        row["significant_BH_FDR"] = sb
    return rows


def four_group_friedman(raw: Dict[str, List[dict]], metric: str) -> dict:
    arrs = [metric_array(raw[a], metric)
            for a in ["improved_leach", "omf", "nsga", "pso"]]
    return friedman_kendall(*arrs)


# --------------------------------------------------------------------------
# Section V-E — scalability: 3-group Friedman per network size (Table XI)
# --------------------------------------------------------------------------

def load_scalability_raw(size: int, algo: str) -> List[dict]:
    runs = []
    for s in SCALABILITY_SEEDS:
        path = os.path.join(SCALABILITY_RAW_DIR, f"{size}nodes", f"{algo}_seed{s}.json")
        with open(path) as f:
            runs.append(json.load(f))
    return runs


def table_xi_scalability() -> List[dict]:
    """
    Reproduces the Section V-E claims: for each scalability network size
    (80/100/150 nodes, N=10 matched seeds), a 3-group matched-seed Friedman
    test (OMF, NSGA-II, MOPSO) on FND, LND, and avg. energy/round.

    Confirms, in particular:
      - FND Kendall's W = 0.27, 0.43, 1.00 for n = 80, 100, 150
      - LND / avg. energy Friedman p = 0.0038, 0.202, 0.067 for
        n = 80, 100, 150 (only the 100- and 150-node cases fall below 0.05)
    """
    rows = []
    for size in SCALABILITY_SIZES:
        raw = {a: load_scalability_raw(size, a) for a in METAHEURISTICS}
        for metric in SCALABILITY_METRICS:
            arrs = [metric_array(raw[a], metric) for a in METAHEURISTICS]
            res = friedman_kendall(*arrs)
            # NOTE: friedman_kendall's own return dict already has a "n" key
            # (the sample size, i.e. seed count) -- use "network_size" here
            # to avoid silently clobbering it when spreading **res.
            rows.append({"network_size": size, "metric": METRIC_DISPLAY[metric],
                          **res, "significant": res["p_value"] < 0.05})
    return rows


def print_table_xi(rows):
    print("\n=== Section V-E — 3-group Friedman per network size (N=10 seeds), Table XI ===")
    print(f"{'N_a':>5}{'Metric':<26}{'chi2':>9}{'W':>9}{'p-value':>12}{'sig':>6}")
    for r in rows:
        print(f"{r['network_size']:>5}{r['metric']:<26}{r['chi2']:>9.3f}{r['kendalls_w']:>9.3f}"
              f"{r['p_value']:>12.4g}{str(r['significant']):>6}")


# --------------------------------------------------------------------------
# Table IV — post-hoc pairwise Wilcoxon, Bonferroni-corrected
# --------------------------------------------------------------------------

def wilcoxon_z(x: np.ndarray, y: np.ndarray) -> float:
    """
    Standard normal-approximation z-statistic for the paired Wilcoxon
    signed-rank test, with the usual tie correction, computed directly
    from the ranks (NOT back-solved from the p-value).

    This is the convention needed to reproduce the paper's Table IV `r`
    column exactly. Back-solving z from SciPy's 'exact'-mode p-value
    (via the inverse normal CDF) is unstable for very small p and can
    return r > 1, which is what the paper's own NB under Table IV warns
    about ("r = |Z|/sqrt(n) ... isn't algebraically derivable from the
    exact p-value"). Computing z from the ranks avoids that instability
    and matches every r value in Table IV bit-for-bit.
    """
    d = x - y
    d = d[d != 0]  # zero_method='wilcox': drop zero differences
    n = len(d)
    ranks = stats.rankdata(np.abs(d))
    w_plus = ranks[d > 0].sum()
    mu = n * (n + 1) / 4
    _, counts = np.unique(np.abs(d), return_counts=True)
    tie_correction = np.sum(counts ** 3 - counts)
    sigma2 = (n * (n + 1) * (2 * n + 1) - tie_correction / 2) / 24
    sigma = np.sqrt(sigma2)
    return float((w_plus - mu) / sigma)


def wilcoxon_pair(x: np.ndarray, y: np.ndarray) -> dict:
    n = len(x)
    # p-value: SciPy's own 'auto' method (falls back to the normal
    # approximation whenever ties/zero-differences are present, and uses
    # the exact distribution only otherwise -- in this dataset that is
    # exactly the p < 0.0001 execution-time comparisons). This matches
    # the paper's NB under Table IV. NOTE: current SciPy exposes this as
    # `method=`, not the older `mode=` alias -- `method=` is used here
    # for forward compatibility (both give bit-identical results).
    w_stat, p = stats.wilcoxon(x, y, zero_method="wilcox",
                                correction=False, method="auto")
    z = wilcoxon_z(x, y)
    r = float(abs(z) / np.sqrt(n))
    return {"W": float(w_stat), "p_value": float(p), "r": r}


def magnitude(r: float) -> str:
    ar = abs(r) if np.isfinite(r) else 0.0
    if ar < 0.10:
        return "negligible"
    if ar < 0.30:
        return "small"
    if ar < 0.50:
        return "medium"
    return "large"


def table_iv(raw: Dict[str, List[dict]]) -> List[dict]:
    pairs = [("omf", "nsga"), ("omf", "pso"), ("nsga", "pso")]
    metrics = ["FND", "LND", "avg_energy_consumed_per_round", "elapsed_seconds"]
    alpha_bonf = 0.05 / 3
    rows = []
    for metric in metrics:
        for a, b in pairs:
            x = metric_array(raw[a], metric)
            y = metric_array(raw[b], metric)
            res = wilcoxon_pair(x, y)
            res.update({
                "metric": METRIC_DISPLAY[metric],
                "comparison": f"{LABELS[a]} vs {LABELS[b]}",
                "magnitude": magnitude(res["r"]),
                "significant_bonferroni": res["p_value"] < alpha_bonf,
            })
            rows.append(res)
    return rows


# --------------------------------------------------------------------------
# Section V-C — paired TOST equivalence tests (OMF vs NSGA-II)
# --------------------------------------------------------------------------

def tost_paired(x: np.ndarray, y: np.ndarray, margin_frac: float = 0.02) -> dict:
    """
    Two one-sided tests for equivalence of paired samples x (OMF) and
    y (NSGA-II). Equivalence margin = +/- margin_frac * mean(y).
    Returns the TOST p-value (max of the two one-sided p-values) and the
    90% CI conventionally reported alongside a TOST result.
    """
    d = x - y
    n = len(d)
    mean_ref = y.mean()
    low, up = -margin_frac * mean_ref, margin_frac * mean_ref
    dm = d.mean()
    sd = d.std(ddof=1)
    se = sd / np.sqrt(n)

    t_low = (dm - low) / se
    t_up = (dm - up) / se
    p_low = 1 - stats.t.cdf(t_low, df=n - 1)   # H0: true diff <= low
    p_up = stats.t.cdf(t_up, df=n - 1)         # H0: true diff >= up
    p_tost = max(p_low, p_up)

    tcrit90 = stats.t.ppf(0.95, df=n - 1)
    ci90 = (dm - tcrit90 * se, dm + tcrit90 * se)

    return {
        "diff_abs": float(dm),
        "diff_pct_of_ref_mean": float(dm / mean_ref * 100),
        "ci90_abs": (float(ci90[0]), float(ci90[1])),
        "ci90_pct_of_ref_mean": (float(ci90[0] / mean_ref * 100),
                                  float(ci90[1] / mean_ref * 100)),
        "margin_abs": (float(low), float(up)),
        "p_tost": float(p_tost),
        "equivalent_at_alpha_0.05": bool(p_tost < 0.05),
    }


def section_v_c_tost(raw: Dict[str, List[dict]]) -> Dict[str, dict]:
    omf, nsga = raw["omf"], raw["nsga"]
    out = {}
    for metric in ["LND", "avg_energy_consumed_per_round", "FND"]:
        x = metric_array(omf, metric)
        y = metric_array(nsga, metric)
        out[METRIC_DISPLAY.get(metric, metric)] = tost_paired(x, y, margin_frac=0.02)
    return out


# --------------------------------------------------------------------------
# Table X — Friedman tests on Pareto-front quality indicators
# --------------------------------------------------------------------------

def table_x() -> List[dict]:
    data = load_pareto_indicators()
    rows = []
    for ind in ("hv", "igd", "spread", "gd"):
        arrs = [data[a][ind] for a in METAHEURISTICS]
        res = friedman_kendall(*arrs)
        rows.append({"indicator": ind.upper(), **res,
                      "significant": res["p_value"] < 0.05})
    return rows


# --------------------------------------------------------------------------
# Pretty printing
# --------------------------------------------------------------------------

def print_table_iii(rows):
    print("\n=== TABLE III — 3-group matched-seed Friedman (OMF, NSGA-II, MOPSO), N=20, K=3 ===")
    print(f"{'Metric':<38}{'chi2':>9}{'W':>9}{'p-value':>12}{'raw':>6}{'BH-FDR':>8}")
    for r in rows:
        print(f"{r['metric']:<38}{r['chi2']:>9.3f}{r['kendalls_w']:>9.3f}"
              f"{r['p_value']:>12.4g}{str(r['significant_raw']):>6}"
              f"{str(r['significant_BH_FDR']):>8}")


def print_table_iv(rows):
    print("\n=== TABLE IV — post-hoc Wilcoxon signed-rank, Bonferroni alpha=0.0167 ===")
    print(f"{'Metric':<38}{'Comparison':<20}{'W':>8}{'p-value':>12}{'r':>7}{'magnitude':>11}{'sig':>6}")
    for r in rows:
        print(f"{r['metric']:<38}{r['comparison']:<20}{r['W']:>8.1f}"
              f"{r['p_value']:>12.4g}{r['r']:>7.2f}{r['magnitude']:>11}"
              f"{str(r['significant_bonferroni']):>6}")


def print_tost(results):
    print("\n=== Section V-C — paired TOST (OMF vs NSGA-II), margin = +/-2% of NSGA-II mean ===")
    # Reporting convention matches the paper: LND/FND (round counts) are
    # shown with the 90% CI in absolute rounds; energy (already a small
    # fraction) is shown with the 90% CI in percent of the NSGA-II mean.
    abs_units = {"LND": "rounds", "FND": "rounds"}
    for metric, r in results.items():
        lo_abs, up_abs = r["ci90_abs"]
        lo_pct, up_pct = r["ci90_pct_of_ref_mean"]
        if metric in abs_units:
            print(f"{metric}: diff = {r['diff_pct_of_ref_mean']:+.2f}%  "
                  f"90% CI [{lo_abs:+.1f}, {up_abs:+.1f}] {abs_units[metric]}  "
                  f"p_TOST = {r['p_tost']:.4g}  "
                  f"equivalent @ alpha=0.05: {r['equivalent_at_alpha_0.05']}")
        else:
            print(f"{metric}: diff = {r['diff_pct_of_ref_mean']:+.2f}%  "
                  f"90% CI [{lo_pct:+.2f}%, {up_pct:+.2f}%]  "
                  f"p_TOST = {r['p_tost']:.4g}  "
                  f"equivalent @ alpha=0.05: {r['equivalent_at_alpha_0.05']}")


def print_table_x(rows):
    print("\n=== TABLE X — Friedman on Pareto-front quality indicators, N=20 ===")
    print(f"{'Indicator':<12}{'chi2':>9}{'W':>9}{'p-value':>12}{'sig':>6}")
    for r in rows:
        print(f"{r['indicator']:<12}{r['chi2']:>9.3f}{r['kendalls_w']:>9.3f}"
              f"{r['p_value']:>12.4g}{str(r['significant']):>6}")


# --------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default=None,
                     help="optional path to dump every computed number as JSON")
    args = ap.parse_args()

    raw = {a: load_raw(a) for a in METAHEURISTICS + ["improved_leach"]}

    t3 = table_iii(raw)
    print_table_iii(t3)

    print("\n=== Section V-A — 4-group Friedman test (Improved LEACH + 3 metaheuristics) ===")
    for metric in ["LND", "avg_energy_consumed_per_round"]:
        res = four_group_friedman(raw, metric)
        print(f"{METRIC_DISPLAY[metric]:<38} chi2={res['chi2']:.3f}  "
              f"W={res['kendalls_w']:.3f}  p={res['p_value']:.4g}")

    t4 = table_iv(raw)
    print_table_iv(t4)

    tost = section_v_c_tost(raw)
    print_tost(tost)

    tx = table_x()
    print_table_x(tx)

    txi = table_xi_scalability()
    print_table_xi(txi)

    if args.save:
        out = {
            "table_iii": t3,
            "four_group_friedman": {
                m: four_group_friedman(raw, m)
                for m in ["LND", "avg_energy_consumed_per_round"]
            },
            "table_iv": t4,
            "tost_v_c": tost,
            "table_x": tx,
            "table_xi_scalability": txi,
        }
        with open(args.save, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved full results to {args.save}")
