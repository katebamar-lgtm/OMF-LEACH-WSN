"""
Regenerates the figures of the OMF-LEACH paper with print-legible text and
black-and-white-safe patterns.

* Fig. 1, 2, 3, 5 are drawn from the mean and 95% CI half-width values printed in
  Tables VI, VII, VIII and X of the paper.
* Fig. 4 uses the means of Table IX; its 95% CI half-widths (CI_FIG4) were computed
  from the 10 raw repetitions per (method, Na) in results_timing_80/call_timing_raw.csv.

Sizes follow a two-column journal layout: one column = 3.45 in (8.8 cm),
full width = 7.16 in (18.2 cm). Minimum text size at final print size: 7.5 pt.
Each figure is saved as 600-dpi PNG and as vector PDF.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import os
OUT = os.environ.get("FIG_OUT", "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "hatch.linewidth": 0.7,
    "pdf.fonttype": 42,   # embed TrueType fonts (no Type 3)
    "ps.fonttype": 42,
})

# One style per method, identical in every figure.
# Fills differ in luminance AND carry a different hatch -> distinguishable in B&W.
STYLE = {
    "Improved LEACH": dict(facecolor="#d9d9d9", hatch="",     edgecolor="black"),
    "NSGA-II":        dict(facecolor="#e69f00", hatch="////", edgecolor="black"),
    "OMF (proposed)": dict(facecolor="#0b3c6f", hatch="",     edgecolor="black"),
    "MOPSO":          dict(facecolor="#ffffff", hatch="xxxx", edgecolor="black"),
}
ERR = dict(elinewidth=0.9, capsize=2.5, capthick=0.9, ecolor="black")
W1, W2 = 3.45, 7.16


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- Fig. 1 (Table VI)
def fig1():
    methods = ["Improved LEACH", "NSGA-II", "OMF (proposed)", "MOPSO"]
    mean = {"Improved LEACH": [312.4, 873.8, 1029.8],
            "NSGA-II":        [332.2, 1070.5, 1672.8],
            "OMF (proposed)": [359.2, 1066.8, 1673.4],
            "MOPSO":          [316.1, 1075.6, 1632.0]}
    ci = {"Improved LEACH": [12.7, 21.6, 4.9],
          "NSGA-II":        [26.8, 11.4, 21.4],
          "OMF (proposed)": [22.8, 9.1, 19.3],
          "MOPSO":          [37.8, 8.3, 21.2]}
    fig, ax = plt.subplots(figsize=(W1, 2.75))
    x = np.arange(3)
    w = 0.2
    for i, m in enumerate(methods):
        ax.bar(x + (i - 1.5) * w, mean[m], w, yerr=ci[m], label=m,
               linewidth=0.7, error_kw=ERR, **STYLE[m])
    ax.set_xticks(x)
    ax.set_xticklabels(["FND", "HND", "LND"])
    ax.set_ylabel("Round")
    ax.set_ylim(0, 2000)
    ax.legend(loc="upper left", frameon=False, ncol=1, handlelength=1.6,
              borderaxespad=0.2)
    style_axes(ax)
    save(fig, "Fig1_lifetime")


# ---------------------------------------------------------------- Fig. 2 (Table VII)
def fig2():
    methods = ["Improved LEACH", "NSGA-II", "OMF (proposed)", "MOPSO"]
    mean = [99.74, 98.45, 98.50, 98.51]
    ci = [0.02, 0.04, 0.05, 0.04]
    fig, ax = plt.subplots(figsize=(W1, 2.55))
    for i, m in enumerate(methods):
        ax.bar(i, mean[i], 0.62, yerr=ci[i], linewidth=0.7, error_kw=ERR, **STYLE[m])
    ax.set_xticks(range(4))
    ax.set_xticklabels(["Improved\nLEACH", "NSGA-II", "OMF\n(proposed)", "MOPSO"])
    ax.set_ylabel("Packet delivery ratio (%)")
    ax.set_ylim(97.5, 100.0)
    style_axes(ax)
    save(fig, "Fig2_pdr")


# ---------------------------------------------------------------- Fig. 3 (Table VIII)
def fig3():
    methods = ["Improved LEACH", "NSGA-II", "OMF (proposed)", "MOPSO"]
    mean = [0.0388, 0.0239, 0.0239, 0.0245]
    ci = [0.0002, 0.0003, 0.0003, 0.0003]
    fig, ax = plt.subplots(figsize=(W1, 2.55))
    for i, m in enumerate(methods):
        ax.bar(i, mean[i], 0.62, yerr=ci[i], linewidth=0.7, error_kw=ERR, **STYLE[m])
    ax.set_xticks(range(4))
    ax.set_xticklabels(["Improved\nLEACH", "NSGA-II", "OMF\n(proposed)", "MOPSO"])
    ax.set_ylabel("Average energy per round (J)")
    ax.set_ylim(0, 0.045)
    style_axes(ax)
    save(fig, "Fig3_energy_per_round")


# ---------------------------------------------------------------- Fig. 4 (Table IX)
CI_FIG4 = {'MOPSO': [14.6, 12.6, 5.7, 11.1], 'NSGA-II': [14.1, 4.9, 5.8, 4.9], 'OMF (proposed)': [15.0, 12.2, 12.3, 4.6]}  # 95% CI half-widths (ms), t(0.975, 9)*sd/sqrt(10), from results_timing_80/call_timing_raw.csv; order Na = 80, 60, 40, 20


def fig4():
    na = ["80", "60", "40", "20"]
    mean = {"NSGA-II":        [459.3, 356.1, 266.7, 184.7],
            "OMF (proposed)": [426.5, 321.0, 232.3, 152.3],
            "MOPSO":          [519.3, 404.0, 323.0, 330.0]}
    with_ci = CI_FIG4 is not None
    if not with_ci:
        print("Fig. 4: CI_FIG4 is None -> bars drawn WITHOUT error bars "
              "(caption must not say 'error bars: 95% CI').")
    fig, ax = plt.subplots(figsize=(W1, 2.75))
    x = np.arange(4)
    w = 0.26
    for i, m in enumerate(["NSGA-II", "OMF (proposed)", "MOPSO"]):
        kw = dict(yerr=CI_FIG4[m], error_kw=ERR) if with_ci else {}
        ax.bar(x + (i - 1) * w, mean[m], w, label=m, linewidth=0.7, **kw, **STYLE[m])
    ax.set_xticks(x)
    ax.set_xticklabels([f"$N_a$ = {v}" for v in na])
    ax.set_ylabel("Time per CH-selection call (ms)")
    ax.set_ylim(0, 650)
    ax.legend(loc="upper right", frameon=False, handlelength=1.6)
    style_axes(ax)
    save(fig, "Fig4_time_per_call")


# ---------------------------------------------------------------- Fig. 5 (Table X)
def fig5():
    methods = ["NSGA-II", "OMF (proposed)", "MOPSO"]
    data = {  # (mean, CI half-width) per method
        "Hypervolume (higher is better)":
            {"NSGA-II": (1.0931, 0.0208), "OMF (proposed)": (1.1047, 0.0031), "MOPSO": (1.0976, 0.0130)},
        "IGD (lower is better)":
            {"NSGA-II": (0.2380, 0.0870), "OMF (proposed)": (0.2371, 0.0888), "MOPSO": (0.2530, 0.0868)},
        "Spread (lower is better)":
            {"NSGA-II": (1.4152, 0.0469), "OMF (proposed)": (1.4523, 0.0488), "MOPSO": (1.4202, 0.0647)},
        "GD (lower is better)":
            {"NSGA-II": (0.0398, 0.0199), "OMF (proposed)": (0.0357, 0.0112), "MOPSO": (0.0498, 0.0349)},
    }
    fig, axes = plt.subplots(2, 2, figsize=(W2, 4.3))
    for k, (ax, (title, d)) in enumerate(zip(axes.ravel(), data.items())):
        for i, m in enumerate(methods):
            mu, c = d[m]
            ax.bar(i, mu, 0.6, yerr=c, linewidth=0.7, error_kw=ERR, **STYLE[m])
        ax.set_xticks(range(3))
        ax.set_xticklabels(["NSGA-II", "OMF (proposed)", "MOPSO"])
        ax.set_ylabel(title)
        ax.text(0.0, 1.02, f"({'abcd'[k]})", transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="bottom")
        style_axes(ax)
    axes[0, 0].set_ylim(0, 1.3)
    axes[0, 1].set_ylim(0, 0.45)
    axes[1, 0].set_ylim(0, 1.7)
    axes[1, 1].set_ylim(0, 0.10)
    fig.tight_layout(h_pad=1.4, w_pad=1.6)
    save(fig, "Fig5_pareto_indicators")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5()
    print("done")
