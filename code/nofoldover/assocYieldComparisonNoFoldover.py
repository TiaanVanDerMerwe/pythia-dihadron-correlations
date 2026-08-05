"""
plot_yield_comparison.py
────────────────────────
Loads integrated yields from four Pythia shower configurations:
  default, nofsr, noisr, nofsrnoisr

Plots near-side and away-side yields together, plus ratio panels
showing default / each variant (default / nofsr, default / noisr, default / nofsrnoisr).

Usage
-----
  python plot_yield_comparison.py

Assumptions
-----------
  • Each config lives in its own sub-folder:
        default/integrated_yields_trig8-15.csv
        nofsr/integrated_yields_trig8-15.csv
        noisr/integrated_yields_trig8-15.csv
        nofsrnoisr/integrated_yields_trig8-15.csv
  • CSV format (after comment lines starting with #):
        assoc_pt_lo, assoc_pt_hi, assoc_pt_center,
        Y_near, err_near, Y_away, err_away
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── file locations ────────────────────────────────────────────────────────────
FILENAME   = "integrated_yields_trig8-15.csv"
CONFIGS    = ["default", "nofsr", "noisr", "nofsrnoisr"]

# ── aesthetics ────────────────────────────────────────────────────────────────
CONFIG_STYLE = {
    'default':      dict(label="FSR & ISR",       color='#4453FF', marker='^', ls='-', lw=1.4),
    'nofsr':        dict(label="No FSR & ISR",    color='#D55E00', marker='o', ls='-', lw=1.4),
    'noisr':        dict(label="FSR & No ISR",    color='#009E73', marker='s', ls='-', lw=1.4),
    'nofsrnoisr':   dict(label="No FSR & No ISR", color='#CC79A7', marker='D', ls='-', lw=1.4),
}

RATIO_COLORS = {
    "nofsr":      '#D55E00',
    "noisr":      '#009E73',
    "nofsrnoisr": '#CC79A7',
}

TRIG_PT = (8.0, 15.0)
SAVE_PATH = "yield_comparison_trig8-15.png"


# ── helpers ───────────────────────────────────────────────────────────────────
def load_csv(folder: str) -> pd.DataFrame:
    path = os.path.join(folder, FILENAME)
    df = pd.read_csv(path, comment="#")
    df.columns = df.columns.str.strip()
    return df



# ── load all data ─────────────────────────────────────────────────────────────
data = {}
for cfg in CONFIGS:
    try:
        data[cfg] = load_csv(f'plots/yields/test/{cfg}')
    except FileNotFoundError:
        print(f"  WARNING: {os.path.join(cfg, FILENAME)} not found — skipping.")

if "default" not in data:
    raise FileNotFoundError("Default config CSV is required but was not found.")

variants = [c for c in CONFIGS[1:] if c in data]   # non-default configs found


# ── build figure ──────────────────────────────────────────────────────────────
# Layout: 3 rows × 2 cols
#   row 0: yield panels (log scale)
#   row 1: ratio default / variant A  (per variant, stacked)
#   ...
n_ratio_rows = len(variants)   # up to 3
n_rows = 1 + n_ratio_rows

height_ratios = [2.5] + [1.0] * n_ratio_rows

fig, axes = plt.subplots(
    n_rows, 2,
    figsize=(10, 3.5 + 2.5 * n_ratio_rows),
    gridspec_kw={"height_ratios": height_ratios, "hspace": 0.40, "wspace": 0.30},
)

yield_axes = [axes[0, 0], axes[0, 1]]
# ratio_axes[v][side] — v indexes variants, side 0=near 1=away
ratio_axes = [[axes[1 + v, s] for s in range(2)] for v in range(n_ratio_rows)]

sides = [
    dict(y_col="Y_near", e_col="err_near", label="Near side",
         phi_label=r"$-\pi/2 < \Delta\phi \leq \pi/2$",
         ylabel=r"$Y^{\mathrm{near}}$", away=False),
    dict(y_col="Y_away", e_col="err_away", label="Away side",
         phi_label=r"$\pi/2 < \Delta\phi \leq 3\pi/2$",
         ylabel=r"$Y^{\mathrm{away}}$", away=True),
]

# ── yield row ─────────────────────────────────────────────────────────────────
for col, side in enumerate(sides):
    ax = yield_axes[col]
    y_col, e_col = side["y_col"], side["e_col"]

    for cfg in [c for c in CONFIGS if c in data]:
        df  = data[cfg]
        st  = CONFIG_STYLE[cfg]
        pt  = df["assoc_pt_center"].values
        y   = df[y_col].values
        err = df[e_col].values
        xerr_lo = pt - df["assoc_pt_lo"].values
        xerr_hi = df["assoc_pt_hi"].values - pt

        ax.errorbar(pt, y, xerr=[xerr_lo, xerr_hi], yerr=err,
                    fmt=st["marker"], color=st["color"], ms=6,
                    markeredgewidth=0.6, ecolor=st["color"],
                    elinewidth=1.4, capsize=3.5,
                    label=st["label"])


    ax.set_yscale("log")
    ax.set_ylabel(side["ylabel"], fontsize=13)
    ax.set_title(
        rf'{side["label"]}' + "\n" + rf'{side["phi_label"]}',
        fontsize=12, fontweight="bold", pad=7,
    )
    ax.tick_params(labelbottom=False, labelsize=11)
    ax.legend(fontsize=8, framealpha=0.2, loc="upper right")
    ax.grid(True, alpha=0.15)
    ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())

# ── ratio rows ────────────────────────────────────────────────────────────────
for v, variant in enumerate(variants):
    df_def = data["default"]
    df_var = data[variant]

    is_last_row = (v == n_ratio_rows - 1)

    for col, side in enumerate(sides):
        ax = ratio_axes[v][col]
        y_col, e_col = side["y_col"], side["e_col"]

        pt_def = df_def["assoc_pt_center"].values
        y_def  = df_def[y_col].values
        e_def  = df_def[e_col].values

        pt_var = df_var["assoc_pt_center"].values
        y_var  = df_var[y_col].values
        e_var  = df_var[e_col].values

        # Interpolate variant onto default pt grid
        y_var_interp = np.interp(pt_def, pt_var, y_var)
        e_var_interp = np.interp(pt_def, pt_var, e_var)

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(y_var_interp != 0, y_def / y_var_interp, np.nan)
            # error propagation: σ_ratio = ratio * sqrt((σ_num/num)² + (σ_den/den)²)
            ratio_err = np.where(
                y_var_interp != 0,
                ratio * np.sqrt((e_def / y_def) ** 2 + (e_var_interp / y_var_interp) ** 2),
                np.nan,
            )

        xerr_lo = pt_def - df_def["assoc_pt_lo"].values
        xerr_hi = df_def["assoc_pt_hi"].values - pt_def
        color = RATIO_COLORS[variant]

        ax.errorbar(pt_def, ratio, xerr=[xerr_lo, xerr_hi], yerr=ratio_err,
                    fmt="o", color=color, ms=5.5, markeredgewidth=0.6,
                    ecolor=color, elinewidth=1.4, capsize=3.5, zorder=4)
        ax.axhline(1, lw=1.4, ls="--", color="black", alpha=0.4)

        # ratio y-label: "Default / No FSR" etc.
        ratio_label = f"FSR & ISR / {CONFIG_STYLE[variant]['label']}"
        ax.set_ylabel(ratio_label, fontsize=9, labelpad=4)
        ax.set_ylim(0.5, 2.0)
        ax.grid(True, alpha=0.15)
        ax.tick_params(labelsize=10, labelbottom=is_last_row)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))

        if is_last_row:
            ax.set_xlabel(r"$p_T^{\rm assoc}$ [GeV/c]", fontsize=12)

# ── title ─────────────────────────────────────────────────────────────────────
fig.suptitle(
    rf"Per-Trigger Integrated Yields"
    "\n"
    rf"Trig $p_T$: {TRIG_PT[0]:.1f}–{TRIG_PT[1]:.1f} GeV/c  |  $|\eta|<0.8$",
    fontsize=13, fontweight="bold", y=0.97,
)

plt.tight_layout()
plt.savefig(f'plots/yields/test/{SAVE_PATH}', dpi=200, bbox_inches="tight")
print(f"Saved → {SAVE_PATH}")
plt.show()
