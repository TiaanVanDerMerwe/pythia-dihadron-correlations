"""
compare_htff.py — Compare HTFF D(zT) curves from multiple Pythia CSV outputs
=============================================================================

Reads two or more CSV files produced by the HTFF analysis script
(plot_integrated_yields → *_pythia.csv) and overlays them on a 2×2 grid:

    Top row   : D_near(zT)  and  D_away(zT)  — log scale, points + exp fit
    Bottom row: ratio to the first file listed (reference)

Expected CSV columns (as written by the analysis script):
    zT_center, zT_lo, zT_hi, D_near, D_near_err, D_away, D_away_err

Usage
-----
Edit the DATASETS list below and run:

    python compare_htff.py

Each entry is a dict with:
    'path'  : path to the *_pythia.csv file
    'label' : legend label (e.g. 'Pythia 8.3 tune A')
    'color' : hex colour string
    'marker': matplotlib marker character  ('o', 's', 'D', '^', 'v', …)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# ─────────────────────────────────────────────────────────────────────────────
# USER CONFIGURATION — edit these entries
# ─────────────────────────────────────────────────────────────────────────────
DATASETS = [
    {
        'path':   'plots/htff/htff_8.0-15.0_fsr_pythia.csv',
        'label':  r'FSR',
        'color':  '#4453FF',
        'marker': 'o',
    },
    {
        'path':   'plots/htff/htff_8.0-15.0_nofsr_pythia.csv',
        'label':  r'No FSR',
        'color':  '#E8700A',
        'marker': 's',
    },
    # Add more entries here as needed:
    # {
    #     'path':   'path/to/another_pythia.csv',
    #     'label':  'My label',
    #     'color':  '#229922',
    #     'marker': 'D',
    # },
]

# Reference dataset index (denominator in the ratio panels)
REFERENCE_IDX = 0

# Output file (set to None to only show interactively)
SAVE_PATH = 'plots/htff/htff_comparison.png'

# Optional: trigger pT range for the suptitle (purely cosmetic)
TRIG_PT_RANGE = (8.0, 15.0)

# ─────────────────────────────────────────────────────────────────────────────
# Exponential fit (weighted log-linear least squares — same as main script)
# ─────────────────────────────────────────────────────────────────────────────

def _exp_fit(x, y, yerr):
    mask = np.isfinite(y) & (y > 0) & np.isfinite(yerr) & (yerr > 0)
    if mask.sum() < 2:
        return None, None, None, None

    lx  = x[mask]
    ly  = np.log(y[mask])
    w   = (y[mask] / yerr[mask]) ** 2

    W    = w.sum()
    Wx   = (w * lx).sum()
    Wx2  = (w * lx ** 2).sum()
    Wy   = (w * ly).sum()
    Wxy  = (w * lx * ly).sum()

    denom = W * Wx2 - Wx ** 2
    if denom == 0:
        return None, None, None, None

    b      = (W * Wxy - Wx * Wy) / denom
    log_A  = (Wy - b * Wx) / W
    A      = np.exp(log_A)

    sigma_b    = np.sqrt(W   / denom)
    sigma_logA = np.sqrt(Wx2 / denom)
    sigma_A    = A * sigma_logA

    return A, b, sigma_A, sigma_b


# ─────────────────────────────────────────────────────────────────────────────
# Load all CSV files
# ─────────────────────────────────────────────────────────────────────────────

def load_datasets(datasets):
    loaded = []
    for ds in datasets:
        path = ds['path']
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"CSV not found: {path}\n"
                "  Check that DATASETS entries point to valid *_pythia.csv files."
            )
        df = pd.read_csv(path)
        required = {'zT_center', 'zT_lo', 'zT_hi',
                    'D_near', 'D_near_err', 'D_away', 'D_away_err'}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV {path} is missing columns: {missing}")
        loaded.append({**ds, 'df': df})
    return loaded


# ─────────────────────────────────────────────────────────────────────────────
# Build plot
# ─────────────────────────────────────────────────────────────────────────────

def build_comparison_figure(loaded, trig_pt_range, ref_idx, save_path):

    ref_df = loaded[ref_idx]['df']

    fig, axes = plt.subplots(
        2, 2,
        figsize=(12, 6),
        gridspec_kw={'height_ratios': [2.5, 1.0], 'hspace': 0.08, 'wspace': 0.30},
    )

    panels = [
        # (top ax, bot ax, y_col, yerr_col, side label, ylabel)
        (axes[0, 0], axes[1, 0],
         'D_near', 'D_near_err',
         'Near side',
         r'$D^{\rm near}(z_T)$'),

        (axes[0, 1], axes[1, 1],
         'D_away', 'D_away_err',
         'Away side',
         r'$D^{\rm away}(z_T)$'),
    ]

    for ax_top, ax_rat, ycol, yerrcol, side_label, ylabel in panels:

        # ── Data points & fits ────────────────────────────────────────────
        for ds in loaded:
            df     = ds['df']
            zt     = df['zT_center'].values
            y      = df[ycol].values
            yerr   = df[yerrcol].values
            xerr   = [zt - df['zT_lo'].values, df['zT_hi'].values - zt]
            color  = ds['color']
            marker = ds['marker']
            label  = ds['label']

            ax_top.errorbar(
                zt, y, xerr=xerr, yerr=yerr,
                fmt=marker, color=color, ms=7,
                markeredgewidth=0.6, ecolor=color,
                elinewidth=1.5, capsize=4,
                label=label, zorder=5,
            )

            A, b, sA, sb = _exp_fit(zt, y, yerr)
            if A is not None:
                x_fit  = np.linspace(zt.min() * 0.95, zt.max() * 1.05, 300)
                y_fit  = A * np.exp(b * x_fit)
                rel_bnd = np.sqrt((sA / A) ** 2 + (x_fit * sb) ** 2)
                ax_top.plot(x_fit, y_fit, '--', color=color, lw=1.6, alpha=0.85,
                            label=rf'{label} fit  $b={b:.2f}\pm{sb:.2f}$',
                            zorder=4)
                ax_top.fill_between(x_fit,
                                    y_fit * (1 - rel_bnd),
                                    y_fit * (1 + rel_bnd),
                                    color=color, alpha=0.15, zorder=3)

        ax_top.set_ylabel(ylabel, fontsize=13)
        ax_top.set_title(side_label, fontsize=13, fontweight='bold', pad=8)
        ax_top.legend(fontsize=8, framealpha=0.2, loc='upper right')
        ax_top.grid(True, alpha=0.15)
        ax_top.set_yscale('log')
        ax_top.tick_params(labelbottom=False, labelsize=13)

        # ── Ratio panels ──────────────────────────────────────────────────
        ref_zt   = ref_df['zT_center'].values
        ref_y    = ref_df[ycol].values
        ref_yerr = ref_df[yerrcol].values

        for i, ds in enumerate(loaded):
            if i == ref_idx:
                continue

            df     = ds['df']
            zt     = df['zT_center'].values
            y      = df[ycol].values
            yerr   = df[yerrcol].values
            color  = ds['color']
            marker = ds['marker']

            ZT_TOL = 0.06   # tolerance for matching zT bin centres
            rat_zt, rat_xerr, rat_y, rat_err = [], [], [], []

            for zt_i, y_i, e_i in zip(zt, y, yerr):
                dists = np.abs(ref_zt - zt_i)
                j_min = int(np.argmin(dists))
                if dists[j_min] > ZT_TOL:
                    continue
                ry   = ref_y[j_min]
                re   = ref_yerr[j_min]
                if ry == 0:
                    continue
                r    = y_i / ry
                with np.errstate(invalid='ignore'):
                    r_e = r * np.sqrt(
                        (e_i / y_i) ** 2 + (re / ry) ** 2
                    ) if y_i != 0 else np.nan
                rat_zt.append(zt_i)
                rat_xerr.append(0.05)
                rat_y.append(r)
                rat_err.append(r_e)

            rat_ylabel = ds['label'] + ' / ' + loaded[ref_idx]['label']
            if rat_zt:
                ax_rat.errorbar(
                    rat_zt, rat_y,
                    xerr=rat_xerr, yerr=rat_err,
                    fmt=marker, color=color, ms=6,
                    markeredgewidth=0.6, ecolor=color,
                    elinewidth=1.5, capsize=4,
                    label=rat_ylabel, zorder=4,
                )
            ax_rat.set_ylabel(rat_ylabel, fontsize=10)

        ax_rat.axhline(1, lw=1.4, ls='--', color='black', alpha=0.4)
        ax_rat.set_xlabel(r'$z_T$', fontsize=13)
        ax_rat.set_ylim(0.0, 2.0)
        ax_rat.set_xlim(ax_top.get_xlim())
        ax_rat.grid(True, alpha=0.15)
        ax_rat.tick_params(labelsize=12)
        if len(loaded) > 2:
            ax_rat.legend(fontsize=8, framealpha=0.2, loc='upper right')

    fig.suptitle(
        rf'Hadron-Triggered Fragmentation Function  —  Comparison'
        '\n'
        rf'Trig $p_T$: {trig_pt_range[0]:.1f}–{trig_pt_range[1]:.1f} GeV/c',
        fontsize=13, fontweight='bold', y=1.03,
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig, axes


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if len(DATASETS) < 2:
        raise ValueError("Add at least two entries to DATASETS before running.")

    print(f"Loading {len(DATASETS)} dataset(s) …")
    loaded = load_datasets(DATASETS)

    for ds in loaded:
        df = ds['df']
        print(f"  [{ds['label']}]  {len(df)} zT points  "
              f"(zT {df['zT_center'].min():.2f}–{df['zT_center'].max():.2f})"
              f"  from {ds['path']}")

    print("\nBuilding comparison figure …")
    build_comparison_figure(loaded, TRIG_PT_RANGE, REFERENCE_IDX, SAVE_PATH)
    plt.show()
    print("Done.")


if __name__ == '__main__':
    main()