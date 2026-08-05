"""
Dihadron Correlation Analysis — pT_assoc Slice Yields
======================================================

Per-pT_assoc slice correlation functions with ZYAM subtraction, compared to
data, with per-trigger integrated near- and away-side yields as a function
of pT_assoc.

Uncertainty method
------------------
Block jackknife, delegated to compute_phi_projection_jackknife from the base
analysis script, which returns both the per-bin central errors and the full
(n_blocks x n_phi) array of per-block projections.

Integrated yield uncertainties are computed by integrating each block's
projection first, then applying the jackknife variance formula to the
resulting scalar yields:

    Y_k    = sum_{phi in side} max(proj_k(phi), 0) * Dphi / DpT
    Var_JK = (N-1)/N * sum_k (Y_k - mean(Y_k))^2
    sigma  = sqrt(Var_JK)

This correctly accounts for the fact that removing one jackknife block shifts
the entire Dphi projection coherently — the phi bins are NOT independent, so
propagating per-bin errors in quadrature would be wrong.

Distribution is UNFOLDED: Delta-phi runs from -pi/2 to 3pi/2.
    Near side : -pi/2 < Delta-phi <= pi/2
    Away side :  pi/2 < Delta-phi <= 3pi/2
    ZYAM window: (0, pi)  (straddles both peaks symmetrically)

pT_assoc slices:  [1,2], [2,3], [3,4], [4,5], [5,6], [6,7], [7,8]  GeV/c
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import sys

# ─────────────────────────────────────────────────────────────────────────────
# Re-use helpers from the base analysis script
# ─────────────────────────────────────────────────────────────────────────────
from projectionsNoFoldover import (
    read_and_combine_bins,
    compute_phi_projection_jackknife,
)

# ─────────────────────────────────────────────────────────────────────────────
# Analysis configuration
# ─────────────────────────────────────────────────────────────────────────────
TRIG_PT_MIN = 8
TRIG_PT_MAX = 15  # GeV/c

ASSOC_PT_EDGES = [4,6]  # GeV/c

# Unfolded phi layout
# Near side : -pi/2 < phi <= pi/2
# Away side :  pi/2 < phi <= 3pi/2
NEAR_SIDE_LO = -np.pi / 2   # left edge of near side
NEAR_SIDE_HI =  np.pi / 2   # boundary between near and away
AWAY_SIDE_HI =  3 * np.pi / 2

ETA_BINS  = 36
PHI_BINS  = 36
ETA_RANGE = (-1.6, 1.6)
PHI_RANGE = (-np.pi / 2, 3 * np.pi / 2)   # unfolded

# ZYAM search window — sits in the trough between the two peaks
ZYAM_PHI_LO =  -np.pi/2
ZYAM_PHI_HI =  3*np.pi/2

zyam = True

N_JACKKNIFE_BLOCKS = 50

# ─────────────────────────────────────────────────────────────────────────────
# Toggle: set to False to skip the Pythia/Data ratio panels in the yield plot
# ─────────────────────────────────────────────────────────────────────────────
PLOT_RATIO = True


# ─────────────────────────────────────────────────────────────────────────────
# Per-slice computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_slice(data, assoc_pt_min, assoc_pt_max, ntrig_override):
    """
    Run the full jackknife pipeline for one pT_assoc slice.

    Filters the combined pair DataFrame to the requested assoc pT window, then
    delegates to compute_phi_projection_jackknife which handles the 2D
    correlation, eta projection, ZYAM subtraction, and jackknife resampling
    internally.  No folding is applied — the distribution runs from -pi/2 to
    3pi/2.

    Returns
    -------
    dict with keys:
        phi_centers      : 1-D array in [-pi/2, 3pi/2]
        phi_proj         : ZYAM-subtracted, JK-central Dphi projection
        phi_proj_err     : per-bin JK 1-sigma errors  (informational only —
                           NOT used for yield uncertainties, see integrate_yield)
        jackknife_projs  : 2-D array (n_blocks x n_phi) of per-block projections
        background       : ZYAM level
        ntrig            : effective trigger count used
        assoc_pt_range   : (min, max)
    or None if the slice is empty.
    """
    assoc_mask = ((data['assoc_pT'] > assoc_pt_min) &
                  (data['assoc_pT'] <= assoc_pt_max))
    sliced = data[assoc_mask].copy()

    if len(sliced) == 0:
        print(f"  WARNING: No pairs in slice pT_assoc [{assoc_pt_min}, {assoc_pt_max}]")
        return None

    print(f"\n── pT_assoc [{assoc_pt_min}, {assoc_pt_max}] GeV/c  "
          f"({len(sliced):,} pairs) ──────────────────────────────────")

    metadata_slice = {
        'CUT_TRIG_PT_RANGE':  [TRIG_PT_MIN, TRIG_PT_MAX],
        'CUT_ASSOC_PT_RANGE': [assoc_pt_min, assoc_pt_max],
    }

    (phi_proj, phi_proj_err, phi_centers,
     ntrig, bkg, jackknife_projs) = compute_phi_projection_jackknife(
        sliced, metadata_slice,
        eta_bins=ETA_BINS,
        phi_bins=PHI_BINS,
        eta_range=ETA_RANGE,
        phi_range=PHI_RANGE,
        zyam=zyam,
        zyam_range={'phi': (ZYAM_PHI_LO, ZYAM_PHI_HI)},
        ntrig_override=ntrig_override,
        n_blocks=N_JACKKNIFE_BLOCKS,
    )
    
    return {
        'phi_centers':     phi_centers,      # in [-pi/2, 3pi/2]
        'phi_proj':        phi_proj,
        'phi_proj_err':    phi_proj_err,
        'jackknife_projs': jackknife_projs,  # shape: (n_blocks, n_phi)
        'background':      bkg,
        'ntrig':           ntrig,
        'assoc_pt_range':  (assoc_pt_min, assoc_pt_max),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Yield integration  (unfolded phi space)
# ─────────────────────────────────────────────────────────────────────────────

def _integrate_one(proj, phi_centers, near_mask, away_mask, bw, delta_pt):
    """Integrate a single (central or per-block) projection into near/away yields."""
    signal  = np.where(proj > 0, proj, 0.0)
    Y_near  = float((signal[near_mask] * bw).sum() / delta_pt)
    Y_away  = float((signal[away_mask] * bw).sum() / delta_pt)
    return Y_near, Y_away


def integrate_yield(phi_centers, phi_proj, jackknife_projs, assoc_pt_range):
    """
    Integrate the unfolded, ZYAM-subtracted Dphi projection into near- and
    away-side per-trigger yields with correct jackknife uncertainties.

    Unfolded phi layout
    -------------------
    Near side : phi_centers in (-pi/2, pi/2]
    Away side : phi_centers in ( pi/2, 3pi/2)

    Each jackknife block's projection is integrated independently, and the
    jackknife variance formula is applied to the resulting scalar yields:

        Y_k    = sum_{phi in side} max(proj_k(phi), 0) * Dphi / DpT
        Var_JK = (N-1)/N * sum_k (Y_k - mean(Y_k))^2

    Parameters
    ----------
    phi_centers     : 1-D array  [-pi/2, 3pi/2]
    phi_proj        : 1-D array  ZYAM-subtracted central projection
    jackknife_projs : 2-D array  (n_blocks x n_phi) per-block projections
    assoc_pt_range  : (lo, hi)

    Returns
    -------
    Y_near, Y_away, sigma_near, sigma_away : float x 4
    """
    bw       = phi_centers[1] - phi_centers[0]
    delta_pt = assoc_pt_range[1] - assoc_pt_range[0]

    # Near:  -pi/2 < phi <= pi/2
    # Away:   pi/2 < phi <= 3pi/2
    near_mask = phi_centers <= NEAR_SIDE_HI
    away_mask = phi_centers >  NEAR_SIDE_HI

    Y_near, Y_away = _integrate_one(phi_proj, phi_centers,
                                    near_mask, away_mask, bw, delta_pt)

    n_blocks = len(jackknife_projs)
    jk_near  = np.array([_integrate_one(jackknife_projs[k], phi_centers,
                                        near_mask, away_mask, bw, delta_pt)[0]
                         for k in range(n_blocks)])
    jk_away  = np.array([_integrate_one(jackknife_projs[k], phi_centers,
                                        near_mask, away_mask, bw, delta_pt)[1]
                         for k in range(n_blocks)])

    # Jackknife variance: Var_JK = (N-1)/N * sum_k (theta_k - theta_bar)^2
    factor     = (n_blocks - 1) / n_blocks
    sigma_near = float(np.sqrt(factor * np.sum((jk_near - jk_near.mean()) ** 2)))
    sigma_away = float(np.sqrt(factor * np.sum((jk_away - jk_away.mean()) ** 2)))

    return Y_near, Y_away, sigma_near, sigma_away


# ─────────────────────────────────────────────────────────────────────────────
# CMS data loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_csv(path):
    """Read a two-column CSV, sort by first column. Returns (x, y) or (None, None)."""
    if not os.path.exists(path):
        return None, None
    try:
        df  = pd.read_csv(path)
        x   = df.iloc[:, 0].values
        y   = df.iloc[:, 1].values
        idx = np.argsort(x)
        return x[idx], y[idx]
    except Exception as e:
        print(f"  Warning: could not read {path}: {e}")
        return None, None


def load_cms_dphi(trig_pt_range, assoc_pt_range, data_dir='datathief'):
    tlo, thi = trig_pt_range
    alo, ahi = assoc_pt_range
    base = f"{data_dir}/CMS_{tlo:.0f}-{thi:.0f}_{alo:.0f}-{ahi:.0f}"
    for ext in ('.csv', '.txt'):
        x, y = _load_csv(base + ext)
        if x is not None:
            return x, y
    return None, None


def load_cms_yield(trig_pt_range, away=True, data_dir='datathief'):
    tlo, thi = trig_pt_range
    side = 'away' if away else 'near'
    base = f"{data_dir}/CMS_{side}_{tlo:.0f}-{thi:.0f}"
    for ext in ('.csv', '.txt'):
        x, y = _load_csv(base + ext)
        if x is not None:
            return x, y
    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# CSV output helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_dphi_slices_csv(slice_results, out_dir, trig_pt_range):
    """
    Write one CSV per pT_assoc slice containing the ZYAM-subtracted Dphi
    projection with per-bin jackknife errors.

    Columns
    -------
    delta_phi_rad      : bin centre in (-pi/2, 3pi/2)
    dNpair_dDeltaPhi   : 1/Ntrig × dNpair/dDeltaPhi  (ZYAM-subtracted)
    stat_err_jackknife : jackknife standard error
    zyam_background    : constant subtracted from every bin
    """
    tlo, thi = trig_pt_range
    for res in (r for r in slice_results if r is not None):
        alo, ahi = res['assoc_pt_range']
        fname = os.path.join(
            out_dir,
            f'dphi_slice_trig{tlo:.0f}-{thi:.0f}_assoc{alo:.0f}-{ahi:.0f}.csv',
        )
        df = pd.DataFrame({
            'delta_phi_rad':      res['phi_centers'],
            'dNpair_dDeltaPhi':   res['phi_proj'],
            'stat_err_jackknife': res['phi_proj_err'],
            'zyam_background':    res['background'],
        })
        header = [
            f'# Dihadron Dphi projection — pT_assoc slice [{alo}, {ahi}] GeV/c',
            f'# trigger pT : {tlo:.1f} – {thi:.1f} GeV/c',
            f'# assoc   pT : {alo:.1f} – {ahi:.1f} GeV/c',
            f'# ZYAM level : {res["background"]:.6e}',
            f'# ntrig      : {res["ntrig"]:.6e}',
            f'# jackknife blocks : {N_JACKKNIFE_BLOCKS}',
            f'#',
        ]
        with open(fname, 'w') as f:
            f.write('\n'.join(header) + '\n')
            df.to_csv(f, index=False)
        print(f"  Saved Dphi slice CSV → {fname}")


def save_integrated_yields_csv(slice_results, out_dir, trig_pt_range):
    """
    Write a single CSV with the integrated near- and away-side yields for
    every pT_assoc slice.

    Columns
    -------
    assoc_pt_lo, assoc_pt_hi : slice edges  [GeV/c]
    assoc_pt_center          : slice midpoint
    Y_near, err_near         : near-side yield ± jackknife 1-sigma
    Y_away, err_away         : away-side yield ± jackknife 1-sigma
    """
    tlo, thi = trig_pt_range
    rows = []
    for res in (r for r in slice_results if r is not None):
        alo, ahi = res['assoc_pt_range']
        Yn, Ya, sn, sa = integrate_yield(
            res['phi_centers'], res['phi_proj'], res['jackknife_projs'],
            assoc_pt_range=res['assoc_pt_range'],
        )
        rows.append({
            'assoc_pt_lo':     alo,
            'assoc_pt_hi':     ahi,
            'assoc_pt_center': (alo + ahi) / 2,
            'Y_near':          Yn,
            'err_near':        sn,
            'Y_away':          Ya,
            'err_away':        sa,
        })

    df = pd.DataFrame(rows)
    fname = os.path.join(
        out_dir,
        f'integrated_yields_trig{tlo:.0f}-{thi:.0f}.csv',
    )
    header = [
        f'# Per-trigger integrated near/away yields vs pT_assoc',
        f'# trigger pT : {tlo:.1f} – {thi:.1f} GeV/c',
        f'# jackknife blocks : {N_JACKKNIFE_BLOCKS}',
        f'# Near side : -pi/2 < Delta-phi <= pi/2',
        f'# Away side :  pi/2 < Delta-phi <= 3pi/2',
        f'#',
    ]
    with open(fname, 'w') as f:
        f.write('\n'.join(header) + '\n')
        df.to_csv(f, index=False)
    print(f"  Saved integrated yields CSV → {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

_COLORS = ['#4453FF']

# Unfolded x-axis ticks: -pi/2 … 3pi/2
_PHI_TICKS  = [-np.pi/2, -np.pi/4, 0,
                np.pi/4,  np.pi/2,
                3*np.pi/4, np.pi,
                5*np.pi/4, 3*np.pi/2]
_PHI_LABELS = [r'$-\pi/2$', r'$-\pi/4$', r'$0$',
               r'$\pi/4$',  r'$\pi/2$',
               r'$3\pi/4$', r'$\pi$',
               r'$5\pi/4$', r'$3\pi/2$']


def plot_dphi_slices(slice_results, trig_pt_range=(TRIG_PT_MIN, TRIG_PT_MAX),
                     cms_dir='datathief', save_path=None):
    """
    One panel per pT_assoc slice: ZYAM-subtracted unfolded Dphi with JK error bars.

    The near/away boundary (phi = pi/2) and the ZYAM window edges (phi = 0
    and phi = pi) are drawn as reference lines.
    """
    valid = [r for r in slice_results if r is not None]
    ncols = 3
    nrows = int(np.ceil(len(valid) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6 * ncols, 4.5 * nrows), squeeze=False)

    for i, res in enumerate(valid):
        ax    = axes[i // ncols][i % ncols]
        color = _COLORS[i % len(_COLORS)]
        alo, ahi = res['assoc_pt_range']

        phi = res['phi_centers']
        sig = res['phi_proj']
        err = res['phi_proj_err']

        # CMS data (assumed to be on the same unfolded range if present)
        cms_phi, cms_y = load_cms_dphi(trig_pt_range, (alo, ahi), data_dir=cms_dir)
        if cms_phi is not None:
            ax.step(cms_phi, cms_y, where='mid', color='black',
                    linestyle='-', linewidth=2, label='CMS data', zorder=10)

        ax.step(phi, sig, where='mid', color=color,
                linestyle='-', linewidth=2,
                label=f'Pythia  ZYAM={res["background"]:.4f}', zorder=4)
        ax.errorbar(phi, sig, yerr=err,
                    fmt='none', ls='none', color=color, ms=5, lw=1.4, zorder=4)

        # Near/away boundary
        ax.axvline(NEAR_SIDE_HI, color='green', lw=1.4, ls='--',
                   alpha=0.7, label=r'$|\Delta\phi|=\pi/2$')
        # ZYAM window edges
        ax.axvline(ZYAM_PHI_LO, color='grey', lw=1.0, ls=':', alpha=0.6)
        ax.axvline(ZYAM_PHI_HI, color='grey', lw=1.0, ls=':', alpha=0.6)
        ax.axhline(0, lw=1.4, ls='--', color='black', alpha=0.4)

        ax.set_xlim(NEAR_SIDE_LO - 0.05, AWAY_SIDE_HI + 0.05)
        ax.set_xticks(_PHI_TICKS)
        ax.set_xticklabels(_PHI_LABELS, fontsize=8)
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=13)
        ax.set_xlabel(r'$\Delta\phi$ [rad]', fontsize=13)
        ax.set_ylabel(r'$\frac{1}{N_{\rm trig}}\frac{dN_{\rm pair}}{d\Delta\phi}$',
                      fontsize=17)
        ax.set_title(
            rf'Assoc $p_T$: {alo:.1f}–{ahi:.1f} GeV/c',
            fontsize=13, fontweight='bold', pad=6,
        )
        ax.legend(fontsize=7, framealpha=0.2)
        ax.grid(True, alpha=0.15)

    for j in range(len(valid), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle(
        rf'FSR & No ISR Dihadron $\Delta\phi$ Distributions' + '\n'
        rf'Trig $p_T$: {trig_pt_range[0]:.1f}–{trig_pt_range[1]:.1f} GeV/c' + '\n'
        rf'$|\eta|$ < 0.8' + '\n',
        fontsize=13, fontweight='bold', y=1.0,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    return fig, axes


def _power_law_fit(x, y, pt_min=5.0):
    mask = (x >= pt_min) & np.isfinite(y) & (y > 0)
    if mask.sum() < 2:
        return None, None
    n, log_A = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)
    return np.exp(log_A), n


def plot_integrated_yields(slice_results, trig_pt_range=(TRIG_PT_MIN, TRIG_PT_MAX),
                           cms_dir='datathief', save_path=None,
                           plot_ratio=PLOT_RATIO):
    """
    Yield panels with optional Pythia/CMS ratio rows underneath.

    When plot_ratio=True  → 2 rows (yield + ratio) per side, 2x2 grid.
    When plot_ratio=False → 1 row (yield only),              1x2 grid.

    Near side  : -pi/2 < Delta-phi <= pi/2
    Away side  :  pi/2 < Delta-phi <= 3pi/2
    """
    valid = [r for r in slice_results if r is not None]
    if not valid:
        print("No valid slices — nothing to plot.")
        return None, None

    pt_centers, xerr_lo, xerr_hi = [], [], []
    y_near, y_away, err_near, err_away = [], [], [], []

    for res in valid:
        alo, ahi = res['assoc_pt_range']
        pt_c = (alo + ahi) / 2
        pt_centers.append(pt_c)
        xerr_lo.append(pt_c - alo)
        xerr_hi.append(ahi - pt_c)

        Yn, Ya, sn, sa = integrate_yield(
            res['phi_centers'], res['phi_proj'], res['jackknife_projs'],
            assoc_pt_range=res['assoc_pt_range'],
        )
        y_near.append(Yn);   y_away.append(Ya)
        err_near.append(sn); err_away.append(sa)

    pt_centers = np.array(pt_centers)

    y_near,   y_away   = np.array(y_near),   np.array(y_away)
    err_near, err_away = np.array(err_near), np.array(err_away)
    xerr  = [xerr_lo, xerr_hi]
    x_fit = np.linspace(pt_centers.min() * 0.95, pt_centers.max() * 1.05, 300)

    color = '#4453FF'

    panels_def = [
        (y_near, err_near, False, 'Near side',
         r'$-\pi/2 < \Delta\phi \leq \pi/2$',
         r'$Y^{\mathrm{near}}$'),
        (y_away, err_away, True,  'Away side',
         r'$\pi/2 < \Delta\phi \leq 3\pi/2$',
         r'$Y^{\mathrm{away}}$'),
    ]

    if plot_ratio:
        fig, axes = plt.subplots(
            2, 2, figsize=(10, 5),
            gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.08, 'wspace': 0.3},
        )
        yield_axes = [axes[0, 0], axes[0, 1]]
        ratio_axes = [axes[1, 0], axes[1, 1]]
    else:
        fig, axes_1d = plt.subplots(1, 2, figsize=(10, 4),
                                    gridspec_kw={'wspace': 0.3})
        yield_axes = list(axes_1d)
        ratio_axes = [None, None]
        axes = axes_1d  # keep a reference for the return value

    for col, (yields, errs, away, label, phi_label, ylabel) in enumerate(panels_def):
        ax_top = yield_axes[col]
        ax_rat = ratio_axes[col]

        cms_pt, cms_y = load_cms_yield(trig_pt_range, away=away, data_dir=cms_dir)

        if cms_pt is not None:
            cms_pt, cms_y = np.array(cms_pt), np.array(cms_y)
            ax_top.errorbar(cms_pt, cms_y, xerr=xerr,
                            fmt='*', color='black', ms=7, markeredgewidth=0.6,
                            ecolor='black', elinewidth=1.5, capsize=4,
                            label='CMS data', zorder=6)
            A_cms, n_cms = _power_law_fit(cms_pt, cms_y)
            if A_cms is not None:
                ax_top.plot(x_fit, A_cms * x_fit ** n_cms,
                            '--', color='black', lw=1.6, alpha=0.85,
                            label=rf'CMS fit  ($n={-n_cms:.2f}$)', zorder=6)
        else:
            ax_top.text(0.97, 0.95, 'no CMS file', transform=ax_top.transAxes,
                        fontsize=7, ha='right', va='top')

        ax_top.errorbar(pt_centers, yields, xerr=xerr, yerr=errs,
                        fmt='o', color=color, ms=7, markeredgewidth=0.6,
                        ecolor=color, elinewidth=1.5, capsize=4,
                        label='Pythia', zorder=5)

        A_py, n_py = _power_law_fit(pt_centers, yields)
        if A_py is not None:
            ax_top.plot(x_fit, A_py * x_fit ** n_py,
                        '--', color=color, lw=1.6, alpha=0.85,
                        label=rf'Pythia fit  ($n={-n_py:.2f}$)', zorder=5)

        ax_top.set_ylabel(ylabel, fontsize=13)
        ax_top.set_title(rf'{label}' + '\n' + rf'{phi_label}',
                         fontsize=13, fontweight='bold', pad=8)
        ax_top.legend(fontsize=8, framealpha=0.2, loc='upper right')
        ax_top.grid(True, alpha=0.15)
        ax_top.set_yscale('log')
        ax_top.tick_params(
            labelbottom=(not plot_ratio),   # hide x labels when ratio row follows
            labelsize=13,
        )
        if not plot_ratio:
            ax_top.set_xlabel(r'$p_T^{\rm assoc}$ [GeV/c]', fontsize=13)

        # ── ratio panel (only when PLOT_RATIO=True) ───────────────────────
        if plot_ratio and ax_rat is not None:
            if cms_pt is not None and len(cms_pt) == len(pt_centers):
                cms_interp = np.interp(pt_centers, cms_pt, cms_y)
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio     = np.where(cms_interp != 0, yields / cms_interp, np.nan)
                    ratio_err = np.where(cms_interp != 0, errs   / cms_interp, np.nan)
                ax_rat.errorbar(pt_centers, ratio, xerr=xerr, yerr=ratio_err,
                                fmt='o', color=color, ms=6, markeredgewidth=0.6,
                                ecolor=color, elinewidth=1.5, capsize=4, zorder=4)
            else:
                ax_rat.text(0.5, 0.5, 'CMS data not loaded',
                            ha='center', va='center',
                            transform=ax_rat.transAxes, fontsize=9, color='gray')

            ax_rat.axhline(1, lw=1.4, ls='--', color='black', alpha=0.4)
            ax_rat.set_xlabel(r'$p_T^{\rm assoc}$ [GeV/c]', fontsize=13)
            ax_rat.set_ylabel('Pythia / CMS', fontsize=13)
            ax_rat.set_ylim(0.5, 2)
            ax_rat.grid(True, alpha=0.15)
            ax_rat.tick_params(labelsize=13)

    fig.suptitle(
        rf'Per-Trigger Integrated Yields'
        '\n'
        rf'Trig $p_T$: {trig_pt_range[0]:.1f}–{trig_pt_range[1]:.1f} GeV/c'
        '\n'
        rf'$|\eta|<0.8$',
        fontsize=13, fontweight='bold', y=1.05,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    return fig, axes


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_yield_table(slice_results):
    W = 130

    w_pt  = 20
    w_y   = 14
    w_rel = 12

    print("\n" + "─" * W)
    header = (
        f"{'pT_assoc [GeV/c]':^{w_pt}}"
        f"{'Y_near':^{w_y}}"
        f"{'rel %(near)':^{w_rel}}"
        f"{'Y_away':^{w_y}}"
        f"{'rel %(away)':^{w_rel}}"
        f"{'Y_total':^{w_y}}"
        f"{'rel %(total)':^{w_rel}}"
    )
    print(header)
    print("─" * W)

    for res in (r for r in slice_results if r is not None):
        alo, ahi = res['assoc_pt_range']

        Yn, Ya, sn, sa = integrate_yield(
            res['phi_centers'], res['phi_proj'], res['jackknife_projs'],
            assoc_pt_range=res['assoc_pt_range'],
        )

        # Total yield over all phi
        bw       = res['phi_centers'][1] - res['phi_centers'][0]
        delta_pt = ahi - alo
        signal   = np.where(res['phi_proj'] > 0, res['phi_proj'], 0.0)
        Ytot     = float((signal * bw).sum() / delta_pt)

        n_blocks = len(res['jackknife_projs'])
        jk_tot   = np.array([
            float((np.where(res['jackknife_projs'][k] > 0,
                            res['jackknife_projs'][k], 0.0) * bw).sum() / delta_pt)
            for k in range(n_blocks)
        ])
        factor = (n_blocks - 1) / n_blocks
        stot   = float(np.sqrt(factor * np.sum((jk_tot - jk_tot.mean()) ** 2)))

        rel_n   = (sn   / Yn   * 100) if Yn   != 0 else 0.0
        rel_a   = (sa   / Ya   * 100) if Ya   != 0 else 0.0
        rel_tot = (stot / Ytot * 100) if Ytot != 0 else 0.0

        row = (
            f"{f'[{alo:>2.0f}, {ahi:>3.0f}]':^{w_pt}}"
            f"{Yn:>{w_y}.5f}"
            f"{rel_n:>{w_rel}.2f}"
            f"{Ya:>{w_y}.5f}"
            f"{rel_a:>{w_rel}.2f}"
            f"{Ytot:>{w_y}.5f}"
            f"{rel_tot:>{w_rel}.2f}"
        )
        print(row)

    print("─" * W + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    TYPE = "test/noisr"
    POW  = 3
    BASE = f'pythiaData/5020/{TYPE}'

    filenames = sorted(glob.glob(f'{BASE}/dihadron_pow{POW}_pT*.csv'))
    if not filenames:
        raise FileNotFoundError(
            f"No files found at {BASE}/dihadron_pow{POW}_pT*.csv"
        )

    print(f"Found {len(filenames)} pTHat bin file(s):")
    for f in filenames:
        print(f"  {f}")

    print("\nLoading and combining bins ...")
    data, metadata = read_and_combine_bins(
        filenames,
        trig_pt_min=TRIG_PT_MIN,
        trig_pt_max=TRIG_PT_MAX,
        assoc_pt_min=ASSOC_PT_EDGES[0],
        assoc_pt_max=ASSOC_PT_EDGES[-1],
        eta_range=ETA_RANGE,
    )
    ntrig_override = metadata['N_TRIG']

    print(f"\nRunning {len(ASSOC_PT_EDGES) - 1} pT_assoc slices "
          f"({N_JACKKNIFE_BLOCKS} JK blocks each) ...")

    slice_results = [
        compute_slice(data, ASSOC_PT_EDGES[j], ASSOC_PT_EDGES[j + 1], ntrig_override)
        for j in range(len(ASSOC_PT_EDGES) - 1)
    ]

    print_yield_table(slice_results)

    out_dir    = f'plots/yields/{TYPE}'
    trig_range = (TRIG_PT_MIN, TRIG_PT_MAX)
    os.makedirs(out_dir, exist_ok=True)

    # ── CSV output (written to the same directory as the images) ─────────────
    print("\nSaving CSVs ...")
    save_dphi_slices_csv(slice_results, out_dir, trig_range)
    save_integrated_yields_csv(slice_results, out_dir, trig_range)

    print("Generating Delta-phi slice plot ...")
    plot_dphi_slices(
        slice_results, trig_pt_range=trig_range, cms_dir='datathief',
        save_path=f'{out_dir}/dphi_slices_{TRIG_PT_MIN:.1f}-{TRIG_PT_MAX:.1f}.png',
    )

    print("Generating integrated yield plot ...")
    plot_integrated_yields(
        slice_results, trig_pt_range=trig_range, cms_dir='datathief',
        save_path=f'{out_dir}/integrated_yields_{TRIG_PT_MIN:.1f}-{TRIG_PT_MAX:.1f}.png',
        plot_ratio=PLOT_RATIO,
    )


if __name__ == '__main__':
    main()