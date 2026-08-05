"""
Δφ Projection Comparison Plot  +  Double-Gaussian Fits
=======================================================
For every dataset loaded (each subdir × each kinematic window, plus
experimental reference data), a double Gaussian is fitted:

    f(Δφ) = A_near · exp(−Δφ²/(2σ_near²))
          + A_away · exp(−(Δφ−π)²/(2σ_away²))
          + C          (constant pedestal / ZYAM level)

Fit results (amplitude, width, and their uncertainties) are printed to
the console in a formatted table and written to a CSV:
    OUTPUT_DIR/double_gaussian_fit_results.csv

Directory layout assumed
------------------------
plots/Correlations/star(alice)/
    DecaysRestricted_HardQCD/  dphi_projection_*.csv
    Default_HardQCD/           dphi_projection_*.csv

STAR/ALICE reference data
------------------
datathief/STAR(ALICE)_{trig_lo}-{trig_hi}_{assoc_lo}-{assoc_hi}.csv
    columns: DeltaPhi, d2Npair, stat +, stat -
"""

import os
import glob
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR    = 'plots/correlations/alice'
EXP_DIR     = 'datathief'
OUTPUT_DIR  = 'plots/comparisons/alice'
SUFFIX = 'nofrs'

SUBDIRS = [
    'default', 
    'nofsr'  
]

SHOW_RATIO = False

# Colour and marker style per configuration (order matches SUBDIRS)
STYLE = {
    'default':      dict(label="FSR & ISR",       color='#4453FF', marker='^', ls='-', lw=1.4, zorder = 5),
    'nofsr':        dict(label="No FSR & ISR",    color='#D55E00', marker='o', ls='-', lw=1.4, zorder = 4),
    'noisr':        dict(label="FSR & No ISR",    color='#009E73', marker='s', ls='-', lw=1.4, zorder = 3),
    'nofsrnoisr':   dict(label="No FSR & No ISR", color='#CC79A7', marker='D', ls='-', lw=1.4, zorder = 2),
}

EXP_STYLE = dict(color='black', marker='*', ms=8, ls='-',
                 markerfacecolor='black', markeredgewidth=0.8,
                 zorder=1, label='STAR data')

MS      = 4   # marker size for Pythia points
CAPSIZE = 2   # error bar cap size


# ──────────────────────────────────────────────────────────────────────────────
# Double-Gaussian model
# ──────────────────────────────────────────────────────────────────────────────
def double_gaussian(phi, A_near, sigma_near, A_away, sigma_away, C):
    """
    Two Gaussians (near-side at 0, away-side at π) plus a constant pedestal.

    Parameters
    ----------
    phi        : array of Δφ values [rad]
    A_near     : amplitude of near-side peak
    sigma_near : width (σ) of near-side peak  [rad]
    A_away     : amplitude of away-side peak
    sigma_away : width (σ) of away-side peak  [rad]
    C          : constant background / pedestal
    """
    return (
        A_near * np.exp(-phi**2 / (2 * sigma_near**2))
        + A_away * np.exp(-(phi - np.pi)**2 / (2 * sigma_away**2))
        + C
    )


def fit_double_gaussian(phi, y, err=None):
    """
    Fit double_gaussian to (phi, y).

    Returns
    -------
    popt : best-fit params  [A_near, sigma_near, A_away, sigma_away, C]
    perr : 1-σ uncertainties from covariance diagonal
    success : bool
    """
    # Initial guesses: peaks at local max near 0 and near π
    y_near_idx  = np.argmin(np.abs(phi))
    y_away_idx  = np.argmin(np.abs(phi - np.pi))
    A_near_0    = max(y[y_near_idx] - np.min(y), 1e-6)
    A_away_0    = max(y[y_away_idx] - np.min(y), 1e-6)
    C_0         = np.min(y)

    p0 = [A_near_0, 0.4, A_away_0, 0.6, C_0]

    # Bounds: amplitudes ≥ 0, widths ∈ (0.05, π), pedestal unconstrained
    bounds = (
        [0,    0.05, 0,    0.05, -np.inf],
        [np.inf, np.pi, np.inf, np.pi,  np.inf],
    )

    # Only use errors as weights if every bin has a finite, positive sigma.
    # A single zero or NaN contaminates the whole fit via divide-by-zero.
    use_sigma = (
        err is not None
        and len(err) == len(y)
        and np.all(np.isfinite(err))
        and np.all(err > 0)
    )
    sigma_fit = err if use_sigma else None

    try:
        popt, pcov = curve_fit(
            double_gaussian, phi, y,
            p0=p0, bounds=bounds, sigma=sigma_fit,
            absolute_sigma=use_sigma,
            maxfev=10_000,
        )
        perr = np.sqrt(np.diag(pcov))
        return popt, perr, True
    except Exception as exc:
        print(f"    [fit failed] {exc}")
        return p0, np.full(len(p0), np.nan), False


def _format_fit_row(source_label, pt_key, popt, perr, success):
    """Return a dict suitable for a DataFrame row."""
    A_near, sig_near, A_away, sig_away, C = popt
    dA_near, dsig_near, dA_away, dsig_away, dC = perr
    trig_lo, trig_hi, assoc_lo, assoc_hi = _pt_key_to_ranges(pt_key)
    return dict(
        source       = source_label,
        pt_key       = pt_key,
        trig_lo      = trig_lo,
        trig_hi      = trig_hi,
        assoc_lo     = assoc_lo,
        assoc_hi     = assoc_hi,
        fit_ok       = success,
        A_near       = A_near,
        A_near_err   = dA_near,
        sigma_near   = sig_near,
        sigma_near_err = dsig_near,
        A_away       = A_away,
        A_away_err   = dA_away,
        sigma_away   = sig_away,
        sigma_away_err = dsig_away,
        pedestal     = C,
        pedestal_err = dC,
    )


def _pt_key_to_ranges(pt_key):
    m = re.match(
        r'trig([\d.]+)-([\d.]+)_assoc([\d.]+)-([\d.]+)',
        pt_key
    )
    if m:
        return tuple(float(x) for x in m.groups())
    return (np.nan,) * 4


# ──────────────────────────────────────────────────────────────────────────────
# Helpers (unchanged from original)
# ──────────────────────────────────────────────────────────────────────────────
def _parse_pt_ranges_from_filename(fname):
    m = re.search(
        r'trig([\d.]+)-([\d.]+)_assoc([\d.]+)-([\d.]+)',
        os.path.basename(fname)
    )
    if m:
        return tuple(float(x) for x in m.groups())
    return None


def load_dphi_csvs(base_dir, subdirs):
    all_data   = {}
    all_ptkeys = set()

    for sd in subdirs:
        path    = os.path.join(base_dir, sd)
        pattern = os.path.join(path, 'dphi_projection_*.csv')
        files   = sorted(glob.glob(pattern))

        if not files:
            print(f"  [warn] No dphi CSVs found in: {path}")
            all_data[sd] = {}
            continue

        all_data[sd] = {}

        for fpath in files:
            pt = _parse_pt_ranges_from_filename(fpath)
            if pt is None:
                print(f"  [warn] Cannot parse pT ranges from: {fpath}")
                continue

            trig_lo, trig_hi, assoc_lo, assoc_hi = pt
            pt_key = f'trig{trig_lo:.1f}-{trig_hi:.1f}_assoc{assoc_lo:.1f}-{assoc_hi:.1f}'

            df = pd.read_csv(fpath, comment='#')
            bkg_col = df['zyam_background'].iloc[0] if 'zyam_background' in df.columns else 0.0

            if 'stat_err_jackknife' in df.columns:
                err = df['stat_err_jackknife'].fillna(0.0).values
            else:
                err = np.zeros(len(df))

            all_data[sd][pt_key] = {
                'phi':        df['delta_phi_rad'].values,
                'central':    df['dNpair_dDeltaPhi'].values,
                'err':        err,
                'background': bkg_col,
                'pt_ranges':  pt,
            }
            all_ptkeys.add(pt_key)
            print(f"  Loaded [{sd}] {pt_key}  ({len(df)} bins)")

    return all_data, sorted(all_ptkeys)


def load_exp_data(exp_dir, trig_lo, trig_hi, assoc_lo, assoc_hi):
    stem = f'ALICE_{trig_lo:.0f}-{trig_hi:.0f}_{assoc_lo:.0f}-{assoc_hi:.0f}'
    for ext in ('.csv', '.txt'):
        fpath = os.path.join(exp_dir, stem + ext)
        if os.path.exists(fpath):
            try:
                df = pd.read_csv(fpath)
                phi       = df['DeltaPhi'].values
                y         = df['d2Npair'].values
                y_err_pos = df['stat +'].values if 'stat +' in df.columns else np.zeros(len(df))
                y_err_neg = np.abs(df['stat -'].values) if 'stat -' in df.columns else np.zeros(len(df))
                idx = np.argsort(phi)
                return phi[idx], y[idx], y_err_pos[idx], y_err_neg[idx]
            except Exception as e:
                print(f"  [warn] Could not read STAR/ALICE file {fpath}: {e}")
    return None, None, None, None


def _phi_ticks():
    ticks  = [-np.pi/2, 0, np.pi/2, np.pi, 3*np.pi/2]
    labels = [r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$']
    return ticks, labels


# ──────────────────────────────────────────────────────────────────────────────
# Printing helpers
# ──────────────────────────────────────────────────────────────────────────────
def print_fit_table(rows):
    """Pretty-print a list of fit-result dicts."""
    hdr = (
        f"{'Source':<22} {'pt_key':<42} {'ok':>3}  "
        f"{'A_near':>10} {'±':>7}  "
        f"{'σ_near':>8} {'±':>7}  "
        f"{'A_away':>10} {'±':>7}  "
        f"{'σ_away':>8} {'±':>7}  "
        f"{'C':>10} {'±':>7}"
    )
    print("\n" + "=" * len(hdr))
    print("DOUBLE-GAUSSIAN FIT RESULTS")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        ok_str = "✓" if r['fit_ok'] else "✗"
        print(
            f"{r['source']:<22} {r['pt_key']:<42} {ok_str:>3}  "
            f"{r['A_near']:>10.4f} {r['A_near_err']:>7.4f}  "
            f"{r['sigma_near']:>8.4f} {r['sigma_near_err']:>7.4f}  "
            f"{r['A_away']:>10.4f} {r['A_away_err']:>7.4f}  "
            f"{r['sigma_away']:>8.4f} {r['sigma_away_err']:>7.4f}  "
            f"{r['pedestal']:>10.4f} {r['pedestal_err']:>7.4f}"
        )
    print("=" * len(hdr) + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Main plotting routine
# ──────────────────────────────────────────────────────────────────────────────
def plot_dphi_comparison(all_data, pt_key, exp_dir, save_dir,
                         subdirs, style, show_ratio=True,
                         fit_rows=None):

    pt_ranges = None
    for sd in subdirs:
        if pt_key in all_data.get(sd, {}):
            pt_ranges = all_data[sd][pt_key]['pt_ranges']
            break
    if pt_ranges is None:
        print(f"  [skip] No data found for {pt_key}")
        return

    trig_lo, trig_hi, assoc_lo, assoc_hi = pt_ranges
    exp_phi, exp_y, exp_y_err_pos, exp_y_err_neg = load_exp_data(
        exp_dir, trig_lo, trig_hi, assoc_lo, assoc_hi
    )

    if show_ratio:
        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1,
            figsize=(8, 9),
            gridspec_kw={'height_ratios': [3, 1.4], 'hspace': 0.08},
            sharex=True,
        )
    else:
        fig, ax_top = plt.subplots(figsize=(8, 6))
        ax_bot = None

    phi_ticks, phi_labels = _phi_ticks()

    # Dense grid for drawing fit curves
    phi_dense = np.linspace(-1.31, 4.98, 500)

    # ── top panel: projections + fit curves ──────────────────────────────────
    for sd in subdirs:
        if pt_key not in all_data.get(sd, {}):
            continue
        entry = all_data[sd][pt_key]
        phi   = entry['phi']
        y     = entry['central']
        err   = entry['err']
        st    = style[sd]

        ax_top.step(phi, y, where='mid', color=st['color'],
                    linestyle="-", linewidth=2, label=st['label'])
        if np.any(err > 0):
            ax_top.errorbar(
                phi, y, yerr=err,
                fmt="none", ms=MS, color=st['color'],
                capsize=CAPSIZE, capthick=1.6, elinewidth=1.6,
            )

        # Overlay fit curve
        popt, perr, success = fit_double_gaussian(
            phi, y, err if np.any(err > 0) else None
        )
        if success:
            ax_top.plot(
                phi_dense, double_gaussian(phi_dense, *popt),
                color=st['color'], ls='--', lw=1.2, alpha=0.8,
            )

        # Collect fit row
        if fit_rows is not None:
            label = style[sd]['label'] if sd in style else sd
            fit_rows.append(_format_fit_row(label, pt_key, popt, perr, success))

    if exp_phi is not None:
        ax_top.step(exp_phi, exp_y, where='mid',
                    color=EXP_STYLE['color'], linestyle=EXP_STYLE['ls'],
                    linewidth=2, zorder=EXP_STYLE['zorder'], label=EXP_STYLE['label'])
        ax_top.errorbar(
            exp_phi, exp_y,
            yerr=[exp_y_err_neg, exp_y_err_pos],
            fmt="none", ms=MS, color=EXP_STYLE['color'],
            linewidth=2, capsize=CAPSIZE, capthick=1.6, elinewidth=1.6,
        )

        # Fit experimental data
        avg_err = (exp_y_err_pos + exp_y_err_neg) / 2.0
        popt_e, perr_e, success_e = fit_double_gaussian(
            exp_phi, exp_y, avg_err
        )
        if success_e:
            ax_top.plot(
                phi_dense, double_gaussian(phi_dense, *popt_e),
                color=EXP_STYLE['color'], ls='--', lw=1.2, alpha=0.8,
            )
        if fit_rows is not None:
            fit_rows.append(
                _format_fit_row('ALICE data', pt_key, popt_e, perr_e, success_e)
            )

    ax_top.axhline(0, color='k', lw=0.6, ls='--', alpha=0.35)
    ax_top.set_ylabel(
        r'$\frac{1}{N_{\rm trig}}\frac{dN_{\rm pair}}{d\Delta\phi}$',
        fontsize=17
    )
    ax_top.set_title(
        f'ALICE Dihadron $\\Delta\\phi$ projection\n'
        f'Trigger $p_T$: {trig_lo:.1f}–{trig_hi:.1f} GeV/c   '
        f'Assoc $p_T$: {assoc_lo:.1f}–{assoc_hi:.1f} GeV/c   '
        f'$|\\eta|$ < 1.0',
        fontsize=13, fontweight='bold',
    )
    ax_top.legend(fontsize=9, framealpha=0.85, loc='upper right')
    ax_top.grid(True, alpha=0.25, lw=0.6)
    ax_top.set_xlim(17*np.pi/41 - 0.1, 65*np.pi/41 + 0.1)
    ax_top.tick_params(axis='y', labelsize=13)

    if not show_ratio:
        ax_top.set_xlabel(r'$\Delta\phi$ [rad]', fontsize=13)
        ax_top.set_xticks(phi_ticks)
        ax_top.set_xticklabels(phi_labels, fontsize=13)

    # ── bottom panel: ratio ───────────────────────────────────────────────────
    if show_ratio and ax_bot is not None:
        if exp_phi is not None:
            for sd in subdirs:
                if pt_key not in all_data.get(sd, {}):
                    continue
                entry = all_data[sd][pt_key]
                phi   = entry['phi']
                y     = entry['central']
                err   = entry['err']
                st    = style[sd]

                exp_interp         = np.interp(phi, exp_phi, exp_y)
                exp_err_pos_interp = np.interp(phi, exp_phi, exp_y_err_pos)
                exp_err_neg_interp = np.interp(phi, exp_phi, exp_y_err_neg)

                with np.errstate(invalid='ignore', divide='ignore'):
                    ratio = np.where(y != 0, exp_interp / y, np.nan)
                    exp_rel    = np.where(
                        ratio >= 0,
                        exp_err_pos_interp / np.abs(exp_interp + 1e-30),
                        exp_err_neg_interp / np.abs(exp_interp + 1e-30),
                    )
                    pythia_rel = err / np.abs(y + 1e-30)
                    ratio_err  = np.where(
                        np.isfinite(ratio),
                        np.abs(ratio) * np.sqrt(exp_rel**2 + pythia_rel**2),
                        np.nan,
                    )

                has_err = np.any(np.isfinite(ratio_err) & (ratio_err > 0))
                if has_err:
                    ax_bot.errorbar(
                        phi, ratio, yerr=ratio_err,
                        fmt=st['marker'], ms=MS, color=st['color'],
                        capsize=CAPSIZE, capthick=0.8, elinewidth=0.8,
                        linestyle=st['ls'], linewidth=st['lw'],
                        label=sd,
                    )
                else:
                    ax_bot.plot(
                        phi, ratio,
                        marker=st['marker'], ms=MS, color=st['color'],
                        linestyle=st['ls'], linewidth=st['lw'],
                        label=sd,
                    )

            ax_bot.axhline(1.0, color='k', lw=1.0, ls='--', alpha=0.5)
            ax_bot.set_ylabel(r'ALICE / Pythia', fontsize=11)
            ax_bot.set_ylim(0, 2)
            ax_bot.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))
            ax_bot.grid(True, alpha=0.25, lw=0.6)
            ax_bot.grid(True, which='minor', alpha=0.12, lw=0.4)
        else:
            ax_bot.text(0.5, 0.5, 'ALICE data not available for this window',
                        ha='center', va='center', transform=ax_bot.transAxes,
                        fontsize=10, color='gray')
            ax_bot.set_ylabel(r'ALICE / Pythia', fontsize=11)

        ax_bot.set_xlabel(r'$\Delta\phi$ [rad]', fontsize=12)
        ax_bot.set_xticks(phi_ticks)
        ax_bot.set_xticklabels(phi_labels, fontsize=11)

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f'dphi_comparison_{pt_key}_{SUFFIX}.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {out_path}")
    plt.close(fig)
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("Loading dphi projection CSVs …")
    all_data, pt_keys = load_dphi_csvs(BASE_DIR, SUBDIRS)

    if not pt_keys:
        raise RuntimeError(
            f"No dphi_projection_*.csv files found under {BASE_DIR}. "
            "Run the main analysis script first."
        )

    print(f"\nFound kinematic windows: {pt_keys}")
    print(f"Saving comparison plots to: {OUTPUT_DIR}\n")

    fit_rows = []   # accumulate fit results across all windows
    saved    = []

    for pt_key in pt_keys:
        print(f"Plotting + fitting: {pt_key}")
        path = plot_dphi_comparison(
            all_data   = all_data,
            pt_key     = pt_key,
            exp_dir    = EXP_DIR,
            save_dir   = OUTPUT_DIR,
            subdirs    = SUBDIRS,
            style      = STYLE,
            show_ratio = SHOW_RATIO,
            fit_rows   = fit_rows,
        )
        if path:
            saved.append(path)

    # ── Print and save fit results ────────────────────────────────────────────
    if fit_rows:
        print_fit_table(fit_rows)

        fit_df   = pd.DataFrame(fit_rows)
        csv_path = os.path.join(OUTPUT_DIR, f'double_gaussian_fit_results_{SUFFIX}.csv')
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        fit_df.to_csv(csv_path, index=False, float_format='%.6f')
        print(f"Fit results CSV written to: {csv_path}")
    else:
        print("[warn] No fit results collected.")

    print(f"\nDone. {len(saved)} figure(s) written.")


if __name__ == '__main__':
    main()