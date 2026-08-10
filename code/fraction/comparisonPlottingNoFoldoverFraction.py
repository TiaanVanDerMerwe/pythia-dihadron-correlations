"""
Δφ Projection Comparison Plot
==============================
Loads the dphi_projection CSVs written by save_projections_to_csv() from
each Pythia filter configuration, overlays them on a single panel, and
compares against STAR/ALICE data.

Directory layout assumed
------------------------
plots/Correlations/star(alice)/
    DecaysRestricted_HardQCD/  dphi_projection_*.csv
    Default_HardQCD/           dphi_projection_*.csv

STAR/ALICE reference data
------------------
datathief/STAR(ALICE)_{trig_lo}-{trig_hi}_{zT_lo}-{zT_hi}.csv
    columns: DeltaPhi, d2Npair, stat +, stat -
"""

import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR = "plots/correlations/test"
EXP_DIR = "datathief"
OUTPUT_DIR = "plots/comparisons/test"

SUBDIRS = [
    "nofsr",
    "fsr",
]

SHOW_RATIO = False

# Colour and marker style per configuration (order matches SUBDIRS)
STYLE = {
    "nofsr": {
        "label": "No FSR",
        "color": "#E8700A",
        "marker": "^",
        "ls": "-",
        "lw": 1.4,
    },
    "fsr": {"label": "FSR", "color": "#4453FF", "marker": "^", "ls": "-", "lw": 1.4},
}

EXP_STYLE = {
    "color": "black",
    "marker": "*",
    "ms": 8,
    "ls": "-",
    "markerfacecolor": "black",
    "markeredgewidth": 0.8,
    "zorder": 10,
    "label": "Test data",
}

MS = 4  # marker size for Pythia points
CAPSIZE = 2  # error bar cap size


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _parse_pt_ranges_from_filename(fname):
    """
    Extract trigger pT and zT ranges from a filename like
        dphi_projection_trig4.0-6.0_zT2.0-4.0_pow3.csv
    Returns (trig_lo, trig_hi, zT_lo, zT_hi) as floats, or None.
    """
    m = re.search(r"trig([\d.]+)-([\d.]+)_zT([\d.]+)-([\d.]+)", os.path.basename(fname))
    if m:
        return tuple(float(x) for x in m.groups())
    return None


def load_dphi_csvs(base_dir, subdirs):
    """
    Scan each subdir for dphi_projection_*.csv files, group them by
    kinematic window, and return a nested dict:

        data[subdir][pt_key] = {
            'phi':        1D array,
            'central':    1D array,
            'err':        1D array,
            'background': float,
            'pt_ranges':  (trig_lo, trig_hi, zT_lo, zT_hi),
        }

    where pt_key = 'trig{lo}-{hi}_zT{lo}-{hi}'.
    """
    all_data = {}
    all_ptkeys = set()

    for sd in subdirs:
        path = os.path.join(base_dir, sd)
        pattern = os.path.join(path, "dphi_projection_*.csv")
        files = sorted(glob.glob(pattern))

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

            _, _, _, _ = pt

            base = os.path.basename(fpath).replace(".csv", "")

            pt_key = base.replace("dphi_projection_", "")

            df = pd.read_csv(fpath, comment="#")
            bkg_col = (
                df["zyam_background"].iloc[0]
                if "zyam_background" in df.columns
                else 0.0
            )

            # read error column — zeros if missing or all-NaN
            if "stat_err_jackknife" in df.columns:
                err = df["stat_err_jackknife"].fillna(0.0).values
            else:
                err = np.zeros(len(df))

            all_data[sd][pt_key] = {
                "phi": df["delta_phi_rad"].values,
                "central": df["dNpair_dDeltaPhi"].values,
                "err": err,
                "background": bkg_col,
                "pt_ranges": pt,
            }
            all_ptkeys.add(pt_key)
            print(f"  Loaded [{sd}] {pt_key}  ({len(df)} bins)")

    print(all_data)
    return all_data, sorted(all_ptkeys)


def load_exp_data(exp_dir, trig_lo, trig_hi, zT_lo, zT_hi):
    """
    Try to load STAR/ALICE reference data for a given kinematic window.
    Returns (phi, y) arrays or (None, None).
    """
    stem = f"STAR_{trig_lo:.0f}-{trig_hi:.0f}_{zT_lo:.0f}-{zT_hi:.0f}"
    for ext in (".csv", ".txt"):
        fpath = os.path.join(exp_dir, stem + ext)
        if os.path.exists(fpath):
            try:
                df = pd.read_csv(fpath)
                phi = df["DeltaPhi"].values
                y = df["d2Npair"].values

                # Use stat columns if present, otherwise fall back to zeros
                y_err_pos = (
                    df["stat +"].values if "stat +" in df.columns else np.zeros(len(df))
                )
                y_err_neg = (
                    np.abs(df["stat -"].values)
                    if "stat -" in df.columns
                    else np.zeros(len(df))
                )

                idx = np.argsort(phi)
                return phi[idx], y[idx], y_err_pos[idx], y_err_neg[idx]
            except Exception as e:  # noqa: BLE001
                print(f"  [warn] Could not read STAR/ALICE file {fpath}: {e}")
    return None, None, None, None


def _phi_ticks():
    ticks = [-np.pi / 2, 0, np.pi / 2, np.pi, 3 * np.pi / 2]
    labels = [r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$"]
    return ticks, labels


# ──────────────────────────────────────────────────────────────────────────────
# Main plotting routine
# ──────────────────────────────────────────────────────────────────────────────
def plot_dphi_comparison(
    all_data, pt_key, exp_dir, save_dir, subdirs, style, show_ratio=True
):

    pt_ranges = None
    for sd in subdirs:
        if pt_key in all_data.get(sd, {}):
            pt_ranges = all_data[sd][pt_key]["pt_ranges"]
            break
    if pt_ranges is None:
        print(f"  [skip] No data found for {pt_key}")
        return

    trig_lo, trig_hi, zT_lo, zT_hi = pt_ranges
    exp_phi, exp_y, exp_y_err_pos, exp_y_err_neg = load_exp_data(
        exp_dir, trig_lo, trig_hi, zT_lo, zT_hi
    )

    if show_ratio:
        fig, (ax_top, ax_bot) = plt.subplots(
            2,
            1,
            figsize=(8, 9),
            gridspec_kw={"height_ratios": [3, 1.4], "hspace": 0.08},
            sharex=True,
        )
    else:
        fig, ax_top = plt.subplots(figsize=(8, 6))
        ax_bot = None

    phi_ticks, phi_labels = _phi_ticks()

    # ── top panel: projections ────────────────────────────────────────────────
    for sd in subdirs:
        if pt_key not in all_data.get(sd, {}):
            continue
        entry = all_data[sd][pt_key]
        phi = entry["phi"]
        y = entry["central"]
        err = entry["err"]
        st = style[sd]

        ax_top.step(
            phi,
            y,
            where="mid",
            color=st["color"],
            linestyle="-",
            linewidth=2,
            label=st["label"],
        )
        if np.any(err > 0):
            ax_top.errorbar(
                phi,
                y,
                yerr=err,
                fmt="none",
                ms=MS,
                color=st["color"],
                capsize=CAPSIZE,
                capthick=1.6,
                elinewidth=1.6,
                linestyle=st["ls"],
                linewidth=2,
            )

    if exp_phi is not None:
        ax_top.step(
            exp_phi,
            exp_y,
            where="mid",
            color=EXP_STYLE["color"],
            linestyle=EXP_STYLE["ls"],
            linewidth=2,
            zorder=EXP_STYLE["zorder"],
            label=EXP_STYLE["label"],
        )
        ax_top.errorbar(
            exp_phi,
            exp_y,
            yerr=[exp_y_err_neg, exp_y_err_pos],
            fmt="none",
            ms=MS,
            color=EXP_STYLE["color"],
            linestyle=st["ls"],
            linewidth=2,
            capsize=CAPSIZE,
            capthick=1.6,
            elinewidth=1.6,
        )

    ax_top.axhline(0, color="k", lw=0.6, ls="--", alpha=0.35)
    ax_top.set_ylabel(
        r"$\frac{1}{N_{\rm trig}}\frac{dN_{\rm pair}}{d\Delta\phi}$", fontsize=17
    )
    ax_top.set_title(
        f"STAR Dihadron $\\Delta\\phi$ projection\n"
        f"Trigger $p_T$: {trig_lo:.1f}–{trig_hi:.1f} GeV/c   "
        f"$z_T$: {zT_lo:.1f}–{zT_hi:.1f} GeV/c   "
        f"$|\\eta|$ < 1",
        fontsize=13,
        fontweight="bold",
    )
    ax_top.legend(fontsize=9, framealpha=0.85, loc="upper right")
    ax_top.grid(True, alpha=0.25, lw=0.6)
    ax_top.set_xlim(17 * np.pi / 41 - 0.1, 65 * np.pi / 41 + 0.1)
    ax_top.tick_params(axis="y", labelsize=13)

    # Apply x-axis labels to top panel if no ratio panel
    if not show_ratio:
        ax_top.set_xlabel(r"$\Delta\phi$ [rad]", fontsize=13)
        ax_top.set_xticks(phi_ticks)
        ax_top.set_xticklabels(phi_labels, fontsize=13)

    # ── bottom panel: ratio ALICE / config ───────────────────────────────────
    if show_ratio and ax_bot is not None:
        if exp_phi is not None:
            for sd in subdirs:
                if pt_key not in all_data.get(sd, {}):
                    continue
                entry = all_data[sd][pt_key]
                phi = entry["phi"]
                y = entry["central"]
                err = entry["err"]
                st = style[sd]

                exp_interp = np.interp(phi, exp_phi, exp_y)
                exp_err_pos_interp = np.interp(phi, exp_phi, exp_y_err_pos)
                exp_err_neg_interp = np.interp(phi, exp_phi, exp_y_err_neg)

                with np.errstate(invalid="ignore", divide="ignore"):
                    ratio = np.where(y != 0, exp_interp / y, np.nan)
                    exp_rel = np.where(
                        ratio >= 0,
                        exp_err_pos_interp / np.abs(exp_interp + 1e-30),
                        exp_err_neg_interp / np.abs(exp_interp + 1e-30),
                    )
                    pythia_rel = err / np.abs(y + 1e-30)
                    ratio_err = np.where(
                        np.isfinite(ratio),
                        np.abs(ratio) * np.sqrt(exp_rel**2 + pythia_rel**2),
                        np.nan,
                    )

                has_err = np.any(np.isfinite(ratio_err) & (ratio_err > 0))
                if has_err:
                    ax_bot.errorbar(
                        phi,
                        ratio,
                        yerr=ratio_err,
                        fmt=st["marker"],
                        ms=MS,
                        color=st["color"],
                        capsize=CAPSIZE,
                        capthick=0.8,
                        elinewidth=0.8,
                        linestyle=st["ls"],
                        linewidth=st["lw"],
                        label=sd,
                    )
                else:
                    ax_bot.plot(
                        phi,
                        ratio,
                        marker=st["marker"],
                        ms=MS,
                        color=st["color"],
                        linestyle=st["ls"],
                        linewidth=st["lw"],
                        label=sd,
                    )

            ax_bot.axhline(1.0, color="k", lw=1.0, ls="--", alpha=0.5)
            ax_bot.set_ylabel(r"ALICE / Pythia", fontsize=11)
            ax_bot.set_ylim(0, 2)
            ax_bot.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))
            ax_bot.grid(True, alpha=0.25, lw=0.6)
            ax_bot.grid(True, which="minor", alpha=0.12, lw=0.4)

        else:
            ax_bot.text(
                0.5,
                0.5,
                "ALICE data not available for this window",
                ha="center",
                va="center",
                transform=ax_bot.transAxes,
                fontsize=10,
                color="gray",
            )
            ax_bot.set_ylabel(r"ALICE / Pythia", fontsize=11)

        ax_bot.set_xlabel(r"$\Delta\phi$ [rad]", fontsize=12)
        ax_bot.set_xticks(phi_ticks)
        ax_bot.set_xticklabels(phi_labels, fontsize=11)

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"star_dphi_comparison_{pt_key}.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
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

    saved = []
    for pt_key in pt_keys:
        print(f"Plotting: {pt_key}")
        path = plot_dphi_comparison(
            all_data=all_data,
            pt_key=pt_key,
            exp_dir=EXP_DIR,
            save_dir=OUTPUT_DIR,
            subdirs=SUBDIRS,
            style=STYLE,
            show_ratio=SHOW_RATIO,
        )
        if path:
            saved.append(path)

    print(f"\nDone. {len(saved)} figure(s) written.")


if __name__ == "__main__":
    main()
