"""
Dihadron Correlation Analysis — zT Slice Yields (Hadron-Triggered FF)
======================================================================

Per-zT slice correlation functions with ZYAM subtraction, with per-trigger
integrated near- and away-side yields as a function of zT — the
hadron-triggered fragmentation function (HTFF).

The HTFF is defined as:
    D(zT) = (1 / N_trig) * dN_pair / dzT

integrated over the near- or away-side Delta-phi window, where
    zT = pT_assoc / pT_trig

Uncertainty method
------------------
Block jackknife: compute_phi_projection_jackknife returns the per-bin
central errors and the full (n_blocks x n_phi) array of per-block
projections.

Integrated yield uncertainties are computed by integrating each block's
projection first, then applying the jackknife variance formula:

    Y_k    = sum_{phi in side} max(proj_k(phi), 0) * Dphi / DzT
    Var_JK = (N-1)/N * sum_k (Y_k - mean(Y_k))^2
    sigma  = sqrt(Var_JK)

zT slices:  [0.3,0.4], [0.4,0.5], [0.5,0.6], [0.6,0.7], [0.7,0.8], [0.8,0.9], [0.9,1.0]
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Re-use helpers from the base analysis script
# ─────────────────────────────────────────────────────────────────────────────
from projectionsNoFoldoverFraction import (
    compute_phi_projection_jackknife,
    read_and_combine_bins,
)

# ─────────────────────────────────────────────────────────────────────────────
# Analysis configuration
# ─────────────────────────────────────────────────────────────────────────────
TRIG_PT_MIN = 8.0
TRIG_PT_MAX = 15.0  # GeV/c

ZT_EDGES = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # dimensionless

# Full unfolded Delta-phi range
PHI_RANGE = (-np.pi / 2, 3 * np.pi / 2)

# Near-side window
NEAR_SIDE_LO = -0.63
NEAR_SIDE_HI = 0.63

# Away-side window: pi ± ~0.63 rad
AWAY_SIDE_LO = 2.51
AWAY_SIDE_HI = 3.77

# ZYAM search window: between the near and away peaks
ZYAM_PHI_LO = 0.63
ZYAM_PHI_HI = 2.51

ETA_CUT = 2
ETA_BINS = 15
PHI_BINS = 36
ETA_RANGE = (-ETA_CUT, ETA_CUT)

zyam = True

N_JACKKNIFE_BLOCKS = 5


# ─────────────────────────────────────────────────────────────────────────────
# Per-slice computation
# ─────────────────────────────────────────────────────────────────────────────


def compute_slice(data, zt_min, zt_max, ntrig_override):
    """
    Run the full jackknife pipeline for one zT slice.
    """
    zt = data["assoc_pT"] / data["trigger_pT"]
    zt_mask = (zt > zt_min) & (zt <= zt_max)
    sliced = data[zt_mask].copy()

    if len(sliced) == 0:
        print(f"  WARNING: No pairs in slice zT [{zt_min}, {zt_max}]")
        return None

    print(
        f"\n── zT [{zt_min:.2f}, {zt_max:.2f}]  "
        f"({len(sliced):,} pairs) ──────────────────────────────────"
    )

    metadata_slice = {
        "CUT_TRIG_PT_RANGE": [TRIG_PT_MIN, TRIG_PT_MAX],
        "CUT_ZT_RANGE": [zt_min, zt_max],
    }

    (phi_proj, phi_proj_err, phi_centers, ntrig, bkg, jackknife_projs) = (
        compute_phi_projection_jackknife(
            sliced,
            metadata_slice,
            eta_bins=ETA_BINS,
            phi_bins=PHI_BINS,
            eta_range=ETA_RANGE,
            phi_range=PHI_RANGE,
            zyam=zyam,
            zyam_range={"phi": (ZYAM_PHI_LO, ZYAM_PHI_HI)},
            ntrig_override=ntrig_override,
            n_blocks=N_JACKKNIFE_BLOCKS,
        )
    )

    return {
        "phi_centers": phi_centers,
        "phi_proj": phi_proj,
        "phi_proj_err": phi_proj_err,
        "jackknife_projs": jackknife_projs,
        "background": bkg,
        "ntrig": ntrig,
        "zt_range": (zt_min, zt_max),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Yield integration — HTFF: divide by DzT, not DpT
# ─────────────────────────────────────────────────────────────────────────────


def _integrate_one(proj, phi_centers, near_mask, away_mask, bw, delta_zt):
    signal = np.where(proj > 0, proj, 0.0)
    Y_near = float((signal[near_mask] * bw).sum() / delta_zt)
    Y_away = float((signal[away_mask] * bw).sum() / delta_zt)
    return Y_near, Y_away


def integrate_yield(
    phi_centers,
    phi_proj,
    jackknife_projs,
    zt_range,
    near_lo=NEAR_SIDE_LO,
    near_hi=NEAR_SIDE_HI,
    away_lo=AWAY_SIDE_LO,
    away_hi=AWAY_SIDE_HI,
):
    bw = phi_centers[1] - phi_centers[0]
    delta_zt = zt_range[1] - zt_range[0]

    near_mask = (phi_centers >= near_lo) & (phi_centers <= near_hi)
    away_mask = (phi_centers >= away_lo) & (phi_centers <= away_hi)

    Y_near, Y_away = _integrate_one(
        phi_proj, phi_centers, near_mask, away_mask, bw, delta_zt
    )

    n_blocks = len(jackknife_projs)
    jk_near = np.array(
        [
            _integrate_one(
                jackknife_projs[k], phi_centers, near_mask, away_mask, bw, delta_zt
            )[0]
            for k in range(n_blocks)
        ]
    )
    jk_away = np.array(
        [
            _integrate_one(
                jackknife_projs[k], phi_centers, near_mask, away_mask, bw, delta_zt
            )[1]
            for k in range(n_blocks)
        ]
    )

    factor = (n_blocks - 1) / n_blocks
    sigma_near = float(np.sqrt(factor * np.sum((jk_near - jk_near.mean()) ** 2)))
    sigma_away = float(np.sqrt(factor * np.sum((jk_away - jk_away.mean()) ** 2)))

    return Y_near, Y_away, sigma_near, sigma_away


# ─────────────────────────────────────────────────────────────────────────────
# STAR data loader
# ─────────────────────────────────────────────────────────────────────────────


def load_star_data(trig_pt_range, data_dir="."):
    """
    Load STAR near- and away-side HTFF data from
        STAR_near_{tlo:.0f}-{thi:.0f}.csv
        STAR_away_{tlo:.0f}-{thi:.0f}.csv

    Each file has the layout (blank-row separated blocks):
        zT, Yield in d+Au,          stat +, stat -
        zT, Yield in Au+Au 0-5%,    stat +, stat -
        zT, Yield in Au+Au 20-40%,  stat +, stat -

    Returns
    -------
    dict with keys 'near' and 'away'; each value is a dict:
        {
            'dAu':       {'zt': array, 'y': array, 'err': array},
            'AuAu_0_5':  {'zt': array, 'y': array, 'err': array},
            'AuAu_20_40':{'zt': array, 'y': array, 'err': array},
        }
    Missing files return an empty dict for that side.
    """
    tlo, thi = trig_pt_range
    result = {}

    for side in ("near", "away"):
        fname = os.path.join(data_dir, f"STAR_{side}_{tlo:.0f}-{thi:.0f}.csv")
        if not os.path.exists(fname):
            print(f"  Warning: STAR data file not found: {fname}")
            result[side] = {}
            continue

        # Read raw text and split on blank rows
        raw = pd.read_csv(fname, header=None, skip_blank_lines=False)
        blocks = []
        current = []
        for _, row in raw.iterrows():
            if row.isnull().all():
                if current:
                    blocks.append(current)
                    current = []
            else:
                current.append(row.values)
        if current:
            blocks.append(current)

        system_keys = ["dAu", "AuAu_0_5", "AuAu_20_40"]
        side_data = {}
        for i, block in enumerate(blocks):
            if i >= len(system_keys):
                break
            arr = np.array(block[1:], dtype=float)  # skip header row
            if arr.shape[1] < 4:
                continue
            zt = arr[:, 0]
            y = arr[:, 1]
            # average of |stat+| and |stat-| as symmetric error
            err = (np.abs(arr[:, 2]) + np.abs(arr[:, 3])) / 2.0
            idx = np.argsort(zt)
            side_data[system_keys[i]] = {
                "zt": zt[idx],
                "y": y[idx],
                "err": err[idx],
            }
        result[side] = side_data

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Plotting — Delta-phi slices
# ─────────────────────────────────────────────────────────────────────────────

_COLORS = ["#4453FF"]

_PHI_TICKS = [-np.pi / 2, 0, np.pi / 2, np.pi, 3 * np.pi / 2]
_PHI_LABELS = [r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$"]


def plot_dphi_slices(
    slice_results, trig_pt_range=(TRIG_PT_MIN, TRIG_PT_MAX), save_path=None
):
    """One panel per zT slice: ZYAM-subtracted Dphi with JK error band."""
    valid = [r for r in slice_results if r is not None]
    ncols = 3
    nrows = int(np.ceil(len(valid) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False
    )

    for i, res in enumerate(valid):
        ax = axes[i // ncols][i % ncols]
        color = _COLORS[i % len(_COLORS)]
        zlo, zhi = res["zt_range"]

        phi = res["phi_centers"]
        sig = res["phi_proj"]
        err = res["phi_proj_err"]

        ax.step(
            phi,
            sig,
            where="mid",
            color=color,
            linestyle="-",
            linewidth=2,
            label=f"Pythia  ZYAM={res['background']:.4f}",
            zorder=4,
        )
        ax.errorbar(
            phi,
            sig,
            yerr=err,
            fmt="none",
            ls="none",
            color=color,
            ms=5,
            lw=1.4,
            zorder=4,
        )

        ax.axvline(NEAR_SIDE_LO, color="green", lw=1.4, ls="--", alpha=0.7)
        ax.axvline(NEAR_SIDE_HI, color="green", lw=1.4, ls="--", alpha=0.7)
        ax.axvline(AWAY_SIDE_LO, color="red", lw=1.4, ls="--", alpha=0.7)
        ax.axvline(AWAY_SIDE_HI, color="red", lw=1.4, ls="--", alpha=0.7)

        ax.axhline(0, lw=1.4, ls="--", color="black", alpha=0.4)
        ax.set_xlim(-np.pi / 2 - 0.05, 3 * np.pi / 2 + 0.05)
        ax.set_xticks(_PHI_TICKS)
        ax.set_xticklabels(_PHI_LABELS, fontsize=11)
        ax.tick_params(axis="x", labelsize=13)
        ax.tick_params(axis="y", labelsize=13)
        ax.set_xlabel(r"$\Delta\phi$ [rad]", fontsize=13)
        ax.set_ylabel(
            r"$\frac{1}{N_{\rm trig}}\frac{dN_{\rm pair}}{d\Delta\phi}$", fontsize=17
        )
        ax.set_title(
            rf"$z_T$: {zlo:.2f}–{zhi:.2f}",
            fontsize=13,
            fontweight="bold",
            pad=6,
        )
        ax.legend(fontsize=7, framealpha=0.2)
        ax.grid(True, alpha=0.15)

    for j in range(len(valid), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle(
        r"Dihadron $\Delta\phi$ Distributions" + "\n"
        rf"Trig $p_T$: {trig_pt_range[0]:.1f}–{trig_pt_range[1]:.1f} GeV/c" + "\n"
        r"$|\eta|$ < 1" + "\n",
        fontsize=13,
        fontweight="bold",
        y=1.0,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    return fig, axes


# ─────────────────────────────────────────────────────────────────────────────
# Exponential fit helper
# ─────────────────────────────────────────────────────────────────────────────


def _exp_fit(x, y, yerr):
    mask = np.isfinite(y) & (y > 0) & np.isfinite(yerr) & (yerr > 0)
    if mask.sum() < 2:
        return None, None, None, None

    lx = x[mask]
    ly = np.log(y[mask])
    w = (y[mask] / yerr[mask]) ** 2

    W = w.sum()
    Wx = (w * lx).sum()
    Wx2 = (w * lx**2).sum()
    Wy = (w * ly).sum()
    Wxy = (w * lx * ly).sum()

    denom = W * Wx2 - Wx**2
    if denom == 0:
        return None, None, None, None

    b = (W * Wxy - Wx * Wy) / denom
    log_A = (Wy - b * Wx) / W

    cov_bb = W / denom
    cov_LALA = Wx2 / denom

    sigma_b = np.sqrt(cov_bb)
    sigma_logA = np.sqrt(cov_LALA)

    A = np.exp(log_A)
    sigma_A = A * sigma_logA

    return A, b, sigma_A, sigma_b


# ─────────────────────────────────────────────────────────────────────────────
# Plotting — integrated HTFF yields
# ─────────────────────────────────────────────────────────────────────────────

# Marker styles and colours for the three STAR systems
_STAR_STYLE = {
    "dAu": {"fmt": "s", "color": "#E8700A", "label": r"d+Au", "ms": 7},
    "AuAu_0_5": {"fmt": "D", "color": "#CC2222", "label": r"Au+Au 0–5%", "ms": 7},
    "AuAu_20_40": {"fmt": "^", "color": "#229922", "label": r"Au+Au 20–40%", "ms": 7},
}


def plot_integrated_yields(
    slice_results,
    trig_pt_range=(TRIG_PT_MIN, TRIG_PT_MAX),
    data_dir=".",
    save_path=None,
):
    """
    2×2 grid: near/away HTFF panels (top) + d+Au / Pythia ratio panels (bottom).

    Top panels show Pythia + exponential fit + all three STAR systems.
    Bottom panels show d+Au / Pythia only.
    """
    valid = [r for r in slice_results if r is not None]
    if not valid:
        print("No valid slices — nothing to plot.")
        return None, None

    # ── Pythia integrated yields ──────────────────────────────────────────
    zt_centers, xerr_lo, xerr_hi = [], [], []
    y_near, y_away, err_near, err_away = [], [], [], []

    for res in valid:
        zlo, zhi = res["zt_range"]
        zt_c = (zlo + zhi) / 2
        zt_centers.append(zt_c)
        xerr_lo.append(zt_c - zlo)
        xerr_hi.append(zhi - zt_c)

        Yn, Ya, sn, sa = integrate_yield(
            res["phi_centers"],
            res["phi_proj"],
            res["jackknife_projs"],
            zt_range=res["zt_range"],
        )
        y_near.append(Yn)
        y_away.append(Ya)
        err_near.append(sn)
        err_away.append(sa)

    zt_centers = np.array(zt_centers)
    y_near, y_away = np.array(y_near), np.array(y_away)
    err_near, err_away = np.array(err_near), np.array(err_away)
    xerr = [xerr_lo, xerr_hi]

    x_fit = np.linspace(zt_centers.min() * 0.95, zt_centers.max() * 1.05, 300)

    # ── Load STAR data ────────────────────────────────────────────────────
    star = load_star_data(trig_pt_range, data_dir=data_dir)

    # ── Build figure ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 6),
        gridspec_kw={"height_ratios": [2.5, 1.0], "hspace": 0.08, "wspace": 0.30},
    )

    pythia_color = "#4453FF"

    panels = [
        (
            axes[0, 0],
            axes[1, 0],
            y_near,
            err_near,
            "near",
            "Near side",
            r"$-0.63 < \Delta\phi < 0.63$ rad",
            r"$D^{\rm near}(z_T)$",
        ),
        (
            axes[0, 1],
            axes[1, 1],
            y_away,
            err_away,
            "away",
            "Away side",
            r"$2.51 < \Delta\phi < 3.77$ rad",
            r"$D^{\rm away}(z_T)$",
        ),
    ]

    for ax_top, ax_rat, yields, errs, side_key, label, phi_label, ylabel in panels:
        star_side = star.get(side_key, {})

        # ── Pythia points ─────────────────────────────────────────────────
        ax_top.errorbar(
            zt_centers,
            yields,
            xerr=xerr,
            yerr=errs,
            fmt="o",
            color=pythia_color,
            ms=7,
            markeredgewidth=0.6,
            ecolor=pythia_color,
            elinewidth=1.5,
            capsize=4,
            label="Pythia",
            zorder=5,
        )

        # ── Exponential fit to Pythia ─────────────────────────────────────
        A_py, b_py, sA_py, sb_py = _exp_fit(zt_centers, yields, errs)
        if A_py is not None:
            y_fit = A_py * np.exp(b_py * x_fit)
            rel_band = np.sqrt((sA_py / A_py) ** 2 + (x_fit * sb_py) ** 2)
            ax_top.plot(
                x_fit,
                y_fit,
                "--",
                color=pythia_color,
                lw=1.6,
                alpha=0.85,
                label=rf"Pythia fit  $b={b_py:.2f}\pm{sb_py:.2f}$",
                zorder=5,
            )
            ax_top.fill_between(
                x_fit,
                y_fit * (1 - rel_band),
                y_fit * (1 + rel_band),
                color=pythia_color,
                alpha=0.15,
                zorder=4,
            )

        # ── STAR systems ──────────────────────────────────────────────────
        for sys_key, sty in _STAR_STYLE.items():
            sys_data = star_side.get(sys_key)
            if sys_data is None:
                continue
            ax_top.errorbar(
                sys_data["zt"],
                sys_data["y"],
                xerr=0.05,  # half of 0.1 zT bin width
                yerr=sys_data["err"],
                fmt=sty["fmt"],
                color=sty["color"],
                ms=sty["ms"],
                markeredgewidth=0.6,
                ecolor=sty["color"],
                elinewidth=1.5,
                capsize=4,
                label=sty["label"],
                zorder=6,
            )

        ax_top.set_ylabel(ylabel, fontsize=13)
        ax_top.set_title(
            rf"{label}" + "\n" + rf"{phi_label}", fontsize=13, fontweight="bold", pad=8
        )
        ax_top.legend(fontsize=8, framealpha=0.2, loc="upper right")
        ax_top.grid(True, alpha=0.15)
        ax_top.set_yscale("log")
        ax_top.tick_params(labelbottom=False, labelsize=13)

        # ── Ratio panel: d+Au / Pythia ────────────────────────────────────
        dau_data = star_side.get("dAu")
        if dau_data is not None:
            # Match each d+Au zT point to the nearest Pythia bin centre.
            # Only keep the pair when they are within half a bin width (0.05);
            # this prevents spurious ratio points where d+Au has no data
            # (e.g. no 0.95 point on the near side).
            ZT_TOL = 0.05  # half of the 0.1 zT bin width

            rat_zt, rat_xerr = [], []
            rat_y, rat_err = [], []

            for zt_d, y_d, e_d in zip(dau_data["zt"], dau_data["y"], dau_data["err"]):
                dists = np.abs(zt_centers - zt_d)
                i_min = int(np.argmin(dists))
                if dists[i_min] > ZT_TOL:
                    continue  # no matching Pythia bin — skip

                py_y = yields[i_min]
                py_e = errs[i_min]
                if py_y == 0:
                    continue

                r = y_d / py_y
                with np.errstate(invalid="ignore"):
                    r_err = (
                        r * np.sqrt((e_d / y_d) ** 2 + (py_e / py_y) ** 2)
                        if y_d != 0
                        else np.nan
                    )

                rat_zt.append(zt_d)
                rat_xerr.append(0.05)
                rat_y.append(r)
                rat_err.append(r_err)

            if rat_zt:
                ax_rat.errorbar(
                    rat_zt,
                    rat_y,
                    xerr=rat_xerr,
                    yerr=rat_err,
                    fmt="s",
                    color=_STAR_STYLE["dAu"]["color"],
                    ms=6,
                    markeredgewidth=0.6,
                    ecolor=_STAR_STYLE["dAu"]["color"],
                    elinewidth=1.5,
                    capsize=4,
                    label="d+Au / Pythia",
                    zorder=4,
                )
            ax_rat.legend(fontsize=8, framealpha=0.2, loc="upper right")
        else:
            ax_rat.text(
                0.5,
                0.5,
                "d+Au data not loaded",
                ha="center",
                va="center",
                transform=ax_rat.transAxes,
                fontsize=9,
                color="gray",
            )

        ax_rat.axhline(1, lw=1.4, ls="--", color="black", alpha=0.4)
        ax_rat.set_xlabel(r"$z_T$", fontsize=13)
        ax_rat.set_ylabel("d+Au / Pythia", fontsize=11)
        ax_rat.set_ylim(0.0, 3.0)
        ax_rat.set_xlim(ax_top.get_xlim())  # keep x-axes aligned
        ax_rat.grid(True, alpha=0.15)
        ax_rat.tick_params(labelsize=12)

    fig.suptitle(
        rf"Hadron-Triggered Fragmentation Function  —  STAR vs Pythia"
        "\n"
        rf"Trig $p_T$: {trig_pt_range[0]:.1f}–{trig_pt_range[1]:.1f} GeV/c",
        fontsize=13,
        fontweight="bold",
        y=1.03,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  Saved: {save_path}")

        # ── Write Pythia D(zT) points to CSV under the parallel csv/ tree ─
        csv_save_path = save_path.replace("plots/", "csv/", 1)
        csv_path = os.path.splitext(csv_save_path)[0] + "_pythia.csv"
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df_out = pd.DataFrame(
            {
                "zT_center": zt_centers,
                "zT_lo": zt_centers - np.array(xerr_lo),
                "zT_hi": zt_centers + np.array(xerr_hi),
                "D_near": y_near,
                "D_near_err": err_near,
                "D_away": y_away,
                "D_away_err": err_away,
            }
        )
        df_out.to_csv(csv_path, index=False, float_format="%.6g")
        print(f"  Saved: {csv_path}")

    return fig, axes


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────


def print_yield_table(slice_results):
    W = 130

    w_zt = 20
    w_y = 14
    w_rel = 12

    print("\n" + "─" * W)

    header = (
        f"{'zT slice':^{w_zt}}"
        f"{'D_near(zT)':^{w_y}}"
        f"{'rel %(near)':^{w_rel}}"
        f"{'D_away(zT)':^{w_y}}"
        f"{'rel %(away)':^{w_rel}}"
        f"{'D_total(zT)':^{w_y}}"
        f"{'rel %(total)':^{w_rel}}"
    )
    print(header)
    print("─" * W)

    for res in (r for r in slice_results if r is not None):
        zlo, zhi = res["zt_range"]

        Yn, Ya, sn, sa = integrate_yield(
            res["phi_centers"],
            res["phi_proj"],
            res["jackknife_projs"],
            zt_range=res["zt_range"],
        )

        bw = res["phi_centers"][1] - res["phi_centers"][0]
        delta_zt = zhi - zlo
        signal = np.where(res["phi_proj"] > 0, res["phi_proj"], 0.0)
        Ytot = float((signal * bw).sum() / delta_zt)

        n_blocks = len(res["jackknife_projs"])
        jk_tot = np.array(
            [
                float(
                    (
                        np.where(
                            res["jackknife_projs"][k] > 0,
                            res["jackknife_projs"][k],
                            0.0,
                        )
                        * bw
                    ).sum()
                    / delta_zt
                )
                for k in range(n_blocks)
            ]
        )
        factor = (n_blocks - 1) / n_blocks
        stot = float(np.sqrt(factor * np.sum((jk_tot - jk_tot.mean()) ** 2)))

        rel_n = (sn / Yn * 100) if Yn != 0 else 0.0
        rel_a = (sa / Ya * 100) if Ya != 0 else 0.0
        rel_tot = (stot / Ytot * 100) if Ytot != 0 else 0.0

        row = (
            f"{f'[{zlo:.2f}, {zhi:.2f}]':^{w_zt}}"
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
    POW = 3
    BASE = "pythiaData/200/star/eta1"

    filenames = sorted(glob.glob(f"{BASE}/dihadron_pow{POW}_pT*.csv"))
    if not filenames:
        raise FileNotFoundError(f"No files found at {BASE}/dihadron_pow{POW}_pT*.csv")

    print(f"Found {len(filenames)} pTHat bin file(s):")
    for f in filenames:
        print(f"  {f}")

    print("\nLoading and combining bins ...")
    data, metadata = read_and_combine_bins(
        filenames,
        trig_pt_min=TRIG_PT_MIN,
        trig_pt_max=TRIG_PT_MAX,
        zt_min=ZT_EDGES[0],
        zt_max=ZT_EDGES[-1],
        eta_range=ETA_RANGE,
    )
    ntrig_override = metadata["N_TRIG"]

    print(
        f"\nRunning {len(ZT_EDGES) - 1} zT slices "
        f"({N_JACKKNIFE_BLOCKS} JK blocks each) ..."
    )

    slice_results = [
        compute_slice(data, ZT_EDGES[j], ZT_EDGES[j + 1], ntrig_override)
        for j in range(len(ZT_EDGES) - 1)
    ]

    print_yield_table(slice_results)

    out_dir = "outputs/plots/htff"
    trig_range = (TRIG_PT_MIN, TRIG_PT_MAX)
    os.makedirs(out_dir, exist_ok=True)

    print("Generating Delta-phi slice plot ...")
    plot_dphi_slices(
        slice_results,
        trig_pt_range=trig_range,
        save_path=f"{out_dir}/dphi_slices_{TRIG_PT_MIN:.1f}-{TRIG_PT_MAX:.1f}_fsr.png",
    )

    print("Generating HTFF plot ...")
    plot_integrated_yields(
        slice_results,
        trig_pt_range=trig_range,
        data_dir="datathief",  # directory containing STAR_near/away CSVs
        save_path=f"{out_dir}/htff_{TRIG_PT_MIN:.1f}-{TRIG_PT_MAX:.1f}_fsr.png",
    )


if __name__ == "__main__":
    main()
