"""
Dihadron Correlation Analysis

Computes the dihadron correlation function:
    1/Ntrig * dNpair/deta*dphi

The idea is simple: for each "trigger" hadron (high-pT), count how many
"associated" hadrons (lower pT) show up nearby in angle space. The result
tells you about jet structure and back-to-back correlations.

When combining multiple pTHat bins, each bin comes with its own sigmaGen
from Pythia. We rescale every pair's weight by sigmaGen/weightSum before
merging, so the final histogram reflects the physically correct
cross-section-weighted distribution rather than just counting events.

Uncertainties on the Δφ projection are estimated via block jackknife:
events are partitioned into N_JACKKNIFE_BLOCKS groups; each sample
leaves one group out, recomputes the full (ZYAM-subtracted) projection,
and the standard jackknife formula gives the standard error per bin.
"""

import glob
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

# ──────────────────────────────────────────────────────────────────────────────
# Analysis cuts — adjust these to match whatever kinematic window you want
# ──────────────────────────────────────────────────────────────────────────────
TRIG_PT_MIN = 8.0  # GeV/c  (lower edge of trigger pT window)
TRIG_PT_MAX = 15.0  # GeV/c  (upper edge of trigger pT window)
ZT_MIN = 0.9
ZT_MAX = 1.0

ETA_RANGE = (-2, 2)  # Δη binning range (also used for projection integral)
PHI_RANGE = (-np.pi / 2, 3 * np.pi / 2)  # Δφ binning range

# Number of blocks used for the block-jackknife uncertainty estimate.
N_JACKKNIFE_BLOCK = 2
N_JACKKNIFE_BLOCKS = [N_JACKKNIFE_BLOCK]


# ──────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────────
def read_dihadron_data(filename):
    """
    Read one dihadron CSV file. The file has a comment-header block
    (lines starting with #) that carries run metadata, followed by
    normal CSV rows.

    We parse the header first, then let pandas handle the data rows.
    Numeric values in the header come back as floats; range strings like
    "1.9e+01 - 2.4e+01" become a [lo, hi] list; anything else stays as a
    plain string.

    Parameters
    ----------
    filename : str
        Path to the data file.

    Returns
    -------
    data : pandas.DataFrame
    metadata : dict
    """
    metadata = {}

    with open(filename, "r") as f:
        for line in f:
            if not line.startswith("#"):
                continue
            if ":" not in line:
                continue

            key, value = line[1:].split(":", 1)
            key = key.strip()
            value = value.strip()

            try:
                metadata[key] = float(value)
                continue
            except ValueError:
                pass

            if " - " in value:
                try:
                    lo, hi = value.split(" - ", 1)
                    metadata[key] = [float(lo), float(hi)]
                    continue
                except ValueError:
                    pass

            metadata[key] = value

    data = pd.read_csv(filename, comment="#")
    return data, metadata


def read_and_combine_bins(
    filenames,
    trig_pt_min=TRIG_PT_MIN,
    trig_pt_max=TRIG_PT_MAX,
    zt_min=ZT_MIN,
    zt_max=ZT_MAX,
    eta_range=ETA_RANGE,
):
    """
    Load every pTHat bin, rescale weights to physical units, and stack
    them into one big DataFrame.

    Pythia gives each event a weight relative to its own run. To compare
    across bins with different cross-sections we multiply by:

        scale = sigmaGen_bin / weightSum_bin

    Parameters
    ----------
    filenames : list of str

    Returns
    -------
    combined_data : pandas.DataFrame
    combined_metadata : dict
    """
    frames = []
    bin_info = []
    total_ntrig = 0.0

    if not filenames:
        raise ValueError("No input files provided.")

    for idx, fname in enumerate(filenames):
        print(f"  Loading bin {idx}: {os.path.basename(fname)}")
        data, meta = read_dihadron_data(fname)

        trig_mask = (data["trigger_pT"] > trig_pt_min) & (
            data["trigger_pT"] <= trig_pt_max
        )
        assoc_mask = (data["assoc_pT"] / data["trigger_pT"] > zt_min) & (
            data["assoc_pT"] / data["trigger_pT"] <= zt_max
        )
        deta_mask = (data["assoc_eta"] - data["trigger_eta"]).abs() < abs(eta_range[1])

        data = data[trig_mask & assoc_mask & deta_mask].copy()

        sigma = meta.get("sigmaGEN", None)
        weightSum = meta.get("weightSum", None)
        trig_weight_sum = meta.get(
            f"triggerWeightSum_{trig_pt_min:e}to{trig_pt_max:e}", None
        )

        if (
            sigma is None
            or weightSum is None
            or weightSum == 0
            or trig_weight_sum is None
            or trig_weight_sum == 0
        ):
            print(
                f"File {fname} is missing 'sigmaGEN' or 'weightSum' or "
                f"weightSum = 0 or 'trig_weight_sum' or trig_weight_sum = 0. "
                f"Cannot rescale weights."
            )
            continue

        scale = sigma / weightSum
        trig_weight_sum_bin = trig_weight_sum * scale
        print(
            f"    sigmaGEN={sigma:.6e}  weightSum={weightSum:.6e}  "
            f"scale={scale:.6e}  pairs={len(data)}"
        )

        data["weight"] = data["weight"] * scale
        data["bin_index"] = idx

        frames.append(data)
        bin_info.append(
            {
                "bin_index": idx,
                "filename": fname,
                "sigmaGEN": sigma,
                "weightSum": weightSum,
                "triggerWeightSum": trig_weight_sum,
                "scale": scale,
                "n_pairs": len(data),
                "pthat_range": meta.get("PTHAT_RANGE", "unknown"),
            }
        )

        total_ntrig += trig_weight_sum_bin

        combined_metadata = {}
        combined_metadata["TRIG_PT_RANGE"] = meta.get("TRIG_PT_RANGE")
        combined_metadata["TRIG_ETA_RANGE"] = meta.get("TRIG_ETA_RANGE")
        combined_metadata["ASSOC_PT_RANGE"] = meta.get("ASSOC_PT_RANGE")
        combined_metadata["ASSOC_ETA_RANGE"] = meta.get("ASSOC_ETA_RANGE")

    combined_data = pd.concat(frames, ignore_index=True)

    combined_metadata["CUT_TRIG_PT_RANGE"] = [trig_pt_min, trig_pt_max]
    combined_metadata["CUT_ZT_RANGE"] = [zt_min, zt_max]
    combined_metadata["N_TRIG"] = total_ntrig
    combined_metadata["N_BINS"] = len(filenames)
    combined_metadata["BIN_INFO"] = bin_info

    print(f"\n  Combined: {len(filenames)} bins, {len(combined_data):,} total pairs")
    return combined_data, combined_metadata


# ──────────────────────────────────────────────────────────────────────────────
# Analysis helpers
# ──────────────────────────────────────────────────────────────────────────────
def calculate_delta_phi(phi1, phi2):
    """
    Return phi2 - phi1 wrapped into [-π, π] using arctan2,
    then shifted into [-π/2, 3π/2] to match the chosen binning range.
    """
    dphi = phi2 - phi1
    dphi = np.arctan2(np.sin(dphi), np.cos(dphi))  # wrap to (-π, π]
    # shift values below -π/2 up by 2π so the full range is (-π/2, 3π/2]
    dphi = np.where(dphi < -np.pi / 2, dphi + 2 * np.pi, dphi)
    return dphi


def _find_zyam_level(phi_proj, phi_centers, zyam_range):
    """
    Find the ZYAM background level from a 1D Δφ projection.

    Scans the distribution within ``zyam_range['phi']`` and returns
    the minimum value.

    Parameters
    ----------
    phi_proj     : 1D array
    phi_centers  : 1D array
    zyam_range   : dict  — {'phi': (lo, hi)}

    Returns
    -------
    background_level : float
    """
    phi_mask = (phi_centers >= zyam_range["phi"][0]) & (
        phi_centers <= zyam_range["phi"][1]
    )
    masked = np.where(phi_mask, phi_proj, np.inf)
    return float(np.min(masked))


def _apply_zyam(phi_proj, phi_centers, zyam_range):
    """
    Apply ZYAM subtraction to a 1D Δφ projection.

    Returns the subtracted projection and the background level.

    Parameters
    ----------
    phi_proj    : 1D array
    phi_centers : 1D array
    zyam_range  : dict  — {'phi': (lo, hi)}

    Returns
    -------
    phi_proj_zyam    : 1D array
    background_level : float
    """
    background_level = _find_zyam_level(phi_proj, phi_centers, zyam_range)
    return phi_proj - background_level, background_level


def compute_dihadron_correlation(
    data,
    metadata,
    eta_bins=40,
    phi_bins=40,
    eta_range=ETA_RANGE,
    phi_range=PHI_RANGE,
    zyam=True,
    zyam_range=None,
    ntrig_override=None,
):
    """
    Build the per-trigger-normalised 2D correlation:

        C(Δη, Δφ) = 1/Ntrig × dNpair / (dΔη dΔφ)

    ZYAM is applied at the projection level (not the 2D surface).
    The background level is always computed but only subtracted from the
    projection when zyam=True; the raw 2D surface is always returned
    unmodified so callers can build the surface plot independently.

    Parameters
    ----------
    data            : pandas.DataFrame
    metadata        : dict
    eta_bins        : int
    phi_bins        : int
    eta_range       : tuple  — (eta_min, eta_max)
    phi_range       : tuple  — (phi_min, phi_max), default (-π/2, 3π/2)
    zyam            : bool
    zyam_range      : dict  {'phi': (lo, hi)}
    ntrig_override  : float or None

    Returns
    -------
    correlation      : 2D array  — raw (no ZYAM) 1/Ntrig × d²N/dΔη dΔφ
    eta_centers      : 1D array
    phi_centers      : 1D array
    ntrig            : float
    background_level : float  — ZYAM level from the Δφ projection
    raw_counts       : 2D array
    weighted_counts  : 2D array
    """
    if zyam_range is None:
        zyam_range = {"phi": (-np.pi / 2, 3 * np.pi / 2)}
    deta = data["assoc_eta"].values - data["trigger_eta"].values
    dphi = calculate_delta_phi(data["trigger_phi"].values, data["assoc_phi"].values)
    weights = data["weight"].values

    if ntrig_override is not None:
        ntrig = ntrig_override
    else:
        group_cols = ["event", "trigger_id"]
        if "bin_index" in data.columns:
            group_cols = ["bin_index"] + group_cols
        trigger_data = data.groupby(group_cols).first()
        ntrig = trigger_data["weight"].sum()

    hist, eta_edges, phi_edges = np.histogram2d(
        deta,
        dphi,
        bins=[eta_bins, phi_bins],
        range=[eta_range, phi_range],
        weights=weights,
    )

    raw_hist, _, _ = np.histogram2d(
        deta,
        dphi,
        bins=[eta_bins, phi_bins],
        range=[eta_range, phi_range],
    )

    deta_bin = (eta_range[1] - eta_range[0]) / eta_bins
    dphi_bin = (phi_range[1] - phi_range[0]) / phi_bins
    bin_area = deta_bin * dphi_bin

    correlation = hist / (ntrig * bin_area)

    eta_centers = 0.5 * (eta_edges[:-1] + eta_edges[1:])
    phi_centers = 0.5 * (phi_edges[:-1] + phi_edges[1:])

    # Compute ZYAM level from the Δφ projection (integral over all Δη).
    # The 2D surface is never modified — only the projection is subtracted.
    phi_projection = np.sum(correlation, axis=0) * deta_bin
    background_level = _find_zyam_level(phi_projection, phi_centers, zyam_range)

    return (
        correlation,
        eta_centers,
        phi_centers,
        ntrig,
        background_level,
        raw_hist,
        hist,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Jackknife uncertainty on the Δφ projection
# ──────────────────────────────────────────────────────────────────────────────
def _ntrig_removed_by_block(block_data, trig_cols):
    """
    Sum the rescaled weights of the unique triggers present in
    ``block_data``.

    Parameters
    ----------
    block_data : pandas.DataFrame
    trig_cols  : list of str

    Returns
    -------
    float
    """
    return block_data.groupby(trig_cols)["weight"].first().sum()


def compute_phi_projection_jackknife(
    data,
    metadata,
    eta_bins=40,
    phi_bins=40,
    eta_range=ETA_RANGE,
    phi_range=PHI_RANGE,
    zyam=True,
    zyam_range=None,
    ntrig_override=None,
    n_blocks=N_JACKKNIFE_BLOCKS,
):
    """
    Estimate statistical uncertainties on the Δφ projection using a
    block-jackknife resampling scheme.

    The projection is an integral (mean × bin-count) over the full Δη
    range ``eta_range``.  No folding is applied.

    For each block k:
        1. Remove all pairs in block k.
        2. Subtract the removed triggers' weight from ntrig_override.
        3. Recompute the 2D correlation.
        4. Project onto Δφ (mean over Δη).
        5. Apply ZYAM if zyam=True.

    Standard jackknife variance:
        Var_JK[θ] = (N-1)/N × Σ_k (θ_k − θ̄_JK)²

    Parameters
    ----------
    data            : pandas.DataFrame
    metadata        : dict
    eta_bins        : int
    phi_bins        : int
    eta_range       : tuple
    phi_range       : tuple
    zyam            : bool
    zyam_range      : dict or None
    ntrig_override  : float  — must be metadata['N_TRIG']
    n_blocks        : int

    Returns
    -------
    phi_proj_central : 1D array  — Δφ projection (ZYAM-subtracted if zyam=True)
    phi_proj_err     : 1D array  — jackknife standard errors
    phi_centers      : 1D array  — bin centres in phi_range
    ntrig            : float
    background_level : float
    jackknife_projs  : 2D array  shape (n_blocks, n_phi_bins)
    """
    if zyam_range is None:
        zyam_range = {"phi": (-np.pi / 2, 3 * np.pi / 2)}
    if ntrig_override is None:
        raise ValueError(
            "ntrig_override is required for the jackknife. "
            "Pass metadata['N_TRIG'] from read_and_combine_bins."
        )

    trig_cols = (
        ["bin_index", "event", "trigger_id"]
        if "bin_index" in data.columns
        else ["event", "trigger_id"]
    )

    # ── central value ────────────────────────────────────────────────────────
    corr_full, _, phi_centers, ntrig, _, _, _ = compute_dihadron_correlation(
        data,
        metadata,
        eta_bins=eta_bins,
        phi_bins=phi_bins,
        eta_range=eta_range,
        phi_range=phi_range,
        zyam=False,
        ntrig_override=ntrig_override,
    )

    deta_bin = (eta_range[1] - eta_range[0]) / eta_bins

    phi_proj_full = np.sum(corr_full, axis=0) * deta_bin

    background_level = _find_zyam_level(phi_proj_full, phi_centers, zyam_range)
    phi_proj_central = (phi_proj_full - background_level) if zyam else phi_proj_full

    # ── assign block labels ──────────────────────────────────────────────────
    event_cols = ["bin_index", "event"] if "bin_index" in data.columns else ["event"]
    unique_events = data[event_cols].drop_duplicates().reset_index(drop=True)
    n_events = len(unique_events)

    if n_blocks > n_events:
        print(
            f"  Warning: n_blocks ({n_blocks}) > n_events ({n_events}). "
            f"Falling back to leave-one-event-out jackknife."
        )
        n_blocks = n_events

    unique_events = unique_events.copy()
    unique_events["_jk_block"] = np.arange(n_events) % n_blocks
    data_jk = data.merge(unique_events, on=event_cols, how="left")

    # ── jackknife loop ───────────────────────────────────────────────────────
    print(f"\nRunning block jackknife ({n_blocks} blocks) …")
    jackknife_projs = np.empty((n_blocks, len(phi_proj_central)))

    for k in range(n_blocks):
        removed = data_jk[data_jk["_jk_block"] == k]
        data_k = data_jk[data_jk["_jk_block"] != k].copy()

        ntrig_k = ntrig_override - _ntrig_removed_by_block(removed, trig_cols)

        corr_k, _, _, _, _, _, _ = compute_dihadron_correlation(
            data_k,
            metadata,
            eta_bins=eta_bins,
            phi_bins=phi_bins,
            eta_range=eta_range,
            phi_range=phi_range,
            zyam=False,
            ntrig_override=ntrig_k,
        )

        phi_proj_k = np.sum(corr_k, axis=0) * deta_bin

        if zyam:
            phi_proj_k, _ = _apply_zyam(phi_proj_k, phi_centers, zyam_range)

        jackknife_projs[k] = phi_proj_k

        if (k + 1) % max(1, n_blocks // 10) == 0:
            print(f"  Block {k + 1}/{n_blocks} done")

    # ── standard jackknife variance ──────────────────────────────────────────
    jk_mean = np.mean(jackknife_projs, axis=0)
    jk_var = (
        (n_blocks - 1) / n_blocks * np.sum((jackknife_projs - jk_mean) ** 2, axis=0)
    )
    phi_proj_err = np.sqrt(jk_var)

    print(
        f"  Jackknife done.  Max error: {phi_proj_err.max():.4e}  "
        f"Median error: {np.median(phi_proj_err):.4e}  "
        f"Mean error: {np.mean(phi_proj_err):.4e}  "
        f"First bin error: {phi_proj_err[0]:.4e}  "
        f"Last bin error: {phi_proj_err[-1]:.4e}"
    )

    rel_err = phi_proj_err / np.abs(phi_proj_central)
    print(
        f"Max relative error: {rel_err.max():.2%}  "
        f"Median relative error: {np.median(rel_err):.2%}  "
        f"Mean relative error: {np.mean(rel_err):.2%}"
    )

    delta_phi = phi_centers[1] - phi_centers[0]
    yield_central = phi_proj_central.sum() * delta_phi
    yield_k = jackknife_projs.sum(axis=1) * delta_phi
    yield_jk_mean = yield_k.mean()
    yield_jk_var = (n_blocks - 1) / n_blocks * np.sum((yield_k - yield_jk_mean) ** 2)
    yield_err = np.sqrt(yield_jk_var)

    print(
        f"\nIntegrated yield:  {yield_central:.4e} ± {yield_err:.4e}  "
        f"(rel. err: {yield_err / abs(yield_central):.2%})"
    )

    return (
        phi_proj_central,
        phi_proj_err,
        phi_centers,
        ntrig,
        background_level,
        jackknife_projs,
    )


def run_jackknife_multiple_blocks(
    data,
    metadata,
    block_sizes,
    eta_bins=40,
    phi_bins=40,
    eta_range=ETA_RANGE,
    phi_range=PHI_RANGE,
    zyam=True,
    zyam_range=None,
    ntrig_override=None,
):
    """
    Run the jackknife multiple times with different block sizes.

    Returns
    -------
    results : dict   keys = block size
    """
    results = {}

    for n_blocks in block_sizes:
        print(f"\n=== Running jackknife with {n_blocks} blocks ===")

        (phi_proj_central, phi_proj_err, phi_centers, ntrig, bkg, jackknife_projs) = (
            compute_phi_projection_jackknife(
                data,
                metadata,
                eta_bins=eta_bins,
                phi_bins=phi_bins,
                eta_range=eta_range,
                phi_range=phi_range,
                zyam=zyam,
                zyam_range=zyam_range,
                ntrig_override=ntrig_override,
                n_blocks=n_blocks,
            )
        )

        results[n_blocks] = {
            "phi_proj_central": phi_proj_central,
            "phi_proj_err": phi_proj_err,
            "phi_centers": phi_centers,
            "ntrig": ntrig,
            "background": bkg,
            "jackknife_projs": jackknife_projs,
        }

    return results


# ──────────────────────────────────────────────────────────────────────────────
# CSV export
# ──────────────────────────────────────────────────────────────────────────────
def save_projections_to_csv(
    phi_proj_central,
    phi_proj_err,
    phi_centers,
    correlation,
    eta_centers,
    phi_centers_2d,
    background_level,
    metadata,
    save_dir="plots/Correlations",
    tag="",
):
    """
    Write the Δφ and Δη projections to separate CSV files.

    Δφ file columns
    ---------------
    delta_phi_rad       : bin centre in (-π/2, 3π/2)  (radians)
    dNpair_dDeltaPhi    : 1/Ntrig × dNpair/dΔφ  (ZYAM-subtracted if applied)
    stat_err_jackknife  : jackknife standard error (0 if not computed)
    zyam_background     : constant ZYAM level subtracted from every bin

    Δη file columns
    ---------------
    delta_eta           : bin centre in eta_range
    dNpair_dDeltaEta    : 1/Ntrig × dNpair/dΔη  (averaged over all Δφ)

    Parameters
    ----------
    phi_proj_central  : 1D array
    phi_proj_err      : 1D array or None
    phi_centers       : 1D array   — centres of Δφ bins (unfolded)
    correlation       : 2D array   — C(Δη, Δφ)
    eta_centers       : 1D array
    phi_centers_2d    : 1D array   — centres from the 2D histogram (may equal phi_centers)
    background_level  : float
    metadata          : dict
    save_dir          : str
    tag               : str

    Returns
    -------
    dphi_path : str
    deta_path : str
    """
    os.makedirs(save_dir, exist_ok=True)

    trig_lo, trig_hi = metadata.get("CUT_TRIG_PT_RANGE", [TRIG_PT_MIN, TRIG_PT_MAX])
    zt_lo, zt_hi = metadata.get("CUT_ZT_RANGE", [ZT_MIN, ZT_MAX])

    base = f"trig{trig_lo:.1f}-{trig_hi:.1f}_zT{zt_lo:.1f}-{zt_hi:.1f}"
    if tag:
        base = f"{base}_{tag}"

    # ── Δφ projection ─────────────────────────────────────────────────────────
    dphi_df = pd.DataFrame(
        {
            "delta_phi_rad": phi_centers,
            "dNpair_dDeltaPhi": phi_proj_central,
            "stat_err_jackknife": (
                phi_proj_err
                if phi_proj_err is not None
                else np.zeros_like(phi_proj_central)
            ),
            "zyam_background": background_level,
        }
    )

    dphi_path = os.path.join(save_dir, f"dphi_projection_{base}.csv")
    header_lines = [
        "# Dihadron Δφ projection",
        f"# trigger pT : {trig_lo:.1f} - {trig_hi:.1f} GeV/c",
        f"# zT : {zt_lo:.1f} - {zt_hi:.1f} GeV/c",
        f"# Δη integral: {ETA_RANGE[0]} to {ETA_RANGE[1]}",
        "# Δφ range   : -π/2 to 3π/2",
        f"# ZYAM level : {background_level:.6e}",
        f"# ntrig      : {metadata.get('N_TRIG', float('nan')):.6e}",
        f"# jackknife blocks : {metadata.get('jackknife_n_blocks', 'not run')}",
        "#",
        "# delta_phi_rad      : bin centre in (-π/2, 3π/2) [rad]",
        "# dNpair_dDeltaPhi   : 1/Ntrig x dNpair/dΔφ  (ZYAM-subtracted)",
        "# stat_err_jackknife : jackknife standard error",
        "# zyam_background    : constant subtracted from every bin",
    ]

    with open(dphi_path, "w") as f:
        f.write("\n".join(header_lines) + "\n")
        dphi_df.to_csv(f, index=False)

    print(f"Δφ projection saved → {dphi_path}  ({len(dphi_df)} bins)")

    # ── Δη projection ─────────────────────────────────────────────────────────
    dphi_bin = phi_centers[1] - phi_centers[0]
    eta_proj = np.sum(correlation, axis=1) * dphi_bin

    deta_df = pd.DataFrame(
        {
            "delta_eta": eta_centers,
            "dNpair_dDeltaEta": eta_proj,
        }
    )

    deta_path = os.path.join(save_dir, f"deta_projection_{base}.csv")
    deta_header = [
        "# Dihadron Δη projection",
        f"# trigger pT : {trig_lo:.1f} - {trig_hi:.1f} GeV/c",
        f"# zT : {zt_lo:.1f} - {zt_hi:.1f} GeV/c",
        "# averaged over all Δφ bins",
        f"# ntrig      : {metadata.get('N_TRIG', float('nan')):.6e}",
        "#",
        "# delta_eta        : bin centre in Δη range",
        "# dNpair_dDeltaEta : 1/Ntrig x dNpair/dΔη",
    ]

    with open(deta_path, "w") as f:
        f.write("\n".join(deta_header) + "\n")
        deta_df.to_csv(f, index=False)

    print(f"Δη projection saved → {deta_path}  ({len(deta_df)} bins)")

    return dphi_path, deta_path


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────
def plot_correlation_with_projections(
    correlation,
    eta_centers,
    phi_centers,
    background_level=0.0,
    phi_proj_central=None,
    phi_proj_err=None,
    metadata=None,
    save_path=None,
):
    """
    Main physics figure: 3D correlation surface on the left, and on the
    right (top to bottom):
        1. Δφ projection with jackknife error bars, compared against STAR/ALICE data
        2. STAR(ALICE) / Pythia ratio
        3. Δη projection

    Parameters
    ----------
    correlation        : 2D array  — raw C(Δη, Δφ)
    eta_centers        : 1D array
    phi_centers        : 1D array  — bin centres in (-π/2, 3π/2)
    background_level   : float     — ZYAM constant; drawn as a dashed line
    phi_proj_central   : 1D array or None  — ZYAM-subtracted projection
    phi_proj_err       : 1D array or None
    metadata           : dict or None
    save_path          : str or None
    """
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, width_ratios=[3, 0.5, 0.5], wspace=0.4, hspace=0.5)

    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_phi = fig.add_subplot(gs[0, 1:])
    ax_rat = fig.add_subplot(gs[1, 1:])
    ax_eta = fig.add_subplot(gs[2, 1:])

    blue = plt.get_cmap("Blues")(0.7)
    orange = plt.get_cmap("Oranges")(0.7)
    red = plt.get_cmap("Reds")(0.7)
    green = plt.get_cmap("Greens")(0.7)

    d_phi = phi_centers[1] - phi_centers[0] if len(phi_centers) > 1 else 0.1

    # ── 3D surface ────────────────────────────────────────────────────────────
    Z = np.clip(correlation, 0, 0.4)
    max_height = Z.max() if Z.max() > 0 else 1.0

    ETA, PHI = np.meshgrid(eta_centers, phi_centers, indexing="ij")

    n_colors = 20
    levels = np.linspace(0, max_height, n_colors + 1)
    colourticks = np.linspace(0, max_height, 5)  # bin edges

    cmap = plt.cm.get_cmap("turbo", n_colors)
    norm = mpl.colors.BoundaryNorm(levels, cmap.N)

    colors = cmap(norm(Z))

    ax3d.plot_surface(
        ETA,
        PHI,
        Z,
        facecolors=colors,
        shade=True,
        alpha=0.9,
        linewidth=0,
        antialiased=True,
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax3d, shrink=0.5, aspect=10, pad=0.15, ticks=colourticks)
    cbar.ax.tick_params(labelsize=12)

    ax3d.set_zlim(0, max_height * 1.05)
    ax3d.set_xlabel(r"$\Delta\eta$", fontsize=11, labelpad=8)
    ax3d.set_ylabel(r"$\Delta\phi$ [rad]", fontsize=11, labelpad=8)
    ax3d.zaxis.set_rotate_label(False)
    ax3d.text2D(
        -0.08,
        0.5,  # x, y in axes-fraction coords; nudge x left/right to taste
        r"$\frac{1}{N_{\rm trig}} \frac{d^2N_{\rm pair}}{d\Delta\eta\, d\Delta\phi}$",
        transform=ax3d.transAxes,
        fontsize=17,
        ha="center",
        va="center",
        rotation=90,
    )
    ax3d.tick_params(axis="both", labelsize=12)
    ax3d.tick_params(axis="z", labelsize=12)
    ax3d.set_yticks([-np.pi / 2, 0, np.pi / 2, np.pi, 3 * np.pi / 2])
    ax3d.set_yticklabels(
        [r"$-\frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$"]
    )
    ax3d.view_init(elev=25, azim=45)

    # ── Δφ projection ─────────────────────────────────────────────────────────
    if phi_proj_central is not None:
        pf_central = phi_proj_central
        pf_err = phi_proj_err
    else:
        dphi_bin = phi_centers[1] - phi_centers[0]
        pf_central = np.sum(correlation, axis=0) * dphi_bin
        pf_central, _ = _apply_zyam(pf_central, phi_centers, {"phi": (0.5, 1.5)})
        pf_err = None

    phi_ticks = [-np.pi / 2, 0, np.pi / 2, np.pi, 3 * np.pi / 2]
    phi_labels = [r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$"]

    if pf_err is not None:
        ax_phi.errorbar(
            phi_centers,
            pf_central,
            yerr=pf_err,
            fmt="o",
            color=blue,
            ms=4,
            capsize=3,
            capthick=1,
            elinewidth=1,
            label="Pythia (JK errors)",
        )
    else:
        ax_phi.plot(phi_centers, pf_central, "o", color=blue, ms=4, label="Pythia")

    ax_phi.axhline(
        y=background_level,
        color=red,
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=f"ZYAM min: {background_level:.4f}",
    )

    # ── try to load STAR/ALICE data ──────────────────────────────────────────────────
    exp_data_loaded = False
    exp_phi = None
    exp_y = None

    if metadata:
        trig_range = metadata.get("CUT_TRIG_PT_RANGE") or metadata.get("TRIG_PT_RANGE")
        zt_range = metadata.get("CUT_ZT_RANGE")

        if isinstance(trig_range, list) and isinstance(zt_range, list):
            exp_path = (
                f"datathief/STAR_{trig_range[0]:.0f}-{trig_range[1]:.0f}"
                f"_zT{zt_range[0]:.2f}-{zt_range[1]:.2f}"
            )
            for ext in (".csv", ".txt"):
                candidate = exp_path + ext
                if os.path.exists(candidate):
                    try:
                        exp_df = pd.read_csv(candidate)
                        exp_phi = exp_df["DeltaPhi"].values
                        exp_y = exp_df["d2Npair"].values

                        # Asymmetric statistical uncertainties.
                        # 'stat -' is the magnitude of the downward error (positive number),
                        # 'stat +' is the magnitude of the upward error.
                        exp_err_lo = (
                            np.abs(exp_df["stat -"].values)
                            if "stat -" in exp_df.columns
                            else np.zeros_like(exp_y)
                        )
                        exp_err_hi = (
                            exp_df["stat +"].values
                            if "stat +" in exp_df.columns
                            else np.zeros_like(exp_y)
                        )

                        sort_idx = np.argsort(exp_phi)
                        exp_phi, exp_y = exp_phi[sort_idx], exp_y[sort_idx]
                        exp_err_lo, exp_err_hi = (
                            exp_err_lo[sort_idx],
                            exp_err_hi[sort_idx],
                        )

                        has_exp_errs = exp_err_lo.any() or exp_err_hi.any()

                        if has_exp_errs:
                            ax_phi.errorbar(
                                exp_phi,
                                exp_y,
                                yerr=[exp_err_lo, exp_err_hi],
                                fmt="s",
                                color=green,
                                ms=4,
                                markerfacecolor="none",
                                markeredgewidth=1.2,
                                capsize=3,
                                capthick=1,
                                elinewidth=1,
                                label="STAR data",
                            )
                        else:
                            ax_phi.plot(
                                exp_phi,
                                exp_y,
                                "s",
                                color=green,
                                ms=4,
                                markerfacecolor="none",
                                markeredgewidth=1.2,
                                label="STAR data",
                            )
                        exp_data_loaded = True
                    except Exception as e:  # noqa: BLE001
                        print(
                            f"Warning: could not load ALICE/STAR data from {candidate}: {e}"
                        )
                    break

            if not exp_data_loaded:
                print(
                    f"Warning: no ALICE/STAR data file found at {exp_path}(.csv/.txt)"
                )

    ax_phi.legend(fontsize=8)
    ax_phi.set_xlabel(r"$\Delta\phi$ [rad]", fontsize=10)
    ax_phi.set_ylabel(
        r"$\frac{1}{N_{\rm trig}} \frac{dN_{\rm pair}}{d\Delta\phi}$", fontsize=10
    )
    ax_phi.set_title(
        rf"$\Delta\phi$ projection  ($\int_{{ {ETA_RANGE[0]} }}^{{ {ETA_RANGE[1]} }} d\Delta\eta$)",
        fontsize=10,
    )
    ax_phi.set_xlim(PHI_RANGE[0] - 0.1, PHI_RANGE[1] + 0.1)
    ax_phi.set_xticks(phi_ticks)
    ax_phi.set_xticklabels(phi_labels, fontsize=8)
    ax_phi.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax_phi.grid(True, alpha=0.3)

    # ── STAR(ALICE) / Pythia ratio ────────────────────────────────────────────────────
    if exp_data_loaded and exp_phi is not None:
        phi_dict = dict(zip(phi_centers, pf_central))
        err_dict = dict(zip(phi_centers, pf_err)) if pf_err is not None else {}

        ratio_phi = []
        ratio_vals = []
        ratio_errs = []

        exp_err_lo_dict = dict(zip(exp_phi, exp_err_lo))
        exp_err_hi_dict = dict(zip(exp_phi, exp_err_hi))

        for cp, cy in zip(exp_phi, exp_y):
            closest = min(phi_centers, key=lambda x: abs(x - cp))
            py_val = phi_dict[closest]
            if abs(closest - cp) <= d_phi / 2.0 and py_val != 0:
                ratio = cy / py_val
                ratio_phi.append(cp)
                ratio_vals.append(ratio)

                # Propagate both STAR/ALICE stat error and Pythia jackknife error.
                # For an asymmetric STAR/ALICE error we use the larger of the two
                # sides as a symmetric approximation in the ratio.
                exp_rel = (
                    max(
                        exp_err_lo_dict.get(cp, 0.0),
                        exp_err_hi_dict.get(cp, 0.0),
                    )
                    / abs(cy)
                    if cy != 0
                    else 0.0
                )
                py_rel = (
                    err_dict[closest] / abs(py_val)
                    if pf_err is not None and py_val != 0
                    else 0.0
                )
                ratio_errs.append(abs(ratio) * np.hypot(exp_rel, py_rel))

        if ratio_phi:
            ratio_phi = np.array(ratio_phi)
            ratio_vals = np.array(ratio_vals)
            ratio_errs = np.array(ratio_errs)

            if pf_err is not None and ratio_errs.any():
                ax_rat.errorbar(
                    ratio_phi,
                    ratio_vals,
                    yerr=ratio_errs,
                    fmt="D",
                    color="purple",
                    ms=4,
                    markerfacecolor="none",
                    markeredgewidth=1.2,
                    capsize=3,
                    capthick=1,
                    elinewidth=1,
                    label="STAR / Pythia",
                )
            else:
                ax_rat.plot(
                    ratio_phi,
                    ratio_vals,
                    "D",
                    color="purple",
                    ms=4,
                    markerfacecolor="none",
                    markeredgewidth=1.2,
                    label="STAR / Pythia",
                )
        else:
            ax_rat.text(
                0.5,
                0.5,
                "No matching bins found\n(STAR & Pythia φ grids differ)",
                ha="center",
                va="center",
                transform=ax_rat.transAxes,
                fontsize=9,
                color="gray",
            )
    else:
        ax_rat.text(
            0.5,
            0.5,
            "STAR data not loaded",
            ha="center",
            va="center",
            transform=ax_rat.transAxes,
            fontsize=9,
            color="gray",
        )

    ax_rat.axhline(y=1.0, color="k", linestyle="--", linewidth=1.0, alpha=0.5)
    ax_rat.set_xlabel(r"$\Delta\phi$ [rad]", fontsize=10)
    ax_rat.set_ylabel(r"STAR Pythia / Pythia", fontsize=10)
    ax_rat.set_title(r"Ratio STAR Pythia / Pythia", fontsize=10)
    ax_rat.set_xlim(PHI_RANGE[0] - 0.1, PHI_RANGE[1] + 0.1)
    ax_rat.set_ylim(0.5, 1.5)
    ax_rat.set_xticks(phi_ticks)
    ax_rat.set_xticklabels(phi_labels, fontsize=8)
    ax_rat.legend(fontsize=8)
    ax_rat.grid(True, alpha=0.3)

    # ── Δη projection ─────────────────────────────────────────────────────────
    deta_bin = eta_centers[1] - eta_centers[0]
    eta_proj = np.sum(correlation, axis=1) * deta_bin  # average over all Δφ

    ax_eta.plot(eta_centers, eta_proj, "o", color=orange, ms=4, label="Pythia")
    ax_eta.axhline(
        y=background_level,
        color=red,
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=f"ZYAM min: {background_level:.4f}",
    )
    ax_eta.legend(fontsize=8)
    ax_eta.set_xlabel(r"$\Delta\eta$", fontsize=10)
    ax_eta.set_ylabel(
        r"$\frac{1}{N_{\rm trig}} \frac{dN_{\rm pair}}{d\Delta\eta}$", fontsize=10
    )
    ax_eta.set_title(r"$\Delta\eta$ projection  (all $\Delta\phi$)", fontsize=10)
    ax_eta.set_xlim(ETA_RANGE[0] - 0.05, ETA_RANGE[1] + 0.05)
    ax_eta.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax_eta.grid(True, alpha=0.3)

    # ── figure-level title ────────────────────────────────────────────────────
    suptitle = "No FSR Dihadron Correlation"
    if metadata:
        parts = []
        trig_range = metadata.get("CUT_TRIG_PT_RANGE") or metadata.get("TRIG_PT_RANGE")
        zt_range = metadata.get("CUT_ZT_RANGE")
        if isinstance(trig_range, list):
            parts.append(
                f"Trigger $p_T$: {trig_range[0]:.1f}-{trig_range[1]:.1f} GeV/c"
            )
        if isinstance(zt_range, list):
            parts.append(rf"$z_T$: {zt_range[0]:.2f}-{zt_range[1]:.2f}")
            parts.append(r"$\eta$ < 1")
        n_blocks_used = metadata.get("jackknife_n_blocks")
        if n_blocks_used:
            parts.append(f"JK blocks: {n_blocks_used}")
        if metadata.get("N_BINS"):
            parts.append(f"{metadata['N_BINS']} pTHat bins combined")
        if parts:
            suptitle += "\n" + "   |   ".join(parts)

    fig.suptitle(suptitle, fontsize=13, fontweight="bold")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    return fig, (ax3d, ax_phi, ax_rat, ax_eta)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    POW = 3
    BASE = "pythiaData/2760/test/fsr"

    filenames = sorted(glob.glob(f"{BASE}/dihadron_pow{POW}_pT*.csv"))

    if not filenames:
        raise FileNotFoundError(
            f"No bin files found matching {BASE}/dihadron_pow{POW}_pT*.csv"
        )

    print(f"Found {len(filenames)} pTHat bin file(s):")
    for f in filenames:
        print(f"  {f}")

    # ------------------------------------------------------------------
    # Load and merge all pTHat bins
    # ------------------------------------------------------------------
    print("\nLoading and combining bins...")
    data, metadata = read_and_combine_bins(filenames)

    print("\nCombined metadata:")
    for key, value in metadata.items():
        if key != "BIN_INFO":
            print(f"  {key}: {value}")

    ntrig_override = metadata.get("N_TRIG")

    # ------------------------------------------------------------------
    # Full 2D correlation (for the 3D surface and Δη projection)
    # ------------------------------------------------------------------
    print("\nComputing 2D dihadron correlation...")
    (correlation, eta_centers, phi_centers, ntrig, bkg, _, _) = (
        compute_dihadron_correlation(
            data,
            metadata,
            eta_bins=36,
            phi_bins=36,
            eta_range=ETA_RANGE,
            phi_range=PHI_RANGE,
            zyam=True,
            zyam_range={"phi": (-np.pi / 2, 3 * np.pi / 2)},
            ntrig_override=ntrig_override,
        )
    )

    print(f"Number of triggers (weighted): {ntrig:.4e}")
    print(f"Max correlation value:         {np.max(correlation):.6f}")
    print(f"Min correlation value:         {np.min(correlation):.6f}")
    print(f"Background level (ZYAM):       {bkg:.6f}")

    # weight-sanity diagnostics
    top_weights = data.nlargest(10, "weight")[
        [
            "event",
            "bin_index",
            "weight",
            "trigger_phi",
            "assoc_phi",
            "trigger_eta",
            "assoc_eta",
        ]
    ]
    print("\nTop-10 pairs by (rescaled) weight:")
    print(top_weights.to_string(index=False))
    print("\nWeight stats:")
    print(
        f"  max/mean ratio:         {data['weight'].max() / data['weight'].mean():.1f}x"
    )
    print(
        f"  top-10 weight fraction: "
        f"{data.nlargest(10, 'weight')['weight'].sum() / data['weight'].sum():.1%}"
    )

    # ------------------------------------------------------------------
    # Jackknife uncertainty on the Δφ projection
    # ------------------------------------------------------------------
    print(
        f"\nComputing jackknife uncertainties "
        f"({N_JACKKNIFE_BLOCKS} blocks) on Δφ projection..."
    )

    jk_results = run_jackknife_multiple_blocks(
        data,
        metadata,
        block_sizes=N_JACKKNIFE_BLOCKS,
        eta_bins=36,
        phi_bins=36,
        eta_range=ETA_RANGE,
        phi_range=PHI_RANGE,
        zyam=True,
        zyam_range={"phi": (-np.pi / 2, 3 * np.pi / 2)},
        ntrig_override=ntrig_override,
    )

    phi_proj_central = jk_results[N_JACKKNIFE_BLOCK]["phi_proj_central"]
    phi_proj_err = jk_results[N_JACKKNIFE_BLOCK]["phi_proj_err"]
    phi_centers_jk = jk_results[N_JACKKNIFE_BLOCK]["phi_centers"]

    metadata["jackknife_n_blocks"] = N_JACKKNIFE_BLOCKS

    # ------------------------------------------------------------------
    # Save projections to CSV
    # ------------------------------------------------------------------
    print("\nSaving projections to CSV...")
    out_dir = "plots/correlations/test/fsr"
    _, _ = save_projections_to_csv(
        phi_proj_central=phi_proj_central,
        phi_proj_err=phi_proj_err,
        phi_centers=phi_centers_jk,
        correlation=correlation,
        eta_centers=eta_centers,
        phi_centers_2d=phi_centers,
        background_level=bkg,
        metadata=metadata,
        save_dir=out_dir,
        tag=f"pow{POW}",
    )

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)

    plot_correlation_with_projections(
        correlation,
        eta_centers,
        phi_centers,
        background_level=bkg,
        phi_proj_central=phi_proj_central,
        phi_proj_err=phi_proj_err,
        metadata=metadata,
        save_path=(
            f"{out_dir}/dihadron_2d_combined_and_projections"
            f"_{TRIG_PT_MIN:.1f}-{TRIG_PT_MAX:.1f}"
            f"_zT{ZT_MIN:.2f}-{ZT_MAX:.2f}.png"
        ),
    )

    plt.show()


if __name__ == "__main__":
    main()
