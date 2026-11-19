"""Single-cell analysis (DIA-NN vs JUMPlion) for datasets with no ground truth.

Shared by make_singlecell_hela.py and make_singlecell_pbmc.py; each dataset has its own config
(configs/singlecell_<name>.json) and is run on its own -- it does not go through
make_figures.py or the differential_expression stage, it reads the precursor matrices directly.

Input per dataset (the "singlecell" config section; one pass per entry of "levels": precursor and protein):
  * levels.<level>.matrices.diann / .jumplion: matrix TSVs (id column + annotation columns + one
    column per cell, raw intensities; 0/NaN = not quantified)
  * sample_meta_file: CSV with a "Run" column (matching the matrix cell columns) and, optionally, a
    grouping column (group_col; e.g. cell type for PBMC) used to colour the PCA/UMAP

Output: <output_dir>/singlecell_<label>/{main,supp,table}. Protein-level panels are Fig_ (main) and
precursor-level panels Supp_ (supp); routing is fixed in the code, not configurable.
  A  IDs per cell                    B  completeness (detected in >= x% of cells)
  C  intensity density, DIA-NN reported vs JUMPlion recovered (supp, protein level)
  D  paired CV (DIA-NN vs JUMPlion)  E  CV by intensity (supp, protein level)
  F  cell-to-cell correlation        G  PCA of cells (supp; when "embedding" is true)
  Integrated  HeLa: 2x2 of A, B, D, F  (Fig_SingleCell_<label>_Integrated)
Panels listed in the config's "skip_analyses" are not drawn (PBMC skips the CV and correlation panels).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

from . import style as S
from .performance import PANEL_LABEL_SIZE as FS, compact_axes
from .common import (OutputLayout, PathResolver, read_raw_config, setup_logging,
                     substitute_config, write_tsv)

plt = S.plt
log = logging.getLogger("jumplion_figures.singlecell")

METHOD_KEYS = [("diann", S.METHOD_ORDER[0]), ("jumplion", S.METHOD_ORDER[1])]
PANEL = (2.4, 2.0)
INT_GAP = 0.12  # inches between panels in the integrated figure
INT_WIDTH = 4.0  # integrated figure width (in); its 2 x 2 panels are drawn at this size, not scaled down
INT_HEIGHT = 3.6  # integrated figure height (in); two rows of equal height
PANEL_INT = ((INT_WIDTH - INT_GAP) / 2, INT_HEIGHT / 2)
RECOVERED_GREEN = "#59A14F"  # same green as the mechanism figures (mechanism.RECOVERED_GREEN)


def _align(fig, bottom=.24, right=.97):
    """Same axes rectangle on every figure-level panel (saved with tight=False), so the panels line up when tiled."""
    fig.subplots_adjust(left=.27, right=right, bottom=bottom, top=.92)


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_matrix(path, meta, id_col):
    """-> (log2 intensity DataFrame [precursor x cell], annotation DataFrame). Non-positive/NaN -> NaN."""
    df = pd.read_csv(path, sep="\t", low_memory=False)
    cells = [c for c in df.columns if c in set(meta["Run"].astype(str))]
    if not cells:
        raise ValueError(f"{path}: no matrix column matches sample_meta 'Run'.")
    df = df.set_index(id_col)
    vals = df[cells].apply(pd.to_numeric, errors="coerce")
    vals = vals.where(vals > 0)
    annot = df.drop(columns=cells)
    return np.log2(vals), annot


# ----------------------------------------------------------------------------
# Computations
# ----------------------------------------------------------------------------
def ids_per_cell(mats, annots, protein_col, level="precursor"):
    """Precursor level: precursors per cell, plus the unique protein groups behind them. Protein
    level: the matrix rows are already protein groups, so only protein_groups is filled."""
    rows = []
    for key, method in METHOD_KEYS:
        m = mats[key]
        prot = annots[key][protein_col] if protein_col in annots[key] else None
        for cell in m.columns:
            ok = m[cell].notna()
            if level == "protein":
                rows.append(dict(method=method, cell=cell, precursors=np.nan, protein_groups=int(ok.sum())))
                continue
            n_prot = prot[ok].nunique() if prot is not None else np.nan
            rows.append(dict(method=method, cell=cell, precursors=int(ok.sum()), protein_groups=n_prot))
    return pd.DataFrame(rows)


PREVALENCE_PCTS = (25, 50, 75, 90, 100)


def min_cells_for(pct, n_cells):
    """Integer cell count for a prevalence threshold: ceil(pct% of n_cells) (72 cells -> 18, 36, 54, 65, 72).
    The product is rounded to 9 places first so float noise (e.g. 0.29 * 100) can't tip the ceil."""
    return int(np.ceil(round(pct / 100 * n_cells, 9)))


def completeness_curve(mats):
    """Entities detected in >= k cells, for every integer k = 1..n_cells (x also given as % of cells)."""
    rows = []
    for key, method in METHOD_KEYS:
        n_det = mats[key].notna().sum(axis=1).to_numpy()
        n_cells = mats[key].shape[1]
        for k in range(1, n_cells + 1):
            rows.append(dict(method=method, min_cells=k, min_pct_cells=k / n_cells * 100,
                             n_features=int((n_det >= k).sum())))
    return pd.DataFrame(rows)


def prevalence_table(mats):
    """Exact counts at PREVALENCE_PCTS (integer cell thresholds), per method, plus JUMPlion - DIA-NN
    absolute and relative (% of the DIA-NN count) differences."""
    n_cells = mats["diann"].shape[1]
    d_name, j_name = S.METHOD_ORDER
    det = {m: mats[k].notna().sum(axis=1).to_numpy() for k, m in METHOD_KEYS}
    rows = []
    for pct in PREVALENCE_PCTS:
        k = min_cells_for(pct, n_cells)
        n_d, n_j = int((det[d_name] >= k).sum()), int((det[j_name] >= k).sum())
        rows.append(dict(prevalence_pct=pct, min_cells=k, n_cells=n_cells, n_diann=n_d, n_jumplion=n_j,
                         abs_diff=n_j - n_d, rel_diff_pct=(n_j - n_d) / n_d * 100 if n_d else np.nan))
    return pd.DataFrame(rows)


def cv_by_intensity(mats, min_cells, n_bins=4):
    """Per-feature CV (%) across cells on the linear scale (each method's own CV), binned by the feature's mean
    JUMPlion log2 intensity quartile: the same feature falls in the same bin for both methods, and the bins are
    quartiles of the JUMPlion means of the features present in both matrices."""
    j_key = METHOD_KEYS[1][0]
    common = mats[METHOD_KEYS[0][0]].index.intersection(mats[j_key].index)
    jl_mean = mats[j_key].loc[common].mean(axis=1)
    per = {}
    for key, method in METHOD_KEYS:
        lin = 2.0 ** mats[key].loc[common]
        n = lin.notna().sum(axis=1)
        cv = (lin.std(axis=1, ddof=1) / lin.mean(axis=1) * 100)[n >= min_cells]
        per[method] = pd.DataFrame({"CV": cv, "mean_log2": jl_mean[cv.index]})
    edges = np.quantile(jl_mean.dropna(), np.linspace(0, 1, n_bins + 1))
    labels = [f"{lo:.1f}–{hi:.1f}" for lo, hi in zip(edges[:-1], edges[1:])]  # mean log2 intensity range per bin
    edges[0], edges[-1] = -np.inf, np.inf
    out = []
    for method, d in per.items():
        d = d.dropna().copy()
        d["bin"] = pd.cut(d["mean_log2"], edges, labels=labels)
        d["method"] = method
        out.append(d.reset_index())
    return pd.concat(out, ignore_index=True), labels


def _cv(lin, min_cells):
    n = lin.notna().sum(axis=1)
    return (lin.std(axis=1, ddof=1) / lin.mean(axis=1) * 100).where(n >= min_cells)


def paired_cv(mats, min_cells, n_bins=4):
    """Protein-wise CV comparison on the SAME features: those quantified in >= min_cells cells in
    both methods, each CV computed over all the cells that method quantifies the feature in.
    Abundance bin = quartile of the feature's mean DIA-NN log2 intensity."""
    common = mats["diann"].index.intersection(mats["jumplion"].index)
    d = 2.0 ** mats["diann"].loc[common]
    j = 2.0 ** mats["jumplion"].loc[common]
    out = pd.DataFrame({"cv_diann": _cv(d, min_cells), "cv_jumplion": _cv(j, min_cells),
                        "mean_log2": mats["diann"].loc[common].mean(axis=1),
                        "n_cells_diann": d.notna().sum(axis=1), "n_cells_jumplion": j.notna().sum(axis=1)})
    out = out.dropna(subset=["cv_diann", "cv_jumplion"])
    labels = [f"Q{i + 1}" for i in range(n_bins)]
    out["bin"] = pd.qcut(out["mean_log2"], n_bins, labels=labels)
    out["delta_cv"] = out["cv_jumplion"] - out["cv_diann"]
    return out.rename_axis("feature").reset_index(), labels


def paired_cv_summary(p):
    rows = []
    for name, g in [("all", p)] + [(str(b), g) for b, g in p.groupby("bin", observed=True)]:
        dc = g.delta_cv
        rows.append(dict(stratum=name, n_features=len(g),
                         median_cv_diann=g.cv_diann.median(), median_cv_jumplion=g.cv_jumplion.median(),
                         median_delta_cv=dc.median(), q25_delta_cv=dc.quantile(.25), q75_delta_cv=dc.quantile(.75),
                         median_abs_delta_cv=dc.abs().median(),
                         spearman_rho=g.cv_diann.rank().corr(g.cv_jumplion.rank())))
    return pd.DataFrame(rows)


def pca_scores(m, min_frac, n_pc=10):
    """Per-cell median-centred log2, features seen in >= min_frac of cells, feature-mean imputed."""
    x = m - m.median(axis=0)
    x = x[m.notna().mean(axis=1) >= min_frac]
    x = x.sub(x.mean(axis=1), axis=0).fillna(0.0)
    u, s, _ = np.linalg.svd(x.to_numpy().T, full_matrices=False)  # cells x features
    n_pc = min(n_pc, len(s))
    scores = u[:, :n_pc] * s[:n_pc]
    var = s ** 2 / np.sum(s ** 2)
    return pd.DataFrame(scores, index=m.columns, columns=[f"PC{i + 1}" for i in range(n_pc)]), var[:n_pc], x.shape[0]


def silhouette(X, labels):
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    if len(uniq) < 2 or len(X) < 3:
        return np.nan
    d = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    sil = np.zeros(len(X))
    for i in range(len(X)):
        same = labels == labels[i]
        same[i] = False
        if not same.any():
            continue
        a = d[i, same].mean()
        b = min(d[i, labels == u].mean() for u in uniq if u != labels[i])
        sil[i] = (b - a) / max(a, b)
    return float(sil.mean())


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
def _legend_above(fig, ax, handles=None):
    """One-column legend above the axes (outside the plot); call after _align, which it overrides by
    lowering the top margin to make room (two 6 pt rows ~ 0.3 in)."""
    h = fig.get_size_inches()[1]
    fig.subplots_adjust(top=1 - .36 / h)
    pos = ax.get_position()
    kw = dict(handles=handles) if handles is not None else {}
    fig.legend(frameon=False, fontsize=FS, loc="lower center", ncol=1, handlelength=1.2,
               bbox_to_anchor=((pos.x0 + pos.x1) / 2, pos.y1 + .01), **kw)


def _count_ticks(ax, level):
    """y tick labels for feature counts: precursor level compact (7500 -> 7.5k, 100000 -> 100k, so
    six-digit counts fit the left margin); protein level full numbers with thousands separators."""
    from matplotlib.ticker import FuncFormatter
    if level == "precursor":
        fmt = lambda v, _: f"{v / 1000:g}k" if abs(v) >= 1000 else f"{v:g}"
    else:
        fmt = lambda v, _: f"{v:,.0f}"
    ax.yaxis.set_major_formatter(FuncFormatter(fmt))



def _box_by_method(ax, data_by_method, ylabel, fmt="{:,.0f}", median_right=False, point_size=1.5):
    pos = np.arange(len(S.METHOD_ORDER))
    rng = np.random.default_rng(0)
    for p, method in zip(pos, S.METHOD_ORDER):
        y = np.asarray(data_by_method[method], float)
        y = y[np.isfinite(y)]
        col = S.method_color(method)
        ax.scatter(p + rng.uniform(-.28, .28, len(y)), y, s=point_size, color=col, alpha=.35, linewidths=0, rasterized=True)
        ax.boxplot([y], positions=[p], widths=.5, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor="none", edgecolor="black", linewidth=.6),
                   medianprops=dict(color="black", linewidth=1.0),
                   whiskerprops=dict(color=S.NEUTRAL, linewidth=.6), capprops=dict(color=S.NEUTRAL, linewidth=.6))
        if median_right:  # median value beside the median line, right of the box
            ax.text(p + .28, np.median(y), fmt.format(np.median(y)), ha="left", va="center", fontsize=FS)
        else:
            ax.text(p, ax.get_ylim()[1], fmt.format(np.median(y)), ha="center", va="bottom", fontsize=FS)
    ax.set_xticks(pos); ax.set_xticklabels([m.replace("DIA-NN + ", "DIA-NN +\n") for m in S.METHOD_ORDER], fontsize=FS)
    if median_right:  # room for the median labels
        ax.set_xlim(-.5, len(pos) - .4)
    ax.set_ylabel(ylabel)


def panel_ids(ids, label, MAIN, sv, level="precursor", size=None):
    # One metric per level: the protein-level run already has its own protein-groups-per-cell panel
    cols = [("protein_groups", "Protein groups per cell")] if level == "protein" else \
        [("precursors", "Precursors per cell")]
    fig, axs = plt.subplots(1, 1, figsize=size or PANEL_INT, squeeze=False)
    for ax, (col, yl) in zip(axs[0], cols):
        _box_by_method(ax, {m: ids[ids.method == m][col] for m in S.METHOD_ORDER}, yl, median_right=True, point_size=5)
        _count_ticks(ax, level)
    axs[0][0].set_title(f"n = {ids.cell.nunique()} cells", fontsize=FS, pad=2)
    compact_axes(*fig.axes)  # benchmark tick/axis-label sizes (performance.compact_axes)
    _align(fig, bottom=.19, right=.985)
    sv(fig, MAIN, f"Fig_SingleCell_{label}_A_ids_per_cell", tight=False)


def panel_completeness(curve, prev, label, MAIN, sv, level="precursor", size=None):
    fig, ax = plt.subplots(figsize=size or PANEL_INT)
    for method in S.METHOD_ORDER:
        d = curve[curve.method == method]
        ax.plot(d.min_pct_cells, d.n_features, color=S.method_color(method), label=method, linewidth=1.2)
    # Number of features detected in every cell (x = 100%); the higher method's label sits above its
    # end point, the lower one's below, so the two never overlap.
    ends = {m: int(curve[(curve.method == m)].sort_values("min_pct_cells").n_features.iloc[-1]) for m in S.METHOD_ORDER}
    upper = max(ends, key=ends.get)
    for m, n in ends.items():
        # Dot on the line's end point (x = 100%), unclipped since it sits on the axes edge.
        ax.plot([100], [n], marker="o", ms=1.5, color=S.method_color(m), linestyle="none", clip_on=False, zorder=5)
        # Upper label sits well above its end point (the curve is flat/higher just to its left, so a
        # small offset would overlap the line); lower label sits below its end point, so the y-axis
        # gets extra room at the bottom for it instead of colliding with the x axis.
        ax.annotate(f"{n:,}", xy=(100, n), xytext=(3, 5 if m == upper else -4), textcoords="offset points",
                    ha="right", va="bottom" if m == upper else "top", fontsize=FS, color=S.method_color(m))
    lo, hi = ax.get_ylim()
    ax.set_ylim(min(lo, min(ends.values())) - .12 * (hi - min(ends.values())), hi)
    ax.set_xlabel("Detected in ≥ x% of cells");ax.set_ylabel("Protein groups" if level == "protein" else "Precursors")
    ax.legend(frameon=False, fontsize=FS, loc="upper right")  # fixed: "best" put it on the 100% label
    _count_ticks(ax, level)
    compact_axes(*fig.axes)  # benchmark tick/axis-label sizes (performance.compact_axes)
    _align(fig, bottom=.19, right=.985)
    sv(fig, MAIN, f"Fig_SingleCell_{label}_B_completeness", tight=False)


def panel_cv(cv, bins, min_cells, label, MAIN, sv):
    fig, ax = plt.subplots(figsize=PANEL_INT)
    w = .34
    for k, method in enumerate(S.METHOD_ORDER):
        groups = [cv[(cv.method == method) & (cv.bin == b)].CV.to_numpy() for b in bins]
        pos = np.arange(len(bins)) + (k - .5) * w * 1.1
        ax.boxplot(groups, positions=pos, widths=w, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor=S.method_color(method), edgecolor="black", linewidth=.5, alpha=.85),
                   medianprops=dict(color="black", linewidth=.9),
                   whiskerprops=dict(color=S.NEUTRAL, linewidth=.5), capprops=dict(color=S.NEUTRAL, linewidth=.5))
    ax.set_xticks(range(len(bins))); ax.set_xticklabels([str(b).replace("–", "–\n") for b in bins], fontsize=FS)  # two lines: the ranges overlap side by side
    ax.set_xlim(-.6, len(bins) - .4)
    ax.set_xlabel("Mean JUMPlion intensity (log$_2$)"); ax.set_ylabel("CV (%)")
    compact_axes(*fig.axes)  # benchmark tick/axis-label sizes (performance.compact_axes)
    _align(fig)
    _legend_above(fig, ax, handles=[S.Patch(facecolor=S.method_color(m), label=m) for m in S.METHOD_ORDER])
    sv(fig, MAIN, f"Supp_SingleCell_{label}_E_CV_by_intensity", tight=False)


def panel_paired_cv(p, bins, summ, min_cells, label, MAIN, sv, level="precursor"):
    """CV in DIA-NN vs CV in JUMPlion for the same features, with y = x. Lower CV is not better here --
    agreement with the DIA-NN CV (points on y = x) is the goal."""
    noun = "protein" if level == "protein" else "precursor"
    fig, ax = plt.subplots(figsize=PANEL_INT)
    # cutoff: 95th percentile of the DIA-NN CVs alone. Features above it in DIA-NN are left out of the plot, n and r;
    # JUMPlion's CV is not cut. Both axes run to the larger of the cutoff and JUMPlion's largest CV among the kept
    # features, so the plot is square and y = x is the diagonal.
    cutoff = float(np.nanquantile(p.cv_diann.dropna(), .95))
    ok = np.isfinite(p.cv_diann) & np.isfinite(p.cv_jumplion) & (p.cv_diann <= cutoff)
    lim = max(cutoff, float(p.cv_jumplion[ok].max()))
    ax.hexbin(p.cv_diann[ok], p.cv_jumplion[ok], gridsize=45, extent=(0, lim, 0, lim), bins="log",
              cmap="Blues", mincnt=1, linewidths=0, rasterized=True)
    ax.plot([0, lim], [0, lim], color=S.REF_LINE, linewidth=.6, linestyle="--")
    from scipy import stats
    r, pval = stats.pearsonr(p.cv_diann[ok], p.cv_jumplion[ok])  # Pearson r on the plotted features
    p_txt = "p < 1e-300" if pval < 1e-300 else (f"p = {pval:.1e}" if pval < .001 else f"p = {pval:.3f}")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.text(.97, .03, f"r = {r:.4f}\n{p_txt}", transform=ax.transAxes, ha="right", va="bottom", fontsize=FS)
    ax.set_title(f"n = {int(ok.sum()):,} {noun}s", fontsize=FS, pad=2)
    ax.set_xlabel("CV, DIA-NN (%)"); ax.set_ylabel("CV, JUMPlion (%)")
    compact_axes(*fig.axes)  # benchmark tick/axis-label sizes (performance.compact_axes)
    _align(fig)
    sv(fig, MAIN, f"Fig_SingleCell_{label}_D_paired_CV", tight=False)


def _group_colors(groups):
    uniq = sorted(pd.unique(groups))
    return {g: S.CYCLE[i % len(S.CYCLE)] for i, g in enumerate(uniq)}


def _scatter_embedding(axs, embeds, groups, xlab_fmt, sil, title_extra=""):
    colors = _group_colors(groups) if groups is not None else None
    for ax, method in zip(axs, S.METHOD_ORDER):
        xy = embeds[method]
        if colors:
            for g, c in colors.items():
                sel = np.asarray(groups) == g
                ax.scatter(xy[sel, 0], xy[sel, 1], s=3, color=c, linewidths=0, alpha=.8, label=str(g), rasterized=True)
        else:
            ax.scatter(xy[:, 0], xy[:, 1], s=3, color=S.method_color(method), linewidths=0, alpha=.8, rasterized=True)
        ax.set_xlabel(xlab_fmt[method][0]); ax.set_ylabel(xlab_fmt[method][1])
        t = method + (f"\nsilhouette = {sil[method]:.2f}" if colors and np.isfinite(sil.get(method, np.nan)) else "")
        ax.set_title(t, fontsize=5.5)
    if colors:
        axs[-1].legend(frameon=False, fontsize=5, markerscale=2, loc="center left", bbox_to_anchor=(1.0, .5))


def panel_pca(scores, vars_, groups, sil, label, MAIN, sv, n_feats=None, level="precursor"):
    # Saved at exactly INT_WIDTH (the width of the two panels above it in the Supp integrated figure), with the
    # group legend inside that width: tight_layout makes room for it, and the figure is not bbox-cropped.
    fig, axs = plt.subplots(1, 2, figsize=(INT_WIDTH, PANEL[1]))
    embeds = {m: scores[m][["PC1", "PC2"]].to_numpy() for m in S.METHOD_ORDER}
    labs = {m: (f"PC1 ({vars_[m][0] * 100:.1f}%)", f"PC2 ({vars_[m][1] * 100:.1f}%)") for m in S.METHOD_ORDER}
    _scatter_embedding(axs, embeds, groups, labs, sil)
    if n_feats:  # features that went into the PCs, per method
        noun = "proteins" if level == "protein" else "precursors"
        for ax, method in zip(axs, S.METHOD_ORDER):
            ax.set_title(f"{ax.get_title()}\n{n_feats[method]:,} {noun} used", fontsize=5.5)
    compact_axes(*fig.axes)  # benchmark tick/axis-label sizes
    fig.tight_layout()
    sv(fig, MAIN, f"Supp_SingleCell_{label}_G_PCA", tight=False)


def find_marker_rows(annot, genes):
    """gene -> matrix row id: first row whose ';'-separated Genes entry contains the gene exactly."""
    if "Genes" not in annot:
        return {}
    gl = annot["Genes"].fillna("").astype(str).str.split(";")
    out = {}
    for g in genes:
        hit = gl.map(lambda xs: g in [x.strip() for x in xs])
        if hit.any():
            out[g] = hit[hit].index[0]
    return out


CELL_ABBR = {"T cells": "T", "B cells": "B", "NK cells": "NK", "Dendritic cells": "DC", "Monocytes": "Mono"}


def collapse_shared_markers(markers):
    """{cell type: genes} -> each marker appears once. Listed under one cell type: stays there. Listed
    under several: goes to a category named for exactly those cell types (e.g. "T/NK", "B/DC/Mono"; "All"
    when it is listed under every cell type), so which types share it is explicit. Cell-type categories
    come first (config order), then the combination categories (order of first appearance)."""
    cts_of = {}
    for ct, gs in markers.items():
        for g in dict.fromkeys(gs):
            cts_of.setdefault(g, []).append(ct)
    out = {ct: [g for g, cts in cts_of.items() if cts == [ct]] for ct in markers}
    out = {ct: gs for ct, gs in out.items() if gs}
    for g, cts in cts_of.items():
        if len(cts) > 1:
            name = "All" if len(cts) == len(markers) else "/".join(CELL_ABBR.get(c, c) for c in cts)
            out.setdefault(name, []).append(g)
    return out


def panel_cell_corr(mats, max_cells, label, OUT, sv, prefix="Supp"):
    rng = np.random.default_rng(0)
    data = {}
    for key, method in METHOD_KEYS:
        m = mats[key]
        cols = list(m.columns)
        if len(cols) > max_cells:
            cols = list(rng.choice(cols, max_cells, replace=False))
        c = m[cols].corr(min_periods=50).to_numpy()
        data[method] = c[np.triu_indices_from(c, k=1)]
    fig, ax = plt.subplots(figsize=PANEL_INT)
    _box_by_method(ax, data, "Pairwise cell-cell correlation", fmt="{:.4f}", median_right=True)
    compact_axes(*fig.axes)  # benchmark tick/axis-label sizes (performance.compact_axes)
    _align(fig)
    sv(fig, OUT, f"{prefix}_SingleCell_{label}_F_cell_correlation", tight=False)


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def run(cfg, layout, saver, resolve):
    """One pass per entry of singlecell.levels (precursor and/or protein), all in one run; every
    figure/table name carries the level (e.g. Fig_SingleCell_HeLa_precursor_A_ids_per_cell)."""
    sc = cfg["singlecell"]
    meta = pd.read_csv(resolve(sc["sample_meta_file"]))
    meta["Run"] = meta["Run"].astype(str)
    for level, lvl in sc["levels"].items():
        log.info("==== %s level ====", level)
        run_level(sc, level, lvl, meta, layout, saver, resolve)


def recovered_values(mats):
    """Log2 values of (feature, cell) entries quantified by DIA-NN, and of those missing in DIA-NN but
    quantified by JUMPlion (the JUMPlion value). Compared on the features and cells both matrices share."""
    d_key, j_key = (k for k, _ in METHOD_KEYS)
    feats = mats[d_key].index.intersection(mats[j_key].index)
    d = mats[d_key].loc[feats].to_numpy(float)
    j = mats[j_key].loc[feats, mats[d_key].columns].to_numpy(float)
    return d[np.isfinite(d)], j[np.isfinite(j) & ~np.isfinite(d)]


def panel_recovered_density(reported, added, label, MAIN, sv, level="precursor", bins=60):
    """Overlaid density of log2 intensity: values DIA-NN reported vs values JUMPlion recovered (missing in
    DIA-NN, quantified by JUMPlion). Each histogram is normalised to unit area."""
    noun = "Protein" if level == "protein" else "Precursor"
    fig, ax = plt.subplots(figsize=PANEL_INT)
    edges = np.linspace(*np.quantile(np.concatenate([reported, added]), [.001, .999]), bins + 1)
    mid = (edges[:-1] + edges[1:]) / 2
    for vals, col, name in ((reported, S.method_color(S.METHOD_ORDER[0]), "DIA-NN reported"),
                               (added, RECOVERED_GREEN, "JUMPlion recovered")):
        dens, _ = np.histogram(vals, bins=edges, density=True)
        ax.fill_between(mid, dens, color=col, alpha=.3, linewidth=0)
        ax.plot(mid, dens, color=col, linewidth=1.0, label=f"{name} (n={len(vals):,})")
    ax.set_xlabel(f"{noun} log$_2$ intensity"); ax.set_ylabel("Density")
    ax.set_ylim(bottom=0)
    compact_axes(*fig.axes)  # benchmark tick/axis-label sizes (performance.compact_axes)
    _align(fig)
    _legend_above(fig, ax)
    sv(fig, MAIN, f"Supp_SingleCell_{label}_C_intensity_density", tight=False)


def assemble_integrated(label, main_dir, saver):
    """Fig_SingleCell_{label}_Integrated -- row 1: A (IDs per cell) | B (completeness); row 2: D (paired CV)
    | F (cell correlation). Panels are read back from their saved PNGs (needs "png" in style.formats) and
    tiled at their own physical size; letters a-d are added here."""
    stems = {"A": "A_ids_per_cell", "B": "B_completeness", "D": "D_paired_CV", "F": "F_cell_correlation"}
    paths = {k: Path(main_dir) / f"Fig_SingleCell_{label}_{s}.png" for k, s in stems.items()}
    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        log.warning("integrated figure skipped: missing panel PNG(s) for %s (is \"png\" in style.formats?)", missing)
        return
    imgs = {k: plt.imread(p) for k, p in paths.items()}
    w = {k: im.shape[1] / saver.dpi for k, im in imgs.items()}
    h = {k: im.shape[0] / saver.dpi for k, im in imgs.items()}
    rows, gap = [["A", "B"], ["D", "F"]], INT_GAP
    row_w = [sum(w[k] for k in r) + gap * (len(r) - 1) for r in rows]
    row_h = [max(h[k] for k in r) for r in rows]
    total_w, total_h = max(row_w), sum(row_h)
    fig = plt.figure(figsize=(total_w, total_h))
    y = 0.0
    for r, rw, rh in zip(rows, row_w, row_h):
        x = (total_w - rw) / 2  # centre a narrower row under/over the wider one
        for k in r:
            n = sum(len(q) for q in rows[:rows.index(r)]) + r.index(k)
            ax = fig.add_axes([x / total_w, 1 - (y + h[k]) / total_h, w[k] / total_w, h[k] / total_h])
            ax.imshow(imgs[k]); ax.axis("off")
            fig.text((x + .06) / total_w, 1 - (y + .06) / total_h, "abcd"[n],
                     fontsize=9, fontweight="bold", ha="left", va="top")
            x += w[k] + gap
        y += rh
    saver.save(fig, main_dir, f"Fig_SingleCell_{label}_Integrated")


def assemble_supp_integrated(label, supp_dir, saver):
    """Supp_SingleCell_{label}_Integrated -- row 1: C (intensity density) | E (CV by intensity); row 2: G (PCA),
    left-aligned with C. Panels are read back from their saved PNGs (needs "png" in style.formats) and tiled at their own
    physical size; letters a-c are added here."""
    stems = {"C": "C_intensity_density", "E": "E_CV_by_intensity", "G": "G_PCA"}
    paths = {k: Path(supp_dir) / f"Supp_SingleCell_{label}_{s}.png" for k, s in stems.items()}
    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        log.warning("supp integrated figure skipped: missing panel PNG(s) for %s (is \"png\" in style.formats?)", missing)
        return
    imgs = {k: plt.imread(p) for k, p in paths.items()}
    w = {k: im.shape[1] / saver.dpi for k, im in imgs.items()}
    h = {k: im.shape[0] / saver.dpi for k, im in imgs.items()}
    rows, gap = [["C", "E"], ["G"]], INT_GAP
    row_w = [sum(w[k] for k in r) + gap * (len(r) - 1) for r in rows]
    row_h = [max(h[k] for k in r) for r in rows]
    total_w, total_h = max(row_w), sum(row_h)
    fig = plt.figure(figsize=(total_w, total_h))
    y, n = 0.0, 0
    for r, rw, rh in zip(rows, row_w, row_h):
        x = 0.0  # left-aligned: G's left edge lines up with C's
        for k in r:
            ax = fig.add_axes([x / total_w, 1 - (y + h[k]) / total_h, w[k] / total_w, h[k] / total_h])
            ax.imshow(imgs[k]); ax.axis("off")
            fig.text((x + .06) / total_w, 1 - (y + .06) / total_h, "abc"[n],
                     fontsize=9, fontweight="bold", ha="left", va="top")
            x += w[k] + gap
            n += 1
        y += rh
    saver.save(fig, supp_dir, f"Supp_SingleCell_{label}_Integrated")


def run_level(sc, level, lvl, meta, layout, saver, resolve):
    label = f"{sc.get('label', 'dataset')}_{level}"
    id_col = lvl.get("id_col", "Protein.Group" if level == "protein" else "Precursor.Id")
    protein_col = sc.get("protein_col", "Protein.Group")
    group_col = sc.get("group_col")

    mats, annots = {}, {}
    for key, _ in METHOD_KEYS:
        mats[key], annots[key] = load_matrix(resolve(lvl["matrices"][key]), meta, id_col)
        log.info("%s %s: %d %s x %d cells", label, key, mats[key].shape[0], level, mats[key].shape[1])
    cells = [c for c in mats["diann"].columns if c in mats["jumplion"].columns]
    for key in mats:
        mats[key] = mats[key][cells]
    groups = None
    if group_col and group_col in meta:
        g = meta.set_index("Run").reindex(cells)[group_col]
        g = g.where(g.notna() & (g.astype(str).str.strip() != ""))  # blank/NaN -> missing
        if g.notna().any():  # an entirely empty group column means no groups: single colour, no silhouette
            groups = g.fillna("Unknown").astype(str).to_numpy()

    # protein level -> main (Fig_); precursor level -> supp (Supp_)
    to_supp = level != "protein"
    main_dir = layout.supp if to_supp else layout.main
    T = layout.table

    def sv(fig, d, stem, **kw):
        saver.save(fig, d, "Supp_" + stem[len("Fig_"):] if to_supp and stem.startswith("Fig_") else stem, **kw)

    psize = tuple(sc["panel_size"]) if sc.get("panel_size") else None  # optional (w, h) in for panels A and B
    ids = ids_per_cell(mats, annots, protein_col, level)
    write_tsv(ids, T, f"ids_per_cell_{label}.tsv")
    panel_ids(ids, label, main_dir, sv, level, size=psize)

    curve = completeness_curve(mats)
    write_tsv(curve, T, f"completeness_curve_{label}.tsv")
    prev = prevalence_table(mats)
    write_tsv(prev, T, f"prevalence_counts_{label}.tsv")
    log.info("%s prevalence (JUMPlion vs DIA-NN):\n%s", label, prev.round(1).to_string(index=False))
    panel_completeness(curve, prev, label, main_dir, sv, level, size=psize)

    skip = set(sc.get("skip_analyses", []))  # any of: cv_by_intensity, paired_cv, cell_correlation
    min_cells = int(sc.get("cv_min_cells", 5))
    if "cv_by_intensity" not in skip:
        cv, bins = cv_by_intensity(mats, min_cells)
        write_tsv(cv.groupby(["method", "bin"], observed=True).CV.agg(["count", "median"]).reset_index(),
                  T, f"CV_by_intensity_{label}.tsv")
        panel_cv(cv, bins, min_cells, label, layout.supp, sv)

    if "paired_cv" not in skip:
        paired, pbins = paired_cv(mats, min_cells)
        summ = paired_cv_summary(paired)
        write_tsv(paired, T, f"paired_CV_per_feature_{label}.tsv.gz")
        write_tsv(summ, T, f"paired_CV_summary_{label}.tsv")
        log.info("%s paired CV:\n%s", label, summ.round(3).to_string(index=False))
        panel_paired_cv(paired, pbins, summ, min_cells, label, main_dir, sv, level)

    reported, added = recovered_values(mats)
    if not added.size:
        log.warning("%s: JUMPlion added no values -- skipping the intensity density panel", label)
    else:
        write_tsv(pd.DataFrame([dict(source=n, n_values=len(v), median_log2=np.median(v),
                                     q25_log2=np.quantile(v, .25), q75_log2=np.quantile(v, .75))
                                for n, v in (("DIA-NN reported", reported), ("JUMPlion recovered", added))]),
                  T, f"intensity_density_summary_{label}.tsv")
        panel_recovered_density(reported, added, label, layout.supp, sv, level)

    if sc.get("embedding", True):  # PCA/UMAP -- off for single-cell-type datasets (HeLa)
        scores, vars_, sil, n_feats = {}, {}, {}, {}
        for key, method in METHOD_KEYS:
            scores[method], vars_[method], n_feat = pca_scores(mats[key], float(sc.get("pca_min_cell_frac", 0.5)))
            n_feats[method] = n_feat
            log.info("%s PCA: %d features", method, n_feat)
            sil[method] = silhouette(scores[method].iloc[:, :min(10, scores[method].shape[1])].to_numpy(), groups) \
                if groups is not None else np.nan
            write_tsv(scores[method].assign(group=groups if groups is not None else "").reset_index(names="cell"),
                      T, f"PCA_scores_{label}_{key}.tsv")
        panel_pca(scores, vars_, groups, sil, label, layout.supp, sv, n_feats, level)
        if level == "protein" and added.size and "cv_by_intensity" not in skip:  # needs panels C and E too
            assemble_supp_integrated(label, layout.supp, saver)
    in_main = level == "protein"
    if "cell_correlation" not in skip:
        panel_cell_corr(mats, int(sc.get("corr_max_cells", 200)), label, layout.main if in_main else layout.supp, sv,
                        "Fig" if in_main else "Supp")
    if level == "protein" and not skip & {"paired_cv", "cell_correlation"}:  # the figure needs both panels
        assemble_integrated(label, main_dir, saver)


def main(default_config, argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=default_config)
    ap.add_argument("--output-dir", help="Override output_dir from the config")
    args = ap.parse_args(argv)
    setup_logging()
    raw = read_raw_config(args.config)
    cfg = substitute_config(raw)
    resolve = PathResolver(cfg["_config_dir"])
    style_cfg = cfg.get("style", {})
    S.apply_style(style_cfg)
    saver = S.FigureSaver(style_cfg.get("formats", ["pdf", "png"]), style_cfg.get("dpi", 600))
    out_root = Path(args.output_dir) if args.output_dir else resolve(cfg.get("output_dir", "jumplion_figures_output"))
    layout = OutputLayout(out_root, f"singlecell_{cfg['singlecell'].get('label', 'dataset')}")  # one folder, both levels
    run(cfg, layout, saver, resolve)
    S.plt.close("all")
    return 0
