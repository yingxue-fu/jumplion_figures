"""Scanpy clustering of the protein-level single-cell matrix (PBMC): DIA-NN vs JUMPlion.

Adapts the scanpy 3k-PBMC clustering workflow (log-normalise -> HVG -> scale -> PCA -> neighbours ->
UMAP -> Leiden -> marker ranking -> annotation) to a proteomics intensity matrix:
  * no count normalisation / log1p: intensities are log2-transformed and each cell centred on its median
  * no mitochondrial QC, no HVG selection, no regress_out: proteins are filtered on prevalence instead
  * missing values (not detected = usually low abundance) are imputed from a left-shifted normal drawn per
    cell (Perseus-style) before scaling/PCA -- config "impute": "left_shift" | "mean"
  * marker ranking is a per-cluster Wilcoxon rank-sum on the OBSERVED (non-imputed) values only
Each method is clustered on its own matrix; both clusterings are drawn on the JUMPlion UMAP
coordinates so they can be compared, and agreement is reported as ARI / NMI.

Config: the "scanpy" section of the PBMC single-cell config. Output:
<output_dir>/singlecell_<label>_scanpy/{main,supp,table}.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

from matplotlib.lines import Line2D
from . import style as S
from .common import (OutputLayout, PathResolver, read_raw_config, setup_logging,
                     substitute_config, write_tsv)
from .performance import compact_axes
from .singlecell import (FS, METHOD_KEYS, RECOVERED_GREEN, collapse_shared_markers, find_marker_rows,
                         load_matrix, silhouette)

plt = S.plt

# integrated PBMC figure: every row is INT_W wide; panel sizes are set so rows and columns line up
INT_W, INT_GAP, INT_ROW_H, INT_ROW4_H = 6.0, 0.15, 1.9, 3.0
INT_AB_W = (INT_W - 2 * INT_GAP) / 3  # panels A, B and H share row 1 equally (config panel_size must match)
log = logging.getLogger("jumplion_figures.singlecell_scanpy")


# ----------------------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------------------
def center_cells(m):
    """log2 matrix (feature x cell) -> each cell centred on its own median over observed values."""
    return m - m.median(axis=0)


def impute(x, mode, rng, shift=1.8, width=0.3):
    """x: cells x features (NaN = missing). 'left_shift': per cell, N(mean - shift*sd, width*sd) of that
    cell's observed values; 'mean': per-feature mean (0 after feature centring happens in scale)."""
    x = x.copy()
    if mode == "mean":
        return x.fillna(x.mean(axis=0))
    for cell in x.index:
        row = x.loc[cell]
        miss = row.isna()
        if miss.any():
            mu, sd = row.mean(), row.std()
            x.loc[cell, miss] = rng.normal(mu - shift * sd, width * sd, miss.sum())
    return x


def _leiden(adata, resolution, key_added="leiden"):
    import scanpy as sc
    try:
        import igraph  # noqa: F401
        sc.tl.leiden(adata, resolution=resolution, random_state=0, flavor="igraph",
                     n_iterations=2, directed=False, key_added=key_added)
    except ImportError:
        sc.tl.leiden(adata, resolution=resolution, random_state=0, key_added=key_added)


def resolution_scan(adata, resolutions):
    """Leiden labels at each resolution on the already-built neighbour graph -> {resolution: labels}."""
    return {r: (_leiden(adata, r, f"leiden_r{r}"), adata.obs[f"leiden_r{r}"].astype(str).to_numpy())[1]
            for r in resolutions}


def cluster_method(centered, cfg, rng, keep=None):
    """centered: feature x cell (observed, median-centred log2, NaN missing). keep: optional fixed feature
    set (default: this matrix's own features seen in >= min_cell_frac of cells).
    -> dict with adata, labels, n_features, obs / imp (cells x kept features, before / after imputation)."""
    import anndata as ad
    import scanpy as sc

    if keep is None:
        frac = centered.notna().mean(axis=1)
        keep = centered.index[frac >= cfg["min_cell_frac"]]
    obs = centered.loc[keep].T
    x = impute(obs, cfg["impute"], rng)
    adata = ad.AnnData(X=x.to_numpy(np.float32), obs=pd.DataFrame(index=x.index.astype(str)),
                       var=pd.DataFrame(index=pd.Index(keep.astype(str))))
    sc.pp.scale(adata, max_value=10)
    n_pcs = min(cfg["n_pcs"], adata.n_obs - 1, adata.n_vars - 1)
    sc.pp.pca(adata, n_comps=n_pcs, svd_solver="arpack", random_state=0)
    sc.pp.neighbors(adata, n_neighbors=cfg["n_neighbors"], n_pcs=n_pcs, random_state=0)
    sc.tl.umap(adata, random_state=0)
    _leiden(adata, cfg["resolution"])
    return dict(adata=adata, n_features=len(keep), labels=adata.obs["leiden"].astype(str).to_numpy(),
                obs=obs, imp=x)


# ----------------------------------------------------------------------------
# Marker ranking on observed values
# ----------------------------------------------------------------------------
def rank_markers(centered, labels, min_obs=3):
    """Per cluster vs rest Wilcoxon rank-sum (Mann-Whitney U) on observed values only; log2FC = difference
    of medians; Benjamini-Hochberg FDR within each cluster. Proteins with < min_obs observed values in
    either group are skipped."""
    from scipy.stats import mannwhitneyu
    X = centered.to_numpy(float)  # feature x cell
    rows = []
    for cl in sorted(set(labels), key=lambda s: int(s)):
        inn = labels == cl
        res = []
        for i in range(X.shape[0]):
            a, b = X[i, inn], X[i, ~inn]
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            if len(a) < min_obs or len(b) < min_obs:
                continue
            p = mannwhitneyu(a, b, alternative="two-sided").pvalue
            res.append((centered.index[i], len(a), len(b), np.median(a) - np.median(b), p))
        if not res:
            continue
        d = pd.DataFrame(res, columns=["feature", "n_obs_cluster", "n_obs_rest", "log2FC", "p"])
        order = d.p.rank(method="first")
        d["fdr"] = (d.p * len(d) / order).clip(upper=1)
        d["fdr"] = d.assign(_o=order).sort_values("_o", ascending=False).fdr.cummin().reindex(d.index)
        d.insert(0, "cluster", cl)
        rows.append(d)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def marker_summary(centered, labels, rows, markers):
    """Per (gene, cluster): mean observed log2, detection fraction; z-scored across clusters per gene."""
    clusters = sorted(set(labels), key=lambda s: int(s))
    recs = []
    for ct, genes in markers.items():
        for g in genes:
            if g not in rows:
                continue
            v = centered.loc[rows[g]].to_numpy(float)
            for cl in clusters:
                x = v[labels == cl]
                ok = np.isfinite(x)
                recs.append(dict(cell_type=ct, gene=g, cluster=cl, mean_log2=x[ok].mean() if ok.any() else np.nan,
                                 frac_detected=ok.mean()))
    d = pd.DataFrame(recs)
    if d.empty:
        return d
    d["z"] = d.groupby("gene").mean_log2.transform(lambda s: (s - s.mean()) / (s.std(ddof=0) or np.nan))
    return d


def suggest_annotation(summ, cell_types):
    """Heuristic label per cluster: marker category with the highest mean z of its markers (needs > 0).
    Only the real cell types are candidates; combination categories ("T/NK", "All", ...) are excluded."""
    cat = summ[summ.cell_type.isin(cell_types)].groupby(["cluster", "cell_type"]).z.mean().unstack()
    best = cat.idxmax(axis=1).where(cat.max(axis=1) > 0, "Unassigned")
    out = cat.copy()
    out["suggested_label"] = best
    return out.reset_index()


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
def _cluster_colors(labels):
    uniq = sorted(set(labels), key=lambda s: (not s.lstrip("-").isdigit(), int(s) if s.lstrip("-").isdigit() else s))
    return {u: S.CYCLE[i % len(S.CYCLE)] for i, u in enumerate(uniq)}


def panel_umap_clusters(xy, results, agree, sil, label, MAIN, sv):
    """JUMPlion Leiden clusters on the JUMPlion UMAP (the DIA-NN UMAP is in the supp panel)."""
    fig, ax = plt.subplots(figsize=(INT_AB_W, INT_ROW_H))
    lab = np.asarray(results["jumplion"]["labels"])
    for cl, col in _cluster_colors(lab).items():
        sel = lab == cl
        ax.scatter(xy[sel, 0], xy[sel, 1], s=5, color=col, linewidths=0, label=cl, rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    ax.set_title(f"{S.METHOD_ORDER[1]} clustering", fontsize=FS, pad=2)
    ax.legend(frameon=False, fontsize=FS, markerscale=1.5, loc="center left", bbox_to_anchor=(1.0, .5),
              title="Cluster", title_fontsize=FS)
    compact_axes(ax)
    # same top / bottom margins as panels A and B (singlecell._align, bottom=.19), so the plot areas line up; no tick
    # labels here, so the left margin is small (only the UMAP2 label)
    # the right margin leaves room for the legend
    fig.subplots_adjust(left=.15, right=.80, bottom=.19, top=.92)
    sv(fig, MAIN, f"Fig_SingleCell_{label}_H_scanpy_clusters", tight=False)


def panel_umap_clusters_diann(xy, results, agree, sil, label, SUPP, sv, xy_jumplion=None):
    """Both independent clusterings drawn on the DIA-NN UMAP (the counterpart of the main H panel,
    which uses the JUMPlion UMAP); with `xy_jumplion`, a second row draws them on the JUMPlion UMAP."""
    embeds = [(S.METHOD_ORDER[0], xy)] + ([(S.METHOD_ORDER[1], xy_jumplion)] if xy_jumplion is not None else [])
    fig = plt.figure(figsize=(2.1 * 2 + .8, 2.1 * len(embeds)), layout="constrained")
    subfigs = np.atleast_1d(fig.subfigures(len(embeds), 1))
    panels = [("diann", S.METHOD_ORDER[0]), ("jumplion", S.METHOD_ORDER[1])]
    for subfig, (emb_name, emb) in zip(subfigs, embeds):
        axs = subfig.subplots(1, 2)
        for ax, (key, method) in zip(axs, panels):
            lab = np.asarray(results[key]["labels"])
            for cl, col in _cluster_colors(lab).items():
                sel = lab == cl
                ax.scatter(emb[sel, 0], emb[sel, 1], s=5, color=col, linewidths=0, label=cl, rasterized=True)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
            ax.legend(frameon=False, fontsize=FS, markerscale=1.5, loc="center left", bbox_to_anchor=(1.0, .5),
                      title="Cluster", title_fontsize=FS)
            ax.set_title(f"{method}: {len(set(lab))} clusters\nsilhouette {sil[key]:.2f}", fontsize=5.5)
            compact_axes(ax)
        subfig.suptitle(f"Clusterings drawn on the {emb_name} UMAP", fontsize=5.5)
    fig.suptitle(f"ARI {agree['ari']:.2f} · NMI {agree['nmi']:.2f} (DIA-NN vs JUMPlion)", fontsize=5.5)
    fig.supxlabel(f"same {results['diann']['n_features']:,} proteins in both methods", fontsize=5)
    sv(fig, SUPP, f"Supp_SingleCell_{label}_N_scanpy_clusters_DIANN_UMAP")


def panel_marker_coverage(mats, annots, markers, label, SUPP, sv, labels=None):
    """Bars: % of cells in which each marker protein is quantified, DIA-NN vs JUMPlion (same cells),
    ordered by DIA-NN coverage (high to low); the number above each pair is the change in the NUMBER of cells."""
    labels = labels or {}
    cats = collapse_shared_markers(markers)
    rows = {k: find_marker_rows(annots[k], [g for gs in cats.values() for g in gs]) for k, _ in METHOD_KEYS}
    recs = []
    for ct, genes in cats.items():
        for g in genes:
            if not any(g in rows[k] for k in rows):
                continue  # not in either matrix
            rec = dict(category=ct, gene=g)
            for k, method in METHOD_KEYS:
                n_obs = int(mats[k].loc[rows[k][g]].notna().sum()) if g in rows[k] else 0
                rec[f"n_{k}"] = n_obs
                rec[k] = n_obs / mats[k].shape[1] * 100
            rec["delta_pp"] = rec["jumplion"] - rec["diann"]
            rec["delta_cells"] = rec["n_jumplion"] - rec["n_diann"]
            recs.append(rec)
    d = pd.DataFrame(recs)
    if d.empty:
        return d
    d = d.sort_values(["diann", "jumplion", "gene"], ascending=[False, False, True], ignore_index=True)  # by DIA-NN coverage
    n = len(d)
    fig, ax = plt.subplots(figsize=(INT_W, INT_ROW_H))
    x = np.arange(n)
    w = .38
    for i, (k, method) in enumerate(METHOD_KEYS):
        ax.bar(x + (i - .5) * w, d[k], width=w, color=S.method_color(method), label=method)
    for xi, dv in zip(x, d.delta_cells):  # one row of changes (cells), same height above every pair
        ax.text(xi, 103, f"{dv:+d}", ha="center", va="bottom", fontsize=FS)
    ax.set_xticks(x); ax.set_xticklabels([labels.get(g, g) for g in d.gene], rotation=90, fontsize=FS)
    ax.set_xlim(-.7, n - .3); ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Cells quantified (%)")
    ax.legend(frameon=False, fontsize=6, loc="lower left", bbox_to_anchor=(0, 1.12), ncol=2)
    fig.tight_layout()
    sv(fig, SUPP, f"Fig_SingleCell_{label}_O_marker_coverage", tight=False)
    return d.rename(columns={"diann": "pct_cells_diann", "jumplion": "pct_cells_jumplion"})


def panel_marker_violin(mats, annots, clusters, genes, label, SUPP, sv, names=None, ncols=3,
                       cluster_source="JUMPlion", stem="P_marker_violin", prefix="Supp", figsize=None, xlabel=None, ylabel_once=False):
    """Per marker: ONE violin per Leiden cluster (cluster_source: which method defined them) -- the distribution of all log2 values
    quantified in the JUMPlion matrix -- with every cell drawn as a jittered point: blue = value DIA-NN
    reported, green = value missing in DIA-NN and recovered by JUMPlion. Cells with no value are not drawn."""
    names = names or {}
    d_key, j_key = (k for k, _ in METHOD_KEYS)
    rows = {k: find_marker_rows(annots[k], list(genes)) for k in (d_key, j_key)}
    use = [g for g in genes if g in rows[j_key]]  # recovered values only exist in the JUMPlion matrix
    if missing := [g for g in genes if g not in use]:
        log.warning("violin markers absent from the JUMPlion matrix (not drawn): %s", missing)
    if not use:
        return pd.DataFrame()
    clusters = np.asarray(clusters)
    order = sorted(set(clusters), key=int)
    nrows = int(np.ceil(len(use) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize or (1.9 * ncols + .3, 1.9 * nrows + .5), squeeze=False,
                            layout="constrained")
    col_rep, col_rec = S.method_color(S.METHOD_ORDER[0]), RECOVERED_GREEN
    recs = []
    jit = np.random.default_rng(0)  # fixed seed: same jitter every run
    for i, (ax, g) in enumerate(zip(axs.ravel(), use)):
        vj = mats[j_key].loc[rows[j_key][g]].to_numpy(float)
        reported = (np.isfinite(mats[d_key].loc[rows[d_key][g]].to_numpy(float)) if g in rows[d_key]
                    else np.zeros(len(vj), bool))
        for ci, cl in enumerate(order):
            sel = clusters == cl
            v = vj[sel]
            ok = np.isfinite(v)
            rep, rec = ok & reported[sel], ok & ~reported[sel]
            recs.append(dict(gene=g, cluster=cl, n_cells=int(sel.sum()), n_reported=int(rep.sum()),
                             n_recovered=int(rec.sum()), median_log2=np.median(v[ok]) if ok.any() else np.nan,
                             median_log2_reported=np.median(v[rep]) if rep.any() else np.nan,
                             median_log2_recovered=np.median(v[rec]) if rec.any() else np.nan))
            if ok.sum() >= 3:
                vp = ax.violinplot([v[ok]], positions=[ci], widths=.8, showextrema=False)
                for body in vp["bodies"]:
                    body.set_facecolor(S.LIGHT); body.set_edgecolor("none"); body.set_alpha(.55)
                ax.hlines(np.median(v[ok]), ci - .22, ci + .22, color="black", linewidth=.8, zorder=4)
            for m, col in ((rep, col_rep), (rec, col_rec)):  # recovered drawn last, on top
                ax.scatter(ci + jit.uniform(-.25, .25, int(m.sum())), v[m], s=6, color=col, alpha=.85,
                           linewidths=0, zorder=3, rasterized=True)
        ax.set_xticks(range(len(order))); ax.set_xticklabels(order)
        ax.set_xlim(-.6, len(order) - .4)
        ax.set_title(names.get(g, g), fontsize=FS + .5)
        ax.set_xlabel(xlabel or f"Cluster ({cluster_source})")
        ax.set_ylabel("log$_2$ intensity" if not ylabel_once or i % ncols == 0 else "")  # once per row, at the left
        compact_axes(ax)
    for ax in axs.ravel()[len(use):]:
        ax.axis("off")
    fig.legend(handles=[Line2D([], [], marker="o", linestyle="", markersize=3, color=c, label=n)
                        for c, n in ((col_rep, "DIA-NN reported"), (col_rec, "JUMPlion recovered"))],
               frameon=False, fontsize=FS, loc="outside upper center", ncol=2)
    sv(fig, SUPP, f"{prefix}_SingleCell_{label}_{stem}", tight=figsize is None)  # given size is kept exactly
    return pd.DataFrame(recs)


def panel_celltype_dotplot(mats, annots, types, type_order, genes, label, MAIN, sv, names=None):
    """Rows: the DIA-NN Leiden clusters (top to bottom as type_order; groups = cluster label per cell); columns: markers
    ordered left to right by their z-scored
    mean log2 across ALL the clusters together (JUMPlion values; see the centre-of-mass ordering below). Top:
    DIA-NN values, bottom: JUMPlion values (same
    cells, same marker order). Dot size = fraction of the type's cells with a value; colour = mean observed log2,
    z-scored across the clusters for each marker."""
    names = names or {}
    d_key, j_key = (k for k, _ in METHOD_KEYS)
    rows = {k: find_marker_rows(annots[k], list(genes)) for k in (d_key, j_key)}
    use = [g for g in genes if g in rows[j_key]]
    if missing := [g for g in genes if g not in use]:
        log.warning("dot plot markers absent from the JUMPlion matrix (not drawn): %s", missing)
    if not use:
        return pd.DataFrame()
    types = np.asarray(types)
    recs = []
    for k, method in METHOD_KEYS:
        for g in use:
            if g not in rows[k]:
                continue
            v = mats[k].loc[rows[k][g]].to_numpy(float)
            for t in type_order:
                x = v[types == t]
                ok = np.isfinite(x)
                recs.append(dict(method=method, gene=g, cell_type=t, n_cells=len(x), n_observed=int(ok.sum()),
                                 frac_detected=ok.mean() if len(x) else np.nan,
                                 mean_log2=x[ok].mean() if ok.any() else np.nan))
    d = pd.DataFrame(recs)
    d["z"] = d.groupby(["method", "gene"]).mean_log2.transform(lambda s: (s - s.mean()) / (s.std(ddof=0) or np.nan))
    # Order by all three cell-type z-scores at once: each marker's centre of mass along the cell-type axis,
    # sum(i * w) / sum(w) with w = z - min(z) (>= 0), i = position in type_order. A marker high only in the first
    # type lands at 0, one split between the first two at 0.5, only the second at 1, ... so the plot reads as a diagonal.
    zref = d[d.method == S.METHOD_ORDER[1]].pivot(index="gene", columns="cell_type", values="z")[type_order]
    w = zref.sub(zref.min(axis=1), axis=0).fillna(0.0)
    centre = (w * np.arange(len(type_order))).sum(axis=1) / w.sum(axis=1).replace(0, np.nan)
    genes_ord = list(centre.fillna(centre.max()).sort_values(kind="stable").index)
    gi = {g: i for i, g in enumerate(genes_ord)}
    ti = {t: i for i, t in enumerate(type_order)}
    fig, axs = plt.subplots(2, 1, figsize=(INT_W, INT_ROW4_H),
                            layout="constrained", sharex=True, sharey=True)
    sc_ = None
    for ax, (k, method) in zip(axs, METHOD_KEYS):
        m = d[(d.method == method) & d.gene.isin(gi)]
        sc_ = ax.scatter(m.gene.map(gi), m.cell_type.map(ti), s=2 + 38 * m.frac_detected.fillna(0).to_numpy(),
                         c=m.z.to_numpy(), cmap="RdBu_r", vmin=-1.5, vmax=1.5, edgecolors="none")
        ax.set_xticks(range(len(genes_ord))); ax.set_xticklabels([names.get(g, g) for g in genes_ord], rotation=90)
        ax.tick_params(labelbottom=ax is axs[-1])  # gene labels once, under the bottom panel
        ax.set_yticks(range(len(type_order))); ax.set_yticklabels(type_order)
        ax.set_xlim(-.6, len(genes_ord) - .4); ax.set_ylim(len(type_order) - .4, -.6)  # first type on top
        ax.set_title(method, fontsize=FS + .5); ax.set_ylabel("DIA-NN cluster")
        compact_axes(ax)
    cb = fig.colorbar(sc_, ax=axs, fraction=.04, pad=.02, shrink=.6)
    cb.set_label("mean log$_2$ (z across clusters)", fontsize=FS); cb.ax.tick_params(labelsize=FS - 1)
    sv(fig, MAIN, f"Supp_SingleCell_{label}_S_celltype_dotplot", tight=False)
    return d


def assemble_pbmc_integrated(label, main_dir, saver):
    """Fig_SingleCell_{label}_Integrated -- row 1: A (IDs per cell) | B (completeness) | H (clusters); row 2: O (marker
    coverage); row 3: I (marker violins, JUMPlion clusters).
    The panels were drawn at sizes that make every row INT_W wide, so they are tiled at their own size (read back
    from the saved PNGs; needs "png" in style.formats); bold letters a-e are added in reading order."""
    stems = {"A": f"{label}_protein_A_ids_per_cell", "B": f"{label}_protein_B_completeness",
             "H": f"{label}_H_scanpy_clusters", "O": f"{label}_O_marker_coverage",
             "I": f"{label}_I_marker_violin"}
    paths = {k: Path(main_dir) / f"Fig_SingleCell_{s}.png" for k, s in stems.items()}
    if missing := [k for k, p in paths.items() if not p.exists()]:
        log.warning("integrated figure skipped: missing panel PNG(s) for %s (is \"png\" in style.formats, and did the "
                    "singlecell step run first?)", missing)
        return
    imgs = {k: plt.imread(p) for k, p in paths.items()}
    w = {k: im.shape[1] / saver.dpi for k, im in imgs.items()}
    h = {k: im.shape[0] / saver.dpi for k, im in imgs.items()}
    rows = [["A", "B", "H"], ["O"], ["I"]]
    row_w = [sum(w[k] for k in r) + INT_GAP * (len(r) - 1) for r in rows]
    row_h = [max(h[k] for k in r) for r in rows]
    total_w, total_h = max(row_w), sum(row_h)
    fig = plt.figure(figsize=(total_w, total_h))
    y, n = 0.0, 0
    for r, rw, rh in zip(rows, row_w, row_h):
        x = (total_w - rw) / 2
        for k in r:
            ax = fig.add_axes([x / total_w, 1 - (y + h[k]) / total_h, w[k] / total_w, h[k] / total_h])
            ax.imshow(imgs[k]); ax.axis("off")
            fig.text((x + .05) / total_w, 1 - (y + .05) / total_h, "abcde"[n], fontsize=9, fontweight="bold",
                     ha="left", va="top")
            x += w[k] + INT_GAP
            n += 1
        y += rh
    saver.save(fig, main_dir, f"Fig_SingleCell_{label}_Integrated", tight=False)


def panel_dotplot(summs, label, SUPP, sv, labels=None):
    """One column per (cell type, marker): a marker listed under several cell types repeats under each."""
    labels = labels or {}
    n_cols = max(d[["cell_type", "gene"]].drop_duplicates().shape[0] for d in summs.values())
    fig, axs = plt.subplots(1, 2, figsize=(max(4.6, .13 * n_cols * 2 + 1.2), 2.9), layout="constrained")
    for ax, (key, method) in zip(axs, METHOD_KEYS):
        d = summs[key]
        genes = list(dict.fromkeys(zip(d.cell_type, d.gene)))
        clusters = sorted(set(d.cluster), key=int)
        gi = {g: i for i, g in enumerate(genes)}
        ci = {c: i for i, c in enumerate(clusters)}
        sc_ = ax.scatter([gi[g] for g in zip(d.cell_type, d.gene)], [ci[c] for c in d.cluster],
                         s=2 + 38 * d.frac_detected.to_numpy(), c=d.z.to_numpy(), cmap="RdBu_r", vmin=-2, vmax=2,
                         edgecolors="none")
        ax.set_xticks(range(len(genes))); ax.set_xticklabels([labels.get(g, g) for _, g in genes], rotation=90, fontsize=5)
        ax.set_yticks(range(len(clusters))); ax.set_yticklabels(clusters, fontsize=5)
        ax.set_xlim(-.6, len(genes) - .4); ax.set_ylim(-.6, len(clusters) - .4)
        ax.set_xlabel(method, fontsize=5.5); ax.set_ylabel("Leiden cluster")
        # marker category separators
        pos = 0
        for ct, gs in d[["cell_type", "gene"]].drop_duplicates().groupby("cell_type", sort=False):
            ax.text(pos + (len(gs) - 1) / 2, len(clusters) - .35, ct.replace("/", "\n"), ha="center",
                    va="bottom", fontsize=4.5, color=S.NEUTRAL)
            pos += len(gs)
            ax.axvline(pos - .5, color=S.LIGHT, linewidth=.4)
    cb = fig.colorbar(sc_, ax=axs, fraction=.04, pad=.02, shrink=.6)
    cb.set_label("mean log$_2$ (z across clusters)", fontsize=5); cb.ax.tick_params(labelsize=4.5)
    fig.suptitle("Marker proteins by cluster · size = fraction of cells detected", fontsize=5.5)
    sv(fig, SUPP, f"Supp_SingleCell_{label}_I_scanpy_marker_dotplot")


def panel_preprocessing(centered, results, label, SUPP, sv):
    """Observed vs imputed values (median-centred log2) per method."""
    fig, axs = plt.subplots(1, 2, figsize=(4.4, 2.0), layout="constrained")
    for ax, (key, method) in zip(axs, METHOD_KEYS):
        obs_df, imp_df = results[key]["obs"], results[key]["imp"]
        o = obs_df.to_numpy(float)
        miss = ~np.isfinite(o)
        bins = np.linspace(np.nanpercentile(imp_df.to_numpy(float), .1), np.nanpercentile(imp_df.to_numpy(float), 99.9), 60)
        ax.hist(o[~miss], bins=bins, density=True, histtype="step", color=S.METHOD_COLORS[method], label="Observed")
        if miss.any():
            ax.hist(imp_df.to_numpy(float)[miss], bins=bins, density=True, histtype="step", color=S.NEUTRAL,
                    linestyle="--", label="Imputed")
        ax.set_xlabel("Median-centred log$_2$ intensity"); ax.set_ylabel("Density")
        ax.set_title(f"{method}\n{miss.sum():,} of {o.size:,} values imputed ({miss.mean() * 100:.1f}%)", fontsize=5.5)
        ax.legend(frameon=False, fontsize=5)
    sv(fig, SUPP, f"Supp_SingleCell_{label}_J_scanpy_preprocessing")


def panel_pca_diag(results, label, SUPP, sv):
    """a: variance ratio per PC (the elbow used to pick n_pcs); b/c: PC1 vs PC2 coloured by Leiden cluster."""
    fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.0), layout="constrained")
    for key, method in METHOD_KEYS:
        vr = np.asarray(results[key]["adata"].uns["pca"]["variance_ratio"]) * 100
        axs[0].plot(np.arange(1, len(vr) + 1), vr, "o-", color=S.method_color(method), label=method,
                    markersize=2.5, linewidth=.9)
    axs[0].set_xlabel("Principal component"); axs[0].set_ylabel("Variance explained (%)")
    axs[0].legend(frameon=False, fontsize=5)
    for ax, (key, method) in zip(axs[1:], METHOD_KEYS):
        a = results[key]["adata"]
        vr = a.uns["pca"]["variance_ratio"]
        lab = results[key]["labels"]
        for cl, col in _cluster_colors(lab).items():
            sel = lab == cl
            ax.scatter(a.obsm["X_pca"][sel, 0], a.obsm["X_pca"][sel, 1], s=5, color=col, linewidths=0,
                       label=cl, rasterized=True)
        ax.set_xlabel(f"PC1 ({vr[0] * 100:.1f}%)"); ax.set_ylabel(f"PC2 ({vr[1] * 100:.1f}%)")
        ax.set_title(method, fontsize=5.5)
        ax.legend(frameon=False, fontsize=FS, markerscale=1.5, loc="center left", bbox_to_anchor=(1.0, .5),
                  title="Cluster", title_fontsize=FS)
    sv(fig, SUPP, f"Supp_SingleCell_{label}_K_scanpy_PCA")


def panel_resolution(scan, results, chosen, label, SUPP, sv):
    """Leiden at several resolutions: number of clusters, silhouette on the PCs, and DIA-NN vs JUMPlion ARI."""
    from sklearn.metrics import adjusted_rand_score
    res = sorted(next(iter(scan.values())))
    fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.0), layout="constrained")
    ari = []
    for key, method in METHOD_KEYS:
        pcs = results[key]["adata"].obsm["X_pca"]
        axs[0].plot(res, [len(set(scan[key][r])) for r in res], "o-", color=S.method_color(method),
                    markersize=2.5, linewidth=.9, label=method)
        axs[1].plot(res, [silhouette(pcs, scan[key][r]) for r in res], "o-", color=S.method_color(method),
                    markersize=2.5, linewidth=.9, label=method)
    ari = [adjusted_rand_score(scan["diann"][r], scan["jumplion"][r]) for r in res]
    axs[2].plot(res, ari, "o-", color=S.NEUTRAL, markersize=2.5, linewidth=.9)
    for ax, yl in zip(axs, ("Leiden clusters", "Silhouette of clusters (PCs)", "ARI, DIA-NN vs JUMPlion")):
        ax.axvline(chosen, color=S.LIGHT, linewidth=.6, linestyle=":")
        ax.set_xlabel("Leiden resolution"); ax.set_ylabel(yl)
    axs[0].legend(frameon=False, fontsize=5)
    sv(fig, SUPP, f"Supp_SingleCell_{label}_L_scanpy_resolution")
    return pd.DataFrame([dict(resolution=r, method=m, n_clusters=len(set(scan[k][r])),
                              silhouette=silhouette(results[k]["adata"].obsm["X_pca"], scan[k][r]),
                              ari_diann_vs_jumplion=a)
                         for r, a in zip(res, ari) for k, m in METHOD_KEYS])


def top_markers(mk, top_n):
    """Top_n up-regulated (log2FC > 0) proteins per cluster by p-value -> list of (feature, cluster)."""
    out = []
    for cl in sorted(mk.cluster.unique(), key=int):
        d = mk[(mk.cluster == cl) & (mk.log2FC > 0)].sort_values("p").head(top_n)
        out += [(f, cl) for f in d.feature]
    return out


def panel_top_marker_heatmap(centered, results, mks, annots, top_n, label, SUPP, sv):
    """Per method: cluster-mean (observed) median-centred log2 of each cluster's top Wilcoxon markers,
    z-scored across clusters."""
    feats = {key: list(dict.fromkeys(f for f, _ in top_markers(mks[key], top_n))) for key, _ in METHOD_KEYS}
    h = max(2.4, .13 * max(len(v) for v in feats.values()) + .9)
    fig, axs = plt.subplots(1, 2, figsize=(5.2, h), layout="constrained")
    for ax, (key, method) in zip(axs, METHOD_KEYS):
        labels = results[key]["labels"]
        clusters = sorted(set(labels), key=int)
        m = centered[key].loc[feats[key]].T.groupby(labels).mean().reindex(clusters).T  # features x clusters
        z = m.sub(m.mean(axis=1), axis=0).div(m.std(axis=1, ddof=0).replace(0, np.nan), axis=0)
        im = ax.imshow(z.to_numpy(float), cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
        names = [str(annots[key].loc[f, "Genes"]).split(";")[0] if "Genes" in annots[key] and pd.notna(annots[key].loc[f, "Genes"])
                 else f for f in z.index]
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=4.5)
        ax.set_xticks(range(len(clusters))); ax.set_xticklabels(clusters, fontsize=5)
        ax.set_xlabel("Leiden cluster"); ax.set_title(method, fontsize=5.5)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    cb = fig.colorbar(im, ax=axs, fraction=.04, pad=.02, shrink=.5)
    cb.set_label(f"cluster mean log$_2$ (z); top {top_n} Wilcoxon markers / cluster", fontsize=5)
    cb.ax.tick_params(labelsize=4.5)
    sv(fig, SUPP, f"Supp_SingleCell_{label}_M_scanpy_top_markers")


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def run(cfg, layout, saver, resolve):
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    sc_cfg = cfg["singlecell"]
    s = dict(min_cell_frac=0.5, impute="left_shift", n_pcs=15, n_neighbors=10, resolution=0.5,
             min_obs_per_group=3, seed=0, resolutions=[0.2, 0.3, 0.5, 0.8, 1.0], top_n_markers=3)
    s.update(cfg.get("scanpy", {}))
    markers = collapse_shared_markers(sc_cfg.get("markers", {}))  # each marker once, under its cell type or the combination it is listed under
    label = sc_cfg.get("label", "PBMC")
    lvl = sc_cfg["levels"]["protein"]
    meta = pd.read_csv(resolve(sc_cfg["sample_meta_file"]))
    meta["Run"] = meta["Run"].astype(str)
    sv, T = saver.save, layout.table

    mats, annots = {}, {}
    for key, _ in METHOD_KEYS:
        mats[key], annots[key] = load_matrix(resolve(lvl["matrices"][key]), meta, lvl.get("id_col", "Protein.Group"))
    cells = [c for c in mats["diann"].columns if c in mats["jumplion"].columns]
    centered = {k: center_cells(mats[k][cells]) for k in mats}
    rng = np.random.default_rng(s["seed"])

    # One protein set for both methods, chosen from the ORIGINAL DIA-NN matrix (>= min_cell_frac of cells), so
    # JUMPlion's higher completeness cannot decide which proteins are clustered. Same cells, preprocessing,
    # number of PCs, graph and resolution; PCA / neighbours / UMAP / Leiden are fitted separately per method.
    common = centered["diann"].index[centered["diann"].notna().mean(axis=1) >= s["min_cell_frac"]]
    common = common.intersection(centered["jumplion"].index)
    results = {k: cluster_method(centered[k], s, rng, keep=common) for k in centered}
    for k, r in results.items():
        log.info("%s: %d proteins used (shared set), %d Leiden clusters", k, r["n_features"], len(set(r["labels"])))

    xy = results["jumplion"]["adata"].obsm["X_umap"]
    agree = dict(ari=adjusted_rand_score(results["diann"]["labels"], results["jumplion"]["labels"]),
                 nmi=normalized_mutual_info_score(results["diann"]["labels"], results["jumplion"]["labels"]))
    sil = {k: silhouette(r["adata"].obsm["X_pca"], r["labels"]) for k, r in results.items()}
    write_tsv(pd.DataFrame(dict(cell=cells, leiden_diann=results["diann"]["labels"],
                                leiden_jumplion=results["jumplion"]["labels"],
                                UMAP1_jumplion=xy[:, 0], UMAP2_jumplion=xy[:, 1],
                                UMAP1_diann=results["diann"]["adata"].obsm["X_umap"][:, 0],
                                UMAP2_diann=results["diann"]["adata"].obsm["X_umap"][:, 1])), T,
              f"leiden_clusters_{label}.tsv")
    write_tsv(pd.DataFrame([dict(method=m, n_proteins_used=results[k]["n_features"],
                                 n_clusters=len(set(results[k]["labels"])), silhouette_leiden=sil[k], **agree)
                            for k, m in METHOD_KEYS]), T, f"cluster_summary_{label}.tsv")

    panel_preprocessing(centered, results, label, layout.supp, sv)
    panel_pca_diag(results, label, layout.supp, sv)
    scan = {k: resolution_scan(r["adata"], s["resolutions"]) for k, r in results.items()}
    write_tsv(panel_resolution(scan, results, s["resolution"], label, layout.supp, sv), T,
              f"resolution_scan_{label}.tsv")

    summs, annot, mks = {}, {}, {}
    for key, method in METHOD_KEYS:
        mk = rank_markers(centered[key], results[key]["labels"], s["min_obs_per_group"])
        mks[key] = mk
        if len(mk):
            mk = mk.merge(annots[key][["Genes"]].reset_index(names="feature"), on="feature", how="left") \
                if "Genes" in annots[key] else mk
            write_tsv(mk.sort_values(["cluster", "p"]), T, f"cluster_markers_wilcoxon_{label}_{key}.tsv")
        rows = find_marker_rows(annots[key], [g for gs in markers.values() for g in gs])
        missing = [g for gs in markers.values() for g in gs if g not in rows]
        if missing:
            log.warning("%s: marker proteins not in matrix: %s", method, missing)
        summs[key] = marker_summary(centered[key], results[key]["labels"], rows, markers)
        write_tsv(summs[key], T, f"marker_summary_by_cluster_{label}_{key}.tsv")
        annot[key] = suggest_annotation(summs[key], list(sc_cfg.get("markers", {}))) if len(summs[key]) else pd.DataFrame()
        if len(annot[key]):
            write_tsv(annot[key], T, f"suggested_annotation_{label}_{key}.tsv")

    panel_umap_clusters(xy, results, agree, sil, label, layout.main, sv)
    panel_umap_clusters_diann(results["diann"]["adata"].obsm["X_umap"], results, agree, sil, label, layout.supp, sv,
                              xy_jumplion=xy)
    # Rows: the DIA-NN Leiden clusters; the same cells are summarised in both the DIA-NN and JUMPlion values
    d_clusters = np.asarray(results["diann"]["labels"])
    dp = panel_celltype_dotplot({k: mats[k][cells] for k in mats}, annots, d_clusters,
                                sorted(set(d_clusters), key=int),
                                list(dict.fromkeys(g for gs in sc_cfg.get("markers", {}).values() for g in gs)),
                                label, layout.supp, sv, names=sc_cfg.get("marker_labels"))
    write_tsv(dp, T, f"celltype_dotplot_{label}.tsv")
    if sc_cfg.get("markers"):  # % of cells quantified per marker, DIA-NN vs JUMPlion
        cov = panel_marker_coverage({k: mats[k][cells] for k in mats}, annots, sc_cfg["markers"], label, layout.main, sv,
                                    labels=sc_cfg.get("marker_labels"))
        write_tsv(cov, T, f"marker_coverage_{label}.tsv")
    if sc_cfg.get("violin_markers"):  # per-cluster intensities: reported vs recovered, for each method's clusters
        vmats, vnames = {k: mats[k][cells] for k in mats}, sc_cfg.get("marker_labels")
        vall = sc_cfg["violin_markers"]
        vmain = [g for g in sc_cfg.get("violin_markers_main", []) if g in vall]  # main-text figure (JUMPlion clusters)
        jl_clusters, dn_clusters = results["jumplion"]["labels"], results["diann"]["labels"]
        if vmain:
            panel_marker_violin(vmats, annots, jl_clusters, vmain, label, layout.main, sv, names=vnames,
                                ncols=len(vmain), cluster_source=S.METHOD_ORDER[1], stem="I_marker_violin",
                                prefix="Fig", figsize=(INT_W, INT_ROW_H), xlabel="Cluster", ylabel_once=True)
        # supplements, 4 panels per row: P = the JUMPlion-cluster violins not already in the main figure;
        # Q = all violin markers on the DIA-NN clusters
        vrest = [g for g in vall if g not in vmain]
        if vrest:
            vio = panel_marker_violin(vmats, annots, jl_clusters, vrest, label, layout.supp, sv, names=vnames,
                                      ncols=4, cluster_source=S.METHOD_ORDER[1], stem="P_marker_violin")
            write_tsv(vio, T, f"marker_violin_{label}.tsv")
        vio_d = panel_marker_violin(vmats, annots, dn_clusters, vall, label, layout.supp, sv, names=vnames,
                                    ncols=4, cluster_source=S.METHOD_ORDER[0], stem="Q_marker_violin_DIANN_clusters")
        write_tsv(vio_d, T, f"marker_violin_DIANN_clusters_{label}.tsv")
    assemble_pbmc_integrated(label, layout.main, saver)


def main(default_config, argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=default_config)
    ap.add_argument("--output-dir", help="Override output_dir from the config")
    args = ap.parse_args(argv)
    setup_logging()
    cfg = substitute_config(read_raw_config(args.config))
    resolve = PathResolver(cfg["_config_dir"])
    style_cfg = cfg.get("style", {})
    S.apply_style(style_cfg)
    saver = S.FigureSaver(style_cfg.get("formats", ["pdf", "png"]), style_cfg.get("dpi", 600))
    out_root = Path(args.output_dir) if args.output_dir else resolve(cfg.get("output_dir", "jumplion_figures_output"))
    layout = OutputLayout(out_root, f"singlecell_{cfg['singlecell'].get('label', 'dataset')}")  # same folder as the singlecell run (file names don't overlap)
    run(cfg, layout, saver, resolve)
    S.plt.close("all")
    return 0
