"""Unified visual style for all JUMPlion manuscript figures.

Every module takes its colors, labels, rcParams and saving logic from here, so
one edit in this file changes all three figure families consistently.

Semantic color system
---------------------
* Methods       : DIA-NN = blue, JUMPlion = red.
* Evidence tier : DIA-NN reported = the DIA-NN blue; JUMPlion evidence tiers are
                  warm colors (MS2 co-elution red, MS1 isotope peak yellow);
                  no qualifying peak = light grey.
* Species       : Human green, Yeast orange, E. coli purple (in every figure).
* Mapping gates : pass green, fail red.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
from matplotlib.patches import FancyBboxPatch, Patch

# ----------------------------------------------------------------------------
# Palette
# ----------------------------------------------------------------------------
NEUTRAL = "#4D4D4D"
LIGHT = "#BDBDBD"
REF_LINE = "#6E6E6E"

CYCLE = ["#4C78A8", "#E45756", "#59A14F", "#F28E2B", "#B07AA1",
         "#76B7B2", "#EDC948", "#9C755F", "#FF9DA7", "#BAB0AC"]

METHOD_ORDER = ["DIA-NN", "JUMPlion"]
METHOD_COLORS = {"DIA-NN": "#4C78A8", "JUMPlion": "#E45756"}

SPECIES_ORDER = ["HUMAN", "YEAST", "ECOLI"]
SPECIES_COLORS = {"HUMAN": "#1B9E77", "YEAST": "#D95F02", "ECOLI": "#7570B3"}
SPECIES_LABELS = {"HUMAN": "Human", "YEAST": "Yeast", "ECOLI": "E. coli"}

TIER_ORDER = ["reported", "ms2_coelution", "ms1_peak", "prediction_only"]
TIER_COLORS = {
    "reported": METHOD_COLORS["DIA-NN"],
    "ms2_coelution": "#E45756",
    "ms1_peak": "#DDAA33",
    "prediction_only": LIGHT,
}
TIER_LABELS = {
    "reported": "DIA-NN reported",
    "ms2_coelution": "MS2 co-elution",
    "ms1_peak": "MS1 isotope peak",
    "prediction_only": "No qualifying peak",
}
_TIER_ALIASES = {
    "reported": "reported", "dia-nn reported": "reported",
    "ms2_coelution": "ms2_coelution", "ms2 co-elution": "ms2_coelution",
    "ms1_peak": "ms1_peak", "ms1 peak": "ms1_peak", "ms1 isotope peak": "ms1_peak",
    "prediction_only": "prediction_only", "prediction only": "prediction_only",
    "no qualifying peak": "prediction_only",
}

ELIGIBILITY_COLORS = {
    "Eligible with both": METHOD_COLORS["DIA-NN"],
    "Newly DE-eligible after JUMPlion": "#F2A541",
    "DIA-NN-only eligible": "#8C8C8C",
    "Eligible with neither": "#D9D9D9",
}
TP_ROUTE_COLORS = {
    "DIA-NN TP": METHOD_COLORS["DIA-NN"],
    "Retained TP": "#9DB9D5",
    "FN -> TP": "#F2A541",
    "Newly DE-eligible -> TP": "#59A14F",
}
OUTCOME_COLORS = {
    "TP": "#59A14F", "FP": "#E15759", "FN": "#F28E2B",
    "TN": LIGHT, "wrong_direction": "#B07AA1",
}
POPULATION_COLORS = {
    "Shared DE-eligible": METHOD_COLORS["DIA-NN"],
    "Newly DE-eligible": "#F2A541",
    "Shared eligible": METHOD_COLORS["DIA-NN"],
    "Newly eligible": "#F2A541",
}
COMPLETENESS_COLORS = {
    "Complete": METHOD_COLORS["DIA-NN"],
    "Partially complete": "#F2A541",
    "Insufficient": LIGHT,
}
RECOVERY_STATE_COLORS = {
    "Remains insufficient": LIGHT,
    "Becomes sufficient": METHOD_COLORS["JUMPlion"],
}
GATE_COLORS = {"pass": "#59A14F", "fail": "#E15759"}
RECOVERY_COLORS = {"Recovered": METHOD_COLORS["JUMPlion"], "Unresolved": LIGHT}


def tier_key(x):
    s = str(x).strip()
    return _TIER_ALIASES.get(s.lower(), s)


def tier_label(x):
    return TIER_LABELS.get(tier_key(x), str(x))


def tier_color(x):
    return TIER_COLORS.get(tier_key(x), NEUTRAL)


def method_color(method):
    return METHOD_COLORS.get(method, NEUTRAL)


def species_label(sp):
    return SPECIES_LABELS.get(str(sp), str(sp))


def clabel(contrast):
    """'S2_vs_S1' -> 'S2/S1'."""
    return str(contrast).replace("_vs_", "/")


# ----------------------------------------------------------------------------
# rcParams
# ----------------------------------------------------------------------------
def apply_style(opts=None):
    o = opts or {}
    fs = float(o.get("font_size", 8))
    fam = o.get("font_family", ["Arial", "Helvetica", "DejaVu Sans"])
    if isinstance(fam, str):
        fam = [fam, "DejaVu Sans"]
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": list(fam),
        "font.size": fs,
        "axes.titlesize": fs + 1,
        "axes.labelsize": fs,
        "xtick.labelsize": fs - 1,
        "ytick.labelsize": fs - 1,
        "legend.fontsize": fs - 1,
        "legend.title_fontsize": fs - 1,
        "figure.titlesize": fs + 1,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.prop_cycle": cycler(color=CYCLE),
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "lines.linewidth": 1.3,
        "lines.markersize": 4,
        "patch.linewidth": 0.5,
        "legend.frameon": False,
        "errorbar.capsize": 3,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    })
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    logging.getLogger("fontTools").setLevel(logging.WARNING)


class FigureSaver:
    """Saves each figure in every configured format and records what was saved."""

    RASTER = {"png", "tif", "tiff", "jpg", "jpeg"}

    def __init__(self, formats=("pdf", "png"), dpi=600):
        self.formats = [f.lower().lstrip(".") for f in formats]
        self.dpi = int(dpi)
        self.saved = []

    def save(self, fig, directory, stem, tight=True):
        """tight=False preserves the figure's exact `figsize` in the saved file instead of the
        default (rcParams `savefig.bbox="tight"`) cropping to the actual content's bounding box.
        Needed for panels whose physical size must stay predictable from figsize alone -- e.g. ones
        tiled into a composite by reading saved-PNG pixel dimensions back -- since tight cropping
        shrinks the output whenever a panel has unused margin (no title, a shorter legend, etc.),
        silently decoupling the saved size from the requested figsize.

        Passing `bbox_inches=None` to `savefig` does NOT disable tight cropping -- it means "use
        rcParams['savefig.bbox']", which is set to "tight" above, so it would silently still crop.
        An explicit Bbox covering the whole figure is the only way to actually override that."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for fmt in self.formats:
            kw = {"dpi": self.dpi} if fmt in self.RASTER else {}
            if not tight:
                w, h = fig.get_size_inches()
                kw.update(bbox_inches=Bbox.from_bounds(0, 0, w, h), pad_inches=0)
            fig.savefig(directory / f"{stem}.{fmt}", **kw)
        plt.close(fig)
        self.saved.append((directory, stem))


# ----------------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------------
def ref_line(ax, value, orient="h", **kw):
    style = dict(ls="--", lw=0.8, color=REF_LINE, zorder=1)
    style.update(kw)
    return (ax.axhline if orient == "h" else ax.axvline)(value, **style)


def fig_legend(fig, handles, labels, ncol=2, y=1.02, **kw):
    style = dict(frameon=False, loc="upper center", bbox_to_anchor=(0.5, y))
    style.update(kw)
    return fig.legend(handles, labels, ncol=ncol, **style)


def line_handles(color_map, keys, labels=None, marker="o"):
    labels = list(labels) if labels is not None else list(keys)
    return [Line2D([], [], color=color_map[k], marker=marker, lw=1.3, ms=4) for k in keys], labels


def patch_handles(color_map, keys, labels=None):
    labels = list(labels) if labels is not None else list(keys)
    return [Patch(facecolor=color_map[k], edgecolor="none") for k in keys], labels


def method_handles():
    return line_handles(METHOD_COLORS, METHOD_ORDER, marker=None)


def color_boxplot(bp, categories, color_map):
    """Color each boxplot artist; `categories` follows box order."""
    for art, cat in zip(bp.get("boxes", []), categories):
        art.set_color(color_map[cat])
    for art, cat in zip(bp.get("medians", []), categories):
        art.set_color(color_map[cat])
        art.set_linewidth(1.3)
    doubled = [c for c in categories for _ in (0, 1)]
    for key in ("whiskers", "caps"):
        for art, cat in zip(bp.get(key, []), doubled):
            art.set_color(color_map[cat])


def boxplot_groups(ax, data, positions, categories, color_map, width=0.28):
    clean = []
    for v in data:
        a = np.asarray(v, float)
        clean.append(a[np.isfinite(a)])
    bp = ax.boxplot(clean, positions=positions, widths=width, showfliers=False,
                    manage_ticks=False)
    color_boxplot(bp, categories, color_map)
    return bp


def violin_box(ax, groups, colors, positions=None, width=0.8,
               max_kde_points=200_000, seed=1):
    """Width-scaled violin (KDE fit on a subsample for speed) + full-data boxplot."""
    rng = np.random.default_rng(seed)
    positions = (np.arange(len(groups), dtype=float) if positions is None
                 else np.asarray(positions, float))
    for pos, vals, col in zip(positions, groups, colors):
        v = np.asarray(vals, float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        kv = v if v.size <= max_kde_points else rng.choice(v, max_kde_points, replace=False)
        if kv.size > 1 and np.ptp(kv) > 0:
            vp = ax.violinplot([kv], positions=[pos], widths=width, showextrema=False)
            for body in vp["bodies"]:
                body.set_facecolor(col)
                body.set_edgecolor(NEUTRAL)
                body.set_linewidth(0.5)
                body.set_alpha(0.85)
        ax.boxplot([v], positions=[pos], widths=width * 0.2, showfliers=False,
                   patch_artist=True, manage_ticks=False,
                   boxprops=dict(facecolor="white", edgecolor=NEUTRAL, linewidth=0.6),
                   medianprops=dict(color="black", linewidth=1.0),
                   whiskerprops=dict(color=NEUTRAL, linewidth=0.6),
                   capprops=dict(color=NEUTRAL, linewidth=0.6))
    ax.set_xticks(positions)
    ax.set_xlim(positions.min() - 0.6, positions.max() + 0.6)


def heatmap(ax, mat, row_labels, col_labels, vmin=-1, vmax=1, cmap="RdBu_r",
            cbar_label=None, annotate=True, fmt="{:.2f}", colorbar=True,
            annot_size=None, tick_size=None, xlabel_rotation=45, xlabel_ha="right"):
    mat = np.asarray(mat, float)
    im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=xlabel_rotation, ha=xlabel_ha)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0, labelsize=tick_size or mpl.rcParams["font.size"])
    if annotate:
        span = max(abs(vmin), abs(vmax))
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, fmt.format(mat[i, j]), ha="center", va="center",
                            fontsize=annot_size or (mpl.rcParams["font.size"] - 2),
                            color="white" if abs(mat[i, j]) > 0.6 * span else "black")
    if colorbar:
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.outline.set_visible(False)
    if cbar_label:
        cb.set_label(cbar_label)
    return im


def annotate_bars(ax, bars, texts, pad_frac=0.015, fontsize=None):
    ymax = max([b.get_height() for b in bars] + [1])
    fs = fontsize if fontsize is not None else mpl.rcParams["font.size"] - 1
    for bar, t in zip(bars, texts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + pad_frac * ymax,
                t, ha="center", va="bottom", fontsize=fs)


def panel_letter(ax, letter):
    ax.set_title(letter, loc="left", fontweight="bold")


class Flowchart:
    """Shared flowchart builder so every workflow panel looks identical.

    One data unit = ``INCH_PER_UNIT`` inches, so fonts, box heights, line widths and
    arrow gaps are physically the same in every workflow figure. Every box has the same
    height (fits two text lines) regardless of how many lines it holds; widths come
    from a small set of standard widths.
    """
    INCH_PER_UNIT = 0.42
    BOX_H1 = 0.6          # one standard height for all boxes (one line of text)
    BOX_H2 = 0.9          # one standard height for all boxes (two lines of text)
    ROW = 1.4            # vertical pitch between chained boxes
    BRANCH_ROW = 1.8     # vertical pitch across a yes/no branch (room for labels)
    WIDE, MID, NARR = 4.0, 3.4, 2.8 # standard widths: main chain / branch boxes
    TINT = 0.45          # fill opacity for colored outcome boxes
    BOXSTYLE = "round,pad=0.04,rounding_size=0.12"

    def __init__(self, xlim, ylim, inch_per_unit=None):
        """inch_per_unit overrides the class default for just this instance -- e.g. to shrink/grow
        one specific workflow panel to fit a target layout, without affecting any other
        Flowchart() call elsewhere, which keeps using the class default."""
        if inch_per_unit is not None:
            self.INCH_PER_UNIT = inch_per_unit
        (x0, x1), (y0, y1) = xlim, ylim
        self.fig, self.ax = plt.subplots(figsize=((x1 - x0) * self.INCH_PER_UNIT,
                                                  (y1 - y0) * self.INCH_PER_UNIT))
        self.fig.subplots_adjust(0, 0, 1, 1)
        self.ax.set_xlim(x0, x1); self.ax.set_ylim(y0, y1)
        self.ax.set_aspect("equal"); self.ax.axis("off")

    def box(self, x, y, text, w=None, h=None, face="white"):
        w, h = w or self.WIDE, h or self.BOX_H2
        fill = "white" if face == "white" else to_rgba(face, self.TINT)
        self.ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=self.BOXSTYLE,
                                         facecolor=fill, edgecolor=NEUTRAL, linewidth=0.9, zorder=2))
        self.ax.text(x, y, text, ha="center", va="center", linespacing=1.25, zorder=3)
        return dict(x=x, y=y, w=w, top=y + h / 2 + 0.04, bottom=y - h / 2 - 0.04)

    def container(self, x, top, bottom, title, w, title_x=None):
        """Light dashed group frame with a title (used to group sequential criteria). `title_x`: where
        the title is centred (default: the frame's own centre `x`)."""
        self.ax.add_patch(FancyBboxPatch((x - w / 2, bottom), w, top - bottom, boxstyle=self.BOXSTYLE,
                                         facecolor=to_rgba(LIGHT, 0.12), edgecolor=REF_LINE,
                                         linewidth=0.8, linestyle=(0, (3, 2)), zorder=0.5))
        self.ax.text(x if title_x is None else title_x, top - 0.38, title, ha="center", va="center")
        return dict(x=x, w=w, top=top + 0.04, bottom=bottom - 0.04)

    def arrow(self, src, dst, label=None, dx=0.0):
        """Arrow from bottom of `src` to top of `dst`; `dx` offsets the start for branches."""
        x1, y1, x2, y2 = src["x"] + dx, src["bottom"], dst["x"], dst["top"]
        self.ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                         arrowprops=dict(arrowstyle="-|>", lw=0.9, color=NEUTRAL,
                                         mutation_scale=8, shrinkA=0, shrinkB=0))
        if label:
            side = np.sign(x2 - x1) or 1
            self.ax.text((x1 + x2) / 2 + 0.28 * side, (y1 + y2) / 2 + 0.12, label,
                         ha="left" if side > 0 else "right", va="center", color=NEUTRAL)


def ecdf_xy(values):
    x = np.asarray(values, float)
    x = np.sort(x[np.isfinite(x)])
    return x, (np.arange(1, x.size + 1) / x.size if x.size else x)
