"""Stage key "generalization" (see make_figures.py): a lean, growable performance analysis
for datasets that don't use the rest of performance.py's Wang-HYE-specific ground-truth panels (expected
log2FC, TP/FP calls, etc) -- unlike that module, nothing here assumes a specific benchmark design.
Currently the 30-run cumulative completeness curve, the completeness-by-method bars, sensitivity/
FDR, and species-split log2FC error/CV box grids (all combined across every dataset into one
figure each -- see below), reusing performance.py's own loading/computation/plotting code so the
figures stay numerically identical to their full-pipeline counterparts; more such dataset-agnostic
analyses are meant to join this module over time. compute_cv_deltafc still runs here too (its
partial_cv/partial_fc/restored_cv/restored_fc feed the box grids and their own written tables),
but its own plot (panel_cv_deltafc, Fig_Generalization_E_quantitative_fidelity) is not drawn for
generalization -- only the full run() pipeline's own Main E still shows it.

Every dataset's output lands in the SAME shared main/supp/table (see make_figures.py's
UNIFIED_DATASET_STAGES), not one subfolder per dataset, so each dataset's files are disambiguated
by a "_<dataset>" filename suffix instead. Table output always goes to table/, matching every
other stage (tables aren't Main/Supp-split in this codebase); the combined (all-datasets) figures
below follow COMBINED_CONTRASTS or go to a fixed destination of their own, not per-dataset
"gradient" routing.

Five lettered panels (A-E, no F -- see below) are combined across datasets rather than drawn once
per dataset, grouped below by how they're computed (unlike performance.py's own Main A-D, this
lettering has no gaps -- there's no representative-examples "C" slot to skip here, so
completeness-by-method's _complete_only variant takes letter C and the combined box grid takes E):
  - A -- Coverage curve: instead of one Fig_Generalization_A_coverage per dataset, every dataset's
    curve is its own subplot in ONE combined Supp_Generalization_A_coverage_all_datasets figure,
    always in Supp (pools every contrast, so there's no per-contrast version of this one).
  - B/C/D -- Completeness-by-method and sensitivity/FDR: instead of one
    Fig_Generalization_{B,D}_... per dataset (which would show every one of that dataset's
    contrasts), these are built once per contrast in COMBINED_CONTRASTS, each using only that
    contrast's rows from every dataset, with dataset name standing in for "contrast" on the x-axis
    -- i.e. comparing that one contrast across datasets, not comparing contrasts within one
    dataset. B (completeness-by-method, stacked) is the one exception to "one combined figure per
    COMBINED_CONTRASTS entry": both entries' stacked bars are drawn as sibling subplots in ONE
    single Supp_Generalization_B_completeness_by_method_all_datasets figure instead
    (panel_completeness_by_method_grid), sharing one legend, regardless of either entry's own
    Main/Supp destination (see BY_METHOD_DEST) -- unlike every other panel here, this one draws its
    own stacked-by-method bars directly (not via performance.py's panel_completeness_by_method,
    which is shared with the full run() pipeline's own single-contrast Main B and has no multi-
    subplot form of its own). C and D DO still follow "one combined figure per COMBINED_CONTRASTS
    entry": each entry also names which destination (Main or Supp) its own C/D (and E, below)
    figures go to, independent of gradient; their "Fig_"/"Supp_" stem prefix follows that
    destination. Completeness-by-method's "_complete_only" figure (C) is the same data as B, but
    just the Complete state's percentage as one plain (not stacked) bar per method, since the full
    stacked version makes "how complete" harder to read off directly than a boundary within the
    stack. Sensitivity/FDR (D) uses panel_sensitivity_fdr_bar here, not panel_sensitivity_fdr --
    bars (dataset is a category, not an ordered sequence a line should connect) and only the
    partially-complete row (both methods); the restored/insufficient row is dropped from this
    combined panel entirely (its TSV is still written, just not plotted here).
  - E -- log2FC error / CV box grid: like completeness-by-method/sensitivity-FDR above, one
    combined {Fig,Supp}_Generalization_E_log2FC_error_CV_box_all_datasets_<contrast> figure per
    COMBINED_CONTRASTS entry (see panel_log2fc_error_cv_box_grid) -- one column per dataset, two
    rows (log2FC error on top, CV on bottom), each with separate Human/Yeast/E. coli boxes, now
    comparing DIA-NN vs JUMPlion for the partially-complete stratum only (paired boxes per
    species, by method color). The restored/insufficient stratum -- JUMPlion-only, no DIA-NN
    counterpart to compare against -- is its own separate
    {Fig,Supp}_Generalization_E_log2FC_error_CV_box_restored_all_datasets_<contrast> figure instead
    (panel_log2fc_error_cv_box_grid_restored), not combined into the same one; it carries its own
    one-entry legend, since (unlike its sibling) it isn't fed into assemble_integrated_figure to
    borrow one from. log2FC error is the signed delta_FC (not its absolute value), with a zero
    reference line; CV is a percentage. Both of E's own figures share one dataset-name title row
    (top only) and one set of species x-tick labels (bottom only) across their two rows, instead of
    being two separate figures that would otherwise each repeat both. Both rows are filtered to
    that entry's one named contrast per dataset rather than pooling every contrast, and the whole
    figure's destination (Main/Supp) follows that entry's destination.
run() is called once per dataset (see make_figures.py's per-dataset loop) and only accumulates
into the relevant module-level cache each time (via _accumulate_and_maybe_combine, whose cached
values are plain DataFrames, for all but the coverage curve -- its cache also carries
non-DataFrame scalars, so it's handled separately); each combined figure is actually built on
whichever dataset's call happens to complete its (contrast's) set (via _all_datasets, which
make_figures.py injects), not necessarily the last one in config order.

Once a COMBINED_CONTRASTS entry's own C, D and E panels have all been built (normally in the same
run() call, since their caches accumulate the same all_datasets set in parallel -- tracked via
built_integrated in run()), they're tiled into that entry's own integrated composite (see
assemble_integrated_figure): row 1 C | D (sharing one method legend), row 2 E -- lettered a-e (a=C,
b/c=D's sensitivity/FDR subplots, d/e=E's log2FC-error/CV rows), one letter per visual panel rather
than one per saved PNG. Every COMBINED_CONTRASTS entry gets this treatment now, not just one
designated "main" contrast -- the Main entry's composite keeps the plain name
Fig_Generalization_Integrated, while each Supp entry's is named Supp_Generalization_Integrated_<contrast>.
Unlike every other combined panel above, C/D/E's calls that feed these composites render at a
smaller native size (MAIN_PANEL_SIZE/MAIN_INNER_GAP, or the matching col_width for E) so each
composite already comes out at MAX_INTEGRATED_WIDTH without assemble_integrated_figure needing to
scale anything down afterward. B itself is excluded from every composite -- it has its own single
combined Supp figure instead (see panel_completeness_by_method_grid above).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import transforms as mtransforms

from . import diff_expression as DE
from . import performance as P
from .common import write_tsv

plt = P.plt
Patch = P.Patch
S = P.S

log = logging.getLogger("jumplion_figures.generalization")

# Which contrasts the combined completeness-by-method/sensitivity-FDR panels compare across
# datasets, and where each one's figure goes -- "main" or "supp" (see the module docstring above).
COMBINED_CONTRASTS = [
    ("B_vs_C", "main"),
    ("A_vs_B", "supp"),
]

# One exception to COMBINED_CONTRASTS' own per-entry destination, specific to the stacked
# completeness-by-method panel (not shared with its "_complete_only" sibling, sensitivity/FDR, or
# the box grids, which all follow COMBINED_CONTRASTS as-is): both entries' stacked bars are drawn
# as ONE subplot each in a single combined figure (see panel_completeness_by_method_grid), which
# always goes to Supp as a whole, regardless of either entry's own "main"/"supp" destination.
BY_METHOD_DEST = "supp"

# Every COMBINED_CONTRASTS entry gets its own integrated composite (see assemble_integrated_figure)
# now, not just one designated "main" contrast -- MAX_INTEGRATED_WIDTH is shared by all of them, so
# every composite comes out the same physical width/proportions regardless of which contrast or
# Main/Supp destination it's built from.
MAX_INTEGRATED_WIDTH = 4.5  # inches -- each integrated composite's target width; same as PVP_WIDTH, so the
                           # plain C/D/E composite matches the precursor-vs-protein one. CAUTION: 4.8in
                           # (tried earlier, before 6.0in) already scaled C/D down enough (to
                           # ~1.7in square) that the longer dataset_label() names ("Astral-15 min",
                           # "timsTOF-15 min", ...) clipped off the left edge of the saved PNG (C's
                           # own y-axis label and the leftmost rotated x-tick label on both C and
                           # D); 3.6in scales them down further still (to ~1.3in square), so that
                           # same clipping is expected to return, likely worse.

# This module's own default panel size, for every panel here -- kept as its own constant, not
# reused from P.PANEL_AB_SIZE, so a future change to one doesn't silently resize the other --
# they're numerically equal right now, but that's incidental.
PANEL_SIZE = (2.5, 2.5)

# C/D/E's own rotated x-tick labels (dataset names, or E's species names) slightly smaller than
# P.compact_axes' own shared PANEL_LABEL_SIZE (6, used for y-tick numbers and both axis labels) --
# applied via a second ax.tick_params(axis="x", ...) call AFTER P.compact_axes (whose own
# tick_params call sets labelsize on both axes and would otherwise overwrite this).
X_TICK_LABEL_SIZE = 5

# panel_sensitivity_fdr_bar's (D's) own axes geometry -- each of its 2 subplots (sensitivity, FDR)
# gets exactly panel_completeness_complete_only's (C's) own axes width
# (_axes_geometry(PANEL_SIZE[0])'s own axes_w), so a bar of the same width fraction renders at the
# same absolute (inch) width in both panels, and both panels' axes extents match when
# assemble_integrated_figure tiles them side by side. Built from
# manual fig.add_axes placement (not plt.subplots' wspace, which would shrink both axes to make
# room for the gap instead of only adding one) -- see panel_sensitivity_fdr_bar. PANEL_D_SIZE's
# width is therefore not an exact "2x PANEL_SIZE[0]" multiple, just close to it (the
# original rationale -- "D is roughly twice as wide as a single-axes panel" -- still holds).
#
# D's own left margin (before its first/sensitivity subplot) uses PANEL_D_INNER_GAP, not C's own
# (much larger) left margin -- the same amount of room that comfortably fits the FDR subplot's own
# y-axis label+ticks in the gap between the two subplots is enough for sensitivity's, and reusing
# it here (rather than a separate, bigger margin) is what lets assemble_integrated_figure's C-to-D
# gap and this gap end up visually the same size (see ROW1_GAP below) -- all three row-1 panels
# (C, sensitivity, FDR) evenly spaced.
#
# Both helpers take a panel WIDTH (not PANEL_AB_SIZE[0] baked in) so a caller that needs a
# smaller rendered panel (MAIN_PANEL_SIZE/MAIN_INNER_GAP below, for Fig_Generalization_Integrated
# itself) can get panel_completeness_complete_only and panel_sensitivity_fdr_bar to render
# natively at that size -- correctly proportioned, matching each other, built at the right
# resolution from the start -- instead of generating them at the default size and shrinking the
# saved image afterward, which would soften them and, if done unevenly, reintroduce the text-size
# mismatches earlier fixes here resolved.
def _axes_geometry(panel_w):
    axes_w = (P.INTEGRATED_RIGHT - P.INTEGRATED_LEFT) * panel_w
    right_margin = (1 - P.INTEGRATED_RIGHT) * panel_w
    return axes_w, right_margin


def _panel_d_size(panel_w, panel_h, inner_gap):
    axes_w, right_margin = _axes_geometry(panel_w)
    return (inner_gap + axes_w + inner_gap + axes_w + right_margin, panel_h)


PANEL_D_INNER_GAP = 1.2  # inches, between the sensitivity and FDR subplots, and before sensitivity
PANEL_D_SIZE = _panel_d_size(PANEL_SIZE[0], PANEL_SIZE[1], PANEL_D_INNER_GAP)

# C's and D's shared axes top edge (figure-fraction) -- one constant, not each panel's own inline
# literal, so C and D always land on the same top margin: both panels' own n= annotations sit just
# above this same edge (rotation=20, va="bottom"), so they need the same headroom to avoid getting
# clipped by the saved PNG's own top edge, whichever panel size/margins mode is in play.
CD_TOP = .85

# C, D and E's own calls that feed each COMBINED_CONTRASTS entry's integrated composite (see
# run()) render NATIVELY at a size that already sums to MAX_INTEGRATED_WIDTH, rather than generated
# at their usual (larger) default size
# and scaled down afterward in assemble_integrated_figure -- post-hoc image scaling would soften
# every panel and, were the scale factor to ever differ between rows, reintroduce the text-size
# mismatches earlier fixes here resolved.
#
# C/D's own small-panel margins (MAIN_PANEL_SIZE/MAIN_INNER_GAP below) are ABSOLUTE inches, not a
# scaled-down fraction of PANEL_SIZE's own P.INTEGRATED_LEFT/RIGHT/BOTTOM margins -- a fraction
# shrinks right along with the panel, and at MAX_INTEGRATED_WIDTH values small enough to matter
# (tried: 4.8in, then 3.6in) that shrank the margin below what a y-axis label or a rotated
# dataset-name tick label actually needs, clipping it off the left edge of the saved PNG. A fixed
# absolute margin doesn't shrink, whatever MAX_INTEGRATED_WIDTH is asked for.
Y_LABEL_MARGIN = 0.4  # one y-axis label (e.g. "Completely quantified (%)") + its tick numbers
X_LABEL_MARGIN = 0.34  # grouped x ticks below the x-axis: number row + underline + instrument name
DECOR_MARGIN = 0.10    # purely decorative margin (a panel's trailing edge -- no label there)
MAIN_PANEL_HEIGHT = 1.5  # inches -- C/D's own height when feeding an integrated composite

# C's one column of axes and each of D's two subplots' axes all share this one width (MAIN_AXES_W)
# -- see panel_sensitivity_fdr_bar's own docstring on why -- solved from row 1's total target
# width: C (Y_LABEL_MARGIN + MAIN_AXES_W + DECOR_MARGIN) + ROW1_GAP (-DECOR_MARGIN, cancelling
# C's own trailing margin -- see ROW1_GAP's own note below) + D (Y_LABEL_MARGIN + MAIN_AXES_W +
# MAIN_INNER_GAP + MAIN_AXES_W + DECOR_MARGIN) == MAX_INTEGRATED_WIDTH, i.e.
# 3*Y_LABEL_MARGIN + 3*MAIN_AXES_W + DECOR_MARGIN == MAX_INTEGRATED_WIDTH (MAIN_INNER_GAP is just
# Y_LABEL_MARGIN again -- D's own gap before FDR needs the same one-y-label room as every other
# margin here).
# UPDATE: C and D's trailing margin is now MAIN_UNIT_MARGIN (room for the "min" unit label right of
# each panel's last grouped tick label -- see grouped_dataset_ticks), so C and D no longer overlap
# (MAIN_ROW1_GAP = 0) and 3*Y_LABEL_MARGIN + 3*MAIN_AXES_W + 2*MAIN_UNIT_MARGIN == MAX_INTEGRATED_WIDTH.
MAIN_UNIT_MARGIN = 0.1
MAIN_ROW1_GAP = 0.0
MAIN_AXES_W = (MAX_INTEGRATED_WIDTH - 3 * Y_LABEL_MARGIN - 2 * MAIN_UNIT_MARGIN) / 3
MAIN_PANEL_SIZE = (Y_LABEL_MARGIN + MAIN_AXES_W + MAIN_UNIT_MARGIN, MAIN_PANEL_HEIGHT)
MAIN_INNER_GAP = Y_LABEL_MARGIN

# assemble_integrated_figure_precursor_vs_protein's own row 1 is C2 (panel_completeness_precursor_
# vs_protein, 2 subplots: precursor | protein) + D2 (a dedicated "Protein"-titled variant of
# panel_sensitivity_fdr_bar's own 2 subplots: sensitivity | FDR) -- 4 subplots total, all sharing
# ONE common axes width (PVP_AXES_W) so every bar (C2's completeness bars, D2's sensitivity/FDR
# bars) renders at the SAME absolute width, solved the same way MAIN_AXES_W itself was (see its
# own note above) from row 1's total target width: C2 (Y_LABEL_MARGIN + axes_w + PVP_INNER_GAP +
# axes_w + DECOR_MARGIN) + ROW1_GAP (-DECOR_MARGIN) + D2 (Y_LABEL_MARGIN + axes_w + MAIN_INNER_GAP
# + axes_w + DECOR_MARGIN) == MAX_INTEGRATED_WIDTH, i.e. 3*Y_LABEL_MARGIN + 4*axes_w +
# PVP_INNER_GAP + DECOR_MARGIN == MAX_INTEGRATED_WIDTH (MAIN_INNER_GAP is Y_LABEL_MARGIN again, same
# as MAIN_AXES_W's own derivation; PVP_INNER_GAP is DECOR_MARGIN again -- C2's own protein-side
# subplot needs no y-label room, same "purely decorative" reasoning as DECOR_MARGIN itself).
PVP_INNER_GAP = DECOR_MARGIN
PVP_FDR_YLIM = (0, 22)  # D2's (protein) empirical-FDR y-axis range, in % -- fixed, not data-matched
PVP_WIDTH = 4.5 # inches -- this composite's own target width (independent of MAX_INTEGRATED_WIDTH)
# C2's and D2's own trailing margin -- wider than DECOR_MARGIN to leave room for the "min" unit
# label written right of each panel's last tick label (see grouped_dataset_ticks's `unit`). With
# it, C2 and D2 no longer overlap (PVP_ROW1_GAP = 0, not ROW1_GAP's -DECOR_MARGIN), so:
# 3*Y_LABEL_MARGIN + 4*axes_w + PVP_INNER_GAP + 2*PVP_UNIT_MARGIN == PVP_WIDTH.
PVP_UNIT_MARGIN = 0.12
PVP_ROW1_GAP = 0.0
PVP_AXES_W = (PVP_WIDTH - 3 * Y_LABEL_MARGIN - 2 * PVP_UNIT_MARGIN - PVP_INNER_GAP) / 4
# Absolute top margin shared by C2 and D2 when feeding this composite: room for the rotated n=
# annotation plus each subplot's own title (pad=14pt) only -- NOT the shared method legend, which
# the OUTER composite draws itself (legend=False) instead of either panel's own standalone/plain
# default.
PVP_TOP_MARGIN = 0.34

# ROW1_GAP (the blank space between C's and D's own saved images) is negative -- it exactly
# cancels out C's own trailing blank margin (DECOR_MARGIN), which would otherwise stack on top of
# D's own leading margin (MAIN_INNER_GAP, sized to fit one subplot's y-axis label) and make the
# C-to-D gap visibly larger than the gap D already uses between its own two subplots. With it
# cancelled, the two gaps are both effectively MAIN_INNER_GAP, so C, D's sensitivity subplot and
# D's FDR subplot end up evenly spaced.
ROW1_GAP = -DECOR_MARGIN

# E's own y-axis labels ("log$_2$FC error", "CV (%)") are shorter than C/D's, so a smaller
# absolute left margin than Y_LABEL_MARGIN is enough; same ABSOLUTE-not-fraction reasoning as
# MAIN_PANEL_SIZE above -- see panel_log2fc_error_cv_box_grid's own `left_margin` parameter.
E_LEFT_MARGIN = 0.4

# E's own top/bottom/hspace fractions -- named constants (not inline literals in
# panel_log2fc_error_cv_box_grid's own subplots_adjust call) so assemble_integrated_figure can
# derive exactly where the CV row (E's row 1) starts, for its own "e" sub-panel label, using the
# same GridSpec math matplotlib itself uses to split rows: each row's own height fraction is
# (E_TOP - E_BOTTOM) / (nrows + (nrows - 1) * E_HSPACE), nrows=2 here.
E_TOP, E_BOTTOM, E_HSPACE = .88, .12, .12
E_RIGHT, E_WSPACE = .98, .15    # the E grid's right margin / column spacing (also used to place its column letters)
E_BOTTOM_COMPACT = .12          # E_BOTTOM when the x tick labels carry a second line (n=)

_coverage_cache = {}  # dataset name -> (curve, n_runs, n_total, pad), accumulated across run() calls
_completeness_caches = {c: {} for c, _ in COMBINED_CONTRASTS}  # contrast -> dataset name -> (coverage, completeness_transition)
_sensitivity_caches = {c: {} for c, _ in COMBINED_CONTRASTS}  # contrast -> dataset name -> (partial_perf, rec_perf)

# Species-split log2FC error/CV box grid (one combined figure, log2FC error on top / CV on bottom
# -- see panel_log2fc_error_cv_box_grid): like _completeness_caches/_sensitivity_caches above,
# built once per entry in COMBINED_CONTRASTS, each filtered to that one contrast per dataset
# rather than pooled across all of a dataset's contrasts. Unlike those two, each dataset's frames
# are kept separate (list of entries, not a pd.concat), so this uses its own per-contrast cache
# rather than reusing _accumulate_and_maybe_combine's concatenation -- each entry is itself a
# (fc_tuple, cv_tuple) pair, one per dataset.
_volcano_caches = {c: {} for c, _ in COMBINED_CONTRASTS}  # contrast -> dataset name -> merged pair table
_box_grid_caches = {c: {} for c, _ in COMBINED_CONTRASTS}  # contrast -> dataset name -> ((partial_fc, restored_fc), (partial_cv, restored_cv))
# Cached tables are already filtered to their contrast and relabeled contrast=dataset name -- see
# _accumulate_and_maybe_combine.


def _accumulate_and_maybe_combine(cache, key, all_datasets, tables):
    """Stores `tables` (a tuple of DataFrames for this one dataset, already filtered to one
    contrast and relabeled contrast=`key`) into `cache[key]`. Once `cache` covers every name in
    `all_datasets`, returns the per-table concatenation across all of them (in `all_datasets`
    order), clearing `cache`; otherwise returns None (not yet complete)."""
    cache[key] = tables
    if not set(all_datasets) <= set(cache):
        return None
    combined = tuple(pd.concat([cache[n][i] for n in all_datasets], ignore_index=True)
                     for i in range(len(tables)))
    cache.clear()
    return combined


def _combined_stem(base, contrast, dest_attr):
    return f"{'Fig' if dest_attr == 'main' else 'Supp'}_Generalization_{base}_all_datasets_{contrast}"


# Display labels for this 4-dataset config's own internal dataset keys -- figures only (titles,
# x-tick categories in the "_all_datasets" combined panels below); the keys themselves (cache
# keys, "_<dataset>" filename suffixes, the config's own "datasets" entries) are untouched.
DATASET_ORDER = ["astral_5m", "astral_15m", "ultra_5m", "ultra_15m"]
DATASET_LABELS = {
    "astral_15m": "Astral, 15 min",
    "astral_5m": "Astral, 5 min",
    "ultra_15m": "timsTOF, 15 min",
    "ultra_5m": "timsTOF, 5 min",
}


def dataset_label(name):
    return DATASET_LABELS.get(name, name)


def grouped_dataset_ticks(ax, contrasts, unit=False):
    """Two-level x tick labels instead of rotated "Astral, 15 min": bare gradient numbers ("5",
    "15") under each tick, a short underline spanning each run of consecutive same-instrument
    datasets, and the instrument name ("Astral", "timsTOF") centered below that line. Splits
    dataset_label() on its last ", " (or "-" for an "Astral-15 min" style label) and drops the
    trailing " min". `unit`: also write "min" just right of the last tick label (needs
    ~PVP_UNIT_MARGIN of free room to the axes' right)."""
    parts = [dataset_label(c).rsplit(", " if ", " in dataset_label(c) else "-", 1) for c in contrasts]
    ax.set_xticks(range(len(contrasts)))
    ax.set_xticklabels([p[-1].removesuffix(" min") for p in parts], rotation=0, ha="center")
    groups = {}
    for i, p in enumerate(parts):
        groups.setdefault(p[0] if len(p) > 1 else "", []).append(i)
    line_trans = mtransforms.offset_copy(ax.get_xaxis_transform(), fig=ax.figure, y=-13, units="points")
    for name, idx in groups.items():
        if name:
            ax.plot([min(idx) - .3, max(idx) + .3], [0, 0], transform=line_trans, color="black",
                    linewidth=.5, clip_on=False, solid_capstyle="butt")
            ax.annotate(name, xy=(sum(idx) / len(idx), 0), xycoords=("data", "axes fraction"),
                        xytext=(0, -15), textcoords="offset points", ha="center", va="top",
                        fontsize=X_TICK_LABEL_SIZE)
    if unit:
        ax.annotate("min", xy=(len(contrasts) - 1 + .3, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -6.5), textcoords="offset points", ha="left", va="top",
                    fontsize=X_TICK_LABEL_SIZE)


def panel_coverage_grid(entries, SUPP, sv, stem="Supp_coverage_all_datasets", ncols=2, total_width=None):
    """One combined figure, one subplot per dataset -- entries: [(dataset_name, curve, n_runs,
    n_total, pad), ...], in display order. Each subplot is `P._draw_coverage`'s own content, titled
    with its dataset name (the per-subplot "Total precursors, n=..." becomes the title's second
    line instead, since the dataset name is more useful at a glance across a grid than a repeated
    per-panel "Total precursors" label). `total_width`: optional inches, overriding the default
    PANEL_SIZE[0]*ncols -- used by run()'s own call so this grid lands at a specific target width
    rather than growing with ncols; each column is simply total_width/ncols wide (close enough to
    PANEL_SIZE[0] that no label clips, unlike the much smaller MAX_INTEGRATED_WIDTH C/D/E use)."""
    n = len(entries)
    ncols = max(1, min(ncols, n))
    nrows = -(-n // ncols)  # ceil
    width = total_width if total_width is not None else PANEL_SIZE[0] * ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(width, PANEL_SIZE[1] * nrows),
                            squeeze=False)
    axs_flat = axs.ravel()
    for ax, (dataset, curve, n_runs, n_total, pad) in zip(axs_flat, entries):
        P._draw_coverage(ax, curve, n_runs, n_total, pad, title=f"{dataset_label(dataset)}\nn={n_total:,}")
    for ax in axs_flat[n:]:
        ax.axis("off")
    S.fig_legend(fig, *axs_flat[0].get_legend_handles_labels(), ncol=2, y=1.0, fontsize=6)
    fig.subplots_adjust(left=.08, right=.97, top=.88, bottom=.12, wspace=.4, hspace=.6)
    sv(fig, SUPP, stem)


def _draw_completeness_complete_only(ax, contrasts, coverage, completeness_transition, stem="",
                                     legend=True, grouped_ticks=False, tick_unit=False):
    """The completeness Complete-only bar content itself -- shared by panel_completeness_complete_only
    (one axes, one figure, this run's own data only) and panel_completeness_precursor_vs_protein
    (two axes, one shared figure -- precursor data in one, protein in the other, genuinely sharing
    one y-axis via matplotlib's own sharey=True, not a post-hoc approximation). `stem`: only used
    in the DIA-NN/JUMPlion total-mismatch warning message, to say which panel it came from.
    Does NOT call compact_axes/set_ylabel/figure-level layout -- the caller owns those, since they
    differ between the two callers (single vs. shared-y pair)."""
    x = np.arange(len(contrasts))
    coverage_by_method = P._coverage_by_method(coverage, completeness_transition)
    wm = 0.38
    totals_by_method = {}
    for mi, method in enumerate(S.METHOD_ORDER):
        cov_m = coverage_by_method[method]
        totals_m = np.array([cov_m[cov_m.contrast == cc].n.sum() for cc in contrasts], float)
        totals_by_method[method] = totals_m
        v = np.array([cov_m[(cov_m.contrast == cc) & (cov_m.state == P.COMPLETE)].n.sum() for cc in contrasts], float)
        pct = np.divide(100 * v, totals_m, out=np.zeros_like(v), where=totals_m > 0)
        off = (-wm / 2, wm / 2)[mi]
        ax.bar(x + off, pct, width=wm, color=S.method_color(method), edgecolor=S.NEUTRAL,
              linewidth=.3, label=method)
    # One total-n annotation per dataset, at the top of the axes -- not per bar (see
    # panel_sensitivity_fdr_bar's matching n= annotation, same convention: only JUMPlion's
    # own total is shown; warn, don't silently pick one, if DIA-NN's own total differs).
    totals_d, totals_j = totals_by_method[P.M_D], totals_by_method[P.M_J]
    for xi, nd, nj in zip(x, totals_d, totals_j):
        if nd != nj:
            log.warning("%s: %s has a different total n between DIA-NN (%d) and JUMPlion "
                        "(%d) -- only JUMPlion's n is annotated.", stem, contrasts[int(xi)], int(nd), int(nj))
        if nj > 0:
            # rotation=45, ha="left": each label leans up-and-to-the-right from its own tick,
            # instead of sitting flat and colliding with its neighbor's -- same reasoning as the
            # rotated x-tick labels below, just angled the other way since these start from the
            # top of the axes, not the bottom.
            ax.text(xi, 1.0, f"n={int(nj):,}", rotation=30, ha="center", va="bottom", fontsize=4.5,
                   transform=ax.get_xaxis_transform())
    if grouped_ticks:
        grouped_dataset_ticks(ax, contrasts, unit=tick_unit)
    else:
        P.contrast_ticks(ax, [dataset_label(c) for c in contrasts], rotation=30)
    if legend:
        ax.legend(frameon=False, fontsize=5.0, loc="upper right")


def panel_completeness_complete_only(contrasts, coverage, completeness_transition, DEST, sv,
                                     stem="Fig_Generalization_C_completeness_by_method_complete_only",
                                     legend=True, panel_size=PANEL_SIZE, margins=None):
    """Same data as P.panel_completeness_by_method, but only the Complete state's percentage --
    one plain (not stacked) bar per method per contrast, so "how many are fully complete" reads
    directly off the bar height instead of a stack boundary. One "n=" annotation per dataset, at
    the top of the axes (JUMPlion's own total only, matching panel_sensitivity_fdr_bar's
    own convention) -- not per bar, since the bar height itself is already a count-derived
    percentage and a second, literal count on top of each bar was redundant. `legend=False` for
    every COMBINED_CONTRASTS entry's own call (run() passes it for both), which each feed their own
    assemble_integrated_figure composite -- that composite gets one shared method legend for its
    C/D row instead; a standalone call outside COMBINED_CONTRASTS keeps the default True. `panel_size`
    (default PANEL_SIZE) is this panel's figsize. `margins`: optional (left, right, bottom) ABSOLUTE
    inches overriding P.INTEGRATED_LEFT/RIGHT/BOTTOM's own fractions -- used by every
    COMBINED_CONTRASTS entry's own call (see MAIN_PANEL_SIZE/Y_LABEL_MARGIN/X_LABEL_MARGIN/
    DECOR_MARGIN), whose panel_size is small enough that a fraction margin (shrinking right along
    with it) would clip the y-axis label or the leftmost rotated x-tick label; a standalone call
    keeps the original proportional fractions (margins=None), unaffected."""
    fig, ax = plt.subplots(figsize=panel_size)
    _draw_completeness_complete_only(ax, contrasts, coverage, completeness_transition, stem=stem, legend=legend,
                                     grouped_ticks=margins is not None, tick_unit=margins is not None)
    ax.set_ylabel("Completely quantified (%)")
    P.compact_axes(ax)
    ax.tick_params(axis="x", labelsize=X_TICK_LABEL_SIZE)
    if margins is None:
        left_frac, right_frac, bottom_frac = P.INTEGRATED_LEFT, P.INTEGRATED_RIGHT, P.INTEGRATED_BOTTOM
    else:
        left_m, right_m, bottom_m = margins
        left_frac, right_frac, bottom_frac = left_m / panel_size[0], 1 - right_m / panel_size[0], bottom_m / panel_size[1]
    fig.subplots_adjust(left=left_frac, right=right_frac, top=CD_TOP, bottom=bottom_frac)
    # tight=False: this panel's saved pixel size must exactly match figsize (PANEL_SIZE) -- see
    # assemble_integrated_figure, which tiles this next to panel_sensitivity_fdr_bar's own output,
    # which shares this same figure height and left/right/top/bottom margins, so their axes
    # (x-axis baselines) land at the same absolute position only if neither panel gets cropped to
    # its own content bbox first.
    sv(fig, DEST, stem, tight=False)


def _read_combined_completeness_tables(table_dir, all_datasets, contrast):
    """Reconstructs ONE contrast's combined (all-datasets) (coverage, completeness_transition)
    pair -- the same shape panel_completeness_complete_only itself is given -- from a
    generalization/table/ directory's own per-dataset completeness_overall_<dataset>.tsv /
    completeness_transition_by_contrast_<dataset>.tsv files. Those are the exact tables run()
    itself writes for EVERY dataset (see write_tsv calls above); the ALL-DATASETS-COMBINED view
    panel_completeness_complete_only actually draws from is otherwise only ever built in memory
    (_accumulate_and_maybe_combine) and never persisted as its own ready file, so this redoes that
    same combination step (filter to `contrast`, relabel contrast=dataset name, concat) reading
    from disk instead -- letting a DIFFERENT run's own saved tables (e.g. a precursor-level run's)
    be recombined here, in a THIRD process/run that has access to neither run's own in-memory
    data. Raises FileNotFoundError (left to the caller -- see panel_completeness_precursor_vs_protein's
    own call site in run(), which checks every file's existence up front and logs/skips instead) if
    `table_dir` doesn't yet have every dataset's own pair, e.g. because that other run hasn't
    happened yet."""
    table_dir = Path(table_dir)
    cov_parts, trans_parts = [], []
    for name in all_datasets:
        cov = pd.read_csv(table_dir / f"completeness_overall_{name}.tsv", sep="\t")
        trans = pd.read_csv(table_dir / f"completeness_transition_by_contrast_{name}.tsv", sep="\t")
        cov_parts.append(cov[cov.contrast == contrast].assign(contrast=name))
        trans_parts.append(trans[trans.contrast == contrast].assign(contrast=name))
    return pd.concat(cov_parts, ignore_index=True), pd.concat(trans_parts, ignore_index=True)


def panel_completeness_precursor_vs_protein(precursor_table_dir, all_datasets, contrast,
                                            protein_coverage, protein_completeness_transition, DEST, sv,
                                            stem="Fig_Generalization_C_completeness_precursor_vs_protein_all_datasets_B_vs_C",
                                            precursor_label="Precursor", protein_label="Protein",
                                            panel_size=PANEL_SIZE, legend=True, margins=None,
                                            axes_w=None, inner_gap=PVP_INNER_GAP):
    """TRUE side-by-side precursor-vs-protein completeness (Complete state only) comparison, in
    ONE matplotlib figure with a genuinely shared y-axis -- unlike splicing two independently-
    rendered PNGs together (an image-compositing approach tried and abandoned here), both sides are
    drawn from their own underlying data tables in this SAME call, so there's a real shared Axes
    object instead of an approximation (hiding one panel's own tick labels and hoping the two line
    up).

    `precursor_table_dir`: the PRECURSOR run's own generalization/table/ directory -- read back via
    _read_combined_completeness_tables (see its own docstring for why this reconstruction, rather
    than one ready file, is needed). `all_datasets`/`contrast`: the same dataset order/contrast
    name the PROTEIN side's own combined data was built with, so both sides share one x-axis
    category order. `protein_coverage`/`protein_completeness_transition`: this (protein) run's own
    already-combined pair, exactly as passed to panel_completeness_complete_only elsewhere -- no
    need to re-read this side's own tables since this function runs inside the same run() call that
    already has them in memory.

    `legend`/`margins`/`axes_w`/`inner_gap`: same ABSOLUTE-inches-margins-override reasoning as
    panel_sensitivity_fdr_bar's own `margins`/`axes_w` parameters (manual fig.add_axes placement,
    not plt.subplots+subplots_adjust, when `margins` is given) -- run() passes legend=False (the
    OUTER composite, assemble_integrated_figure_precursor_vs_protein, draws its own shared method
    legend instead), margins=(left, right, bottom, top) and axes_w=PVP_AXES_W (see its own note,
    shared with panel_sensitivity_fdr_bar's own D2 variant so every bar in that composite's row 1
    renders at the SAME absolute width) when feeding that composite's row 1; all default to this
    panel's own larger standalone look (own legend, generous fractional margins, sharey=True via
    plt.subplots) otherwise."""
    precursor_coverage, precursor_transition = _read_combined_completeness_tables(
        precursor_table_dir, all_datasets, contrast)
    if margins is None:
        fig, axs = plt.subplots(1, 2, figsize=(panel_size[0] * 2, panel_size[1]), sharey=True)
    else:
        left_m, right_m, bottom_m, top_m = margins
        aw = axes_w if axes_w is not None else (panel_size[0] - left_m - right_m)
        total_w = left_m + aw + inner_gap + aw + right_m
        fig = plt.figure(figsize=(total_w, panel_size[1]))
        y0_frac = bottom_m / panel_size[1]
        h_frac = 1 - (bottom_m + top_m) / panel_size[1]
        aw_frac = aw / total_w
        x0_fracs = [left_m / total_w, (left_m + aw + inner_gap) / total_w]
        axs = [fig.add_axes([x0, y0_frac, aw_frac, h_frac]) for x0 in x0_fracs]
        axs[1].sharey(axs[0])
    _draw_completeness_complete_only(axs[0], all_datasets, precursor_coverage, precursor_transition,
                                     stem=f"{stem} ({precursor_label})", legend=False,
                                     grouped_ticks=margins is not None)
    _draw_completeness_complete_only(axs[1], all_datasets, protein_coverage, protein_completeness_transition,
                                     stem=f"{stem} ({protein_label})", legend=False,
                                     grouped_ticks=margins is not None, tick_unit=margins is not None)
    axs[0].set_ylabel("Completely quantified (%)")
    axs[0].set_title(precursor_label, fontsize=6, pad=14)
    axs[1].set_title(protein_label, fontsize=6, pad=14)
    for ax in axs:
        P.compact_axes(ax)
        ax.tick_params(axis="x", labelsize=X_TICK_LABEL_SIZE)
    axs[1].tick_params(axis="y", labelleft=False)  # sharey already shares the axis; this just
                                                    # drops the (redundant) right-hand tick numbers
    # tight=False when `margins` is given: same as panel_sensitivity_fdr_bar's own sv() -- a tight
    # crop would trim this panel's own saved image differently from D2's, so C2's and D2's axes
    # wouldn't line up (top/bottom) once the composite tiles them side by side.
    if legend:
        method_handles = [Patch(facecolor=S.method_color(m), label=m) for m in S.METHOD_ORDER]
        S.fig_legend(fig, method_handles, [h.get_label() for h in method_handles], ncol=2, y=1.0, fontsize=5)
    if margins is None:
        # top=.65: generous, since this panel's own top margin has to fit the shared legend, both
        # panels' own titles, AND the rotated n= annotations, all stacked above the axes.
        fig.subplots_adjust(left=.14, right=.98, top=.65, bottom=.26, wspace=.08)
    sv(fig, DEST, stem, tight=margins is None)


def panel_sensitivity_fdr_bar(contrasts, partial_perf, sens_ylim, fdr_ylim, DEST, sv,
                              stem="Supp_Generalization_D_sensitivity_fdr_all_datasets", legend=True,
                              panel_size=PANEL_SIZE, inner_gap=PANEL_D_INNER_GAP, margins=None,
                              title=None, title_pad=14, grouped_ticks=False):
    """Partially-complete sensitivity/FDR only (both methods), as bars instead of
    P.panel_sensitivity_fdr's line+marker plot -- for the "_all_datasets" combined panels here,
    where the x-axis is dataset name rather than a meaningfully ordered sequence of contrasts, so
    a bar per category reads more naturally than a connecting line. The first argument is one
    stratum's table: partially complete (both methods) or insufficient/restored (JUMPlion only,
    one centered bar per dataset). One "n=" annotation per dataset, at the top of each
    subplot (JUMPlion's own total only -- see panel_completeness_complete_only's own
    matching annotation and docstring note). Each of the two
    subplots gets exactly panel_completeness_complete_only's own axes width/height at `panel_size`
    (see _panel_d_size) via manual fig.add_axes placement, with `inner_gap` between them -- so a
    bar of the same width fraction (`w`, matching panel_completeness_complete_only's own `wm`)
    renders at the same absolute width in both panels, and both panels' axes (x-axis baselines)
    land at the same absolute position, when assemble_integrated_figure tiles them side by side.
    `legend=False` for every COMBINED_CONTRASTS entry's own call, same reasoning as
    panel_completeness_complete_only's own `legend` parameter -- see its docstring. `margins`:
    optional (left, right, bottom, axes_w) ABSOLUTE inches -- same reasoning and same
    per-COMBINED_CONTRASTS-entry usage as panel_completeness_complete_only's own `margins`
    parameter (see its docstring), but D additionally needs an explicit shared `axes_w` (matching
    C's own) since D's total size is otherwise derived from `panel_size`/`inner_gap`, not set
    directly; when given, `panel_size`/`inner_gap` are ignored for layout (still accepted so the
    call signature stays uniform). A 5-element `margins` (left, right, bottom, axes_w, top) adds an
    explicit ABSOLUTE top margin too (instead of the implicit CD_TOP-fraction-of-panel_size[1] the
    4-element form uses) -- needed for the "Protein"-titled D2 variant feeding
    assemble_integrated_figure_precursor_vs_protein's row 1 (see PVP_AXES_W/PVP_TOP_MARGIN's own
    note), which needs extra room above the axes for `title` that plain D's own n=-only top margin
    doesn't reserve. `title`/`title_pad`: when given, set on BOTH subplots (sensitivity, FDR) --
    this D variant shows protein-only data, same as C2's own "Protein"-titled subplot, so the two
    read consistently in that composite."""
    x = np.arange(len(contrasts))
    w = 0.38  # matches panel_completeness_complete_only's own wm -- see PANEL_D_SIZE's own note.
    if margins is None:
        axes_w, _right_margin = _axes_geometry(panel_size[0])
        d_size = _panel_d_size(panel_size[0], panel_size[1], inner_gap)
        y0_frac, h_frac = P.INTEGRATED_BOTTOM, CD_TOP - P.INTEGRATED_BOTTOM
        x0_fracs = [inner_gap / d_size[0], (inner_gap + axes_w + inner_gap) / d_size[0]]
    elif len(margins) == 5:
        left_m, right_m, bottom_m, axes_w, top_m = margins
        d_size = (left_m + axes_w + inner_gap + axes_w + right_m, panel_size[1])
        y0_frac = bottom_m / d_size[1]
        h_frac = 1 - (bottom_m + top_m) / d_size[1]
        x0_fracs = [left_m / d_size[0], (left_m + axes_w + inner_gap) / d_size[0]]
    else:
        left_m, right_m, bottom_m, axes_w = margins
        d_size = (left_m + axes_w + inner_gap + axes_w + right_m, panel_size[1])
        y0_frac, h_frac = bottom_m / d_size[1], CD_TOP - bottom_m / d_size[1]
        x0_fracs = [left_m / d_size[0], (left_m + axes_w + inner_gap) / d_size[0]]
    fig = plt.figure(figsize=d_size)
    ax_w_frac = axes_w / d_size[0]
    axs = [fig.add_axes([x0, y0_frac, ax_w_frac, h_frac]) for x0 in x0_fracs]
    # Methods actually present: both for the partially-complete stratum, JUMPlion only for the
    # insufficient (restored) one, whose single bar is centered on the tick.
    present = [m for m in S.METHOD_ORDER if (partial_perf.method == m).any()]
    for mi, method in enumerate(present):
        zzd = partial_perf[partial_perf.method == method].set_index("contrast").reindex(contrasts)
        off = (-w / 2, w / 2)[mi] if len(present) > 1 else 0.0
        axs[0].bar(x + off, zzd.sensitivity * 100, width=w, color=S.method_color(method),
                  edgecolor=S.NEUTRAL, linewidth=.3, label=method)
        axs[1].bar(x + off, zzd.empirical_fdr * 100, width=w, color=S.method_color(method),
                  edgecolor=S.NEUTRAL, linewidth=.3)
    labels = [dataset_label(c) for c in contrasts]
    if grouped_ticks or margins is not None:
        grouped_dataset_ticks(axs[0], contrasts, unit=True); grouped_dataset_ticks(axs[1], contrasts, unit=True)
    else:
        P.contrast_ticks(axs[0], labels, rotation=30); P.contrast_ticks(axs[1], labels, rotation=30)
    axs[0].set_ylabel("Sensitivity (%)"); axs[0].set_ylim(*sens_ylim)
    axs[1].set_ylabel("Empirical FDR (%)"); axs[1].set_ylim(*fdr_ylim)
    if title is not None:
        for ax in axs:
            ax.set_title(title, fontsize=6, pad=title_pad)

    # One total-n annotation per dataset, at the top of each subplot -- same denominator on both
    # (sensitivity and FDR come from the same partially-complete stratum), and only DIA-NN+
    # JUMPlion's own n (not DIA-NN's), matching panel_completeness_complete_only's own convention;
    # warn, don't silently pick one, if DIA-NN's own n for a dataset differs.
    n_d_map = partial_perf[partial_perf.method == P.M_D].set_index("contrast")["n"].reindex(contrasts)
    n_j_map = partial_perf[partial_perf.method == P.M_J].set_index("contrast")["n"].reindex(contrasts)
    for c, nd, nj in zip(contrasts, n_d_map, n_j_map):
        if pd.notna(nd) and pd.notna(nj) and int(nd) != int(nj):
            log.warning("%s: %s has different n between DIA-NN (%d) and JUMPlion (%d) in "
                        "the partial-completeness stratum -- only JUMPlion's n is annotated.",
                        stem, c, int(nd), int(nj))
    for ax in axs:
        for xi, nj in zip(x, n_j_map):
            if pd.notna(nj):
                # rotation=45, ha="left": see panel_completeness_complete_only's own matching n=
                # annotation and its comment -- same reasoning, kept consistent between the two.
                ax.text(xi, 1.0, f"n={int(nj):,}", rotation=30, ha="center", va="bottom", fontsize=4.5,
                       transform=ax.get_xaxis_transform())

    P.compact_axes(*axs)
    for ax in axs:
        ax.tick_params(axis="x", labelsize=X_TICK_LABEL_SIZE)
    if legend:
        axs[0].legend(frameon=False, fontsize=5, loc="upper right")
    # No subplots_adjust -- both axes were already placed directly (fig.add_axes above), matching
    # panel_completeness_complete_only's own geometry exactly; see this function's docstring.
    # tight=False: see the matching note on panel_completeness_complete_only's own sv() call.
    sv(fig, DEST, stem, tight=False)


def panel_sensitivity_fdr_insufficient_grid(entries, datasets, DEST, sv,
                                            stem="Supp_Generalization_D_sensitivity_fdr_insufficient_all_datasets",
                                            size=(3.6, 3.2)):
    """Insufficient (restored by JUMPlion) stratum, ONE figure for every COMBINED_CONTRASTS entry --
    entries: [(contrast, combined_rec), ...] (combined_rec: JUMPlion-only, one row per dataset). Sensitivity
    in the top row, empirical FDR in the bottom row, one column per contrast; each row's y-range is matched
    across the contrasts (computed on every contrast's values together), so the bars read on the same scale."""
    sens_ylim = P._matched_ylim(pd.concat([r.sensitivity for _, r in entries]) * 100, floor_zero=True)
    fdr_ylim = P._matched_ylim(pd.concat([r.empirical_fdr for _, r in entries]) * 100, floor_zero=True)
    fig, axs = plt.subplots(2, len(entries), figsize=size, squeeze=False, layout="constrained")
    x = np.arange(len(datasets))
    for col, (contrast, rec) in enumerate(entries):
        zz = rec[rec.method == P.M_J].set_index("contrast").reindex(datasets)
        ax_s, ax_f = axs[0, col], axs[1, col]
        ax_s.bar(x, zz.sensitivity * 100, width=.38, color=S.method_color(P.M_J), edgecolor=S.NEUTRAL, linewidth=.3)
        ax_f.bar(x, zz.empirical_fdr * 100, width=.38, color=S.method_color(P.M_J), edgecolor=S.NEUTRAL, linewidth=.3)
        ax_s.set_ylim(*sens_ylim)
        ax_f.set_ylim(*fdr_ylim)
        if col == 0:
            ax_s.set_ylabel("Sensitivity (%)"); ax_f.set_ylabel("Empirical FDR (%)")
        for ax in (ax_s, ax_f):
            grouped_dataset_ticks(ax, datasets, unit=True)
            for xi, nj in zip(x, zz["n"]):  # same n= annotation as panel_sensitivity_fdr_bar
                if pd.notna(nj):
                    ax.text(xi, 1.0, f"n={int(nj):,}", ha="center", va="bottom", fontsize=5.5,
                            transform=ax.get_xaxis_transform())
        ax_s.set_title(S.clabel(contrast), fontsize=6, pad=8)
    P.compact_axes(*axs.ravel())
    for ax in axs.ravel():
        ax.tick_params(axis="x", labelsize=X_TICK_LABEL_SIZE)
    sv(fig, DEST, stem, tight=False)


def _box_grid_figure(entries_by_row, col_width, panel_height, left_frac, legend_handles=None):
    """Shared scaffold for both panel_log2fc_error_cv_box_grid and its own
    _restored sibling -- one figure, 2 rows (log2FC error on top, CV on bottom), one column per
    dataset, sharey="row", dataset-name titles on row 0 only, species x-tick labels on row 1 only.
    `entries_by_row`: {0: [(dataset_name, draw_fn), ...], 1: [...]} where each `draw_fn(ax)` draws
    that one subplot's own box(es) and returns the data group(s) actually plotted (for
    P._matched_box_ylim). Returns (fig, axs) with everything already drawn except the caller's own
    fig.subplots_adjust/sv -- margins differ between the two callers (one has a legend to leave
    room for, one doesn't)."""
    n = len(entries_by_row[0])
    pos = list(range(len(P.CLEAN)))
    fig, axs = plt.subplots(2, n, figsize=(col_width * n, panel_height), sharey="row")
    axs = np.asarray(axs).reshape(2, n)
    row_labels = {0: "log$_2$FC error", 1: "CV (%)"}
    row_zero_ref = {0: True, 1: False}
    row_groups = {row_i: [] for row_i in (0, 1)}
    for row_i, entries in entries_by_row.items():
        for j, (name, draw_fn) in enumerate(entries):
            ax = axs[row_i, j]
            row_groups[row_i] += draw_fn(ax)
            if row_zero_ref[row_i]:
                S.ref_line(ax, 0)
            ax.set_xticks(pos)
            ax.set_xticklabels([S.species_label(s) for s in P.CLEAN] if row_i == 1 else [],
                               rotation=30, ha="right")
            if row_i == 0:
                ax.set_title(dataset_label(name), fontsize=6)
            if j == 0:
                ax.set_ylabel(row_labels[row_i])
            P.compact_axes(ax)
            ax.tick_params(axis="x", labelsize=X_TICK_LABEL_SIZE)
    for row_i in (0, 1):
        ylim = P._matched_box_ylim(row_groups[row_i], floor_zero=(row_i == 1))
        for ax in axs[row_i]:
            ax.set_ylim(*ylim)
    if legend_handles is not None:
        # y=0.98 (not >1.0, unlike most of this codebase's other fig_legend calls): tight=False
        # below saves exactly figsize's own [0, w] x [0, h] bbox, not auto-expanded to include
        # whatever spills outside it the way the default tight=True crop would -- a legend anchored
        # above y=1.0 (relying on that auto-expansion) would otherwise be silently clipped out of
        # the saved file.
        S.fig_legend(fig, legend_handles, [h.get_label() for h in legend_handles], ncol=len(legend_handles),
                    y=1.0, fontsize=5)
    return fig, axs


def panel_log2fc_error_cv_box_grid(fc_entries, cv_entries, DEST, sv,
                                   stem="Supp_Generalization_E_log2FC_error_CV_box_all_datasets_B_vs_C",
                                   col_width=PANEL_SIZE[0], height_scale=0.8, left_margin=None,
                                   restored=False, compact_x=False):
    """Species-split log2FC error (top row) and CV (bottom row) box grids for the PARTIALLY
    COMPLETE stratum only, combined into ONE figure -- one column per dataset, shared across both
    rows (dataset name title only on the top row, species x-tick labels only on the bottom row)
    instead of two separate figures that would otherwise each repeat both. Compares DIA-NN vs
    JUMPlion (two box series per species, by method color only -- no completeness-state
    shading needed here anymore, since every box drawn is the same state). The Insufficient
    (restored) stratum -- JUMPlion-only, no DIA-NN counterpart to compare against -- is its own
    separate figure now (see panel_log2fc_error_cv_box_grid_restored), no longer combined into
    this one. No legend of its own -- method color (blue=DIA-NN, red=JUMPlion) is
    identified by the shared C/D legend in assemble_integrated_figure. log2FC error is the signed
    delta_FC (not its absolute value), with a dashed zero reference line (no systematic
    over/under-estimation); CV is a percentage. `fc_entries`/`cv_entries`: [(dataset_name,
    partial_df), ...] in the same dataset order, each already filtered to one contrast (e.g. B/C)
    but BOTH methods (unlike the stratum-vs-stratum form this replaced, no longer pre-filtered to
    JUMPlion only). `col_width` (default PANEL_SIZE[0]) is the per-dataset column width; every
    COMBINED_CONTRASTS entry's own call passes a narrower one sized to match row 1 (C+D) in
    assemble_integrated_figure, so that composite can tile this panel at its own native size (no
    cropping/stretching afterward) instead of needing to shrink it -- shrinking a saved PNG to fit
    changes its apparent font size relative to a panel that wasn't shrunk, making text sizes
    inconsistent across the composite. `left_margin`: optional ABSOLUTE inches overriding the
    default `left=.08` fraction margin -- that fraction, like C/D's own fraction margins, shrinks
    right along with a narrow `col_width` and can clip the y-axis label; every COMBINED_CONTRASTS
    entry's own call passes E_LEFT_MARGIN. `height_scale` (default 1.1, multiplies PANEL_SIZE[1]):
    lower than earlier (1.4, 1.3) -- this panel's height was more than its tick-label content
    actually needed. `restored=True`: the Insufficient (restored) stratum instead -- JUMPlion-only
    (no DIA-NN counterpart, and no "method" column in the input), so one centered box per species,
    same sizes/margins/n= annotations as the partially-complete form. `compact_x=True` (the
    precursor-vs-protein E): horizontal species tick labels, each followed on a second line by its
    n= (at the bottom of the plot) instead of the n= above the CV axes."""
    width = col_width * len(fc_entries)

    def make_draw(col, scale):
        def draw(ax, partial):
            box_pos, data, cat = [], [], []
            for i, sp in enumerate(P.CLEAN):
                if restored:
                    box_pos.append(i)
                    data.append(partial[partial.species == sp][col].to_numpy(float) * scale)
                    cat.append(P.M_J)
                    continue
                for off, method in [(-.18, P.M_D), (.18, P.M_J)]:
                    box_pos.append(i + off)
                    data.append(partial[(partial.species == sp) & (partial.method == method)][col]
                               .to_numpy(float) * scale)
                    cat.append(method)
            S.boxplot_groups(ax, data, box_pos, cat, S.METHOD_COLORS, width=.3)
            return data
        return draw

    draw_fc, draw_cv = make_draw("delta_FC", 1.0), make_draw("CV", 100.0)
    entries_by_row = {
        0: [(name, lambda ax, partial=partial: draw_fc(ax, partial)) for name, partial in fc_entries],
        1: [(name, lambda ax, partial=partial: draw_cv(ax, partial)) for name, partial in cv_entries],
    }
    left_frac = .08 if left_margin is None else left_margin / width
    fig, axs = _box_grid_figure(entries_by_row, col_width, PANEL_SIZE[1] * height_scale, left_frac)
    # One "n=" per species group, in the gap between the log2FC-error row and the CV row (just above
    # the CV axes): JUMPlion's own count of precursors with a delta_FC -- same
    # JUMPlion-only-n convention as C/D's own n= annotations. DIA-NN's own count is expected to
    # match (same partially-complete precursors); warn, don't silently pick one, if it doesn't.
    for j, (name, partial) in enumerate(fc_entries):
        n_labels = []
        for i, sp in enumerate(P.CLEAN):
            if restored:
                n_sp = int(partial[partial.species == sp]["delta_FC"].notna().sum())
                n_by_method = {P.M_D: n_sp, P.M_J: n_sp}
            else:
                n_by_method = {m: int(partial[(partial.species == sp) & (partial.method == m)]["delta_FC"].notna().sum())
                               for m in (P.M_D, P.M_J)}
                n_sp = n_by_method[P.M_J]
            if n_by_method[P.M_D] != n_sp:
                log.warning("%s / %s: %s has different n between DIA-NN (%d) and JUMPlion (%d) "
                            "in the log2FC-error/CV box grid -- only JUMPlion's n is annotated.",
                            stem, dataset_label(name), sp, n_by_method[P.M_D], n_sp)
            if compact_x:
                n_labels.append(f"{S.species_label(sp)}\nn={n_sp:,}")
                continue
            axs[1, j].annotate(f"n={n_sp:,}", xy=(i, 1.0), xycoords=axs[1, j].get_xaxis_transform(),
                               xytext=(0, 1.5), textcoords="offset points", ha="center", va="bottom",
                               fontsize=4.5)
        if compact_x:
            axs[1, j].set_xticklabels(n_labels, rotation=0, ha="center")
    # top/bottom are tight on purpose -- this panel's overall figsize is already shorter than its
    # width warrants (one row per metric, not per dataset), so both margins are sized to the
    # title/tick-label content itself rather than to a generic fraction that would otherwise leave
    # a lot of visibly empty space above/below a short canvas.
    # left: PANEL_SIZE[1] fraction by default, matching every other panel's own proportional
    # margin; `left_margin`, when given, is an ABSOLUTE-inch override instead (see its docstring).
    fig.subplots_adjust(left=left_frac, right=E_RIGHT, top=E_TOP, bottom=E_BOTTOM_COMPACT if compact_x else E_BOTTOM,
                        wspace=E_WSPACE, hspace=E_HSPACE)
    # tight=False: this panel's saved pixel WIDTH must exactly match col_width * n for
    # assemble_integrated_figure's width-matching (see col_width's docstring note) to actually
    # land on the intended width instead of whatever a content-bbox crop happens to trim it to.
    sv(fig, DEST, stem, tight=False)


def panel_log2fc_error_cv_box_grid_restored(fc_entries, cv_entries, DEST, sv,
                                            stem="Supp_Generalization_E_log2FC_error_CV_box_restored_all_datasets_B_vs_C",
                                            **kwargs):
    """The Insufficient (restored) stratum's log2FC error / CV box grid: panel_log2fc_error_cv_box_grid
    in its `restored=True` form (JUMPlion-only, one box per species), so it has exactly the same
    format -- column width, margins, n= annotations, no legend -- as the partially-complete grid.
    `fc_entries`/`cv_entries`: [(dataset_name, restored_df), ...]."""
    panel_log2fc_error_cv_box_grid(fc_entries, cv_entries, DEST, sv, stem=stem, restored=True, **kwargs)


def _integrated_panel_files(contrast, dest_attr):
    """The three C/D/E panel stems one integrated composite tiles together, for one
    COMBINED_CONTRASTS entry (contrast, dest_attr) -- B itself (the stacked completeness-by-method
    panel) is always excluded: it has its own single combined Supp figure instead (see
    panel_completeness_by_method_grid), independent of this entry's own dest_attr."""
    return {
        "C": _combined_stem("C_completeness_by_method_complete_only", contrast, dest_attr),
        "D": _combined_stem("D_sensitivity_fdr", contrast, dest_attr),
        "E": _combined_stem("E_log2FC_error_CV_box", contrast, dest_attr),
    }


def _e_column_labels(ex, ey, ebw, n_datasets, e_width):
    """Bold-letter positions for E's dataset columns (d, e, f, ...), each at its column's left axes
    edge (minus a small offset, a bit more for the first), derived from the same left/right/wspace
    panel_log2fc_error_cv_box_grid uses. (ex, ey, ebw) is E's placed position/width; `e_width` its
    native width in inches (col_width * n_datasets)."""
    e_left = E_LEFT_MARGIN / e_width
    e_col_w = (E_RIGHT - e_left) / (n_datasets + (n_datasets - 1) * E_WSPACE)
    return [(chr(ord("d") + j), ex + ebw * (e_left + j * e_col_w * (1 + E_WSPACE)) - (0.4 if j == 0 else 0.2), ey)
            for j in range(n_datasets)]


def assemble_integrated_figure(L, saver, contrast, dest_attr, n_datasets):
    """One integrated composite for one COMBINED_CONTRASTS entry (distinct from performance.py's
    own Fig_Generalization_Integrated, which covers Main A-D of the full Wang-HYE pipeline) -- named
    Fig_Generalization_Integrated for the Main entry (unsuffixed, matching this composite's original
    name before every COMBINED_CONTRASTS entry got one), or Supp_Generalization_Integrated_<contrast>
    for a Supp entry. Row 1: C (completeness Complete-only, lettered a) | D (sensitivity, b | FDR,
    c); row 2: E (the combined log2FC error / CV box grid, panel_log2fc_error_cv_box_grid -- its own
    two rows share dataset titles/species x-labels/legend internally; one letter per dataset
    column, d, e, f, ...) -- this entry's own three C/D/E figures, saved as three PNGs but
    labelled one letter per VISUAL panel, not per saved PNG -- see the sub_labels block below).

    C and D share ONE method (blue=DIA-NN, red=JUMPlion) legend, in LEGEND_GAP's reserved
    strip above row 1, instead of each carrying its own -- both are generated with legend=False for
    every COMBINED_CONTRASTS entry (see run()). E needs no legend of its own here -- it already
    uses the same two method colors, identified by the shared C/D legend above it.

    Each panel's physical size is read back from its own saved PNG (pixel shape / saver.dpi), like
    performance.py's own assemble_integrated_figure -- but C, D and E are each already generated at
    a size that sums to MAX_INTEGRATED_WIDTH natively (see MAIN_PANEL_SIZE/MAIN_INNER_GAP and
    run()'s own col_width kwarg), not at their usual (larger) default size, so there is normally no
    scaling left to do here at all: row_w_natural[0] (C+D) already comes out ~MAX_INTEGRATED_WIDTH,
    and E's own call already requests col_width=MAX_INTEGRATED_WIDTH/n_datasets, so its saved PNG
    is already that width before this function ever runs. The per-row scaling below only exists as
    a safety net -- e.g. if MAX_INTEGRATED_WIDTH is changed without also regenerating C/D/E at
    their new native size -- and, if it ever does anything, still scales every row by the same
    shared factor (not one row alone by a different one, which would otherwise make its text
    visibly smaller than the other rows', despite every panel using the same absolute font sizes
    internally).

    See panel_completeness_precursor_vs_protein for a genuine (true-sharey, one-matplotlib-figure)
    precursor-vs-protein comparison -- NOT part of this composite; this composite's own "C" panel
    is always just this run's own data, exactly as before."""
    dest_dir = getattr(L, dest_attr)
    panel_files = _integrated_panel_files(contrast, dest_attr)
    paths = {k: dest_dir / f"{stem}.png" for k, stem in panel_files.items()}
    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        log.warning("assemble_integrated_figure skipped for %s (%s): missing panel PNG(s) for %s "
                    "(is \"png\" in the configured style.formats?)", contrast, dest_attr, missing)
        return

    imgs = {k: plt.imread(p) for k, p in paths.items()}
    w = {k: img.shape[1] / saver.dpi for k, img in imgs.items()}
    h = {k: img.shape[0] / saver.dpi for k, img in imgs.items()}

    rows = [["C", "D"], ["E"]]
    # Horizontal gap between C and D (row 1 only, see ROW1_GAP's own docstring note) -- E is a
    # single-column row.
    row_gaps = [[MAIN_ROW1_GAP], []]
    row_w_natural = [sum(w[k] for k in row) + sum(gaps) for row, gaps in zip(rows, row_gaps)]
    row_h_natural = [max(h[k] for k in row) for row in rows]

    # Target width every row is scaled to match (see docstring): row 1's (C+D's) own natural
    # width, capped at MAX_INTEGRATED_WIDTH.
    target_w = min(row_w_natural[0], MAX_INTEGRATED_WIDTH)
    row_scale = [target_w / rw for rw in row_w_natural]
    row_h = [rh * s for rh, s in zip(row_h_natural, row_scale)]
    LEGEND_GAP = 0.18  # inches reserved above row 1 for the shared C/D method legend
    total_w, total_h = target_w, sum(row_h) + LEGEND_GAP

    positions = {}
    y = LEGEND_GAP
    for row, gaps, s, rh in zip(rows, row_gaps, row_scale, row_h):
        x = 0.0  # every row is scaled to exactly total_w, so none needs centering
        for i, k in enumerate(row):
            bw, bh = w[k] * s, h[k] * s
            positions[k] = (x, y, bw, bh)
            x += bw + (gaps[i] * s if i < len(gaps) else 0)
        y += rh

    fig = plt.figure(figsize=(total_w, total_h))
    for k, (x0_, y0_, bw, bh) in positions.items():
        ax = fig.add_axes([x0_ / total_w, 1 - (y0_ + bh) / total_h, bw / total_w, bh / total_h])
        ax.imshow(imgs[k])
        ax.axis("off")
    method_handles = [Patch(facecolor=S.method_color(m), label=m) for m in S.METHOD_ORDER]
    # fontsize=6 (not 7): at MAX_INTEGRATED_WIDTH's own small scale, 7pt stood out as noticeably
    # larger than every other label in the composite (C/D/E's own text all sits around 4.5-6.5pt).
    fig.legend(handles=method_handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              frameon=False, fontsize=5, handlelength=1.2, columnspacing=1.2)
    # Bold letter labels -- a (C), b/c (D's sensitivity/FDR subplots), d/e (E's log2FC-error/CV
    # rows): one per VISUAL panel, not one per saved PNG, since D and E each save as a single
    # image covering two sub-panels. b/c and d/e are placed at sub-positions computed from the
    # same geometry constants those two images were actually drawn with (D's FDR axes left edge:
    # Y_LABEL_MARGIN/MAIN_AXES_W/MAIN_INNER_GAP; E's CV row top: E_TOP/E_BOTTOM/E_HSPACE's own
    # GridSpec math), not estimated/guessed positions, and scaled by each panel's own ACTUAL
    # placed size (bw/bh, already read back from its saved PNG) so this stays correct even if
    # assemble_integrated_figure's own row-scaling (see its docstring) ever actually does
    # something.
    label_pad = 0.05  # inches inset from each sub-panel's own top-left corner
    cx, cy, _cbw, _cbh = positions["C"]
    dx, dy, dbw, _dbh = positions["D"]
    ex, ey, ebw, _ebh = positions["E"]
    _native_d_width = Y_LABEL_MARGIN + MAIN_AXES_W + MAIN_INNER_GAP + MAIN_AXES_W + MAIN_UNIT_MARGIN
    fdr_x_offset = dbw * (Y_LABEL_MARGIN + MAIN_AXES_W + MAIN_INNER_GAP) / _native_d_width
    # E's letters are one per dataset column (d, e, f, ...), as in the precursor-vs-protein composite.
    sub_labels = ([("a", cx, cy), ("b", dx, dy), ("c", dx + fdr_x_offset - 0.35, dy)]
                  + _e_column_labels(ex, ey, ebw, n_datasets, MAX_INTEGRATED_WIDTH))
    for letter, x0_, y0_ in sub_labels:
        fig.text((x0_ + label_pad) / total_w, 1 - (y0_ + label_pad) / total_h, letter,
                 fontsize=9, fontweight="bold", ha="left", va="top")
    out_stem = "Fig_Generalization_Integrated" if dest_attr == "main" else f"Supp_Generalization_Integrated_{contrast}"
    saver.save(fig, dest_dir, out_stem)


def _integrated_panel_files_pvp(contrast, dest_attr):
    """The C2/D2/E panel stems assemble_integrated_figure_precursor_vs_protein tiles together --
    "C2" and "D2" are dedicated composite-sized/titled variants (panel_completeness_precursor_vs_
    protein and panel_sensitivity_fdr_bar's own `title="Protein"` call -- see PVP_AXES_W/
    PVP_TOP_MARGIN's own notes), NOT the same saved PNGs assemble_integrated_figure itself uses for
    its own C/D (those stay untitled, at MAIN_AXES_W); "E" is the one PNG shared between both
    composites, unchanged."""
    return {
        "C2": _combined_stem("C_completeness_precursor_vs_protein", contrast, dest_attr),
        "D2": _combined_stem("D_sensitivity_fdr_precursor_vs_protein", contrast, dest_attr),
        "E": _combined_stem("E_log2FC_error_CV_box_precursor_vs_protein", contrast, dest_attr),
    }


def assemble_integrated_figure_precursor_vs_protein(L, saver, contrast, dest_attr, n_datasets):
    """A SEPARATE integrated composite from assemble_integrated_figure, built only for the Main
    entry (the only one with a C2/D2 panel at all -- see run()): row 1 is C2 (the precursor-vs-
    protein true-shared-y-axis comparison, already carrying its own "Precursor"/"Protein" titles
    internally, so it gets a single letter rather than one per sub-axes) | D2 (the "Protein"-titled
    sensitivity, b | FDR, c variant of D), same row; row 2 is E (log2FC error, d | CV, e) -- this
    entry's own same E PNG assemble_integrated_figure itself tiles, just paired with the wider C2 +
    titled D2 instead of the narrower single-dataset C + plain D. Does not replace or alter
    assemble_integrated_figure's own output -- that composite keeps its original "C"/"D" (this
    run's own data alone, untitled).

    Unlike assemble_integrated_figure's own row 1 (whose C+D natural widths already sum to
    MAX_INTEGRATED_WIDTH via MAIN_AXES_W's own derivation), here C2 and D2 are BOTH generated at
    PVP_AXES_W (see its own note) specifically so their natural widths already sum to
    MAX_INTEGRATED_WIDTH too, AND so every bar across all 4 subplots (C2's 2, D2's 2) renders at
    the same absolute width -- the row-scaling below is just the same safety net
    assemble_integrated_figure's own docstring describes, not doing real work in the normal case."""
    dest_dir = getattr(L, dest_attr)
    panel_files = _integrated_panel_files_pvp(contrast, dest_attr)
    paths = {k: dest_dir / f"{stem}.png" for k, stem in panel_files.items()}
    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        log.warning("assemble_integrated_figure_precursor_vs_protein skipped for %s (%s): missing "
                    "panel PNG(s) for %s (is \"png\" in the configured style.formats?)",
                    contrast, dest_attr, missing)
        return

    imgs = {k: plt.imread(p) for k, p in paths.items()}
    w = {k: img.shape[1] / saver.dpi for k, img in imgs.items()}
    h = {k: img.shape[0] / saver.dpi for k, img in imgs.items()}

    rows = [["C2", "D2"], ["E"]]
    # Horizontal gap between C2 and D2 (row 1 only) -- same ROW1_GAP assemble_integrated_figure
    # uses between C and D.
    row_gaps = [[PVP_ROW1_GAP], []]
    row_w_natural = [sum(w[k] for k in row) + sum(gaps) for row, gaps in zip(rows, row_gaps)]
    row_h_natural = [max(h[k] for k in row) for row in rows]

    target_w = min(row_w_natural[0], PVP_WIDTH)
    row_scale = [target_w / rw for rw in row_w_natural]
    row_h = [rh * s for rh, s in zip(row_h_natural, row_scale)]
    LEGEND_GAP = 0.18  # inches reserved above row 1 for the shared method legend
    total_w, total_h = target_w, sum(row_h) + LEGEND_GAP

    positions = {}
    y = LEGEND_GAP
    for row, gaps, s, rh in zip(rows, row_gaps, row_scale, row_h):
        x = 0.0
        for i, k in enumerate(row):
            bw, bh = w[k] * s, h[k] * s
            positions[k] = (x, y, bw, bh)
            x += bw + (gaps[i] * s if i < len(gaps) else 0)
        y += rh

    fig = plt.figure(figsize=(total_w, total_h))
    for k, (x0_, y0_, bw, bh) in positions.items():
        ax = fig.add_axes([x0_ / total_w, 1 - (y0_ + bh) / total_h, bw / total_w, bh / total_h])
        ax.imshow(imgs[k])
        ax.axis("off")
    method_handles = [Patch(facecolor=S.method_color(m), label=m) for m in S.METHOD_ORDER]
    fig.legend(handles=method_handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              frameon=False, fontsize=5, handlelength=1.2, columnspacing=1.2)

    # Bold letter labels -- a (C2), b/c (D2's sensitivity/FDR), d/e (E's log2FC-error/CV rows) --
    # same sub-offset math as assemble_integrated_figure's own b/c/d/e (see its own comment), with
    # D2's own PVP_AXES_W/MAIN_INNER_GAP geometry (not MAIN_AXES_W/MAIN_INNER_GAP, since D2's
    # per-subplot axes width differs from plain D's own).
    label_pad = 0.05
    c2x, c2y, _c2bw, _c2bh = positions["C2"]
    dx, dy, dbw, _dbh = positions["D2"]
    ex, ey, ebw, _ebh = positions["E"]
    _native_d2_width = Y_LABEL_MARGIN + PVP_AXES_W + MAIN_INNER_GAP + PVP_AXES_W + PVP_UNIT_MARGIN
    fdr_x_offset = dbw * (Y_LABEL_MARGIN + PVP_AXES_W + MAIN_INNER_GAP) / _native_d2_width
    # E's letters are one per dataset COLUMN (d, e, f, ...), each at its column's left axes edge.
    sub_labels = ([("a", c2x, c2y), ("b", dx, dy), ("c", dx + fdr_x_offset - 0.35, dy)]
                  + _e_column_labels(ex, ey, ebw, n_datasets, PVP_WIDTH))
    for letter, x0_, y0_ in sub_labels:
        fig.text((x0_ + label_pad) / total_w, 1 - (y0_ + label_pad) / total_h, letter,
                 fontsize=9, fontweight="bold", ha="left", va="top")
    out_stem = ("Fig_Generalization_Integrated_precursor_vs_protein" if dest_attr == "main"
               else f"Supp_Generalization_Integrated_precursor_vs_protein_{contrast}")
    saver.save(fig, dest_dir, out_stem)


B_GRID_WIDTH = 4.0  # inches -- Supp_Generalization_B_completeness_by_method_all_datasets' own total
                    # width, one subplot per COMBINED_CONTRASTS entry.
B_GRID_HEIGHT = 1.8  # inches -- its height


def panel_completeness_by_method_grid(entries, SUPP, sv,
                                      stem="Supp_Generalization_B_completeness_by_method_all_datasets",
                                      total_width=B_GRID_WIDTH):
    """One combined figure, one subplot per COMBINED_CONTRASTS entry -- entries:
    [(contrast, datasets, coverage, completeness_transition), ...], each already the per-contrast
    combined (all-datasets) data _accumulate_and_maybe_combine returns. Draws the SAME stacked-by-
    method completeness bars as P.panel_completeness_by_method (method color, completeness-state
    alpha, per-segment raw counts) -- re-implemented here, not called, since that function always
    draws into (and saves) its own single-axes figure, with no way to target one subplot of a
    larger one; it's shared with the full run() pipeline's own single-contrast Main B, so changing
    its own signature to support this wasn't the simplest path. One shared legend (method colors +
    completeness-state alphas) above both subplots, one shared y-axis (sharey, left subplot only
    keeps tick labels; later ones show tick labels again, as the y-axis is the same but easier to read) instead of each subplot repeating both -- this is what makes it a genuine
    single figure rather than panel_completeness_by_method's own output merged after the fact."""
    n = len(entries)
    fig, axs = plt.subplots(1, n, figsize=(total_width, B_GRID_HEIGHT), sharey=True, squeeze=False)
    axs = axs[0]
    wm = 0.38
    state_text_color = {P.COMPLETE: "white", P.PARTIAL: "black", P.INSUFFICIENT: "black"}
    for ax, (contrast, datasets, coverage, completeness_transition) in zip(axs, entries):
        x = np.arange(len(datasets))
        coverage_by_method = P._coverage_by_method(coverage, completeness_transition)
        for mi, method in enumerate(S.METHOD_ORDER):
            cov_m = coverage_by_method[method]
            totals_m = np.array([cov_m[cov_m.contrast == dd].n.sum() for dd in datasets], float)
            bottom_m = np.zeros(len(datasets))
            off = (-wm / 2, wm / 2)[mi]
            for state in P.COMPLETENESS_ORDER:
                v = np.array([cov_m[(cov_m.contrast == dd) & (cov_m.state == state)].n.sum()
                             for dd in datasets], float)
                pct = np.divide(100 * v, totals_m, out=np.zeros_like(v), where=totals_m > 0)
                ax.bar(x + off, pct, width=wm, bottom=bottom_m, color=S.method_color(method),
                      alpha=P.STATE_ALPHA[state], edgecolor=S.NEUTRAL, linewidth=.3)
                for xi, bi, pi, vi in zip(x, bottom_m, pct, v):
                    if pi > 4:  # skip labels on slivers too thin to hold text
                        ax.text(xi + off, bi + pi / 2, f"{int(vi):,}", ha="center", va="center",
                               fontsize=4.0, color=state_text_color[state])
                bottom_m += pct
        grouped_dataset_ticks(ax, datasets, unit=True)
        ax.set_title(S.clabel(contrast), fontsize=6)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        P.compact_axes(ax)
        ax.tick_params(axis="x", labelsize=X_TICK_LABEL_SIZE)
        ax.tick_params(axis="y", labelleft=True)  # sharey hides them on every subplot but the first
    axs[0].set_ylabel(f"{P.unit_label()} (%)")
    # Same table-style legend as the Performance section's completeness-by-method panel (same
    # absolute size: 1.73 x 0.315 in), centered above the subplots.
    fig_w, fig_h = fig.get_size_inches()
    leg_w, leg_h = .96 * P.bw, .14 * P.bh
    P.completeness_table_legend(fig, [(1 - leg_w / fig_w) / 2, .80, leg_w / fig_w, leg_h / fig_h])
    fig.subplots_adjust(left=.125, right=.95, top=.72, bottom=.16, wspace=.25)  # right: room for the last "min"
    sv(fig, SUPP, stem, tight=False)


def panel_benchmark_design(contrasts, SUPP, sv, stem="Supp_Generalization_benchmark_design"):
    """Stacked H/Y/E composition per benchmark sample (same layout as Supp_Performance_A_benchmark_design).
    The design is shared by every dataset, so this is drawn once, from P.GROUPS/P.MIX."""
    table_w = 1.75
    pw, ph = P.PANEL_A_SIZE
    labs = list(P.GROUPS)
    # Same bar width / pitch as Supp_Performance_A_benchmark_design (its axes hold P.DESIGN_REF_N_GROUPS
    # samples in (.92 - .16) * pw): this design has fewer samples, so the axes get narrower. The figure width and
    # the table stay where they are in the performance panel, so the two tables line up.
    unit = (.92 - .16) * pw / (P.DESIGN_REF_N_GROUPS + 1.25)
    fig_w = pw + table_w
    fig = plt.figure(figsize=(fig_w, ph))
    ax = fig.add_axes([.16 * pw / fig_w, .12, unit * (len(labs) + 1.25) / fig_w, .75 - .12])
    fmt = lambda v: int(v) if float(v).is_integer() else v
    P.hye_stack_bar_plot(ax, np.arange(len(labs)),
                         [np.array([fmt(P.MIX[l][k]) for l in labs]) for k in range(3)],
                         [S.SPECIES_COLORS[s] for s in S.SPECIES_ORDER], 0.8, 5,
                         labs, [S.species_label(s) for s in S.SPECIES_ORDER],
                         x_label_offset=1.15, y_label_offset=3)
    ax.set_yticks([]); ax.set_xlim(-1.65, len(labs) - 0.4); ax.set_ylim(0, 108)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(axis="both", length=0)
    # centred under the reference (performance) axes, not this narrower one, so the footnotes line up
    ref_center = (.16 + (.92 - .16) / 2) * pw / fig_w
    ax.text(ref_center, -0.08, f"{P.GROUP_SIZE} replicates per sample, 250 pg injection",
            transform=mtransforms.blended_transform_factory(fig.transFigure, ax.transAxes),
            ha="center", va="top", fontsize=5)
    P.benchmark_design_table(fig, [(pw + .05) / fig_w, .12, (table_w - .1) / fig_w, .75 - .12], contrasts)
    sv(fig, SUPP, stem, tight=False)  # exact size, like Supp_Performance_A_benchmark_design


def run(cfg, L, saver, resolve):
    P.PADJ_CUT = float(cfg.get("padj_cutoff", 0.05))
    # Overridable matrix row id column (see P.ID_COL's own module-level note) -- lets this stage run
    # on a non-precursor-level matrix (e.g. "Protein.Group" for a protein-level quantification
    # matrix) via a new "id_col" key in this stage's own config section, without touching any of the
    # shared performance.py functions' own logic. Defaults to the normal precursor-level column, so
    # every existing config (no "id_col" key) is unaffected.
    P.ID_COL = cfg.get("id_col", "Precursor.Id")
    # Optional: a DIFFERENT run()'s own "generalization/table" directory (e.g. a precursor-level
    # run's), read back via _read_combined_completeness_tables and compared against this run's own
    # MAIN_INTEGRATED_CONTRAST completeness in one true-shared-y-axis figure -- see
    # panel_completeness_precursor_vs_protein. None (the default, no "precursor_table_dir" config
    # key) skips that extra standalone figure entirely.
    precursor_table_dir = resolve(cfg.get("precursor_table_dir"))
    # Protein-level run (precursor_table_dir configured): draw ONLY the precursor-vs-protein figures
    # (C2/D2/E and their composite), for every COMBINED_CONTRASTS entry (A_vs_B and B_vs_C alike),
    # plus the all-datasets coverage grid (Supp A), completeness-by-method grid (Supp B), volcano
    # grid (Supp D) and the benchmark design -- none of the other standalone single-run panels.
    pvp_only = precursor_table_dir is not None
    bench = cfg.get("benchmark", {})
    # "contrasts" (an explicit [[g1, g2], ...] pair list) takes precedence when present -- see
    # configure_benchmark -- for benchmark designs that aren't a star around one shared reference
    # (e.g. a path: A_vs_B, B_vs_C). Falls back to "reference" (star topology) otherwise.
    P.configure_benchmark(bench.get("groups"), bench.get("reference", "S1"), bench.get("contrasts"))
    sv = saver.save
    T = L.table
    dataset = cfg.get("_dataset")
    suffix = f"_{dataset}" if dataset else ""
    key = dataset or "default"
    # make_figures.py injects _all_datasets alphabetically (astral_15m, astral_5m, ...); reorder to
    # 5 min before 15 min within each instrument so every combined panel's x-axis/columns read
    # shortest-to-longest gradient. Unlisted dataset keys keep their relative order, after the rest.
    all_datasets = sorted(cfg.get("_all_datasets") or [key],
                          key=lambda d: DATASET_ORDER.index(d) if d in DATASET_ORDER else len(DATASET_ORDER))
    # Tracks, per COMBINED_CONTRASTS entry (by contrast name), which of its C/D/E combined panels
    # were freshly built in THIS call -- each entry's own integrated composite
    # (assemble_integrated_figure) is attempted only once all three land in the same call (the
    # normal case: every cache accumulates the same all_datasets set in parallel).
    built_integrated = {letter: set() for letter in "CDE"}
    # "C2"/"D2": the Main entry's own precursor-vs-protein comparison panel and its matching
    # "Protein"-titled D variant (see panel_completeness_precursor_vs_protein/
    # panel_sensitivity_fdr_bar's own `title` parameter) -- tracked separately from "C"/"D" since
    # they only exist for the Main entry, and only when precursor_table_dir is configured; they
    # feed their own composite (assemble_integrated_figure_precursor_vs_protein), not
    # assemble_integrated_figure.
    built_integrated["C2"] = set()
    built_integrated["D2"] = set()
    # B's own per-contrast combined data (coverage, completeness_transition), collected here
    # instead of saved as a standalone panel -- panel_completeness_by_method_grid draws both
    # COMBINED_CONTRASTS entries as subplots of ONE figure, attempted only once both entries' data
    # land in the same call (see b_entries' use below).
    b_entries = {}

    # No fold-change cutoff: a detection is significance alone (p_adj), and the sign of log2FC only
    # decides TP vs wrong_direction (yeast up, E. coli down) -- same rule as the performance stage.
    log2fc_cutoff = None
    contrasts, prepared, preprepared = P.load_pairs_and_prepped(cfg, resolve, log2fc_cutoff=log2fc_cutoff)

    completeness, curve, n_runs, n_total, pad = P.compute_coverage_curve(contrasts, preprepared)
    write_tsv(completeness, T, f"precursor_completeness_30runs{suffix}.tsv.gz")
    write_tsv(curve, T, f"precursor_completeness_cumulative_curve{suffix}.tsv")
    # Accumulate only -- see module docstring. The combined figure is built once every dataset in
    # _all_datasets has been accumulated (by whichever call completes that set).
    _coverage_cache[key] = (curve, n_runs, n_total, pad)
    if set(all_datasets) <= set(_coverage_cache):
        entries = [(name, *_coverage_cache[name]) for name in all_datasets]
        panel_coverage_grid(entries, L.supp, sv, stem="Supp_Generalization_A_coverage_all_datasets",
                           total_width=4.8)
        _coverage_cache.clear()
        # The benchmark design is dataset-independent, so it is drawn once, by the protein-level run.
        if pvp_only:
            panel_benchmark_design(contrasts, L.supp, sv)

    coverage, completeness_transition, restored = P.compute_completeness_by_method(contrasts, prepared, preprepared)
    write_tsv(coverage, T, f"completeness_overall{suffix}.tsv")
    write_tsv(completeness_transition, T, f"completeness_transition_by_contrast{suffix}.tsv")
    for contrast, dest_attr in COMBINED_CONTRASTS:
        if contrast not in set(coverage["contrast"]):
            log.warning("generalization: contrast '%s' not found for %s -- skipping it in the "
                        "combined completeness-by-method panel.", contrast, key)
            continue
        combined = _accumulate_and_maybe_combine(
            _completeness_caches[contrast], key, all_datasets,
            (coverage[coverage.contrast == contrast].assign(contrast=key),
             completeness_transition[completeness_transition.contrast == contrast].assign(contrast=key)))
        if combined is not None:
            b_entries[contrast] = (all_datasets, *combined)
            # Every COMBINED_CONTRASTS entry's own call renders at MAIN_PANEL_SIZE (smaller than
            # the function's own default) with ABSOLUTE-inch margins (not P.INTEGRATED_LEFT/RIGHT/
            # BOTTOM's own fractions, which would shrink right along with it) so each entry's own
            # integrated composite row 1 is already the right size natively, labels intact.
            size_kwargs = {"panel_size": MAIN_PANEL_SIZE,
                           "margins": (Y_LABEL_MARGIN, MAIN_UNIT_MARGIN, X_LABEL_MARGIN)}
            if not pvp_only:
                panel_completeness_complete_only(all_datasets, *combined, getattr(L, dest_attr), sv,
                                                  stem=_combined_stem("C_completeness_by_method_complete_only",
                                                                      contrast, dest_attr),
                                                  legend=False, **size_kwargs)
                built_integrated["C"].add(contrast)
            # True shared-y-axis precursor-vs-protein comparison (see
            # panel_completeness_precursor_vs_protein) -- only for the Main entry, and only when a
            # DIFFERENT run()'s own generalization/table directory was given (see
            # precursor_table_dir's own note above). Feeds its own composite, row 1, alongside this
            # entry's own D and E -- see assemble_integrated_figure_precursor_vs_protein below; it
            # never replaces assemble_integrated_figure's own "C" (still this run's own data alone).
            if pvp_only:
                missing =[f for name in all_datasets
                          for f in (f"completeness_overall_{name}.tsv",
                                    f"completeness_transition_by_contrast_{name}.tsv")
                          if not (precursor_table_dir / f).exists()]
                if missing:
                    log.warning("panel_completeness_precursor_vs_protein skipped: %s is missing %s "
                                "-- has the precursor-level run happened yet?",
                                precursor_table_dir, missing)
                else:
                    # panel_size/legend/margins/axes_w: sized and marginned to feed
                    # assemble_integrated_figure_precursor_vs_protein's row 1 at its native size
                    # (no post-hoc down-scaling), with axes_w shared with D2's own call below so
                    # every bar in that row renders at the same absolute width -- see
                    # PVP_AXES_W/PVP_TOP_MARGIN's own notes.
                    panel_completeness_precursor_vs_protein(
                        precursor_table_dir, all_datasets, contrast, *combined, getattr(L, dest_attr), sv,
                        stem=_combined_stem("C_completeness_precursor_vs_protein", contrast, dest_attr),
                        panel_size=MAIN_PANEL_SIZE, legend=False, axes_w=PVP_AXES_W,
                        margins=(Y_LABEL_MARGIN, PVP_UNIT_MARGIN, X_LABEL_MARGIN, PVP_TOP_MARGIN))
                    built_integrated["C2"].add(contrast)

    if set(c for c, _ in COMBINED_CONTRASTS) <= set(b_entries):
        entries = [(contrast, *b_entries[contrast]) for contrast, _ in COMBINED_CONTRASTS]
        panel_completeness_by_method_grid(entries, getattr(L, BY_METHOD_DEST), sv)

    partial_perf, rec_perf, _sens_ylim, _fdr_ylim = P.compute_sensitivity_fdr(contrasts, prepared, restored)
    # Volcano of the exact proteins behind D's sensitivity/FDR (partially-complete stratum, both
    # methods): one combined figure per COMBINED_CONTRASTS entry (all datasets, Astral on row 1,
    # timsTOF on row 2, shared axis limits) -- always Supp. Built by whichever call completes the set.
    for contrast, _dest in COMBINED_CONTRASTS:
        if contrast in prepared:
            _volcano_caches[contrast][key] = prepared[contrast][2]
            if set(all_datasets) <= set(_volcano_caches[contrast]):
                names = sorted(all_datasets, key=lambda n: DATASET_ORDER.index(n) if n in DATASET_ORDER else len(DATASET_ORDER))
                fig = DE.plot_benchmark_volcano_grid(
                    [(dataset_label(n), _volcano_caches[contrast][n]) for n in names],
                    [(name, suf) for name, suf in P.METHOD_SUFFIX], P.PARTIAL, P.PADJ_CUT,
                    logfc_cutoff=log2fc_cutoff)
                sv(fig, L.supp, f"Supp_Generalization_D_volcano_all_datasets_{contrast}")
                _volcano_caches[contrast].clear()
    write_tsv(partial_perf, T, f"partial_sensitivity_fdr{suffix}.tsv")
    write_tsv(rec_perf, T, f"restored_sensitivity_fdr{suffix}.tsv")
    insufficient_combined = {}  # contrast -> all-datasets restored table, filled once the last dataset arrives
    for contrast, dest_attr in COMBINED_CONTRASTS:
        if contrast not in set(partial_perf["contrast"]):
            log.warning("generalization: contrast '%s' not found for %s -- skipping it in the "
                        "combined sensitivity/FDR panel.", contrast, key)
            continue
        combined = _accumulate_and_maybe_combine(
            _sensitivity_caches[contrast], key, all_datasets,
            (partial_perf[partial_perf.contrast == contrast].assign(contrast=key),
             rec_perf[rec_perf.contrast == contrast].assign(contrast=key)))
        if combined is not None:
            combined_partial, combined_rec = combined
            # Re-matched across the combined (4-dataset) data -- the per-dataset sens_ylim/fdr_ylim
            # compute_sensitivity_fdr returned above aren't reused here, since this combined
            # figure's y-axis has to span every dataset's values, not just one. Matched against
            # the partially-complete stratum only, since that's all this bar panel now plots.
            sens_ylim = P._matched_ylim(combined_partial.sensitivity * 100, floor_zero=True)
            fdr_ylim = P._matched_ylim(combined_partial.empirical_fdr * 100, floor_zero=True)
            # Same MAIN_PANEL_SIZE/MAIN_INNER_GAP/ABSOLUTE-margin reasoning as
            # panel_completeness_complete_only's own call above, for every COMBINED_CONTRASTS entry
            # -- see that call's comment. D's own margins tuple also carries MAIN_AXES_W (shared
            # with C, so their bars render at the same absolute width -- see
            # panel_sensitivity_fdr_bar's own docstring).
            size_kwargs = {"panel_size": MAIN_PANEL_SIZE, "inner_gap": MAIN_INNER_GAP,
                           "margins": (Y_LABEL_MARGIN, MAIN_UNIT_MARGIN, X_LABEL_MARGIN, MAIN_AXES_W)}
            if not pvp_only:
                panel_sensitivity_fdr_bar(all_datasets, combined_partial, sens_ylim, fdr_ylim,
                                            getattr(L, dest_attr), sv,
                                            stem=_combined_stem("D_sensitivity_fdr", contrast, dest_attr),
                                            legend=False, **size_kwargs)
                built_integrated["D"].add(contrast)
            # A SEPARATE "Protein"-titled D2 variant, its axes_w matching C2's own (PVP_AXES_W) --
            # feeds assemble_integrated_figure_precursor_vs_protein's row 1 instead of
            # assemble_integrated_figure's own row 1 (which keeps the original, untitled,
            # MAIN_AXES_W-wide D). Same (dest_attr == "main" and precursor_table_dir is not None)
            # gate as C2's own build above.
            if pvp_only:
                panel_sensitivity_fdr_bar(
                    all_datasets, combined_partial, sens_ylim, PVP_FDR_YLIM, getattr(L, dest_attr), sv,
                    stem=_combined_stem("D_sensitivity_fdr_precursor_vs_protein", contrast, dest_attr),
                    legend=False, title="Protein", grouped_ticks=True,
                    panel_size=MAIN_PANEL_SIZE, inner_gap=MAIN_INNER_GAP,
                    margins=(Y_LABEL_MARGIN, PVP_UNIT_MARGIN, X_LABEL_MARGIN, PVP_AXES_W, PVP_TOP_MARGIN))
                built_integrated["D2"].add(contrast)
            insufficient_combined[contrast] = combined_rec
    # Insufficient (restored by JUMPlion) stratum: JUMPlion-only sensitivity/FDR bars, always Supp,
    # both contrasts in ONE figure with y-ranges matched across them.
    if set(c for c, _ in COMBINED_CONTRASTS) <= set(insufficient_combined):
        panel_sensitivity_fdr_insufficient_grid(
            [(c, insufficient_combined[c]) for c, _ in COMBINED_CONTRASTS], all_datasets, L.supp, sv)

    # Only the per-precursor tables are kept here -- Fig_Generalization_E_quantitative_fidelity
    # (panel_cv_deltafc, still used by the full run() pipeline's own Main E) isn't drawn for
    # generalization; partial_cv/partial_fc/restored_cv/restored_fc still feed the species-split
    # box grids below.
    partial_cv, partial_fc, restored_cv, restored_fc, _cv_ylim, _fc_ylim = P.compute_cv_deltafc(
        contrasts, prepared, restored)
    write_tsv(partial_cv, T, f"partial_CV_pooled_values{suffix}.tsv")
    write_tsv(partial_fc, T, f"partial_delta_FC_precursor_level{suffix}.tsv.gz")
    write_tsv(restored_cv, T, f"restored_CV_pooled_values{suffix}.tsv")
    write_tsv(restored_fc, T, f"restored_delta_FC_precursor_level{suffix}.tsv.gz")

    # Species-split log2FC error / CV box grid (one combined figure -- see
    # panel_log2fc_error_cv_box_grid), built once per entry in COMBINED_CONTRASTS (not pooled
    # across a dataset's other contrasts) -- one column per dataset, kept separate (not
    # concatenated), so _accumulate_and_maybe_combine doesn't fit here.
    for contrast, dest_attr in COMBINED_CONTRASTS:
        fc_ok = contrast in set(partial_fc["contrast"])
        cv_ok = contrast in set(partial_cv["contrast"])
        if not (fc_ok and cv_ok):
            log.warning("generalization: contrast '%s' not found for %s in %s -- skipping it in "
                        "the combined log2FC_error/CV box grid.", contrast, key,
                        "partial_fc" if not fc_ok else "partial_cv")
            continue
        # Partial kept as BOTH methods now (no more .method == P.M_J filter) -- the combined figure
        # compares DIA-NN vs JUMPlion for the partially-complete stratum; restored stays
        # JUMPlion-only (it has no DIA-NN counterpart by definition) and feeds its own separate
        # figure below instead of being combined into the same one.
        _box_grid_caches[contrast][key] = (
            (partial_fc[partial_fc.contrast == contrast], restored_fc[restored_fc.contrast == contrast]),
            (partial_cv[partial_cv.contrast == contrast], restored_cv[restored_cv.contrast == contrast]))
        if set(all_datasets) <= set(_box_grid_caches[contrast]):
            fc_entries = [(name, _box_grid_caches[contrast][name][0][0]) for name in all_datasets]
            cv_entries = [(name, _box_grid_caches[contrast][name][1][0]) for name in all_datasets]
            fc_restored_entries = [(name, _box_grid_caches[contrast][name][0][1]) for name in all_datasets]
            cv_restored_entries = [(name, _box_grid_caches[contrast][name][1][1]) for name in all_datasets]
            # Every COMBINED_CONTRASTS entry's own call gets a narrower col_width, sized so this
            # panel's saved width already matches row 1 (C+D) in assemble_integrated_figure -- see
            # panel_log2fc_error_cv_box_grid's col_width docstring note -- plus E_LEFT_MARGIN
            # (ABSOLUTE, not the default `left=.08` fraction, which would shrink along with that
            # narrower col_width and clip the y-axis label). The restored-only sibling isn't part
            # of any composite, so it keeps its own default (standalone) sizing instead.
            kwargs = {"col_width": MAX_INTEGRATED_WIDTH / len(all_datasets), "left_margin": E_LEFT_MARGIN,
                      "compact_x": True}  # same E format as the precursor-vs-protein one
            if not pvp_only:
                panel_log2fc_error_cv_box_grid(
                    fc_entries, cv_entries, getattr(L, dest_attr), sv,
                    stem=_combined_stem("E_log2FC_error_CV_box", contrast, dest_attr), **kwargs)
            # Same E grid at PVP_WIDTH (not MAX_INTEGRATED_WIDTH) for
            # assemble_integrated_figure_precursor_vs_protein, so it tiles unscaled under C2/D2.
            if pvp_only:
                panel_log2fc_error_cv_box_grid(
                    fc_entries, cv_entries, getattr(L, dest_attr), sv,
                    stem=_combined_stem("E_log2FC_error_CV_box_precursor_vs_protein", contrast, dest_attr),
                    col_width=PVP_WIDTH / len(all_datasets), left_margin=E_LEFT_MARGIN, compact_x=True)
            else:
                panel_log2fc_error_cv_box_grid_restored(
                    fc_restored_entries, cv_restored_entries, getattr(L, dest_attr), sv,
                    stem=_combined_stem("E_log2FC_error_CV_box_restored", contrast, dest_attr), **kwargs)
            _box_grid_caches[contrast].clear()
            built_integrated["E"].add(contrast)

    for contrast, dest_attr in COMBINED_CONTRASTS:
        if not pvp_only and all(contrast in built_integrated[letter] for letter in "CDE"):
            assemble_integrated_figure(L, saver, contrast, dest_attr, len(all_datasets))
        if all(contrast in built_integrated[letter] for letter in ("C2", "D2", "E")):
            assemble_integrated_figure_precursor_vs_protein(L, saver, contrast, dest_attr, len(all_datasets))

    log.info("generalization finished%s.", f" ({dataset})" if dataset else "")
