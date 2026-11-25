"""Mechanism: localization-tier recovery + intensity mapping + final-recovered precursor-run
outcomes, merged into one pipeline stage (former signal_recovery_mechanism, former
intensity_mapping_mechanism, former final_recovered).

Inputs: the shared precursor-run features/quantification parquet (evidence_table ==
quantification_parquet, read ONCE), precursor_mapping_model_metrics.tsv, a precursor-level
species annotation CSV (Precursor.Id, Species, read ONCE), plus the RT/IM calibration tables and
XIC files used by the localization-tier panels.

Main panels (the end-to-end recovered-subset story): Fig_Mechanism_A/B/C/D +
Fig_Mechanism_combined. Everything else -- the localization-tier workflow, evidence-feature
distributions, XIC examples, intensity-mapping gates/funnel, and reliability diagnostics -- is
Supp. The two mechanisms' workflow flowcharts are combined into one Supp figure
(Supp_Mechanism_recovery_and_mapping_workflow).

Outputs go to mechanism/{main,supp,table}. Tables are prefixed by the panel they support; tables
that back no panel are prefixed "Table_".
"""
from __future__ import annotations

import ast
import io
import json
import logging
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.transforms import Bbox

from . import style as S
from .common import read_table, write_tsv

try:
    from .intensity_mapping_reliability import run_reliability
except ImportError:
    from intensity_mapping_reliability import run_reliability

log = logging.getLogger("jumplion_figures.mechanism")

# ============================================================================
# Constants -- localization-tier recovery (former signal_recovery_mechanism.py)
# ============================================================================
# Evidence.Tier is a decision-tree outcome, not an ordinal quality scale:
#   ms2_coelution   MS2-primary route succeeds
#   ms1_peak        MS2 route fails, MS1 fallback succeeds
#   prediction_only neither route succeeds (shown as "No qualifying peak")
#   reported        DIA-NN-reported precursor-run
TIER_COL = "Evidence.Tier"
TIERS = S.TIER_ORDER  # reported, ms2_coelution, ms1_peak, prediction_only
# Tiers with no meaningful "detected peak" distinct from the predicted RT: prediction_only never
# found one. reported has its own real apex/fwhm (it still went through XIC extraction and peak
# fitting -- it just skipped the MS1/MS2 qualifying-peak decision), so it's not included here; XIC
# example plots skip the apex line/shading only for the tiers below.
NO_DETECTED_PEAK_TIERS = {"prediction_only"}
MS1_FEATURES = ["ms1_xics_signal_rank", "ms1_corr_mean", "ms1_isotope_cosine"]
MS2_FEATURES = ["ms2_xics_signal_rank", "ms2_corr_mean", "ms2_fragment_cosine"]
ALL_FEATURES = MS1_FEATURES + MS2_FEATURES
OPTIONAL_COLS = ["ms1_usable_xic_fraction_local", "ms2_usable_xic_fraction_local", "RT.Query", "RT.Apex",
                 "ms1_corr_sum", "ms2_corr_sum"]
PRECURSOR_COLS = ["Precursor.Id", "precursor_id", "PrecursorID"]
RUN_COLS = ["run_name", "Run", "run", "File.Name", "FileName"]
COELUTION_COLS = {"ms1_corr_mean", "ms2_corr_mean", "ms1_coelution_score", "ms2_coelution_score",
                  "ms1_corr_sum", "ms2_corr_sum"}

# Localization-workflow Flowchart geometry -- used only by the combined workflow panel
# (_build_localization_workflow below), not a standalone composite.
PANEL_A_XLIM = (0.95, 11.5)  # left: "Tier 1" (F.MID-wide) reaches x=1.35, 1.5 clipped it; right:
                             # "Tier 4" reaches x=11.1, 11 clipped it; both widened by 0.3 for the
                             # "Signal localization" container, which extends 0.25 past the boxes
PANEL_A_CONTAINER_HEAD = 0.75  # extra gap between XIC extraction and the status box (container title room)
PANEL_A_SCALE = 0.42
PANEL_A_BRANCH_GAP = 0.5  # horizontal edge-to-edge gap between sibling branch boxes
PANEL_A_BRANCH_LEFT_BIAS = 0.8  # pulls the ms2/q1 and ms1/none branch pairs left of their
                                # parent's x, so the tree drifts right less and stays compact
PANEL_A_Y0 = 7.05
WORKFLOW_A_YLIM = (-4.7, 8.95)  # (was -3.95 before the container head and the "RT boundaries" box; the H1 boxes then took 1.8 back) bottom: raw_quant sits at yc-BRANCH_ROW (not yc-ROW), so its
                                 # bottom edge is at -3.85, not -3.6 -- -3.7 was clipping it

# Row/branch pitch shared by BOTH combined-workflow flowcharts (_build_localization_workflow AND
# _build_mapping_workflow) -- one knob to tune box spacing in either, instead of each workflow
# having its own independently-tuned values. Taller than either flowchart's box widths alone
# would need (same box sizes, just more vertical gap). The two workflows still come out at
# different absolute heights despite sharing this pitch, since they chain through a different
# number of boxes -- that's deliberate (see _build_mapping_workflow's own comment). Changing this
# value changes each workflow's total content height, so each workflow's own ylim (WORKFLOW_A_YLIM
# here; the ylim passed to S.Flowchart directly in _build_mapping_workflow) needs recomputing to
# match -- see either function's own geometry comments for the margin convention used.
WORKFLOW_ROW, WORKFLOW_BRANCH_ROW = 1.35, 1.6

# ============================================================================
# Constants -- intensity mapping (former intensity_mapping_mechanism.py)
# ============================================================================
ID, RUN, DIANN, RAW = "Precursor.Id", "Run", "DIANN.Quantity", "JUMPlion.Raw.Quantity"
FINAL, STATUS, TIER, SPECIES = "Final.Quantity", "Final.Status", "evidence_tier_label", "Species"
REQ_PARQUET = [ID, RUN, DIANN, RAW, FINAL, STATUS, TIER]
REQ_METRICS = [ID, "N.Overlap", "N.Overlap.Fraction", "Effective.Min.Overlap", "Slope", "Intercept", "R2",
               "RMSE", "MAE", "Pearson", "Spearman", "Model.Good", "Model.Fail.Reason"]
SPECIES_ORDER = S.SPECIES_ORDER
INT_XLABEL = "Mean observed DIA-NN intensity (log2)"
RAW_XLABEL = "JUMPlion raw precursor quantity (log2)"
DIANN_YLABEL = "DIA-NN precursor quantity (log2)"

# ============================================================================
# Constants -- final recovered (former final_recovered.py; stays Main, unchanged)
# ============================================================================
# The 3 localization/recovery tiers -- "reported" (already DIA-NN-reported, no recovery involved)
# is excluded from this list, since most of this section is specifically about precursor-runs that
# needed recovery; Fig_Mechanism_A below adds "reported" back in as a reference bar.
RECOVERY_TIERS = ["ms2_coelution", "ms1_peak", "prediction_only"]
PANEL_A_TIERS = ["reported"] + RECOVERY_TIERS

# Fig_Mechanism_combined's left column (A over C) are two independently-saved, same-width
# (2.4in) panels; each one's own fig.tight_layout() picks its own left/right margin from its own
# content (e.g. A's y-axis needs more room for 6-digit tick labels than C's 5-digit one), so their
# plot boxes don't line up even though the two canvases are the same width. Forcing the SAME
# explicit left/right fraction on both after tight_layout (generous enough for A's wider labels)
# makes their axes boxes -- and so their bars/x-axis -- align in the composite.
COL1_LEFT, COL1_RIGHT = 0.25, 0.95

# Right column (B over D): both are now ONE 4.4in canvas with 2 internal subplots (D's Yeast/E.
# coli sharey subplots mirror B's signal-rank/co-elution subplots structurally), so using the
# exact same left/right/wspace for both directly lines up their columns and their inner seam.
# COL2_B_RIGHT leaves a wide right margin (not just the usual thin edge) -- panel B's shared
# legend sits there, just outside its second subplot's axes; D doesn't need that margin but has to
# share the same value so the two panels' subplot axes boxes still line up (see both callers).
COL2_B_LEFT, COL2_B_RIGHT, COL2_B_WSPACE = 0.10, 0.97, 0.27

# The left/right fix above only aligns columns (A-over-C, B-over-D); it does nothing for the
# x-axis height WITHIN a row, which is set independently by each panel's own bottom margin --
# auto (tight_layout) for A/C/D, hardcoded at .28 for B (four_tier_violins' own default). A's
# x-tick labels are the same rotated tier names as B's, so .28 suits both; pinning it explicitly
# on A (instead of leaving it to tight_layout) puts their x-axis baselines at the same height.
# C and D both carry an xlabel ("Sample group") below their non-rotated group-label ticks, so
# both need the same bottom room -- pin both to one explicit value.
ROW1_BOTTOM = 0.22
ROW2_BOTTOM = 0.18

# Shared size for every in-panel annotation (bar-count labels, legends) across the final-recovered
# section, so they read as one consistent text size.
ANNOT_FONTSIZE = 6

# For panel B only (not Supp or other tier-colored panels elsewhere): the 3 recovery tiers all
# share one green instead of their usual per-tier red/yellow/gray -- de-emphasizes the
# tier-vs-tier distinction here since this panel is about reported-vs-recovered, not which
# recovery tier. "reported" keeps its own color.
RECOVERED_GREEN = "#59A14F"
PANEL_AB_COLORS = {t: (RECOVERED_GREEN if t in RECOVERY_TIERS else S.tier_color(t)) for t in PANEL_A_TIERS}

MAIN_PANEL_FILES_COMBINED = {
    "A": "Fig_Mechanism_A_tier_counts_before_after",
    "B": "Fig_Mechanism_B_signal_rank_coelution",
    "C": "Fig_Mechanism_C_yeast_ecoli_by_group",
    "D": "Fig_Mechanism_D_signal_rank_by_group",
}
ROW_GAP = 0.12  # inches of blank space between the A/B row and the C/D row
TOP_GAP = 0.12  # inches of blank space above the A/B row
LABEL_PAD = 0  # inches inset from a panel's top-left corner for its bold letter label


# ----------------------------------------------------------------------------
# Shared loading / validation
# ----------------------------------------------------------------------------
def _tier_filtered(raw):
    d = raw.copy()
    if TIER_COL not in d.columns and "evidence_tier" in d.columns:
        d[TIER_COL] = d["evidence_tier"]
    missing = [c for c in [TIER_COL] + ALL_FEATURES if c not in d.columns]
    if missing:
        raise KeyError(f"Evidence table missing columns: {missing}")
    d[TIER_COL] = d[TIER_COL].astype(str)
    return d[d[TIER_COL].isin(TIERS)].copy()


def validate_precursor_run_universe(evidence):
    pcol = next((c for c in PRECURSOR_COLS if c in evidence.columns), None)
    rcol = next((c for c in RUN_COLS if c in evidence.columns), None)
    res = dict(n_rows=int(len(evidence)), precursor_col=pcol, run_col=rcol, n_precursors=None, n_runs=None,
               expected_cartesian_n=None, is_complete_cartesian=None, n_duplicate_precursor_runs=None)
    if pcol is None or rcol is None:
        log.warning("Could not verify n_precursor x n_run: precursor/run ID columns not found.")
        return res
    dup = int(evidence.duplicated([pcol, rcol]).sum())
    np_, nr_ = int(evidence[pcol].nunique()), int(evidence[rcol].nunique())
    complete = dup == 0 and len(evidence) == np_ * nr_
    res.update(n_precursors=np_, n_runs=nr_, expected_cartesian_n=np_ * nr_,
               is_complete_cartesian=bool(complete), n_duplicate_precursor_runs=dup)
    log.info("Evidence universe: rows=%s precursors=%s runs=%s duplicates=%s complete=%s",
             f"{len(evidence):,}", f"{np_:,}", f"{nr_:,}", f"{dup:,}", complete)
    if not complete:
        log.warning("Evidence rows are not a duplicate-free precursor x run Cartesian product; "
                    "describe the denominator as the eligible precursor-run universe.")
    return res


def _col(evidence, feature):
    v = pd.to_numeric(evidence[feature], errors="coerce")
    # Negative correlation = zero positive co-elution evidence (same definition everywhere).
    return v.clip(lower=0.0) if feature in COELUTION_COLS else v


def _tier_groups(evidence, feature, tiers=TIERS):
    v = _col(evidence, feature)
    t = evidence[TIER_COL]
    return [v[t.eq(tier)].dropna().to_numpy(float) for tier in tiers]


# Short horizontal tick labels for the four evidence groups (Fig_Mechanism A and B).
TIER_TICK_LABELS = {"reported": "DIA-NN\nreported", "ms2_coelution": "Co-eluting\nfragments",
                    "ms1_peak": "Isotope\nenvelope", "prediction_only": "No qualifying\npeak"}


def two_level_tier_axis(fig, ax, tiers):
    """x labels shared by Fig_Mechanism A and B: horizontal two-line names under the recovery tiers
    (none under "reported"), then a group row below -- "DIA-NN reported" under the first group, and
    a bracket spanning the recovery tiers labelled "JUMPlion evaluated". Groups sit at x = 0..n-1."""
    from matplotlib.transforms import ScaledTranslation
    x = np.arange(len(tiers))
    ax.set_xticks(x)
    ax.set_xticklabels(["" if t == "reported" else TIER_TICK_LABELS[t] for t in tiers], fontsize=5.5)

    def _below(dy_pt):  # x in data units, y just below the axes by dy_pt points
        return ax.get_xaxis_transform() + ScaledTranslation(0, -dy_pt / 72, fig.dpi_scale_trans)
    ax.plot([x[1] - .4, x[-1] + .4], [0, 0], transform=_below(20), color=S.NEUTRAL, lw=.6, clip_on=False)
    ax.text(x[0], 0 + 0.06, "DIA-NN\nreported", transform=_below(22), ha="center", va="top", fontsize=6.5)
    ax.text((x[1] + x[-1]) / 2, 0, "JUMPlion evaluated", transform=_below(22), ha="center", va="top",
            fontsize=6.5)


def _tier_axis(ax, tiers=TIERS, rotation=22, short_labels=False, fig=None):
    if short_labels:
        two_level_tier_axis(fig or ax.figure, ax, tiers)
        return
    ax.set_xticks(range(len(tiers)))
    ax.set_xticklabels([S.tier_label(t) for t in tiers], rotation=rotation, ha="right")


def as_bool(s):
    if s.dtype == bool:
        return s.fillna(False)
    return s.fillna(False).astype(str).str.lower().isin(["true", "1", "yes", "y"])


def log2_series(s):
    x = pd.to_numeric(s, errors="coerce")
    return np.log2(x.where(x > 0))


def validate(df, required, label):
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise KeyError(f"{label} is missing required columns: {miss}")


def _load_species_annotation(sp_path):
    """Precursor.Id -> upper-cased, stripped Species, as a Series named "Species" -- shared by
    the intensity-mapping section (joined onto per-precursor tables) and the final-recovered
    section (mapped onto per-row tables), read from disk exactly once."""
    sp = pd.read_csv(sp_path, usecols=["Precursor.Id", "Species"]).drop_duplicates("Precursor.Id")
    sp["Species"] = sp["Species"].astype(str).str.upper().str.strip()
    return sp.set_index("Precursor.Id")["Species"]


# ----------------------------------------------------------------------------
# Combined recovery + mapping workflow (NEW: the two former Main-A flowcharts, one Supp figure)
# ----------------------------------------------------------------------------
def _build_localization_workflow():
    F = S.Flowchart(xlim=PANEL_A_XLIM, ylim=WORKFLOW_A_YLIM, inch_per_unit=PANEL_A_SCALE)
    F.ROW, F.BRANCH_ROW = WORKFLOW_ROW, WORKFLOW_BRANCH_ROW  # see the comment above their definition

    def sib(parent_x, lw, rw):
        """x for a left/right sibling pair centered under `parent_x`, PANEL_A_BRANCH_GAP apart
        edge-to-edge given their own widths `lw`/`rw` -- shrink PANEL_A_BRANCH_GAP to pull
        branches closer together."""
        half_sep = (lw + rw) / 2 + PANEL_A_BRANCH_GAP
        return parent_x - half_sep / 2, parent_x + half_sep / 2

    H1, H2 = F.BOX_H1, F.BOX_H2

    def drop(h_above, h_below, pitch=None):
        """Centre-to-centre pitch that leaves the same edge-to-edge gap as `pitch` (default BRANCH_ROW; ROW for
        the main chain) does between two full-height (H2) boxes, whatever the heights of the two boxes are."""
        return (pitch or F.BRANCH_ROW) - (2 * H2 - h_above - h_below) / 2

    y = PANEL_A_Y0
    missing_q = F.box(5.0, y + F.ROW, "Precursor-run entries", h=H1)
    b1 = F.box(5.0, missing_q["y"] - drop(H1, H1, F.ROW), "Initial RT (reported/predicted)", h=H1)
    b2 = F.box(5.0, b1["y"] - drop(H1, H1, F.ROW), "RT search region (RT±window)", h=H1)
    b3 = F.box(5.0, b2["y"] - drop(H1, H1, F.ROW), "XIC extraction", h=H1)
    # extra drop below XIC extraction: room for the "Signal localization" container's title (drawn at the end).
    # Placed from b3's bottom edge so its distance to the container is independent of b3's height: with
    # two-line (H2) boxes above and below, the status box's centre sat ROW + PANEL_A_CONTAINER_HEAD below b3's.
    quant_status = F.box(5.0, b3["bottom"] - (F.ROW + PANEL_A_CONTAINER_HEAD - H2 / 2 - 0.04),
                         "DIA-NN quantification status?", h=H1)
    yr = quant_status["y"] - drop(H1, H1)
    xr_l, xr_r = sib(quant_status["x"], F.MID, F.MID)
    reported = F.box(xr_l, yr, "DIA-NN reported RT", w=F.MID, h=H1, face=S.tier_color("reported"))
    q2 = F.box(xr_r, yr, "Qualifying MS2 co-elution?", w=F.MID, h=H1)
    yb = q2["y"] - drop(H1, H2)  # q2 (H1) -> ms2 / q1 (H2)
    xb_l, xb_r = sib(q2["x"] - PANEL_A_BRANCH_LEFT_BIAS, F.MID, F.WIDE)
    ms2 = F.box(xb_l, yb, "MS2 co-elution-refined RT", w=F.MID, h=H1, face=S.tier_color("ms2_coelution"))
    q1 = F.box(xb_r, yb, "Qualifying MS1 isotope peak\nnear predicted RT?")
    yc = yb - drop(H2, H1)  # q1 (H2) -> ms1 / none (H1)
    xc_l, xc_r = sib(q1["x"] - PANEL_A_BRANCH_LEFT_BIAS, F.MID, F.MID)
    ms1 = F.box(xc_l, yc, "MS1 isotope-refined RT", w=F.MID, h=H1, face=S.tier_color("ms1_peak"))
    none = F.box(xc_r, yc, "Predicted RT only", w=F.MID, h=H1, face=S.tier_color("prediction_only"))
    # Convergence box: all four localization-tier outcomes feed into the same raw precursor
    # quantity that the mapping workflow (_build_mapping_workflow) starts from -- same label/style
    # as that workflow's own first box, so the two visually read as the same handoff object.
    # The four tiers converge on "RT boundaries for quantification" (inside the "Signal localization"
    # container), which then hands off to the raw quantity box below the container.
    rt_box = F.box(5.0, yc - drop(H1, H2), "RT-centered\nquantification window")
    raw_quant = F.box(5.0, rt_box["y"] - F.ROW - 0.1, "JUMPlion raw\nprecursor quantity")
    F.arrow(missing_q, b1)
    F.arrow(b1, b2); F.arrow(b2, b3)  # b3 -> container arrow is drawn with the container, below
    F.arrow(quant_status, reported, "reported", dx=-0.3); F.arrow(quant_status, q2, "missing", dx=0.3)
    F.arrow(q2, ms2, "yes", dx=-0.3); F.arrow(q2, q1, "no", dx=0.3)
    F.arrow(q1, ms1, "yes", dx=-0.3); F.arrow(q1, none, "no", dx=0.3)

    # Three of the four convergence arrows are routed as TWO plain line segments (down a clear
    # vertical lane, then across a shared "manifold" height) plus one short real arrow for the
    # final vertical drop into raw_quant -- so only that last segment carries an arrowhead, and
    # it points straight down into the box like every other arrow here, instead of sideways (a
    # single-corner elbow straight into raw_quant's center would arrive horizontally). Landing
    # points are spread across raw_quant's top edge (not all at its center) in the same
    # left-to-right order as the lanes, so the paths never cross each other. ms1 skips this --
    # it sits directly above raw_quant with nothing in between, so a plain straight arrow already
    # does the right thing.
    manifold_y = rt_box["top"] + 0.35  # clear of ms1/none's bottom edge; above rt_box's top

    def converge(src, x_lane, x_land):
        ax = F.ax
        ax.plot([x_lane, x_lane], [src["bottom"], manifold_y], color=S.NEUTRAL, lw=0.9)
        ax.plot([x_lane, x_land], [manifold_y, manifold_y], color=S.NEUTRAL, lw=0.9)
        ax.annotate("", xy=(x_land, rt_box["top"]), xytext=(x_land, manifold_y),
                   arrowprops=dict(arrowstyle="-|>", lw=0.9, color=S.NEUTRAL, mutation_scale=8,
                                   shrinkA=0, shrinkB=0))

    # x_lane: <2.35 clears both ms2[2.35,5.75] and ms1[3.8,7.2] (and q1[6.25,10.25]); <3.8 clears
    # ms1/q1 alone; none has nothing below its own row so uses its own x as the lane.
    converge(reported, 1.9, 3.2)
    converge(ms2, 3.3, 4.2)
    # Plain vertical line, not F.arrow(ms1, raw_quant): that would land on raw_quant's own
    # center (x=5.0), making a straight but slightly diagonal line since ms1 is at x=5.5 --
    # landing at ms1's own x instead keeps it perfectly vertical.
    F.ax.annotate("", xy=(ms1["x"], rt_box["top"]), xytext=(ms1["x"], ms1["bottom"]),
                 arrowprops=dict(arrowstyle="-|>", lw=0.9, color=S.NEUTRAL, mutation_scale=8,
                                 shrinkA=0, shrinkB=0))
    converge(none, none["x"], 6.6)

    # Gray container "Signal localization": from below XIC extraction to below the "RT boundaries" box,
    # spanning the whole status/tier tree (and the leftmost convergence lane). The XIC extraction arrow
    # points at the container itself (vertical, at the chain's x), and the arrow into raw_quant starts at the "RT boundaries" box, crossing the container's bottom edge.
    tree = [quant_status, reported, q2, ms2, q1, ms1, none, rt_box]
    left = min(min(b["x"] - b["w"] / 2 for b in tree), 1.8)
    right = max(b["x"] + b["w"] / 2 for b in tree)
    cont_top = b3["bottom"] - 0.4  # long enough for a visible arrowhead
    cont = F.container((left + right) / 2, cont_top, rt_box["bottom"] - 0.2, "Signal localization (Evidence-guided RT refinement)",
                       w=right - left + 0.5, title_x=quant_status["x"])  # title over the status box, not the frame centre
    arrow = dict(arrowstyle="-|>", lw=0.9, color=S.NEUTRAL, mutation_scale=8, shrinkA=0, shrinkB=0)
    F.ax.annotate("", xy=(5.0, cont["top"]), xytext=(5.0, b3["bottom"]), arrowprops=arrow)
    F.ax.annotate("", xy=(5.0, raw_quant["top"]), xytext=(5.0, rt_box["bottom"]), arrowprops=arrow)
    return F


def _letters_left_of_ylabels(fig, axs):
    """Move each panel letter just left of its y label (S.panel_letter puts it over the axes' left edge).
    Call after tight_layout; panels without a y label keep their letter where it is."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in np.ravel(axs):
        ylab = ax.yaxis.label
        if not ax.axison or not ylab.get_text():
            continue
        title = ax._left_title
        x0 = ylab.get_window_extent(renderer).x0 - title.get_window_extent(renderer).width - 0.05 * fig.dpi
        title.set_x(ax.transAxes.inverted().transform((x0, 0))[0])  # not set_title(): that resets the bold weight


def _build_mapping_workflow():
    # xlim: content spans x=[0.8, 8.7] (widest boxes are "mapped"/"unresolved2" on the left and
    # "bad" on the right) -- 0.15 margin each side. ylim gives this workflow enough headroom (at
    # the shared WORKFLOW_ROW/BRANCH_ROW pitch -- see that constant's comment) to fit the "within
    # extrapolation limit?" branch added below "Mapping-supported precursor model" (mirroring
    # Supp_Mechanism2_E_extrapolation_validation's panel a) without clipping; it ends up taller
    # than the localization workflow since it chains through more boxes at the same pitch.
    F = S.Flowchart(xlim=(0.65, 8.85), ylim=(0.6, 14.4))
    F.ROW, F.BRANCH_ROW = WORKFLOW_ROW, WORKFLOW_BRANCH_ROW
    y = 13.8
    b1 = F.box(5.0, y, "JUMPlion raw\nprecursor quantity")
    b2 = F.box(5.0, y - F.ROW, "Paired DIA-NN and JUMPlion\nprecursor quantities")
    b3 = F.box(5.0, y - 2 * F.ROW, "Precursor-specific linear\nmapping (log$_2$ scale)")
    top = b3["y"] - F.ROW + F.BOX_H2 / 2
    # one-line criteria boxes (BOX_H1): shrink the pitch by the height saved vs. a two-line box (BOX_H2), so the
    # gap between the boxes stays what it was with ROW
    crit_row = F.ROW - (F.BOX_H2 - F.BOX_H1)
    g1 = F.box(5.0, top - 1.25, "Enough paired observations", h=F.BOX_H1)
    g2 = F.box(5.0, g1["y"] - crit_row, "Mapping slope: 0.5–1.5", h=F.BOX_H1)
    g3 = F.box(5.0, g2["y"] - crit_row, "Fitting MAE ≤0.5 log$_2$ units", h=F.BOX_H1)
    crit = F.container(5.0, top, g3["y"] - F.BOX_H1 / 2 - 0.35,
                       "Is precursor-specific mapping reliable?", w=F.WIDE + 0.8)
    yo = crit["bottom"] - F.BRANCH_ROW + F.BOX_H2 / 2 - 0.1
    ok = F.box(3.2, yo, "Apply mapping to candidate\nmissing quantities", w=F.MID, face=S.GATE_COLORS["pass"])
    bad = F.box(7.0, yo, "Mapping unsupported\n(remain missing)", w=F.MID, face=S.tier_color("prediction_only"))
    # Applicability branch (mirrors Supp_Mechanism2_E_extrapolation_validation's panel a): a
    # mapping-supported model still needs the recovered precursor-run's raw quantity to fall
    # within the fitted range (production guardrail = MAX_EXTRAPOLATION log2) before it's actually
    # used -- a second, separate "Unresolved" outcome from the reliability check's "fail" branch.
    extrap = F.box(3.2, yo - F.ROW, "Within extrapolation limit?\n(≤2 log$_2$ units)", w=F.MID)
    mapped = F.box(2.0, extrap["y"] - F.BRANCH_ROW, "JUMPlion-recovered\n precursor quantity", w=2.6, #h=F.BOX_H1,
                   face=S.GATE_COLORS["pass"])
    unresolved2 = F.box(4.8, extrap["y"] - F.BRANCH_ROW, "Remain missing", w=2.2, #h=F.BOX_H1,
                        face=S.tier_color("prediction_only"))
    F.arrow(b1, b2); F.arrow(b2, b3); F.arrow(b3, crit); F.arrow(g1, g2); F.arrow(g2, g3)
    F.arrow(crit, ok, "pass", dx=-0.3); F.arrow(crit, bad, "fail", dx=0.3)
    F.arrow(ok, extrap)
    F.arrow(extrap, mapped, "yes", dx=-0.3); F.arrow(extrap, unresolved2, "no", dx=0.3)
    return F


def _render_flowchart_rgba(F, dpi):
    """Render a Flowchart's figure to an in-memory RGBA array at `dpi` (no intermediate PNG file
    on disk) and close it -- bbox pinned to the figure's own size for the same reason
    style.FigureSaver.save(tight=False) does: a Flowchart's figsize is its physical size by
    construction (subplots_adjust(0,0,1,1), axis off), so there's no content to crop to anyway,
    but pinning it keeps this robust to that changing."""
    w, h = F.fig.get_size_inches()
    buf = io.BytesIO()
    F.fig.savefig(buf, format="png", dpi=dpi, bbox_inches=Bbox.from_bounds(0, 0, w, h), pad_inches=0)
    plt.close(F.fig)
    buf.seek(0)
    return plt.imread(buf)


def panel_combined_workflow(L, sv, dpi):
    """Both mechanism workflows -- localization-tier recovery and intensity mapping -- side by
    side, top-aligned, in one Supp figure, each under its own bold heading. Side by side (not
    stacked) avoids an even taller stacked figure. Both flowcharts share one row/branch pitch
    (WORKFLOW_ROW/BRANCH_ROW), but the mapping workflow ends up taller since it chains through
    more boxes at that same pitch (each workflow's own ylim is tuned to match, not shared --
    WORKFLOW_A_YLIM for localization, the ylim passed directly in _build_mapping_workflow for
    mapping); top-aligned (within the body band, below the headings) so that height difference
    doesn't misalign either chart's "input" row."""
    dpi = int(dpi)
    img1 = _render_flowchart_rgba(_build_localization_workflow(), dpi)
    img2 = _render_flowchart_rgba(_build_mapping_workflow(), dpi)
    w1, h1 = img1.shape[1] / dpi, img1.shape[0] / dpi
    w2, h2 = img2.shape[1] / dpi, img2.shape[0] / dpi
    gap = 0.3
    # title_h: reserved above the body for each column's one-line heading, at axes.titlesize
    # (font_size+1) bold.
    title_h = 0.26
    total_w = w1 + w2 + gap
    body_h = max(h1, h2)
    total_h = body_h + title_h
    fig = plt.figure(figsize=(total_w, total_h))
    ax1 = fig.add_axes([0, (body_h - h1) / total_h, w1 / total_w, h1 / total_h])
    ax1.imshow(img1); ax1.axis("off")
    # x=0.377: PANEL_A_XLIM isn't symmetric around the chain's own x=5.0 (there's extra room on
    # the right for the "none" box) -- a plain centered title (x=0.5, the default) would sit
    # visibly right of the actual box chain. 0.377 is where x=5.0 falls as a fraction of
    # PANEL_A_XLIM's current width, so the title lines up with the chain instead of the whole
    # (asymmetric) canvas -- recompute if PANEL_A_XLIM changes.
    ax1.set_title("a. Hierarchical ion evidence evaluation", fontweight="bold", x=0.384)
    ax2 = fig.add_axes([(w1 + gap) / total_w, (body_h - h2) / total_h, w2 / total_w, h2 / total_h])
    ax2.imshow(img2); ax2.axis("off")
    ax2.set_title("b. Intensity mapping and selective recovery", fontweight="bold")
    sv(fig, L.supp, "Supp_Mechanism_recovery_and_mapping_workflow", tight=False)


# ----------------------------------------------------------------------------
# Signal-recovery mechanism panels (former signal_recovery_mechanism.py; all Supp now --
# Panel A absorbed into panel_combined_workflow above)
# ----------------------------------------------------------------------------
def panel_tier_frequencies(evidence, L, sv, universe):
    counts = evidence[TIER_COL].value_counts().reindex(TIERS).fillna(0).astype(int)
    total = int(counts.sum())
    frac = counts / total if total else counts.astype(float)
    before_counts = pd.DataFrame({"tier": TIERS, "label": [S.tier_label(t) for t in TIERS],
                                  "n": counts.values, "fraction": frac.values})
    write_tsv(before_counts, L.table, "Supp_Mechanism1_B_evidence_tiers.tsv")
    exp, np_, nr_ = (universe.get(k) for k in ["expected_cartesian_n", "n_precursors", "n_runs"])
    if exp is not None and total != exp:
        log.warning("Mechanism1 B tier sum (%s) differs from n_precursor*n_run (%s).", f"{total:,}", f"{exp:,}")
    qc = {t: int(counts[t]) for t in TIERS}
    qc.update(sum_four_tiers=total, n_precursors=np_, n_runs=nr_, n_precursor_x_n_run=exp,
              sum_equals_n_precursor_x_n_run=None if exp is None else total == exp)
    write_tsv(pd.DataFrame([qc]), L.table, "Supp_Mechanism1_B_evidence_tier_QC.tsv")
    write_tsv(pd.DataFrame({"tier": TIERS, "n": counts.values, "fraction": frac.values}),
              L.table, "Table_evidence_tier_fractions.tsv")
    return before_counts


def _rt_delta(evidence):
    if any(c not in evidence.columns for c in ["RT.Query", "RT.Apex"]):
        return None
    d = evidence[evidence[TIER_COL].isin(["ms2_coelution", "ms1_peak"])].copy()
    d["RT.Query"] = pd.to_numeric(d["RT.Query"], errors="coerce")
    d["delta"] = pd.to_numeric(d["RT.Apex"], errors="coerce") - d["RT.Query"]
    d = d[np.isfinite(d["delta"]) & np.isfinite(d["RT.Query"])]
    return d if len(d) else None




def four_tier_violins(evidence, features, ylabels, titles, directory, table_dir, stem, sv, max_pts,
                      figsize=None, ylim=None, tight=True, top=None, tiers=None,
                      left=None, right=None, wspace=None, bottom=.28, colors=None,
                      legend_handles=None, short_labels=False):
    """`short_labels`: the two-level x labels of Fig_Mechanism_A (two_level_tier_axis) instead of the
    rotated full names.
    `tiers` overrides the module-global 4-tier TIERS (e.g. a 3-tier subset) for both the
    plotted groups and the x-axis -- despite the name, this isn't limited to exactly four.
    `ylim` is either one (lo, hi) tuple shared by every subplot, or a list with one entry (a
    tuple, or None to leave that subplot auto-scaled) per feature, for when features that don't
    share a natural y-range are combined into one figure. `left`/`right`/`wspace`/`bottom` override
    the auto-computed subplots_adjust margins -- for pinning this figure's axes boxes to line up
    with another, separately-saved panel in a composite (e.g. the final-recovered section's
    assemble_main_figure). `colors` overrides the default one-color-per-tier (S.tier_color(t) for
    t in tiers) list, e.g. to gray out specific tiers for one caller without changing S.tier_color
    globally. `legend_handles`, if given, draws one legend (shared across every subplot) on the
    last one."""
    tiers = tiers if tiers is not None else TIERS
    colors = colors if colors is not None else [S.tier_color(t) for t in tiers]
    present = [f in evidence.columns for f in features]
    feats = [ft for ft, p in zip(zip(features, ylabels, titles), present) if p]
    if not feats:
        return
    # Filter a per-feature ylim list the same way as feats, so they stay aligned even if a
    # feature was dropped above for not being in evidence.columns.
    ylims = ([yl for yl, p in zip(ylim, present) if p] if isinstance(ylim, list)
            else [ylim] * len(feats))
    fig, axes = plt.subplots(1, len(feats), figsize=figsize or (min(7.2, 3.6 * len(feats)), 3.6), squeeze=False)
    rows = []
    for i, (ax, (feature, ylabel, title)) in enumerate(zip(axes.ravel(), feats)):
        groups = _tier_groups(evidence, feature, tiers=tiers)
        S.violin_box(ax, groups, colors, max_kde_points=max_pts)
        _tier_axis(ax, tiers=tiers, short_labels=short_labels)
        if title:
            ax.set_title(title)
        ax.set_ylabel(ylabel)
        if ylims[i] is not None:
            ax.set_ylim(*ylims[i])
        for tier, v in zip(tiers, groups):
            rows.append(dict(feature=feature, tier=tier, n=len(v),
                             median=np.median(v) if len(v) else np.nan,
                             q25=np.quantile(v, .25) if len(v) else np.nan,
                             q75=np.quantile(v, .75) if len(v) else np.nan))
    if legend_handles:
        # Anchored just outside the last subplot's own right edge (not "upper right" inside it),
        # so it never sits on top of violin data -- callers that use this must leave enough right
        # margin (via `right`) for it; see COL2_B_RIGHT.
        axes.ravel()[-1].legend(handles=legend_handles, frameon=False, fontsize=ANNOT_FONTSIZE,
                                loc="lower right", bbox_to_anchor=(1.1, 1.0), borderaxespad=0)
    fig.subplots_adjust(bottom=bottom, wspace=wspace if wspace is not None else .25,
                        **({"top": top} if top is not None else {}),
                        **({"left": left} if left is not None else {}),
                        **({"right": right} if right is not None else {}))
    sv(fig, directory, stem, tight=tight)
    write_tsv(pd.DataFrame(rows), table_dir, f"{stem}_summary.tsv")


# ----------------------------------------------------------------------------
# Signal-recovery: supplementary panels (unchanged stems)
# ----------------------------------------------------------------------------
def _weighted_hist_quantile(x, count, q):
    order = np.argsort(x)
    x, count = np.asarray(x, float)[order], np.asarray(count, float)[order]
    cs = np.cumsum(count)
    return float(x[min(np.searchsorted(cs, q * cs[-1], side="left"), len(x) - 1)])


def _coarsen_bins(x, count, target_bins):
    """Merge adjacent pre-binned (x, count) pairs so at most ~target_bins bars remain."""
    order = np.argsort(x)
    x, count = np.asarray(x, float)[order], np.asarray(count, float)[order]
    factor = max(1, int(np.ceil(len(x) / target_bins)))
    if factor == 1:
        return x, count
    groups = np.arange(len(x)) // factor
    uniq = np.unique(groups)
    x_c = np.array([x[groups == g].mean() for g in uniq])
    count_c = np.array([count[groups == g].sum() for g in uniq])
    return x_c, count_c


def _ensure_rt_cross_validation(cfg, resolve):
    """Return (error_histogram_path, representative_run_path), running JUMPlion's RT cross-validation
    first when those files are missing (production runs turn the validation off).

    The cross-validation is JUMPlion's own code (jumplion.coordinate_validation.run_rt_prediction with
    run_validation=True), run with the production settings on the saved pre-prediction RT matrix and the
    library RTs. `mechanism.rt_cv_regenerate` = {"rt_matrix", "library", "min_run_fraction", "output_dir"};
    results go to output_dir (default: <rt_error_histogram's folder>/regenerated), not over production files."""
    hist_p, rep_p = resolve(cfg.get("rt_error_histogram")), resolve(cfg.get("rt_representative_run"))
    if hist_p is None or rep_p is None or (Path(hist_p).exists() and Path(rep_p).exists()):
        return hist_p, rep_p
    rg = cfg.get("rt_cv_regenerate") or {}
    rt_matrix, library = resolve(rg.get("rt_matrix")), resolve(rg.get("library"))
    if rt_matrix is None or library is None:
        raise FileNotFoundError(
            f"{hist_p} is missing (RT cross-validation was off in the JUMPlion run). Set mechanism.rt_cv_regenerate "
            '= {"rt_matrix": ".../rt_prediction/rt_mat_before_prediction.tsv", "library": "<spectral library parquet>"} '
            "to rerun it here.")
    from jumplion.calibration import calc_spline_vectorized, refine_spline
    from jumplion.coordinate_validation import ModelConfig, run_rt_prediction

    out_dir = Path(resolve(rg.get("output_dir")) or Path(hist_p).parent / "regenerated")
    log.info("RT cross-validation files not found; running it on %s (output: %s)", rt_matrix, out_dir)
    rt_mat = pd.read_csv(rt_matrix, sep="\t", index_col=0)
    lib = pd.read_parquet(library, columns=["Precursor.Id", "RT"]).drop_duplicates("Precursor.Id")
    mcfg = ModelConfig(coordinate_label="RT", max_segments=30, min_run_fraction=float(rg.get("min_run_fraction", 0.5)),
                       adjustment_mode="clip")  # as JUMPlion's production RT prediction
    run_rt_prediction(rt_mat=rt_mat, lib_df_uniq_prec=lib, refine_spline_fn=refine_spline,
                      calc_spline_vectorized_fn=calc_spline_vectorized, out_dir=out_dir, config=mcfg,
                      run_validation=True, save_plots=False)
    return out_dir / Path(hist_p).name, out_dir / Path(rep_p).name


def panel_pre_extraction_rt_validation(hist, rep, half_window, L, sv, cycle_time=None):
    if hist is None or rep is None or half_window is None:
        return
    obs = next((c for c in ["observed_coordinate", "observed_rt", "Observed.RT"] if c in rep.columns), None)
    pred = next((c for c in ["predicted_coordinate_final", "predicted_rt_final", "predicted_rt_oof"] if c in rep.columns), None)
    ccol = next((c for c in ["bin_center", "error_bin_center"] if c in hist.columns), None)
    ncol = next((c for c in ["count", "n"] if c in hist.columns), None)
    if None in (obs, pred, ccol, ncol):
        log.warning("Supp_Mechanism1_A_01 skipped: RT histogram/representative-run columns not recognized.")
        return
    d = rep[[obs, pred]].apply(pd.to_numeric, errors="coerce").dropna()
    xh = pd.to_numeric(hist[ccol], errors="coerce")
    cnt = pd.to_numeric(hist[ncol], errors="coerce").fillna(0)
    ok = np.isfinite(xh) & (cnt >= 0)
    xh, cnt = xh[ok].to_numpy(), cnt[ok].to_numpy()
    if d.empty or not len(xh) or cnt.sum() <= 0:
        return
    fig, axes = plt.subplots(1, 2, figsize=(5, 2.4))
    ax = axes[0]
    lo, hi = min(d[obs].min(), d[pred].min()), max(d[obs].max(), d[pred].max())
    ax.scatter(d[obs], d[pred], s=6, alpha=.35, color=S.NEUTRAL, edgecolors="none", rasterized=True)
    ax.plot([lo, hi], [lo, hi], ls="--", lw=.8, color=S.REF_LINE)
    ax.set_xlabel("Observed RT (min)"); ax.set_ylabel("OOF-predicted RT (min)")
    ax.set_title("An example run", fontsize=7); S.panel_letter(ax, "a")
    ax.text(.1, .97, f"n={len(d):,}", transform=ax.transAxes, ha="left", va="top", fontsize=6)
    ax = axes[1]
    xh_plot, cnt_plot = _coarsen_bins(xh, cnt, target_bins=20)
    widths = np.diff(np.sort(np.unique(xh_plot)))
    width = float(np.median(widths)) if len(widths) else 1.0
    ax.bar(xh_plot, cnt_plot / cnt_plot.sum(), width=width * .95, color=S.NEUTRAL, edgecolor="none")
    S.ref_line(ax, 0, "v")
    for s in (-1, 1):
        S.ref_line(ax, s * float(half_window), "v", ls=":", color=S.METHOD_COLORS["JUMPlion"])
    med = _weighted_hist_quantile(xh, cnt, .50)
    p95 = _weighted_hist_quantile(np.abs(xh), cnt, .95)
    within = float(cnt[np.abs(xh) <= float(half_window)].sum() / cnt.sum())
    ax.set_xlabel("OOF-predicted RT − observed RT (min)"); ax.set_ylabel("Fraction")
    ax.set_title("All 30 runs", fontsize=7); S.panel_letter(ax, "b")
    n_total = int(round(cnt.sum()))
    ax.text(.1, .97, f"n={n_total:,}", transform=ax.transAxes, ha="left", va="top", fontsize=6)
    lines = [f"median ΔRT={med:.4f} min", f"p95 |ΔRT|={p95:.4f} min"]
    lines.append(f"RT window ±{float(half_window):g} min")
    #lines.append(f"{within:.1%} within window")
    ax.text(.95, .97, "\n".join(lines), transform=ax.transAxes, ha="right", va="top", fontsize=5)
    fig.tight_layout()
    _letters_left_of_ylabels(fig, axes)
    sv(fig, L.supp, "Supp_Mechanism1_A_pre_extraction_RT_validation")
    write_tsv(pd.DataFrame([dict(n_representative=len(d), median_signed_error=med, p95_abs_error=p95,
                                 half_window=float(half_window), fraction_within_window=within)]),
              L.table, "Supp_Mechanism1_A_pre_extraction_RT_validation_summary.tsv")


def panel_rt_refinement_by_rt(evidence, L, sv, n_bins=20):
    """Left column: the tier's overall refinement-error distribution as a histogram -- former
    standalone Supp_Mechanism1_C, merged in here instead of its own panel so the per-tier overall
    distribution and gradient trend read together. Right column: refinement error (median,
    p10-p90 band) binned across the RT gradient -- former Supp_Mechanism1_A_02."""
    d = _rt_delta(evidence)
    if d is None:
        return
    rows = []
    for tier, g in d.groupby(TIER_COL):
        try:
            g = g.assign(bin=pd.qcut(g["RT.Query"], q=n_bins, duplicates="drop"))
        except ValueError:
            continue
        for _, h in g.groupby("bin", observed=True):
            if len(h) < 20:
                continue
            rows.append(dict(tier=tier, rt=h["RT.Query"].median(), n=len(h), median=h["delta"].median(),
                             p10=h["delta"].quantile(.10), p90=h["delta"].quantile(.90),
                             median_abs=h["delta"].abs().median(), p95_abs=h["delta"].abs().quantile(.95)))
    z = pd.DataFrame(rows)
    if z.empty:
        return
    maxabs = max(float(np.nanquantile(np.abs(d["delta"]), .995)), 1e-6)
    bins = np.linspace(-maxabs, maxabs, 61)
    hist_rows = []
    fig, axes = plt.subplots(2, 2, figsize=(7, 4.8), sharex="col",
                             gridspec_kw={"width_ratios": [1, 1.5]})
    for (ax_hist, ax_rt), tier in zip(axes, ["ms2_coelution", "ms1_peak"]):
        clr = S.tier_color(tier)
        q = z[z["tier"].eq(tier)].sort_values("rt")
        if not q.empty:
            ax_rt.fill_between(q["rt"].to_numpy(), q["p10"].to_numpy(), q["p90"].to_numpy(), color=clr, alpha=.25, lw=0,
                               label="10th–90th percentile")
            ax_rt.plot(q["rt"], q["median"], marker="o", ms=3, color=clr, label="Median (per RT bin)")
            ax_rt.legend(frameon=False, fontsize=6, loc="upper right", handlelength=1.4)
        S.ref_line(ax_rt, 0)
        ax_rt.set_ylabel("Refined RT − predicted RT (min)")
        ax_rt.set_title(S.tier_label(tier), loc="left")

        zz = d.loc[d[TIER_COL].eq(tier), "delta"].to_numpy()
        if len(zz):
            ax_hist.hist(zz, bins=bins, density=True, color=clr, edgecolor="none")
            S.ref_line(ax_hist, 0, "v")
            med, absz = float(np.median(zz)), np.abs(zz)
            p95 = float(np.quantile(absz, .95))
            ax_hist.text(.98, .94, f"n={len(zz):,}\nmedian ΔRT={med:.4f} min\np95 |ΔRT|={p95:.4f} min",
                        transform=ax_hist.transAxes, ha="right", va="top", fontsize=7)
            hist_rows.append(dict(tier=tier, n=len(zz), median_delta=med, median_abs_delta=float(np.median(absz)),
                                  p90_abs_delta=float(np.quantile(absz, .90)), p95_abs_delta=p95))
        else:
            ax_hist.axis("off")
        ax_hist.set_ylabel("Density")
        ax_hist.set_title(S.tier_label(tier), loc="left")  # same title as the gradient plot beside it
        # Both panels show the same quantity (refinement delta) on different axes -- histogram on
        # x, RT-gradient plot on y -- so pin them to the same range for a direct visual comparison.
        ax_hist.set_xlim(-maxabs, maxabs)
        ax_rt.set_ylim(-maxabs, maxabs)
    axes[-1, 0].set_xlabel("Refined RT − predicted RT (min)")
    axes[-1, 1].set_xlabel("Predicted RT (min)")
    fig.tight_layout()
    # panel letters a-d (row by row), just left of each y label and level with the axes' top; drawn as figure
    # text because the right column's left-aligned title (the tier name) already occupies the title slot
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, letter in zip(axes.ravel(), "abcd"):
        x0 = ax.yaxis.label.get_window_extent(renderer).x0 - 0.05 * fig.dpi
        fig.text(x0 / fig.bbox.width, ax.get_position().y1 + .01, letter, ha="right", va="bottom",
                 fontsize=plt.rcParams["axes.titlesize"], fontweight="bold")
    sv(fig, L.supp, "Supp_Mechanism1_B_RT_refinement_across_gradient")
    write_tsv(z, L.table, "Supp_Mechanism1_B_RT_refinement_across_gradient_summary.tsv")
    write_tsv(pd.DataFrame(hist_rows), L.table,
             "Supp_Mechanism1_B_RT_refinement_across_gradient_histogram_summary.tsv")


def _ensure_abs_error(df, label):
    out = df.copy()
    pred = next((c for c in [f"predicted_{label.lower()}_final", "predicted_coordinate_final"] if c in out.columns), None)
    if "final_absolute_error" not in out.columns:
        if pred is None or "observed_coordinate" not in out.columns:
            raise KeyError(f"{label} CV table needs final_absolute_error or observed_coordinate + prediction.")
        out["final_absolute_error"] = (out[pred].astype(float) - out["observed_coordinate"].astype(float)).abs()
    if "fit_status" in out.columns:
        out = out[out["fit_status"] == "ok"]
    return out[np.isfinite(pd.to_numeric(out["final_absolute_error"], errors="coerce"))].copy(), pred


def _ecdf_panel(ax, errors, boundary, p95, coverage, boundary_text, fmt):
    x, y = S.ecdf_xy(errors)
    ax.plot(x, y, color=S.NEUTRAL)
    S.ref_line(ax, boundary, "v")
    S.ref_line(ax, p95, "v", ls=":")
    ax.set_ylabel("Cumulative fraction"); ax.set_ylim(0, 1.01); ax.set_xlim(left=0)
    ax.text(p95, .95, f"p95 = {p95:{fmt}}", ha="left", va="top", fontsize=7)
    ax.text(boundary, .06, boundary_text, ha="right", va="bottom", fontsize=7)
    ax.text(.98, .20, f"{coverage:.1%} within window", transform=ax.transAxes, ha="right", fontsize=7)


def panel_rt_localization_legacy(rt_cv, half_points, cycle_time, L, sv):
    cv, _ = _ensure_abs_error(rt_cv, "RT")
    if "final_absolute_error_points" in cv.columns:
        err = pd.to_numeric(cv["final_absolute_error_points"], errors="coerce")
    else:
        if not cycle_time or cycle_time <= 0:
            raise ValueError("cycle_time required unless RT CV has final_absolute_error_points.")
        err = pd.to_numeric(cv["final_absolute_error"], errors="coerce") / float(cycle_time)
    vals = err.replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    p95, cov = float(np.nanpercentile(vals, 95)), float(np.mean(vals <= half_points))
    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    _ecdf_panel(ax, vals, half_points, p95, cov, f"Extraction boundary\n±{half_points:g} points", ".1f")
    ax.set_xlabel("Absolute RT prediction error (acquisition points)")
    fig.tight_layout()
    sv(fig, L.supp, "Supp_Mechanism1_01_RT_localization")
    write_tsv(pd.DataFrame([dict(n=len(vals), rt_half_window_points=half_points, p95_absolute_error_points=p95,
                                 fraction_within_extraction_window=cov)]), L.table, "Supp_Mechanism1_01_RT_localization_summary.tsv")


def panel_im(im_cv, im_window, im_window_table, L, sv):
    if im_cv is None:
        return
    cv, pred = _ensure_abs_error(im_cv, "IM")
    if im_window_table is not None:
        dd = cv.merge(im_window_table[["run_name", "im_half_window"]], on="run_name", how="inner")
        errors, windows = pd.to_numeric(dd["final_absolute_error"], errors="coerce"), pd.to_numeric(dd["im_half_window"], errors="coerce")
        boundary, txt = float(np.nanmedian(windows)), "Median calibrated boundary"
    elif im_window is not None:
        errors = pd.to_numeric(cv["final_absolute_error"], errors="coerce")
        windows = pd.Series(float(im_window), index=cv.index)
        boundary, txt = float(im_window), f"Extraction boundary ±{float(im_window):g}"
    else:
        errors = None
    if errors is not None:
        ok = np.isfinite(errors) & np.isfinite(windows)
        e, w = errors[ok].to_numpy(), windows[ok].to_numpy()
        p95, cov = float(np.nanpercentile(e, 95)), float(np.mean(e <= w))
        fig, ax = plt.subplots(figsize=(3.6, 2.9))
        _ecdf_panel(ax, e, boundary, p95, cov, txt, ".4f")
        ax.set_xlabel("Absolute IM prediction error")
        fig.tight_layout()
        sv(fig, L.supp, "Supp_Mechanism1_02_IM_localization")
    # Observed vs out-of-fold predicted IM scatter
    if pred is not None and "observed_coordinate" in cv.columns:
        d = cv[["observed_coordinate", pred]].dropna()
        if len(d) > 30000:
            d = d.sample(30000, random_state=1)
        fig, ax = plt.subplots(figsize=(3.1, 3.0))
        ax.scatter(d["observed_coordinate"], d[pred], s=3, alpha=.2, color=S.NEUTRAL, edgecolors="none", rasterized=True)
        lo, hi = min(d.min()), max(d.max())
        ax.plot([lo, hi], [lo, hi], lw=.8, ls="--", color=S.REF_LINE)
        ax.set_xlabel("Observed IM"); ax.set_ylabel("Out-of-fold predicted IM")
        fig.tight_layout()
        sv(fig, L.supp, "Supp_Mechanism1_05_IM_out_of_fold_scatter")


# ----------------------------------------------------------------------------
# Signal-recovery: XIC examples (unchanged stems)
# ----------------------------------------------------------------------------
def parse_array_cell(x):
    """Robust conversion of nested parquet object arrays to numeric arrays."""
    if x is None:
        return np.asarray([], dtype=float)
    if isinstance(x, np.ndarray):
        if x.dtype != object:
            return np.asarray(x, dtype=float)
        try:
            rows = [np.asarray(r, dtype=float).ravel() for r in x]
            if rows and len({len(r) for r in rows}) == 1:
                return np.vstack(rows)
            return np.asarray(x.tolist(), dtype=float)
        except (TypeError, ValueError):
            return np.asarray([], dtype=float)
    if isinstance(x, (list, tuple)):
        try:
            return np.asarray(x, dtype=float)
        except (TypeError, ValueError):
            try:
                return np.vstack([np.asarray(r, dtype=float).ravel() for r in x])
            except (TypeError, ValueError):
                return np.asarray([], dtype=float)
    if isinstance(x, str):
        for parser in (ast.literal_eval, json.loads):
            try:
                return parse_array_cell(parser(x))
            except Exception:
                pass
        return np.asarray([], dtype=float)
    try:
        return np.asarray(x, dtype=float)
    except (TypeError, ValueError):
        return np.asarray([], dtype=float)


def _valid_xic(r):
    a, b = parse_array_cell(r.get("ms1", [])), parse_array_cell(r.get("ms2", []))
    return a.ndim == 2 and b.ndim == 2 and a.shape[0] >= 5 and b.shape[0] >= 5 and a.shape[1] > 0 and b.shape[1] > 0


def _xic_tier_col(d):
    for c in ("evidence_tier", TIER_COL):
        if c in d.columns:
            return c
    raise KeyError("XIC data lack evidence_tier/Evidence.Tier.")


def _xic_files(xic_dir):
    fs = sorted(Path(xic_dir).glob("*.xic.parquet"))
    if not fs:
        raise FileNotFoundError(f"No *.xic.parquet files in {xic_dir}")
    return {f.name.replace(".xic.parquet", ""): f for f in fs}


def _manual_example(files, spec, expected):
    if "|" not in spec:
        raise ValueError('Manual XIC example must be "Precursor.Id|run_name".')
    precursor, run = [z.strip() for z in spec.rsplit("|", 1)]
    path = files.get(run) or next((v for k, v in files.items() if k.lower() == run.lower()), None)
    if path is None:
        raise KeyError(f"Run not found: {run}")
    try:
        d = pd.read_parquet(path, filters=[("Precursor.Id", "==", precursor)])
    except Exception:
        d = pd.read_parquet(path)
        d = d[d["Precursor.Id"].astype(str) == precursor]
    if d.empty:
        raise KeyError(f"{precursor} not found in {run}")
    d = d.copy(); d["run_name"] = run
    actual = str(d.iloc[0][_xic_tier_col(d)])
    if actual != expected:
        log.warning("Requested %s example but actual tier=%s; plotting anyway.", expected, actual)
    d = d[d.apply(_valid_xic, axis=1)]
    if d.empty:
        raise ValueError(f"No usable XIC for {spec}")
    return d.iloc[0]


def _auto_examples(files, tiers, mode="median", max_per_run=250):
    """Scan each run file ONCE and pick one representative example per tier.

    `mode` is either one selection mode applied to every tier, or a {tier: mode} dict for
    tier-specific selection (e.g. {"reported": "high"}) without a second file scan. Modes:
    "median" (closest to the tier's median signal), "strongest" (max signal), "high" (closest to
    the 85th percentile -- relatively high without picking a potential outlier)."""
    cands = {t: [] for t in tiers}
    for run, path in files.items():
        d = pd.read_parquet(path)
        tc = _xic_tier_col(d)
        for tier in tiers:
            z = d[d[tc].astype(str) == tier]
            if len(z) > max_per_run:
                z = z.sample(max_per_run, random_state=1)
            for _, r in z.iterrows():
                if not _valid_xic(r):
                    continue
                s1, s2 = float(np.nansum(parse_array_cell(r["ms1"]))), float(np.nansum(parse_array_cell(r["ms2"])))
                sig = np.log2((s2 if tier == "ms2_coelution" else s1 if tier == "ms1_peak" else s1 + s2) + 1)
                if np.isfinite(sig):
                    rr = r.copy(); rr["run_name"] = run
                    cands[tier].append((sig, rr))
    modes = mode if isinstance(mode, dict) else {t: mode for t in tiers}
    out = {}
    for tier, c in cands.items():
        m = modes.get(tier, "median")
        if not c:
            out[tier] = None
        elif m == "strongest":
            out[tier] = max(c, key=lambda z: z[0])[1]
        elif m == "high":
            target = np.quantile([z[0] for z in c], 0.85)
            out[tier] = min(c, key=lambda z: abs(z[0] - target))[1]
        else:
            med = np.median([z[0] for z in c])
            out[tier] = min(c, key=lambda z: abs(z[0] - med))[1]
    return out


def plot_xic_example(row, tier, L, sv, stem):
    if row is None or not _valid_xic(row):
        return
    ms1, ms2 = parse_array_cell(row["ms1"]), parse_array_cell(row["ms2"])
    r1, r2 = parse_array_cell(row.get("ms1_rt", [])).ravel(), parse_array_cell(row.get("ms2_rt", [])).ravel()
    if len(r1) != len(ms1): r1 = np.arange(len(ms1))
    if len(r2) != len(ms2): r2 = np.arange(len(ms2))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(4.3, 4.0), sharex=True, gridspec_kw={"hspace": .18})
    for jj in range(ms1.shape[1]):
        a1.plot(r1, ms1[:, jj], lw=1.4 if jj == 0 else .9, alpha=.95 if jj == 0 else .7,
               color=MS1_ZOOM_COLORS.get(f"M+{jj}"))
    for pos, jj in enumerate(np.argsort(np.nansum(ms2, axis=0))[::-1][:min(6, ms2.shape[1])]):
        a2.plot(r2, ms2[:, jj], lw=.9, alpha=.8, color=MS2_ZOOM_COLORS[pos % len(MS2_ZOOM_COLORS)])
    num = lambda k: pd.to_numeric(pd.Series([row.get(k, np.nan)]), errors="coerce").iloc[0]
    initial, apex, fwhm = num("initial_apex"), num("apex"), num("fwhm")
    idx_ok = lambda v: np.isfinite(v) and 0 <= int(round(v)) < len(r2)
    prt = r2[int(round(initial))] if idx_ok(initial) else None
    art = r2[int(round(apex))] if idx_ok(apex) else None
    for ax in (a1, a2):
        if prt is not None:
            S.ref_line(ax, prt, "v")
        if art is not None and tier not in NO_DETECTED_PEAK_TIERS:
            ax.axvline(art, lw=1, color="black")
    if art is not None and np.isfinite(fwhm) and fwhm > 0 and tier not in NO_DETECTED_PEAK_TIERS:
        ai, h = int(round(apex)), max(1, int(round(fwhm)))  # shaded span = apex +/- fwhm
        for ax in (a1, a2):
            ax.axvspan(r2[max(0, ai - h)], r2[min(len(r2) - 1, ai + h)], color=S.tier_color(tier), alpha=.18, lw=0)
    a1.set_ylabel("MS1 intensity"); a2.set_ylabel("MS2 intensity"); a2.set_xlabel("Retention time")
    for ax in (a1, a2):
        ax.set_ylim(bottom=0)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        ax.yaxis.get_offset_text().set_fontsize(7)
    a1.set_title(S.tier_label(tier), loc="left", color="black")
    # Wrapped, right-aligned, figure-level caption (not an in-axes a1.text(ha="right", ...)): a
    # long run_name can overflow past the axes' left edge at that anchor, and since savefig uses
    # bbox_inches="tight" (style.apply_style), that overflow silently re-crops/distorts the whole
    # saved figure. Wrapping each row to a fixed width keeps every line short enough to never
    # overflow the canvas that way. ma="right" right-aligns each line to the anchor individually
    # (ha="right" alone only right-aligns the text block as a whole, not each line within it).
    run_wrapped = textwrap.fill(str(row.get("run_name", "")), width=100)
    caption = f'{row.get("Precursor.Id", "")}\n{run_wrapped}'
    n_lines = caption.count("\n") + 1
    fig.subplots_adjust(left=.16, right=.97, top=.92, bottom=.12 + .03 * n_lines)
    fig.text(.97, .01, caption, ha="right", va="bottom", ma="right", fontsize=5.5, linespacing=1.3)
    sv(fig, L.supp, stem)


# Plotly qualitative palettes, hardcoded (plotly isn't a project dependency) -- used only for the
# zoom XIC plots' channel coloring: px.colors.qualitative.Set2 (MS1 isotopes M+0.. ) and
# px.colors.qualitative.Dark24_r (MS2 fragments, by descending-signal rank).
_PX_SET2 = ["#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854", "#FFD92F", "#E5C494", "#B3B3B3"]
_PX_DARK24 = ["#2E91E5", "#E15F99", "#1CA71C", "#FB0D0D", "#DA16FF", "#222A2A", "#B68100", "#750D86",
              "#EB663B", "#511CFB", "#00A08B", "#FB00D1", "#FC0080", "#B2828D", "#6C7C32", "#778AAE",
              "#862A16", "#A777F1", "#620042", "#1616A7", "#DA60CA", "#6C4516", "#0D2A63", "#AF0038"]
MS1_ZOOM_COLORS = dict(zip([f"M+{i}" for i in range(9)], _PX_SET2[:9]))
MS2_ZOOM_COLORS = list(reversed(_PX_DARK24))


def _zoom_window(r, rt_value, fwhm, fwhm_mult=2):
    """RT range in `r` centered on `rt_value`, spanning +/- fwhm_mult*fwhm points."""
    if rt_value is None or not len(r) or not np.isfinite(fwhm) or fwhm <= 0:
        return None
    idx = int(np.argmin(np.abs(r - rt_value)))
    half_pts = max(1, int(round(fwhm_mult * fwhm)))
    lo, hi = max(0, idx - half_pts), min(len(r) - 1, idx + half_pts)
    return r[lo], r[hi]


def plot_xic_example_zoom(row, tier, channel, L, sv, stem, fallback_fwhm=3):
    """Small, un-annotated crop on just the central (apex-centered) part of one XIC channel --
    no RT reference line(s), no FWHM shading. `channel` is "ms1" or "ms2" -- one figure, one
    channel; call it twice (separate stems) for tiers that need both, e.g. prediction_only/
    reported, where neither channel is "the" recovery channel. Tiers in NO_DETECTED_PEAK_TIERS
    have no meaningful detected-peak fwhm, so `fallback_fwhm` stands in for it there."""
    if row is None or not _valid_xic(row):
        return
    ms1, ms2 = parse_array_cell(row["ms1"]), parse_array_cell(row["ms2"])
    r1, r2 = parse_array_cell(row.get("ms1_rt", [])).ravel(), parse_array_cell(row.get("ms2_rt", [])).ravel()
    if len(r1) != len(ms1): r1 = np.arange(len(ms1))
    if len(r2) != len(ms2): r2 = np.arange(len(ms2))
    num = lambda k: pd.to_numeric(pd.Series([row.get(k, np.nan)]), errors="coerce").iloc[0]
    apex, fwhm = num("apex"), num("fwhm")
    fwhm = fallback_fwhm if tier in NO_DETECTED_PEAK_TIERS else fwhm
    apex_rt = r2[int(round(apex))] if np.isfinite(apex) and 0 <= int(round(apex)) < len(r2) else None

    arr, r = (ms1, r1) if channel == "ms1" else (ms2, r2)
    fig, ax = plt.subplots(figsize=(0.75, 0.75))
    if channel == "ms1":
        cols = list(range(arr.shape[1]))
        for jj in cols:
            ax.plot(r, arr[:, jj], lw=1.4 if jj == 0 else .9, alpha=.95 if jj == 0 else .7,
                    color=MS1_ZOOM_COLORS.get(f"M+{jj}"))
    else:
        cols = list(np.argsort(np.nansum(arr, axis=0))[::-1][:min(6, arr.shape[1])])
        for pos, jj in enumerate(cols):
            ax.plot(r, arr[:, jj], lw=.9, alpha=.8, color=MS2_ZOOM_COLORS[pos % len(MS2_ZOOM_COLORS)])
    win = _zoom_window(r, apex_rt, fwhm)
    ymax = np.nan
    if win is not None:
        ax.set_xlim(*win)
        mask = (r >= win[0]) & (r <= win[1])
        if cols and mask.any():
            ymax = np.nanmax(arr[np.ix_(mask, cols)])
    ax.set_ylim(0, ymax * 1.05 if np.isfinite(ymax) and ymax > 0 else None)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    fig.subplots_adjust(left=.03, right=.98, top=.98, bottom=.02)
    sv(fig, L.supp / "xic_zoom", stem)


def make_xic_panels(xic_dir, manual, mode, L, sv):
    files = _xic_files(xic_dir)
    spec = [("ms2_coelution", "Supp_Mechanism1_XIC_A_MS2_coelution"), ("ms1_peak", "Supp_Mechanism1_XIC_B_MS1_peak"),
            ("prediction_only", "Supp_Mechanism1_XIC_C_no_qualifying_peak"),
            ("reported", "Supp_Mechanism1_XIC_D_DIANN_reported")]
    zoom_channels = {"ms2_coelution": ["ms2"], "ms1_peak": ["ms1"], "prediction_only": ["ms1", "ms2"],
                     "reported": ["ms1", "ms2"]}
    # "reported" always picks the single strongest-signal example, regardless of the configured
    # xic_auto_selection mode, which still governs the other three tiers.
    auto_tiers = [t for t, _ in spec if not manual.get(t)]
    auto_modes = {t: ("strongest" if t == "reported" else mode) for t in auto_tiers}
    auto = _auto_examples(files, auto_tiers, auto_modes) if auto_tiers else {}
    used = []
    for tier, stem in spec:
        row = _manual_example(files, manual[tier], tier) if manual.get(tier) else auto.get(tier)
        plot_xic_example(row, tier, L, sv, stem)
        channels = zoom_channels[tier]
        for ch in channels:
            suffix = f"_zoom_{ch}" if len(channels) > 1 else "_zoom"
            plot_xic_example_zoom(row, tier, ch, L, sv, f"{stem}{suffix}")
        if row is not None:
            sel_mode = "manual" if manual.get(tier) else f"automatic_{auto_modes.get(tier, mode)}"
            used.append(dict(requested_tier=tier, selection_mode=sel_mode,
                             **{"Precursor.Id": row.get("Precursor.Id", ""), "run_name": row.get("run_name", "")}))
    if used:
        write_tsv(pd.DataFrame(used), L.table, "xic_examples_used.tsv")


# ----------------------------------------------------------------------------
# Intensity mapping (former intensity_mapping_mechanism.py; all Supp now --
# Panel A absorbed into panel_combined_workflow above)
# ----------------------------------------------------------------------------
def pooled_fit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.var(x) == 0:
        return np.nan, np.nan
    slope = float(np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2))
    return slope, float(y.mean() - slope * x.mean())


def err_metrics(e):
    e = np.asarray(e, float)
    e = e[np.isfinite(e)]
    if not len(e):
        return {"N": 0, "MAE": np.nan, "RMSE": np.nan, "Median.Signed.Error": np.nan}
    return {"N": len(e), "MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e ** 2))),
            "Median.Signed.Error": float(np.median(e))}


def coverage_bins(max_n, min_overlap):
    max_n = max(int(min_overlap), int(max_n))
    edges = sorted({int(v) for v in [0, min_overlap - 1, 10, 15, 20, 25, max_n] if 0 <= v <= max_n})
    if edges[-1] < max_n:
        edges.append(max_n)
    labels = [f"<{min_overlap}" if (a == 0 and b == min_overlap - 1) else f"{a + 1}-{b}"
              for a, b in zip(edges[:-1], edges[1:])]
    return edges, labels


def choose_examples(metrics, paired):
    """Choose one clearly supported and one clearly rejected mapping example."""
    available = metrics[metrics.index.isin(paired[ID].unique())].copy()
    good = available[available["Model.Good"]].copy()
    bad = available[(~available["Model.Good"]) & available["Slope"].notna() & available["Intercept"].notna()].copy()
    rows = []
    if len(good):
        g = good.sort_values(["MAE", "N.Overlap"], ascending=[True, False]).iloc[[0]].copy()
        g["Example.Type"] = "Accepted mapping"
        rows.append(g)
    if len(bad):
        # Prefer a clearly interpretable slope failure near 0.25, then ~2.0,
        # rather than an extreme pathological fit.
        low = bad[bad["Model.Fail.Reason"].eq("SLOPE_TOO_LOW")].copy()
        high = bad[bad["Model.Fail.Reason"].eq("SLOPE_TOO_HIGH")].copy()
        if len(low):
            low["__d"] = (low["Slope"] - 0.25).abs()
            b = low.sort_values(["__d", "N.Overlap"], ascending=[True, False]).iloc[[0]].drop(columns="__d").copy()
        elif len(high):
            high["__d"] = (high["Slope"] - 2.0).abs()
            b = high.sort_values(["__d", "N.Overlap"], ascending=[True, False]).iloc[[0]].drop(columns="__d").copy()
        else:
            b = bad.sort_values(["MAE", "N.Overlap"], ascending=[False, False]).iloc[[0]].copy()
        b["Example.Type"] = "Rejected mapping"
        rows.append(b)
    return pd.concat(rows) if rows else available.head(0)


def choose_rejected_examples(metrics, paired):
    available = metrics[metrics.index.isin(paired[ID].unique())].copy()
    out = []
    for label, reason, target in [("Insufficient overlap", "INSUFFICIENT_OVERLAP", None),
                                  ("Slope too low", "SLOPE_TOO_LOW", 0.25),
                                  ("Slope too high", "SLOPE_TOO_HIGH", 2.0),
                                  ("High MAE", "HIGH_MAE", None)]:
        z = available[available["Model.Fail.Reason"].eq(reason)].copy()
        if target is not None:
            z = z[z["Slope"].notna()]
        if z.empty:
            continue
        if label == "Insufficient overlap":
            row = z.sort_values("N.Overlap", ascending=False).iloc[[0]].copy()
        elif target is not None:  # representative, not extreme, slope failure
            z["__d"] = (z["Slope"] - target).abs()
            row = z.sort_values(["__d", "N.Overlap"], ascending=[True, False]).iloc[[0]].drop(columns="__d").copy()
        else:
            row = z.sort_values(["MAE", "N.Overlap"], ascending=[False, False]).iloc[[0]].copy()
        row["Rejected.Gate"] = label
        out.append(row)
    return pd.concat(out) if out else available.head(0)


def _species_text(row):
    v = row.get(SPECIES, np.nan)
    return S.species_label(v) if pd.notna(v) else "Unannotated"


def mapping_scatter(ax, p, pid, row, header, passed):
    z = p[p[ID].eq(pid)]
    ax.scatter(z["JUMPlion.Raw.log2"], z["DIANN.log2"], s=13, color=S.NEUTRAL, alpha=.8, edgecolors="none")
    if pd.notna(row["Slope"]) and pd.notna(row["Intercept"]) and len(z):
        xx = np.linspace(z["JUMPlion.Raw.log2"].min(), z["JUMPlion.Raw.log2"].max(), 100)
        ax.plot(xx, row["Intercept"] + row["Slope"] * xx, lw=1.4,
                color=S.GATE_COLORS["pass" if passed else "fail"])
    txt = f"{header}\n{pid} | {_species_text(row)}\nn={int(row['N.Overlap'])}"
    if pd.notna(row["Slope"]): txt += f"; slope={row['Slope']:.2f}"
    if pd.notna(row["MAE"]): txt += f"; MAE={row['MAE']:.2f}"
    ax.set_title(txt, fontsize=7.5)
    ax.set_xlabel(RAW_XLABEL); ax.set_ylabel(DIANN_YLABEL)


def _run_intensity_mapping(cfg, L, sv, resolve, q, sp):
    """Everything intensity_mapping_mechanism.run() used to do except loading `q`/`sp` (now done
    once, shared with the signal-recovery/final-recovered sections -- see run() below). Returns
    the final-recovery-by-tier DataFrame (former Fig_Mechanism2_F's table) for
    panel_before_after_tiers in the final-recovered section, so it no longer has to be read back
    off disk."""
    T, MAIN, SUPP = L.table, L.main, L.supp
    MIN_SLOPE = float(cfg.get("min_slope", 0.5))
    MAX_SLOPE = float(cfg.get("max_slope", 1.5))
    MAX_MAE = float(cfg.get("max_mae", 0.5))
    # Production applicability safeguard: a guardrail against extreme extrapolation beyond the
    # fitted range, not a claim that prediction is uniformly accurate anywhere within it.
    MAX_EXTRAPOLATION = float(cfg.get("max_extrapolation_log2", 2.0))
    for k in ["mapping_metrics", "species_annotation", "quantity_diann", "quantity_jumplion"]:
        if not cfg.get(k):
            raise KeyError(f"mechanism.{k} is required.")
    if sp is None:
        raise KeyError("mechanism.species_annotation is required.")

    m = read_table(resolve(cfg["mapping_metrics"]))
    validate(m, REQ_METRICS, "mapping metrics")
    m = m[REQ_METRICS].drop_duplicates(ID).set_index(ID)
    m["Model.Good"] = as_bool(m["Model.Good"])
    eff = pd.to_numeric(m["Effective.Min.Overlap"], errors="coerce").dropna().unique()
    if len(eff) != 1:
        raise ValueError(f"Expected one Effective.Min.Overlap value, found: {eff}")
    MIN_OVERLAP = int(eff[0])

    write_tsv(sp.value_counts(dropna=False).rename_axis(SPECIES).reset_index(name="N.Precursors"),
              T, "Table_species_annotation_summary.tsv")
    sp3 = sp[sp.isin(SPECIES_ORDER)]
    m = m.join(sp3, how="left")

    # ---------- precursor outcomes ----------
    miss = q[DIANN].isna()
    rec_flag = q[STATUS].astype(str).str.lower().eq("recovered")
    ps = pd.DataFrame({ID: q[ID], RUN: q[RUN], "obs": ~miss, "miss": miss, "rec": rec_flag}).groupby(ID, sort=False).agg(
        N_Runs=(RUN, "nunique"), N_DIANN_Observed=("obs", "sum"), N_DIANN_Missing=("miss", "sum"),
        N_Recovered=("rec", "sum"))
    ps["Outcome"] = np.where(ps.N_DIANN_Missing == 0, "No missing",
                             np.where(ps.N_Recovered > 0, "Recovered", "Unresolved"))
    ps = ps.join(m, how="left")
    write_tsv(ps, T, "Supp_Mechanism2_B0_precursor_outcomes.tsv", index=True)

    # ---------- paired points + mapped reconciliation ----------
    p = q[q[DIANN].notna() & q[RAW].notna() & (q[DIANN] > 0) & (q[RAW] > 0)][[ID, RUN, DIANN, RAW]].copy()
    p["DIANN.log2"] = log2_series(p[DIANN])
    p["JUMPlion.Raw.log2"] = log2_series(p[RAW])
    p = p.join(m[["Slope", "Intercept", "Model.Good"]], on=ID)
    good_mask = p["Model.Good"].astype("boolean").fillna(False).to_numpy(bool)
    p["Mapped.log2"] = np.where(good_mask, p["Intercept"] + p["Slope"] * p["JUMPlion.Raw.log2"], np.nan)
    p["Raw.Signed.Error"] = p["JUMPlion.Raw.log2"] - p["DIANN.log2"]
    p["Mapped.Signed.Error"] = p["Mapped.log2"] - p["DIANN.log2"]

    ex = choose_examples(m, p)
    write_tsv(ex, T, "Supp_Mechanism2_E0_example_precursors.tsv", index=True)
    rej = choose_rejected_examples(m, p)
    write_tsv(rej, T, "Supp_Mechanism2_A_rejected_example_precursors.tsv", index=True)

    # ---------- feasibility by coverage (table only) ----------
    mm = m.copy()
    max_n = int(max(q.groupby(ID)[RUN].nunique().max(), mm["N.Overlap"].max()))
    edges, labels = coverage_bins(max_n, MIN_OVERLAP)
    mm["Coverage.Bin"] = pd.cut(mm["N.Overlap"], bins=edges, labels=labels, include_lowest=True)
    write_tsv(mm.groupby("Coverage.Bin", observed=False).agg(
        N_Precursors=("Model.Good", "size"), N_Mapping_Supported=("Model.Good", "sum"),
        Mapping_Supported_Fraction=("Model.Good", "mean")).reset_index(),
        T, "Table_mapping_feasibility_by_coverage.tsv")
    write_tsv(mm[["N.Overlap", "N.Overlap.Fraction", "Slope", "MAE", "R2", "Spearman", "Model.Good", "Coverage.Bin"]],
              T, "Table_overlap_vs_mapping_quality.tsv", index=True)

    # ---------- final recovery by tier ----------
    rec = q[rec_flag].copy()
    tier_counts = rec.groupby(TIER, dropna=False).size().rename("N.Recovered.PrecursorRuns").reset_index()
    order = {t: i for i, t in enumerate(["ms2_coelution", "ms1_peak", "prediction_only"])}
    tier_counts["__o"] = tier_counts[TIER].map(lambda t: order.get(S.tier_key(t), 9))
    tier_counts = tier_counts.sort_values(["__o", TIER]).drop(columns="__o").reset_index(drop=True)
    write_tsv(tier_counts, T, "Supp_Mechanism2_F0_final_recovery_by_localization_tier.tsv")

    # ---------- failure reasons / full metrics ----------
    write_tsv(m[["Model.Good", "Model.Fail.Reason"]], T, "Table_mapping_failure_reason_by_precursor.tsv", index=True)
    write_tsv(m["Model.Fail.Reason"].fillna("MISSING_REASON").value_counts().rename_axis("Reason")
              .reset_index(name="N.Precursors"), T, "Table_mapping_failure_reason_counts.tsv")
    write_tsv(m, T, "Table_mapping_model_metrics.tsv", index=True)

    # ---------- intensity bins (global quintiles of mean observed DIA-NN log2) ----------
    obs = q[q[DIANN].notna() & (q[DIANN] > 0)][[ID, DIANN]].copy()
    obs["DIANN.log2"] = log2_series(obs[DIANN])
    abundance = obs.groupby(ID)["DIANN.log2"].mean().rename("Mean.DIANN.log2")
    mech = m.join(abundance).dropna(subset=["Mean.DIANN.log2"]).copy()
    try:
        _, iedges = pd.qcut(mech["Mean.DIANN.log2"], q=5, retbins=True, duplicates="drop")
        iedges = np.unique(iedges)
    except ValueError:
        iedges = np.array([mech["Mean.DIANN.log2"].min() - 1e-9, mech["Mean.DIANN.log2"].max() + 1e-9])
    ilabels = [f"{iedges[i]:.2f}-{iedges[i + 1]:.2f}" for i in range(len(iedges) - 1)]
    mech["Intensity.Bin"] = pd.cut(mech["Mean.DIANN.log2"], bins=iedges, labels=ilabels, include_lowest=True)

    def gates(df):
        df["Overlap.Pass"] = df["N.Overlap"] >= MIN_OVERLAP
        df["Slope.Pass"] = df["Overlap.Pass"] & df["Slope"].between(MIN_SLOPE, MAX_SLOPE, inclusive="both")
        df["MAE.Pass"] = df["Slope.Pass"] & (df["MAE"] <= MAX_MAE)
        return df
    mech = gates(mech)
    isum = mech.groupby("Intensity.Bin", observed=False).agg(
        N_Precursors=("Model.Good", "size"), Median_Overlap=("N.Overlap", "median"),
        Overlap_Pass_Fraction=("Overlap.Pass", "mean"), Slope_Pass_Fraction=("Slope.Pass", "mean"),
        MAE_Pass_Fraction=("MAE.Pass", "mean"), Mapping_Supported_Fraction=("Model.Good", "mean")).reset_index()
    gb = lambda df, col: df.groupby("Intensity.Bin", observed=False)[col]
    for prefix, src, col in [("Overlap", mech, "N.Overlap"), ("Slope", mech[mech["Overlap.Pass"]], "Slope"),
                             ("MAE", mech[mech["Slope.Pass"]], "MAE")]:
        if prefix != "Overlap":
            isum[f"Median_{prefix}"] = isum["Intensity.Bin"].map(gb(src, col).median()).astype(float)
        qq = gb(src, col).quantile([.25, .75]).unstack()
        for qv, tag in [(.25, "Q25"), (.75, "Q75")]:
            isum[f"{prefix}_{tag}"] = isum["Intensity.Bin"].map(qq[qv] if qv in qq else pd.Series(dtype=float)).astype(float)
    write_tsv(mech, T, "Supp_Mechanism2_B_intensity_precursors.tsv", index=True)
    write_tsv(isum, T, "Supp_Mechanism2_D_intensity_mechanism_summary.tsv")
    write_tsv(pd.DataFrame({"Intensity.Bin": ilabels, "Lower.log2": iedges[:-1], "Upper.log2": iedges[1:]}),
              T, "Supp_Mechanism2_B_intensity_bin_ranges.tsv")

    rm = ps[["N_DIANN_Missing", "N_Recovered", "Outcome"]].join(abundance, how="left").join(sp3, how="left")
    rm = rm[rm["N_DIANN_Missing"] > 0].copy()
    rm["Recovered"] = rm["N_Recovered"] > 0
    rm["Intensity.Bin"] = pd.cut(rm["Mean.DIANN.log2"], bins=iedges, labels=ilabels, include_lowest=True)
    rec_agg = dict(N_Requiring_Recovery=("Recovered", "size"), N_Recovered=("Recovered", "sum"),
                   Recovery_Fraction=("Recovered", "mean"))
    irec = rm.groupby("Intensity.Bin", observed=False).agg(**rec_agg).reset_index()
    write_tsv(rm, T, "Supp_Mechanism2_B_intensity_recovery_precursors.tsv", index=True)
    write_tsv(irec, T, "Supp_Mechanism2_B_intensity_recovery_summary.tsv")

    # ---------- species ----------
    smech = mech[mech[SPECIES].isin(SPECIES_ORDER)].copy()
    rows = []
    for spc in SPECIES_ORDER:
        zz = smech[smech[SPECIES].eq(spc)]
        for ib in ilabels:
            z = zz[zz["Intensity.Bin"].astype(str).eq(ib)]
            rows.append({SPECIES: spc, "Intensity.Bin": ib, "N.Precursors": len(z),
                         "Median.Overlap": z["N.Overlap"].median(),
                         "Median.Slope": z.loc[z["Overlap.Pass"], "Slope"].median(),
                         "Median.MAE": z.loc[z["Slope.Pass"], "MAE"].median()})
    s_int = pd.DataFrame(rows)
    write_tsv(s_int, T, "Supp_Mechanism2_C_species_intensity_mechanism_summary.tsv")
    gs_rows = []
    for spc in SPECIES_ORDER:
        z = smech[smech[SPECIES].eq(spc)]
        n = len(z)
        gs_rows.append({SPECIES: spc, "N.MappingEvaluated": n, "N.Overlap.Pass": int(z["Overlap.Pass"].sum()),
                        "N.Slope.Pass": int(z["Slope.Pass"].sum()), "N.MAE.Pass": int(z["MAE.Pass"].sum()),
                        "N.Model.Good": int(z["Model.Good"].sum()),
                        "Overlap.Pass.Fraction": z["Overlap.Pass"].mean() if n else np.nan,
                        "Slope.Pass.Fraction.of.All": z["Slope.Pass"].mean() if n else np.nan,
                        "MAE.Pass.Fraction.of.All": z["MAE.Pass"].mean() if n else np.nan,
                        "Mapping.Supported.Fraction": z["Model.Good"].mean() if n else np.nan})
    write_tsv(smech, T, "Table_species_gate_precursors.tsv", index=True)
    write_tsv(pd.DataFrame(gs_rows), T, "Table_species_mapping_gate_summary.tsv")
    srec = rm[rm[SPECIES].isin(SPECIES_ORDER)].copy()
    write_tsv(srec.groupby(SPECIES).agg(**rec_agg).reindex(SPECIES_ORDER).reset_index(),
              T, "Table_species_recovery_summary.tsv")
    s_int_rec = srec.groupby([SPECIES, "Intensity.Bin"], observed=False).agg(**rec_agg).reset_index()
    write_tsv(s_int_rec, T, "Supp_Mechanism2_C_species_by_intensity_recovery.tsv")

    # ---------- dynamic range ----------
    dr = obs.groupby(ID)["DIANN.log2"].agg(P05=lambda v: v.quantile(.05), P95=lambda v: v.quantile(.95), N_Observed="size")
    dr["Dynamic.Range.log2"] = dr["P95"] - dr["P05"]
    dr = dr.join(sp3, how="left").join(m[["N.Overlap", "N.Overlap.Fraction", "Slope", "MAE", "Model.Good"]], how="left")
    dr = gates(dr)
    dre = dr[dr["Overlap.Pass"] & dr["Dynamic.Range.log2"].notna()].copy()
    if len(dre):
        _, dedges = pd.qcut(dre["Dynamic.Range.log2"], q=5, retbins=True, duplicates="drop")
        dedges = np.unique(dedges)
        if len(dedges) < 2:
            dedges = np.array([dedges[0] - 1e-9, dedges[0] + 1e-9])
    else:
        dedges = np.array([0.0, 1.0])
    dlabels = [f"{dedges[i]:.2f}-{dedges[i + 1]:.2f}" for i in range(len(dedges) - 1)]
    dre["Dynamic.Range.Bin"] = pd.cut(dre["Dynamic.Range.log2"], bins=dedges, labels=dlabels, include_lowest=True)
    dr_agg = dict(N_Precursors=("Slope", "size"), Median_Slope=("Slope", "median"), Slope_Pass_Fraction=("Slope.Pass", "mean"))
    write_tsv(dr, T, "Supp_Mechanism2_D_species_dynamic_range_precursors.tsv", index=True)
    dr_sum = dre.groupby("Dynamic.Range.Bin", observed=False).agg(**dr_agg).reset_index()
    write_tsv(dr_sum, T, "Supp_Mechanism2_D_dynamic_range_vs_slope_summary.tsv")
    s_dr = dre[dre[SPECIES].isin(SPECIES_ORDER)].groupby([SPECIES, "Dynamic.Range.Bin"], observed=False).agg(**dr_agg).reset_index()
    write_tsv(s_dr, T, "Supp_Mechanism2_D_species_dynamic_range_vs_slope.tsv")

    # ---------- raw vs global vs precursor-specific (in-sample) ----------
    pg = p[good_mask].copy()
    gsl, gin = pooled_fit(pg["JUMPlion.Raw.log2"], pg["DIANN.log2"])
    pg["Global.Mapped.log2"] = gin + gsl * pg["JUMPlion.Raw.log2"]
    pg["Global.Signed.Error"] = pg["Global.Mapped.log2"] - pg["DIANN.log2"]
    pg["PrecursorSpecific.Signed.Error"] = pg["Mapped.Signed.Error"]
    write_tsv(pg, T, "Supp_Mechanism2_E_mapping_comparison_points.tsv.gz")
    comp = pd.DataFrame([{"Mapping": n, **err_metrics(pg[c])} for n, c in
                         [("Raw", "Raw.Signed.Error"), ("Global", "Global.Signed.Error"),
                          ("Precursor-specific", "PrecursorSpecific.Signed.Error")]])
    comp["Global.Slope"], comp["Global.Intercept"] = gsl, gin
    write_tsv(comp, T, "Supp_Mechanism2_E_mapping_comparison_summary.tsv")

    # ---------- interpolation / extrapolation of recovered cells ----------
    fr = pg.groupby(ID)["JUMPlion.Raw.log2"].agg(["min", "max"]).rename(
        columns={"min": "Observed.Raw.Min.log2", "max": "Observed.Raw.Max.log2"})
    ext = rec[rec[RAW].notna() & (rec[RAW] > 0)].copy()
    ext["Raw.JUMPlion.log2"] = log2_series(ext[RAW])
    ext = ext.join(fr, on=ID).join(m[["Slope", "Intercept", "Model.Good"]], on=ID)
    ext = ext[ext["Model.Good"].astype("boolean").fillna(False).to_numpy(bool) & ext["Observed.Raw.Min.log2"].notna()].copy()
    lo, hi, v = ext["Observed.Raw.Min.log2"], ext["Observed.Raw.Max.log2"], ext["Raw.JUMPlion.log2"]
    ext["Range.Class"] = np.select([v < lo, v > hi], ["Below observed range", "Above observed range"], default="Within observed range")
    ext["Extrapolation.Distance.log2"] = np.where(v < lo, lo - v, np.where(v > hi, v - hi, 0.0))
    dd = ext["Extrapolation.Distance.log2"]
    sev_order = ["Within fitted range", "0-0.5 log2 outside", "0.5-1 log2 outside", "1-2 log2 outside", ">2 log2 outside"]
    ext["Severity"] = np.select([dd.eq(0), (dd > 0) & (dd <= .5), (dd > .5) & (dd <= 1), (dd > 1) & (dd <= 2), dd > 2],
                                sev_order, default="Unknown")
    ext["Mapped.log2"] = ext["Intercept"] + ext["Slope"] * v
    write_tsv(ext[[ID, RUN, RAW, "Raw.JUMPlion.log2", "Observed.Raw.Min.log2", "Observed.Raw.Max.log2", "Range.Class",
                   "Extrapolation.Distance.log2", "Severity", "Mapped.log2", TIER]],
              T, "Supp_Mechanism2_E_recovered_extrapolation.tsv.gz")
    write_tsv(ext.groupby(["Range.Class", "Severity"], dropna=False).size().rename("N.Recovered.PrecursorRuns").reset_index(),
              T, "Supp_Mechanism2_E_extrapolation_summary.tsv")

    # ================================ formerly Main, now Supp ================================
    # (former 4A workflow absorbed into panel_combined_workflow; see run())

    # former 4C distributions of the three sequential mapping criteria -- plotted later, combined
    # with former 4D/4E and S4E's panels b/c/d into Supp_Mechanism2_A_mapping_diagnostics (see
    # below, after the reliability analysis that panel's third row also needs).
    gate = gates(m.copy())
    write_tsv(gate[["N.Overlap", "Slope", "MAE", "Overlap.Pass", "Slope.Pass", "MAE.Pass", "Model.Good", "Model.Fail.Reason"]],
              T, "Supp_Mechanism2_C0_gate_distributions.tsv", index=True)
    ov = pd.to_numeric(gate["N.Overlap"], errors="coerce").dropna()

    # former 4D sequential, precursor-level mapping funnel: strictly overlap -> slope -> MAE,
    # ending at mapping-supported precursor models (no precursor-run or localization stages).
    funnel_stages = ["Mapping\nevaluated", "Overlap\npass", "Slope\npass", "MAE\npass",
                     "Mapping\nsupported"]
    funnel_counts = [len(gate), int(gate["Overlap.Pass"].sum()), int(gate["Slope.Pass"].sum()),
                     int(gate["MAE.Pass"].sum()), int(gate["Model.Good"].sum())]
    funnel = pd.DataFrame({"Stage": funnel_stages, "N.Precursors": funnel_counts})
    funnel["Percent.of.MappingEvaluated"] = 100 * funnel["N.Precursors"] / max(funnel_counts[0], 1)
    write_tsv(funnel, T, "Supp_Mechanism2_D0_mapping_funnel.tsv")

    # former 4E one accepted and one rejected mapping example; each panel has its own y-axis.

    # Applicability-stage counts (mapping-supported model -> candidate mapped precursor-run ->
    # within supported range) back the pass-branch numbers in the mapping workflow; no longer
    # plotted directly, since the applicability step itself is shown in that workflow panel.
    fit_ranges = p.groupby(ID)["JUMPlion.Raw.log2"].agg(["min", "max"]).rename(
        columns={"min": "Observed.Raw.Min.log2", "max": "Observed.Raw.Max.log2"}
    )
    cand = q[q[DIANN].isna() & q[RAW].notna() & (q[RAW] > 0)][[ID, RUN, RAW]].copy()
    cand = cand.join(m[["Model.Good", "Slope", "Intercept"]], on=ID).join(fit_ranges, on=ID)
    cand = cand[cand["Model.Good"].fillna(False)].copy()
    cand["Raw.log2"] = log2_series(cand[RAW])
    cand["Extrapolation.Distance.log2"] = np.maximum(
        cand["Observed.Raw.Min.log2"] - cand["Raw.log2"],
        cand["Raw.log2"] - cand["Observed.Raw.Max.log2"],
    ).clip(lower=0)
    cand["Within.Supported.Range"] = cand["Extrapolation.Distance.log2"] <= MAX_EXTRAPOLATION
    n_candidate = len(cand)
    n_supported = int(cand["Within.Supported.Range"].sum())
    n_final = int((q[STATUS].astype(str).str.lower() == "recovered").sum())

    write_tsv(pd.DataFrame({
        "Stage": ["Mapping supported", "Candidate mapped quantities", "Within supported range", "Final recovered"],
        "N.PrecursorRuns": [int(gate["Model.Good"].sum()), n_candidate, n_supported, n_final],
    }), T, "Table_Mechanism2_applicability_decision_stages.tsv")

    # ================================ SUPP (unchanged stems) ================================
    # S4A: two good examples (best fit, then a typical fit) followed by several rejected examples
    # by active failure mode -- formerly rejected-only (Supp_Mechanism2_A_rejected_mapping_
    # examples); renamed since it now opens with accepted examples too. `~m.index.isin(ex.index)`
    # excludes the accepted/rejected precursors already shown in Supp_Mechanism2_G0_mapping_
    # diagnostics (row 2, cols e/f -- built from the same `ex = choose_examples(m, p)`), so no
    # precursor appears in both panels.
    good_rows = []
    good_pool = m[m["Model.Good"].astype("boolean").fillna(False) & m.index.isin(p[ID].unique())
                 & ~m.index.isin(ex.index)].copy()
    # "High-agreement fit" is restricted to N.Overlap>=18 (enough paired observations that the fit isn't
    # just a lucky low-n MAE) -- "Representative accepted fit" below still draws from the full good_pool, and
    # doesn't depend on "High-agreement fit" having found a candidate.
    best_pool = good_pool[good_pool["N.Overlap"] >= 18]
    best = None
    if len(best_pool):
        best = best_pool.sort_values(["MAE", "N.Overlap"], ascending=[True, False]).iloc[[0]].copy()
        best["Example.Label"] = "High-agreement fit"
        good_rows.append(best)
    rest = good_pool.drop(index=best.index) if best is not None else good_pool
    if len(rest):
        med_mae = good_pool["MAE"].median()
        rest = rest.copy()
        rest["__d"] = (rest["MAE"] - med_mae).abs()
        typical = rest.sort_values(["__d", "N.Overlap"], ascending=[True, False]).iloc[[0]].drop(columns="__d").copy()
        typical["Example.Label"] = "Representative accepted fit"
        good_rows.append(typical)

    # Show multiple clear slope failures near 0.25 and 2.0, plus representative overlap/MAE failures.
    rejected_rows = []
    for label, reason, target, n_take in [
        ("Insufficient overlap", "INSUFFICIENT_OVERLAP", None, 2),
        ("Slope too low", "SLOPE_TOO_LOW", 0.25, 2),
        ("Slope too high", "SLOPE_TOO_HIGH", 2.0, 2),
        ("High MAE", "HIGH_MAE", None, 2),
    ]:
        z = m[(m["Model.Fail.Reason"] == reason) & m.index.isin(p[ID].unique())
             & ~m.index.isin(ex.index)].copy()
        if z.empty:
            continue
        if target is not None:
            z = z[z["Slope"].notna()].copy()
            z["__d"] = (z["Slope"] - target).abs()
            z = z.sort_values(["__d", "N.Overlap"], ascending=[True, False]).drop(columns="__d")
        elif label == "Insufficient overlap":
            z = z.sort_values("N.Overlap", ascending=False)
        else:
            z = z.sort_values(["MAE", "N.Overlap"], ascending=[False, False])
        z = z.head(n_take).copy(); z["Example.Label"] = label; rejected_rows.append(z)

    combo = pd.concat(good_rows + rejected_rows) if (good_rows or rejected_rows) else rej.head(0)
    write_tsv(combo, T, "Supp_Mechanism2_B_mapping_examples.tsv", index=True)
    ncombo = max(len(combo), 1)
    ncol = min(4, ncombo); nrow = int(np.ceil(ncombo / ncol))
    # Width capped at <=7.2in (regular document width) regardless of ncol.
    fig, axs = plt.subplots(nrow, ncol, figsize=(2.7 * ncol, 2.7 * nrow), squeeze=False)
    for jj, ax in enumerate(axs.ravel()):
        if jj >= len(combo):
            ax.axis("off"); continue
        row = combo.iloc[jj]
        passed = row["Example.Label"] in ("High-agreement fit", "Representative accepted fit")
        mapping_scatter(ax, p, combo.index[jj], row, row["Example.Label"], passed)
    fig.tight_layout()
    sv(fig, SUPP, "Supp_Mechanism2_B_mapping_examples")

    def bin_axes(axes, labs, xlabel):
        for ax in axes:
            ax.set_xticks(np.arange(len(labs))); ax.set_xticklabels(labs, rotation=25, ha="right")
            ax.set_xlabel(xlabel)

    # S4B: precursor abundance and the three mapping criteria + recovery.
    xb = np.arange(len(isum))
    fig, axs = plt.subplots(2, 2, figsize=(5, 4.8))
    for ax, prefix, ylab, refs, letter in [
        (axs[0, 0], "Overlap", "Paired observations", [MIN_OVERLAP], "a"),
        (axs[0, 1], "Slope", "Slope", [MIN_SLOPE, MAX_SLOPE], "b"),
        (axs[1, 0], "MAE", "MAE (log2)", [MAX_MAE], "c"),
    ]:
        med = isum[f"Median_{prefix}"].to_numpy(float)
        q25 = isum[f"{prefix}_Q25"].to_numpy(float)
        q75 = isum[f"{prefix}_Q75"].to_numpy(float)
        yerr = np.vstack([med - q25, q75 - med])
        ax.errorbar(xb, med, yerr=yerr, marker="o", color=S.NEUTRAL, capsize=3, lw=1.1)
        for r0 in refs:
            S.ref_line(ax, r0)
        ax.set_ylabel(ylab); S.panel_letter(ax, letter)
    axs[1, 1].plot(np.arange(len(irec)), 100 * irec["Recovery_Fraction"], marker="o", color=S.RECOVERY_COLORS["Recovered"])
    axs[1, 1].set_ylabel("Recovered (%)"); axs[1, 1].set_ylim(0, 100); S.panel_letter(axs[1, 1], "d")
    bin_axes(axs.ravel(), list(isum["Intensity.Bin"].astype(str)), INT_XLABEL)
    fig.tight_layout()
    sv(fig, SUPP, "Supp_Mechanism2_D_intensity_mechanism")

    # S4C: species x the same global intensity bins.
    fig, axs = plt.subplots(2, 2, figsize=(5, 4.8))
    xx = np.arange(len(ilabels))
    for spc in SPECIES_ORDER:
        clr, lab = S.SPECIES_COLORS[spc], S.species_label(spc)
        z = s_int[s_int[SPECIES].eq(spc)].set_index("Intensity.Bin").reindex(ilabels)
        axs[0, 0].plot(xx, z["Median.Overlap"], marker="o", color=clr, label=lab)
        axs[0, 1].plot(xx, z["Median.Slope"], marker="o", color=clr, label=lab)
        axs[1, 0].plot(xx, z["Median.MAE"], marker="o", color=clr, label=lab)
        zr = s_int_rec[s_int_rec[SPECIES].eq(spc)].copy(); zr["Intensity.Bin"] = zr["Intensity.Bin"].astype(str)
        zr = zr.set_index("Intensity.Bin").reindex(ilabels)
        axs[1, 1].plot(xx, 100 * zr["Recovery_Fraction"], marker="o", color=clr, label=lab)
    S.ref_line(axs[0, 0], MIN_OVERLAP); axs[0, 0].set_ylabel("Median paired observations")
    S.ref_line(axs[0, 1], MIN_SLOPE); S.ref_line(axs[0, 1], MAX_SLOPE); axs[0, 1].set_ylabel("Median slope")
    S.ref_line(axs[1, 0], MAX_MAE); axs[1, 0].set_ylabel("Median MAE (log2)")
    axs[1, 1].set_ylabel("Recovered (%)"); axs[1, 1].set_ylim(0, 100)
    for ax, letter in zip(axs.ravel(), "abcd"): S.panel_letter(ax, letter)
    bin_axes(axs.ravel(), ilabels, INT_XLABEL)
    S.fig_legend(fig, *axs[0, 0].get_legend_handles_labels(), ncol=3, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, .96])
    sv(fig, SUPP, "Supp_Mechanism2_E_species_by_intensity_mechanism")

    # Supp_Mechanism2_A_mapping_diagnostics: former standalone C0 (gate distributions), D0
    # (mapping funnel), and E0 (accepted/rejected examples), combined into one 2x3 grid -- row 1:
    # the three sequential gate distributions; row 2: the mapping funnel plus two mapping
    # examples. No LOO/rel dependency (the extrapolation-distance panels that used to be this
    # figure's row 3 now live in Supp_Mechanism2_C_threshold_sensitivity_and_robustness instead,
    # built later by _run_mapping_reliability).
    fig, axs = plt.subplots(2, 3, figsize=(7.8, 4.8))

    # Row 1 (former C0): the three sequential gate distributions.
    axs[0, 0].hist(ov, bins=np.arange(-.5, int(ov.max()) + 1.5, 1), color=S.NEUTRAL, edgecolor="white", linewidth=.3)
    def gate_label(ax, x, text, side):
        """Pass criterion, horizontal, beside a dashed gate line at x (side +1: right of the line, -1: left), at the top."""
        ax.annotate(text, xy=(x, .97), xycoords=ax.get_xaxis_transform(), xytext=(2.5 * side, 0),
                    textcoords="offset points", ha="left" if side > 0 else "right", va="top",
                    fontsize=5.5, color=S.NEUTRAL)

    S.ref_line(axs[0, 0], MIN_OVERLAP - .5, "v")
    gate_label(axs[0, 0], MIN_OVERLAP - .5, f"pass: ≥ {MIN_OVERLAP}", +1)
    axs[0, 0].set_xlabel("Paired observations")
    axs[0, 1].hist(gate.loc[gate["Overlap.Pass"], "Slope"].dropna(), bins=50, color=S.NEUTRAL, edgecolor="white", linewidth=.3)
    S.ref_line(axs[0, 1], MIN_SLOPE, "v"); S.ref_line(axs[0, 1], MAX_SLOPE, "v")
    # outside the passing range (left of the lower line, right of the upper one) so the two labels never meet
    gate_label(axs[0, 1], MIN_SLOPE, f"pass: ≥ {MIN_SLOPE:g}", -1)
    gate_label(axs[0, 1], MAX_SLOPE, f"pass: ≤ {MAX_SLOPE:g}", +1)
    S.ref_line(axs[0, 1], 1, "v", color="red"); axs[0, 1].set_xlim(-2.5, 4.5)
    axs[0, 1].set_xlabel("Mapping slope")
    axs[0, 2].hist(gate.loc[gate["Slope.Pass"], "MAE"].dropna(), bins=50, color=S.NEUTRAL, edgecolor="white", linewidth=.3)
    S.ref_line(axs[0, 2], MAX_MAE, "v")
    gate_label(axs[0, 2], MAX_MAE, f"pass: ≤ {MAX_MAE:g}", +1)  # right of the line: the tall bars are on the passing side
    axs[0, 2].set_xlim(-0.1, 1.6)
    axs[0, 2].set_xlabel("MAE (log2)")
    for ax, letter in zip(axs[0], "abc"):
        ax.set_ylabel("Precursors"); S.panel_letter(ax, letter)

    # Row 2, col 0 (former D0): the mapping funnel.
    xxf = np.arange(len(funnel_stages))
    axs[1, 0].plot(xxf, funnel["Percent.of.MappingEvaluated"], color=S.NEUTRAL, marker="o", lw=1.2)
    axs[1, 0].set_xticks(xxf); axs[1, 0].set_xticklabels(funnel_stages, fontsize=7)
    axs[1, 0].set_ylabel("% of precursors"); axs[1, 0].set_ylim(0, 118)
    axs[1, 0].set_xlim(-0.8, len(funnel_stages) - 1 + 0.4)  # room left of the first point for its annotation
    for i, row in funnel.iterrows():
        axs[1, 0].text(i, row["Percent.of.MappingEvaluated"] + 4,
                       f"{int(row['N.Precursors']):,}\n({row['Percent.of.MappingEvaluated']:.1f}%)",
                       ha="center", va="bottom", fontsize=7)
    S.panel_letter(axs[1, 0], "d")

    # Row 2, cols 1-2 (former E0): one accepted and one rejected mapping example.
    for jj, ax in enumerate([axs[1, 1], axs[1, 2]]):
        if jj >= len(ex):
            ax.axis("off"); continue
        row = ex.iloc[jj]
        header = str(row.get("Example.Type", "Example"))
        if not bool(row["Model.Good"]):
            header += f" ({str(row.get('Model.Fail.Reason', '')).replace('_', ' ').lower()})"
        mapping_scatter(ax, p, ex.index[jj], row, header, bool(row["Model.Good"]))
    S.panel_letter(axs[1, 1], "e"); S.panel_letter(axs[1, 2], "f")

    fig.tight_layout()
    _letters_left_of_ylabels(fig, axs)
    sv(fig, SUPP, "Supp_Mechanism2_A_mapping_diagnostics")

    log.info("Intensity-mapping section finished (reliability/LOO analysis runs later).")
    return tier_counts, comp


def _run_mapping_reliability(cfg, L, sv, resolve, comp):
    """The LOO-based reliability analysis (slow, and can fail production-fit validation) plus the
    one panel that depends on it, Supp_Mechanism2_C_threshold_sensitivity_and_robustness. Split
    out of _run_intensity_mapping and called separately, after the Main final-recovered section in
    run(), so a LOO failure can't prevent the Main panels (or anything else in this module) from
    being generated -- `comp` (the raw/global/precursor-specific MAE comparison, panel h) is the
    only other-section data this still needs, passed in since it's otherwise cheap to keep from
    _run_intensity_mapping rather than recompute."""
    T, SUPP = L.table, L.supp
    MIN_SLOPE = float(cfg.get("min_slope", 0.5))
    MAX_SLOPE = float(cfg.get("max_slope", 1.5))
    MAX_MAE = float(cfg.get("max_mae", 0.5))
    MAX_EXTRAPOLATION = float(cfg.get("max_extrapolation_log2", 2.0))

    # Panels a/b: retention-vs-fragility tradeoff, one point per candidate threshold, reusing the
    # existing full-model and LOO results (no refitting). Retention (% of overlap/MAE(or
    # slope)-eligible models accepted) and LOO-fragility (% of THOSE accepted models that are
    # LOO-fragile) have different denominators, so they are plotted as x/y of the same point
    # rather than as two lines sharing one axis; all points use the same marker size (held-out
    # MAE values are in the source-data table, not annotated on the plot). This is a robustness
    # check -- does shifting/narrowing the window materially change retention, fragility, or
    # held-out accuracy -- not a search for a data-optimized cutoff, so the production point is
    # marked but never called "optimal".
    def _tradeoff_plot(ax, sens, labels, is_production):
        xr = 100 * sens["Accepted.Fraction"].to_numpy(float)
        yf = 100 * sens["LOO.Fragile.Fraction"].to_numpy(float)
        prod = np.asarray(is_production, dtype=bool)
        ax.plot(xr, yf, "-", color=S.LIGHT, lw=0.8, zorder=1)
        ax.scatter(xr[~prod], yf[~prod], s=42, color=S.NEUTRAL, alpha=.8, edgecolors="none", zorder=2)
        if prod.any():
            ax.scatter(xr[prod], yf[prod], s=42, color=S.GATE_COLORS["pass"], alpha=.9,
                      edgecolors="none", zorder=3, label="Production")
        for xi, yi, lab in zip(xr, yf, labels):
            ax.annotate(str(lab), (xi, yi), textcoords="offset points", xytext=(4, 3), fontsize=5.5)
        ax.set_xlabel("Models retained (%)"); ax.set_ylabel("LOO-fragile among retained models (%)")

    # S4D/S4E: unified reliability analysis. Each LOO fit is computed once and reused for
    # robustness, threshold sensitivity, and extrapolation validation. A single max_mae (from
    # config, same MAX_MAE used by former Main Figure 4 and S4B/S4C) is passed through so the
    # production criterion is identical everywhere in this module.
    reliability_out = T / "reliability_analysis"
    rel = run_reliability(
        quantity_diann=resolve(cfg["quantity_diann"]),
        quantity_jumplion=resolve(cfg["quantity_jumplion"]),
        mapping_metrics=resolve(cfg["mapping_metrics"]),
        output_dir=reliability_out,
        species_annotation=resolve(cfg["species_annotation"]),
        quantification_parquet=resolve(cfg["quantification_parquet"]),
        production_module=cfg.get("production_module", "jumplion.intensity_mapping"),
        production_module_dir=resolve(cfg["production_module_dir"]) if cfg.get("production_module_dir") else None,
        min_slope=MIN_SLOPE, max_slope=MAX_SLOPE, max_mae=MAX_MAE,
        max_extrapolation_log2=MAX_EXTRAPOLATION,
        require_production_match=True,
    )
    vcheck = rel["validation_summary"]
    write_tsv(vcheck,T,"Table_Mechanism2_reliability_validation.tsv")
    match = vcheck.loc[vcheck["Metric"].eq("Fit.Match.Fraction"),"Value"]
    if len(match) and not np.isclose(float(match.iloc[0]), 1.0):
        raise RuntimeError("Reliability analysis failed production-fit validation; S4D/S4E were not generated.")

    # S4D: slope/MAE threshold sensitivity + mapping robustness.
    slope_sens = rel["slope_threshold_sensitivity"]
    mae_sens = rel["mae_threshold_sensitivity"]
    rb_overlap = rel["stability_by_overlap"]
    rb_huber = rel["huber_groups"]
    rb_conc = rel["huber_concordance"]
    write_tsv(slope_sens,T,"Supp_Mechanism2_D1_slope_threshold_sensitivity.tsv")
    write_tsv(mae_sens,T,"Supp_Mechanism2_D2_MAE_threshold_sensitivity.tsv")
    write_tsv(rb_overlap,T,"Supp_Mechanism2_D3_LOO_slope_stability_vs_overlap.tsv")
    write_tsv(rb_huber,T,"Supp_Mechanism2_D4_OLS_Huber_decision_groups.tsv")
    write_tsv(rb_conc,T,"Supp_Mechanism2_D4_OLS_Huber_concordance.tsv")
    write_tsv(rel["gate_instability"],T,"Table_Mechanism2_D_gate_specific_instability.tsv")

    # Extrapolation validation data (former S4E) and the production extrapolation safeguard
    # (d_extrap <= MAX_EXTRAPOLATION log2) -- a guardrail against extreme extrapolation, not a
    # claim that held-out prediction is uniformly accurate anywhere within that distance. Computed
    # here (moved up from after the D panel) since D's row 2 is now these extrapolation-bin plots.
    ex_bin=rel["extrapolation_summary"]; ex_ret=rel["cutoff_retention"]; ex_int=rel["cutoff_by_intensity"]; ex_tier=rel["cutoff_by_localization_tier"]
    write_tsv(ex_bin,T,"Supp_Mechanism2_E1_extrapolation_LOO_bin_summary.tsv"); write_tsv(ex_ret,T,"Supp_Mechanism2_E2_cutoff_retention.tsv"); write_tsv(ex_int,T,"Table_Mechanism2_E_cutoff_by_intensity.tsv")
    if len(ex_tier): write_tsv(ex_tier,T,"Supp_Mechanism2_E4_cutoff_by_localization_tier.tsv")
    labs = ex_bin["Extrapolation.Bin"].astype(str).tolist()
    xx = np.arange(len(labs))
    # DISTANCE_LABELS bins are fixed ("Within fitted range", "0-0.5", "0.5-1", "1-2", ">2 log2
    # outside"); the guardrail at MAX_EXTRAPOLATION=2 falls between the last two bins. The x-axis
    # is categorical, so a dashed vline there reads oddly -- shade the ">2 log2 outside" category
    # instead and label the boundary directly.
    def _mark_guardrail(ax, n):
        ax.axvspan(n - 1.5, n - 0.5, color=S.LIGHT, alpha=0.25, zorder=0)
        ax.text(n - 1.5, 0.97, "production limit: 2 log$_2$", transform=ax.get_xaxis_transform(),
                rotation=90, ha="center", va="top", fontsize=6, color=S.REF_LINE)

    # Supp_Mechanism2_C_threshold_sensitivity_and_robustness: 3x3 grid -- row 1 = a/b/c (threshold
    # sensitivity/robustness, as before); row 2 = d/e/f, extrapolation-distance validation (former
    # Supp_Mechanism2_E_extrapolation_validation panels b/c/d, moved in from
    # Supp_Mechanism2_A_mapping_diagnostics' former row 3, which freed G0 of any LOO/rel
    # dependency); row 3 = g (OLS/Huber decision groups, was "d") and h (former standalone
    # Supp_Mechanism2_F_quantitative_reconciliation, was "e"). The third cell of row 3 is empty.
    fig, axs = plt.subplots(3, 3, figsize=(7.5, 7.2))
    ax=axs[0,0]
    _tradeoff_plot(ax, slope_sens, slope_sens["Label"], slope_sens["Is.Production"])
    S.panel_letter(ax,"a")
    ax=axs[0,1]
    _tradeoff_plot(ax, mae_sens, [f"{v:.2f}" for v in mae_sens["MAE.Cutoff"]], mae_sens["Is.Production"])
    S.panel_letter(ax,"b")
    ax=axs[0,2]; ax.plot(rb_overlap["Full.N.Overlap"],rb_overlap["Median_Abs_Delta_Slope"],marker="o",color=S.NEUTRAL,label="Median"); ax.plot(rb_overlap["Full.N.Overlap"],rb_overlap["P95_Abs_Delta_Slope"],marker="o",color=S.LIGHT,label="P95")
    ax.set_xlabel("Paired observations in full model"); ax.set_ylabel("|Delta slope| after eligible LOO"); ax.legend(frameon=False); S.panel_letter(ax,"c")

    _mark_guardrail(axs[1,0], len(labs))
    axs[1,0].plot(xx,ex_bin["MAE"],marker="o",color=S.NEUTRAL,zorder=2)
    axs[1,0].set_xticks(xx); axs[1,0].set_xticklabels(labs,rotation=28,ha="right"); axs[1,0].set_ylabel("Held-out prediction MAE (log2)"); S.panel_letter(axs[1,0],"d")
    _mark_guardrail(axs[1,1], len(labs))
    axs[1,1].plot(xx,ex_bin["Median.Signed.Error"],marker="o",color=S.NEUTRAL,zorder=2); axs[1,1].axhline(0,ls="--",lw=.8,color=S.LIGHT)
    axs[1,1].set_xticks(xx); axs[1,1].set_xticklabels(labs,rotation=28,ha="right"); axs[1,1].set_ylabel("Median signed error (log2)"); S.panel_letter(axs[1,1],"e")
    xr2=np.arange(len(ex_ret)); axs[1,2].plot(xr2,100*ex_ret["Retained.Fraction"],marker="o",color=S.RECOVERY_COLORS["Recovered"])
    prod_c = np.where(ex_ret["Is.Production"].to_numpy(bool))[0]
    if len(prod_c):
        i = int(prod_c[0])
        axs[1,2].scatter([xr2[i]],[100*ex_ret["Retained.Fraction"].iloc[i]],s=80,facecolors="none",
                         color=S.GATE_COLORS["pass"],linewidths=1.7,zorder=3)
    axs[1,2].set_xticks(xr2); axs[1,2].set_xticklabels(ex_ret["Cutoff"],rotation=25,ha="right"); axs[1,2].set_ylabel("Current recoveries retained (%)"); axs[1,2].set_ylim(0,105); S.panel_letter(axs[1,2],"f")

    ax=axs[2,0]; order=["Both pass","OLS only","Huber only","Both fail"]; zh=rb_huber.set_index("Decision.Group").reindex(order).fillna(0); ax.bar(np.arange(len(zh)),zh["N.Precursors"],color=S.NEUTRAL,width=.65); ax.set_xticks(np.arange(len(zh))); ax.set_xticklabels(order,rotation=20,ha="right"); ax.set_ylabel("Precursors")
    cr=rb_conc.loc[rb_conc["Metric"].eq("Decision.Concordance.Fraction"),"Value"]
    if len(cr): ax.text(.98,.95,f"Concordance = {100*float(cr.iloc[0]):.1f}%",transform=ax.transAxes,ha="right",va="top",fontsize=7)
    S.panel_letter(ax,"g")

    write_tsv(comp,T,"Supp_Mechanism2_F_quantitative_reconciliation.tsv")
    ax=axs[2,1]
    bars=ax.bar(range(len(comp)),comp["MAE"],color=[S.LIGHT,S.NEUTRAL,S.METHOD_COLORS["JUMPlion"]],width=.65)
    S.annotate_bars(ax,bars,[f"{v:.2f}" for v in comp["MAE"]])
    ax.set_xticks(range(len(comp))); ax.set_xticklabels(comp["Mapping"],rotation=20,ha="right")
    ax.set_ylabel("MAE to paired DIA-NN (log2)"); S.panel_letter(ax,"h")
    axs[2,2].axis("off")

    fig.tight_layout(); _letters_left_of_ylabels(fig, axs)
    sv(fig,SUPP,"Supp_Mechanism2_C_threshold_sensitivity_and_robustness")
    log.info("Reliability/LOO analysis finished.")


# ----------------------------------------------------------------------------
# Final recovered (former final_recovered.py; stays Main, unchanged panels/stems)
# ----------------------------------------------------------------------------
def panel_before_after_tiers(before_counts, after_counts, L, sv):
    """Grouped bar chart: precursor-run entries per tier before intensity mapping
    (localization-tier assignment only) vs after (final recovered -- also passed the
    intensity-mapping/applicability gate). Includes "reported" as a reference bar alongside the 3
    recovery tiers -- reported entries never go through the intensity-mapping recovery gate
    (they're already final), so their "after" is just the same count as "before" (100% retention
    by definition), unlike the recovery tiers' real attrition."""
    # Normalize tier spellings via S.tier_key() before indexing -- evidence_tier_label in the
    # intensity-mapping table isn't guaranteed to use the same literal snake_case strings as
    # PANEL_A_TIERS (that's what tier_key()'s alias table is for), and a silent mismatch here
    # would reindex everything to 0 instead of erroring.
    before_counts = before_counts.assign(tier=before_counts["tier"].map(S.tier_key))
    after_counts = after_counts.assign(evidence_tier_label=after_counts["evidence_tier_label"].map(S.tier_key))
    before = (before_counts.groupby("tier")["n"].sum()
             .reindex(PANEL_A_TIERS).fillna(0).astype(int))
    after = (after_counts.groupby("evidence_tier_label")["N.Recovered.PrecursorRuns"].sum()
            .reindex(RECOVERY_TIERS).fillna(0).astype(int))
    after["reported"] = before["reported"]
    after = after.reindex(PANEL_A_TIERS)
    write_tsv(_tier_counts_table(before, after), L.table, "Fig_Mechanism_A_tier_counts.tsv")

    # Two adjacent bars per recovery tier ("Missing" gray, "Recovered" green), not one stacked
    # bar -- the two counts are meant to be compared directly, which a shared baseline (both bars
    # starting at 0) makes easier to read than one on top of the other. "Missing" is the total
    # originally missing for that tier (= the sum of the two segments the old stacked bar had,
    # i.e. "recovered" + "still missing"), not just the leftover-after-recovery amount -- so it
    # equals `before`, the same total the stacked bar's full height used to show. "reported"
    # stays its own single blue bar (before == after by definition, so there's nothing to split).
    x = np.arange(len(PANEL_A_TIERS))
    fig, ax = plt.subplots(figsize=(2.4, 2.1))
    blue = S.tier_color("reported")
    missing_gray, recovered_green = "#BDBDBD", RECOVERED_GREEN
    bar_w, gap = 0.32, 0.03

    # "reported" is the same width as every other bar (not double-width) -- it's its own single
    # bar, not a missing/recovered pair, but it should still look like one bar among four.
    ax.bar(x[:1], before.values[:1], width=bar_w, color=blue, edgecolor=S.NEUTRAL, linewidth=.4)
    # ha="left" (not "center"): a centered label on the leftmost bar grows half its width to the
    # left, which crosses the y-axis spine for a wide 6-digit count -- anchoring it at the bar's
    # left edge instead keeps the whole label inside the axes.
    if before.values[0] > 0:
        ax.text(x[0] - bar_w / 2, before.values[0], f"{before.values[0]:,}", ha="left", va="bottom",
               fontsize=ANNOT_FONTSIZE)

    rec_x = x[1:]
    xm, xr = rec_x - (bar_w + gap) / 2, rec_x + (bar_w + gap) / 2
    miss_vals, rec_vals = before.values[1:], after.values[1:]
    ax.bar(xm, miss_vals, width=bar_w, color=missing_gray, edgecolor=S.NEUTRAL, linewidth=.4, label="Missing")
    ax.bar(xr, rec_vals, width=bar_w, color=recovered_green, edgecolor=S.NEUTRAL, linewidth=.4, label="Recovered")
    # Same approach as Fig_Mechanism_C_yeast_ecoli_by_group: each label is centered above its own
    # bar (not leaning outward -- that just traded within-pair collisions for between-group ones,
    # since the two adjacent pairs' outward-growing labels met in the gap between them). Instead,
    # enforce a minimum vertical gap between a pair's two labels, pushing the taller bar's label
    # further up when they'd otherwise sit at nearly the same height and collide horizontally.
    ymax_ref = max(int(before.max()), 1) * 1.15
    pad, min_gap = .015 * ymax_ref, .05 * ymax_ref
    for xi, a, r in zip(rec_x, miss_vals, rec_vals):
        a_label_y, r_label_y = a + pad, r + pad
        if a > 0 and r > 0 and abs(a - r) < min_gap:
            if a >= r:
                a_label_y = r_label_y + min_gap  # missing is taller -- push its label further up
            else:
                r_label_y = a_label_y + min_gap  # recovered is taller -- push its label further up
        if a > 0:
            ax.text(xi - (bar_w + gap) / 2, a_label_y, f"{a:,.0f}", ha="center", va="bottom", fontsize=ANNOT_FONTSIZE)
        if r > 0:
            ax.text(xi + (bar_w + gap) / 2, r_label_y, f"{r:,.0f}", ha="center", va="bottom", fontsize=ANNOT_FONTSIZE)
    two_level_tier_axis(fig, ax, PANEL_A_TIERS)
    ax.set_ylabel("Number of precursor quantities")
    ax.set_ylim(0, max(int(before.max()), 1) * 1.15)
    #ax.set_xlim(x[0] - 0.8, x[-1] + 0.7)  # extra left room so the first (widest) label clears the y-axis
    # Explicit swatched handles instead of the auto-generated ones -- those would otherwise pick
    # up "reported"'s blue (the first bar's color), not representative since the other 3 bars are
    # now a gray/green pair. "Reported" gets its own entry in its actual bar color (blue), so the
    # legend also explains why the first bar looks different from the other three.
    legend_handles = [
        Patch(facecolor=blue, edgecolor=S.NEUTRAL, linewidth=.4, label="DIA-NN reported"),
        Patch(facecolor=missing_gray, edgecolor=S.NEUTRAL, linewidth=.4, label="DIA-NN missing"),
        Patch(facecolor=recovered_green, edgecolor=S.NEUTRAL, linewidth=.4, label="JUMPlion recovered"),
    ]
    ax.legend(handles=legend_handles, frameon=False, fontsize=ANNOT_FONTSIZE, loc="upper right")
    fig.tight_layout()
    # left/right shared with C (COL1_LEFT/RIGHT); bottom shared with B (ROW1_BOTTOM) -- see those
    # constants above.
    fig.subplots_adjust(left=COL1_LEFT+0.05, right=COL1_RIGHT, bottom=ROW1_BOTTOM)
    sv(fig, L.main, "Fig_Mechanism_A_tier_counts_before_after", tight=False)


def _recovered_evidence_view(evidence, recovered_only=True):
    """Rows of the shared evidence DataFrame restricted to two groups kept side by side for
    comparison: "reported" tier rows (all of them, unfiltered), and the 3 recovery tiers but ONLY
    rows that ended up with Final.Status=="recovered" after intensity mapping (the actual final
    recovered subset). `recovered_only=False` keeps every precursor-run entry of the 3 recovery
    tiers instead (no Final.Status filter). Returns None if Final.Status is needed but isn't
    present at all."""
    tier = evidence[TIER_COL].astype(str)
    is_reported = tier.eq("reported")
    if not recovered_only:
        return evidence[is_reported | tier.isin(RECOVERY_TIERS)].copy()
    if "Final.Status" not in evidence.columns:
        log.warning("mechanism: 'Final.Status' not found -- skipping the "
                    "recovered-subset distribution panels.")
        return None
    is_recovered_tier = tier.isin(RECOVERY_TIERS) & evidence["Final.Status"].astype(str).str.lower().eq("recovered")
    return evidence[is_reported | is_recovered_tier].copy()


def panel_recovered_distributions(evidence, L, sv, max_pts, stem="Fig_Mechanism_B_signal_rank_coelution"):
    """Signal rank and co-elution score, recovery tiers restricted to the recovered subset (see
    _recovered_evidence_view) while "reported" stays as the same unfiltered baseline, so
    recovered-vs-reported is a direct comparison, across all 4 tiers. Per-subplot ylim: signal
    rank auto-scales (it's already a 0-1 percentile); co-elution score is capped around (0, 7)
    but with matplotlib's usual ~5% autoscale margin added on each side, rather than cutting off
    exactly at the limit. figsize is set explicitly (rather than four_tier_violins' own
    min(7.2, 3.6*n_features) default) to control this panel's column width in
    Fig_Mechanism_combined -- see assemble_main_figure. left/right/wspace are pinned to
    COL2_B_LEFT/RIGHT/WSPACE -- the same values panel_recovered_signal_rank_by_group (Main D)
    uses for its own 2-subplot figure -- so the two panels' subplot axes boxes line up. colors:
    PANEL_AB_COLORS greens out the 3 recovery tiers, de-emphasizing which tier in favor of
    reported-vs-recovered."""
    coelution_margin = 0.05 * (7 - 0)  # matplotlib's default axes.margins fraction
    four_tier_violins(evidence, ["ms2_xics_signal_rank", "ms2_corr_sum"],
                      ["Relative MS2 intensity", "Fragment co-elution score"], [None, None],
                      L.main, L.table, stem, sv, max_pts,
                      left=COL2_B_LEFT, right=COL2_B_RIGHT, wspace=COL2_B_WSPACE, bottom=ROW1_BOTTOM,
                      # top=0.88: like panel C, this panel has no title eating into its top
                      # margin, so its y-axis label reaches the canvas top and collides with the
                      # composite's bold letter labels ("b"/"c") -- reserve headroom for them.
                      top=0.95,
                      # One colour per group (S.tier_color), named on the x axis -- no legend.
                      # TIERS (four_tier_violins' default tier order) matches PANEL_A_TIERS exactly.
                      colors=[S.tier_color(t) for t in PANEL_A_TIERS],
                      ylim=[None, (0 - coelution_margin, 7 + coelution_margin)],
                      figsize=(4.4, 2.1), tight=False,
                      short_labels=True)


def _recovery_tier_rows_view(evidence):
    """ALL rows of the shared evidence DataFrame in one of the 3 recovery tiers, regardless of
    Final.Status -- the "totally evaluated" population the group/species panels compare against
    the Final.Status=="recovered" subset ("finally recovered"). Unlike
    _recovered_evidence_view(), this is NOT pre-filtered to recovered rows and excludes "reported"
    entirely."""
    if "Final.Status" not in evidence.columns:
        log.warning("mechanism: 'Final.Status' not found -- skipping the group/species panels.")
        return None
    tier = evidence[TIER_COL].astype(str)
    return evidence[tier.isin(RECOVERY_TIERS)].copy()


def classify_group_species(d, cfg, resolve, sp):
    """Adds sample_group (from Run via sample_meta_file) and species (from Precursor.Id via the
    shared species annotation `sp`) columns to `d`; drops rows missing either. Returns
    (classified_df, group_labels) or (None, None) if required config/data is missing."""
    meta_path = resolve(cfg.get("sample_meta_file"))
    if meta_path is None or sp is None:
        log.warning("mechanism: sample_meta_file/species_annotation not configured -- "
                    "skipping the group/species panels.")
        return None, None
    bench = cfg.get("benchmark", {})
    groups_cfg = bench.get("groups") or {}
    if not groups_cfg:
        log.warning("mechanism: no benchmark.groups configured -- skipping the "
                    "group/species panels.")
        return None, None
    group_labels = list(groups_cfg)  # e.g. ["S1", ..., "S5"], in config order

    meta = pd.read_csv(meta_path)
    run_to_mixture = dict(zip(meta["Run"].astype(str), meta["Group"].astype(str)))
    mixture_to_label = {mix: lab for lab, mix in groups_cfg.items()}  # "H50Y10E40" -> "S1"
    d = d.copy()
    d["sample_group"] = d["Run"].astype(str).map(run_to_mixture).map(mixture_to_label)
    d["species"] = d["Precursor.Id"].map(sp)

    has_group = d["sample_group"].notna()
    has_species = d["species"].isin(S.SPECIES_ORDER)
    if (~has_group).sum():
        log.warning("mechanism: %s/%s rows had no Run -> sample-group match "
                    "(unknown Run in sample_meta_file).", f"{(~has_group).sum():,}", f"{len(d):,}")
    if (~has_species).sum():
        log.warning("mechanism: %s rows had no/unrecognized species annotation "
                    "(not in %s) and were dropped.", f"{(~has_species).sum():,}", S.SPECIES_ORDER)
    d = d[has_group & has_species]
    if d.empty:
        log.warning("mechanism: no rows had both a sample group and a recognized species "
                    "-- skipping the group/species panels.")
        return None, None
    return d, group_labels


def _recovered_mask(d):
    return d["Final.Status"].astype(str).str.lower().eq("recovered")


def panel_recovered_by_species(classified, group_labels, L, sv):
    """Supp_Mechanism_A: one figure, 3 subplots left-to-right (Human | Yeast | E. coli,
    S.SPECIES_ORDER), grouped bars of totally-evaluated (every recovery-tier row for that
    species, regardless of Final.Status) vs finally-recovered (Final.Status=="recovered"),
    across the 5 sample groups -- the per-species breakdown of Fig_Mechanism_A's
    before/after comparison. Width <7.2in; each species keeps its own y-scale since the 3
    species' counts differ a lot in magnitude."""
    x = np.arange(len(group_labels))
    w = 0.34
    fig, axes = plt.subplots(1, len(S.SPECIES_ORDER), figsize=(7.2, 2.4))
    for ax, sp_key in zip(axes, S.SPECIES_ORDER):
        sub = classified[classified["species"] == sp_key]
        if sub.empty:
            log.warning("mechanism: zero %s rows -- skipping its group panel.", sp_key)
            ax.axis("off")
            continue
        sub_recovered = sub[_recovered_mask(sub)]
        before = sub.groupby("sample_group").size().reindex(group_labels).fillna(0).astype(int)
        after = sub_recovered.groupby("sample_group").size().reindex(group_labels).fillna(0).astype(int)
        retained_pct = [100 * a / b if b else float("nan") for a, b in zip(after.values, before.values)]
        write_tsv(pd.DataFrame({
            "sample_group": group_labels, "n_totally_evaluated": before.values,
            "n_finally_recovered": after.values, "retained_percent": retained_pct,
        }), L.table, f"Supp_Mechanism_A_{sp_key.lower()}_tier_counts.tsv")

        color = S.SPECIES_COLORS[sp_key]
        b1 = ax.bar(x - w / 2, before.values, width=w, color=color,
                   edgecolor=S.NEUTRAL, linewidth=.4, alpha=.45, label="Missing")
        b2 = ax.bar(x + w / 2, after.values, width=w, color=color,
                   edgecolor=S.NEUTRAL, linewidth=.4, label="Recovered")
        S.annotate_bars(ax, b1, [f"{v:,}" for v in before.values], pad_frac=.02, fontsize=ANNOT_FONTSIZE)
        S.annotate_bars(ax, b2, [f"{v:,}\n({p:.1f}%)" for v, p in zip(after.values, retained_pct)],
                        pad_frac=.02, fontsize=ANNOT_FONTSIZE)
        ax.set_xticks(x)
        ax.set_xticklabels(group_labels)
        ax.set_ylim(0, max(int(before.max()), int(after.max()), 1) * 1.32)
        ax.set_title(S.species_label(sp_key), loc="left")
    # one shared legend above the panels (the longer wording would not fit inside each 2.4in panel)
    S.fig_legend(fig, [Patch(facecolor=S.NEUTRAL, edgecolor=S.NEUTRAL, linewidth=.4, alpha=.45),
                       Patch(facecolor=S.NEUTRAL, edgecolor=S.NEUTRAL, linewidth=.4)],
                 ["Light bars: DIA-NN missing", "Dark bars: JUMPlion recovered"], ncol=2, y=1.02,
                 fontsize=ANNOT_FONTSIZE)
    axes[0].set_ylabel("Number of precursor quantities")
    fig.tight_layout(rect=[0, 0, 1, .96])
    sv(fig, L.supp, "Supp_Mechanism_A_by_group")


def panel_recovered_only_by_species(classified, group_labels, L, sv):
    """Related to Supp_Mechanism_A, but final-recovered counts only (no totally-evaluated
    comparison): precursor-run entries per sample group, by species. Yeast and E. coli share one
    main plot (Fig_Mechanism_C) with adjacent bars per group."""
    recovered = classified[_recovered_mask(classified)]
    x = np.arange(len(group_labels))

    w = 0.34
    species_counts, rows = {}, []
    for sp_key in ["YEAST", "ECOLI"]:
        sub = recovered[recovered["species"] == sp_key]
        counts = sub.groupby("sample_group").size().reindex(group_labels).fillna(0).astype(int)
        species_counts[sp_key] = counts
        rows.append(pd.DataFrame({"species": sp_key, "sample_group": group_labels,
                                  "n_finally_recovered": counts.values}))
    write_tsv(pd.concat(rows, ignore_index=True), L.table, "Fig_Mechanism_C_yeast_ecoli_by_group.tsv")

    fig, ax = plt.subplots(figsize=(2.4, 2.1))
    for off, sp_key in zip((-w / 2, w / 2), ["YEAST", "ECOLI"]):
        counts = species_counts[sp_key]
        # alpha/edge/linewidth match S.violin_box's violin-body styling (panel D), so the two
        # panels read as the same species color scheme rather than a solid vs. translucent one.
        ax.bar(x + off, counts.values, width=w, color=S.SPECIES_COLORS[sp_key], alpha=0.85,
              edgecolor=S.NEUTRAL, linewidth=.5, label=S.species_label(sp_key))
    # Not S.annotate_bars here -- it places each bar's label independently, so when Yeast and
    # E. coli land at similar heights within a group (e.g. S5) their labels collide since the two
    # bars sit edge-to-edge. Enforce a minimum vertical gap between the two labels in a group,
    # pushing the taller bar's label further up when they're too close instead.
    ymax_c = max(int(species_counts["YEAST"].max()), int(species_counts["ECOLI"].max()), 1)
    pad, min_gap = .02 * ymax_c, .09 * ymax_c
    for i in range(len(group_labels)):
        yv, ev = int(species_counts["YEAST"].iloc[i]), int(species_counts["ECOLI"].iloc[i])
        y_label_y, e_label_y = yv + pad, ev + pad
        if abs(yv - ev) < min_gap:
            if yv >= ev:
                y_label_y = e_label_y + min_gap  # yeast is taller -- push its label further up
            else:
                e_label_y = y_label_y + min_gap  # e. coli is taller -- push its label further up
        ax.text(x[i] - w / 2, y_label_y, f"{yv:,}", ha="center", va="bottom", fontsize=ANNOT_FONTSIZE)
        ax.text(x[i] + w / 2, e_label_y, f"{ev:,}", ha="center", va="bottom", fontsize=ANNOT_FONTSIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.set_xlabel("Sample group")
    ax.set_ylabel("Number of precursor quantities\n(JUMPlion recovered)")
    ax.set_ylim(0, max(int(species_counts["YEAST"].max()), int(species_counts["ECOLI"].max()), 1) * 1.2)
    ax.legend(frameon=False, fontsize=ANNOT_FONTSIZE, loc="upper right")
    ax.set_title('', loc="center", fontsize=6)
    fig.tight_layout()
    # left/right shared with a (COL1_LEFT/RIGHT); bottom shared with d (ROW2_BOTTOM) -- see those
    # constants above. top=0.88: unlike a/b/e, this panel has no legend/title eating into its top
    # margin, so its y-axis label reaches close to the canvas top and collides with this panel's
    # bold letter label ("d") in the composite -- reserve a little headroom for it.
    fig.subplots_adjust(left=COL1_LEFT+0.05, right=COL1_RIGHT, bottom=ROW2_BOTTOM, top=0.95)
    sv(fig, L.main, "Fig_Mechanism_C_yeast_ecoli_by_group", tight=False)


def panel_recovered_signal_rank_by_group(classified, group_labels, L, sv, max_pts):
    """Main D: Yeast and E. coli final-recovered MS2 XIC signal rank by sample group, as two
    sharey subplots in ONE figure -- figsize/left/right/wspace/bottom match panel B's own 4.4in,
    2-subplot layout exactly (COL2_B_LEFT/RIGHT/WSPACE, ROW2_BOTTOM), so this panel's column
    lines up with B's above it."""
    recovered = classified[_recovered_mask(classified)]
    value_col, ylabel = "ms2_xics_signal_rank", "Relative MS2 intensity"
    fig, axes = plt.subplots(1, 2, figsize=(4.4, 2.1), sharey=True)
    rows = []
    for ax, sp_key in zip(axes, ["YEAST", "ECOLI"]):
        sub = recovered[recovered["species"] == sp_key].dropna(subset=[value_col])
        if sub.empty:
            log.warning("mechanism: zero final-recovered %s rows with %s -- skipping "
                        "%s's by-group plot.", sp_key, value_col, sp_key)
            continue
        groups = [sub.loc[sub["sample_group"] == g, value_col].to_numpy() for g in group_labels]
        rows.append(pd.DataFrame({
            "species": sp_key, "sample_group": group_labels, "n": [len(v) for v in groups],
            "median": [np.median(v) if len(v) else np.nan for v in groups],
            "q25": [np.quantile(v, .25) if len(v) else np.nan for v in groups],
            "q75": [np.quantile(v, .75) if len(v) else np.nan for v in groups],
        }))
        S.violin_box(ax, groups, [S.SPECIES_COLORS[sp_key]] * len(group_labels), max_kde_points=max_pts)
        ax.set_xticklabels(group_labels)
        ax.set_xlabel("Sample group")
        n_plotted = sum(len(v) for v in groups)  # entries actually drawn in this subplot
        ax.set_title(f"{S.species_label(sp_key)} (n = {n_plotted:,})", loc="center", fontsize=6)
    axes[1].tick_params(labelleft=False)  # sharey already links the scale; avoid duplicate labels
    axes[0].set_ylabel(ylabel)
    if rows:
        write_tsv(pd.concat(rows, ignore_index=True), L.table, "Fig_Mechanism_D_signal_rank_by_group.tsv")
    fig.subplots_adjust(left=COL2_B_LEFT+0.02, right=COL2_B_RIGHT, wspace=0.08, bottom=ROW2_BOTTOM, top=0.88)
    sv(fig, L.main, "Fig_Mechanism_D_signal_rank_by_group", tight=False)


def _tier_counts_table(before, after):
    return pd.DataFrame({
        "tier": PANEL_A_TIERS, "label": [S.tier_label(t) for t in PANEL_A_TIERS],
        "n_before_intensity_mapping": before.values, "n_after_intensity_mapping": after.values,
        "retained_fraction": [(a / b if b else float("nan")) for a, b in zip(after.values, before.values)],
    })


def _second_subplot_left_frac(left, right, wspace):
    """Where a 2-subplot figure's right subplot starts, as a fraction of the whole figure width
    -- derived from the same left/right/wspace passed to fig.subplots_adjust(), so this stays
    correct if those margins change, instead of a separately hand-tuned label position."""
    gap = wspace * (right - left) / (2 + wspace)
    return (left + right) / 2 + gap / 2


def assemble_main_figure(L, saver, b_stem=None):
    """Stitch Main A/B/C/D into one composite laid out as a shared 2x2 grid -- top: A (tier
    counts before/after intensity mapping) | B (signal-rank + co-elution violins); bottom: C
    (Yeast/E. coli finally-recovered counts by group) | D (Yeast/E. coli signal-rank-by-group
    sharey violins) -- so A sits directly above C and B above D, column-aligned. Column widths
    (max of A/C, max of B/D) and row heights (max of A/B, max of C/D) are shared across the grid,
    with ROW_GAP inches of blank space between the two rows and TOP_GAP above the first row; a
    panel smaller than its cell is centered within it. Requires "png" in the configured output
    formats.

    Bold letter labels: A (tier counts), B/C (panel B's two subplots -- signal rank and
    co-elution are separate measurements, so each gets its own letter), D (panel C, yeast/ecoli
    counts), E (panel D -- Yeast/E. coli share one y-axis and read as a single panel, so they get
    one letter, not two).

    Each panel's physical size is read back from its own saved PNG (pixel shape / saver.dpi)
    rather than a hand-maintained size table, so any panel's figsize can change without a
    matching edit here."""
    files = {**MAIN_PANEL_FILES_COMBINED, **({"B": b_stem} if b_stem else {})}
    paths = {k: L.main / f"{stem}.png" for k, stem in files.items()}
    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        log.warning("assemble_main_figure skipped: missing panel PNG(s) for %s "
                    "(is \"png\" in the configured style.formats?)", missing)
        return

    imgs = {k: plt.imread(p) for k, p in paths.items()}
    sizes = {k: (img.shape[1] / saver.dpi, img.shape[0] / saver.dpi) for k, img in imgs.items()}

    grid = [["A", "B"], ["C", "D"]]
    col_w = [max(sizes[grid[r][c]][0] for r in range(2)) for c in range(2)]
    row_h = [max(sizes[k][1] for k in row) for row in grid]
    total_w, total_h = sum(col_w), sum(row_h) + ROW_GAP + TOP_GAP

    positions = {}
    row_y0 = []  # each row's un-centered top y, for labels -- see below
    y_top = TOP_GAP
    for ri, (row, rh) in enumerate(zip(grid, row_h)):
        row_y0.append(y_top)
        x_left = 0.0
        for k, cw in zip(row, col_w):
            w, h = sizes[k]
            positions[k] = (x_left + (cw - w) / 2, y_top + (rh - h) / 2, w, h)  # centered in its cell
            x_left += cw
        y_top += rh + (ROW_GAP if ri == 0 else 0)

    fig = plt.figure(figsize=(total_w, total_h))
    for k, (x, y, w, h) in positions.items():
        ax = fig.add_axes([x / total_w, 1 - (y + h) / total_h, w / total_w, h / total_h])
        ax.imshow(imgs[k])
        ax.axis("off")

    def _label(x_in, y_in, text):
        fig.text(x_in / total_w, 1 - y_in / total_h, text, fontsize=11, fontweight="bold",
                 ha="left", va="top")

    # a/b/c share row_y0[0] and d/e share row_y0[1] (not each panel's own, vertically-centered
    # `positions[k]` y) so same-row labels sit at exactly the same height regardless of any one
    # panel being shorter than its row and centered within it.
    y_row1, y_row2 = row_y0[0] + LABEL_PAD, row_y0[1] + LABEL_PAD

    xA, _, _, _ = positions["A"]
    _label(xA + LABEL_PAD, y_row1, "a")

    xB, _, wB, _ = positions["B"]
    _label(xB + LABEL_PAD, y_row1, "b")
    # b2_frac marks subplot2's axes SPINE, but its y-tick numbers (e.g. "6") are drawn right up
    # against that spine, just outside it -- LABEL_PAD alone isn't enough clearance, so push "c"
    # further in with a bigger, dedicated offset instead.
    b2_frac = _second_subplot_left_frac(COL2_B_LEFT, COL2_B_RIGHT, COL2_B_WSPACE)
    _label(xB + b2_frac * wB - 0.5, y_row1, "c")

    xC, _, _, _ = positions["C"]
    _label(xC + LABEL_PAD - 0.02, y_row2, "d")

    xD, _, _, _ = positions["D"]
    _label(xD + LABEL_PAD, y_row2, "e")

    saver.save(fig, L.main, "Fig_Mechanism_combined")


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
def run(cfg, L, saver, resolve):
    sv = saver.save
    max_pts = int(cfg.get("max_violin_kde_points", 200_000))

    # ---- shared loading: read the precursor-run features/quantification parquet ONCE ----
    ev_path = resolve(cfg.get("evidence_table"))
    qp_path = resolve(cfg.get("quantification_parquet"))
    if ev_path is None:
        raise KeyError("mechanism.evidence_table is required.")
    if qp_path is None:
        raise KeyError("mechanism.quantification_parquet is required.")
    wanted = sorted(set([TIER_COL, "evidence_tier", "Final.Status"] + ALL_FEATURES + OPTIONAL_COLS
                        + PRECURSOR_COLS + RUN_COLS + REQ_PARQUET))
    log.info("Loading precursor-run features: %s", ev_path)
    if ev_path == qp_path:
        raw = read_table(ev_path, columns=wanted)
        raw_qp = raw
    else:
        log.warning("mechanism.evidence_table and quantification_parquet differ (%s vs %s); "
                    "reading the features table twice instead of once.", ev_path, qp_path)
        raw = read_table(ev_path, columns=wanted)
        raw_qp = read_table(qp_path, columns=wanted)

    evidence = _tier_filtered(raw)
    counts = evidence[TIER_COL].value_counts().reindex(TIERS).fillna(0).astype(int)
    log.info("Evidence tiers | %s", " | ".join(f"{t}={counts[t]:,}" for t in TIERS))
    universe = validate_precursor_run_universe(evidence)
    write_tsv(pd.DataFrame([universe]), L.table, "precursor_run_universe_check.tsv")

    validate(raw_qp, REQ_PARQUET, "quantification parquet")
    q = raw_qp[REQ_PARQUET].copy()
    q[RUN] = q[RUN].astype(str)

    sp_path = resolve(cfg.get("species_annotation"))
    sp = _load_species_annotation(sp_path) if sp_path is not None else None

    # ---- combined workflow (Supp) ----
    panel_combined_workflow(L, sv, saver.dpi)

    # ---- signal-recovery mechanism (Supp) ----
    before_counts_df = panel_tier_frequencies(evidence, L, sv, universe)

    hist_path, rep_path = _ensure_rt_cross_validation(cfg, resolve)
    hist = read_table(hist_path)
    rep = read_table(rep_path)
    panel_pre_extraction_rt_validation(hist, rep, cfg.get("rt_extraction_half_window"), L, sv,
                                       cycle_time=cfg.get("cycle_time"))
    panel_rt_refinement_by_rt(evidence, L, sv)
    rt_cv = read_table(resolve(cfg.get("rt_cv")))
    if rt_cv is not None and cfg.get("rt_half_window_points") is not None:
        panel_rt_localization_legacy(rt_cv, float(cfg["rt_half_window_points"]), cfg.get("cycle_time"), L, sv)
    panel_im(read_table(resolve(cfg.get("im_cv"))), cfg.get("im_window"),
             read_table(resolve(cfg.get("im_window_table"))), L, sv)

    # XIC examples last (slowest step: scans per-run parquet files)
    xic_dir = resolve(cfg.get("xic_dir"))
    if xic_dir is not None:
        log.info("Generating XIC examples from %s", xic_dir)
        make_xic_panels(xic_dir, cfg.get("xic_examples") or {}, cfg.get("xic_auto_selection", "median"), L, sv)

    # ---- intensity-mapping mechanism (Supp) ----
    # The LOO reliability analysis (_run_mapping_reliability) is called separately, after the Main
    # final-recovered section below, so a LOO failure there can't prevent Main from being built.
    after_counts_df, comp = _run_intensity_mapping(cfg, L, sv, resolve, q, sp)

    # ---- final recovered (Main) ----
    if after_counts_df.empty or after_counts_df["N.Recovered.PrecursorRuns"].sum() == 0:
        log.warning("mechanism: %s has zero total recovered precursor-runs -- the 'after "
                    "intensity mapping' bars will all be 0. If that's unexpected, check "
                    "the intensity-mapping section's Final.Status values upstream.",
                    "Supp_Mechanism2_F0_final_recovery_by_localization_tier.tsv")
    panel_before_after_tiers(before_counts_df, after_counts_df, L, sv)

    recovered = _recovered_evidence_view(evidence)
    # Fig_Mechanism_B is drawn twice in the main folder: _recovered (the finally recovered
    # precursor-runs of each recovery tier) and _all (every JUMPlion precursor-run entry of each
    # recovery tier). mechanism.signal_rank_population ("recovered", the default, or "all") picks
    # which one goes into the integrated figure. Panel D below always uses the recovered subset.
    b_choice = str(cfg.get("signal_rank_population", "recovered")).lower()
    if b_choice not in ("recovered", "all"):
        raise ValueError(f"mechanism.signal_rank_population must be 'recovered' or 'all', got {b_choice!r}")
    b_stems = {}
    for population in ("recovered", "all"):
        b_view = _recovered_evidence_view(evidence, recovered_only=population == "recovered")
        if b_view is not None and len(b_view):
            b_stems[population] = f"Fig_Mechanism_B_signal_rank_coelution_{population}"
            panel_recovered_distributions(b_view, L, sv, max_pts, stem=b_stems[population])
    if recovered is not None and len(recovered):
        classified_recovered, recovered_group_labels = classify_group_species(recovered, cfg, resolve, sp)
        if classified_recovered is not None:
            panel_recovered_signal_rank_by_group(classified_recovered, recovered_group_labels, L, sv, max_pts)
    elif recovered is not None:
        log.warning("mechanism: zero rows matched 'reported' or a recovery tier with "
                    "Final.Status=='recovered' -- skipping the distribution panels.")

    all_tier_rows = _recovery_tier_rows_view(evidence)
    if all_tier_rows is not None and len(all_tier_rows):
        classified, group_labels = classify_group_species(all_tier_rows, cfg, resolve, sp)
        if classified is not None:
            panel_recovered_by_species(classified, group_labels, L, sv)
            panel_recovered_only_by_species(classified, group_labels, L, sv)
    elif all_tier_rows is not None:
        log.warning("mechanism: zero recovery-tier rows -- skipping the group/species panels.")

    assemble_main_figure(L, saver, b_stem=b_stems.get(b_choice))

    # LOO reliability analysis + Supp_Mechanism2_C_threshold_sensitivity_and_robustness, last:
    # everything above (including the Main panels) is already written by the time this runs, so a
    # LOO/production-fit failure here doesn't take the rest of the mechanism stage down with it.
    _run_mapping_reliability(cfg, L, sv, resolve, comp)

    log.info("mechanism finished.")
