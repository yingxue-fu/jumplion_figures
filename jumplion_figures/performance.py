"""Performance benchmark on the Wang HYE 5-mixture design (DIA-NN vs JUMPlion).

Organized around DIA-NN's own completeness, contrast by contrast, BEFORE JUMPlion:
    Complete            6/6 DIA-NN observations in both groups.
    Partially complete  >=3/6 in both groups (DIA-NN already DE-eligible), but <6/6 in >=1 group.
    Insufficient        <3/6 in >=1 group (DIA-NN cannot support the predefined DE analysis).
These three states -- not "shared" vs "newly DE-eligible" -- are the primary organizing lens:
DE-eligible (>=3/6 both groups) remains the operational rule the DE pipeline itself uses, but it
is Complete + Partially complete, not a population in its own right. Within Insufficient, JUMPlion
may push a precursor's own n_obs to >=3/6 in both groups ("restored" / "becomes sufficient") or
not ("remains insufficient").

Key definitions
---------------
* Precursors mapped to more than one organism are excluded before ANY analysis.
* Detection = adjusted P < padj_cutoff plus the known species direction
  (Human invariant -> FP/TN; Yeast up; E. coli down). No fold-change cutoff here in the full
  run() pipeline (assign_benchmark_state's own `log2fc_cutoff` stays None); the standalone
  `generalization` stage (not this module's own run()) does apply one -- see its own run().
* Completeness is computed from DIA-NN's own n_obs (pre-DE table), independent of whether
  JUMPlion actually filled anything in. Primary abundance reference = JUMPlion S1 mean log2.
"""
from __future__ import annotations

import logging
import math
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

from . import style as S
from .common import write_tsv

log = logging.getLogger("jumplion_figures.performance")

PADJ_CUT = 0.05
# The matrix row identifier column -- "Precursor.Id" for the normal precursor-level pipeline, but
# overridable (see generalization.run()'s own P.ID_COL line) so prepare_pair/load_pairs_and_prepped/
# compute_coverage_curve/compute_completeness_by_method -- the subset of this module's functions
# `generalization` actually calls -- can instead run on a protein-level matrix (row id
# "Protein.Group") without duplicating that logic. Same reset-per-stage pattern as PADJ_CUT above:
# run() resets this to the precursor-level default at its own top, so a prior generalization() call
# earlier in the same process can never leak its own override into the full pipeline.
ID_COL = "Precursor.Id"
REF = "H50Y10E40"
# Intensity bins are quantile-based: one global set of 20th/40th/60th/80th percentile cutpoints
# (see quantile_bin_edges() in run()), computed once from the pooled common-universe intensity
# distribution and reused unchanged everywhere, rather than a fixed absolute log2 value or a
# value recomputed per contrast/population.
BIN_LABELS = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
INT_XLABEL = "JUMPlion S1 mean log2 intensity"
M_D, M_J = S.METHOD_ORDER
METHOD_SUFFIX = [(M_D, "D"), (M_J, "J")]

# Benchmark design. Defaults = Wang HYE 5-mixture; overwritten from the config's
# "benchmark" section by configure_benchmark() so nothing here is hard-coded downstream.
GROUPS = {"S1": "H50Y10E40", "S2": "H50Y20E30", "S3": "H50Y25E25", "S4": "H50Y30E20", "S5": "H50Y40E10"}
REF_LABEL = "S1"
MIX = {}        # label -> (human, yeast, ecoli) percentages
EXPECTED = {}   # "S2_vs_S1" -> {"HUMAN": log2FC, "YEAST": ..., "ECOLI": ...}


def parse_mix(group_name):
    m = re.fullmatch(r"H(\d+(?:\.\d+)?)Y(\d+(?:\.\d+)?)E(\d+(?:\.\d+)?)", str(group_name))
    if not m:
        raise ValueError(f"Group name '{group_name}' is not of the form H<h>Y<y>E<e> (e.g. H50Y10E40).")
    return tuple(float(v) for v in m.groups())


def configure_benchmark(groups=None, reference="S1", contrasts=None):
    """Derive mixture composition and expected log2FC from group names -- either for every
    non-reference group vs one fixed `reference` (star topology, the original/default design:
    EXPECTED["label_vs_reference"]), or for an explicit `contrasts` list of (label1, label2) pairs
    (e.g. a path -- A_vs_B, B_vs_C -- where no single group is the hub). REF/REF_LABEL (the fixed
    reference group, used by run()'s other, reference-only panels like the pooled intensity-bin
    setup) are only meaningful, and only set, in reference mode; `contrasts` mode leaves them at
    whatever they were (unused by anything `contrasts` mode actually calls)."""
    global GROUPS, REF_LABEL, REF, MIX, EXPECTED
    GROUPS = dict(groups or GROUPS)
    MIX = {lab: parse_mix(g) for lab, g in GROUPS.items()}
    EXPECTED = {}
    if contrasts:
        for g1, g2 in contrasts:
            h1, y1, e1 = MIX[g1]
            h2, y2, e2 = MIX[g2]
            EXPECTED[f"{g1}_vs_{g2}"] = {"HUMAN": math.log2(h1 / h2), "YEAST": math.log2(y1 / y2),
                                         "ECOLI": math.log2(e1 / e2)}
    else:
        REF_LABEL = reference
        REF = GROUPS[reference]
        h0, y0, e0 = MIX[reference]
        for lab, (h, y, e) in MIX.items():
            if lab == reference:
                continue
            EXPECTED[f"{lab}_vs_{reference}"] = {"HUMAN": math.log2(h / h0), "YEAST": math.log2(y / y0),
                                                 "ECOLI": math.log2(e / e0)}


configure_benchmark()
CLEAN = ["HUMAN", "YEAST", "ECOLI"]

# DIA-NN completeness states (per contrast, from DIA-NN alone, BEFORE JUMPlion). GROUP_SIZE and
# MIN_OBS_ELIGIBLE mirror the DE pipeline's own missingness rule (max_missing_pct=0.5 of 6 reps).
GROUP_SIZE = 6
MIN_OBS_ELIGIBLE = 3
COMPLETE = "Complete"
PARTIAL = "Partially complete"
INSUFFICIENT = "Insufficient"
COMPLETENESS_ORDER = [COMPLETE, PARTIAL, INSUFFICIENT]
REMAINS_INSUFFICIENT = "Remains insufficient"
RESTORED = "Becomes sufficient"

# Shading per completeness state -- a method's own color (DIA-NN blue / JUMPlion red) at
# this alpha, used both by panel_completeness_by_method's stacked bars and (for PARTIAL) by
# _panel_box_grid_by_dataset's "Partially complete" boxes, so the two kinds of panel agree on what
# "partial" looks like.
STATE_ALPHA = {COMPLETE: 1.0, PARTIAL: 0.6, INSUFFICIENT: 0.3}

# Minimum-denominator reporting rules: a rate computed from too few observations is unstable
# (e.g. one FP among 3 discoveries swings FDR enormously), so it is reported as NaN instead of
# a misleading number -- gaps in a line plot, not a bar pinned at 0% or 100%.
MIN_N_SENSITIVITY = 20     # min ground-truth differential precursors (TP+FN+wrong_direction)
MIN_N_DISCOVERIES = 20     # min total discoveries (TP+FP) to report empirical (raw) FDR
MIN_N_SPECIES_BALANCED = 10  # min per-species count (human/yeast/ecoli) to include that
                             # species' rate in the species-balanced FDR combination

# Standalone benchmark-design panel (Supp_Performance_A_benchmark_design) size. The other main
# panels get their figsize from their own aw/ah ... ew/eh constants below.
PANEL_A_SIZE = (1.70, 1.60)
# Samples in the reference design panel (hye_wang: S1-S5); generalization's design panel matches its bar width.
DESIGN_REF_N_GROUPS = 5
PANEL_LABEL_SIZE = 6  # tick/axis-label size for the shrunken main panels
# Main A (30-run cumulative coverage) and Main B (completeness-by-method) share this one canvas
# size, for the Fig_Performance_Integrated composite below -- not the individual heatmaps. Fixed
# size (not derived from PANEL_A_SIZE): shorter than C/D's 3.00in column height, so it's
# centered vertically against them in that composite rather than filling the column exactly.
aw, ah = 1.8, 2.25
bw, bh = 1.8, 2.25
cw, ch = 1.8, 2.25
dw, dh = 2.7, 2.925
ew, eh = 2.7, 2.925
# D and E (row 2 of Fig_Performance_Integrated) share these same top/bottom/hspace fractions for
# their own 2-row sharex grids (partially-complete row on top, restored row below) -- named
# constants, not each panel's own inline subplots_adjust literal, so assemble_integrated_figure can
# derive exactly where each one's own row 1 (restored) starts, for its own "f"/"g" sub-panel label,
# using the same GridSpec math matplotlib itself uses to split rows.
ROW2_TOP, ROW2_BOTTOM, ROW2_HSPACE = .90, .12, .65
# Transition heatmaps (Supp) -- width matches the old combined-B-panel subfigure width
# (width_ratios=[2, 1.5, 2] of the old 5.5in panel), 2.0in tall.
PANEL_B0_SIZE = (2.4, 2.0)  # transition heatmaps
# Fig_Performance_A_coverage and Fig_Performance_B_completeness_by_method are the same
# PANEL_AB_SIZE canvas, side by side as row 1 of Fig_Performance_Integrated -- sharing
# left/right lines up their axes boxes horizontally, and sharing bottom puts their x-axis (the
# actual axis line, where the tick labels sit) at the same height, instead of each panel's own
# previous bottom margin (.14 vs .20) leaving the two x-axes at different vertical positions.
INTEGRATED_LEFT, INTEGRATED_RIGHT, INTEGRATED_BOTTOM = 0.24, 0.98, 0.1
# panel_sensitivity_fdr_bar's own axes geometry (_axes_geometry/_panel_d_size/PANEL_D_INNER_GAP/
# PANEL_D_SIZE) lives in generalization.py now -- it's the only caller, along with
# panel_completeness_complete_only/panel_sensitivity_fdr_bar/panel_log2fc_error_cv_box_grid/
# panel_coverage_grid themselves, which moved there with it.


def benchmark_design_table(fig, rect, contrasts):
    """Expected Yeast / E. coli log2FC per contrast (from the mixture design, EXPECTED) as a text
    table in `rect` (figure fractions); shared by Supp_Performance_A/Supp_Generalization benchmark
    design panels."""
    axt = fig.add_axes(rect)
    axt.axis("off"); axt.set_xlim(0, 1); axt.set_ylim(0, 1)
    signed = lambda v: f"{v:+.2f}".replace("-", "−")
    cols_x = [0.0, 0.50, 0.90]
    cols_ha = ["left", "right", "right"]
    header = ["Contrast", "Yeast log2FC", "E. coli log2FC"]
    n_rows = len(contrasts) + 1
    ys = np.linspace(1, 0, n_rows + 1)[:-1] - .5 / n_rows
    for x_, ha_, txt in zip(cols_x, cols_ha, header):
        axt.text(x_, ys[0], txt, ha=ha_, va="center", fontsize=5, fontweight="bold")
    axt.plot([0, .92], [1 - 1 / n_rows] * 2, color="black", lw=.5, clip_on=False)
    for y_, c in zip(ys[1:], contrasts):
        cells = [S.clabel(c), signed(EXPECTED[c]["YEAST"]), signed(EXPECTED[c]["ECOLI"])]
        for x_, ha_, txt in zip(cols_x, cols_ha, cells):
            axt.text(x_, y_, txt, ha=ha_, va="center", fontsize=5)


def compact_axes(*axes, labelsize=PANEL_LABEL_SIZE):
    """Shrink tick and axis-label text to suit the narrow tiled main panels."""
    for ax in axes:
        ax.tick_params(labelsize=labelsize)
        ax.xaxis.label.set_size(labelsize + 0.5)
        ax.yaxis.label.set_size(labelsize + 0.5)


def classify_completeness(n1, n2):
    """DIA-NN completeness state from each group's observed-replicate count (array-like in, array out)."""
    n1 = np.asarray(n1, dtype=float)
    n2 = np.asarray(n2, dtype=float)
    full = (n1 >= GROUP_SIZE) & (n2 >= GROUP_SIZE)
    eligible = (n1 >= MIN_OBS_ELIGIBLE) & (n2 >= MIN_OBS_ELIGIBLE)
    return np.select([full, eligible], [COMPLETE, PARTIAL], default=INSUFFICIENT)


def quantile_bin_edges(values):
    """20/40/60/80th percentile cutpoints of `values`, computed ONCE from the pooled common-universe
    intensity distribution and reused unchanged everywhere, so "intensity bin N" means the same
    absolute log2-intensity range in every analysis/contrast rather than a different range per
    contrast or per population. Returns (edges for pd.cut, display labels showing the actual
    log2-intensity boundaries, e.g. ['<8.8', '8.8-9.5', '9.5-10.2', '10.2-11.0', '>=11.0'])."""
    v = pd.to_numeric(values, errors="coerce")
    _, edges = pd.qcut(v, q=[0, .2, .4, .6, .8, 1.0], retbins=True)
    cuts = edges[1:-1]
    display_labels = ([f"<{cuts[0]:.1f}"] +
                      [f"{cuts[i]:.1f}-{cuts[i + 1]:.1f}" for i in range(len(cuts) - 1)] +
                      [f"≥{cuts[-1]:.1f}"])
    return [-np.inf, *cuts, np.inf], display_labels


# ----------------------------------------------------------------------------
# Core helpers
# ----------------------------------------------------------------------------
def norm_species(x):
    s = str(x).upper()
    hits = []
    if "HUMAN" in s: hits.append("HUMAN")
    if "YEAST" in s: hits.append("YEAST")
    if "ECOLI" in s or "E.COLI" in s or "E. COLI" in s: hits.append("ECOLI")
    return hits[0] if len(hits) == 1 else "AMBIGUOUS"


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return center - half, center + half


def assign_benchmark_state(df, log2fc_cutoff=None):
    """`log2fc_cutoff`: if given, a detection additionally requires |log2FC| > this value (not
    just p_adj < PADJ_CUT) -- None (the default) reproduces the original padj-only rule, with no
    fold-change cutoff at all (see this module's own docstring)."""
    species = df["Species_std"]
    sig = df["p_adj"] < PADJ_CUT
    fc = df["log2FC"]
    if log2fc_cutoff is not None:
        sig = sig & (fc.abs() > log2fc_cutoff)
    state = np.full(len(df), "NE", dtype=object)
    human, yeast, ecoli = species.eq("HUMAN"), species.eq("YEAST"), species.eq("ECOLI")
    state[human & ~sig] = "TN"
    state[human & sig] = "FP"
    state[yeast & ~sig] = "FN"
    state[yeast & sig & (fc > 0)] = "TP"
    state[yeast & sig & (fc <= 0)] = "wrong_direction"
    state[ecoli & ~sig] = "FN"
    state[ecoli & sig & (fc < 0)] = "TP"
    state[ecoli & sig & (fc >= 0)] = "wrong_direction"
    return pd.Series(state, index=df.index)


def benchmark_state_one(row):
    sp = norm_species(row.get("Species", ""))
    sig = pd.notna(row.get("p_adj", np.nan)) and row["p_adj"] < PADJ_CUT
    fc = row.get("log2FC", np.nan)
    if sp == "HUMAN":
        return "FP" if sig else "TN"
    if sp == "YEAST":
        return "FN" if not sig else ("TP" if pd.notna(fc) and fc > 0 else "wrong_direction")
    if sp == "ECOLI":
        return "FN" if not sig else ("TP" if pd.notna(fc) and fc < 0 else "wrong_direction")
    return "NE"


def prepare_pair(diann_file, jump_file, contrast, g1, g2, log2fc_cutoff=None):
    """g1/g2: this contrast's two group MIXTURE names (e.g. "H65Y30E05"), not labels -- g1 is the
    comparison side, g2 the reference/denominator side its log2FC/CV/intensity are measured
    against. Generalizes the original fixed-REF design to any (g1, g2) pair, including one where
    different contrasts have different g2 (e.g. a path -- A_vs_B, B_vs_C -- not a star around one
    shared reference). g2 is also stashed on `m` itself (as "ref_group") so later per-contrast
    code (compute_completeness_by_method, compute_cv_deltafc) can read it back off `m` instead of
    relying on a single global reference. `log2fc_cutoff`: passed straight through to
    assign_benchmark_state -- see its own docstring."""
    d = pd.read_csv(diann_file)
    j = pd.read_csv(jump_file)
    d["Species_std"] = d["Species"].map(norm_species)
    j["Species_std"] = j["Species"].map(norm_species)
    # Clean benchmark universe: drop multi-organism precursors before anything else.
    d = d[d["Species_std"].isin(CLEAN)].copy()
    j = j[j["Species_std"].isin(CLEAN)].copy()
    for x in (d, j):
        x["benchmark_state"] = assign_benchmark_state(x, log2fc_cutoff=log2fc_cutoff)
        x["DE_call"] = x["p_adj"] < PADJ_CUT
    m = d.merge(j, on=ID_COL, suffixes=("_D", "_J"))
    m["contrast"] = contrast
    m["other_group"] = g1
    m["ref_group"] = g2
    m["S1_intensity"] = m[f"mean_log2_{g2}_J"]
    m["S1_intensity_diann"] = m[f"mean_log2_{g2}_D"]
    # S1_bin is assigned later in run() once the global, common-universe bin edges are known.
    # DIA-NN completeness state (Complete / Partially complete only -- d is already DE-eligible,
    # i.e. n_obs_D >= 3/6 in both groups, so m never contains Insufficient precursors).
    m["completeness_state"] = classify_completeness(m[f"n_obs_{g2}_D"], m[f"n_obs_{g1}_D"])
    return d, j, m, g1


def load_pairs_and_prepped(cfg, resolve, log2fc_cutoff=None):
    """Load every configured contrast's DE + pre-DE tables, species-filtered to CLEAN organisms.
    Returns (contrasts, prepared, preprepared) -- prepared[name] is prepare_pair()'s own tuple
    (used by most of run()'s panels), preprepared[name] is the (pd0, pj0) species-filtered pre-DE
    pair (used by the coverage curve, the intensity-bin setup, and the exclusion audit). Each
    comparison item must carry "g1"/"g2" (its two group MIXTURE names -- see prepare_pair), in
    addition to the 4 file paths. Shared by the full `run()` pipeline and the standalone
    `generalization` stage, which only needs the latter. `log2fc_cutoff`: passed straight through
    to prepare_pair/assign_benchmark_state -- see assign_benchmark_state's own docstring. run()
    never passes this (always None, the original padj-only rule); only `generalization` does."""
    prepared, preprepared = {}, {}
    ref_pd0_ids = ref_pj0_ids = ref_name = None
    for item in cfg["comparisons"]:
        name = item["name"]
        if name not in EXPECTED:
            raise KeyError(f"Unknown contrast '{name}'. Expected one of {list(EXPECTED)}.")
        paths = {k: resolve(item[k]) for k in ["diann", "jumplion", "diann_pre_de", "jumplion_pre_de"]}
        item["_paths"] = paths
        log.info("Loading %s", name)
        prepared[name] = prepare_pair(paths["diann"], paths["jumplion"], name, item["g1"], item["g2"],
                                      log2fc_cutoff=log2fc_cutoff)
        pd0, pj0 = pd.read_csv(paths["diann_pre_de"]), pd.read_csv(paths["jumplion_pre_de"])
        for xx in (pd0, pj0):
            xx["Species_std"] = xx["Species"].map(norm_species)
            if "DE_eligible" not in xx.columns:
                raise ValueError("pre-DE table must contain DE_eligible")
            xx["DE_eligible"] = xx["DE_eligible"].astype(bool)
        n_pd0_before, n_pj0_before = len(pd0), len(pj0)
        # Every contrast's pre-DE table (per method) is expected to cover the exact same universe
        # of precursor IDs -- they're all derived from the same master precursor matrix, just
        # re-evaluated against a different reference pair. Warn (don't silently diverge) if a
        # later contrast's table doesn't match the first one seen, per method.
        pd0_ids, pj0_ids = set(pd0[ID_COL]), set(pj0[ID_COL])
        same_as_ref = True
        if ref_pd0_ids is None:
            ref_pd0_ids, ref_pj0_ids, ref_name = pd0_ids, pj0_ids, name
        else:
            if pd0_ids != ref_pd0_ids:
                same_as_ref = False
                log.warning("%s: DIA-NN pre-DE precursor set differs from %s's (%s vs %s, %s not "
                            "shared).", name, ref_name, f"{len(pd0_ids):,}", f"{len(ref_pd0_ids):,}",
                            f"{len(pd0_ids ^ ref_pd0_ids):,}")
            if pj0_ids != ref_pj0_ids:
                same_as_ref = False
                log.warning("%s: JUMPlion pre-DE precursor set differs from %s's (%s vs "
                            "%s, %s not shared).", name, ref_name, f"{len(pj0_ids):,}",
                            f"{len(ref_pj0_ids):,}", f"{len(pj0_ids ^ ref_pj0_ids):,}")
        valid_ids = (set(pd0.loc[pd0["Species_std"].isin(CLEAN), ID_COL]) |
                     set(pj0.loc[pj0["Species_std"].isin(CLEAN), ID_COL]))
        pd0 = pd0[pd0[ID_COL].isin(valid_ids) & pd0["Species_std"].isin(CLEAN)].copy()
        pj0 = pj0[pj0[ID_COL].isin(valid_ids) & pj0["Species_std"].isin(CLEAN)].copy()
        # Every contrast's counts are identical when its pre-DE precursor sets matched the
        # reference, so the counts are only worth logging once -- for the reference contrast, or
        # again for any contrast that turned out to actually differ (same_as_ref False above).
        if name == ref_name or not same_as_ref:
            log.info("%s: pre-DE precursors before/after species filtering (keep %s only) -- "
                    "DIA-NN %s -> %s, JUMPlion %s -> %s", name, CLEAN, f"{n_pd0_before:,}",
                    f"{len(pd0):,}", f"{n_pj0_before:,}", f"{len(pj0):,}")
        preprepared[name] = (pd0, pj0)
    return list(prepared), prepared, preprepared


def compute_coverage_curve(contrasts, preprepared):
    """30-run cumulative coverage curve (method-agnostic completeness diagnostic, pooled across
    every group that appears in ANY contrast -- not a fixed reference + each contrast's own
    "other" group, so this works whether the contrasts form a star (one shared reference) or an
    arbitrary pair list (e.g. a path: A_vs_B, B_vs_C, where no single group is common to both in
    the same role). A group appearing in more than one contrast's table (e.g. B in both of the
    above) only needs its n_obs read once -- first occurrence wins. Returns (completeness, curve,
    n_runs, n_total, pad). Shared by the full `run()` pipeline's Main A panel and the standalone
    `generalization` stage."""
    obs = {}
    all_ids = set()
    groups = []
    for method, idx in [(M_D, 0), (M_J, 1)]:
        obs[method] = {}
        for c in contrasts:
            tab = preprepared[c][idx]
            all_ids.update(tab[ID_COL])
            for col in tab.columns:
                if not col.startswith("n_obs_"):
                    continue
                g = col[len("n_obs_"):]
                if g not in obs[method]:
                    obs[method][g] = tab.drop_duplicates(ID_COL).set_index(ID_COL)[col]
                    if g not in groups:
                        groups.append(g)
    ids = sorted(all_ids)
    n_runs = GROUP_SIZE * len(groups)
    comp_frames = []
    for method in S.METHOD_ORDER:
        mat = pd.DataFrame({g: pd.to_numeric(obs[method][g], errors="coerce").reindex(ids) for g in groups})
        nobs = mat.fillna(0).sum(axis=1).clip(0, n_runs)
        comp_frames.append(pd.DataFrame({"Precursor_Id": ids, "method": method, f"n_obs_{n_runs}": nobs.values,
                                         f"n_missing_{n_runs}": (n_runs - nobs).values,
                                         "missing_fraction": ((n_runs - nobs) / n_runs).values}))
    completeness = pd.concat(comp_frames, ignore_index=True)
    thresholds = np.arange(0, n_runs, dtype=float) / n_runs
    n_total = len(ids)
    curve = []
    for method in S.METHOD_ORDER:
        vals = completeness.loc[completeness.method == method, "missing_fraction"].to_numpy(float)
        for t in thresholds:
            n_cum = int((vals <= t + 1e-12).sum())
            curve.append(dict(method=method, missing_fraction=t, cumulative_n=n_cum, n_total=n_total,
                              cumulative_percent=100 * n_cum / n_total if n_total else np.nan))
    curve = pd.DataFrame(curve)
    pad = 0.5 / n_runs
    return completeness, curve, n_runs, n_total, pad


def _draw_coverage(ax, curve, n_runs, n_total, pad, title=None):
    """The coverage-curve axis content itself -- shared by the single-panel panel_coverage
    (run()'s Main A / generalization's own per-dataset figure) and panel_coverage_grid (one
    multi-dataset combined figure)."""
    # Each method's 0-missing-fraction starting value (precursors observed in every one of the
    # n_runs runs) -- used below to decide each annotation's side. The curve that starts HIGHER
    # gets its label placed ABOVE the line, the one that starts LOWER gets its label placed BELOW
    # -- whichever method that happens to be (not a fixed per-method direction, since which curve
    # starts higher can vary by dataset). Either side is safe from this curve's OWN rise (a
    # monotonically non-decreasing cumulative curve never dips back under its own starting value),
    # and the two curves start far enough apart at x=0 that neither label's small offset reaches
    # the other curve.
    starts = {}
    for method in S.METHOD_ORDER:
        zz0 = curve[(curve.method == method) & (curve.missing_fraction == 0)]
        if not zz0.empty:
            starts[method] = (int(zz0.cumulative_n.iloc[0]), float(zz0.cumulative_percent.iloc[0]))
    upper_method = max(starts, key=lambda m: starts[m][1]) if starts else None

    for method in S.METHOD_ORDER:
        zz = curve[curve.method == method]
        ax.plot(zz.missing_fraction * 100, zz.cumulative_percent, marker="o", ms=1.6,
                color=S.method_color(method), label=method)
        if method in starts:
            n0, pct0 = starts[method]
            above = method == upper_method
            ax.annotate(f"{n0:,}\n({pct0:.1f}%)", xy=(0, pct0), xytext=(1, 6 if above else -4),
                       textcoords="offset points", ha="left", va="bottom" if above else "top",
                       fontsize=4.2, color=S.method_color(method))
    ax.set_xlabel("Missing percentage (%)")
    ax.set_ylabel(f"{unit_label()} (%)")
    ax.set_xlim((-pad) * 100, ((n_runs - 1) / n_runs + pad) * 100)
    ax.set_xticks([0, 25, 50, 75])
    ax.set_ylim(0, 105)
    if title is not None:
        ax.set_title(title, fontsize=5.5)
    else:
        # One-row total just above the axes (panel_coverage leaves room for it under the legend);
        # panel_coverage_grid passes its own per-dataset `title` instead.
        ax.text(0.5, 1.02, f"Total {unit_label().lower()}: n={n_total:,}", transform=ax.transAxes,
               ha="center", va="bottom", fontsize=5.5)
    compact_axes(ax)


def panel_coverage(curve, n_runs, n_total, pad, MAIN, sv, stem="Fig_Performance_A_coverage"):
    """The coverage-curve plot itself -- same figure for the full `run()` pipeline's Main A and
    the standalone `coverage` stage."""
    fig, axl2 = plt.subplots(figsize=(aw, ah))
    _draw_coverage(axl2, curve, n_runs, n_total, pad)
    # Legend outside the plot (above the axes) rather than the small in-axes "lower right" one --
    # B's fixed 2.5x2.5 size leaves enough top margin for it now.
    S.fig_legend(fig, *axl2.get_legend_handles_labels(), ncol=2, y=1.0, fontsize=6,
                 handlelength=1.2, columnspacing=1.2)
    fig.subplots_adjust(left=INTEGRATED_LEFT, right=INTEGRATED_RIGHT, top=.80, bottom=INTEGRATED_BOTTOM+0.06)
    # tight=False: see the matching note on Fig_Performance_B_completeness_by_method's sv() call
    # -- both must keep their exact figsize canvas (not cropped to their own content bbox) for
    # INTEGRATED_LEFT/RIGHT/BOTTOM to actually align the two panels in Fig_Performance_Integrated.
    sv(fig, MAIN, stem, tight=False)


def compute_completeness_by_method(contrasts, prepared, preprepared):
    """DIA-NN completeness state per contrast (before JUMPlion) and JUMPlion's own completeness
    state after, needed by panel_completeness_by_method; also the RESTORED ("becomes sufficient")
    subset needed by compute_sensitivity_fdr/panel_sensitivity_fdr's JUMPlion-recovery row. Lean
    subset of the full run() pipeline's "DIA-NN completeness universe" block -- the same n_obs-
    derived completeness_state/_J and restored-precursor merge, just without the intensity-bin
    columns (bin/I_ref) run() also attaches there for its other (intensity-bin) panels, since
    neither of these two panels needs them.
    Returns (coverage, completeness_transition, restored)."""
    cov_rows, u_list, restored_rows = [], [], []
    for c, (d, j, m, other) in prepared.items():
        pd0, pj0 = preprepared[c]
        ref = m["ref_group"].iloc[0]  # this contrast's own reference/denominator side (see prepare_pair)
        dd = pd0[[ID_COL, f"n_obs_{ref}", f"n_obs_{other}"]].rename(
            columns={f"n_obs_{ref}": "n1", f"n_obs_{other}": "n2"})
        jj = pj0[[ID_COL, "DE_eligible", f"n_obs_{ref}", f"n_obs_{other}"]].rename(
            columns={"DE_eligible": "testable_J", f"n_obs_{ref}": "n1_J", f"n_obs_{other}": "n2_J"})
        u = dd.merge(jj, on=ID_COL, how="outer")
        u["n1"] = u["n1"].fillna(0)
        u["n2"] = u["n2"].fillna(0)
        u["n1_J"] = u["n1_J"].fillna(0)
        u["n2_J"] = u["n2_J"].fillna(0)
        u["testable_J"] = u["testable_J"].fillna(False).astype(bool)
        u["completeness_state"] = classify_completeness(u["n1"], u["n2"])
        u["completeness_state_J"] = classify_completeness(u["n1_J"], u["n2_J"])
        u["contrast"] = c
        u_list.append(u)
        for state in COMPLETENESS_ORDER:
            cov_rows.append(dict(contrast=c, state=state, n=int((u.completeness_state == state).sum())))
        ins = u[u.completeness_state == INSUFFICIENT].copy()
        ins["restoration"] = np.where(ins.testable_J, RESTORED, REMAINS_INSUFFICIENT)
        rec = (ins.loc[ins.restoration == RESTORED, ["contrast", ID_COL, "n1", "n2"]]
               .merge(j[[ID_COL, "benchmark_state", "log2FC", "p_adj", "Species_std",
                        f"cv_{ref}", f"cv_{other}"]].rename(
                   columns={f"cv_{ref}": "cv_ref", f"cv_{other}": "cv_comparison"}),
                      on=ID_COL, how="left"))
        restored_rows.append(rec)
    u_all = pd.concat(u_list, ignore_index=True)
    coverage = pd.DataFrame(cov_rows)
    transition_rows = []
    for c in contrasts:
        zz = u_all[u_all.contrast == c]
        for base in COMPLETENESS_ORDER:
            g = zz[zz.completeness_state == base]
            n = len(g)
            for after in COMPLETENESS_ORDER:
                k = int((g.completeness_state_J == after).sum())
                transition_rows.append(dict(contrast=c, baseline_state=base, state_after_jumplion=after,
                                            n=k, pct=100 * k / n if n else np.nan))
    completeness_transition = pd.DataFrame(transition_rows)
    restored = pd.concat(restored_rows, ignore_index=True)
    return coverage, completeness_transition, restored


def _coverage_by_method(coverage, completeness_transition):
    """{method: per-contrast per-state counts}, DIA-NN's straight from `coverage`, JUMPlion's own
    completeness_state_J distribution derived from `completeness_transition` (summed over all
    baseline states, each contrast's state_after_jumplion counts partition all precursors) since
    it isn't computed anywhere else. Shared by panel_completeness_by_method and
    panel_completeness_complete_only."""
    coverage_j = (completeness_transition.groupby(["contrast", "state_after_jumplion"])["n"].sum()
                 .reset_index().rename(columns={"state_after_jumplion": "state"}))
    return {M_D: coverage, M_J: coverage_j}


def completeness_table_legend(fig, rect):
    """Table-style completeness legend (see panel_completeness_by_method's note above its call):
    method-color x state-alpha swatches, method names as column headers, state names down the right.
    Drawn in its own small axes at `rect` (figure fractions); its inner coordinates are fixed
    (0, 1) axes fractions, so the same absolute inch size (rect width/height x figure size) gives
    the same look in any figure."""
    leg_ax = fig.add_axes(rect)
    leg_ax.axis("off")
    leg_ax.set_xlim(0, 1)
    leg_ax.set_ylim(0, 1)
    col_x = [0.25, 0.45]        # each method column's swatch center
    label_x = col_x[-1] + 0.08  # row-label text's left-aligned anchor, right of the rightmost swatch
    row_y = [0.00, 0.30, 0.60]  # Complete (bottom) -> Insufficient (top), matching COMPLETENESS_ORDER
    sw_w, sw_h = 0.05, 0.14     # swatch size
    for ri, state in enumerate(COMPLETENESS_ORDER):
        for ci, method in enumerate(S.METHOD_ORDER):
            leg_ax.add_patch(Rectangle((col_x[ci] - sw_w / 2, row_y[ri]), sw_w, sw_h,
                                       facecolor=S.method_color(method), alpha=STATE_ALPHA[state],
                                       edgecolor=S.NEUTRAL, linewidth=.3))
        leg_ax.text(label_x, row_y[ri] + sw_h / 2, state, ha="left", va="center", fontsize=5.0)
    for ci, method in enumerate(S.METHOD_ORDER):
        leg_ax.text(col_x[ci] - sw_w / 2 - 0.06, row_y[-1] + sw_h + 0.12, method, ha="left", va="bottom",
                   fontsize=5.0)


def unit_label():
    """Row-entity noun for axis labels: proteins for the protein-level run (ID_COL overridden by
    generalization.run), precursors otherwise."""
    return "Proteins" if ID_COL == "Protein.Group" else "Precursors"


def panel_completeness_by_method(contrasts, coverage, completeness_transition, MAIN, sv,
                                 stem="Fig_Performance_B_completeness_by_method"):
    """Stacked-by-method completeness bars -- same figure for the full run() pipeline's Main B and
    the standalone `generalization` stage. Stacked by METHOD color (blue=DIA-NN, red=JUMPlion)
    instead of by completeness-state color -- each method's own Complete/Partially complete/
    Insufficient breakdown is one stacked bar, shaded from solid (Complete) down to translucent
    (Insufficient) of that method's own color, with the two methods' bars side by side per
    contrast. Annotations are only on the DIA-NN bar: its Partially complete count, and (between
    its Partially complete top and a dashed line at JUMPlion's Complete + Partially complete top)
    the Insufficient (restored) count. The shared legend states each segment's own
    completeness-state name, and the bar height reads directly as the percentage."""
    x = np.arange(len(contrasts))
    coverage_by_method = _coverage_by_method(coverage, completeness_transition)

    fig_compm, axm = plt.subplots(figsize=(bw, bh))
    wm = 0.38
    for mi, method in enumerate(S.METHOD_ORDER):
        cov_m = coverage_by_method[method]
        totals_m = np.array([cov_m[cov_m.contrast == cc].n.sum() for cc in contrasts], float)
        bottom_m = np.zeros(len(contrasts))
        off = (-wm / 2, wm / 2)[mi]
        for state in COMPLETENESS_ORDER:
            v = np.array([cov_m[(cov_m.contrast == cc) & (cov_m.state == state)].n.sum() for cc in contrasts], float)
            pct = np.divide(100 * v, totals_m, out=np.zeros_like(v), where=totals_m > 0)
            axm.bar(x + off, pct, width=wm, bottom=bottom_m, color=S.method_color(method),
                    alpha=STATE_ALPHA[state], edgecolor=S.NEUTRAL, linewidth=.3)
            if method == M_D and state == PARTIAL:  # DIA-NN's partially complete count
                for xi, (b0, h, n) in enumerate(zip(bottom_m, pct, v)):
                    if h > 0:
                        axm.text(xi + off, b0 + h / 2, f"{int(n):,}", ha="center", va="center", fontsize=4.5)
            bottom_m += pct
        if method == M_J:
            # JUMPlion's Complete + Partially complete top, drawn across the DIA-NN bar: the stretch
            # between it and DIA-NN's own Partially complete top is the Insufficient (restored) share.
            cov_d = coverage_by_method[M_D]
            tot_d = {cc: cov_d[cov_d.contrast == cc].n.sum() for cc in contrasts}
            for xi, cc in enumerate(contrasts):
                if tot_d[cc] <= 0:
                    continue
                cj = cov_m[cov_m.contrast == cc]
                top_j = 100 * cj[cj.state.isin([COMPLETE, PARTIAL])].n.sum() / cj.n.sum()
                dd = cov_d[(cov_d.contrast == cc) & cov_d.state.isin([COMPLETE, PARTIAL])].n.sum()
                top_d = 100 * dd / tot_d[cc]
                axm.plot([xi - wm, xi], [top_j, top_j], color=S.NEUTRAL, lw=.6, ls="--", zorder=4)
                tr = completeness_transition
                n_rest = int(tr[(tr.contrast == cc) & (tr.baseline_state == INSUFFICIENT)
                                & tr.state_after_jumplion.isin([COMPLETE, PARTIAL])].n.sum())
                if n_rest > 0 and top_j > top_d:
                    axm.text(xi - wm / 2, (top_d + top_j) / 2, f"{n_rest:,}", ha="center", va="center",
                             fontsize=4.5, zorder=5)
    contrast_ticks(axm, contrasts, rotation=0)
    axm.set_ylabel("Precursors (%)")
    axm.set_ylim(0, 100)
    axm.set_yticks([0, 20, 40, 60, 80, 100])
    # Custom table-style legend (not Axes.legend(), which has no row/column-header concept): six
    # color swatches in two columns (one per method, each swatch the actual method-color x
    # state-alpha its row's state uses), state names (Complete/Partially complete/Insufficient)
    # listed once, down the RIGHT side of the swatches bottom to top, method names as column
    # headers above -- states only stated once each instead of repeated into all six labels (the
    # one-label-per-swatch version tried first). Built as its own small axes in the figure's top
    # margin, with xlim/ylim fixed to (0, 1) so its own data coordinates already ARE axes-fraction
    # coordinates.
    completeness_table_legend(fig_compm, [0.02, .80, .96, .14])
    compact_axes(axm)
    fig_compm.subplots_adjust(left=INTEGRATED_LEFT, right=INTEGRATED_RIGHT, top=.75, bottom=INTEGRATED_BOTTOM)
    # tight=False: this panel's saved pixel size must exactly match figsize (PANEL_AB_SIZE)
    # for its INTEGRATED_LEFT/RIGHT/BOTTOM margins to land at the same absolute position as
    # Fig_Performance_A_coverage's -- the default tight=True crops to each panel's own content
    # bbox, which differs between the two (different legend/label extents), silently breaking the
    # alignment even though both start from the same figsize.
    sv(fig_compm, MAIN, stem, tight=False)


def metric_rows(df, contrast, method, stratum="all", stratum_value="all"):
    """Sensitivity and empirical FDR, each reported (not NaN) only above its own minimum
    denominator -- sensitivity needs enough ground-truth differential precursors (TP+FN+wrong),
    FDR needs enough discoveries (TP+FP); a handful of either makes the rate unstable."""
    x = df[df["benchmark_state"] != "NE"]
    c = x["benchmark_state"].value_counts()
    TP, FP, FN, TN = [int(c.get(k, 0)) for k in ["TP", "FP", "FN", "TN"]]
    wrong = int(c.get("wrong_direction", 0))
    n_truth, n_disc = TP + FN + wrong, TP + FP
    sens = TP / n_truth if n_truth >= MIN_N_SENSITIVITY else np.nan
    sl, sh = wilson(TP, n_truth)
    # FDR is STRICT by default (empirical_fdr, used by every plot): a significant call in the wrong
    # direction counts as a false discovery along with the human FP, (FP + wrong) / (TP + FP + wrong).
    # The earlier human-only definition, FP / (TP + FP), is kept in the *_fp_only columns. Both use the
    # same minimum-discoveries rule.
    n_disc_strict = n_disc + wrong
    fdr = (FP + wrong) / n_disc_strict if n_disc_strict >= MIN_N_DISCOVERIES else np.nan
    fl, fh = wilson(FP + wrong, n_disc_strict)
    fdr_fp_only = FP / n_disc if n_disc >= MIN_N_DISCOVERIES else np.nan
    fpl, fph = wilson(FP, n_disc)
    return dict(contrast=contrast, method=method, stratum=stratum, stratum_value=stratum_value,
                n=len(x), TP=TP, FP=FP, FN=FN, TN=TN, wrong_direction=wrong, sensitivity=sens,
                sensitivity_ci_low=sl, sensitivity_ci_high=sh,
                empirical_fdr=fdr, fdr_ci_low=fl, fdr_ci_high=fh,
                empirical_fdr_fp_only=fdr_fp_only, fdr_fp_only_ci_low=fpl, fdr_fp_only_ci_high=fph)


def _matched_ylim(*series, frac=0.1, floor_zero=False):
    vals = pd.concat(series).replace([np.inf, -np.inf], np.nan).dropna()
    if not len(vals):
        return (0.0, 100.0) if floor_zero else (-1.0, 1.0)
    lo, hi = float(vals.min()), float(vals.max())
    span = hi - lo if hi > lo else max(abs(hi), 1.0)
    return (0.0 if floor_zero else lo - frac * span), hi + frac * span


def compute_sensitivity_fdr(contrasts, prepared, restored):
    """Sensitivity/FDR for the partially-complete stratum (both methods) and the restored
    (Insufficient -> sufficient via JUMPlion) stratum (JUMPlion only), plus their shared,
    y-axis-matched (sensitivity, FDR) ranges. Shared by the full run() pipeline's Main D panel and
    the standalone `generalization` stage. Returns (partial_perf, rec_perf, sens_ylim, fdr_ylim)."""
    partial_perf = []
    for c, (d, j, m, other) in prepared.items():
        g = m[m.completeness_state == PARTIAL]
        for method, suf in METHOD_SUFFIX:
            temp = pd.DataFrame({"benchmark_state": g[f"benchmark_state_{suf}"],
                                 "Species_std": g[f"Species_std_{suf}"]})
            partial_perf.append(metric_rows(temp, c, method, "state", PARTIAL))
    partial_perf = pd.DataFrame(partial_perf)

    rec_perf = []
    for c in contrasts:
        rr = restored[restored.contrast == c]
        rec_perf.append(metric_rows(rr, c, M_J, "state", INSUFFICIENT))
    rec_perf = pd.DataFrame(rec_perf)

    sens_ylim = _matched_ylim(partial_perf.sensitivity * 100, rec_perf.sensitivity * 100, floor_zero=True)
    fdr_ylim = _matched_ylim(partial_perf.empirical_fdr * 100, rec_perf.empirical_fdr * 100, floor_zero=True)
    return partial_perf, rec_perf, sens_ylim, fdr_ylim


def panel_sensitivity_fdr(contrasts, partial_perf, rec_perf, sens_ylim, fdr_ylim, MAIN, sv,
                          stem="Fig_Performance_D_sensitivity_fdr"):
    """Row 0: partially complete (both methods). Row 1: restored (insufficient), JUMPlion-only.
    One sharex figure (both rows plot the same contrast categories); both rows keep their own
    (rotated) contrast tick labels, not just the bottom row. Same figure for the full run()
    pipeline's Main D and the standalone `generalization` stage."""
    x = np.arange(len(contrasts))
    zz = rec_perf.set_index("contrast").reindex(contrasts)
    fig, axs = plt.subplots(2, 2, figsize=(dw, dh), sharex=True)
    for method in S.METHOD_ORDER:
        zzd = partial_perf[partial_perf.method == method].set_index("contrast").reindex(contrasts)
        axs[0, 0].plot(x, zzd.sensitivity * 100, marker="o", ms=3, lw=1, color=S.method_color(method), label=method)
        axs[0, 1].plot(x, zzd.empirical_fdr * 100, marker="o", ms=3, lw=1, color=S.method_color(method))
    axs[1, 0].plot(x, zz.sensitivity * 100, marker="o", ms=3, lw=1, color=S.method_color(M_J))
    axs[1, 1].plot(x, zz.empirical_fdr * 100, marker="o", ms=3, lw=1, color=S.method_color(M_J))
    for row in (0, 1):
        contrast_ticks(axs[row, 0], contrasts, rotation=30); contrast_ticks(axs[row, 1], contrasts, rotation=30)
        axs[row, 0].set_ylabel("Sensitivity (%)"); axs[row, 0].set_ylim(*sens_ylim)
        axs[row, 1].set_ylabel("Empirical FDR (%)"); axs[row, 1].set_ylim(*fdr_ylim)
    # sharex=True's automatic label_outer() hides row 0's tick labels (labelbottom=False), a
    # visibility flag that contrast_ticks()'s set_xticklabels() (label TEXT only) doesn't undo.
    for ax in (axs[0, 0], axs[0, 1]):
        ax.tick_params(labelbottom=True)

    # n annotations: the number of precursors behind each contrast's sensitivity/FDR (metric_rows'
    # own "n" -- same denominator shown on both subplots in a row, since sensitivity and FDR are
    # computed from the same stratum). Row 0 plots both methods, but only JUMPlion's n is
    # shown (matches row 1, which only has JUMPlion); warn instead of silently picking one if
    # DIA-NN's n for a contrast differs from it.
    n_d_map = partial_perf[partial_perf.method == M_D].set_index("contrast")["n"].reindex(contrasts)
    n_j_map = partial_perf[partial_perf.method == M_J].set_index("contrast")["n"].reindex(contrasts)
    for c, nd, nj in zip(contrasts, n_d_map, n_j_map):
        if pd.notna(nd) and pd.notna(nj) and int(nd) != int(nj):
            log.warning("%s: %s has different n between DIA-NN (%d) and JUMPlion (%d) in "
                        "the partial-completeness stratum -- only JUMPlion's n is annotated.",
                        stem, c, int(nd), int(nj))
    n_f_map = rec_perf.set_index("contrast")["n"].reindex(contrasts)
    for row, n_map in ((0, n_j_map), (1, n_f_map)):
        for ax in (axs[row, 0], axs[row, 1]):
            for xi, n in zip(x, n_map):
                if pd.notna(n):
                    ax.text(xi, 1.02, f"n={int(n)}", ha="center", va="bottom", fontsize=4.5,
                           transform=ax.get_xaxis_transform())

    compact_axes(*axs.ravel())
    # No per-figure legend here -- method color is identified by Main B's own legend elsewhere in
    # the composite.
    fig.subplots_adjust(left=.15, right=.95, top=ROW2_TOP, bottom=ROW2_BOTTOM, hspace=ROW2_HSPACE, wspace=.55)
    # tight=False: this panel's saved pixel size must exactly match figsize (dw, dh) -- D and E
    # share the same nominal figsize AND the same ROW2_TOP/ROW2_BOTTOM/ROW2_HSPACE row-split, so
    # only an EXACT (not cropped) saved size guarantees their row 0/row 1 baselines land at the
    # same absolute height when assemble_integrated_figure tiles them side by side; the default
    # tight=True crops to each panel's own content bbox, which differs between the two (different
    # y-axis label text, n= annotation extents, etc.), breaking that alignment even though both
    # start from the same figsize.
    sv(fig, MAIN, stem, tight=False)


def _whisker_bounds(x, whis=1.5):
    # Matplotlib's own boxplot whisker rule (Q1/Q3 +/- whis*IQR, clipped to the most extreme
    # in-range point). The CV/delta-FC panels draw boxplots with showfliers=False, so the data's
    # raw min/max (which can be far-out fliers) would pad the shared axis much more than what is
    # ever actually drawn.
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if not len(x):
        return np.nan, np.nan
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    within = x[(x >= q1 - whis * iqr) & (x <= q3 + whis * iqr)]
    return (float(within.min()), float(within.max())) if len(within) else (float(q1), float(q3))


def _matched_box_ylim(groups, frac=0.15, floor_zero=False):
    # groups: one array per rendered box (already on display scale); range is set just beyond the
    # most extreme whisker across all of them, not the raw data extremes.
    los, his = [], []
    for g in groups:
        lo, hi = _whisker_bounds(g)
        if np.isfinite(lo):
            los.append(lo); his.append(hi)
    if not los:
        return (0.0, 100.0) if floor_zero else (-1.0, 1.0)
    lo, hi = min(los), max(his)
    span = hi - lo if hi > lo else max(abs(hi), 1.0)
    return (0.0 if floor_zero else lo - frac * span), hi + frac * span


def compute_cv_deltafc(contrasts, prepared, restored):
    """Pooled CV and delta-FC (observed - expected log2FC) for the partially-complete stratum
    (both methods) and the restored stratum (JUMPlion only), plus their shared, y-axis-matched
    (CV, delta-FC) ranges, needed by panel_cv_deltafc. Lean subset of the full run() pipeline's
    Main E CV/delta-FC block -- no intensity_bin column, since panel_cv_deltafc (unlike the
    intensity-bin composites elsewhere in run()) groups by species only.
    partial_fc/restored_fc also carry "abs_log2FC_error" (|delta_FC|), written into their own
    tables for reference, for accuracy regardless of direction rather than a signed,
    over/under-estimation view (panel_cv_deltafc's own "delta_FC" boxes).
    Returns (partial_cv, partial_fc, restored_cv, restored_fc, cv_ylim, fc_ylim)."""
    partial_cv_rows, partial_fc_rows = [], []
    for c, (d, j, m, other) in prepared.items():
        ref = m["ref_group"].iloc[0]  # this contrast's own reference/denominator side (see prepare_pair)
        g = m[m.completeness_state == PARTIAL]
        for method, suf in METHOD_SUFFIX:
            for grp in (ref, other):
                col = f"cv_{grp}_{suf}"
                if col not in g.columns:
                    continue
                cvs = pd.to_numeric(g[col], errors="coerce")
                frame = pd.DataFrame({"contrast": c, "method": method,
                                     "species": g[f"Species_std_{suf}"].values,
                                     "CV": cvs.values}).dropna(subset=["CV"])
                partial_cv_rows.append(frame[frame.species.isin(CLEAN)])
            for sp in CLEAN:
                z = g[g[f"Species_std_{suf}"] == sp]
                delta = pd.to_numeric(z[f"log2FC_{suf}"], errors="coerce") - EXPECTED[c][sp]
                ok = delta.notna()
                if ok.any():
                    partial_fc_rows.append(pd.DataFrame({"contrast": c, "method": method, "species": sp,
                                                         "delta_FC": delta[ok].values}))
    partial_cv = pd.concat(partial_cv_rows, ignore_index=True)
    partial_fc = pd.concat(partial_fc_rows, ignore_index=True)

    restored_cv_rows = []
    for col in ("cv_ref", "cv_comparison"):
        cvs = pd.to_numeric(restored[col], errors="coerce")
        frame = pd.DataFrame({"contrast": restored["contrast"].values, "species": restored["Species_std"].values,
                              "CV": cvs.values}).dropna(subset=["CV"])
        restored_cv_rows.append(frame[frame.species.isin(CLEAN)])
    restored_cv = pd.concat(restored_cv_rows, ignore_index=True)

    restored_fc_rows = []
    for sp in CLEAN:
        z = restored[restored.Species_std == sp]
        if len(z):
            delta = pd.to_numeric(z["log2FC"], errors="coerce") - z["contrast"].map(lambda cc: EXPECTED[cc][sp])
            ok = delta.notna()
            if ok.any():
                restored_fc_rows.append(pd.DataFrame({"contrast": z.loc[ok, "contrast"].values, "species": sp,
                                                       "delta_FC": delta[ok].values}))
    restored_fc = pd.concat(restored_fc_rows, ignore_index=True)

    partial_fc["abs_log2FC_error"] = partial_fc["delta_FC"].abs()
    restored_fc["abs_log2FC_error"] = restored_fc["delta_FC"].abs()

    cv_box_groups = ([partial_cv[(partial_cv.species == sp) & (partial_cv.method == me)].CV.to_numpy(float) * 100
                      for sp in CLEAN for me in S.METHOD_ORDER] +
                     [restored_cv[restored_cv.species == sp].CV.to_numpy(float) * 100 for sp in CLEAN])
    fc_box_groups = ([partial_fc[(partial_fc.species == sp) & (partial_fc.method == me)].delta_FC.to_numpy(float)
                      for sp in CLEAN for me in S.METHOD_ORDER] +
                     [restored_fc[restored_fc.species == sp].delta_FC.to_numpy(float) for sp in CLEAN])
    cv_ylim = _matched_box_ylim(cv_box_groups, floor_zero=True)
    fc_ylim = _matched_box_ylim(fc_box_groups)
    return partial_cv, partial_fc, restored_cv, restored_fc, cv_ylim, fc_ylim


def panel_cv_deltafc(partial_cv, partial_fc, restored_cv, restored_fc, cv_ylim, fc_ylim, MAIN, sv,
                     stem="Fig_Performance_E_quantitative_fidelity"):
    """Row 0: partially complete, paired DIA-NN vs JUMPlion boxes. Row 1: restored
    (insufficient), JUMPlion-only (restored precursors have no DIA-NN counterpart). One sharex
    figure (both rows plot the same species categories). Both rows keep their own x tick labels
    (species names), not just the bottom row. No legend of its own -- same as D, the other panel in
    Fig_Performance_Integrated's own row 2; method color (blue=DIA-NN, red=JUMPlion) is
    identified by Main B's own legend elsewhere in the composite."""
    partial_cv_pct = partial_cv.assign(CV=partial_cv["CV"] * 100)
    fig, axs = plt.subplots(2, 2, figsize=(ew, eh), sharex=True)
    paired_boxes(axs[0, 0], partial_cv_pct, "species", CLEAN, "CV")
    axs[0, 0].set_ylabel("CV (%)"); axs[0, 0].set_ylim(*cv_ylim)
    paired_boxes(axs[0, 1], partial_fc, "species", CLEAN, "delta_FC")
    S.ref_line(axs[0, 1], 0)
    axs[0, 1].set_ylabel("log$_2$FC error"); axs[0, 1].set_ylim(*fc_ylim)

    pos = list(range(len(CLEAN)))
    data_cv = [restored_cv[restored_cv.species == sp].CV.to_numpy(float) * 100 for sp in CLEAN]
    S.boxplot_groups(axs[1, 0], data_cv, pos, [M_J] * len(CLEAN), S.METHOD_COLORS)
    axs[1, 0].set_ylabel("CV (%)"); axs[1, 0].set_ylim(*cv_ylim)
    data_fc = [restored_fc[restored_fc.species == sp].delta_FC.to_numpy(float) for sp in CLEAN]
    S.boxplot_groups(axs[1, 1], data_fc, pos, [M_J] * len(CLEAN), S.METHOD_COLORS)
    S.ref_line(axs[1, 1], 0)
    axs[1, 1].set_ylabel("log$_2$FC error"); axs[1, 1].set_ylim(*fc_ylim)

    for row in (0, 1):
        for col in (0, 1):
            axs[row, col].set_xticks(pos)
            axs[row, col].set_xticklabels([S.species_label(s) for s in CLEAN], rotation=30, ha="right")
    # sharex=True's automatic label_outer() hides row 0's tick labels (labelbottom=False), a
    # visibility flag that set_xticklabels()'s label TEXT alone doesn't undo.
    for ax in (axs[0, 0], axs[0, 1]):
        ax.tick_params(labelbottom=True)
    compact_axes(*axs.ravel())
    fig.subplots_adjust(left=.15, right=.95, top=ROW2_TOP, bottom=ROW2_BOTTOM, hspace=ROW2_HSPACE, wspace=.55)
    # Two-column method legend in the top margin (stays inside the 1 - ROW2_TOP strip, so the axes
    # positions, and the alignment with D, are unchanged).
    fig.legend(handles=[S.Patch(facecolor=S.method_color(m), edgecolor="none", linewidth=0, label=m)
                        for m in S.METHOD_ORDER],
               loc="upper center", bbox_to_anchor=(.55, .995), ncol=2, frameon=False, fontsize=5.0,
               handlelength=1.2, columnspacing=1.5)
    # tight=False: see panel_sensitivity_fdr's own matching note -- D and E share the same nominal
    # figsize and row-split fractions, so only an exact (uncropped) saved size keeps their row 0/
    # row 1 baselines aligned in the composite.
    sv(fig, MAIN, stem, tight=False)


def balanced(g, sp_col, st_col):
    """Species-balanced FDR helper (1:1:1 weighting): returns (human, yeast, ecoli subsets,
    human FP rate, yeast TP rate, ecoli TP rate, balanced FDR). Balancing combines a per-species
    rate from each of the three species, so each rate -- not just the total discovery count --
    needs its own minimum denominator (MIN_N_SPECIES_BALANCED); if any one species is too thin
    (e.g. 2 E. coli calls in a bin with 50 total), the whole balanced FDR is reported as NaN
    rather than silently resting on that one unstable component.
    The balanced FDR is STRICT, like metric_rows': wrong-direction yeast / E. coli calls count as
    false discoveries along with the human FP, (hr + yw + ew) / (hr + yw + ew + yr + er) -- which
    reduces to the earlier hr / (hr + yr + er) when there are none."""
    h, y, e = g[g[sp_col] == "HUMAN"], g[g[sp_col] == "YEAST"], g[g[sp_col] == "ECOLI"]
    hr = (h[st_col] == "FP").mean() if len(h) >= MIN_N_SPECIES_BALANCED else np.nan
    yr = (y[st_col] == "TP").mean() if len(y) >= MIN_N_SPECIES_BALANCED else np.nan
    er = (e[st_col] == "TP").mean() if len(e) >= MIN_N_SPECIES_BALANCED else np.nan
    yw = (y[st_col] == "wrong_direction").mean() if len(y) >= MIN_N_SPECIES_BALANCED else np.nan
    ew = (e[st_col] == "wrong_direction").mean() if len(e) >= MIN_N_SPECIES_BALANCED else np.nan
    false_part, true_part = hr + yw + ew, yr + er
    den = false_part + true_part
    return h, y, e, hr, yr, er, (false_part / den if np.isfinite(den) and den > 0 else np.nan)


def paired_boxes(ax, frame, key_col, keys, val_col):
    pos, data, cat = [], [], []
    for i, k in enumerate(keys):
        for off, method in [(-.18, M_D), (.18, M_J)]:
            pos.append(i + off)
            data.append(frame[(frame[key_col] == k) & (frame.method == method)][val_col].to_numpy(float))
            cat.append(method)
    S.boxplot_groups(ax, data, pos, cat, S.METHOD_COLORS)
    ax.set_xticks(range(len(keys)))


def contrast_ticks(ax, contrasts, rotation=0):
    ax.set_xticks(range(len(contrasts)))
    if rotation:
        ax.set_xticklabels([S.clabel(c) for c in contrasts], rotation=rotation, ha="right")
    else:
        ax.set_xticklabels([S.clabel(c) for c in contrasts])


def bin_ticks(ax, labels):
    """labels: the global display labels from quantile_bin_edges() (actual log2-intensity
    cutoffs), not the generic BIN_LABELS category keys used for data grouping."""
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    # Pin the view to all 5 bin positions even if trailing/leading ones are all-NaN (e.g. masked
    # by a minimum-denominator rule) -- otherwise autoscale, which ignores NaN points entirely,
    # can crop those tick positions out of view and make it look like the bin was never plotted.
    ax.set_xlim(-0.6, len(labels) - 0.4)


# ----------------------------------------------------------------------------
# Replicate-level representative examples
# ----------------------------------------------------------------------------
def infer_matrix_group_columns(columns, sample_meta=None):
    """Map group label (S1..S5) -> replicate columns.

    Uses the sample metadata (Run -> Group) when available; otherwise falls back to
    matching the mixture pattern (e.g. '_50-10-40_') in the column names.
    """
    columns = list(columns)
    if sample_meta is not None:
        run_to_group = dict(zip(sample_meta["Run"].astype(str), sample_meta["Group"].astype(str)))
        out = {lab: [c for c in columns if run_to_group.get(str(c)) == g] for lab, g in GROUPS.items()}
    else:
        pat = lambda mix: "_" + "-".join(f"{v:g}" for v in mix) + "_"
        out = {lab: [c for c in columns if pat(MIX[lab]) in c] for lab in GROUPS}
    bad = [g for g, v in out.items() if not v]
    if bad:
        raise ValueError(f"Could not infer replicate columns for groups: {bad}")
    return out


def plot_example_method(ax, matrix, pid, group_cols, comparison_group, method, stat_row, state_text,
                        stat_fontsize=6, state_fontsize=6, other_matrix=None, show_xticklabels=True,
                        completeness_label=None):
    """other_matrix: when given (the DIA-NN matrix, while plotting the JUMPlion matrix),
    points are split by whether DIA-NN already had a value at that exact precursor+replicate cell
    ("shared", colored like the DIA-NN panel) vs. not ("additional" -- a value only JUMPlion
    reports, colored like the JUMPlion panel) -- instead of one flat method color.
    `completeness_label`: optional (e.g. "Partially complete"/"Insufficient"), the ONLY thing shown
    in the bottom annotation now. The benchmark-state call (FN/TP/wrong_direction) is a different
    axis (what DE concluded, not how complete the data was) and is shown in-plot instead, at the
    bottom corner opposite the top stat box -- left if log2FC < 0, right if log2FC > 0 -- not
    stated at all when stat_row is None (not DE-eligible, so there's no log2FC to place it by;
    completeness_label alone already conveys that)."""
    color = S.method_color(method)
    shared_color, additional_color = S.method_color(M_D), S.method_color(M_J)
    for xi, g in enumerate([REF_LABEL, comparison_group]):
        cols = group_cols[g]
        vals = pd.to_numeric(matrix.loc[pid, cols], errors="coerce").to_numpy(dtype=float, copy=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            lv = np.log2(np.where(vals > 0, vals, np.nan))
        xs = np.full(len(lv), xi, dtype=float) + np.linspace(-0.05, 0.05, len(lv))
        if other_matrix is not None and pid in other_matrix.index:
            other_vals = pd.to_numeric(other_matrix.loc[pid, cols], errors="coerce").to_numpy(dtype=float)
            additional = np.isfinite(lv) & ~np.isfinite(other_vals)
            pt_colors = [additional_color if a else shared_color for a in additional]
        else:
            pt_colors = color
        ax.scatter(xs, lv, s=12, color=pt_colors, alpha=0.85, edgecolors="none")
        if np.isfinite(lv).any():
            med = np.nanmedian(lv)
            ax.plot([xi - .11, xi + .11], [med, med], lw=1.2, color=S.NEUTRAL)
    # show_xticklabels=False hides the tick marks entirely (not just their text) -- no caller wants
    # bare unlabeled tick marks, so one flag controls both.
    if show_xticklabels:
        ax.set_xticks([0, 1])
        ax.set_xticklabels([REF_LABEL, comparison_group])
    else:
        ax.set_xticks([])
    ax.set_xlim(-0.5, 1.5)
    if stat_row is not None:
        log2fc = stat_row["log2FC"]
        padj = stat_row["p_adj"]
        padj_txt = f"{padj:.2e}" if padj < 0.01 else f"{padj:.3g}"
        lines = [f"log2FC {log2fc:.2f}", f"Padj {padj_txt}"]
        ha = "left" if log2fc > 0 else "right"
        tx = .97 if ha == "right" else .03
        ax.text(tx, .97, "\n".join(lines), transform=ax.transAxes, ha=ha, va="top", fontsize=stat_fontsize)
        if state_text:
            # Opposite corner from the stat box above, at the bottom instead of the top -- left if
            # log2FC < 0, right if log2FC > 0.
            state_ha = "left" if log2fc < 0 else "right"
            state_tx = .03 if state_ha == "left" else .97
            ax.text(state_tx, .03, state_text, transform=ax.transAxes, ha=state_ha, va="bottom",
                   fontsize=state_fontsize)
    if completeness_label:
        # color (this subplot's own method color, already computed above for the scatter points)
        # -- blue=DIA-NN, red=JUMPlion -- identifies which method this completeness state
        # belongs to without needing a separate legend.
        if show_xticklabels:
            ax.text(.5, -.24, completeness_label, transform=ax.transAxes, ha="center", va="top",
                   fontsize=state_fontsize, color=color)
        else:
            ax.text(.5, -.12, completeness_label, transform=ax.transAxes, ha="center", va="top",
                   fontsize=state_fontsize, color=color)


def _axes_data_yrange(ax):
    """Actual finite y-data range plotted on `ax` (scatter + line artists), or None if it has
    none -- unlike ax.get_ylim(), which falls back to matplotlib's default (0, 1) range for an
    empty axes, so naively combining get_ylim() across two side-by-side axes (one of which may
    have no DIA-NN quantification at all) drags that (0, 1) default into the shared range."""
    ys = []
    for coll in ax.collections:
        off = coll.get_offsets()
        if len(off):
            ys.append(np.asarray(off)[:, 1])
    for ln in ax.lines:
        ys.append(np.asarray(ln.get_ydata(), dtype=float))
    if not ys:
        return None
    y = np.concatenate(ys)
    y = y[np.isfinite(y)]
    return (float(y.min()), float(y.max())) if len(y) else None


def _nfinite(matrix, pid, cols):
    return int(np.isfinite(pd.to_numeric(matrix.loc[pid, cols], errors="coerce").to_numpy(dtype=float)).sum())


def generate_replicate_examples(cfg, resolve, prepared, restored, L, sv):
    """Replicate-level scatter examples of the two TP-gain routes: FN->TP among precursors DIA-NN
    could already test (Partially complete), and TP among precursors DIA-NN could not test at all
    until JUMPlion (Insufficient -> restored)."""
    dm_path, jm_path = resolve(cfg.get("diann_matrix")), resolve(cfg.get("jumplion_matrix"))
    if dm_path is None or jm_path is None:
        log.info("Representative examples skipped: diann_matrix / jumplion_matrix not configured.")
        return False
    for p in (dm_path, jm_path):
        if not p.exists():
            raise FileNotFoundError(f"Replicate matrix not found: {p}")
    sep = cfg.get("matrix_sep", "\t")
    md = pd.read_csv(dm_path, sep=sep).drop_duplicates("Precursor.Id").set_index("Precursor.Id")
    mj = pd.read_csv(jm_path, sep=sep).drop_duplicates("Precursor.Id").set_index("Precursor.Id")
    meta_path = resolve(cfg.get("sample_meta_file"))
    meta = pd.read_csv(meta_path) if meta_path is not None and meta_path.exists() else None
    group_cols = infer_matrix_group_columns(md.columns, meta)

    fn_rows, nt_rows = [], []
    for c, (d, j, m, other_name) in prepared.items():
        comp = c.split("_vs_")[0]
        z = m[(m.benchmark_state_D == "FN") & (m.benchmark_state_J == "TP") &
              (m.Species_std_J.isin(["YEAST", "ECOLI"]))]
        for _, r in z.iterrows():
            pid = r["Precursor.Id"]
            if pid not in md.index or pid not in mj.index:
                continue
            sp = r.Species_std_J
            low = REF_LABEL if sp == "YEAST" else comp
            nD, nJ = _nfinite(md, pid, group_cols[low]), _nfinite(mj, pid, group_cols[low])
            err = abs(r.log2FC_J - EXPECTED[c][sp])
            fn_rows.append(dict(contrast=c, Precursor_Id=pid, Species=sp, comparison_group=comp,
                                low_group=low, nD_low=nD, nJ_low=nJ,
                                log2FC_D=r.log2FC_D, padj_D=r.p_adj_D,
                                log2FC_J=r.log2FC_J, padj_J=r.p_adj_J,
                                fc_error_J=err, score=err - 0.05 * (nJ - nD)))
        ntc = restored[(restored.contrast == c) & (restored.benchmark_state == "TP")]
        jidx = j.drop_duplicates("Precursor.Id").set_index("Precursor.Id")
        for _, r in ntc.iterrows():
            pid = r["Precursor.Id"]
            if pid not in md.index or pid not in mj.index or pid not in jidx.index:
                continue
            sp = norm_species(jidx.at[pid, "Species"])
            if sp not in ["YEAST", "ECOLI"]:
                continue
            nD1, nD2 = _nfinite(md, pid, group_cols[REF_LABEL]), _nfinite(md, pid, group_cols[comp])
            nJ1, nJ2 = _nfinite(mj, pid, group_cols[REF_LABEL]), _nfinite(mj, pid, group_cols[comp])
            js = jidx.loc[pid]
            err = abs(js.log2FC - EXPECTED[c][sp])
            nt_rows.append(dict(contrast=c, Precursor_Id=pid, Species=sp, comparison_group=comp,
                                nD_S1=nD1, nD_other=nD2, nJ_S1=nJ1, nJ_other=nJ2,
                                log2FC_J=js.log2FC, padj_J=js.p_adj, fc_error_J=err,
                                score=err - 0.05 * ((nJ1 + nJ2) - (nD1 + nD2))))

    fnc, ntc = pd.DataFrame(fn_rows), pd.DataFrame(nt_rows)
    if fnc.empty or ntc.empty:
        raise ValueError("Representative examples: no FN->TP or restored->TP replicate-level candidates.")
    fnc = fnc.sort_values(["contrast", "score"])
    ntc = ntc.sort_values(["contrast", "score"])
    write_tsv(fnc, L.table, "partial_FN_to_TP_replicate_example_candidates.tsv")
    write_tsv(ntc, L.table, "restored_TP_replicate_example_candidates.tsv")

    def stats_for(c, pid):
        d, j, m, g1 = prepared[c]
        didx = d.drop_duplicates("Precursor.Id").set_index("Precursor.Id")
        jidx = j.drop_duplicates("Precursor.Id").set_index("Precursor.Id")
        ds = didx.loc[pid] if pid in didx.index else None
        js = jidx.loc[pid] if pid in jidx.index else None
        # JUMPlion's OWN completeness state for this precursor (Complete/Partially complete only --
        # j is already DE-eligible-only, same reasoning as prepare_pair's own
        # m["completeness_state"] comment, so this is never "Insufficient" either). Computed from
        # j's own n_obs columns directly, not m's -- m's inner merge drops any precursor DIA-NN
        # itself never reached DE-eligibility for, exactly the Insufficient-route examples this
        # needs to cover (pid absent from m, but still present in j).
        g2 = m["ref_group"].iloc[0]
        jlabel = (classify_completeness(np.array([js[f"n_obs_{g2}"]]), np.array([js[f"n_obs_{g1}"]]))[0]
                 if js is not None else None)
        return (ds, benchmark_state_one(ds) if ds is not None else None,
                js, benchmark_state_one(js) if js is not None else None, jlabel)

    # Curated: one example per gain route in the chosen contrast.
    main_c = cfg.get("example_contrast", "S4_vs_S1")
    if main_c not in prepared:
        main_c = list(prepared)[-1]
    nt_pool = ntc[ntc.contrast == main_c]
    fn_pool = fnc[fnc.contrast == main_c]
    if nt_pool.empty or fn_pool.empty:
        raise ValueError(f"No example candidates for {main_c}.")
    nt_pref = nt_pool[(nt_pool.nD_S1 >= 1) & (nt_pool.nD_other >= 1)]
    nt_row = (nt_pref if len(nt_pref) else nt_pool).sort_values("score").iloc[0]
    fn_row = fn_pool.sort_values("score").iloc[0]

    # Standalone Fig_Performance_C1/C2_representative_example_* were removed -- C1+C2 combined
    # below is the only representative-example panel now. row_defs is still needed by it.
    row_defs = [("C1", "Partially complete", fn_row), ("C2", "Insufficient", nt_row)]

    # ======================= Main C: representative examples (for the B-detail composite) =======================
    # Same two examples as C1/C2 above, but as one sharex figure (both rows use the same
    # REF_LABEL/comparison_group categories) instead of two separately-tiled images, and without
    # the gray row-group rectangles. No x tick labels (show_xticklabels=False) -- the completeness
    # label annotated at the bottom of each subplot already states the group pairing in context,
    # and REF_LABEL/comparison_group are generic "S1"/"S4"-style names not meaningful on their own.
    #cw, ch = 2.0, 2.7
    fig, axs = plt.subplots(2, 2, figsize=(cw, ch), sharex=True)
    axs[0, 1].sharey(axs[0, 0])
    axs[1, 1].sharey(axs[1, 0])
    for ax in (axs[0, 1], axs[1, 1]):
        ax.tick_params(labelleft=False)  # sharey() (unlike sharey=True at creation) doesn't auto-hide these
    for row, (letter, lab, row_data) in enumerate(row_defs):
        pid, comp = row_data["Precursor_Id"], row_data.comparison_group
        ds, dstate, js, jstate, jlabel = stats_for(main_c, pid)
        plot_example_method(axs[row, 0], md, pid, group_cols, comp, M_D, ds, dstate, stat_fontsize=4.5,
                           completeness_label=lab, show_xticklabels=False)
        plot_example_method(axs[row, 1], mj, pid, group_cols, comp, M_J, js, jstate, stat_fontsize=4.5,
                           other_matrix=md, completeness_label=jlabel, show_xticklabels=False)
        axs[row, 0].set_ylabel("log$_2$ intensity")
        axs[row, 0].set_title(f"{pid} ({S.species_label(row_data.Species)}), {S.clabel(main_c)}",
                              loc="left", fontsize=5)
    compact_axes(*axs.ravel())
    # No per-column "DIA-NN"/"JUMPlion" header here -- method color is identified by
    # Main B's own legend elsewhere in the composite.
    fig.subplots_adjust(left=0.20, right=1.0, top=.90, bottom=INTEGRATED_BOTTOM, hspace=.56, wspace=.12)
    # tight=False: see panel_coverage's/panel_completeness_by_method's own matching note -- this
    # panel's saved pixel size must exactly match its own nominal figsize (cw, ch) for
    # Fig_Performance_Integrated's total width to be predictable from the nominal figsize constants
    # alone, the same as the composite's other four panels (A, B, D, E all already do this).
    sv(fig, L.main, "Fig_Performance_C_representative_examples", tight=False)

    # Expanded: one yeast and one E. coli example of each route per contrast (contrasts horizontally). Width capped at
    # <=7.2in (regular document width) regardless of how many contrasts there are -- columns just
    # get narrower (with smaller fonts below) rather than the figure growing unbounded with nc.
    # mode -> completeness label, same vocabulary as Main C's own row_defs (route identity doubles
    # as DIA-NN's completeness state for that route -- see generate_replicate_examples' own
    # docstring): "Restored -> TP" precursors were Insufficient for DIA-NN, "Partial FN -> TP"
    # ones were Partially complete.
    MODE_LABEL = {"Restored -> TP": "Insufficient", "Partial FN -> TP": "Partially complete"}
    contrasts = list(prepared)
    fig, axs = plt.subplots(4, 2 * len(contrasts), figsize=(8, 5), squeeze=False)
    for ci, c in enumerate(contrasts):
        sel = []
        for mode, pool in [("Partial FN -> TP", fnc), ("Restored -> TP", ntc)]:
            rows = pool[pool.contrast == c].sort_values("score")
            # one yeast and one E. coli example per route (best score within each species)
            for sp in ("YEAST", "ECOLI"):
                rs = rows[rows.Species == sp]
                sel.append((mode, rs.iloc[0] if len(rs) else None))
        for rr, (mode, row) in enumerate(sel):
            axD, axJ = axs[rr, 2 * ci], axs[rr, 2 * ci + 1]
            if row is None:
                axD.axis("off"); axJ.axis("off")
                continue
            pid, comp = row["Precursor_Id"], row.comparison_group
            ds, dstate, js, jstate, jlabel = stats_for(c, pid)
            # Same layout as Main C: other_matrix=md on the JUMPlion column (shared-vs-additional
            # point coloring, not a flat method color) and completeness_label on both (DIA-NN's own
            # route-level label here, JUMPlion's own computed label from stats_for).
            plot_example_method(axD, md, pid, group_cols, comp, M_D, ds, dstate, stat_fontsize=4.5,
                               completeness_label=MODE_LABEL[mode])
            plot_example_method(axJ, mj, pid, group_cols, comp, M_J, js, jstate, stat_fontsize=4.5,
                               other_matrix=md, completeness_label=jlabel)
            # Use only the axes that actually have finite data -- ax.get_ylim() falls back to
            # matplotlib's default (0, 1) range for an empty axes (e.g. DIA-NN has no
            # quantification at all for this precursor), which otherwise gets dragged into the
            # shared range and leaves the real data squashed into a sliver at the top.
            ranges = [r for r in (_axes_data_yrange(axD), _axes_data_yrange(axJ)) if r is not None]
            if ranges:
                ymin = min(r[0] for r in ranges); ymax = max(r[1] for r in ranges)
                pad = 0.08 * (ymax - ymin) if ymax > ymin else 0.5
                axD.set_ylim(ymin - pad, ymax + pad); axJ.set_ylim(ymin - pad, ymax + pad)
            axJ.tick_params(axis="y", labelleft=False)
            if ci == 0:
                axD.set_ylabel("log$_2$ intensity", fontsize=6.5)
            # Same title CONTENT/style as Main C's own axs[row, 0].set_title(...) -- pid, species,
            # contrast, left-aligned, fontsize=5 -- but still placed via ax.text (not set_title)
            # at this same tuned offset: unlike Main C, this panel also carries its own per-column
            # "DIA-NN"/"JUMPlion" headers (fig.text below, at a fixed y=.99) since there's no
            # neighboring composite panel's legend to borrow method-color identification from here
            # -- set_title() would land at nearly that same height for row 0 and collide with them
            # (see that block's own comment). Content dropped the old two-line "mode\npid" text --
            # mode is now redundant with the completeness_label shown at the bottom of each subplot.
            axD.text(-.02, 1.02, f"{pid} ({S.species_label(row.Species)}), {S.clabel(c)}",
                    transform=axD.transAxes, ha="left", va="bottom", fontsize=5)
    compact_axes(*axs.ravel())
    # Column headers (contrast + method) for row 0 only, placed at a fixed height near the figure
    # top -- using each axes' own set_title() put them at nearly the same height as the per-example
    # "mode / pid" text above (both anchored just above the axes), so they overlapped.
    fig.subplots_adjust(left=.065, right=.995, top=.915, bottom=.08, hspace=.65, wspace=.24)
    # Extra gap between contrasts (on top of the DIA-NN | JUMPlion wspace): shrink every column uniformly
    # to make room, then shift each contrast's pair of columns right by its index times the gap.
    contrast_gap = .03  # figure fraction
    if len(contrasts) > 1:
        left, right = .065, .995
        s = ((right - left) - contrast_gap * (len(contrasts) - 1)) / (right - left)
        for col in range(axs.shape[1]):
            for ax in axs[:, col]:
                p = ax.get_position()
                x0 = left + (p.x0 - left) * s + (col // 2) * contrast_gap
                ax.set_position([x0, p.y0, p.width * s, p.height])
    for col in range(axs.shape[1]):
        pos = axs[0, col].get_position()
        c = contrasts[col // 2]
        m_ = "DIA-NN" if col % 2 == 0 else "JUMPlion"
        fig.text((pos.x0 + pos.x1) / 2, .99, f"{S.clabel(c)}\n{m_}", ha="center", va="top", fontsize=5.5, fontweight="bold")
    # Dashed gray separators: between contrasts (vertical, clear of the next contrast's y tick labels) and between
    # the "Partially complete" (rows 0-1) and "Insufficient" (rows 2-3) rows (horizontal, just above the lower
    # rows' titles).
    from matplotlib.lines import Line2D
    sep = dict(color=S.LIGHT, linestyle="--", linewidth=.7, transform=fig.transFigure)
    y_top = .985
    y_bot = axs[-1, 0].get_position().y0 - .06
    for k in range(len(contrasts) - 1):
        x_prev = axs[0, 2 * k + 1].get_position().x1
        x_next = axs[0, 2 * k + 2].get_position().x0
        x_sep = (x_prev + x_next - .025) / 2
        fig.add_artist(Line2D([x_sep, x_sep], [y_bot, y_top], **sep))
    y_head = .943  # under the bold column headers, above row 0's per-example titles
    fig.add_artist(Line2D([.015, .995], [y_head, y_head], **sep))
    y_sep = axs[2, 0].get_position().y1 + .035
    fig.add_artist(Line2D([.015, .995], [y_sep, y_sep], **sep))  # starts at the y labels: a line further left would widen the tight crop
    sv(fig, L.supp, "Supp_Performance_E_representative_examples_expanded")
    return True


def hye_stack_bar_plot(ax, x, y_list, colors, width, label_size, sample_names, species,
                       x_label_offset=1, y_label_offset=5, text_color="white"):
    """Stacked HYE composition bar plot with value labels."""
    bottom = np.zeros(len(x))
    for i, y in enumerate(y_list):
        ax.bar(x, y, width, bottom=bottom, color=colors[i], label=species[i])
        for xi, (b, v) in enumerate(zip(bottom, y)):
            ax.text(xi, b + v / 2, str(v), fontsize=label_size, ha="center", va="center", color=text_color)
        bottom = bottom + y
    ax.set_xticks(x)
    ax.set_xticklabels([])
    for xi, total in enumerate(bottom):
        ax.text(xi, total + y_label_offset, sample_names[xi], fontsize=label_size + 1,
                ha="center", va="bottom")
    cum = 0
    for sp, y in zip(species, y_list):
        ax.text(-x_label_offset, cum + y[0] / 2, sp, fontsize=label_size + 1, ha="center", va="center")
        cum += y[0]


MAIN_PANEL_FILES_INTEGRATED = {
    "A": "Fig_Performance_A_coverage",
    "B": "Fig_Performance_B_completeness_by_method",
    "C": "Fig_Performance_C_representative_examples",
    "D": "Fig_Performance_D_sensitivity_fdr",
    "E": "Fig_Performance_E_quantitative_fidelity",
}


def assemble_integrated_figure(L, saver):
    """Fig_Performance_Integrated -- row 1: A (coverage curve) | B (completeness broken down by
    method+state) | C (representative examples); row 2: D (sensitivity/FDR) | E (quantitative
    fidelity). Each row is laid out on its own (own total width) rather than forced into a shared
    grid with the other row, and a narrower row is centered under/over the wider one -- there's no
    semantic reason one row's panels should share column boundaries with the other's. Requires
    "png" in the configured output formats; all 5 panels are Main panels. Lettered a-g: a-e one per
    panel (A-E), plus f/g for D's and E's own row 1 (restored population) -- see the sub_labels
    block below.

    Each panel's physical size is read back from its own saved PNG (pixel shape / saver.dpi)
    rather than from a hand-maintained size table, so this stays correct if any panel's figsize
    changes."""
    paths = {k: L.main / f"{stem}.png" for k, stem in MAIN_PANEL_FILES_INTEGRATED.items()}
    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        log.warning("assemble_integrated_figure skipped: missing panel PNG(s) for %s "
                    "(is \"png\" in the configured style.formats?)", missing)
        return

    imgs = {k: plt.imread(p) for k, p in paths.items()}
    w = {k: img.shape[1] / saver.dpi for k, img in imgs.items()}
    h = {k: img.shape[0] / saver.dpi for k, img in imgs.items()}

    rows = [["A", "B", "C"], ["D", "E"]]
    # Horizontal gap after each panel in a row EXCEPT the last (one value per adjacent pair, not
    # one per row) -- A-B sit close edge-to-edge fine, but B-C (completeness next to representative
    # examples) and D-F (sensitivity/FDR next to quantitative fidelity) need breathing room.
    row_gaps = [[0.15, 0.15], [0.15]]
    row_w = [sum(w[k] for k in row) + sum(gaps) for row, gaps in zip(rows, row_gaps)]
    row_h = [max(h[k] for k in row) for row in rows]
    total_w, total_h = max(row_w), sum(row_h)

    positions = {}
    y = 0.0
    for row, rw, rh, gaps in zip(rows, row_w, row_h, row_gaps):
        x = (total_w - rw) / 2  # center this row under/over the wider one
        for i, k in enumerate(row):
            positions[k] = (x, y, w[k], h[k])
            x += w[k] + (gaps[i] if i < len(gaps) else 0)
        y += rh

    fig = plt.figure(figsize=(total_w, total_h))
    for k, (x0_, y0_, bw, bh) in positions.items():
        ax = fig.add_axes([x0_ / total_w, 1 - (y0_ + bh) / total_h, bw / total_w, bh / total_h])
        ax.imshow(imgs[k])
        ax.axis("off")
    # Bold letter labels, one per panel -- inset slightly from each panel's own top-left corner
    # (computed from `positions`, not baked into each source panel, since this is the one place
    # that actually knows where every panel landed in the composite). Lettered by VISUAL reading
    # order (row-major: A,B,C,D,E -> a,b,c,d,e), not by each panel's own source letter, same
    # convention as generalization.py's own integrated composite. D and E each ALSO get a second
    # sub-label (f, g respectively) at their own row 1 (the restored population, below the
    # partially-complete row) -- derived from ROW2_TOP/ROW2_BOTTOM/ROW2_HSPACE's own GridSpec math
    # (same approach as generalization.py's own sub-panel labels), scaled by each panel's ACTUAL
    # placed height (read back from its saved PNG) so this stays correct even if
    # assemble_integrated_figure's own row-scaling (see this function's docstring) ever actually
    # does something.
    letters = {"A": "a", "B": "b", "C": "c", "D": "d", "E": "e"}
    label_pad = 0.06      # inches inset from each panel's top-left corner (A, B, C)
    label_pad_de = 0.03   # D/E's own top-level letter sits a bit higher (smaller inset) than the rest
    _row2_h = (ROW2_TOP - ROW2_BOTTOM) / (2 + ROW2_HSPACE)  # nrows=2
    _row1_top_frac_from_bottom = (ROW2_TOP - _row2_h) - ROW2_HSPACE * _row2_h
    dx, dy, _dbw, dbh = positions["D"]
    ex, ey, _ebw, ebh = positions["E"]
    sub_labels = [("f", dx, dy + dbh * (1 - _row1_top_frac_from_bottom - 0.1)),
                 ("g", ex, ey + ebh * (1 - _row1_top_frac_from_bottom - 0.1))]
    for k, (x0_, y0_, bw, bh) in positions.items():
        pad = label_pad_de if k in ("D", "E") else label_pad
        fig.text((x0_ + pad) / total_w, 1 - (y0_ + pad) / total_h, letters[k],
                 fontsize=9, fontweight="bold", ha="left", va="top")
    for letter, x0_, y0_ in sub_labels:
        fig.text((x0_ + label_pad) / total_w, 1 - (y0_ + label_pad) / total_h, letter,
                 fontsize=9, fontweight="bold", ha="left", va="top")
    saver.save(fig, L.main, "Fig_Performance_Integrated")


def assemble_supp_b_integrated(L, saver, gap=0.02):
    """Supp_Performance_B_integrated -- a: transition heatmaps (Supp_Performance_B0_heatmaps) |
    b: overall species composition (Supp_Performance_B1_species_composition_overall). Both panels
    keep their native size (so their fonts match), top-aligned, with sizes read back from each
    saved PNG. Requires "png" in the
    configured output formats."""
    stems = {"a": "Supp_Performance_B0_heatmaps", "b": "Supp_Performance_B1_species_composition_overall"}
    paths = {k: L.supp / f"{s}.png" for k, s in stems.items()}
    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        log.warning("assemble_supp_b_integrated skipped: missing panel PNG(s) for %s "
                    "(is \"png\" in the configured style.formats?)", missing)
        return
    imgs = {k: plt.imread(p) for k, p in paths.items()}
    w = {k: img.shape[1] / saver.dpi for k, img in imgs.items()}
    h = {k: img.shape[0] / saver.dpi for k, img in imgs.items()}
    # native size (no rescaling, so the fonts match); the shorter panel is top-aligned
    H = max(h.values())
    total_w = sum(w.values()) + gap
    fig = plt.figure(figsize=(total_w, H))
    x = 0.0
    for k in ("a", "b"):
        ax = fig.add_axes([x / total_w, 1 - h[k] / H, w[k] / total_w, h[k] / H])
        ax.imshow(imgs[k])
        ax.axis("off")
        fig.text((x + 0.06) / total_w, 1 - 0.06 / H, k, fontsize=9, fontweight="bold", ha="left", va="top")
        x += w[k] + gap
    saver.save(fig, L.supp, "Supp_Performance_B_integrated", tight=False)


# ----------------------------------------------------------------------------
# Main entry
# ----------------------------------------------------------------------------
def run(cfg, L, saver, resolve):
    global PADJ_CUT, ID_COL
    PADJ_CUT = float(cfg.get("padj_cutoff", 0.05))
    # Always the precursor-level default here -- this pipeline's own id column is never
    # configurable (see ID_COL's own module-level note); resetting it defensively means a prior
    # generalization() call earlier in the same process (which may have overridden it for a
    # protein-level run) can never leak into this one.
    ID_COL = "Precursor.Id"
    bench = cfg.get("benchmark", {})
    if bench.get("contrasts"):
        # The full Wang-HYE pipeline (Main C examples, the pooled intensity-bin setup, etc.) still
        # assumes one shared reference group across every contrast; only the standalone
        # `generalization` stage supports an explicit, non-star benchmark.contrasts pair list.
        raise KeyError("performance.run() does not support benchmark.contrasts (non-star pairs); "
                       "use the 'generalization' stage for those, or set benchmark.reference instead.")
    configure_benchmark(bench.get("groups"), bench.get("reference", "S1"))
    sv = saver.save
    T, MAIN, SUPP = L.table, L.main, L.supp

    # ---------- load DE tables + pre-DE tables ----------
    contrasts, prepared, preprepared = load_pairs_and_prepped(cfg, resolve)
    x = np.arange(len(contrasts))
    nc = len(contrasts)

    # ---------- global, common-universe intensity bin boundaries ----------
    # One set of 20/40/60/80th-percentile cutpoints from the pooled (deduplicated by Precursor.Id)
    # JUMPlion S1 reference intensity across the full common precursor universe, computed once and
    # reused unchanged for every analysis below -- so "intensity bin N" is the same absolute
    # log2-intensity range everywhere in this report, not a different range per contrast or population.
    pooled_ref = (pd.concat([pj0[["Precursor.Id", f"mean_log2_{REF}"]] for _, pj0 in preprepared.values()],
                            ignore_index=True)
                  .drop_duplicates("Precursor.Id")[f"mean_log2_{REF}"])
    bin_edges, bin_labels_display = quantile_bin_edges(pooled_ref)
    for m in (mm for _, _, mm, _ in prepared.values()):
        m["S1_bin"] = pd.cut(m["S1_intensity"], bin_edges, labels=BIN_LABELS, right=False)

    # ---------- exclusion audit ----------
    rows = []
    for item in cfg["comparisons"]:
        for kind, key in [("DE DIA-NN", "diann"), ("DE JUMPlion", "jumplion"),
                          ("pre-DE DIA-NN", "diann_pre_de"), ("pre-DE JUMPlion", "jumplion_pre_de")]:
            sp = pd.read_csv(item["_paths"][key], usecols=["Precursor.Id", "Species"])["Species"].map(norm_species)
            rows.append(dict(contrast=item["name"], source=kind, n_input=len(sp),
                             n_unambiguous=int(sp.isin(CLEAN).sum()),
                             n_excluded_ambiguous=int((sp == "AMBIGUOUS").sum())))
    write_tsv(pd.DataFrame(rows), T, "ambiguous_organism_exclusion_audit.tsv")

    # ======================= DIA-NN completeness universe (before/after JUMPlion) =======================
    # Per contrast: classify every clean precursor's DIA-NN completeness state from n_obs alone
    # (independent of whether JUMPlion touched it), then check whether JUMPlion's own n_obs
    # crosses the DE-eligibility threshold for precursors DIA-NN could not test (Insufficient).
    cov_rows, u_list, restored_rows = [], [], []
    for c, (d, j, m, other) in prepared.items():
        pd0, pj0 = preprepared[c]
        dd = pd0[["Precursor.Id", f"n_obs_{REF}", f"n_obs_{other}"]].rename(
            columns={f"n_obs_{REF}": "n1", f"n_obs_{other}": "n2"})
        jj = pj0[["Precursor.Id", "DE_eligible", f"n_obs_{REF}", f"n_obs_{other}",
                 f"mean_log2_{REF}", "Species_std"]].rename(
            columns={"DE_eligible": "testable_J", f"n_obs_{REF}": "n1_J", f"n_obs_{other}": "n2_J",
                    f"mean_log2_{REF}": "I_ref"})
        u = dd.merge(jj, on="Precursor.Id", how="outer")
        u["n1"] = u["n1"].fillna(0)
        u["n2"] = u["n2"].fillna(0)
        u["n1_J"] = u["n1_J"].fillna(0)
        u["n2_J"] = u["n2_J"].fillna(0)
        u["testable_J"] = u["testable_J"].fillna(False).astype(bool)
        u["completeness_state"] = classify_completeness(u["n1"], u["n2"])
        u["completeness_state_J"] = classify_completeness(u["n1_J"], u["n2_J"])
        u["bin"] = pd.cut(u["I_ref"], bin_edges, labels=BIN_LABELS, right=False)
        u["contrast"] = c
        u_list.append(u)
        for state in COMPLETENESS_ORDER:
            cov_rows.append(dict(contrast=c, state=state, n=int((u.completeness_state == state).sum())))
        ins = u[u.completeness_state == INSUFFICIENT].copy()
        ins["restoration"] = np.where(ins.testable_J, RESTORED, REMAINS_INSUFFICIENT)
        rec = (ins.loc[ins.restoration == RESTORED, ["contrast", "Precursor.Id", "n1", "n2", "bin"]]
               .merge(j[["Precursor.Id", "benchmark_state", "log2FC", "p_adj", "Species_std",
                        f"cv_{REF}", f"cv_{other}"]].rename(
                   columns={f"cv_{REF}": "cv_ref", f"cv_{other}": "cv_comparison"}),
                      on="Precursor.Id", how="left"))
        restored_rows.append(rec)

    u_all = pd.concat(u_list, ignore_index=True)
    write_tsv(u_all, T, "completeness_precursor_level.tsv.gz")
    coverage = pd.DataFrame(cov_rows)
    write_tsv(coverage, T, "completeness_overall.tsv")
    transition_rows = []
    for c in contrasts:
        zz = u_all[u_all.contrast == c]
        for base in COMPLETENESS_ORDER:
            g = zz[zz.completeness_state == base]
            n = len(g)
            for after in COMPLETENESS_ORDER:
                k = int((g.completeness_state_J == after).sum())
                transition_rows.append(dict(contrast=c, baseline_state=base, state_after_jumplion=after,
                                            n=k, pct=100 * k / n if n else np.nan))
    completeness_transition = pd.DataFrame(transition_rows)
    write_tsv(completeness_transition, T, "completeness_transition_by_contrast.tsv")
    insufficient_all = u_all[u_all.completeness_state == INSUFFICIENT].copy()
    insufficient_all["restoration"] = np.where(insufficient_all.testable_J, RESTORED, REMAINS_INSUFFICIENT)
    write_tsv(insufficient_all, T, "insufficient_precursors.tsv.gz")
    restored = pd.concat(restored_rows, ignore_index=True)
    write_tsv(restored, T, "restored_precursors.tsv.gz")
    restoration_rate = (insufficient_all.groupby("contrast")["restoration"]
                     .agg(n_insufficient="count", n_restored=lambda s: int((s == RESTORED).sum()))
                     .reset_index())
    restoration_rate["restoration_rate"] = restoration_rate["n_restored"] / restoration_rate["n_insufficient"]
    write_tsv(restoration_rate, T, "insufficient_restoration_rate.tsv")
    m_all = pd.concat([m for (_, _, m, _) in prepared.values()], ignore_index=True)
    write_tsv(m_all, T, "complete_partial_precursor_master.tsv.gz")

    # ---------- shared data for Main D/F (sensitivity, FDR) and Main E (CV, delta FC), computed
    # here so the two pairs of panels can be drawn with matching y-axis ranges ----------
    partial_perf, rec_perf, DF_SENS_YLIM, DF_FDR_YLIM = compute_sensitivity_fdr(contrasts, prepared, restored)
    write_tsv(partial_perf, T, "partial_sensitivity_fdr.tsv")
    write_tsv(rec_perf, T, "restored_sensitivity_fdr.tsv")

    partial_cv_rows, partial_fc_rows = [], []
    for c, (d, j, m, other) in prepared.items():
        g = m[m.completeness_state == PARTIAL]
        for method, suf in METHOD_SUFFIX:
            for grp in (REF, other):
                col = f"cv_{grp}_{suf}"
                if col not in g.columns:
                    continue
                cvs = pd.to_numeric(g[col], errors="coerce")
                frame = pd.DataFrame({"contrast": c, "method": method,
                                     "intensity_bin": g["S1_bin"].astype(str).values,
                                     "species": g[f"Species_std_{suf}"].values,
                                     "CV": cvs.values}).dropna(subset=["CV"])
                partial_cv_rows.append(frame[frame.species.isin(CLEAN)])
            for sp in CLEAN:
                z = g[g[f"Species_std_{suf}"] == sp]
                delta = pd.to_numeric(z[f"log2FC_{suf}"], errors="coerce") - EXPECTED[c][sp]
                ok = delta.notna()
                if ok.any():
                    partial_fc_rows.append(pd.DataFrame({"contrast": c, "method": method, "species": sp,
                                                         "intensity_bin": z.loc[ok, "S1_bin"].astype(str).values,
                                                         "delta_FC": delta[ok].values}))
    partial_cv = pd.concat(partial_cv_rows, ignore_index=True)
    write_tsv(partial_cv, T, "partial_CV_pooled_values.tsv")
    partial_fc = pd.concat(partial_fc_rows, ignore_index=True)
    write_tsv(partial_fc, T, "partial_delta_FC_precursor_level.tsv.gz")

    restored_cv_rows = []
    for col in ("cv_ref", "cv_comparison"):
        cvs = pd.to_numeric(restored[col], errors="coerce")
        frame = pd.DataFrame({"contrast": restored["contrast"].values, "species": restored["Species_std"].values,
                              "intensity_bin": restored["bin"].astype(str).values, "CV": cvs.values}).dropna(subset=["CV"])
        restored_cv_rows.append(frame[frame.species.isin(CLEAN)])
    restored_cv = pd.concat(restored_cv_rows, ignore_index=True)
    write_tsv(restored_cv, T, "restored_CV_pooled_values.tsv")

    restored_fc_rows = []
    for sp in CLEAN:
        z = restored[restored.Species_std == sp]
        if len(z):
            delta = pd.to_numeric(z["log2FC"], errors="coerce") - z["contrast"].map(lambda cc: EXPECTED[cc][sp])
            ok = delta.notna()
            if ok.any():
                restored_fc_rows.append(pd.DataFrame({"contrast": z.loc[ok, "contrast"].values, "species": sp,
                                                       "intensity_bin": z.loc[ok, "bin"].astype(str).values,
                                                       "delta_FC": delta[ok].values}))
    restored_fc = pd.concat(restored_fc_rows, ignore_index=True)
    write_tsv(restored_fc, T, "restored_delta_FC_precursor_level.tsv.gz")

    cv_box_groups = ([partial_cv[(partial_cv.species == sp) & (partial_cv.method == me)].CV.to_numpy(float) * 100
                      for sp in CLEAN for me in S.METHOD_ORDER] +
                     [restored_cv[restored_cv.species == sp].CV.to_numpy(float) * 100 for sp in CLEAN])
    fc_box_groups = ([partial_fc[(partial_fc.species == sp) & (partial_fc.method == me)].delta_FC.to_numpy(float)
                      for sp in CLEAN for me in S.METHOD_ORDER] +
                     [restored_fc[restored_fc.species == sp].delta_FC.to_numpy(float) for sp in CLEAN])
    E_CV_YLIM = _matched_box_ylim(cv_box_groups, floor_zero=True)
    E_FC_YLIM = _matched_box_ylim(fc_box_groups)

    # ======================= Supp A: benchmark design =======================
    # Bar plot keeps PANEL_A_SIZE's own geometry (left=.16, right=.92, bottom=.12, top=.75 of that
    # width); the expected-log2FC table sits to its right in extra figure width.
    table_w = 1.75
    pw, ph = PANEL_A_SIZE
    fig_w = pw + table_w
    fig = plt.figure(figsize=(fig_w, ph))
    ax = fig.add_axes([.16 * pw / fig_w, .12, (.92 - .16) * pw / fig_w, .75 - .12])
    labs = list(GROUPS)
    fmt = lambda v: int(v) if float(v).is_integer() else v
    hye_stack_bar_plot(ax, np.arange(len(labs)),
                       [np.array([fmt(MIX[l][k]) for l in labs]) for k in range(3)],
                       [S.SPECIES_COLORS[s] for s in S.SPECIES_ORDER], 0.8, 5,
                       labs, [S.species_label(s) for s in S.SPECIES_ORDER],
                       x_label_offset=1.15, y_label_offset=3)
    ax.set_yticks([]); ax.set_xlim(-1.65, len(labs) - 0.4); ax.set_ylim(0, 108)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(axis="both", length=0)
    ax.text(0.5, -0.08, f"{GROUP_SIZE} replicates per sample, 200 pg injection", transform=ax.transAxes,
            ha="center", va="top", fontsize=5)#, style="italic")
    # Expected log2FC per contrast (derived from the mixture design, EXPECTED), right of the bars.
    benchmark_design_table(fig, [(pw + .05) / fig_w, .12, (table_w - .1) / fig_w, .75 - .12], contrasts)
    # tight=False: a bbox crop would trim each design panel differently (by its widest label), so the
    # panels would not line up in Supp_Benchmark_design
    sv(fig, SUPP, "Supp_Performance_A_benchmark_design", tight=False)

    # ======================= Shared setup: completeness + coverage data (feeds Main A/B) =======================
    # 30-run completeness (method-agnostic coverage diagnostic) computed here for Main A.
    completeness, curve, n_runs, n_total, pad = compute_coverage_curve(contrasts, preprepared)
    write_tsv(completeness, T, "precursor_completeness_30runs.tsv.gz")
    write_tsv(curve, T, "precursor_completeness_cumulative_curve.tsv")

    # Left-to-right: 30-run cumulative coverage | DIA-NN completeness | per-contrast state-transition heatmaps.
    # Heatmaps stack one per row (not a 2x2 grid): each gets the full column width, and tick/axis
    # labels are drawn only on the bottom-most one (they're identical across contrasts) to avoid
    # the label collisions a compact 2x2 grid produces.
    heat_rows = nc

    # ======================= Main B: completeness, by method =======================
    panel_completeness_by_method(contrasts, coverage, completeness_transition, MAIN, sv)

    # Complete always transitions 100% -> Complete (trivial), so only Partial/Insufficient are
    # shown as baseline rows; all three states remain as "after JUMPlion" columns.
    short = {COMPLETE: "Complete", PARTIAL: "Partial", INSUFFICIENT: "Insuff."}
    heat_row_states = [PARTIAL, INSUFFICIENT]
    row_labels = [short[s] for s in heat_row_states]
    col_labels = [short[s] for s in COMPLETENESS_ORDER]
    fig_heat = plt.figure(figsize=PANEL_B0_SIZE)
    axd_h = fig_heat.subplot_mosaic([[f"H{k}"] for k in range(heat_rows)])
    for k, c in enumerate(contrasts):
        zz = completeness_transition[completeness_transition.contrast == c]
        mat = (zz.pivot(index="baseline_state", columns="state_after_jumplion", values="pct")
               .reindex(index=heat_row_states, columns=COMPLETENESS_ORDER).to_numpy(float))
        ax_h = axd_h[f"H{k}"]
        S.heatmap(ax_h, mat, row_labels, col_labels, vmin=0, vmax=100, cmap="Blues",
                  annotate=True, fmt="{:.1f}%", colorbar=False, annot_size=5, tick_size=5,
                  xlabel_rotation=0, xlabel_ha="center")
        ax_h.set_title(S.clabel(c), fontsize=5, pad=2)
        # n= = number of DIA-NN precursors in each baseline-state row (sum over after-states)
        n_row = zz.groupby("baseline_state")["n"].sum().reindex(heat_row_states)
        for i, s in enumerate(heat_row_states):
            ax_h.text(1.03, i, f"n={int(n_row[s]):,}", transform=ax_h.get_yaxis_transform(),
                      ha="left", va="center", fontsize=5)
        if k == nc - 1:  # bottom-most: the only one that also keeps the column (x) tick labels
            ax_h.set_xlabel("JUMPlion", fontsize=5.5)
        else:
            ax_h.set_xticklabels([])
    # tight_layout() treats each of the 4 "H" rows independently and resizes them to fit their own
    # tick labels -- since only the bottom-most keeps labels (the other 3 are blank), it shrinks
    # the labeled one far more than the others, leaving the heatmap column squashed into the top
    # with dead space below. subplots_adjust uses the GridSpec's own equal-height row split
    # instead, so all 4 heatmaps actually occupy 1/4 of the figure each.
    fig_heat.subplots_adjust(left=.18, right=.80, top=.9, bottom=.13, hspace=.55)
    # Shared "DIA-NN baseline state" label, centered on the whole stack of heatmaps rather than
    # tied to one individual (now that every heatmap keeps its own Partial/Insuff. row labels).
    # Positions (and this text call) are read back after fig_heat.subplots_adjust so they reflect
    # the final layout.
    top_pos, bot_pos = axd_h["H0"].get_position(), axd_h[f"H{nc - 1}"].get_position()
    fig_heat.text(top_pos.x0 - .15, (top_pos.y1 + bot_pos.y0) / 2, "DIA-NN",
                 ha="center", va="center", rotation=90, fontsize=5.5)
    sv(fig_heat, SUPP, "Supp_Performance_B0_heatmaps", tight=False)  # exact size: tiled in Supp_Performance_B_integrated

    # ======================= Main A: 30-run cumulative coverage =======================
    panel_coverage(curve, n_runs, n_total, pad, MAIN, sv)

    # ======================= Supp B2: completeness + species composition by intensity bin =======================
    # Row 1: completeness-state composition by intensity bin. Rows 2-3: number/percentage of
    # Human/Yeast/E. coli precursors per contrast x intensity bin, for the two populations where
    # species-balanced (not raw) FDR is used (Supp C, Supp D) because HYE composition itself can
    # shift across bins -- these rows show that shift directly.
    comp_rows = []
    for c, (d, j, m, other) in prepared.items():
        g = m[m.completeness_state == PARTIAL]
        for b, gg in g.groupby("S1_bin", observed=True):
            tot = len(gg)
            for sp in CLEAN:
                n = int((gg["Species_std_J"] == sp).sum())
                comp_rows.append(dict(contrast=c, population="Partially complete", intensity_bin=str(b),
                                      species=sp, n=n, pct=100 * n / tot if tot else np.nan))
        rr = restored[restored.contrast == c]
        for b, gg in rr.groupby("bin", observed=True):
            tot = len(gg)
            for sp in CLEAN:
                n = int((gg["Species_std"] == sp).sum())
                comp_rows.append(dict(contrast=c, population="Insufficient (restored)", intensity_bin=str(b),
                                      species=sp, n=n, pct=100 * n / tot if tot else np.nan))
    species_composition = pd.DataFrame(comp_rows)
    write_tsv(species_composition, T, "species_composition_by_intensity.tsv")

    pops = ["Partially complete", "Insufficient (restored)"]
    # Width capped at <=7.2in (regular document width); height scaled to match so the panel
    # keeps its original aspect ratio rather than being stretched/squashed.
    fig, axs = plt.subplots(3, nc, figsize=(7.0, 9.2 * 7.0 / 10.6), sharey="row", squeeze=False)
    for col, c in enumerate(contrasts):
        ax0 = axs[0, col]
        zz = u_all[u_all.contrast == c]
        bottom = np.zeros(len(BIN_LABELS))
        for state in COMPLETENESS_ORDER:
            counts = (zz[zz.completeness_state == state].groupby("bin", observed=True).size()
                      .reindex(BIN_LABELS).fillna(0).to_numpy(float))
            ax0.bar(range(len(BIN_LABELS)), counts, bottom=bottom, color=S.COMPLETENESS_COLORS[state], label=state)
            bottom += counts
        ax0.set_title(S.clabel(c), fontsize=6.5); bin_ticks(ax0, bin_labels_display)
    axs[0, 0].set_ylabel("Precursors (DIA-NN,\nbefore JUMPlion)", fontsize=6.5)
    completeness_handles, completeness_labels = axs[0, 0].get_legend_handles_labels()

    for row, pop in enumerate(pops, start=1):
        for col, c in enumerate(contrasts):
            ax = axs[row, col]
            zz = species_composition[(species_composition.population == pop) & (species_composition.contrast == c)]
            bottom = np.zeros(len(BIN_LABELS))
            for sp in S.SPECIES_ORDER:
                vals = (zz[zz.species == sp].set_index("intensity_bin").reindex(BIN_LABELS)["pct"]
                        .fillna(0).to_numpy(float))
                ax.bar(range(len(BIN_LABELS)), vals, bottom=bottom, color=S.SPECIES_COLORS[sp],
                       label=S.species_label(sp))
                bottom += vals
            totals = zz.groupby("intensity_bin")["n"].sum().reindex(BIN_LABELS).fillna(0).to_numpy(float)
            for i, t in enumerate(totals):
                ax.text(i, 103, f"n={int(t)}", ha="center", va="bottom", fontsize=4.2, rotation=90)
            bin_ticks(ax, bin_labels_display)
            ax.set_ylim(0, 118)
        axs[row, 0].set_ylabel(f"{pop}\ncomposition (%)", fontsize=6.5)
    compact_axes(*axs.ravel())
    fig.supxlabel(INT_XLABEL, y=.01, fontsize=6.5)
    S.fig_legend(fig, completeness_handles, completeness_labels, ncol=3, y=1.06, fontsize=5.5)
    S.fig_legend(fig, *axs[1, 0].get_legend_handles_labels(), ncol=3, y=1.03, fontsize=5.5)
    fig.subplots_adjust(left=.10, right=.95, bottom=.10, top=.96, wspace=.18, hspace=.4)
    sv(fig, SUPP, "Supp_Performance_B2_completeness_and_species_composition_by_intensity")

    # Supp B1: overall HYE composition per contrast, Partial vs Restored panels
    overall_comp_rows = []
    for c, (d, j, m, other) in prepared.items():
        g = m[m.completeness_state == PARTIAL]
        tot = len(g)
        for sp in CLEAN:
            n = int((g["Species_std_J"] == sp).sum())
            overall_comp_rows.append(dict(contrast=c, population="Partially complete", species=sp,
                                          n=n, pct=100 * n / tot if tot else np.nan))
        rr = restored[restored.contrast == c]
        tot = len(rr)
        for sp in CLEAN:
            n = int((rr["Species_std"] == sp).sum())
            overall_comp_rows.append(dict(contrast=c, population="Insufficient (restored)", species=sp,
                                          n=n, pct=100 * n / tot if tot else np.nan))
    overall_composition = pd.DataFrame(overall_comp_rows)
    write_tsv(overall_composition, T, "species_composition_overall.tsv")

    # Width capped at <=7.2in (regular document width); height scaled to keep the aspect ratio.
    fig, axs = plt.subplots(1, 2, figsize=(3.6, PANEL_B0_SIZE[1]))
    panels = [("Partially complete", "Partially complete"),
             ("Insufficient (restored)", "Insufficient → restored")]
    for ax, (pop, caption) in zip(axs, panels):
        bottom = np.zeros(nc)
        for sp in S.SPECIES_ORDER:
            vals = np.array([
                overall_composition[(overall_composition.contrast == c) & (overall_composition.population == pop) &
                                    (overall_composition.species == sp)].pct.iloc[0]
                for c in contrasts])
            ax.bar(x, vals, .6, bottom=bottom, color=S.SPECIES_COLORS[sp], label=S.species_label(sp))
            bottom += vals
        totals = np.array([
            overall_composition[(overall_composition.contrast == c) & (overall_composition.population == pop)].n.sum()
            for c in contrasts])
        for i, t in enumerate(totals):
            ax.text(x[i], 102, f"n={int(t)}", ha="center", va="bottom", fontsize=5)
        contrast_ticks(ax, contrasts)
        ax.set_ylim(0, 100)  # the n= labels now sit above the axes, so the title is lifted clear of them
        ax.set_title(caption, fontsize=5, pad=10)
    axs[0].set_ylabel("Composition (%)")
    # same text sizes as the B0 heatmap panel it is tiled with in Supp_Performance_B_integrated (ticks/n 5, axis labels 5.5)
    compact_axes(*axs, labelsize=5)
    S.fig_legend(fig, *axs[0].get_legend_handles_labels(), ncol=3, y=.99, fontsize=5)
    fig.tight_layout(rect=[0, .02, 1, .95])  # .05: extra bottom margin
    sv(fig, SUPP, "Supp_Performance_B1_species_composition_overall", tight=False)  # exact size, as B0
    assemble_supp_b_integrated(L, saver)

    # Standalone Fig_Performance_D_partial_sensitivity_fdr was removed -- DF combined (below) is
    # the only sensitivity/FDR panel for the partial-completeness stratum now. partial_perf
    # (computed above) is shared with Main E's matched y-axis range.

    # Data for Supp C (species-balanced sensitivity/FDR rows): balanced (not pooled) here because
    # HYE composition itself shifts sharply across intensity bins (Supp B2) -- e.g. Yeast:E.coli
    # among changing species can go from ~94:6 in the lowest bin to ~25:75 in the highest for the
    # same contrast, and yeast/E.coli sensitivity can differ substantially, so a pooled rate here
    # would really be tracking the composition shift, not a robustness trend. Each species'
    # contribution requires its own minimum denominator (MIN_N_SPECIES_BALANCED), same rule as
    # balanced FDR. Plotted below as Supp C, combined with the CV/ΔFC rows computed in Main E.
    partial_bybin = []
    for c, (d, j, m, other) in prepared.items():
        g = m[m.completeness_state == PARTIAL]
        for method, suf in METHOD_SUFFIX:
            temp = pd.DataFrame({"benchmark_state": g[f"benchmark_state_{suf}"],
                                 "Species_std": g[f"Species_std_{suf}"], "bin": g["S1_bin"]})
            for b, idx in temp.groupby("bin", observed=True).groups.items():
                sub = temp.loc[idx]
                row = metric_rows(sub, c, method, "intensity", str(b))
                _, _, _, _, yr, er, bf = balanced(sub, "Species_std", "benchmark_state")
                row["species_balanced_sensitivity"] = (yr + er) / 2
                row["species_balanced_fdr"] = bf
                partial_bybin.append(row)
    partial_bybin = pd.DataFrame(partial_bybin)
    write_tsv(partial_bybin, T, "partial_sensitivity_fdr_by_intensity.tsv")

    # Standalone Fig_Performance_E_partial_quantitative_fidelity was removed -- panel_cv_deltafc
    # (below, Main E) is the only CV/delta-FC panel for the partial-completeness stratum now.

    # ======================= Supp C: Partially complete -- sensitivity/FDR + CV/ΔFC by intensity =======================
    # Rows 1-2 reuse partial_bybin (species-balanced, computed above); rows 3-4 reuse partial_cv /
    # partial_fc (computed in Main E) -- same underlying data as the former Supp D0 + Supp E0,
    # combined into one figure.
    # Width capped at <=7.2in (regular document width); height scaled to keep the aspect ratio.
    fig, axs = plt.subplots(4, nc, figsize=(7.0, 11.4 * 7.0 / 10.5), sharey="row", squeeze=False)
    for col, c in enumerate(contrasts):
        for method in S.METHOD_ORDER:
            y = (partial_bybin[(partial_bybin.contrast == c) & (partial_bybin.method == method)]
                 .set_index("stratum_value").reindex(BIN_LABELS))
            axs[0, col].plot(range(len(BIN_LABELS)), y.species_balanced_sensitivity * 100, marker="o", ms=3, lw=1, color=S.method_color(method), label=method)
            axs[1, col].plot(range(len(BIN_LABELS)), y.species_balanced_fdr * 100, marker="o", ms=3, lw=1, color=S.method_color(method), label=method)
        axs[0, col].set_title(S.clabel(c), fontsize=6.5)
        bin_ticks(axs[0, col], bin_labels_display); bin_ticks(axs[1, col], bin_labels_display)

        ax = axs[2, col]
        zz = partial_cv[partial_cv.contrast == c]
        pos, data, cat = [], [], []
        for i, b in enumerate(BIN_LABELS):
            for off, method in [(-.18, M_D), (.18, M_J)]:
                pos.append(i + off); cat.append(method)
                data.append(zz[(zz.intensity_bin == b) & (zz.method == method)].CV.to_numpy(float) * 100)
        S.boxplot_groups(ax, data, pos, cat, S.METHOD_COLORS)
        bin_ticks(ax, bin_labels_display)

        ax2 = axs[3, col]
        zz2 = partial_fc[partial_fc.contrast == c]
        pos, data, cat = [], [], []
        for i, b in enumerate(BIN_LABELS):
            for off, method in [(-.18, M_D), (.18, M_J)]:
                pos.append(i + off); cat.append(method)
                data.append(zz2[(zz2.intensity_bin == b) & (zz2.method == method)].delta_FC.to_numpy(float))
        S.boxplot_groups(ax2, data, pos, cat, S.METHOD_COLORS)
        S.ref_line(ax2, 0)
        bin_ticks(ax2, bin_labels_display)
    axs[0, 0].set_ylabel("Species-balanced sensitivity (%)", fontsize=6.5); axs[1, 0].set_ylabel("Species-balanced FDR (%)", fontsize=6.5)
    axs[2, 0].set_ylabel("CV (%)", fontsize=6.5); axs[3, 0].set_ylabel(r"$\Delta$FC", fontsize=6.5)
    compact_axes(*axs.ravel())
    fig.supxlabel(INT_XLABEL, y=.005, fontsize=6.5)
    S.fig_legend(fig, *axs[0, 0].get_legend_handles_labels(), ncol=2, y=1.015, fontsize=5.5)
    # bottom needs enough room for the 45-deg-rotated bin labels on the last row, or they collide
    # with the supxlabel right below them.
    fig.subplots_adjust(left=.08, right=.99, bottom=.08, top=.95, wspace=.18, hspace=.40)
    sv(fig, SUPP, "Supp_Performance_C_partial_sensitivity_fdr_CV_deltaFC_by_intensity")

    # ======================= Main D: sensitivity/FDR =======================
    panel_sensitivity_fdr(contrasts, partial_perf, rec_perf, DF_SENS_YLIM, DF_FDR_YLIM, MAIN, sv)

    # Data for Supp D (species-balanced sensitivity/FDR rows) -- same composition-confound
    # reasoning as Supp C (the restored-Insufficient population shows similarly large
    # Yeast:E.coli shifts across bins). Plotted below as Supp D, combined with the CV/ΔFC rows
    # computed above (shared with Main E).
    rec_bybin = []
    for c in contrasts:
        rr = restored[restored.contrast == c]
        for b, g in rr.groupby("bin", observed=True):
            ch = g[g.Species_std.isin(["YEAST", "ECOLI"])]
            tp = int((ch.benchmark_state == "TP").sum())
            den = int(ch.benchmark_state.isin(["TP", "FN", "wrong_direction"]).sum())
            _, _, _, _, yr, er, bf = balanced(g, "Species_std", "benchmark_state")
            rec_bybin.append(dict(contrast=c, intensity_bin=str(b), n=den,
                                  pooled_sensitivity=tp / den if den >= MIN_N_SENSITIVITY else np.nan,
                                  species_balanced_sensitivity=(yr + er) / 2,
                                  species_balanced_fdr=bf))
    rec_bybin = pd.DataFrame(rec_bybin)
    write_tsv(rec_bybin, T, "restored_sensitivity_fdr_by_intensity.tsv")

    # Standalone Fig_Performance_G_restored_quantitative_fidelity was removed -- Main E combined
    # (below) is the only CV/delta-FC panel for the restored-Insufficient population now.
    # restored_cv/restored_fc (computed above) are reused there (shared with Main E's matched
    # y-axis range). Restored precursors have no DIA-NN counterpart (DIA-NN was Insufficient), so
    # that row is JUMPlion-only -- the same CV / Delta-FC question as the partially-complete row,
    # asked of the more severely incomplete population.

    # ======================= Main E: quantitative fidelity =======================
    panel_cv_deltafc(partial_cv, partial_fc, restored_cv, restored_fc, E_CV_YLIM, E_FC_YLIM, MAIN, sv)

    # ======================= Supp D: Insufficient -- sensitivity/FDR + CV/ΔFC of restored precursors, by intensity =======================
    # Rows 1-2 reuse rec_bybin (species-balanced, computed above); rows 3-4 reuse restored_cv /
    # restored_fc (computed above) -- same underlying data as the former Supp F0 + Supp G0,
    # combined into one figure.
    # Width capped at <=7.2in (regular document width); height scaled to keep the aspect ratio.
    fig, axs = plt.subplots(4, nc, figsize=(7.0, 11.4 * 7.0 / 10.3), sharey="row", squeeze=False)
    for col, c in enumerate(contrasts):
        y = rec_bybin[rec_bybin.contrast == c].set_index("intensity_bin").reindex(BIN_LABELS)
        axs[0, col].plot(range(len(BIN_LABELS)), y.species_balanced_sensitivity * 100, marker="o", ms=3, lw=1, color=S.method_color(M_J))
        axs[0, col].set_title(S.clabel(c), fontsize=6.5)
        axs[1, col].plot(range(len(BIN_LABELS)), y.species_balanced_fdr * 100, marker="o", ms=3, lw=1, color=S.method_color(M_J))
        bin_ticks(axs[0, col], bin_labels_display); bin_ticks(axs[1, col], bin_labels_display)

        ax = axs[2, col]
        zz = restored_cv[restored_cv.contrast == c]
        data = [zz[zz.intensity_bin == b].CV.to_numpy(float) * 100 for b in BIN_LABELS]
        S.boxplot_groups(ax, data, list(range(len(BIN_LABELS))), [M_J] * len(BIN_LABELS), S.METHOD_COLORS, width=.5)
        bin_ticks(ax, bin_labels_display)

        ax2 = axs[3, col]
        zz2 = restored_fc[restored_fc.contrast == c]
        data2 = [zz2[zz2.intensity_bin == b].delta_FC.to_numpy(float) for b in BIN_LABELS]
        S.boxplot_groups(ax2, data2, list(range(len(BIN_LABELS))), [M_J] * len(BIN_LABELS), S.METHOD_COLORS, width=.5)
        S.ref_line(ax2, 0)
        bin_ticks(ax2, bin_labels_display)
    axs[0, 0].set_ylabel("Species-balanced sensitivity (%)", fontsize=6.5); axs[1, 0].set_ylabel("Species-balanced FDR (%)", fontsize=6.5)
    axs[2, 0].set_ylabel("CV (%)", fontsize=6.5); axs[3, 0].set_ylabel("log$_2$FC error", fontsize=6.5)
    compact_axes(*axs.ravel())
    fig.supxlabel(INT_XLABEL, y=.005, fontsize=6.5)
    # Only one method (restored precursors have no DIA-NN counterpart), but still label its color.
    S.fig_legend(fig, *S.line_handles(S.METHOD_COLORS, [M_J]), ncol=1, y=1.015, fontsize=5.5)
    # bottom needs enough room for the 45-deg-rotated bin labels on the last row, or they collide
    # with the supxlabel right below them.
    fig.subplots_adjust(left=.08, right=.99, bottom=.08, top=.95, wspace=.18, hspace=.40)
    sv(fig, SUPP, "Supp_Performance_D_restored_sensitivity_fdr_CV_deltaFC_by_intensity")

    # ---------- representative examples (Main C / Supp E) ----------
    generate_replicate_examples(cfg, resolve, prepared, restored, L, sv)

    # ---------- composite: row 1 A (coverage), B (completeness by method), C (examples); row 2 D (sensitivity/FDR), F (quantitative fidelity) ----------
    assemble_integrated_figure(L, saver)

    # ---------- figure plan + master table ----------
    plan = [
        ["Main A", "30-run cumulative coverage", "All clean single-organism precursors"],
        ["Main B", "DIA-NN/JUMPlion completeness, by method and state", "All clean single-organism precursors"],
        ["Main C", "Representative examples (curated): Partial FN->TP and Insufficient->restored->TP", "Partial FN->TP and Insufficient->restored->TP"],
        ["Main D", "Sensitivity + overall FDR: partially complete, and restored (insufficient)", "Partially complete; Insufficient (restored)"],
        ["Main E", "Quantitative fidelity (CV, ΔFC): partially complete, and restored (insufficient)", "Partially complete; Insufficient (restored)"],
        ["Supp A", "Benchmark design", "All samples"],
        ["Supp B1", "Species composition (n, %) overall per contrast", "Partially complete and Insufficient (restored)"],
        ["Supp B2", "Completeness composition + species composition (n, %) by intensity", "All clean single-organism precursors; Partially complete and Insufficient (restored)"],
        ["Supp C", "Partial species-balanced sensitivity + FDR, and CV / ΔFC, by intensity", "Partially complete"],
        ["Supp D", "Restored species-balanced sensitivity + FDR, and CV / ΔFC, by intensity", "Insufficient (restored)"],
        ["Supp E", "Representative examples (expanded)", "Partial FN->TP and Insufficient->restored->TP"],
    ]
    write_tsv(pd.DataFrame(plan, columns=["panel", "analysis", "precursor_universe"]), T, "final_figure_plan.tsv")
    log.info("Performance analysis finished.")
