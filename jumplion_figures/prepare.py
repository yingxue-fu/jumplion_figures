"""Data preparation used by the pipeline stages.

Pipeline stages (3):
  differential_expression      precursor matrices + sample metadata + FASTA
                               --diff_benchmark_analysis()--> <name>_dea_results/ DE + pre-DE tables
  performance                  uses the DE + pre-DE tables
  mechanism                    FIRST builds precursor-run features
                               (xic_quantification.parquet --add_precursor_run_features()-->
                               mechanism/table/precursor_run_features.parquet),
                               then runs its analyses (localization-tier recovery, intensity
                               mapping, final-recovered) on them, sharing one load of the file

Prepared outputs are reused when they already exist (set the matching "overwrite" key to
rebuild); a warning is printed if an input is newer than the output being reused.
"""
from __future__ import annotations

import importlib
import logging
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

log = logging.getLogger("jumplion_figures.prepare")

DEFAULT_FEATURE_FN = "jumplion_figures.precursor_features:add_precursor_run_features"
DEFAULT_DE_FN = "jumplion_figures.diff_expression:diff_benchmark_analysis"
DEFAULT_GROUPS = {"S1": "H50Y10E40", "S2": "H50Y20E30", "S3": "H50Y25E25", "S4": "H50Y30E20", "S5": "H50Y40E10"}
FEATURES_NAME = "precursor_run_features.parquet"


def load_callable(spec):
    """'package.module:function' -> function."""
    mod_name, _, fn_name = spec.partition(":")
    try:
        mod = importlib.import_module(mod_name)
    except ImportError as exc:
        raise ImportError(f"Cannot import '{mod_name}' ({exc}).") from exc
    return getattr(mod, fn_name)


def _resolve_matrices(de):
    """differential_expression.matrices.<role> is either the plain {"name", "file"} it's always
    been (used as-is, "normalization" has no effect on it), or nested {"normalized": {...},
    "raw": {...}} -- for a role defined both ways, "normalization" (default "normalized") picks
    which {"name", "file"} pair to use, so the rest of the pipeline (DE, performance, mechanism)
    never has to know the two quantifications exist; it just sees one flat matrices dict, as
    before."""
    mode = de.get("normalization", "normalized")
    if mode not in ("normalized", "raw"):
        raise KeyError(f'differential_expression.normalization must be "normalized" or "raw", got {mode!r}.')
    matrices = {}
    for role, spec in de.get("matrices", {}).items():
        if isinstance(spec.get(mode), dict):
            matrices[role] = spec[mode]
        elif "file" in spec:
            matrices[role] = spec
        else:
            raise KeyError(f'differential_expression.matrices.{role} has no "{mode}" entry '
                           f'and no top-level "file".')
    return matrices


def _stage_dir(out_root, stage, dataset):
    """<out_root>/<stage>, or <out_root>/<stage>/<dataset> when `dataset` is given -- stage first,
    dataset second, matching OutputLayout (see common.py) so every stage groups all datasets'
    output together instead of the other way around."""
    d = Path(out_root) / stage
    return d / dataset if dataset else d


class Context:
    """Paths shared by all stages, derived from the one config.

    benchmark.contrasts (an explicit [[g1, g2], ...] group-label pair list), when present,
    overrides benchmark.reference -- for a benchmark design that isn't a star around one shared
    reference (e.g. a path: A_vs_B, B_vs_C, where no single group is common to every contrast in
    the same role). Either way, `contrast_pairs` is the one list everything downstream (run_de,
    split_shared_new, apply_derived_inputs) iterates: in reference mode it's built as
    [(label, reference) for label in labels] -- same contrasts, same names, as the old
    reference-only design, just expressed as the general (g1, g2) form so there's one code path
    instead of two. `reference`/`labels` stay available (as None in contrasts mode) only because
    a few call sites outside this class still read them directly."""

    def __init__(self, cfg, resolve, out_root, dataset=None):
        self.cfg, self.resolve, self.out_root, self.dataset = cfg, resolve, Path(out_root), dataset
        bench = cfg.get("benchmark", {})
        self.groups = dict(bench.get("groups", DEFAULT_GROUPS))
        explicit = bench.get("contrasts")
        if explicit:
            self.reference = None
            self.labels = None
            self.contrast_pairs = [tuple(p) for p in explicit]
            bad = [p for p in self.contrast_pairs if p[0] not in self.groups or p[1] not in self.groups]
            if bad:
                raise KeyError(f"benchmark.contrasts pair(s) {bad} reference a group not in benchmark.groups")
        else:
            self.reference = bench.get("reference", "S1")
            if self.reference not in self.groups:
                raise KeyError(f"benchmark.reference '{self.reference}' not in benchmark.groups")
            self.labels = [g for g in self.groups if g != self.reference]
            self.contrast_pairs = [(lab, self.reference) for lab in self.labels]
        self.sig = cfg.get("mechanism", {})
        de = cfg.get("differential_expression", {})
        self.features_file = (resolve(self.sig.get("features_file"))
                              or _stage_dir(self.out_root, "mechanism", dataset) / "table" / FEATURES_NAME)
        self.de_dir = (resolve(de.get("output_dir"))
                       or _stage_dir(self.out_root, "differential_expression", dataset))
        self.matrices = _resolve_matrices(de)

    def contrast(self, g1, g2):
        return f"{g1}_vs_{g2}"

    def pair(self, g1, g2):
        return self.groups[g1], self.groups[g2]

    def de_file(self, role, g1, g2, pre=False):
        name = self.matrices[role]["name"]
        a, b = self.pair(g1, g2)
        return self.de_dir / f"{name}_dea_results" / f"{name}_{a}_vs_{b}{'_pre_DE' if pre else ''}.csv"


def _warn_if_stale(inputs, outputs, key):
    ins = [Path(p) for p in inputs if p is not None and Path(p).exists()]
    outs = [Path(p) for p in outputs if Path(p).exists()]
    if ins and outs and max(p.stat().st_mtime for p in ins) > min(p.stat().st_mtime for p in outs):
        log.warning("Inputs are NEWER than the reused outputs; set %s=true to rebuild.", key)


# ----------------------------------------------------------------------------
# Precursor-run features (first step of mechanism)
# ----------------------------------------------------------------------------
def ensure_features(ctx):
    """Build the precursor-run feature table if needed; return its path."""
    sec, out = ctx.sig, ctx.features_file
    src = ctx.resolve(sec.get("xic_quant_file"))
    if out.exists() and not sec.get("overwrite_features", False):
        log.info("Precursor-run features exist, reusing %s", out)
        _warn_if_stale([src], [out], "mechanism.overwrite_features")
        return out
    if src is None:
        raise KeyError("mechanism.xic_quant_file is required to build precursor-run features.")
    fn = load_callable(sec.get("features_function", DEFAULT_FEATURE_FN))
    log.info("Building precursor-run features from %s", src)
    df = fn(pd.read_parquet(src), **sec.get("features_kwargs", {}))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    tmp.replace(out)  # atomic: a crashed run never leaves a half-written file behind
    log.info("Wrote %s (%s rows, %s columns)", out, f"{len(df):,}", df.shape[1])
    return out


# ----------------------------------------------------------------------------
# Stage: differential_expression
# ----------------------------------------------------------------------------
def run_de(section, ctx):
    roles = ["diann", "jumplion"]
    for r in roles:
        if r not in ctx.matrices or not {"name", "file"} <= set(ctx.matrices[r]):
            raise KeyError(f"differential_expression.matrices.{r} must define 'name' and 'file'.")
    expected = [ctx.de_file(r, g1, g2, pre) for r in roles for (g1, g2) in ctx.contrast_pairs for pre in (False, True)]
    if all(p.exists() for p in expected) and not section.get("overwrite", False):
        log.info("DE tables exist in %s, reusing (set differential_expression.overwrite=true to rebuild).", ctx.de_dir)
        _warn_if_stale([ctx.resolve(ctx.matrices[r]["file"]) for r in roles] +
                       [ctx.resolve(section.get("sample_meta_file"))], expected,
                       "differential_expression.overwrite")
    else:
        _compute_de(section, ctx, roles, expected)
    split_shared_new(section, ctx)


def _compute_de(section, ctx, roles, expected):
    fn = load_callable(section.get("function", DEFAULT_DE_FN))
    ctx.de_dir.mkdir(parents=True, exist_ok=True)
    fasta = section.get("fasta_files")
    kwargs = dict(
        mat_files=[ctx.resolve(ctx.matrices[r]["file"]) for r in roles],
        mat_names=[ctx.matrices[r]["name"] for r in roles],
        mat_type=section.get("mat_type", "Precursor.Id"),
        mat_sep=section.get("mat_sep", "\t"),
        log_transform_input=bool(section.get("log_transform_input", True)),
        sample_meta_file=ctx.resolve(section["sample_meta_file"]),
        fasta_files=[str(ctx.resolve(f)) for f in fasta] if fasta else None,
        pairs=[ctx.pair(g1, g2) for (g1, g2) in ctx.contrast_pairs],
        out_dir=ctx.de_dir,
        p_adj_cutoff=float(section.get("p_adj_cutoff", 0.05)),
        log2fc_cutoff=float(section.get("log2fc_cutoff", 0.5)),
        padj_col=section.get("padj_col", "p_adj"),
        log2fc_col=section.get("log2fc_col", "log2FC"),
    )
    log.info("Running DE for %d contrasts x %d matrices -> %s", len(ctx.contrast_pairs), len(roles), ctx.de_dir)
    try:
        fn(**kwargs)
    finally:
        plt.close("all")  # the DE script leaves its volcano/FC figures open
    missing = [str(p) for p in expected if not p.exists()]
    if missing:
        raise FileNotFoundError(f"DE step finished but expected outputs are missing: {missing}")


def split_shared_new(section, ctx):
    """Split each JUMPlion DE table into shared / new precursors and plot each part.

    shared : precursors also present in the DIA-NN DE table of the same contrast
    new    : the rest, i.e. precursors DE-testable only after JUMPlion
    Outputs go next to the JUMPlion DE tables:
      <name>_<pair>_{shared,new}.csv, _volcano.svg, _fc.svg  (+ <name>_shared_new_summary.tsv)
    Plots use the DE script's own plot_volcano / plot_scatter_box, so they match its other DE plots.
    """
    from . import diff_expression as dea

    id_col = section.get("mat_type", "Precursor.Id")
    name = ctx.matrices["jumplion"]["name"]
    out_dir = ctx.de_dir / f"{name}_dea_results"
    rows = []
    for lab1, lab2 in ctx.contrast_pairs:
        a, b = ctx.pair(lab1, lab2)
        pair_name = f"{a}_vs_{b}"
        jl = pd.read_csv(ctx.de_file("jumplion", lab1, lab2), index_col=0)
        dn = pd.read_csv(ctx.de_file("diann", lab1, lab2), index_col=0)
        in_diann = jl[id_col].isin(set(dn[id_col]))
        for part, sub in (("shared", jl[in_diann]), ("new", jl[~in_diann])):
            stem = f"{name}_{pair_name}_{part}"
            sub.to_csv(out_dir / f"{stem}.csv")
            counts = sub["DE_result"].value_counts()
            rows.append({"contrast": ctx.contrast(lab1, lab2), "pair": pair_name, "part": part, "n": len(sub),
                         **{k.replace(" ", "_"): int(counts.get(k, 0)) for k in
                            ["true positive", "false positive", "true negative", "false negative"]}})
            if sub.empty:
                log.warning("%s: no %s precursors; plots skipped.", pair_name, part)
                continue
            try:
                fig, ax = plt.subplots(1, 1, figsize=(2.4, 2.4))
                dea.plot_volcano(ax, sub, -3.2, 3.2, 0.5, 0.05, f"{pair_name} ({part})", 2, None)
                plt.savefig(out_dir / f"{stem}_volcano.svg", bbox_inches="tight")
                dea.plot_scatter_box(sub, title=f"{name}_{pair_name} ({part})", x_col=b,
                                     figsize=(3.2, 2.4), width_ratios=[4, 1])
                plt.savefig(out_dir / f"{stem}_fc.svg", bbox_inches="tight")
            finally:
                plt.close("all")
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / f"{name}_shared_new_summary.tsv", sep="\t", index=False)
    log.info("Shared/new split written to %s", out_dir)


# ----------------------------------------------------------------------------
# Wire prepared outputs into the analysis sections (explicit settings win)
# ----------------------------------------------------------------------------
def apply_derived_inputs(cfg, ctx):
    bench = {"groups": ctx.groups}
    if ctx.reference is not None:
        bench["reference"] = ctx.reference
    else:
        bench["contrasts"] = [list(p) for p in ctx.contrast_pairs]
    de_sec = cfg.get("differential_expression", {})

    def comparisons():
        out = []
        for g1, g2 in ctx.contrast_pairs:
            a, b = ctx.pair(g1, g2)
            out.append({
                "name": ctx.contrast(g1, g2),
                "g1": a, "g2": b,
                "diann": str(ctx.de_file("diann", g1, g2)), "jumplion": str(ctx.de_file("jumplion", g1, g2)),
                "diann_pre_de": str(ctx.de_file("diann", g1, g2, True)),
                "jumplion_pre_de": str(ctx.de_file("jumplion", g1, g2, True)),
            })
        return out

    perf = cfg.get("performance")
    if perf is not None:
        perf.setdefault("benchmark", bench)
        perf.setdefault("padj_cutoff", cfg.get("benchmark", {}).get("padj_cutoff", 0.05))
        if "comparisons" not in perf:
            perf["comparisons"] = comparisons()
        for role in ("diann", "jumplion"):
            if f"{role}_matrix" not in perf and role in ctx.matrices:
                perf[f"{role}_matrix"] = str(ctx.resolve(ctx.matrices[role]["file"]))
        perf.setdefault("matrix_sep", de_sec.get("mat_sep", "\t"))
        if de_sec.get("sample_meta_file"):
            perf.setdefault("sample_meta_file", str(ctx.resolve(de_sec["sample_meta_file"])))

    genl = cfg.get("generalization")
    if genl is not None:
        genl.setdefault("benchmark", bench)
        if "comparisons" not in genl:
            genl["comparisons"] = comparisons()

    mech = cfg.get("mechanism")
    if mech is not None:
        mech.setdefault("evidence_table", str(ctx.features_file))
        mech.setdefault("quantification_parquet", str(ctx.features_file))
        mech.setdefault("benchmark", bench)
        if de_sec.get("sample_meta_file"):
            mech.setdefault("sample_meta_file", str(ctx.resolve(de_sec["sample_meta_file"])))
        if "species_annotation" not in mech and "diann" in ctx.matrices:
            # Any pre-DE table carries the precursor -> species annotation; use the last contrast.
            g1, g2 = ctx.contrast_pairs[-1]
            mech["species_annotation"] = str(ctx.de_file("diann", g1, g2, pre=True))
