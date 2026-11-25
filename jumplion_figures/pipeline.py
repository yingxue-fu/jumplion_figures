#!/usr/bin/env python3
"""JUMPlion manuscript pipeline: 4 stages driven by ONE config.

Stages (always executed in this order; choose a subset with "run" in the config or --only):
    differential_expression       precursor matrices -> DE + pre-DE tables per contrast
    generalization                growable, dataset-agnostic performance analysis (currently the
                                  30-run cumulative completeness curve, completeness-by-method
                                  bars, sensitivity/FDR, and CV/delta-FC quantitative fidelity,
                                  reusing performance.py's own loading/computation/plotting code) --
                                  for datasets that don't use the rest of performance.py's
                                  Wang-HYE-specific ground-truth panels. Every dataset's output
                                  lands in that ONE shared main/supp/table (see
                                  UNIFIED_DATASET_STAGES) rather than one subfolder per dataset --
                                  filenames carry a "_<dataset>" suffix instead. Coverage,
                                  completeness-by-method and sensitivity/FDR are the exception to
                                  per-dataset figures: each is combined into one figure across
                                  every dataset instead (see generalization.py).
    performance                   DIA-NN vs JUMPlion benchmark (uses the DE tables)
    mechanism                     builds precursor-run features from xic_quantification.parquet,
                                  then localization-tier recovery, intensity mapping, and the
                                  end-to-end final-recovered analysis, all sharing one load of
                                  the features/quantification table

Usage
-----
    python -m jumplion_figures.pipeline --config configs/hye_wang.json
    python -m jumplion_figures.pipeline --config configs/hye_wang.json --only performance
    python -m jumplion_figures.pipeline --config configs/hye_wang.json --only differential_expression performance

Prepared data (DE tables, features file) is reused unless the matching "overwrite" key is true.

Multiple datasets
-----------------
    A top-level "datasets": {name: {"jumplion_outdir": ..., "sample_meta_file": ...}, ...} block runs
    every stage once per dataset against an otherwise identical config -- for datasets that share
    the same benchmark groups/comparisons and differ only in where their files live and what their
    samples are named. Each dataset's jumplion_outdir is substituted for ${jumplion_outdir} in its own copy of the
    config, and its output lands under <output_dir>/<stage>/<dataset_name>/... (stage first,
    dataset second, so each stage's folder holds every dataset). Omit "datasets" to run a single
    dataset as before (output lands directly under <output_dir>/<stage>/...).

Output layout
-------------
    <output_dir>/
        differential_expression/[<dataset_name>/]  <name>_dea_results/ (DE + pre-DE tables, volcano/FC plots)
        generalization/                             main/  supp/  table/ (every dataset's files
                                                    share this ONE folder, named "..._<dataset_name>")
        performance/[<dataset_name>/]               main/  supp/  table/
        mechanism/[<dataset_name>/]                 main/  supp/  table/  (table/ holds precursor_run_features.parquet)
        run_summary.tsv                             one row per dataset x stage
"""
from __future__ import annotations

import argparse
import copy
import logging
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import pandas as pd

from . import style
from .common import (OutputLayout, PathResolver, read_raw_config,
                                     substitute_config, setup_logging, write_tsv)
from . import prepare, performance, mechanism, generalization

PREP_STAGES = {"differential_expression": prepare.run_de}
ANALYSES = {
    "generalization": generalization.run,
    "performance": performance.run,
    "mechanism": mechanism.run,
}
ORDER = ["differential_expression", "generalization", "performance", "mechanism"]
DEPENDS_ON = {  # skip an analysis (with a clear reason) if a stage it needs failed in this run
    "generalization": ["differential_expression"],
    "performance": ["differential_expression"],
    "mechanism": ["differential_expression"],
}
# Analyses that read the precursor-run feature table build/reuse it first.
FEATURE_INPUT_KEY = {"mechanism": "evidence_table"}

# Stages whose datasets all write into ONE shared main/supp/table (not one subfolder per dataset)
# -- the stage itself must then disambiguate its own output filenames by dataset, via the
# "_dataset" key injected into its config section below.
UNIFIED_DATASET_STAGES = {"generalization"}


def write_panel_manifest(layout):
    rows = []
    for group, d in [("main", layout.main), ("supp", layout.supp)]:
        stems = sorted({p.stem for p in d.iterdir() if p.is_file()})
        rows += [{"group": group, "panel": s} for s in stems]
    rows += [{"group": "table", "panel": p.name}
             for p in sorted(layout.table.iterdir()) if p.is_file() and p.name != "panel_manifest.tsv"]
    write_tsv(pd.DataFrame(rows, columns=["group", "panel"]), layout.table, "panel_manifest.tsv")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="Unified JSON config")
    ap.add_argument("--only", nargs="+", choices=ORDER, help="Run only these stages")
    ap.add_argument("--output-dir", help="Override output_dir from the config")
    ap.add_argument("--normalization", choices=["normalized", "raw"],
                    help="Override differential_expression.normalization from the config")
    ap.add_argument("--fail-fast", action="store_true", help="Stop at the first failing stage")
    args = ap.parse_args(argv)

    setup_logging()
    log = logging.getLogger("jumplion_figures")
    raw_cfg = read_raw_config(args.config)

    # "datasets": {name: {"jumplion_outdir": ..., "sample_meta_file": ...}, ...} runs every stage once per
    # dataset, substituting each dataset's own jumplion_outdir (and sample_meta_file) into an otherwise
    # identical config -- for several datasets that share the same benchmark groups/comparisons
    # and differ only in where their files live and what their samples are named.
    datasets = raw_cfg.get("datasets")

    # strict=False when "datasets" is present: this "base" pass only reads top-level style/
    # output_dir/run below, never the dataset-scoped (per-dataset-substituted) sections, so it
    # shouldn't fail just because ${jumplion_outdir} (supplied per dataset, not here) can't resolve yet.
    base_cfg = substitute_config(raw_cfg, strict=not datasets)

    base_out_root = (Path(args.output_dir) if args.output_dir
                     else PathResolver(base_cfg["_config_dir"])(base_cfg.get("output_dir", "jumplion_figures_output")))
    base_out_root.mkdir(parents=True, exist_ok=True)

    style_cfg = base_cfg.get("style", {})
    style.apply_style(style_cfg)
    saver = style.FigureSaver(style_cfg.get("formats", ["pdf", "png"]), style_cfg.get("dpi", 600))

    requested = set(args.only or base_cfg.get("run", ORDER))
    unknown = requested - set(ORDER)
    if unknown:
        raise SystemExit(f"Unknown stage(s) in 'run': {sorted(unknown)}. Valid: {ORDER}")
    dataset_items = list(datasets.items()) if datasets else [(None, {})]

    all_summary = []
    for ds_name, ds_overrides in dataset_items:
        if ds_name is None:
            cfg = base_cfg
        else:
            raw_ds = copy.deepcopy(raw_cfg)
            if "sample_meta_file" in ds_overrides:
                raw_ds.setdefault("differential_expression", {})["sample_meta_file"] = ds_overrides["sample_meta_file"]
            cfg = substitute_config(raw_ds, {"jumplion_outdir": ds_overrides["jumplion_outdir"]})
        if args.normalization:
            cfg.setdefault("differential_expression", {})["normalization"] = args.normalization

        # Every stage's own folder groups all datasets under it (<output_dir>/<stage>/<dataset>/...)
        # rather than each dataset owning a folder with every stage inside it -- so pass the shared
        # base_out_root plus ds_name, not a per-dataset root, down to Context/OutputLayout.
        resolve = PathResolver(cfg["_config_dir"])
        ctx = prepare.Context(cfg, resolve, base_out_root, dataset=ds_name)
        prepare.apply_derived_inputs(cfg, ctx)

        log.info("==== dataset: %s ====", ds_name or "(default)")
        failed = set()
        for name in [s for s in ORDER if s in requested]:
            if name not in cfg:
                log.warning("Skipping %s: no '%s' section in config.", name, name)
                all_summary.append({"dataset": ds_name or "default", "stage": name, "status": "skipped (no config section)"})
                continue
            blockers = [d for d in DEPENDS_ON.get(name, []) if d in failed]
            if blockers:
                log.warning("Skipping %s: required stage(s) %s failed.", name, blockers)
                all_summary.append({"dataset": ds_name or "default", "stage": name,
                                    "status": f"skipped ({', '.join(blockers)} failed)"})
                continue
            t0, n_before = time.time(), len(saver.saved)
            is_prep = name in PREP_STAGES
            unified = name in UNIFIED_DATASET_STAGES
            layout = None if is_prep else OutputLayout(base_out_root, name, dataset=None if unified else ds_name)
            log.info("==== %s ====", name)
            try:
                if is_prep:
                    PREP_STAGES[name](cfg[name], ctx)
                else:
                    key = FEATURE_INPUT_KEY.get(name)
                    if key and str(cfg[name].get(key)) == str(ctx.features_file):
                        prepare.ensure_features(ctx)
                    # A unified-output stage can't tell its datasets' files apart by folder (they
                    # all land in the same main/supp/table), so it gets told its own dataset name
                    # (to fold into filenames), gradient (main vs supp routing, see "datasets" in
                    # the config), and the full dataset list (so it can tell, across its repeated
                    # per-dataset calls, when it has now seen every one -- e.g. to build one
                    # combined cross-dataset figure on the last call) -- a plain dict copy, not a
                    # mutation of the shared cfg[name], since this loop reuses that section across
                    # every dataset.
                    stage_cfg = (dict(cfg[name], _dataset=ds_name, _gradient=ds_overrides.get("gradient"),
                                     _all_datasets=sorted(datasets) if datasets else [ds_name or "default"])
                                if unified else cfg[name])
                    ANALYSES[name](stage_cfg, layout, saver, resolve)
                status = "ok"
            except Exception as exc:  # keep going so independent stages still produce output
                log.exception("%s failed: %s", name, exc)
                status = f"failed: {type(exc).__name__}: {exc}"
                failed.add(name)
                if args.fail_fast:
                    raise
            finally:
                style.plt.close("all")
                if layout is not None:
                    write_panel_manifest(layout)
            all_summary.append({"dataset": ds_name or "default", "stage": name, "status": status,
                                "n_figures": len(saver.saved) - n_before,
                                "seconds": round(time.time() - t0, 1),
                                "output": str(layout.root if layout else ctx.de_dir)})

    write_tsv(pd.DataFrame(all_summary), base_out_root, "run_summary.tsv")
    for row in all_summary:
        log.info("%-15s %-30s %s", row["dataset"], row["stage"], row["status"])
    return 0 if all(r["status"] == "ok" for r in all_summary) else 1


if __name__ == "__main__":
    sys.exit(main())
