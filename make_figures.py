#!/usr/bin/env python3
"""Generate the manuscript figures. One entry point; pick datasets, or run everything.

    python make_figures.py                          # all targets, in the order below
    python make_figures.py hye_wang hye_genbeta
    python make_figures.py hye_wang --stages mechanism      # only one stage of a pipeline target
    python make_figures.py singlecell_pbmc --output-dir D:/out

Targets (run in this order; each step in its own Python process):
    hye_wang          HYE benchmark: differential_expression, performance, mechanism
                      (configs/hye_wang.json)
    hye_genbeta       generalization to other instruments/gradients: precursor level, then protein level
                      (configs/hye_genbeta_precursor.json, then configs/hye_genbeta_protein.json,
                      which reads the precursor run's tables)
    singlecell_hela   single-cell HeLa (configs/singlecell_hela.json)
    singlecell_pbmc   single-cell PBMC: general panels + scanpy clustering (configs/singlecell_pbmc.json)
    imputation_comparison   (opt-in: only runs when named) JUMPlion vs KNN / missForest / MinProb imputation of DIA-NN on the
                      generalization protein tables, in R (Supp_Generalization_F, G, H); needs Rscript and the
                      hye_genbeta tables first. Stages: impute, compare, plots; settings in the
                      "imputation_comparison" block of configs/hye_genbeta_protein.json

--stages limits a target to some of its stages. hye_wang / hye_genbeta: differential_expression, generalization,
performance, mechanism; imputation_comparison: impute, compare, plots. A target with none of the chosen stages is skipped. --output-dir overrides each
config's output_dir. For the remaining options (--normalization, --fail-fast, a custom config) run
the pipeline directly:  python -m jumplion_figures.pipeline --config <json> [options]

A failing target is reported and the others still run; the exit code is non-zero if any failed.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIGS = HERE / "configs"

PIPELINE = "pipeline"
SINGLECELL = "singlecell"
# target -> ordered steps of (kind, config file); a step that fails does not stop the next one.
TARGETS = {
    "hye_wang": [(PIPELINE, "hye_wang.json")],
    "hye_genbeta": [(PIPELINE, "hye_genbeta_precursor.json"), (PIPELINE, "hye_genbeta_protein.json")],
    "singlecell_hela": [(SINGLECELL, "singlecell_hela.json")],
    "singlecell_pbmc": [(SINGLECELL, "singlecell_pbmc.json")],
    "imputation_comparison": [],  # built by imputation_commands(); R scripts, not a config step
}
OPT_IN = {"imputation_comparison"}  # skipped unless named on the command line
STAGES = ["differential_expression", "generalization", "performance", "mechanism"]
IMPUTATION_STAGES = ["impute", "compare", "plots"]
IMPUTATION_CONFIG = "hye_genbeta_protein.json"

# Single-cell targets run in-process through the modules' own main(default_config, argv).
SINGLECELL_CODE = {
    "singlecell_hela": ("import sys; from jumplion_figures import singlecell; "
                        "sys.exit(singlecell.main({cfg!r}, sys.argv[1:]))"),
    "singlecell_pbmc": ("import sys; from jumplion_figures import singlecell, singlecell_scanpy; "
                        "rc = singlecell.main({cfg!r}, sys.argv[1:]); "
                        "sys.exit(singlecell_scanpy.main({cfg!r}, sys.argv[1:]) or rc)"),
}


def find_rscript(configured):
    """Rscript from the config, $RSCRIPT, the PATH, or the newest R under Program Files (Windows)."""
    candidates = [configured, os.environ.get("RSCRIPT"), shutil.which("Rscript")]
    found = sorted(Path("C:/Program Files/R").glob("R-*/bin/Rscript.exe"), reverse=True)
    candidates += [str(p) for p in found]
    for c in candidates:
        if c and Path(c).exists() or (c and shutil.which(c)):
            return c
    raise FileNotFoundError("Rscript not found: set imputation_comparison.rscript in "
                            f"configs/{IMPUTATION_CONFIG}, or the RSCRIPT environment variable")


def imputation_commands(output_dir, stages):
    """Ordered (label, command) list for the imputation_comparison target: the R scripts next to this file.

    Settings come from the imputation_comparison block of the protein-level generalization config; the
    dataset names and the gen_beta root (the parent of each run_dir) from the same config."""
    cfg = json.loads((CONFIGS / IMPUTATION_CONFIG).read_text(encoding="utf-8"))
    opts = cfg.get("imputation_comparison", {})
    datasets = list(cfg["datasets"])
    root = Path(cfg["datasets"][datasets[0]]["run_dir"]).parent
    ana = output_dir or cfg["output_dir"]
    rscript = find_rscript(opts.get("rscript"))
    methods = ",".join(opts.get("methods", ["knn", "missforest", "minprob"]))
    seed = str(opts.get("seed", 1))
    missing = [str(Path(cfg["datasets"][d]["run_dir"]) / "publication" / f)
               for d in datasets for f in ("prot_matrix_diann.tsv", "prot_matrix_diann_jumplion.tsv")
               if not (Path(cfg["datasets"][d]["run_dir"]) / "publication" / f).exists()]
    if missing:
        raise FileNotFoundError("imputation_comparison needs these protein matrices first:\n  " + "\n  ".join(missing))
    run = lambda script, *a: [rscript, str(HERE / "r_scripts" / script), *a]  # noqa: E731
    steps = []
    if not stages or "impute" in stages:
        steps.append(("impute", run("impute_protein_matrices.R", str(root), ",".join(datasets), methods, seed)))
    if not stages or "compare" in stages:
        steps += [(f"compare {d}", run("compare_imputation.R", str(root), d, str(ana))) for d in datasets]
    if not stages or "plots" in stages:
        steps += [("figures F, G", run("compare_imputation_combined.R", str(ana))),
                  ("figure H", run("compare_imputation_curves.R", str(root), str(ana)))]
    return steps


def command(target, kind, cfg_name, output_dir, stages):
    cfg = str(CONFIGS / cfg_name)
    extra = ["--output-dir", output_dir] if output_dir else []
    if kind == PIPELINE:
        return [sys.executable, "-m", "jumplion_figures.pipeline", "--config", cfg, *extra,
                *(["--only", *stages] if stages else [])]
    return [sys.executable, "-c", SINGLECELL_CODE[target].format(cfg=cfg), *extra]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="*", metavar="target", help=f"targets to run (default: all except opt-in {sorted(OPT_IN)}): {', '.join(TARGETS)}")
    ap.add_argument("--skip", nargs="+", choices=list(TARGETS), default=[], help="targets to skip")
    ap.add_argument("--stages", nargs="+", choices=STAGES + IMPUTATION_STAGES,
                    help="stages to run: pipeline targets take " + ", ".join(STAGES) + "; imputation_comparison takes "
                    + ", ".join(IMPUTATION_STAGES) + ". A target none of the chosen stages apply to is skipped")
    ap.add_argument("--output-dir", help="override output_dir from the config(s)")
    args = ap.parse_args(argv)

    bad = [t for t in args.targets if t not in TARGETS]
    if bad:
        ap.error(f"unknown target(s) {bad}; choose from {list(TARGETS)}")
    chosen = [t for t in TARGETS if (t in args.targets if args.targets else t not in OPT_IN) and t not in args.skip]

    results = {}
    for target in chosen:
        print(f"\n==== {target} ====", flush=True)
        rcs = []
        if target == "imputation_comparison":
            stages = [s for s in args.stages or [] if s in IMPUTATION_STAGES]
            if args.stages and not stages:
                print("-- skipped (no imputation_comparison stage chosen)", flush=True)
            else:
                try:
                    for label, cmd in imputation_commands(args.output_dir, stages):
                        print(f"-- {label}", flush=True)
                        rc = subprocess.call(cmd, cwd=HERE)
                        rcs.append(rc)
                        if rc:  # later steps read this one's outputs
                            break
                except FileNotFoundError as e:
                    print(f"   {e}", flush=True)
                    rcs.append(1)
            results[target] = max(rcs, default=0)
            continue
        for kind, cfg_name in TARGETS[target]:
            stages = [s for s in args.stages or [] if s in STAGES] if kind == PIPELINE else None
            if kind == PIPELINE and args.stages and not stages:
                print(f"-- {cfg_name}: skipped (no pipeline stage chosen)", flush=True)
                continue
            print(f"-- {cfg_name}", flush=True)
            rcs.append(subprocess.call(command(target, kind, cfg_name, args.output_dir, stages), cwd=HERE))
        results[target] = max(rcs, default=0)
    # Supp_Benchmark_design stacks panels from both HYE runs, so it follows whenever both ran.
    if {"hye_wang", "hye_genbeta"} <= set(chosen) and not args.stages:
        print("\n==== benchmark design (hye_wang + hye_genbeta) ====", flush=True)
        extra = ["--output-dir", args.output_dir] if args.output_dir else []
        results["benchmark_design"] = subprocess.call(
            [sys.executable, "-m", "jumplion_figures.design_figure", "--top", str(CONFIGS / "hye_wang.json"),
             "--bottom", str(CONFIGS / "hye_genbeta_protein.json"), *extra], cwd=HERE)
    print("\nSummary")
    for target, rc in results.items():
        print(f"  {target:<24} {'ok' if rc == 0 else f'FAILED (exit {rc})'}")
    return 0 if all(rc == 0 for rc in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
