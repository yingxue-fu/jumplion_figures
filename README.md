# JUMPlion manuscript figures

Figures comparing DIA-NN with JUMPlion: the HYE benchmark (performance, mechanism), its
generalization to other instruments/gradients (precursor and protein level), and the single-cell
datasets (HeLa, PBMC). One visual style (`jumplion_figures/style.py`) for everything.

## Quick start

One entry point, `make_figures.py`. Name the targets you want, or give none to run all of them:

```
python make_figures.py                                  # everything, in the order below
python make_figures.py hye_wang hye_genbeta
python make_figures.py --skip singlecell_pbmc
python make_figures.py hye_wang --stages mechanism      # one stage of a pipeline target
python make_figures.py singlecell_hela --output-dir D:/out
python make_figures.py imputation_comparison            # opt-in, never part of the default run (needs R)
python make_figures.py imputation_comparison --stages plots   # redraw its figures only
```

| Target | Config | Produces | Needs |
|---|---|---|---|
| `hye_wang` | `configs/hye_wang.json` | HYE benchmark: differential expression, performance **and mechanism** figures | DIA-NN / JUMPlion precursor matrices; `xic_quantification.parquet` and the RT/XIC files for mechanism |
| `hye_genbeta` | `configs/hye_genbeta_precursor.json`, then `configs/hye_genbeta_protein.json` | other instruments/gradients, precursor level then protein level | precursor and protein matrices for each dataset; the protein step reads the precursor step's tables (`precursor_table_dir`) |
| `singlecell_hela` | `configs/singlecell_hela.json` | single-cell HeLa | HeLa precursor + protein matrices, sample meta |
| `singlecell_pbmc` | `configs/singlecell_pbmc.json` | single-cell PBMC (general panels + scanpy) | PBMC matrices, sample meta; scanpy |
| `imputation_comparison` (opt-in) | `imputation_comparison` block of `configs/hye_genbeta_protein.json` | JUMPlion vs KNN / missForest / MinProb imputation of DIA-NN, protein level, in R: `Supp_Generalization_F` (combined), `G` (restored proteins), `H` (FDR vs true-positive curves) | R (`Rscript`; packages impute, missForest, imputeLCMD, data.table, ggplot2, patchwork), the `hye_genbeta` protein matrices (DIA-NN and JUMPlion) |

`imputation_comparison` is opt-in: it only runs when named (`python make_figures.py imputation_comparison`), never in the default run. Its stages are `impute` (writes `prot_matrix_diann_{knn,missforest,minprob}.tsv` beside each dataset's DIA-NN matrix), `compare` (per-dataset tables `imputation_comparison_<dataset>.tsv` and `imputation_fdr_sensitivity_curves.tsv`) and `plots`; `--stages plots` redraws the figures without redoing the slow imputation. The R scripts sit in `r_scripts/` (`impute_protein_matrices.R`, `compare_imputation.R`, `compare_imputation_combined.R`, `compare_imputation_curves.R`, shared theme `jumplion_theme.R`, which mirrors `style.py`). `Rscript` is taken from `imputation_comparison.rscript` in the config, `$RSCRIPT`, the PATH, or `C:/Program Files/R`; the seed and methods are in the same block.

Run it from this folder. Before the first run the DIA-NN and JUMPlion protein matrices (`prot_matrix_diann.tsv`, `prot_matrix_diann_jumplion.tsv` in each dataset's `publication/` folder) must exist for all four datasets; the target lists any that are missing and stops. Stages combine freely: `--stages compare plots` rebuilds the tables and figures without redoing the slow imputation (missForest dominates the run time). The figures go to `<output_dir>/generalization/supp/` (`Supp_Generalization_F_imputation_comparison_all_datasets_combined`, `G_imputation_comparison_restored`, `H_imputation_fdr_tp_curves`), the tables to `<output_dir>/generalization/table/`. A failing R step stops the later ones and the summary shows `FAILED`.

Troubleshooting: if `Rscript` is not found, set `imputation_comparison.rscript` in `configs/hye_genbeta_protein.json` (e.g. `C:/Program Files/R/R-4.4.3/bin/Rscript.exe`) or the `RSCRIPT` environment variable. If a PNG or PDF cannot be written ("cairo error" / "agg could not write"), close the file in any viewer that has it open, or pause OneDrive sync, and rerun with `--stages plots`. The results come from a single imputation seed.

Targets run in the order above, each in its own Python process. A failing target is reported and the
rest still run. Within `hye_genbeta` the protein step reads tables written by the precursor step, so
it always runs second.

Before the first run, edit the paths in each config in `configs/` (every input/output path is an
absolute path in the config; none is hard-coded in the code). `--output-dir DIR` overrides the
configs' `output_dir`. `--stages` (differential_expression, generalization, performance, mechanism)
limits the pipeline targets to some of their stages (`imputation_comparison` has its own: impute, compare, plots; a target none of the chosen stages apply to is skipped). For the remaining pipeline options
(`--normalization`, `--fail-fast`, a custom config):

```
python -m jumplion_figures.pipeline --config configs/hye_wang.json --normalization raw
```

Setup: `pip install -r requirements.txt` (Python packages; developed with Python 3.12). The `imputation_comparison` target also needs R and its packages: `Rscript r_scripts/install_packages.R`. The paths in `configs/*.json` are placeholders (`/path/to/...`): replace them with the locations of your data before running.

Requirements: Python 3.9+, numpy, pandas, scipy, matplotlib, pyarrow. `differential_expression`
also needs statsmodels, seaborn and biopython. The PBMC scanpy step needs scanpy, anndata,
scikit-learn and python-igraph (or leidenalg). `umap-learn` is optional. No `jumplion` install is
required.

## Project layout

```
make_figures.py                single entry point (see above)
configs/                       one JSON per target
r_scripts/                     R scripts of the opt-in imputation_comparison target (+ jumplion_theme.R, the ggplot twin of style.py)
jumplion_figures/
  pipeline.py                  pipeline driver: differential_expression, generalization, performance, mechanism
  style.py                     palette, rcParams, plot helpers  <- restyle everything here
  common.py                    config (${var} expansion), paths, output folders, table I/O
  prepare.py                   differential-expression stage, feature building, input wiring
  diff_expression.py           vendored differential-expression script (DE, volcano, benchmark volcano grid)
  precursor_features.py        vendored per-precursor-run feature builder (input to mechanism)
  performance.py               HYE benchmark: coverage, completeness, sensitivity/FDR, CV/dFC, examples
  generalization.py            multi-dataset version of performance (precursor and protein level)
  mechanism.py                 localization-tier recovery, intensity mapping, final recovered
  intensity_mapping.py         production intensity-mapping model (loaded by the reliability module)
  intensity_mapping_reliability.py   mapping-reliability analysis used by mechanism
  singlecell.py                single-cell panels shared by HeLa and PBMC (HeLa integrated figure)
  singlecell_scanpy.py         PBMC protein-level clustering, markers, integrated figure
```

## Pipeline stages

Stages always run in this order; choose a subset with `"run"` in the config or `--stages`:

```
python make_figures.py hye_wang --stages performance
python make_figures.py hye_wang --stages differential_expression performance
```

| Stage | Does | Output |
|---|---|---|
| `differential_expression` | both quantification matrices -> DE and pre-DE tables per contrast | `differential_expression/[<dataset>/]<name>_dea_results/` |
| `generalization` | dataset-agnostic performance figures across datasets | `generalization/{main,supp,table}` (all datasets share one folder; names end `_<dataset>` or `_all_datasets`) |
| `performance` | DIA-NN vs JUMPlion benchmark on the DE tables | `performance/{main,supp,table}` |
| `mechanism` | builds `precursor_run_features.parquet`, then localization-tier recovery, intensity mapping, final-recovered analysis | `mechanism/{main,supp,table}` |

* DE tables and the features file are reused when they exist (`differential_expression.overwrite`,
  `mechanism.overwrite_features` force a rebuild).
* If `differential_expression` fails, the stages that need it are skipped; the reason is in
  `run_summary.tsv`. A failing stage does not stop the others (use `--fail-fast` to stop).
* DE also splits each JUMPlion table into *shared* (precursor also in the DIA-NN table) and
  *new* precursors and draws volcano/FC plots for each.
* `--normalization raw|normalized` switches which matrix pair DE uses (when the config offers both),
  without editing the config.

### performance

Organized around DIA-NN's own completeness per contrast: **Complete** (6/6 in both groups),
**Partially complete** (>= 3/6 in both groups but < 6/6 in at least one) and **Insufficient**
(< 3/6 in at least one group). Insufficient precursors are split into those that *remain
insufficient* and those that *become sufficient* once JUMPlion fills gaps.

* Main: `Fig_Performance_A` coverage, `B` completeness by method, `C` representative examples,
  `D` sensitivity/FDR, `E` quantitative fidelity, and `Fig_Performance_Integrated` (a-g).
* Supp: `A_benchmark_design` (with the expected-log2FC table), `B0_heatmaps` (state transitions),
  `B1_species_composition_overall`, `B2_completeness_and_species_composition_by_intensity`,
  `B_integrated` (B0 + B1), then C-F and `E_representative_examples_expanded`.

**FDR definition.** Every FDR shown (sensitivity/FDR panels, species-balanced FDR, volcano empirical FDR) is the
*strict* empirical FDR: significant calls in the wrong direction count as false discoveries,
(FP + wrong direction) / (TP + FP + wrong direction). The earlier human-only FP / (TP + FP) is still
written to the tables as `empirical_fdr_fp_only` (with `fdr_fp_only_ci_low/high`).

### generalization

Run once per config; with several `datasets`, every dataset's data goes into the same figures.

* Precursor run (`hye_genbeta_precursor.json`): `A_coverage_all_datasets`,
  `B_completeness_by_method_all_datasets`, and per `COMBINED_CONTRASTS` entry
  (`B_vs_C` -> main, `A_vs_B` -> supp): `C_completeness_by_method_complete_only`, `D_sensitivity_fdr`,
  `D_sensitivity_fdr_insufficient`, `D_volcano_all_datasets`, `E_log2FC_error_CV_box` (and the
  `_restored` version), and `Integrated`.
* Protein run (`hye_genbeta_protein.json`, `id_col: Protein.Group`, `precursor_table_dir` set):
  draws `A_coverage_all_datasets`, `B_completeness_by_method_all_datasets`, `D_volcano_all_datasets`,
  `D_sensitivity_fdr_insufficient`, the one-off `benchmark_design`, and the precursor-vs-protein
  figures (`C_completeness_precursor_vs_protein`, `D_..._precursor_vs_protein`,
  `E_..._precursor_vs_protein`, `Integrated`). Other per-run panels are not drawn at protein level.
* `{Fig,Supp}` prefix follows each entry's destination. The exact list of files written is in
  `generalization/table/panel_manifest.tsv`.

## Single-cell

```
python make_figures.py singlecell_hela
python make_figures.py singlecell_pbmc
```

Neither goes through the pipeline stages; they read the matrices directly (no ground truth). Output:
`<output_dir>/singlecell_<label>/{main,supp,table}`. Protein-level panels are `Fig_` (main) and
precursor-level panels `Supp_` (supp); this routing is fixed in the code.

* **HeLa**: A IDs per cell, B completeness (with the count detected in every cell), C intensity
  density (DIA-NN reported vs JUMPlion recovered), D paired CV, E CV by intensity, F cell-cell
  correlation, G PCA. `Fig_SingleCell_HeLa_Integrated` is a 2x2 of A, B, D, F.
* **PBMC**: A, B, C from `singlecell.py`; the CV and correlation panels are skipped
  (`skip_analyses`). `singlecell_scanpy.py` clusters both methods on one shared protein set (Leiden),
  and draws the cluster UMAP (H), marker coverage (O), marker violins (I, P, Q), the DIA-NN-cluster
  UMAP (N) and cluster dot plot (S), plus `Fig_SingleCell_PBMC_Integrated`. No cell types are assigned.
* `singlecell.panel_size` in the PBMC config (`[1.9, 1.9]`) must match the integrated figure's A/B
  width (`INT_AB_W` in `singlecell_scanpy.py`).

## Config

* `paths`: variables usable as `${name}` anywhere in the config (they may reference each other).
* `output_dir`: where everything is written. `style`: font, size, `formats` (include `png` -- the
  integrated figures are assembled from the saved PNGs), `dpi`.
* `benchmark`: group labels -> names (`H<h>Y<y>E<e>`), plus either `reference` (every other group vs
  one, a star) or `contrasts` (explicit `[["A","B"],["B","C"]]` pairs). `contrasts` wins if both are
  given; `performance` supports only `reference`. `padj_cutoff` sets significance; no log2FC
  cutoff is applied in `generalization` (the sign of log2FC only separates TP from wrong direction).
* `datasets` (optional): `{name: {"run_dir", "sample_meta_file", "gradient"}}` runs every stage once
  per dataset; each `run_dir` replaces `${run_dir}` in its own copy of the config.
* `differential_expression.matrices.<role>`: `{"name", "file"}`, or nested `normalized`/`raw` pairs
  (give them different `name`s so their outputs do not overwrite each other).
* `generalization.id_col` / `precursor_table_dir`: switch a run to protein level and point it at the
  precursor run's `generalization/table`.
* `mechanism.signal_rank_population`: `"recovered"` (default) or `"all"`. Both versions of
  `Fig_Mechanism_B_signal_rank_coelution_{recovered,all}` are written to `mechanism/main` (2nd-4th groups =
  finally recovered precursor-runs, or every JUMPlion entry of the three recovery tiers); this setting
  picks which one goes into the integrated figure. "DIA-NN reported" is unfiltered in both.
* `singlecell`: `label`, `sample_meta_file`, `levels.<precursor|protein>.{id_col, matrices}`,
  `group_col`, `cv_min_cells`, `embedding`, `skip_analyses`, `markers`, `violin_markers(_main)`,
  `marker_labels`. `scanpy` (PBMC): `min_cell_frac`, `n_pcs`, `n_neighbors`, `resolution`, `seed`.

## Output layout

```
<output_dir>/
  differential_expression/[<dataset>/]  <name>_dea_results/
  generalization/        main/  supp/  table/          (all datasets together)
  performance/           main/  supp/  table/
  mechanism/             main/  supp/  table/          (table/ also holds precursor_run_features.parquet)
  singlecell_<label>/    main/  supp/  table/          (single-cell runs; own output_dir)
  run_summary.tsv        one row per dataset x stage
```

Every `main/`, `supp/` and `table/` folder gets a `panel_manifest.tsv` listing its files.
Figures are written as PDF and PNG (see `style.formats`). If you rename or drop a panel, delete the
old files in the output folders yourself; the scripts only add and overwrite.

## Unified colors

| Meaning | Color |
|---|---|
| DIA-NN / DIA-NN reported | blue `#4C78A8` |
| JUMPlion | red `#E45756` |
| JUMPlion recovered (values missing in DIA-NN) | green `#59A14F` |
| Human / Yeast / E. coli | green / orange / purple |
| MS1 isotope peak / MS2 co-elution / unresolved | gold `#DDAA33` / red / grey `#BDBDBD` |

## Notes on the vendored scripts

`diff_expression.py` is the DE script with minimal fixes: `plot_volcano` no longer fails when a
contrast has zero calls of some outcome (counts use `.get(key, 0)`, the FDR division is guarded),
`add_all_protein_names` / `define_species` are defined in the file (FASTA via Biopython), and an
unused `import polars` was removed. Analysis logic is unchanged. `precursor_features.py` is unchanged.
