#!/usr/bin/env Rscript
# Impute the DIA-NN protein matrix of each generalization dataset with KNN, missForest and MinProb, so the
# imputed DIA-NN tables can go through the same differential-expression benchmark as DIA-NN and JUMPlion.
#
#   Rscript r_scripts/impute_protein_matrices.R <gen_beta dir> [datasets, comma separated] [methods, comma separated] [seed]
#   e.g. Rscript r_scripts/impute_protein_matrices.R /path/to/gen_beta ultra_5m,ultra_15m,astral_5m,astral_15m knn,missforest,minprob
#
# Reads  <dir>/<dataset>/publication/prot_matrix_diann.tsv  (linear intensities; empty / 0 = missing)
# Writes <dir>/<dataset>/publication/prot_matrix_diann_<method>.tsv  in the same layout (annotation columns kept,
# linear intensities), so differential_expression's "log_transform_input": true treats them like the originals.
#
# Imputation is done on log2 intensities, across the whole protein x sample matrix (global, not per sample):
#   knn        impute::impute.knn, k = 10 nearest proteins
#   missforest missForest::missForest, proteins = observations, samples = variables
#   minprob    imputeLCMD::impute.MinProb (q = 0.01, tune.sigma = 1): per sample, N(1st percentile of that sample,
#              median sd of the proteins observed in > 50% of samples)
# Only values that are missing are replaced; observed values are written back unchanged.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("usage: Rscript r_scripts/impute_protein_matrices.R <gen_beta dir> [datasets] [methods] [seed]")
root <- args[1]
datasets <- if (length(args) >= 2) strsplit(args[2], ",")[[1]] else c("ultra_5m", "ultra_15m", "astral_5m", "astral_15m")
methods <- if (length(args) >= 3) strsplit(args[3], ",")[[1]] else c("knn", "missforest", "minprob")
SEED <- if (length(args) >= 4) as.integer(args[4]) else 1L
ANNOT <- c("Protein.Group", "Protein.Names", "Genes", "Precursors", "n.Precursors", "Peptides", "n.Peptides")

need <- function(pkg, bioc = FALSE) {
  if (requireNamespace(pkg, quietly = TRUE)) return(invisible(TRUE))
  message("installing ", pkg)
  if (bioc) {
    if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager", repos = "https://cloud.r-project.org")
    BiocManager::install(pkg, update = FALSE, ask = FALSE)
  } else install.packages(pkg, repos = "https://cloud.r-project.org")
  requireNamespace(pkg, quietly = TRUE) || stop("could not install ", pkg)
}

impute_knn <- function(x) {
  need("impute", bioc = TRUE)
  set.seed(SEED)
  impute::impute.knn(x, k = 10, rowmax = 0.9, colmax = 0.9)$data  # rows = proteins: neighbours are similar proteins
}

impute_missforest <- function(x) {
  need("missForest")
  set.seed(SEED)
  missForest::missForest(x, ntree = 100, verbose = FALSE)$ximp
}

impute_minprob <- function(x) {
  need("imputeLCMD")
  set.seed(SEED)
  invisible(capture.output(imp <- imputeLCMD::impute.MinProb(x, q = 0.01, tune.sigma = 1)))  # it prints the sd
  imp
}

for (ds in datasets) {
  f <- file.path(root, ds, "publication", "prot_matrix_diann.tsv")
  if (!file.exists(f)) { warning("missing: ", f); next }
  tab <- data.table::fread(f, sep = "\t", data.table = FALSE, check.names = FALSE)
  samples <- setdiff(names(tab), ANNOT)
  lin <- as.matrix(tab[, samples, drop = FALSE])
  lin[!is.finite(lin) | lin <= 0] <- NA
  x <- log2(lin)
  rownames(x) <- tab$Protein.Group
  miss <- is.na(x)
  message(sprintf("%s: %d proteins x %d samples, %.1f%% missing", ds, nrow(x), ncol(x), 100 * mean(miss)))
  for (m in methods) {
    imp <- switch(m, knn = impute_knn(x), missforest = impute_missforest(x), minprob = impute_minprob(x),
                  stop("unknown method: ", m))
    imp <- as.matrix(imp)
    x_out <- x
    x_out[miss] <- imp[miss]  # observed values are never changed
    out <- tab
    out[, samples] <- 2^x_out
    fo <- file.path(root, ds, "publication", sprintf("prot_matrix_diann_%s.tsv", m))
    data.table::fwrite(out, fo, sep = "\t", na = "")
    message(sprintf("  %-10s -> %s (%d values imputed)", m, fo, sum(miss)))
  }
}
