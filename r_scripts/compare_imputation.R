#!/usr/bin/env Rscript
# JUMPlion vs imputation of DIA-NN: sensitivity / empirical FDR on the HYE benchmark, protein level, one dataset (writes the table only;
# the figures come from compare_imputation_combined.R and compare_imputation_curves.R).
#
#   Rscript r_scripts/compare_imputation.R <gen_beta dir> <dataset> [analyses dir]
#   e.g. Rscript r_scripts/compare_imputation.R /path/to/gen_beta ultra_5m /path/to/gen_beta/analyses4_protein
#
# Needs <dir>/<dataset>/publication/prot_matrix_{diann, diann_knn, diann_missforest, diann_minprob, diann_jumplion}.tsv
# (impute_protein_matrices.R makes the three imputed ones) and <dir>/sample_meta_<dataset>.csv (Run, Group2 = A/B/C).
#
# Mirrors the Python benchmark (performance.py / diff_expression.py) so the numbers are comparable:
#   * Welch t-test per protein on log2 intensities, BH-adjusted, p_adj < 0.05, no log2FC cutoff; a protein is tested
#     when it has <= 50% missing values in each group (an imputed matrix has none, so every protein is tested)
#   * truth: HUMAN unchanged, YEAST up and ECOLI down in the first group of the contrast; wrong-direction calls
#     count as false discoveries; sensitivity = TP / (TP + FN + wrong), FDR = (FP + wrong) / (TP + FP + wrong),
#     each reported only with >= 20 in its denominator
#   * strata are defined by the ORIGINAL DIA-NN observations, as in the paper's completeness classes:
#       Partially complete: >= 3 observed in both groups, but not all 6 in both
#       Insufficient: < 3 observed in a group in DIA-NN (all of them; each method scores the ones it can test: JUMPlion
#       the proteins its matrix makes testable, i.e. the restored ones; the imputations every one, as they fill all gaps)
#     DIA-NN itself cannot test the Insufficient ones. Partially complete: every method is scored on the same proteins.

suppressMessages(library(data.table))
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: Rscript r_scripts/compare_imputation.R <gen_beta dir> <dataset> [analyses dir]")
root <- args[1]; ds <- args[2]
ana <- if (length(args) >= 3) args[3] else file.path(root, "analyses4_protein")
pub <- file.path(root, ds, "publication")

PADJ <- 0.05; MIN_N <- 20; GROUP_SIZE <- 6; MIN_OBS <- 3
CONTRASTS <- list(A_vs_B = c("A", "B"), B_vs_C = c("B", "C"))  # first group = numerator of log2FC
METHODS <- c("DIA-NN" = "diann", "KNN" = "diann_knn", "missForest" = "diann_missforest",
             "MinProb" = "diann_minprob", "JUMPlion" = "diann_jumplion")
COLORS <- c("DIA-NN" = "#4C78A8", "KNN" = "#59A14F", "missForest" = "#F28E2B", "MinProb" = "#B07AA1", "JUMPlion" = "#E45756")

meta <- fread(file.path(root, sprintf("sample_meta_%s.csv", ds)))
read_mat <- function(tag) {
  tab <- fread(file.path(pub, sprintf("prot_matrix_%s.tsv", tag)), sep = "\t", data.table = FALSE, check.names = FALSE)
  m <- as.matrix(tab[, meta$Run, drop = FALSE]); rownames(m) <- tab$Protein.Group
  m[!is.finite(m) | m <= 0] <- NA
  list(log2 = log2(m), info = tab[, setdiff(names(tab), meta$Run), drop = FALSE])
}
mats <- lapply(METHODS, read_mat)
ids <- rownames(mats[["DIA-NN"]]$log2)  # every matrix is aligned to the DIA-NN protein order (JUMPlion rows are ordered differently)
for (k in names(mats)) mats[[k]]$log2 <- mats[[k]]$log2[match(ids, rownames(mats[[k]]$log2)), , drop = FALSE]

info <- mats[["DIA-NN"]]$info
sp <- sapply(strsplit(info$Protein.Names, ";"), function(x) {
  s <- unique(sub("^[^_]*_", "", trimws(x))); s <- s[s %in% c("HUMAN", "YEAST", "ECOLI")]; if (length(s) == 1) s else NA_character_
})
names(sp) <- info$Protein.Group
keep <- !is.na(sp)  # clean benchmark universe: single-organism proteins only
cat(sprintf("%s: %d proteins, %d in the clean HUMAN/YEAST/ECOLI universe\n", ds, length(sp), sum(keep)))

cols <- function(g) meta$Run[meta$Group2 == g]
nobs <- function(x, g) rowSums(is.finite(x[, cols(g), drop = FALSE]))

de <- function(x, g1, g2) {  # Welch t-test per row, BH over testable rows
  a <- x[, cols(g1), drop = FALSE]; b <- x[, cols(g2), drop = FALSE]
  na <- rowSums(is.finite(a)); nb <- rowSums(is.finite(b))
  ma <- rowMeans(a, na.rm = TRUE); mb <- rowMeans(b, na.rm = TRUE)
  va <- apply(a, 1, var, na.rm = TRUE); vb <- apply(b, 1, var, na.rm = TRUE)
  se2 <- va / na + vb / nb
  t <- (ma - mb) / sqrt(se2)
  df <- se2^2 / ((va / na)^2 / (na - 1) + (vb / nb)^2 / (nb - 1))
  p <- 2 * pt(-abs(t), df)
  testable <- (1 - na / ncol(a) <= 0.5) & (1 - nb / ncol(b) <= 0.5)
  p[!testable | na < 2 | nb < 2] <- NA
  padj <- rep(NA_real_, length(p)); ok <- !is.na(p); padj[ok] <- p.adjust(p[ok], "BH")
  data.frame(id = rownames(x), log2FC = ma - mb, padj = padj, testable = testable & !is.na(p), stringsAsFactors = FALSE)
}

state <- function(species, fc, padj) {
  sig <- !is.na(padj) & padj < PADJ
  s <- rep(NA_character_, length(fc))
  s[species == "HUMAN"] <- ifelse(sig[species == "HUMAN"], "FP", "TN")
  y <- species == "YEAST"; e <- species == "ECOLI"
  s[y] <- ifelse(!sig[y], "FN", ifelse(fc[y] > 0, "TP", "wrong"))
  s[e] <- ifelse(!sig[e], "FN", ifelse(fc[e] < 0, "TP", "wrong"))
  s
}

rows <- list()
for (cn in names(CONTRASTS)) {
  g1 <- CONTRASTS[[cn]][1]; g2 <- CONTRASTS[[cn]][2]
  d_obs <- list(nobs(mats[["DIA-NN"]]$log2, g1), nobs(mats[["DIA-NN"]]$log2, g2))
  j_ok <- (1 - nobs(mats[["JUMPlion"]]$log2, g1) / length(cols(g1)) <= 0.5) & (1 - nobs(mats[["JUMPlion"]]$log2, g2) / length(cols(g2)) <= 0.5)
  strata <- list(
    "Partially complete" = keep & d_obs[[1]] >= MIN_OBS & d_obs[[2]] >= MIN_OBS & !(d_obs[[1]] >= GROUP_SIZE & d_obs[[2]] >= GROUP_SIZE),
    "Insufficient" = keep & (d_obs[[1]] < MIN_OBS | d_obs[[2]] < MIN_OBS),
    "Insufficient, restored by JUMPlion" = keep & (d_obs[[1]] < MIN_OBS | d_obs[[2]] < MIN_OBS) & j_ok,  # the same proteins for every method
    "All proteins" = keep)  # every protein the method can test (strata pooled, incl. complete); not in the per-dataset figure
  for (meth in names(METHODS)) {
    r <- de(mats[[meth]]$log2, g1, g2)
    for (st in names(strata)) {
      sel <- strata[[st]] & r$testable
      s <- state(sp[sel], r$log2FC[sel], r$padj[sel])
      n <- function(k) sum(s == k, na.rm = TRUE)
      TP <- n("TP"); FP <- n("FP"); FN <- n("FN"); TN <- n("TN"); W <- n("wrong")
      rows[[length(rows) + 1]] <- data.frame(
        dataset = ds, contrast = cn, stratum = st, method = meth, n_proteins = sum(strata[[st]]), n_tested = sum(sel),
        TP, FP, FN, TN, wrong_direction = W,
        sensitivity = if (TP + FN + W >= MIN_N) 100 * TP / (TP + FN + W) else NA_real_,
        empirical_fdr = if (TP + FP + W >= MIN_N) 100 * (FP + W) / (TP + FP + W) else NA_real_)
    }
  }
}
res <- rbindlist(rows)
res[, method := factor(method, levels = names(METHODS))]
res[, contrast_lab := sub("_vs_", " vs ", contrast)]
dir.create(file.path(ana, "generalization", "table"), recursive = TRUE, showWarnings = FALSE)
fwrite(res, file.path(ana, "generalization", "table", sprintf("imputation_comparison_%s.tsv", ds)), sep = "\t")
print(as.data.frame(res[, .(contrast, stratum, method, n_proteins, n_tested, TP, FP, FN, wrong_direction,
                            sensitivity = round(sensitivity, 1), empirical_fdr = round(empirical_fdr, 1))]), row.names = FALSE)

