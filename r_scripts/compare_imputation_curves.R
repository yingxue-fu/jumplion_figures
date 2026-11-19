#!/usr/bin/env Rscript
# Empirical FDR vs sensitivity curves for DIA-NN, KNN / missForest / MinProb imputation of DIA-NN and JUMPlion,
# traced by sweeping the adjusted p-value cutoff (HYE benchmark, protein level, all four generalization datasets).
#
#   Rscript r_scripts/compare_imputation_curves.R <gen_beta dir> [analyses dir]
#   analyses dir defaults to <gen_beta dir>/analyses4_protein
#
# Same statistics, truth and strata as compare_imputation.R (see there); only the cutoff changes. The padj < 0.05
# operating point of each curve is marked. A point is dropped when fewer than 20 proteins are called (FDR denominator).
# Strata: Partially complete, Insufficient (every method scores the proteins it can test), as in figure F.

suppressMessages({library(data.table); library(ggplot2); library(patchwork)})
script_dir <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
source(file.path(script_dir, "jumplion_theme.R"))  # Python-matching ggplot theme (Arial 8 pt etc.)
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("usage: Rscript r_scripts/compare_imputation_curves.R <gen_beta dir> [analyses dir]")
root <- args[1]
ana <- if (length(args) >= 2) args[2] else file.path(root, "analyses4_protein")

DATASETS <- c(astral_5m = "Astral, 5 min", astral_15m = "Astral, 15 min", ultra_5m = "timsTOF, 5 min", ultra_15m = "timsTOF, 15 min")
CONTRASTS <- list(A_vs_B = c("A", "B"), B_vs_C = c("B", "C"))
METHODS <- c("DIA-NN" = "diann", "KNN" = "diann_knn", "missForest" = "diann_missforest",
             "MinProb" = "diann_minprob", "JUMPlion" = "diann_jumplion")
COLORS <- c("DIA-NN" = "#4C78A8", "KNN" = "#59A14F", "missForest" = "#F28E2B", "MinProb" = "#B07AA1", "JUMPlion" = "#E45756")
CUTOFFS <- sort(unique(c(10^seq(-4, 0, length.out = 80), 0.05)))
MIN_N <- 20; GROUP_SIZE <- 6; MIN_OBS <- 3

de <- function(x, cols1, cols2) {  # Welch t-test per row; padj = BH over testable rows
  a <- x[, cols1, drop = FALSE]; b <- x[, cols2, drop = FALSE]
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
  data.frame(log2FC = ma - mb, padj = padj, testable = testable & !is.na(p))
}

curve_rows <- list()
for (ds in names(DATASETS)) {
  pub <- file.path(root, ds, "publication")
  meta <- fread(file.path(root, sprintf("sample_meta_%s.csv", ds)))
  mats <- lapply(METHODS, function(tag) {
    tab <- fread(file.path(pub, sprintf("prot_matrix_%s.tsv", tag)), sep = "\t", data.table = FALSE, check.names = FALSE)
    m <- as.matrix(tab[, meta$Run, drop = FALSE]); rownames(m) <- tab$Protein.Group
    m[!is.finite(m) | m <= 0] <- NA
    list(log2 = log2(m), info = tab[, setdiff(names(tab), meta$Run), drop = FALSE])
  })
  ids <- rownames(mats[["DIA-NN"]]$log2)  # align every matrix to the DIA-NN protein order
  for (k in names(mats)) mats[[k]]$log2 <- mats[[k]]$log2[match(ids, rownames(mats[[k]]$log2)), , drop = FALSE]
  info <- mats[["DIA-NN"]]$info
  sp <- sapply(strsplit(info$Protein.Names, ";"), function(x) {
    s <- unique(sub("^[^_]*_", "", trimws(x))); s <- s[s %in% c("HUMAN", "YEAST", "ECOLI")]; if (length(s) == 1) s else NA_character_
  })
  keep <- !is.na(sp)
  cols <- function(g) meta$Run[meta$Group2 == g]
  nobs <- function(x, g) rowSums(is.finite(x[, cols(g), drop = FALSE]))

  for (cn in names(CONTRASTS)) {
    g1 <- CONTRASTS[[cn]][1]; g2 <- CONTRASTS[[cn]][2]
    d1 <- nobs(mats[["DIA-NN"]]$log2, g1); d2 <- nobs(mats[["DIA-NN"]]$log2, g2)
    strata <- list("Partially complete" = keep & d1 >= MIN_OBS & d2 >= MIN_OBS & !(d1 >= GROUP_SIZE & d2 >= GROUP_SIZE),
                   "Insufficient" = keep & (d1 < MIN_OBS | d2 < MIN_OBS))
    for (meth in names(METHODS)) {
      r <- de(mats[[meth]]$log2, cols(g1), cols(g2))
      for (st in names(strata)) {
        sel <- strata[[st]] & r$testable
        if (!any(sel)) next
        s <- sp[sel]; fc <- r$log2FC[sel]; pa <- r$padj[sel]
        up <- (s == "YEAST" & fc > 0) | (s == "ECOLI" & fc < 0)  # right-direction change
        truth_de <- s != "HUMAN"
        for (cut in CUTOFFS) {
          sig <- pa < cut
          TP <- sum(sig & truth_de & up); W <- sum(sig & truth_de & !up); FP <- sum(sig & !truth_de)
          pos <- sum(truth_de)  # TP + FN + wrong
          curve_rows[[length(curve_rows) + 1]] <- data.table(
            dataset = ds, contrast = cn, stratum = st, method = meth, cutoff = cut, TP, FP, wrong_direction = W,
            sensitivity = if (pos >= MIN_N) 100 * TP / pos else NA_real_,
            empirical_fdr = if (TP + FP + W >= MIN_N) 100 * (FP + W) / (TP + FP + W) else NA_real_)
        }
      }
    }
  }
  cat("done", ds, "\n")
}
cur <- rbindlist(curve_rows)
fwrite(cur, file.path(ana, "generalization", "table", "imputation_fdr_sensitivity_curves.tsv"), sep = "\t")

cur[, method := factor(method, names(COLORS))]
cur[, dataset_lab := factor(DATASETS[dataset], unname(DATASETS))]
cur[, stratum := factor(stratum, c("Partially complete", "Insufficient"))]
dir.create(file.path(ana, "generalization", "supp"), recursive = TRUE, showWarnings = FALSE)
# Same sweep, y = number of true positives (the strata hold different numbers of proteins).
# facet_grid shares a y scale across a whole row, so Astral and timsTOF are drawn as two sub-plots side by side: each
# has its own y axis (shared by the two gradients of that instrument within a stratum row).
# One figure, a: B vs C (top), b: A vs B (bottom).
tp_row <- function(ct) {
  x <- cur[contrast == ct & !is.na(empirical_fdr)]
  sub_plot <- function(dss, first) {
    xs <- x[dataset %in% dss]
    ggplot(xs, aes(empirical_fdr, TP, colour = method)) +
      geom_vline(xintercept = 5, linetype = "dashed", colour = "grey60", linewidth = LW_AXIS) +
      geom_path(linewidth = LW_LINE) +
      geom_point(data = xs[abs(cutoff - 0.05) < 1e-12], size = 4 / ggplot2::.pt) +
      facet_grid(stratum ~ dataset_lab, scales = "free", switch = "y") +
      scale_colour_manual(values = COLORS, name = NULL, drop = FALSE) +
      scale_y_continuous(limits = c(0, NA), expand = expansion(mult = c(0.02, 0.05))) +
      labs(x = "Empirical FDR (%)", y = if (first) "True positives (n)" else NULL) +
      theme_jumplion() +
      theme(strip.text.y = if (first) element_text(size = FS) else element_blank(), panel.spacing = unit(.5, "lines"),
            legend.position = "top")
  }
  (sub_plot(c("astral_5m", "astral_15m"), TRUE) | sub_plot(c("ultra_5m", "ultra_15m"), FALSE)) +
    plot_layout(guides = "collect") +
    plot_annotation(title = sprintf("%s (dot: padj < 0.05; dashed: 5%% FDR)", sub("_vs_", " vs ", ct)),
                    theme = theme(plot.title = element_text(size = FS + 1, face = "bold", hjust = 0))) &
    theme(legend.position = "top")
}
p <- (wrap_elements(tp_row("B_vs_C")) / wrap_elements(tp_row("A_vs_B"))) + plot_layout(guides = "collect") +
  plot_annotation(tag_levels = "a") & theme(plot.tag = element_text(size = FS + 1, face = "bold"))
stem <- file.path(ana, "generalization", "supp", "Supp_Generalization_H_imputation_fdr_tp_curves")
save_fig(stem, p, 7.2, 8.4)
cat("wrote", paste0(stem, ".png"), "\n")
