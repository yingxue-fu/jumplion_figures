#!/usr/bin/env Rscript
# One supplementary figure from the four per-dataset tables compare_imputation.R writes:
# DIA-NN vs KNN / missForest / MinProb imputation of DIA-NN vs JUMPlion, sensitivity and empirical FDR, every dataset.
#
#   Rscript r_scripts/compare_imputation_combined.R <analyses dir>
#
# Figure F (both contrasts, panels a and b) and figure G (restored proteins). Columns = datasets, rows = sensitivity / FDR, one block per stratum. Under each contrast: the number of proteins
# scored. Partially complete: the same proteins for every method. Insufficient: the imputations score every
# protein with < 3 observations in a group in DIA-NN; JUMPlion only the ones its matrix makes testable (in brackets).

suppressMessages({library(data.table); library(ggplot2); library(patchwork)})
script_dir <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
source(file.path(script_dir, "jumplion_theme.R"))  # Python-matching ggplot theme (Arial 8 pt etc.)
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("usage: Rscript r_scripts/compare_imputation_combined.R <analyses dir>")
ana <- args[1]
DATASETS <- c(astral_5m = "Astral\n5 min", astral_15m = "Astral\n15 min", ultra_5m = "timsTOF\n5 min", ultra_15m = "timsTOF\n15 min")  # two-line strips fit the narrow facets
COLORS <- c("DIA-NN" = "#4C78A8", "KNN" = "#59A14F", "missForest" = "#F28E2B", "MinProb" = "#B07AA1", "JUMPlion" = "#E45756")

res <- rbindlist(lapply(names(DATASETS), function(d)
  fread(file.path(ana, "generalization", "table", sprintf("imputation_comparison_%s.tsv", d)))))
res[, method := factor(method, names(COLORS))]
res[, dataset_lab := factor(DATASETS[dataset], unname(DATASETS))]

# x labels: contrast + number of proteins scored
fmt <- function(n) formatC(n, format = "d", big.mark = ",")
lab <- res[, {
  d <- .SD[method == "DIA-NN"]; j <- .SD[method == "JUMPlion"]
  .(x_lab = if (stratum[1] == "Partially complete") sprintf("%s\nn = %s", sub("_vs_", " vs ", contrast[1]), fmt(d$n_tested))
            else if (stratum[1] == "Insufficient, restored by JUMPlion") sprintf("%s\nn = %s", sub("_vs_", " vs ", contrast[1]), fmt(j$n_tested))
            else sprintf("%s\nn = %s\n(JUMPlion %s)", sub("_vs_", " vs ", contrast[1]), fmt(d$n_proteins), fmt(j$n_tested)))
}, by = .(dataset, contrast, stratum)]
res <- merge(res, lab, by = c("dataset", "contrast", "stratum"))

res[, false_positives := FP + wrong_direction]  # same definition as the numerator of the empirical FDR

draw <- function(ct, st, title, show_legend, counts = FALSE) {
  x <- res[contrast == ct & stratum == st]
  mv <- c("sensitivity", "empirical_fdr", if (counts) "TP")
  ml <- c("Sensitivity (%)", "Empirical FDR (%)", if (counts) "True positives (n)")
  long <- melt(x, id.vars = c("dataset_lab", "x_lab", "method", "n_tested"),
               measure.vars = mv, variable.name = "metric", value.name = "value")
  long[, metric := factor(metric, mv, ml)]
  long[, lab := ifelse(is.na(value) & n_tested == 0, "n/a", ifelse(is.na(value), "n<20", ""))]
  ggplot(long, aes(x_lab, value, fill = method)) +
    geom_col(position = position_dodge(.8), width = .75, colour = "#4D4D4D", linewidth = LW_PATCH, na.rm = TRUE) +
    geom_text(aes(label = lab, y = 0), position = position_dodge(.8), vjust = -.4, hjust = 0, angle = 90, size = pt_mm(5.5),
              colour = "#4D4D4D") +
    facet_grid(metric ~ dataset_lab, scales = "free", switch = "y") +
    scale_fill_manual(values = COLORS, name = NULL, drop = FALSE) +
    scale_y_continuous(expand = expansion(mult = c(0, .08)), limits = c(0, NA)) +
    labs(x = NULL, y = NULL, title = title) +
    theme_jumplion() +
    theme(axis.text.x = element_text(lineheight = .9), panel.spacing = unit(.4, "lines"),
          legend.position = if (show_legend) "top" else "none")
}
dir.create(file.path(ana, "generalization", "supp"), recursive = TRUE, showWarnings = FALSE)
# Restored proteins, both contrasts in one figure (a: B vs C on top, b: A vs B below)
restored <- function(ct, legend) draw(ct, "Insufficient, restored by JUMPlion",
                                      paste(sub("_vs_", " vs ", ct), "- Insufficient, restored by JUMPlion (same proteins for every method)"), legend)
pr <- (restored("B_vs_C", TRUE) / restored("A_vs_B", FALSE)) + plot_layout(guides = "collect") + plot_annotation(tag_levels = "a") &
  theme(legend.position = "top", plot.tag = element_text(size = FS + 1, face = "bold"))
stem <- file.path(ana, "generalization", "supp", "Supp_Generalization_G_imputation_comparison_restored")
save_fig(stem, pr, 6, 6.8)
cat("wrote", paste0(stem, ".png"), "
")

# Both contrasts in one figure (a: B vs C, b: A vs B), under 8 in wide; one shared legend in its own row above the panel letters
f_block <- function(ct, tag) {
  nm <- sub("_vs_", " vs ", ct)
  wrap_plots(list(draw(ct, "Partially complete", paste(nm, "- Partially complete"), TRUE) + labs(tag = tag),
                  draw(ct, "Insufficient", paste(nm, "- Insufficient"), FALSE, counts = TRUE)), ncol = 1, heights = c(2, 3))
}
pc <- guide_area() / (f_block("B_vs_C", "a") | f_block("A_vs_B", "b")) +
  plot_layout(guides = "collect", heights = c(0.05, 1)) &
  theme(plot.tag = element_text(size = FS + 1, face = "bold"))
stem <- file.path(ana, "generalization", "supp", "Supp_Generalization_F_imputation_comparison_all_datasets_combined")
save_fig(stem, pc, 7.8, 8)
cat("wrote", paste0(stem, ".png"), "\n")
