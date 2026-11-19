#!/usr/bin/env Rscript
# Installs the R packages the imputation_comparison scripts need (skips the ones already installed).
#   Rscript r_scripts/install_packages.R
# Versions used for the manuscript analysis: R 4.4.3; impute 1.80.0 (Bioconductor 3.20), missForest 1.6.1,
# randomForest 4.7.1.2, imputeLCMD 2.1, pcaMethods 1.98.0, data.table 1.18.2.1, ggplot2 4.0.3, patchwork 1.3.2,
# systemfonts 1.3.2, ragg 1.5.2. Figures use the Arial font (the Python figures do too) and cairo for the PDFs.
repos <- "https://cloud.r-project.org"
cran <- c("data.table", "ggplot2", "patchwork", "missForest", "systemfonts", "ragg")
bioc <- c("impute", "pcaMethods")  # imputeLCMD needs pcaMethods; imputeLCMD itself is on CRAN
for (p in cran) if (!requireNamespace(p, quietly = TRUE)) install.packages(p, repos = repos)
if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager", repos = repos)
for (p in bioc) if (!requireNamespace(p, quietly = TRUE)) BiocManager::install(p, update = FALSE, ask = FALSE)
if (!requireNamespace("imputeLCMD", quietly = TRUE)) install.packages("imputeLCMD", repos = repos)
