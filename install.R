# R dependencies for the Everyday Causal Inference code examples
# Run with: Rscript install.R   (or source("install.R") from an R session)

# Use the official CRAN cloud mirror if none is configured (required for Rscript)
if (getOption("repos")["CRAN"] %in% c(NA, "@CRAN@")) {
  options(repos = c(CRAN = "https://cloud.r-project.org"))
}

install.packages(c(
  "tidyverse",
  "fixest",
  "did",
  "plm",
  "AER",
  "broom",
  "forecast",
  "pwr",
  "RCT",
  "rdrobust",
  "rddensity",
  "lpdensity",
  "CausalImpact",
  "skimr",
  "gridExtra",
  "patchwork",
  "grf",
  "policytree",
  "DiagrammeR"  # policytree suggests it
))

# CausalArima is not on CRAN; install it from GitHub
if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes")
remotes::install_github("FMenchetti/CausalArima", dependencies = TRUE)
