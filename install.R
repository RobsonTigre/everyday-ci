# R dependencies for the Everyday Causal Inference code examples
# Run with: Rscript install.R   (or source("install.R") from an R session)

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
  "patchwork"
))

# CausalArima is not on CRAN; install it from GitHub
if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes")
remotes::install_github("FMenchetti/CausalArima", dependencies = TRUE)
