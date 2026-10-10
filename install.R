# R dependencies for the Everyday Causal Inference code examples
# Run with: Rscript install.R   (or source("install.R") from an R session)

# Use the official CRAN cloud mirror if none is configured (required for Rscript)
if (getOption("repos")["CRAN"] %in% c(NA, "@CRAN@")) {
  options(repos = c(CRAN = "https://cloud.r-project.org"))
}

packages <- c(
  "tidyverse",
  "fixest",
  "did",
  "plm",
  "AER",
  "broom",
  "forecast",
  "tidybayes",  # CausalArima imports it but omits it from DESCRIPTION
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
  "sandwich",
  "MatchIt",
  "WeightIt",
  "cobalt",
  "sensemakr",
  "dagitty",
  "lmtest",
  "readr",
  "DiagrammeR"  # policytree suggests it
)
install.packages(packages)

# CausalArima is not on CRAN; install it from GitHub
if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes")
remotes::install_github("FMenchetti/CausalArima", dependencies = TRUE)

required_packages <- c(packages, "remotes", "CausalArima")
missing_packages <- required_packages[!vapply(required_packages, requireNamespace,
                                            logical(1), quietly = TRUE)]
if (length(missing_packages)) {
  stop("Installation failed: could not load ", paste(missing_packages, collapse = ", "))
}
