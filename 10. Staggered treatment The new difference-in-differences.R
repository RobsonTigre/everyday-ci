##########################################################
## Everyday Causal Inference: How to estimate, test, and explain impacts with R and Python
## www.everydaycausal.com
## Copyright (c) 2025 by Robson Tigre. All rights reserved.
## You may read, run, adapt, and cite this code, provided you credit the source.
## It should not be used to create competing educational or commercial products
##########################################################
## Code for Chapter 10 - Staggered treatment: The new difference-in-differences
## Created: May 08, 2026
## Last modified: 2026-05-18
##########################################################

# ---- Libraries (union of all four sections) ---------------------------------
# If you haven't already, run this once to install the packages:
# install.packages(c("tidyverse", "fixest", "did", "broom", "patchwork"))

suppressPackageStartupMessages({
  library(tidyverse)
  library(fixest)
  library(did)
  library(broom)
  library(patchwork)
})

# ---- Project theme + palette (shared by Sections 3 and 4) -------------------
book_colors <- list(
  primary    = "#2E86AB",
  secondary  = "#A23B72",
  accent     = "#F18F01",
  success    = "#C73E1D",
  muted      = "#6C757D",
  light_gray = "grey90",
  dark_gray  = "grey30"
)

theme_book <- function(base_size = 12) {
  theme_minimal(base_size = base_size) +
    theme(
      plot.title = element_text(face = "bold", color = book_colors$dark_gray),
      plot.subtitle = element_text(color = book_colors$muted),
      axis.title = element_text(color = book_colors$dark_gray),
      panel.grid.minor = element_blank(),
      legend.position = "bottom"
    )
}

# Cohort series colors. Pastel tones of blue / radish / forest green / violet,
# kept separate from book_colors$primary/$secondary/$accent/$success so the
# cohort-trend figures in Sections 3 and 4 stay visually consistent without
# disturbing the other figures' palette.
cohort_color_5 <- "#5C8FAF" # deeper pastel steel blue
cohort_color_7 <- "#D77A91" # deeper pastel radish
cohort_color_9 <- "#5E9A78" # deeper pastel forest green (sage)
cohort_color_11 <- "#A07AC1" # deeper pastel violet (lavender)

cohort_palette <- c(
  "Treated in period 5"  = cohort_color_5,
  "Treated in period 7"  = cohort_color_7,
  "Treated in period 9"  = cohort_color_9,
  "Treated in period 11" = cohort_color_11,
  "Never treated"        = book_colors$muted
)


# =============================================================================
# SECTION 1 — DATA: main DGP  (writes data/staggered_did.csv)
# -----------------------------------------------------------------------------
# Design choices (and why):
#   1. Unequal cohort sizes (700 / 500 / 300 / 200 treated + 800 never-treated)
#      following a "front-loaded" rollout: the chain pilots big in its flagship
#      state, then decelerates. This makes the simple-vs-cohort-size-weighted
#      aggregation gap visible — a teachable moment.
#   2. AR(1) marginal SD set to 4.5 so event-study CIs are visibly wide and
#      representative of a real retail panel.
#   3. tau_schedule slopes (0.50, 0.40, 0.28, 0.16). Front-loading amplifies
#      TWFE contamination via g=5's larger contribution to forbidden
#      comparisons; this slope range keeps the bias inside the [25%, 35%]
#      attenuation target relative to the CS overall ATT.
#
# Calibration targets (hard = stopifnot, soft = warning):
#   HARD: sign(TWFE) == sign(true ATT)
#   HARD: |TWFE - CS_simple| / CS_simple in [0.25, 0.35]
#   HARD: ATT(g=5) is the largest of {ATT(g=5), ATT(g=7), ATT(g=9), ATT(g=11)}
#   HARD: ATT(g=11) <= 0.85 * ATT(g=5)   (early-adopter dominance is visible)
#   SOFT: SE on CS_simple in [0.13, 0.25]
#   SOFT: |CS_simple - CS_group| / true_att >= 0.02
#   SOFT: SE(g=11) > SE(g=5)
#   SOFT: strict monotone ATT(g=5) > ATT(g=7) > ATT(g=9) > ATT(g=11)
#
# Seed selection: seed 64 was chosen by a seed search over [43, 80] as the
# unique seed in that range that passes all four hard stops AND yields a
# strictly monotone-decreasing ATT(g). Under seed 64: ATT(g) = 3.56, 2.81,
# 2.67, 2.23 (point estimates from bstrap=FALSE search; full-bootstrap run
# may differ slightly in SE but not in point estimates).
# =============================================================================

set.seed(64)

# ---- Panel structure --------------------------------------------------------
# Front-loaded rollout: the chain pilots big in its flagship state (PB) where it
# has the most stores and the deepest ops support, then decelerates as it
# extends into smaller markets. The 3.5x ratio between g=5 (700) and g=11
# (200) makes cohort-size weighting in aggte() visibly matter.
#   g=5  : 700 flagship pilot stores (PB)
#   g=7  : 500 secondary rollout     (RN)
#   g=9  : 300 mid-tier markets      (CE)
#   g=11 : 200 final batch           (PE)
#   g=0  : 800 never-treated holdout (BA)
cohort_sizes <- c("5" = 700, "7" = 500, "9" = 300, "11" = 200, "0" = 800)
cohorts <- as.integer(names(cohort_sizes))
states <- c("5" = "PB", "7" = "RN", "9" = "CE", "11" = "PE", "0" = "BA")
T <- 12

n_total <- sum(cohort_sizes)

stores <- tibble(
  store_id = seq_len(n_total),
  cohort = rep(cohorts, times = cohort_sizes),
  state = rep(states[as.character(cohorts)], times = cohort_sizes),
  store_FE = rnorm(n_total, mean = 10, sd = 3.5),
  pre_treatment_sales = rnorm(n_total, mean = 8, sd = 1.5)
)

# Cohort-level baseline shifts. Earlier rollout cohorts start at higher baseline
# sales because they are flagship urban stores. The shift is constant in time,
# so it differences out of every DiD and parallel trends in expectation still
# holds by construction.
cohort_intercept <- tibble(
  cohort = c(5, 7, 9, 11, 0),
  c_g    = c(2.0, 0.5, -0.5, -1.5, 0.0)
)

period_FE <- tibble(period = 1:T, period_FE = rnorm(T, mean = 0, sd = 0.3))

panel <- expand_grid(store_id = stores$store_id, period = 1:T) |>
  left_join(stores, by = "store_id") |>
  left_join(cohort_intercept, by = "cohort") |>
  left_join(period_FE, by = "period") |>
  mutate(
    relative_period = if_else(cohort == 0, NA_integer_, period - cohort),
    treated         = as.integer(cohort != 0 & period >= cohort)
  )

# AR(1) residuals at the store level. Marginal SD = 4.5 widens event-study CIs
# to a level representative of a real retail panel. rho = 0.3 captures modest
# month-to-month persistence. Innovation SD is rescaled so the stationary
# marginal SD matches the target.
ar1_panel <- function(n_stores, T, rho, sd_marginal) {
  sd_innov <- sd_marginal * sqrt(1 - rho^2)
  e <- matrix(0, nrow = n_stores, ncol = T)
  e[, 1] <- rnorm(n_stores, 0, sd_marginal) # stationary start
  innov <- matrix(rnorm(n_stores * (T - 1), 0, sd_innov), nrow = n_stores)
  for (t in 2:T) e[, t] <- rho * e[, t - 1] + innov[, t - 1]
  as.vector(t(e)) # store-major order
}

panel$ar1_error <- ar1_panel(n_total, T, rho = 0.3, sd_marginal = 4.5)

# ---- Heterogeneous treatment effect schedule --------------------------------
# tau(g, k) = base(g) + slope(g) * k for k >= 0 (relative_period); 0 otherwise.
# Later adopters get larger base; early adopters get steeper ramp. Mixing the
# two directions creates the cross-cohort heterogeneity that drives TWFE bias.
tau_schedule <- tibble(
  cohort = c(5, 7, 9, 11),
  base   = c(1.5, 1.8, 2.1, 2.4),
  slope  = c(0.50, 0.40, 0.28, 0.16)
)

panel_full <- panel |>
  left_join(tau_schedule, by = "cohort") |>
  mutate(
    tau = if_else(treated == 1, base + slope * relative_period, 0),
    sales = store_FE + c_g + period_FE + 0.4 * pre_treatment_sales +
      tau + ar1_error
  )

# Slim down to the columns we ship in the CSV.
panel <- panel_full |>
  select(
    store_id, period, state, pre_treatment_sales,
    cohort, relative_period, treated, sales
  )

# ---- Write CSV --------------------------------------------------------------
out_path <- "data/staggered_did.csv"
write_csv(panel, out_path)
message("Wrote ", out_path, " (", nrow(panel), " rows).")

# ---- Compute true ATT, CS estimates, and TWFE estimate ----------------------
true_att <- panel_full |>
  filter(treated == 1) |>
  summarise(true_att = mean(tau)) |>
  pull(true_att)

cs_fit <- att_gt(
  yname = "sales", tname = "period", idname = "store_id", gname = "cohort",
  data = panel,
  control_group = "nevertreated", est_method = "dr",
  bstrap = TRUE, cband = TRUE, clustervars = "store_id"
)

cs_simple <- aggte(cs_fit, type = "simple")
cs_group <- aggte(cs_fit, type = "group", cband = TRUE)
cs_calendar <- aggte(cs_fit, type = "calendar", cband = TRUE)
cs_dynamic <- aggte(cs_fit,
  type = "dynamic", cband = TRUE,
  min_e = -6, max_e = 7
)

# Also estimate with not-yet-treated as a sensitivity check (used in chapter S3.1)
cs_fit_notyet <- att_gt(
  yname = "sales", tname = "period", idname = "store_id", gname = "cohort",
  data = panel,
  control_group = "notyettreated", est_method = "dr",
  bstrap = TRUE, cband = TRUE, clustervars = "store_id",
  base_period = "universal", anticipation = 0
)
cs_notyet_simple <- aggte(cs_fit_notyet, type = "simple")

twfe_fit <- feols(
  sales ~ treated + pre_treatment_sales | store_id + period,
  data = panel, cluster = ~store_id
)
twfe_est <- coef(twfe_fit)[["treated"]]
twfe_se <- se(twfe_fit)[["treated"]]

bias_rel_cs <- abs(twfe_est - cs_simple$overall.att) / cs_simple$overall.att
bias_rel_true <- abs(twfe_est - true_att) / true_att
gap_abs <- abs(cs_simple$overall.att - cs_group$overall.att)
gap_pct <- gap_abs / true_att

# ---- Headline numbers for chapter prose -------------------------------------
# Cohort summary
cohort_summary <- panel |>
  distinct(store_id, cohort, state) |>
  count(cohort, state, name = "n_stores") |>
  arrange(cohort)
print(cohort_summary)

# Headline numbers
cat(sprintf("True overall ATT             : %.3f\n", true_att))
cat(sprintf(
  "CS overall (simple)          : %.3f (SE %.3f)\n",
  cs_simple$overall.att, cs_simple$overall.se
))
cat(sprintf(
  "CS overall (group, size-wtd) : %.3f (SE %.3f)\n",
  cs_group$overall.att, cs_group$overall.se
))
cat(sprintf(
  "|simple - group|             : %.3f  (%.1f%% of true ATT)\n",
  gap_abs, 100 * gap_pct
))
cat(sprintf(
  "CS overall (not-yet-treated) : %.3f (SE %.3f)\n",
  cs_notyet_simple$overall.att, cs_notyet_simple$overall.se
))
cat(sprintf(
  "TWFE                         : %.3f (SE %.3f)\n",
  twfe_est, twfe_se
))
cat(sprintf(
  "|TWFE - CS_simple|/CS_simple : %.3f  (target [0.25, 0.35])\n",
  bias_rel_cs
))
cat(sprintf("|TWFE bias| / true ATT       : %.3f\n", bias_rel_true))

# Group-specific ATT(g)
group_tbl <- tibble(
  cohort = cs_group$egt,
  att_g  = cs_group$att.egt,
  se_g   = cs_group$se.egt
)
print(group_tbl)

# Event-study ATT(e)
event_tbl <- tibble(
  event_time = cs_dynamic$egt,
  att_e      = cs_dynamic$att.egt,
  se_e       = cs_dynamic$se.egt
)
print(event_tbl)

# Calendar-time ATT(t)
cal_tbl <- tibble(
  period = cs_calendar$egt,
  att_t  = cs_calendar$att.egt,
  se_t   = cs_calendar$se.egt
)
print(cal_tbl)

# ---- Soft assertions (warn) -------------------------------------------------
se_overall <- cs_simple$overall.se
se_g5 <- group_tbl$se_g[group_tbl$cohort == 5]
se_g11 <- group_tbl$se_g[group_tbl$cohort == 11]

# Soft #1: SE on CS overall ATT in [0.13, 0.25]
if (!(se_overall >= 0.13 && se_overall <= 0.25)) {
  warning(sprintf(
    "SOFT-FAIL: SE on CS overall ATT = %.3f (target [0.13, 0.25]).",
    se_overall
  ))
}

# Soft #2: |simple - group| / true ATT >= 2%
if (gap_pct < 0.02) {
  warning(sprintf(
    "SOFT-FAIL: |simple - group| / true ATT = %.3f (target >= 0.02).",
    gap_pct
  ))
}

# Soft #3: smallest cohort (g=11, 200 stores) should have wider CI than the
# largest treated cohort (g=5, 700 stores).
if (!is.na(se_g5) && !is.na(se_g11) && !(se_g11 > se_g5)) {
  warning(sprintf(
    "SOFT-FAIL: SE(g=11) = %.3f should exceed SE(g=5) = %.3f (smallest cohort, widest CI).",
    se_g11, se_g5
  ))
}

# Pull the four ATT(g) point estimates for the hard stops below.
att_g5 <- group_tbl$att_g[group_tbl$cohort == 5]
att_g7 <- group_tbl$att_g[group_tbl$cohort == 7]
att_g9 <- group_tbl$att_g[group_tbl$cohort == 9]
att_g11 <- group_tbl$att_g[group_tbl$cohort == 11]

# Soft #4: strictly monotone-decreasing ATT(g). Nice-to-have but not required;
# the chapter prose handles non-strict cases with "cluster lower" phrasing.
is_strict_monotone <- att_g5 > att_g7 && att_g7 > att_g9 && att_g9 > att_g11
if (!is_strict_monotone) {
  warning(sprintf(
    "SOFT-FAIL: ATT(g) is not strictly monotone-decreasing (got %.2f, %.2f, %.2f, %.2f).",
    att_g5, att_g7, att_g9, att_g11
  ))
}

# ---- Hard assertions (stop) -------------------------------------------------
stopifnot(
  "HARD-FAIL: TWFE sign does not match true ATT — design broken." =
    sign(twfe_est) == sign(true_att)
)

stopifnot(
  "HARD-FAIL: |TWFE - CS_simple|/CS_simple outside [0.25, 0.35] — adjust tau_schedule$slope." =
    bias_rel_cs >= 0.25 && bias_rel_cs <= 0.35
)

stopifnot(
  "HARD-FAIL: ATT(g=5) is not the largest — early-adopter dominance broken." =
    att_g5 > max(att_g7, att_g9, att_g11)
)

stopifnot(
  "HARD-FAIL: ATT(g=11) too close to ATT(g=5) — try a different seed (target ATT(g=11) <= 0.85 * ATT(g=5))." =
    att_g11 <= 0.85 * att_g5
)



# =============================================================================
# SECTION 2 — DATA: covariate-driven DGP  (writes data/staggered_did_cov.csv)
# -----------------------------------------------------------------------------
# Companion to Section 1. The main dataset already satisfies unconditional
# parallel trends, so it cannot motivate conditioning on covariates. Here we
# generate a sibling dataset where:
#
#   1. Cohort assignment is driven by two observable covariates:
#        - urban (binary, Bernoulli(0.5))
#        - pre_treatment_sales (continuous, mean 9 if urban else 7, sd 1.5)
#      Stores with a higher selection score are assigned to earlier cohorts.
#      This induces selection on observables: cohort 5 is urban-heavy and
#      high-baseline; never-treated is the opposite.
#
#   2. The outcome path has a covariate-driven linear trend. Since cohort
#      composition in (urban, pre_treatment_sales) differs by cohort, this
#      trend term breaks unconditional parallel trends.
#
#   3. The treatment effect schedule tau(g, k) is identical to Section 1, so the
#      conditional CS estimate (with xformla = ~ pre_treatment_sales + urban)
#      should land near Section 1's overall ATT. The reader sees
#      "conditioning recovers the right answer."
#
# Same cohort sizes (700/500/300/200/800), 12 periods, AR(1) error with rho
# 0.3 and marginal SD 4.5, store FE N(10, 3.5^2). State is randomized per
# store and is not used in any analysis; included for schema parity.
#
# Calibration targets (HARD = stopifnot, SOFT = warning):
#   HARD: at least 2 of the 6 pre-treatment leads are outside the uniform
#         95% band in the UNCONDITIONAL event study; mean |lead| >= 1.0.
#   HARD: ALL pre-treatment leads inside the uniform 95% band in the
#         CONDITIONAL event study; mean |lead| <= 0.4.
#   HARD: |unconditional_ATT - conditional_ATT| / conditional_ATT >= 0.15
#         (unconditional is meaningfully biased).
#   HARD: |conditional_ATT - true_ATT| / true_ATT <= 0.10 (within ~1-2 SE).
#   HARD: Each cohort contains both urban and non-urban stores
#         (urban share in (0.05, 0.95)).
#   SOFT: Strict monotone bias in leads (more negative as e -> -6).
# =============================================================================

set.seed(64)

# ---- Panel structure --------------------------------------------------------
cohort_sizes <- c("5" = 700, "7" = 500, "9" = 300, "11" = 200, "0" = 800)
cohorts <- as.integer(names(cohort_sizes))
T <- 12
n_total <- sum(cohort_sizes)
states_pool <- c("PB", "RN", "CE", "PE", "BA")

# ---- Store-level attributes -------------------------------------------------
# urban first, then pre_treatment_sales conditional on urban. Both are
# time-invariant store characteristics and will be the covariates passed to
# att_gt() via xformla in the conditional analysis.
urban <- rbinom(n_total, size = 1, prob = 0.5)
pre_treatment_sales <- ifelse(
  urban == 1,
  rnorm(n_total, mean = 9, sd = 1.5),
  rnorm(n_total, mean = 7, sd = 1.5)
)

# Selection-into-treatment score. Higher score = earlier cohort. The noise
# SD is intentionally larger than the covariate weights so every cohort
# retains genuine within-cohort variation in urban and pre_treatment_sales.
# This is what makes the conditional analysis well-posed (overlap holds).
score <- 0.5 * urban + 0.25 * (pre_treatment_sales - 8) + rnorm(n_total, 0, 1.2)

# Sort stores by score (desc) and assign cohorts top-down: top 700 -> g=5,
# next 500 -> g=7, next 300 -> g=9, next 200 -> g=11, last 800 -> g=0.
ord <- order(-score)
assigned <- rep(0L, n_total)
breaks <- cumsum(c(700, 500, 300, 200))
assigned[ord[1:breaks[1]]] <- 5L
assigned[ord[(breaks[1] + 1):breaks[2]]] <- 7L
assigned[ord[(breaks[2] + 1):breaks[3]]] <- 9L
assigned[ord[(breaks[3] + 1):breaks[4]]] <- 11L
assigned[ord[(breaks[4] + 1):n_total]] <- 0L

stores <- tibble(
  store_id            = seq_len(n_total),
  urban               = urban,
  pre_treatment_sales = pre_treatment_sales,
  score               = score,
  cohort              = assigned,
  state               = sample(states_pool, n_total, replace = TRUE),
  store_FE            = rnorm(n_total, mean = 10, sd = 3.5)
)

# Cohort-level baseline intercepts. Set to zero here because cohort
# assignment via the score already generates substantial baseline-level
# differences across cohorts (urban + high pre_sales -> earlier cohort).
# Layering additional intercepts on top would obscure pedagogy without
# changing the conditional/unconditional contrast.
cohort_intercept <- tibble(cohort = c(5, 7, 9, 11, 0), c_g = rep(0, 5))

period_FE <- tibble(period = 1:T, period_FE = rnorm(T, mean = 0, sd = 0.3))

panel <- expand_grid(store_id = stores$store_id, period = 1:T) |>
  left_join(stores, by = "store_id") |>
  left_join(cohort_intercept, by = "cohort") |>
  left_join(period_FE, by = "period") |>
  mutate(
    relative_period = if_else(cohort == 0, NA_integer_, period - cohort),
    treated         = as.integer(cohort != 0 & period >= cohort)
  )

# ---- AR(1) residuals (same form as Section 1) -------------------------------
ar1_panel <- function(n_stores, T, rho, sd_marginal) {
  sd_innov <- sd_marginal * sqrt(1 - rho^2)
  e <- matrix(0, nrow = n_stores, ncol = T)
  e[, 1] <- rnorm(n_stores, 0, sd_marginal)
  innov <- matrix(rnorm(n_stores * (T - 1), 0, sd_innov), nrow = n_stores)
  for (t in 2:T) e[, t] <- rho * e[, t - 1] + innov[, t - 1]
  as.vector(t(e))
}

panel$ar1_error <- ar1_panel(n_total, T, rho = 0.3, sd_marginal = 4.5)

# ---- Treatment effect schedule (identical to Section 1) ---------------------
tau_schedule <- tibble(
  cohort = c(5, 7, 9, 11),
  base   = c(1.5, 1.8, 2.1, 2.4),
  slope  = c(0.50, 0.40, 0.28, 0.16)
)

# Covariate-driven trend coefficients. These are what break unconditional PT.
# Calibrated so that with softened selection (sd_noise=1.2), the average lead
# at e = -6 lands around -2.5 to -3.0 (clearly outside the 95% band) while
# the conditional fit recovers leads near zero.
trend_coef_urban <- 0.75
trend_coef_pre_sales <- 0.40

panel_full <- panel |>
  left_join(tau_schedule, by = "cohort") |>
  mutate(
    tau = if_else(treated == 1, base + slope * relative_period, 0),
    cov_trend = (trend_coef_urban * urban +
      trend_coef_pre_sales * (pre_treatment_sales - 8)) * period,
    sales = store_FE + c_g + period_FE +
      0.4 * pre_treatment_sales +
      1.5 * urban +
      cov_trend +
      tau + ar1_error
  )

# Ship a slim CSV. xformla in att_gt() needs the covariates as columns in the
# data frame (their value at g-1 is what gets used).
panel <- panel_full |>
  select(
    store_id, period, state, urban, pre_treatment_sales,
    cohort, relative_period, treated, sales
  )

out_path <- "data/staggered_did_cov.csv"
write_csv(panel, out_path)
message("Wrote ", out_path, " (", nrow(panel), " rows).")

# ---- True ATT from the DGP --------------------------------------------------
true_att <- panel_full |>
  filter(treated == 1) |>
  summarise(true_att = mean(tau)) |>
  pull(true_att)

# ---- Cohort composition: overlap check --------------------------------------
cohort_composition <- panel |>
  distinct(store_id, cohort, urban, pre_treatment_sales) |>
  group_by(cohort) |>
  summarise(
    n_stores = n(),
    urban_share = mean(urban),
    mean_pre_sales = mean(pre_treatment_sales),
    sd_pre_sales = sd(pre_treatment_sales),
    .groups = "drop"
  )

# Cohort composition (overlap diagnostic)
print(cohort_composition)

# ---- Unconditional CS estimator ---------------------------------------------
cs_uncond <- att_gt(
  yname = "sales", tname = "period", idname = "store_id", gname = "cohort",
  data = panel,
  control_group = "nevertreated",
  est_method = "dr",
  base_period = "universal",
  bstrap = TRUE, cband = TRUE, clustervars = "store_id"
)
uncond_simple <- aggte(cs_uncond, type = "simple")
uncond_dynamic <- aggte(cs_uncond,
  type = "dynamic", cband = TRUE,
  min_e = -6, max_e = 7
)

# ---- Conditional CS estimator with xformla ----------------------------------
cs_cond <- att_gt(
  yname = "sales", tname = "period", idname = "store_id", gname = "cohort",
  data = panel,
  xformla = ~ pre_treatment_sales + urban,
  control_group = "nevertreated",
  est_method = "dr",
  base_period = "universal",
  bstrap = TRUE, cband = TRUE, clustervars = "store_id"
)
cond_simple <- aggte(cs_cond, type = "simple")
cond_dynamic <- aggte(cs_cond,
  type = "dynamic", cband = TRUE,
  min_e = -6, max_e = 7
)

# ---- Headline numbers -------------------------------------------------------
# Overall ATT comparison
cat(sprintf("True overall ATT          : %.3f\n", true_att))
cat(sprintf(
  "Unconditional CS (simple) : %.3f (SE %.3f)\n",
  uncond_simple$overall.att, uncond_simple$overall.se
))
cat(sprintf(
  "Conditional   CS (simple) : %.3f (SE %.3f)\n",
  cond_simple$overall.att, cond_simple$overall.se
))
bias_uncond <- (uncond_simple$overall.att - true_att) / true_att
bias_cond <- (cond_simple$overall.att - true_att) / true_att
gap_rel <- abs(uncond_simple$overall.att - cond_simple$overall.att) /
  cond_simple$overall.att
cat(sprintf("Bias (uncond)             : %+.1f%% of true\n", 100 * bias_uncond))
cat(sprintf("Bias (cond)               : %+.1f%% of true\n", 100 * bias_cond))
cat(sprintf("|uncond - cond| / cond    : %.3f  (target >= 0.15)\n", gap_rel))

# ---- Event-study lead diagnostics ------------------------------------------
lead_diag <- function(agg, label) {
  # With base_period = "universal" and anticipation = 0, e = -1 is normalized
  # to exactly 0 with NA SE (per the did::att_gt help). Exclude it from the
  # diagnostic so it doesn't dilute the mean |lead| with a mechanical zero.
  pre <- agg$egt < 0 & agg$egt != -1
  tibble(
    label        = label,
    event_time   = agg$egt[pre],
    att          = agg$att.egt[pre],
    se           = agg$se.egt[pre],
    crit_val     = agg$crit.val.egt,
    band_lo      = agg$att.egt[pre] - agg$crit.val.egt * agg$se.egt[pre],
    band_hi      = agg$att.egt[pre] + agg$crit.val.egt * agg$se.egt[pre],
    outside_band = (band_lo > 0) | (band_hi < 0)
  )
}

leads_uncond <- lead_diag(uncond_dynamic, "unconditional")
leads_cond <- lead_diag(cond_dynamic, "conditional")

# Pre-treatment leads (unconditional)
print(leads_uncond)
# Pre-treatment leads (conditional)
print(leads_cond)

n_out_uncond <- sum(leads_uncond$outside_band)
n_out_cond <- sum(leads_cond$outside_band)
mean_abs_uncond <- mean(abs(leads_uncond$att))
mean_abs_cond <- mean(abs(leads_cond$att))

cat(sprintf(
  "\nUnconditional: %d of %d leads outside uniform band, mean |lead| = %.3f\n",
  n_out_uncond, nrow(leads_uncond), mean_abs_uncond
))
cat(sprintf(
  "Conditional  : %d of %d leads outside uniform band, mean |lead| = %.3f\n",
  n_out_cond, nrow(leads_cond), mean_abs_cond
))

# ---- Soft assertions --------------------------------------------------------
# Strict monotone bias in unconditional leads (more negative as e -> -6).
uncond_leads_sorted <- leads_uncond |> arrange(event_time)
is_monotone <- all(diff(uncond_leads_sorted$att) > 0)
if (!is_monotone) {
  warning(sprintf(
    "SOFT-FAIL: unconditional leads not monotone in event_time (got %s).",
    paste(sprintf("%.2f", uncond_leads_sorted$att), collapse = ", ")
  ))
}

# ---- Hard assertions --------------------------------------------------------
stopifnot(
  "HARD-FAIL: too few unconditional leads outside the 95% band (need >= 2)." =
    n_out_uncond >= 2
)

stopifnot(
  "HARD-FAIL: unconditional mean |lead| too small (need >= 1.0)." =
    mean_abs_uncond >= 1.0
)

stopifnot(
  "HARD-FAIL: conditional leads still drifting outside the band (need 0)." =
    n_out_cond == 0
)

stopifnot(
  "HARD-FAIL: conditional mean |lead| too large (need <= 0.4)." =
    mean_abs_cond <= 0.4
)

stopifnot(
  "HARD-FAIL: unconditional vs conditional ATT gap too small (need >= 15%)." =
    gap_rel >= 0.15
)

stopifnot(
  "HARD-FAIL: conditional ATT not close enough to true (need within 10%)." =
    abs(bias_cond) <= 0.10
)

stopifnot(
  "HARD-FAIL: some cohort has degenerate urban composition (need share in (0.05, 0.95))." =
    all(cohort_composition$urban_share > 0.05 &
      cohort_composition$urban_share < 0.95)
)



# =============================================================================
# SECTION 3 — FIGURES: main (7 PNGs)
# -----------------------------------------------------------------------------
# Reads data/staggered_did.csv (written by Section 1) and writes the seven
# canonical PNGs the chapter references:
#   cohort_trends, attgt_matrix, aggte_simple, aggte_group, aggte_calendar,
#   aggte_dynamic, twfe_vs_cs.
# Y-axis units are "R$ millions" (matching Chapter 9's convention).
#
# Reproducibility convention: every att_gt()/aggte() call below runs a
# multiplier bootstrap, whose draws depend on the RNG state at the call site.
# To make each result reproducible *independently of call order* — and
# identical to the seeded calls in staggered-did.qmd — we re-seed with
# set.seed(64) immediately before every bootstrap-bearing call. (64 matches
# the DGP seed.) The seed here covers anything before the first such call.
# =============================================================================

set.seed(64)

# ---- Load main data ---------------------------------------------------------
panel <- read_csv("data/staggered_did.csv", show_col_types = FALSE) |>
  mutate(cohort_label = if_else(cohort == 0, "Never treated",
    paste0("Treated in period ", cohort)
  ))

# ============================================================================
# A. Data inspection (mirrors chapter S4 "Implementing")
# ============================================================================
print(head(panel))

cohort_summary <- panel |>
  distinct(store_id, cohort) |>
  count(cohort) |>
  mutate(cohort_label = case_when(
    cohort == 0 ~ "Never Treated",
    TRUE ~ paste0("Treated in Period ", cohort)
  ))
print(cohort_summary)

# ============================================================================
# F1. Cohort sales trajectories (chapter @fig-cohort-trends)
# ============================================================================
cohort_means <- panel |>
  group_by(cohort, cohort_label, period) |>
  summarise(mean_sales = mean(sales), .groups = "drop")

# cohort_color_* and cohort_palette are defined in the shared block at the top.
p_f1 <- ggplot(
  cohort_means,
  aes(
    x = period, y = mean_sales,
    color = cohort_label, group = cohort_label
  )
) +
  geom_vline(xintercept = 5, linetype = "22", color = cohort_color_5, alpha = 0.6, linewidth = 1.2) +
  geom_vline(xintercept = 7, linetype = "22", color = cohort_color_7, alpha = 0.6, linewidth = 1.2) +
  geom_vline(xintercept = 9, linetype = "22", color = cohort_color_9, alpha = 0.6, linewidth = 1.2) +
  geom_vline(xintercept = 11, linetype = "22", color = cohort_color_11, alpha = 0.6, linewidth = 1.2) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  scale_color_manual(values = cohort_palette, name = NULL) +
  scale_x_continuous(breaks = 1:12) +
  guides(color = guide_legend(nrow = 2, byrow = TRUE)) +
  labs(
    x = "Period (month)",
    y = "Average sales (R$ millions)",
    title = "Cohort treatment and outcome trends",
    subtitle = "Dashed lines mark each cohort's first treated period"
  ) +
  theme_book(base_size = 16)

ggsave("images/cohort_trends.png", p_f1,
  width = 9, height = 5.5, dpi = 200, bg = "white"
)
message("Wrote images/cohort_trends.png")

# ============================================================================
# C. Main CS estimation (never-treated control, doubly-robust, bootstrap)
#     base_period = "universal" matches the chapter's att_gt() call: it fixes
#     the pre-treatment baseline at g - 1 so the event-study leads display
#     cumulatively. Post-treatment ATT(g, t) and every aggregation overall are
#     identical under "varying"; only the pre-treatment cells/leads change.
#
#     Per the seeding convention above, re-seed immediately before att_gt().
# ============================================================================
set.seed(64)
cs_results <- att_gt(
  yname = "sales", tname = "period", idname = "store_id", gname = "cohort",
  data = panel,
  control_group = "nevertreated", est_method = "dr",
  bstrap = TRUE, cband = TRUE, clustervars = "store_id",
  base_period = "universal", anticipation = 0
)

# C. summary(cs_results) — ATT(g, t) table
print(summary(cs_results))

# Extract ATT(5, 7) -- the specific cell the chapter quotes
att57_idx <- which(cs_results$group == 5 & cs_results$t == 7)
att57_est <- cs_results$att[att57_idx]
att57_se <- cs_results$se[att57_idx]
cat(sprintf("\nATT(5, 7) = %.3f (SE %.3f)\n", att57_est, att57_se))

# ============================================================================
# B. Sensitivity check: never-treated vs not-yet-treated (chapter S3.1)
# ============================================================================
set.seed(64)
cs_notyet <- att_gt(
  yname = "sales", tname = "period", idname = "store_id", gname = "cohort",
  data = panel,
  control_group = "notyettreated", est_method = "dr",
  bstrap = TRUE, cband = TRUE, clustervars = "store_id",
  anticipation = 0
)

set.seed(64)
agg_never_simple <- aggte(cs_results, type = "simple")
set.seed(64)
agg_notyet_simple <- aggte(cs_notyet, type = "simple")

sensitivity_tbl <- tibble(
  control = c("Never-treated", "Not-yet-treated"),
  att     = c(agg_never_simple$overall.att, agg_notyet_simple$overall.att),
  se      = c(agg_never_simple$overall.se, agg_notyet_simple$overall.se)
)

# B. Sensitivity: never-treated vs not-yet-treated overall ATT
print(sensitivity_tbl)

# ============================================================================
# D + G. Four aggregations + their figures
# ============================================================================

# --- Aggregation 1: simple (single overall ATT) ------------------------------
agg_simple <- agg_never_simple # already computed above

# D. summary(aggte(..., type = 'simple'))
print(summary(agg_simple))

# --- Aggregation 2: group (one per cohort) -----------------------------------
set.seed(64)
agg_group <- aggte(cs_results, type = "group", cband = TRUE)

# D. summary(aggte(..., type = 'group'))
print(summary(agg_group))

# --- Aggregation 3: calendar (one per period) --------------------------------
set.seed(64)
agg_calendar <- aggte(cs_results, type = "calendar", cband = TRUE)

# D. summary(aggte(..., type = 'calendar'))
print(summary(agg_calendar))

# --- Aggregation 4: dynamic / event study ------------------------------------
set.seed(64)
agg_dynamic <- aggte(cs_results,
  type = "dynamic", cband = TRUE,
  min_e = -6, max_e = 7
)

# D. summary(aggte(..., type = 'dynamic', min_e=-6, max_e=7))
print(summary(agg_dynamic))

# Explicit values at e = 0 and e = 4 (prose anchors)
e0_idx <- which(agg_dynamic$egt == 0)
e4_idx <- which(agg_dynamic$egt == 4)
dyn_e0 <- agg_dynamic$att.egt[e0_idx]
dyn_e0s <- agg_dynamic$se.egt[e0_idx]
dyn_e4 <- agg_dynamic$att.egt[e4_idx]
dyn_e4s <- agg_dynamic$se.egt[e4_idx]
cat(sprintf("\nDynamic ATT at e=0 : %.3f (SE %.3f)\n", dyn_e0, dyn_e0s))
cat(sprintf("Dynamic ATT at e=4 : %.3f (SE %.3f)\n", dyn_e4, dyn_e4s))

# True overall ATT (main tau schedule) -- used as the ground-truth line in F3
tau_schedule_main <- tibble(
  cohort = c(5, 7, 9, 11),
  base   = c(1.5, 1.8, 2.1, 2.4),
  slope  = c(0.50, 0.40, 0.28, 0.16)
)

panel_treated <- panel |>
  filter(cohort != 0, period >= cohort) |>
  left_join(tau_schedule_main, by = "cohort") |>
  mutate(tau = base + slope * (period - cohort))

true_att <- mean(panel_treated$tau)
cat(sprintf("\nTrue overall ATT (from main tau schedule) : %.3f\n", true_att))

# ---- F2. ATT(g, t) heat-map -------------------------------------------------
attgt_df <- tibble(
  cohort = cs_results$group,
  period = cs_results$t,
  att    = cs_results$att
) |>
  mutate(phase = if_else(period < cohort, "pre", "post"))

p_f2 <- ggplot(attgt_df, aes(x = period, y = factor(cohort), fill = att)) +
  geom_tile(aes(alpha = phase), color = "white") +
  geom_text(aes(label = sprintf("%.2f", att)),
    size = 3.6,
    color = book_colors$dark_gray
  ) +
  scale_fill_gradient2(
    low = book_colors$success, mid = "white",
    high = book_colors$primary, midpoint = 0,
    name = "ATT(g, t)"
  ) +
  scale_alpha_manual(values = c(pre = 0.35, post = 1.0), guide = "none") +
  scale_x_continuous(breaks = 1:12) +
  labs(
    x = "Calendar period (t)",
    y = "Cohort (g)",
    title = "Group-time average treatment effects",
    subtitle = "Pre-treatment cells are faded; post-treatment cells show ATT(g, t)"
  ) +
  theme_book(base_size = 14.4)

ggsave("images/attgt_matrix.png", p_f2,
  width = 9, height = 4.5, dpi = 200, bg = "white"
)
message("Wrote images/attgt_matrix.png")

# ---- F4. Group-specific ATT (one point per cohort) --------------------------
hline_f4 <- tibble(overall = agg_group$overall.att)
p_f4 <- tibble(
  cohort = agg_group$egt,
  att    = agg_group$att.egt,
  se     = agg_group$se.egt,
  c_uni  = agg_group$crit.val.egt
) |>
  mutate(lo = att - c_uni * se, hi = att + c_uni * se) |>
  ggplot(aes(x = factor(cohort), y = att)) +
  geom_hline(
    data = hline_f4,
    aes(yintercept = overall, linetype = "Overall Treatment Effect"),
    color = book_colors$accent, linewidth = 0.8
  ) +
  geom_pointrange(aes(ymin = lo, ymax = hi),
    color = book_colors$primary, size = 1, linewidth = 1
  ) +
  annotate("text",
    x = Inf, y = 3.5,
    label = sprintf(
      "Overall treatment effect:\naverage ATT across cohorts = R$ %.2f M",
      agg_group$overall.att
    ),
    hjust = 1, vjust = -0.4, size = 3.6, color = book_colors$accent
  ) +
  scale_linetype_manual(
    name = NULL,
    values = c("Overall Treatment Effect" = "dashed")
  ) +
  guides(linetype = guide_legend(
    override.aes = list(color = book_colors$accent)
  )) +
  labs(
    x = "Cohort (first-treated period)", y = "Average ATT for cohort",
    title = "Group-specific ATT and uniform 95% bands"
  ) +
  theme_book()
ggsave("images/aggte_group.png", p_f4,
  width = 7, height = 4.5,
  dpi = 200, bg = "white"
)
message("Wrote images/aggte_group.png")

# ---- F5. Calendar-time ATT --------------------------------------------------
hline_f5 <- tibble(overall = agg_calendar$overall.att)
p_f5 <- tibble(
  period = agg_calendar$egt,
  att    = agg_calendar$att.egt,
  se     = agg_calendar$se.egt,
  c_uni  = agg_calendar$crit.val.egt
) |>
  mutate(lo = att - c_uni * se, hi = att + c_uni * se) |>
  ggplot(aes(x = period, y = att)) +
  geom_hline(
    data = hline_f5,
    aes(yintercept = overall, linetype = "Overall Treatment Effect"),
    color = book_colors$accent, linewidth = 0.8
  ) +
  geom_ribbon(aes(ymin = lo, ymax = hi), fill = book_colors$primary, alpha = 0.2) +
  geom_line(color = book_colors$primary, linewidth = 1) +
  geom_point(color = book_colors$primary, size = 2) +
  annotate("text",
    x = -Inf, y = agg_calendar$overall.att,
    label = sprintf(
      "Overall treatment effect:\naverage ATT across calendar periods = R$ %.2f M",
      agg_calendar$overall.att
    ),
    hjust = 0, vjust = -0.4, size = 3.96, color = book_colors$accent
  ) +
  scale_linetype_manual(
    name = NULL,
    values = c("Overall Treatment Effect" = "dashed")
  ) +
  guides(linetype = guide_legend(
    override.aes = list(color = book_colors$accent)
  )) +
  scale_x_continuous(breaks = seq(min(agg_calendar$egt),
    max(agg_calendar$egt),
    by = 1
  )) +
  labs(
    x = "Calendar period (t)", y = "Average ATT in period t",
    title = "Calendar-time ATT and uniform 95% bands"
  ) +
  theme_book()
ggsave("images/aggte_calendar.png", p_f5,
  width = 8, height = 4.5,
  dpi = 200, bg = "white"
)
message("Wrote images/aggte_calendar.png")

# ---- F6. Dynamic ATT (event study) ------------------------------------------
hline_f6 <- tibble(overall = agg_dynamic$overall.att)
p_f6 <- tibble(
  e      = agg_dynamic$egt,
  att    = agg_dynamic$att.egt,
  se     = agg_dynamic$se.egt,
  c_uni  = agg_dynamic$crit.val.egt
) |>
  mutate(
    lo = att - c_uni * se, hi = att + c_uni * se,
    phase = if_else(e < 0, "Pre", "Post")
  ) |>
  ggplot(aes(x = e, y = att, color = phase)) +
  geom_hline(yintercept = 0, color = book_colors$muted) +
  geom_vline(xintercept = -0.5, linetype = "dashed", color = book_colors$muted) +
  geom_hline(
    data = hline_f6,
    aes(yintercept = overall, linetype = "Overall Treatment Effect"),
    color = book_colors$accent, linewidth = 0.8,
    inherit.aes = FALSE
  ) +
  geom_pointrange(aes(ymin = lo, ymax = hi), size = 0.8, linewidth = 0.8) +
  annotate("text",
    x = -Inf, y = agg_dynamic$overall.att,
    label = sprintf(
      "Overall treatment effect:\naverage post-treatment ATT (e ≥ 0) = R$ %.2f M",
      agg_dynamic$overall.att
    ),
    hjust = 0, vjust = -0.4, size = 3.6, color = book_colors$accent
  ) +
  scale_color_manual(
    values = c(
      Pre = book_colors$secondary,
      Post = book_colors$primary
    ),
    name = NULL
  ) +
  scale_linetype_manual(
    name = NULL,
    values = c("Overall Treatment Effect" = "dashed")
  ) +
  scale_x_continuous(breaks = seq(min(agg_dynamic$egt),
    max(agg_dynamic$egt),
    by = 1
  )) +
  guides(linetype = guide_legend(
    override.aes = list(color = book_colors$accent)
  )) +
  labs(
    x = "Event time (periods since treatment)",
    y = "Average ATT",
    title = "Dynamic ATT (event study) and uniform 95% bands"
  ) +
  theme_book()
ggsave("images/aggte_dynamic.png", p_f6,
  width = 8, height = 4.5,
  dpi = 200, bg = "white"
)
message("Wrote images/aggte_dynamic.png")

# ============================================================================
# E. TWFE vs CS
# ============================================================================

# --- E.1 Overall ATT comparison ----------------------------------------------
twfe_fit <- feols(
  sales ~ treated + pre_treatment_sales | store_id + period,
  data = panel, cluster = ~store_id
)
twfe_est <- coef(twfe_fit)[["treated"]]
twfe_se <- se(twfe_fit)[["treated"]]

# CS side uses the `group` overall — the chapter's recommended headline ATT.
side_by_side <- tibble(
  estimator = c("Naive TWFE", "Callaway-Sant'Anna"),
  estimate  = c(twfe_est, agg_group$overall.att),
  se        = c(twfe_se, agg_group$overall.se)
)

attenuation_pct <- 100 * (1 - twfe_est / agg_group$overall.att)

# E.1 TWFE vs CS overall ATT (side-by-side)
print(side_by_side)
cat(sprintf("\nAttenuation (1 - TWFE/CS) * 100 = %.1f%%\n", attenuation_pct))

# --- E.1b Figure: overall ATT, TWFE vs CS vs ground truth --------------------
p_e1 <- tibble(
  estimator = factor(
    c("True ATT", "Naive TWFE", "Callaway-Sant'Anna"),
    levels = c("True ATT", "Naive TWFE", "Callaway-Sant'Anna")
  ),
  value = c(true_att, twfe_est, agg_group$overall.att),
  se = c(NA, twfe_se, agg_group$overall.se)
) |>
  mutate(lo = value - 1.96 * se, hi = value + 1.96 * se) |>
  ggplot(aes(x = estimator, y = value, color = estimator)) +
  geom_pointrange(aes(ymin = lo, ymax = hi), size = 1, linewidth = 1) +
  scale_color_manual(
    values = c(
      "True ATT"           = book_colors$muted,
      "Naive TWFE"         = book_colors$success,
      "Callaway-Sant'Anna" = book_colors$primary
    ),
    guide = "none"
  ) +
  labs(
    x = NULL, y = "Overall ATT (R$ millions)",
    title = "Overall ATT — naive TWFE vs Callaway-Sant'Anna vs ground truth"
  ) +
  theme_book()
ggsave("images/twfe_vs_cs_aggregated.png", p_e1,
  width = 7, height = 4.5, dpi = 200, bg = "white"
)
message("Wrote images/twfe_vs_cs_aggregated.png")

# --- E.2 Event-study comparison: with vs without the never-treated holdout ---
# Figure A keeps the BA never-treated holdout in the sample, so both estimators
# use it as a clean control. Figure B drops the holdout entirely (cohort != 0):
# now neither estimator has a clean control, and TWFE is forced to use already-
# treated cohorts as implicit controls -- the forbidden comparisons.

# Helper: tidy a fixest event-study fit into the plotting format.
tidy_twfe_es <- function(fit) {
  ref_row <- tibble(
    e = -1, att = 0, se = NA_real_, lo = 0, hi = 0, src = "TWFE"
  )
  broom::tidy(fit, conf.int = TRUE) |>
    filter(str_detect(term, "rel::")) |>
    mutate(e = as.integer(str_extract(term, "-?\\d+")), src = "TWFE") |>
    filter(e >= -6, e <= 7) |>
    select(
      e,
      att = estimate, se = std.error,
      lo = conf.low, hi = conf.high, src
    ) |>
    bind_rows(ref_row) |>
    arrange(e)
}

# Helper: tidy a did::aggte(type = "dynamic") object into the same format.
tidy_cs_es <- function(agg) {
  tibble(
    e = agg$egt, att = agg$att.egt, se = agg$se.egt, c_uni = agg$crit.val.egt
  ) |>
    mutate(lo = att - c_uni * se, hi = att + c_uni * se, src = "CS") |>
    select(e, att, se, lo, hi, src)
}

# Helper: overlaid TWFE-vs-CS event-study plot. Each estimator also gets a
# dashed, colour-matched line at its overall effect -- the average of its own
# post-treatment ATT (e >= 0), the same summary the F6 figure draws for CS.
plot_es_compare <- function(df, title, subtitle) {
  overall_lbl <- "Overall effect (avg post-treatment ATT)"
  overall_df <- df |>
    filter(e >= 0) |>
    group_by(src) |>
    summarise(overall = mean(att), .groups = "drop")

  ggplot(df, aes(x = e, y = att, color = src, group = src)) +
    geom_hline(yintercept = 0, color = book_colors$muted) +
    geom_vline(
      xintercept = -0.5, linetype = "dashed", color = book_colors$muted
    ) +
    geom_hline(
      data = overall_df,
      aes(yintercept = overall, color = src, linetype = overall_lbl),
      linewidth = 0.8, alpha = 0.65
    ) +
    geom_pointrange(aes(ymin = lo, ymax = hi),
      position = position_dodge(width = 0.3), size = 0.8, linewidth = 0.8,
      alpha = 0.65
    ) +
    scale_color_manual(
      values = c(CS = book_colors$primary, TWFE = book_colors$success), name = NULL
    ) +
    scale_linetype_manual(
      name = NULL, values = setNames("dashed", overall_lbl)
    ) +
    guides(linetype = guide_legend(
      override.aes = list(color = book_colors$muted)
    )) +
    labs(
      x = "Event time (periods since treatment)", y = "Average ATT",
      title = title, subtitle = subtitle
    ) +
    theme_book()
}

# --- E.2a WITH the never-treated holdout -------------------------------------
# Never-treated stores (cohort == 0) get rel = -1 so they fold into the omitted
# reference bin and serve as pure clean controls (they get no relative-time
# dummy). CS uses the never-treated control -- the chapter's main `agg_dynamic`.
panel_es_hold <- panel |>
  mutate(rel = if_else(cohort == 0L, -1L, period - cohort))

twfe_es_hold <- feols(
  sales ~ i(rel, ref = -1) + pre_treatment_sales | store_id + period,
  data = panel_es_hold, cluster = ~store_id
)
twfe_hold_df <- tidy_twfe_es(twfe_es_hold)
cs_hold_df <- tidy_cs_es(agg_dynamic)

# E.2a TWFE event-study coefficients (WITH never-treated holdout)
print(twfe_hold_df)

p_es_hold <- plot_es_compare(
  bind_rows(twfe_hold_df, cs_hold_df),
  "TWFE vs Callaway-Sant'Anna event study — with the never-treated holdout",
  "Both estimators use never-treated stores as a clean control"
)
ggsave("images/twfe_vs_cs_es_holdout.png", p_es_hold,
  width = 9, height = 5, dpi = 200, bg = "white"
)
message("Wrote images/twfe_vs_cs_es_holdout.png")

# --- E.2b WITHOUT the never-treated holdout ----------------------------------
# Drop the holdout entirely: TWFE leans on already-treated cohorts (forbidden
# comparisons); CS has no never-treated units left and falls back to a
# not-yet-treated control.
panel_noh <- panel |> filter(cohort != 0)

twfe_es_noh <- feols(
  sales ~ i(rel, ref = -1) + pre_treatment_sales | store_id + period,
  data = panel_noh |> mutate(rel = period - cohort), cluster = ~store_id
)
twfe_noh_df <- tidy_twfe_es(twfe_es_noh)

set.seed(64)
cs_noh <- att_gt(
  yname = "sales", tname = "period", idname = "store_id", gname = "cohort",
  data = panel_noh,
  control_group = "notyettreated", est_method = "dr",
  bstrap = TRUE, cband = TRUE, clustervars = "store_id",
  base_period = "universal", anticipation = 0
)
set.seed(64)
agg_dynamic_noh <- aggte(cs_noh, type = "dynamic", cband = TRUE, min_e = -6, max_e = 7)
cs_noh_df <- tidy_cs_es(agg_dynamic_noh)

# E.2b TWFE vs CS event-study coefficients (WITHOUT never-treated holdout)
print(twfe_noh_df)
print(cs_noh_df)

p_es_noh <- plot_es_compare(
  bind_rows(twfe_noh_df, cs_noh_df),
  "TWFE vs Callaway-Sant'Anna event study — holdout removed",
  "With no never-treated control, TWFE leans on already-treated cohorts"
)
ggsave("images/twfe_vs_cs_es_noholdout.png", p_es_noh,
  width = 9, height = 5, dpi = 200, bg = "white"
)
message("Wrote images/twfe_vs_cs_es_noholdout.png")

# ============================================================================
# F. Headline-number summary block — prose anchors
# ============================================================================

att_g5 <- agg_group$att.egt[agg_group$egt == 5]
att_g7 <- agg_group$att.egt[agg_group$egt == 7]
att_g9 <- agg_group$att.egt[agg_group$egt == 9]
att_g11 <- agg_group$att.egt[agg_group$egt == 11]

# Event-study anchors -- TWFE with vs without the never-treated holdout.
twfe_hold_em6 <- twfe_hold_df$att[twfe_hold_df$e == -6]
twfe_hold_e0 <- twfe_hold_df$att[twfe_hold_df$e == 0]
twfe_hold_e7 <- twfe_hold_df$att[twfe_hold_df$e == 7]
twfe_noh_em6 <- twfe_noh_df$att[twfe_noh_df$e == -6]
twfe_noh_e0 <- twfe_noh_df$att[twfe_noh_df$e == 0]
twfe_noh_e5 <- twfe_noh_df$att[twfe_noh_df$e == 5]
cs_es_e0 <- cs_hold_df$att[cs_hold_df$e == 0]
cs_es_e7 <- cs_hold_df$att[cs_hold_df$e == 7]

# F. PROSE ANCHORS — values the chapter text quotes
cat(sprintf(
  "%-44s : %s\n",
  "S3.1  never-treated overall ATT",
  sprintf(
    "%.2f (SE %.2f)", agg_never_simple$overall.att,
    agg_never_simple$overall.se
  )
))
cat(sprintf(
  "%-44s : %s\n",
  "S3.1  not-yet-treated overall ATT",
  sprintf(
    "%.2f (SE %.2f)", agg_notyet_simple$overall.att,
    agg_notyet_simple$overall.se
  )
))
cat(sprintf(
  "%-44s : %s\n",
  "S4    ATT(5, 7)",
  sprintf("%.2f (SE %.2f)", att57_est, att57_se)
))
cat(sprintf(
  "%-44s : %s\n",
  "S4    overall ATT (CS simple)",
  sprintf(
    "%.2f (SE %.2f)", agg_simple$overall.att,
    agg_simple$overall.se
  )
))
cat(sprintf(
  "%-44s : %s\n",
  "S4    overall ATT (CS group, headline)",
  sprintf(
    "%.2f (SE %.2f)", agg_group$overall.att,
    agg_group$overall.se
  )
))
cat(sprintf("%-44s : %.2f\n", "S4    true ATT (simulation)", true_att))
cat(sprintf("%-44s : %.2f\n", "S4    group ATT(g=5)", att_g5))
cat(sprintf("%-44s : %.2f\n", "S4    group ATT(g=7)", att_g7))
cat(sprintf("%-44s : %.2f\n", "S4    group ATT(g=9)", att_g9))
cat(sprintf("%-44s : %.2f\n", "S4    group ATT(g=11)", att_g11))
cat(sprintf("%-44s : %.2f\n", "S4    dynamic ATT e=0", dyn_e0))
cat(sprintf("%-44s : %.2f\n", "S4    dynamic ATT e=4", dyn_e4))
cat(sprintf(
  "%-44s : %s\n",
  "S5    TWFE overall",
  sprintf("%.2f (SE %.2f)", twfe_est, twfe_se)
))
cat(sprintf("%-44s : %.1f%%\n", "S5    TWFE attenuation", attenuation_pct))
cat(sprintf("%-44s : %+.2f\n", "S5    TWFE ES e=-6 (with holdout)", twfe_hold_em6))
cat(sprintf("%-44s : %+.2f\n", "S5    TWFE ES e=0  (with holdout)", twfe_hold_e0))
cat(sprintf("%-44s : %+.2f\n", "S5    TWFE ES e=7  (with holdout)", twfe_hold_e7))
cat(sprintf("%-44s : %+.2f\n", "S5    TWFE ES e=-6 (no holdout)", twfe_noh_em6))
cat(sprintf("%-44s : %+.2f\n", "S5    TWFE ES e=0  (no holdout)", twfe_noh_e0))
cat(sprintf("%-44s : %+.2f\n", "S5    TWFE ES e=5  (no holdout)", twfe_noh_e5))
cat(sprintf("%-44s : %+.2f\n", "S5    CS ES e=0", cs_es_e0))
cat(sprintf("%-44s : %+.2f\n", "S5    CS ES e=7", cs_es_e7))

# SECTION 3 complete.


# =============================================================================
# SECTION 4 — FIGURES: covariate-driven (3 PNGs)
# -----------------------------------------------------------------------------
# For the "Adding covariates: a conditional-parallel-trends example" subsection.
# Reads data/staggered_did_cov.csv (written by Section 2) and writes:
#   1. images/cov_cohort_trends.png      — raw average sales by cohort
#                                          (pre-treatment slopes visibly differ
#                                          across cohorts, signalling the PT
#                                          problem before any estimation).
#   2. images/cov_overlap.png            — overlap diagnostic: pre_treatment_sales
#                                          distribution by cohort + urban share
#                                          by cohort.
#   3. images/cov_eventstudy_compare.png — side-by-side event study, left
#                                          panel is CS unconditional (leads
#                                          drift outside the band), right
#                                          panel is CS conditional on both
#                                          covariates (leads hug zero).
#
# Seed the multiplier bootstrap so band widths and SEs are reproducible.
# =============================================================================

set.seed(64)

# ---- Load the covariate-DGP data --------------------------------------------
panel <- read_csv("data/staggered_did_cov.csv", show_col_types = FALSE) |>
  mutate(cohort_label = if_else(cohort == 0, "Never treated",
    paste0("Treated in period ", cohort)
  ))

# cohort_color_* and cohort_palette are defined in the shared block at the top.

# ============================================================================
# F1. Cohort sales trajectories — same layout as cohort_trends.png but on the
#     covariate dataset; cohorts visibly fan out in the pre-treatment window.
# ============================================================================
cohort_means <- panel |>
  group_by(cohort, cohort_label, period) |>
  summarise(mean_sales = mean(sales), .groups = "drop")

p_f1 <- ggplot(
  cohort_means,
  aes(
    x = period, y = mean_sales,
    color = cohort_label, group = cohort_label
  )
) +
  geom_vline(
    xintercept = 5, linetype = "22", color = cohort_color_5,
    alpha = 0.6, linewidth = 1.2
  ) +
  geom_vline(
    xintercept = 7, linetype = "22", color = cohort_color_7,
    alpha = 0.6, linewidth = 1.2
  ) +
  geom_vline(
    xintercept = 9, linetype = "22", color = cohort_color_9,
    alpha = 0.6, linewidth = 1.2
  ) +
  geom_vline(
    xintercept = 11, linetype = "22", color = cohort_color_11,
    alpha = 0.6, linewidth = 1.2
  ) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  scale_color_manual(values = cohort_palette, name = NULL) +
  scale_x_continuous(breaks = 1:12) +
  guides(color = guide_legend(nrow = 2, byrow = TRUE)) +
  labs(
    x = "Period (month)",
    y = "Average sales (R$ millions)",
    title = "Cohort trends when treatment timing depends on observables",
    subtitle = "Urban, high-baseline cohorts trend up faster even before they are treated"
  ) +
  theme_book(base_size = 16)

ggsave("images/cov_cohort_trends.png", p_f1,
  width = 9, height = 5.5, dpi = 200, bg = "white"
)
message("Wrote images/cov_cohort_trends.png")

# ============================================================================
# F2. Overlap diagnostic — two panels in one figure.
#     Left  : distribution of pre_treatment_sales by cohort (boxplot).
#     Right : urban share by cohort (bar chart).
#     Both confirm every cohort has comparable units in the covariate space.
# ============================================================================
stores <- panel |>
  distinct(store_id, cohort, cohort_label, urban, pre_treatment_sales)

# Force a consistent cohort order on the x-axis for both panels.
cohort_order <- c(
  "Never treated", "Treated in period 5",
  "Treated in period 7", "Treated in period 9",
  "Treated in period 11"
)
stores <- stores |>
  mutate(cohort_label = factor(cohort_label, levels = cohort_order))

p_overlap_left <- ggplot(
  stores,
  aes(
    x = cohort_label, y = pre_treatment_sales,
    fill = cohort_label
  )
) +
  geom_boxplot(
    alpha = 0.7, outlier.size = 0.6,
    outlier.color = book_colors$muted
  ) +
  scale_fill_manual(values = cohort_palette, guide = "none") +
  labs(
    x = NULL, y = "pre_treatment_sales",
    title = "Pre-treatment sales by cohort",
    subtitle = "Boxplots overlap — comparable stores in every cohort"
  ) +
  theme_book() +
  theme(axis.text.x = element_text(angle = 25, hjust = 1))

urban_share <- stores |>
  group_by(cohort_label) |>
  summarise(urban_share = mean(urban), n_stores = n(), .groups = "drop")

p_overlap_right <- ggplot(
  urban_share,
  aes(
    x = cohort_label, y = urban_share,
    fill = cohort_label
  )
) +
  geom_col(alpha = 0.85, width = 0.65) +
  geom_text(aes(label = scales::percent(urban_share, accuracy = 1)),
    vjust = -0.4, size = 3.5, color = book_colors$dark_gray
  ) +
  scale_fill_manual(values = cohort_palette, guide = "none") +
  scale_y_continuous(
    labels = scales::percent_format(accuracy = 1),
    expand = expansion(mult = c(0, 0.15))
  ) +
  labs(
    x = NULL, y = "Share of urban stores",
    title = "Urban share by cohort",
    subtitle = "Earlier cohorts skew urban, but every cohort has both types"
  ) +
  theme_book() +
  theme(axis.text.x = element_text(angle = 25, hjust = 1))

p_f2 <- p_overlap_left + p_overlap_right + plot_layout(ncol = 2)

ggsave("images/cov_overlap.png", p_f2,
  width = 11, height = 5, dpi = 200, bg = "white"
)
message("Wrote images/cov_overlap.png")

# ============================================================================
# CS estimation — unconditional and conditional, both with universal base
# period so leads display cumulatively in the event-study plot.
# ============================================================================
cs_uncond <- att_gt(
  yname = "sales", tname = "period", idname = "store_id", gname = "cohort",
  data = panel,
  control_group = "nevertreated",
  est_method = "dr",
  base_period = "universal",
  bstrap = TRUE, cband = TRUE, clustervars = "store_id"
)
uncond_simple <- aggte(cs_uncond, type = "simple")
uncond_dynamic <- aggte(cs_uncond,
  type = "dynamic", cband = TRUE,
  min_e = -6, max_e = 7
)

cs_cond <- att_gt(
  yname = "sales", tname = "period", idname = "store_id", gname = "cohort",
  data = panel,
  xformla = ~ pre_treatment_sales + urban,
  control_group = "nevertreated",
  est_method = "dr",
  base_period = "universal",
  bstrap = TRUE, cband = TRUE, clustervars = "store_id"
)
cond_simple <- aggte(cs_cond, type = "simple")
cond_dynamic <- aggte(cs_cond,
  type = "dynamic", cband = TRUE,
  min_e = -6, max_e = 7
)

# ============================================================================
# F3. Two-panel event study: unconditional vs conditional.
# ============================================================================
make_es_df <- function(agg, panel_label) {
  tibble(
    e      = agg$egt,
    att    = agg$att.egt,
    se     = agg$se.egt,
    c_uni  = agg$crit.val.egt
  ) |>
    mutate(
      lo    = att - c_uni * se,
      hi    = att + c_uni * se,
      phase = if_else(e < 0, "Pre", "Post"),
      panel = panel_label
    )
}

es_uncond <- make_es_df(
  uncond_dynamic,
  "Without covariates"
)
es_cond <- make_es_df(
  cond_dynamic,
  "With covariates"
)

# Force panel order so unconditional is on the left.
es_combined <- bind_rows(es_uncond, es_cond) |>
  mutate(panel = factor(panel, levels = c(
    "Without covariates",
    "With covariates"
  )))

y_range <- range(c(es_combined$lo, es_combined$hi), na.rm = TRUE)

# Per-panel overall ATT (dynamic aggregation): one line + label per facet.
hline_es <- tibble(
  panel = factor(levels(es_combined$panel),
    levels = levels(es_combined$panel)
  ),
  overall = c(uncond_dynamic$overall.att, cond_dynamic$overall.att)
)

p_f3 <- ggplot(es_combined, aes(x = e, y = att, color = phase)) +
  geom_hline(yintercept = 0, color = book_colors$muted) +
  geom_vline(
    xintercept = -0.5, linetype = "dashed",
    color = book_colors$muted
  ) +
  geom_hline(
    data = hline_es,
    aes(yintercept = overall, linetype = "Overall Treatment Effect"),
    color = book_colors$accent, linewidth = 0.8,
    inherit.aes = FALSE
  ) +
  geom_pointrange(aes(ymin = lo, ymax = hi),
    size = 0.7, linewidth = 0.8,
    na.rm = TRUE
  ) +
  geom_text(
    data = hline_es,
    aes(x = -Inf, y = overall, label = sprintf(
      "Overall treatment effect:\naverage post-treatment ATT (e ≥ 0) = R$ %.2f M",
      overall
    )),
    hjust = 0, vjust = -0.4, size = 4.32, color = book_colors$accent,
    inherit.aes = FALSE
  ) +
  facet_wrap(~panel, ncol = 2) +
  scale_color_manual(
    values = c(
      Pre = book_colors$secondary,
      Post = book_colors$primary
    ),
    name = NULL
  ) +
  scale_linetype_manual(
    name = NULL,
    values = c("Overall Treatment Effect" = "dashed")
  ) +
  guides(linetype = guide_legend(
    override.aes = list(color = book_colors$accent)
  )) +
  scale_x_continuous(breaks = seq(-6, 7, 1)) +
  coord_cartesian(ylim = y_range) +
  labs(
    x = "Event time (periods since treatment)",
    y = "Average ATT (R$ millions)",
    title = "Event study, before and after conditioning on covariates",
    subtitle = "Unconditional leads drift below zero; conditioning on pre_treatment_sales + urban restores parallel trends"
  ) +
  theme_book() +
  # Enlarge title, subtitle, panel titles, axis titles, and tick labels by 20%.
  theme(
    plot.title    = element_text(size = rel(1.44)),
    plot.subtitle = element_text(size = rel(1.2)),
    axis.title    = element_text(size = rel(1.2)),
    axis.text     = element_text(size = rel(0.96)),
    strip.text    = element_text(face = "bold", size = rel(0.96))
  )

ggsave("images/cov_eventstudy_compare.png", p_f3,
  width = 12, height = 5, dpi = 200, bg = "white"
)
message("Wrote images/cov_eventstudy_compare.png")

# ============================================================================
# Console summary — numbers the chapter prose will quote verbatim.
# ============================================================================

# True overall ATT recomputed from the same tau schedule used in the DGP.
tau_schedule <- tibble(
  cohort = c(5, 7, 9, 11),
  base   = c(1.5, 1.8, 2.1, 2.4),
  slope  = c(0.50, 0.40, 0.28, 0.16)
)
true_att <- panel |>
  filter(cohort != 0, period >= cohort) |>
  left_join(tau_schedule, by = "cohort") |>
  mutate(tau = base + slope * (period - cohort)) |>
  summarise(true_att = mean(tau)) |>
  pull(true_att)

# PROSE ANCHORS for the conditional-PT subsection
cat(sprintf("True overall ATT (from DGP)   : %.2f\n", true_att))
cat(sprintf(
  "Unconditional CS  overall ATT : %.2f (SE %.2f)\n",
  uncond_simple$overall.att, uncond_simple$overall.se
))
cat(sprintf(
  "Conditional   CS  overall ATT : %.2f (SE %.2f)\n",
  cond_simple$overall.att, cond_simple$overall.se
))
cat(sprintf(
  "Bias (uncond)                 : %+.0f%% of true\n",
  100 * (uncond_simple$overall.att - true_att) / true_att
))
cat(sprintf(
  "Bias (cond)                   : %+.1f%% of true\n",
  100 * (cond_simple$overall.att - true_att) / true_att
))

cat("\nUnconditional event-study leads (e < 0, with universal base period):\n")
print(es_uncond |> filter(e < 0) |> select(e, att, se, lo, hi))

cat("\nConditional event-study leads (e < 0):\n")
print(es_cond |> filter(e < 0) |> select(e, att, se, lo, hi))

cat("\nUrban share by cohort:\n")
print(urban_share)

cat("\nPre_treatment_sales summary by cohort:\n")
print(stores |>
  group_by(cohort_label) |>
  summarise(
    mean = mean(pre_treatment_sales),
    sd = sd(pre_treatment_sales),
    min = min(pre_treatment_sales),
    max = max(pre_treatment_sales),
    .groups = "drop"
  ))

# SECTION 4 complete.
