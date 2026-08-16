##########################################################
## Everyday Causal Inference: How to estimate, test, and explain impacts with R and Python
## www.everydaycausal.com
## Copyright (c) 2025 by Robson Tigre. All rights reserved.
## You may read, run, adapt, and cite this code, provided you credit the source.
## It should not be used to create competing educational or commercial products
##########################################################
## Code for Chapter 13 - Falsification and robustness: Stress test your results before someone else does
## Created: Aug 07, 2026
## Last modified: Aug 12, 2026
##########################################################

# ==========================================================
# SETUP
# ==========================================================

# If you haven't already, run this once to install the packages:
# install.packages(c("tidyverse", "fixest", "sandwich", "MatchIt", "WeightIt",
#                     "cobalt", "sensemakr", "dagitty", "lmtest", "specr", "readr"))

# You must run the lines below at the start of every new R session.
library(tidyverse)   # data wrangling and plots
library(fixest)      # fast regression with robust standard errors
library(sandwich)    # HC1 and HC3 covariance matrices for lm()
library(MatchIt)     # nearest-neighbour and other matching estimators
library(WeightIt)    # inverse-probability and balancing weights
library(cobalt)      # balance diagnostics and love plots
library(sensemakr)   # sensitivity analysis for unobserved confounding
library(dagitty)     # draw DAGs and read off testable implications
library(lmtest)      # coeftest() for robust-covariance inference
library(specr)       # appendix: specification-curve analysis
library(readr)       # used only inside the optional DGP block below

# ==========================================================
# Book-wide Theme and Color Palette
# ==========================================================
# Define a consistent color palette (colorblind-friendly)
book_colors <- list(
    primary = "#2E86AB", # Steel blue - main data
    secondary = "#A23B72", # Magenta - secondary data
    accent = "#F18F01", # Orange - highlights/warnings
    success = "#C73E1D", # Red-orange - thresholds/targets
    muted = "#6C757D", # Gray - reference lines
    light_gray = "grey90", # Light gray - backgrounds
    dark_gray = "grey30" # Dark gray - text
)

# Unified theme for all plots
theme_book <- function(base_size = 14) {
    theme_minimal(base_size = base_size) +
        theme(
            plot.title = element_text(
                face = "bold", size = base_size * 1.3,
                color = "grey20", margin = margin(b = 5)
            ),
            plot.subtitle = element_text(
                size = base_size * 0.9, color = "grey40",
                margin = margin(b = 15)
            ),
            plot.caption = element_text(
                size = base_size * 0.7, color = "grey50",
                hjust = 0, margin = margin(t = 10)
            ),
            axis.title = element_text(size = base_size, color = "grey30"),
            axis.text = element_text(size = base_size * 0.85, color = "grey40"),
            panel.grid.major = element_line(color = "grey90", linewidth = 0.5),
            panel.grid.minor = element_blank(),
            legend.position = "bottom",
            legend.title = element_text(size = base_size * 0.9, face = "bold"),
            legend.text = element_text(size = base_size * 0.85),
            plot.margin = margin(20, 20, 20, 20)
        )
}

# ==========================================================
# DATA GENERATING PROCESS (DGP) -- optional, non-writing block
# ==========================================================
# Simulating a free-shipping banner on a marketplace. Assignment is NOT
# random: a targeting model shows the banner to users it expects to buy.
# Confounding is planted in two deliberate layers: (1) observed-but-omitted
# pre-treatment behavior (prior_orders, prior_profit, sessions_pre) that a
# lazy analyst would leave out, and (2) a truly-unobserved latent
# shopping_intent that no observed covariate captures. R generates and
# freezes this dataset once; the Python companion only ever reads the same
# frozen CSV below, so R and Python never diverge on different RNGs.
#
# This block is EDUCATIONAL ONLY. The write_csv() calls stay commented out:
# do not regenerate data/free_shipping_banner.csv from here, or every number
# in the chapter stops matching what is printed on the page. The section
# right after this one reads the already-frozen file instead.

## ---- config (single source of truth) ----
cfg <- list(
  seed = 20260606,
  n    = 30000,

  ## latent shopping intent (truly-unobserved layer) ~ N(0, 1)
  intent_sd = 1,

  ## pre-treatment behaviour (observed-but-omitted layer), each driven by intent
  ## so the omitted covariates carry part -- but only part -- of the confounding
  prior_orders_base   = 3,     # Poisson base rate before intent shifts it
  prior_orders_intent = 1.0,   # how strongly intent lifts prior order rate
  prior_profit_base   = 40,    # R$ baseline pre-period profit level
  prior_profit_intent = 12,    # R$ per unit of intent
  prior_profit_orders = 4,     # R$ per prior order
  prior_profit_sd     = 8,     # idiosyncratic noise on prior_profit
  sessions_base       = 6,     # Poisson base session count
  sessions_intent     = 1.2,   # intent lift on sessions

  ## targeting score the business actually computed: built from OBSERVED
  ## covariates AND the latent intent, squashed to (0, 1)
  score_intercept     = -0.4,
  score_intent        = 0.9,   # hidden ingredient -> residual confounding
  score_prior_profit  = 0.012,
  score_prior_orders  = 0.10,
  score_sessions      = 0.05,
  score_eligible      = 0.25,
  score_noise_sd      = 0.6,

  ## treatment assignment (targeted): propensity rises with the score, prior
  ## profit, shipping eligibility, and (hiddenly) intent. Logistic.
  assign_intercept    = -0.5,
  assign_score        = 2.6,
  assign_prior_profit = 0.010,
  assign_eligible     = 0.5,
  assign_intent       = 0.45,  # extra hidden push beyond what score captures

  ## shipping eligibility (can the user actually redeem free shipping?)
  eligible_prob       = 0.6,

  ## true treatment effect: positive ONLY for treated & shipping-eligible users,
  ## EXACTLY zero for ineligible users. Target ATT ~ R$1.5 among treated&eligible.
  effect_base         = 1.5,   # mean true effect for eligible users (R$)
  effect_het_sd       = 0.25,  # mild heterogeneity around the base

  ## profit_30d outcome (post-period), driven by observed covariates AND intent
  profit_intercept    = 30,
  profit_intent       = 4.2,   # intent drives the outcome -> residual naive bias
  profit_prior_profit = 0.12,
  profit_prior_orders = 0.4,
  profit_sessions     = 0.2,
  profit_eligible     = 2,
  profit_sd           = 10,

  ## profit_pre (a pre-treatment-period outcome) -- SAME selection structure, no
  ## treatment term, so treated users look better even before the banner.
  ## Powers placebo-in-time. Residual from intent survives adjustment.
  profit_pre_intercept    = 28,
  profit_pre_intent       = 7,
  profit_pre_prior_profit = 0.5,
  profit_pre_prior_orders = 1.4,
  profit_pre_sessions     = 0.5,
  profit_pre_sd           = 10,

  ## eligible_category_profit -- outcome the banner CAN move (carries the effect)
  elig_cat_intercept = 12,
  elig_cat_intent    = 4,
  elig_cat_share     = 0.6,    # share of the true effect that lands here
  elig_cat_sd        = 6,

  ## noneligible_category_profit -- KNOWN-NULL outcome (banner cannot move it) but
  ## correlated with intent -> naive/adjusted estimate spuriously positive.
  ## Powers the negative-control outcome.
  noneL_cat_intercept = 10,
  noneL_cat_intent    = 5,
  noneL_cat_sd        = 6
)
set.seed(cfg$seed)
n <- cfg$n

## ---- latent shopping intent (truly-unobserved layer) ----
shopping_intent <- rnorm(n, 0, cfg$intent_sd)

## ---- observed context covariates ----
tenure_days <- round(rgamma(n, shape = 2, rate = 1 / 200))
loyalty_tier <- sample(c("bronze", "silver", "gold"), n, replace = TRUE,
                       prob = c(0.5, 0.35, 0.15))
region <- sample(c("north", "south", "east", "west"), n, replace = TRUE)
device <- sample(c("ios", "android", "web"), n, replace = TRUE,
                 prob = c(0.4, 0.4, 0.2))
shipping_eligible <- rbinom(n, 1, cfg$eligible_prob)

## ---- observed-but-omitted layer: pre-treatment behaviour driven by intent ----
prior_orders <- rpois(n, lambda = pmax(0.1, cfg$prior_orders_base +
                                         cfg$prior_orders_intent * shopping_intent))
sessions_pre <- rpois(n, lambda = pmax(0.1, cfg$sessions_base +
                                         cfg$sessions_intent * shopping_intent))
prior_profit <- cfg$prior_profit_base +
  cfg$prior_profit_intent * shopping_intent +
  cfg$prior_profit_orders * prior_orders +
  rnorm(n, 0, cfg$prior_profit_sd)

## ---- business targeting score: observed covariates AND hidden intent ----
score_lin <- cfg$score_intercept +
  cfg$score_intent       * shopping_intent +
  cfg$score_prior_profit * (prior_profit - cfg$prior_profit_base) +
  cfg$score_prior_orders * (prior_orders - cfg$prior_orders_base) +
  cfg$score_sessions     * (sessions_pre - cfg$sessions_base) +
  cfg$score_eligible     * shipping_eligible +
  rnorm(n, 0, cfg$score_noise_sd)
predicted_purchase_score <- plogis(score_lin)  # in (0, 1)

## ---- targeted treatment assignment (logistic propensity) ----
assign_lin <- cfg$assign_intercept +
  cfg$assign_score        * (predicted_purchase_score - 0.5) +
  cfg$assign_prior_profit * (prior_profit - cfg$prior_profit_base) +
  cfg$assign_eligible     * shipping_eligible +
  cfg$assign_intent       * shopping_intent
treatment_propensity <- plogis(assign_lin)
treated <- rbinom(n, 1, treatment_propensity)

## ---- known ground-truth per-user effect (R$) ----
## Positive (mild heterogeneity) for shipping-eligible users, EXACTLY zero for
## ineligible users. Heterogeneity does NOT use shopping_intent, so the public
## proxy never reveals the hidden layer.
het <- rnorm(n, 0, cfg$effect_het_sd)
true_effect <- shipping_eligible * (cfg$effect_base + het)
true_effect_proxy <- true_effect  # public, non-revealing copy

## ---- outcomes ----
profit_30d <- cfg$profit_intercept +
  cfg$profit_intent       * shopping_intent +
  cfg$profit_prior_profit * (prior_profit - cfg$prior_profit_base) +
  cfg$profit_prior_orders * (prior_orders - cfg$prior_orders_base) +
  cfg$profit_sessions     * (sessions_pre - cfg$sessions_base) +
  cfg$profit_eligible     * shipping_eligible +
  treated * true_effect +
  rnorm(n, 0, cfg$profit_sd)

profit_pre <- cfg$profit_pre_intercept +
  cfg$profit_pre_intent       * shopping_intent +
  cfg$profit_pre_prior_profit * (prior_profit - cfg$prior_profit_base) +
  cfg$profit_pre_prior_orders * (prior_orders - cfg$prior_orders_base) +
  cfg$profit_pre_sessions     * (sessions_pre - cfg$sessions_base) +
  rnorm(n, 0, cfg$profit_pre_sd)

eligible_category_profit <- cfg$elig_cat_intercept +
  cfg$elig_cat_intent * shopping_intent +
  treated * (cfg$elig_cat_share * true_effect) +
  rnorm(n, 0, cfg$elig_cat_sd)

noneligible_category_profit <- cfg$noneL_cat_intercept +
  cfg$noneL_cat_intent * shopping_intent +
  rnorm(n, 0, cfg$noneL_cat_sd)

## ---- assemble & split public vs truth ----
user_id <- seq_len(n)
public <- data.frame(
  user_id, treated, shipping_eligible,
  profit_30d, profit_pre,
  eligible_category_profit, noneligible_category_profit,
  prior_orders, prior_profit, sessions_pre,
  tenure_days, loyalty_tier, region, device,
  predicted_purchase_score, true_effect_proxy
)
truth <- data.frame(
  user_id, shopping_intent, true_effect, treatment_propensity
)

# dir.create("data", showWarnings = FALSE)
# readr::write_csv(public, "data/free_shipping_banner.csv")
# readr::write_csv(truth,  "data/free_shipping_banner_truth.csv")
# ^ left commented out on purpose -- see the note above the DGP block.

# ==========================================================
# Read the already-frozen dataset (this is what every section below uses)
# ==========================================================
df <- read.csv("data/free_shipping_banner.csv")

# ==========================================================
# Shared reporting helpers
# ==========================================================
# Every regression result in this script prints the estimate, the standard
# error, and a 95% Student-t interval built from the model's residual degrees
# of freedom. The Python companion prints the identical schema, so the two
# languages can be compared line by line.

report_regression <- function(label, model, term = "treated",
                              covariance = "HC1") {
  test <- coeftest(model, vcov. = vcovHC(model, type = covariance))
  estimate <- test[term, "Estimate"]
  se <- test[term, "Std. Error"]
  critical <- qt(0.975, df = df.residual(model))
  cat(sprintf(
    "%s: estimate = %.4f; %s SE = %.4f; 95%% Student-t CI = [%.4f, %.4f]\n",
    label, estimate, covariance, se,
    estimate - critical * se, estimate + critical * se
  ))
}

report_classical <- function(label, model, term = "treated") {
  test <- summary(model)$coefficients
  estimate <- test[term, "Estimate"]
  se <- test[term, "Std. Error"]
  critical <- qt(0.975, df = df.residual(model))
  cat(sprintf(
    "%s: estimate = %.4f; classical SE = %.4f; 95%% Student-t CI = [%.4f, %.4f]\n",
    label, estimate, se, estimate - critical * se, estimate + critical * se
  ))
}

report_correlation <- function(label, x, y) {
  keep <- complete.cases(x, y)
  r <- cor(x[keep], y[keep], method = "pearson")
  delta <- qnorm(0.975) / sqrt(sum(keep) - 3)
  cat(sprintf(
    "%s: Pearson r = %.4f; 95%% Fisher CI = [%.4f, %.4f]\n",
    label, r, tanh(atanh(r) - delta), tanh(atanh(r) + delta)
  ))
}

CORE <- "treated + prior_orders + prior_profit + sessions_pre"

# ==========================================================
# ANALYSIS STAGES: the raw comparison and the intermediate core model
# ==========================================================
# The two numbers the chapter opens with, and the simulated truths they are
# graded against. Both interval rows use classical (homoskedastic) standard
# errors; every diagnostic below uses HC1 unless it says otherwise.

report_classical("raw comparison", lm(profit_30d ~ treated, data = df))
report_classical("intermediate core controls",
                 lm(as.formula(paste("profit_30d ~", CORE)), data = df))
#> raw comparison: estimate = 7.8558; classical SE = 0.1373; 95% Student-t CI = [7.5867, 8.1249]
#> intermediate core controls: estimate = 2.1866; classical SE = 0.1331; 95% Student-t CI = [1.9257, 2.4475]

# Simulated truths. true_effect_proxy is the per-user ground truth that ships
# in the public CSV; the latent confounder itself never does.
cat(sprintf("targeted-user ATT truth   = %.4f\n",
            mean(df$true_effect_proxy[df$treated == 1])))
cat(sprintf("whole-base ATE truth      = %.4f\n", mean(df$true_effect_proxy)))
cat(sprintf("untreated ATC truth       = %.4f\n",
            mean(df$true_effect_proxy[df$treated == 0])))
cat(sprintf("eligible-targeted truth   = %.4f\n",
            mean(df$true_effect_proxy[df$treated == 1 & df$shipping_eligible == 1])))
cat(sprintf("ineligible-targeted truth = %.4f\n",
            mean(df$true_effect_proxy[df$treated == 1 & df$shipping_eligible == 0])))
#> 0.9877 targeted ATT; 0.8981 whole-base ATE; 0.8176 untreated ATC;
#> 1.4982 eligible-targeted; 0.0000 ineligible-targeted

# ==========================================================
# PLACEBO IN TIME: a "fake effect" before the banner existed
# ==========================================================
# The banner cannot have changed anything before it launched. Swap the
# outcome for the pre-period profit and re-run the intermediate spec; the
# 'treated' coefficient should be ~0 if the comparison is clean.
# It is not: intermediate placebo-in-time = R$1.59 [R$1.32, R$1.86].

placebo_time <- lm(as.formula(paste("profit_pre ~", CORE)), data = df)
report_regression("intermediate placebo in time", placebo_time)
#> intermediate placebo in time: estimate = 1.5918; HC1 SE = 0.1366; 95% Student-t CI = [1.3239, 1.8596]

# ==========================================================
# NEGATIVE-CONTROL OUTCOMES: an effect where the banner cannot reach
# ==========================================================
# Swap the outcome for spend in categories the free-shipping offer does not
# apply to. The banner has no mechanism to move it, so treated should be ~0.
# It is not: intermediate negative-control = R$1.11 [R$0.95, R$1.28].

negctrl <- lm(
  as.formula(paste("noneligible_category_profit ~", CORE)), data = df
)
report_regression("intermediate negative control", negctrl)
#> intermediate negative control: estimate = 1.1148; HC1 SE = 0.0838; 95% Student-t CI = [0.9505, 1.2791]

# ==========================================================
# PLACEBO TREATMENTS: an effect from an assignment that did nothing
# ==========================================================
# Two checks: a deterministic null permutation (destroys targeting
# structure entirely) and a score-preserving fake exposure (keeps the
# selection pressure in the targeting score, but among users who were
# never actually treated). Both condition on the real 'treated' indicator.

n_rows <- nrow(df)

# fake_exposure_n gives the fake label the same share of the untreated pool
# that real exposure has of the whole base: floor(14209 / 30000 * 15791) = 7479.
placebo_cfg <- list(
  lcg_multiplier = 104729,
  lcg_offset = 12345,
  fake_exposure_n = floor(sum(df$treated) / n_rows * sum(df$treated == 0))
)
stopifnot(placebo_cfg$fake_exposure_n == 7479)

# A deterministic null permutation, identical in R and Python.
i0 <- 0:(n_rows - 1)
idx <- ((placebo_cfg$lcg_multiplier * i0 + placebo_cfg$lcg_offset) %% n_rows) + 1
stopifnot(length(unique(idx)) == n_rows)
df$placebo_null <- df$treated[idx]

# A score-preserving fake exposure among users who were not treated.
untreated <- which(df$treated == 0)
ordered <- untreated[order(-df$predicted_purchase_score[untreated],
                           df$user_id[untreated])]
df$fake_score_exposure <- 0L
df$fake_score_exposure[head(ordered, placebo_cfg$fake_exposure_n)] <- 1L
stopifnot(sum(df$fake_score_exposure) == placebo_cfg$fake_exposure_n)

null_fit <- lm(
  profit_30d ~ placebo_null + treated +
    prior_orders + prior_profit + sessions_pre,
  data = df
)
score_fit <- lm(
  profit_30d ~ fake_score_exposure + treated +
    prior_orders + prior_profit + sessions_pre,
  data = df
)

report_regression("deterministic null", null_fit, "placebo_null")
report_regression("score-preserving fake exposure", score_fit, "fake_score_exposure")
#> deterministic null: estimate = -0.0626; HC1 SE = 0.1197; 95% Student-t CI = [-0.2972, 0.1720]
#> score-preserving fake exposure: estimate = 1.5868; HC1 SE = 0.1838; 95% Student-t CI = [1.2266, 1.9471]

# ==========================================================
# PLACEBO UNITS: groups the banner could not have helped
# ==========================================================
# shipping_eligible == 0 users cannot redeem free shipping, so their true
# banner effect is R$0.00 by construction. A design that removes selection
# should return roughly zero for their banner contrast.

ineligible <- df |> filter(shipping_eligible == 0)

# 1) The observational estimator on the placebo units.
#    Need NOT be zero: selection still operates among the ineligible.
obs_placebo <- lm(
  profit_30d ~ treated + prior_orders + prior_profit + sessions_pre,
  data = ineligible
)
report_regression("ineligible observational contrast", obs_placebo)
#> ineligible observational contrast: estimate = 1.1231; HC1 SE = 0.2146; 95% Student-t CI = [0.7025, 1.5437]

# 2) Permutation calibration: fake the assignment within a group that
#    cannot benefit, and see how often we reject the null.
set.seed(42)
reject <- replicate(1000, {
  ineligible$fake <- sample(ineligible$treated)
  m <- lm(profit_30d ~ fake + prior_orders + prior_profit + sessions_pre,
          data = ineligible)
  test <- coeftest(m, vcov. = vcovHC(m, type = "HC1"))
  abs(test["fake", "t value"]) > qt(0.975, df = m$df.residual)
})
mean(reject)   # 0.046: RNG-specific Monte Carlo result near 0.05

# ==========================================================
# BALANCE AND OVERLAP: are we comparing comparable users?
# ==========================================================
# Standardized mean differences (SMDs) put covariate gaps on a common
# scale; propensity-score overlap asks whether every kind of user appears
# in both groups. Good observed balance is necessary, not sufficient --
# it says nothing about the latent shopping_intent no column captures.

propensity <- treated ~ prior_orders + prior_profit + sessions_pre +
  predicted_purchase_score + tenure_days + shipping_eligible
ps_model <- glm(propensity, data = df, family = binomial())
matched <- matchit(
  propensity,
  data = df,
  method = "nearest",
  distance = predict(ps_model, type = "response"),
  estimand = "ATT",
  replace = TRUE,
  ratio = 1,
  normalize = FALSE
)

# Overlap: the estimated propensity distribution in each group.
df$propensity_score <- predict(ps_model, type = "response")
print(
  ggplot(df, aes(x = propensity_score, fill = factor(treated))) +
    geom_density(alpha = 0.5, color = NA) +
    labs(x = "Estimated probability of targeting", y = "Density",
         fill = "Targeted")
)

# SMDs use the treated group's SD because the target is the ATT.
bal.tab(matched, un = TRUE, binary = "std", s.d.denom = "treated")
love.plot(matched, abs = TRUE, binary = "std", s.d.denom = "treated",
          thresholds = c(m = 0.10))

matched_df <- match_data(matched, data = df)
control_w <- matched_df$weights[matched_df$treated == 0]
sum(matched_df$treated == 0)            # 6017 unique controls
sum(control_w)^2 / sum(control_w^2)     # control ESS = 2727.325

# ==========================================================
# D-SEPARATION: the falsification you do not yet do
# ==========================================================
# A candidate DAG implies conditional independencies that can sometimes be
# checked in observed data. A failed check is evidence against that
# maintained graph-plus-model implication, not a full verdict on the story.

# Candidate graph under test.
g <- dagitty('dag {
  prior_orders -> predicted_purchase_score
  prior_profit -> predicted_purchase_score
  sessions_pre -> predicted_purchase_score
  predicted_purchase_score -> treated
  prior_orders -> treated;  prior_profit -> treated;  sessions_pre -> treated
  prior_orders -> profit_30d;  prior_profit -> profit_30d;  sessions_pre -> profit_30d
  treated -> profit_30d
}')

# The DAG IMPLIES a list of conditional independencies. Read them off.
impliedConditionalIndependencies(g)

# Report all four candidate-graph implications.
report_correlation("prior_orders _||_ prior_profit",
                   df$prior_orders, df$prior_profit)
report_correlation("prior_orders _||_ sessions_pre",
                   df$prior_orders, df$sessions_pre)
report_correlation("prior_profit _||_ sessions_pre",
                   df$prior_profit, df$sessions_pre)

# The primary test residualizes score and profit on exposure and core controls.
r_score  <- resid(lm(predicted_purchase_score ~ treated + prior_orders +
                       prior_profit + sessions_pre, data = df))
r_profit <- resid(lm(profit_30d ~ treated + prior_orders +
                       prior_profit + sessions_pre, data = df))
report_correlation("predicted_purchase_score _||_ profit_30d after intermediate conditioning",
                   r_score, r_profit)
#> prior_orders _||_ prior_profit: Pearson r = 0.7343; 95% Fisher CI = [0.7290, 0.7395]
#> prior_orders _||_ sessions_pre: Pearson r = 0.2166; 95% Fisher CI = [0.2058, 0.2274]
#> prior_profit _||_ sessions_pre: Pearson r = 0.3655; 95% Fisher CI = [0.3557, 0.3753]
#> predicted_purchase_score _||_ profit_30d after intermediate conditioning: Pearson r = 0.1188; 95% Fisher CI = [0.1076, 0.1300]

# ==========================================================
# ALIGN THE FOCAL MODEL WITH THE TARGETED-USER ATT
# ==========================================================
# The intermediate model assumes one constant effect even though eligibility
# determines whether free shipping can matter. Add treated * eligibility,
# centered so the 'treated' coefficient reads as the targeted-population
# contrast: R$1.87 [R$1.60, R$2.13] with HC1.

treated_eligibility_share <- mean(df$shipping_eligible[df$treated == 1])
df$eligibility_centered <-
  df$shipping_eligible - treated_eligibility_share
FOCAL_RHS <- paste("treated * eligibility_centered +",
                   "prior_orders + prior_profit + sessions_pre")

focal <- lm(as.formula(paste("profit_30d ~", FOCAL_RHS)), data = df)
report_regression("focal targeted-standardized contrast", focal)
#> focal targeted-standardized contrast: estimate = 1.8657; HC1 SE = 0.1341; 95% Student-t CI = [1.6028, 2.1285]

raw <- lm(
  profit_30d ~ treated * shipping_eligible +
    prior_orders + prior_profit + sessions_pre,
  data = df
)
# Both parameterizations are the same fit with different labels.
stopifnot(max(abs(fitted(focal) - fitted(raw))) < 1e-9)
df$ineligibility <- 1 - df$shipping_eligible
raw_eligible <- lm(
  profit_30d ~ treated * ineligibility +
    prior_orders + prior_profit + sessions_pre,
  data = df
)
report_regression("raw ineligible-user contrast", raw)
report_regression("raw treatment-by-eligibility interaction", raw,
                  "treated:shipping_eligible")
report_regression("raw eligible-user contrast", raw_eligible)
#> raw ineligible-user contrast: estimate = 1.2491; HC1 SE = 0.2006; 95% Student-t CI = [0.8558, 1.6423]
#> raw treatment-by-eligibility interaction: estimate = 0.9353; HC1 SE = 0.2444; 95% Student-t CI = [0.4563, 1.4143]
#> raw eligible-user contrast: estimate = 2.1844; HC1 SE = 0.1638; 95% Student-t CI = [1.8634, 2.5054]

# ==========================================================
# FOCAL RECHECK: rerun every applicable diagnostic on the focal model
# ==========================================================
# The placebo-treatment rows add the placebo indicator as one more regressor
# on the focal right-hand side. Placebo-in-time and negative-control keep the
# focal specification and swap only the outcome. The ineligible placebo-unit
# recheck is the "raw ineligible-user contrast" printed just above: the same
# fitted model written with raw shipping_eligible instead of the centered
# version, which reads the contrast at eligibility zero.

report_regression("focal placebo in time",
                  lm(as.formula(paste("profit_pre ~", FOCAL_RHS)), data = df))
report_regression("focal negative control",
                  lm(as.formula(paste("noneligible_category_profit ~", FOCAL_RHS)),
                     data = df))
report_regression("focal deterministic null",
                  lm(as.formula(paste("profit_30d ~ placebo_null +", FOCAL_RHS)),
                     data = df),
                  "placebo_null")
report_regression("focal score-preserving fake exposure",
                  lm(as.formula(paste("profit_30d ~ fake_score_exposure +", FOCAL_RHS)),
                     data = df),
                  "fake_score_exposure")
report_correlation(
  "predicted_purchase_score _||_ profit_30d after focal conditioning",
  resid(lm(as.formula(paste("predicted_purchase_score ~", FOCAL_RHS)), data = df)),
  resid(lm(as.formula(paste("profit_30d ~", FOCAL_RHS)), data = df))
)
#> focal placebo in time: estimate = 1.6037; HC1 SE = 0.1384; 95% Student-t CI = [1.3325, 1.8750]
#> focal negative control: estimate = 1.1267; HC1 SE = 0.0849; 95% Student-t CI = [0.9604, 1.2931]
#> focal deterministic null: estimate = -0.0507; HC1 SE = 0.1187; 95% Student-t CI = [-0.2835, 0.1820]
#> focal score-preserving fake exposure: estimate = 1.3766; HC1 SE = 0.1828; 95% Student-t CI = [1.0182, 1.7349]
#> predicted_purchase_score _||_ profit_30d after focal conditioning: Pearson r = 0.1058; 95% Fisher CI = [0.0946, 0.1170]

# ==========================================================
# ROBUSTNESS GRID: six defensible specifications, one lever at a time
# ==========================================================
# Core controls, context covariates, the targeting score, common-support
# trimming, ATT weighting, and 1:1 nearest-neighbour matching -- read this
# grid for effect-size stability, never as an average.

focal_formula <- as.formula(paste("profit_30d ~", FOCAL_RHS))

core <- lm(focal_formula, data = df)
context <- lm(
  update(focal_formula, . ~ . + tenure_days + factor(loyalty_tier) +
           factor(region) + factor(device)),
  data = df
)
score <- lm(update(formula(context), . ~ . + predicted_purchase_score), data = df)

ps_t <- range(df$predicted_purchase_score[df$treated == 1])
ps_c <- range(df$predicted_purchase_score[df$treated == 0])
cs_lo <- max(ps_t[1], ps_c[1]); cs_hi <- min(ps_t[2], ps_c[2])
df_trim <- df |> filter(predicted_purchase_score >= cs_lo,
                        predicted_purchase_score <= cs_hi)
trimmed <- lm(focal_formula, data = df_trim)

# ATT weighting and 1:1 nearest-neighbour ATT matching with replacement.
w <- weightit(propensity, data = df, method = "glm", estimand = "ATT")
ipw <- lm(focal_formula, data = df, weights = w$weights)
ipw_placebo_time <- lm(
  update(focal_formula, profit_pre ~ .), data = df, weights = w$weights
)
ipw_negative_control <- lm(
  update(focal_formula, noneligible_category_profit ~ .),
  data = df, weights = w$weights
)
ps_model2 <- glm(propensity, data = df, family = binomial())
mm <- matchit(
  propensity, data = df, method = "nearest",
  distance = predict(ps_model2, type = "response"), estimand = "ATT",
  replace = TRUE, ratio = 1, normalize = FALSE
)
matched_grid <- match_data(mm, data = df)
nn <- lm(focal_formula, data = matched_grid, weights = weights)

# HC1 for five rows; HC3 for the fitted match. Every row prints the
# estimate, robust SE, and a Student-t interval using its residual df.
models <- list(core = core, context = context, score = score,
               trimmed = trimmed, ipw = ipw, matching = nn)
walk(names(models), function(name) {
  covariance <- if (name == "matching") "HC3" else "HC1"
  report_regression(name, models[[name]], "treated", covariance = covariance)
})
report_regression("IPW placebo in time", ipw_placebo_time)
report_regression("IPW noneligible-category placebo", ipw_negative_control)
cat(sprintf(
  "common-support trim: %d rows removed; %d treated removed; targeted truth = %.6f\n",
  nrow(df) - nrow(df_trim),
  sum(df$treated == 1) - sum(df_trim$treated == 1),
  mean(df_trim$true_effect_proxy[df_trim$treated == 1])
))
# adjusted estimates: 1.87, 1.86, 1.21, 1.86, 1.22, 1.08
# IPW placebo in time: estimate = 0.6296; HC1 SE = 0.1733; 95% Student-t CI = [0.2900, 0.9692]
# IPW noneligible-category placebo: estimate = 0.3902; HC1 SE = 0.1047; 95% Student-t CI = [0.1849, 0.5955]
# common-support trim: 18 rows removed; 10 treated removed; targeted truth = 0.987911

# Changing the population changes the question. Restricting the core model to
# shipping-eligible users is graded against the eligible-targeted truth
# (R$1.50), not the R$0.99 truth for all targeted users, so it is NOT part of
# the six-specification same-population grid above.
report_regression("eligible-only core",
                  lm(as.formula(paste("profit_30d ~", CORE)), data = df,
                     subset = shipping_eligible == 1))
#> eligible-only core: estimate = 2.2664; HC1 SE = 0.1707; 95% Student-t CI = [1.9317, 2.6010]

# ==========================================================
# SENSITIVITY TO UNOBSERVED CONFOUNDING
# ==========================================================
# The focal model still rests on the assumption that observed controls
# capture the selection the targeting system acted on. Sensitivity
# analysis bounds how strong a hidden confounder would have to be to
# overturn the result. Uses the partial-R^2 framework (sensemakr in R,
# PySensemakr in Python), applied to the focal classical OLS coefficient.

model_sens <- lm(as.formula(paste("profit_30d ~", FOCAL_RHS)), data = df)

# Sensitivity of the `treated` coefficient, benchmarked against the
# strongest observed control, prior_profit.
sens <- sensemakr(
  model_sens, treatment = "treated",
  benchmark_covariates = "prior_profit", kd = 1
)
summary(sens)
# RV = 0.077; RV-alpha = 0.067; treatment partial R2 = 0.0065
# formal 1x prior-profit bound: R$-1.19 [R$-1.44, R$-0.94]
plot(sens)

# Closed form from the t-stat and df alone (classical SEs):
est <- summary(model_sens)$coefficients["treated", ]
t_stat  <- est["t value"]; dfree <- model_sens$df.residual
partial_r2 <- t_stat^2 / (t_stat^2 + dfree)          # 0.0065
f_stat  <- t_stat / sqrt(dfree)
rv <- 0.5 * (sqrt(f_stat^4 + 4 * f_stat^2) - f_stat^2)    # 0.077

# ==========================================================
# MECHANISM AND SPECIFICITY: does the effect show up only where it can?
# ==========================================================
# The mechanism predicts a larger contrast where free shipping can act
# than where it cannot. Fit the same focal model to eligible-category
# profit, noneligible-category profit, and their per-user difference.

df$category_difference <-
  df$eligible_category_profit - df$noneligible_category_profit

elig <- lm(as.formula(paste("eligible_category_profit ~", FOCAL_RHS)), data = df)
nonelig <- lm(as.formula(paste("noneligible_category_profit ~", FOCAL_RHS)), data = df)
difference <- lm(as.formula(paste("category_difference ~", FOCAL_RHS)), data = df)

walk2(
  c("eligible", "noneligible", "difference"),
  list(elig, nonelig, difference),
  report_regression
)
#> eligible: estimate = 1.4938; HC1 SE = 0.0824; 95% Student-t CI = [1.3324, 1.6552]
#> noneligible: estimate = 1.1267; HC1 SE = 0.0849; 95% Student-t CI = [0.9604, 1.2931]
#> difference: estimate = 0.3671; HC1 SE = 0.1109; 95% Student-t CI = [0.1497, 0.5845]

# ==========================================================
# HETEROGENEITY AS FALSIFICATION: leave-one-region-out
# ==========================================================
# Re-estimate the focal model after dropping each region in turn. If one
# deletion moves the estimate sharply, the aggregate depends on that slice.

loo <- map_dfr(sort(unique(df$region)), function(r) {
  m <- feols(as.formula(paste("profit_30d ~", FOCAL_RHS)),
             data = filter(df, region != r), vcov = "hetero")
  tibble(dropped = r, treated = coef(m)["treated"])
})

full <- coef(feols(as.formula(paste("profit_30d ~", FOCAL_RHS)),
                   data = df, vcov = "hetero"))["treated"]
loo |> mutate(full = full)   # if one row swings far from `full`, be suspicious
# range R$1.72-R$1.95; spread R$0.23

# ==========================================================
# APPENDIX: SPECIFICATION CURVE
# ==========================================================
# When the grid of defensible choices grows large, a specification curve
# plots the estimate across every pre-declared combination rather than
# selecting one preferred path.

specs <- setup(
  data = df,
  y = "profit_30d",
  x = "treated",
  model = "lm",
  controls = c("prior_orders", "prior_profit", "sessions_pre",
               "predicted_purchase_score", "tenure_days")
)
results <- specr(specs)   # one row per defensible specification
plot(results)             # the specification curve: read the spread, not a point
