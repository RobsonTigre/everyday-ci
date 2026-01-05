##########################################################
## Everyday Causal Inference: How to estimate, test, and explain impacts with R and Python
## www.everydaycausal.com
## Copyright © 2025 by Robson Tigre. All rights reserved.
## You may read, share, and cite for learning purposes, provided you credit the source.
## It should not be used to create competing educational or commercial products
##########################################################
## Code for Chapter 7 - Instrumental variables: When people don't do what they were assigned to
## Created: Jan 02, 2026
## Last modified: Jan 04, 2026
##########################################################

#########################################
# Setup: Load required libraries
#########################################

# If you haven't already, run this once to install the packages:
# install.packages(c("tidyverse", "AER", "fixest"))

library(tidyverse)
library(AER) # for IV regression (ivreg)
library(fixest) # for regular OLS with robust SE (feols)

#########################################
# Data Simulation: Email Frequency Experiment
#########################################
#
# KEY INSIGHT: The simulation includes an UNOBSERVABLE VARIABLE that the
# data scientist cannot see but affects both:
#   1. Treatment uptake (who opts into daily emails)
#   2. The outcome (revenue)
#
# This creates SELECTION BIAS in naive as-treated comparisons.
# IV/2SLS works because the instrument (invitation) is randomly assigned
# and therefore independent of the unobservable variable.
#
# IV Assumptions satisfied by design:
#   - Relevance: invitation -> daily_emails
#   - Independence: invitation is randomly assigned, independent of U
#   - Exclusion: invitation affects revenue ONLY through daily_emails
#   - Monotonicity: no defiers (invitation only encourages, never discourages)
#   - Two-sided non-compliance: some control users are "always-takers"
#
#########################################

# Set seed for reproducibility
set.seed(42)

# Simulate data
n <- 10000

# Random assignment to invitation (instrument Z)
# This is the KEY: random assignment ensures independence from U
Z <- rbinom(n, 1, 0.5)

# Observed covariates (pre-treatment characteristics)
tenure_months <- pmax(rnorm(n, mean = 24, sd = 12), 1)
past_purchase <- pmax(rnorm(n, mean = 200, sd = 100), 0)
engagement_score <- pmax(rnorm(n, mean = 50, sd = 20), 0)

#########################################
# THE UNOBSERVABLE VARIABLE
# This is the source of selection bias. Think of it as "user enthusiasm"
# or "deal-seeking propensity" - something that drives both:
#   1. Willingness to opt into daily emails
#   2. Baseline spending behavior
#
# The data scientist CANNOT observe this variable. It's in the error term.
#########################################
unobservable_variable <- rnorm(n, mean = 0, sd = 1)

#########################################
# Treatment uptake (daily emails, D)
# Probability depends on:
#   - Z (invitation): randomly assigned, increases opt-in probability
#   - U (unobservable): enthusiastic users are more likely to opt-in
#
# This creates TWO-SIDED non-compliance:
#   - Invited (Z=1): ~40% opt-in (compliers + always-takers)
#   - Control (Z=0): ~5% opt-in (always-takers who find the setting)
#
# The unobservable_variable shifts probabilities for BOTH groups
#########################################

# Logistic model for treatment uptake
# Baseline (Z=0): plogis(-2.9) ≈ 0.05 (5% always-takers)
# Invited (Z=1): plogis(-2.9 + 2.3) = plogis(-0.6) ≈ 0.35 + effect of U
# The 0.5 coefficient on U means enthusiastic users are more likely to opt-in
logit_prob <- -2.9 + 2.3 * Z + 0.5 * unobservable_variable
prob_daily_email <- plogis(logit_prob)
D <- rbinom(n, 1, prob_daily_email)

# Baseline revenue (depends on customer characteristics)
# Note: baseline does NOT depend on unobservable_variable yet
baseline_revenue <- 100 +
    0.5 * tenure_months +
    0.2 * past_purchase +
    0.3 * engagement_score +
    rnorm(n, 0, 30)

# True treatment effect for those who get daily emails
true_LATE <- 15

#########################################
# Observed revenue (Y)
# Depends on:
#   - D (treatment): true causal effect of daily emails
#   - Observed covariates: captured in baseline_revenue
#   - U (unobservable): enthusiastic users spend more REGARDLESS of emails
#
# The key insight: U affects Y directly (8 * unobservable_variable)
# This is the confounding path that biases as-treated analysis
#########################################
revenue <- baseline_revenue + D * true_LATE + 8 * unobservable_variable + rnorm(n, 0, 20)

# Create dataframe (note: unobservable_variable is NOT included - analyst can't see it)
df <- data.frame(
    customer_id = 1:n,
    invitation = Z,
    daily_emails = D,
    revenue = revenue,
    tenure_months = tenure_months,
    past_purchase = past_purchase,
    engagement_score = engagement_score
)

# Save for later use
# write.csv(df, "/Users/robsontigre/Desktop/everyday-ci/data/daily_communications.csv", row.names = FALSE)

head(df)
# skimr::skim(df)

# df %>%
#     janitor::tabyl(invitation, daily_emails) %>%
#     janitor::adorn_totals("row") %>%
#     janitor::adorn_percentages("row") %>%
#     janitor::adorn_pct_formatting(digits = 1) %>%
#     janitor::adorn_ns()
#  invitation             0             1
#           0 94.5% (4,727)  5.5%   (274)
#           1 63.6% (3,178) 36.4% (1,821)
#       Total 79.0% (7,905) 21.0% (2,095)
# Check compliance rates
# Proportion opted in (invited, Z=1): ~36%
# Proportion opted in (control, Z=0): ~5.5%
# First stage (difference): ~31 percentage points

#########################################
# Checking the First Stage (Relevance Assumption)
#########################################

# Read the data from csv
df <- read.csv("/Users/robsontigre/Desktop/everyday-ci/data/daily_communications.csv")

# First stage: effect of invitation on daily email adoption
first_stage <- feols(
    daily_emails ~ invitation + tenure_months +
        past_purchase + engagement_score,
    data = df, vcov = "hetero"
)

# Display results
summary(first_stage)
#                     Estimate Std. Error   t value   Pr(>|t|)
# (Intercept)       0.08276067   0.014693  5.632708 1.8220e-08 ***
# invitation        0.30943768   0.007530 41.094854  < 2.2e-16 *** <---
# tenure_months    -0.00078414   0.000318 -2.466853 1.3647e-02 *
# past_purchase     0.00000962   0.000038  0.255261 7.9853e-01
# engagement_score -0.00022094   0.000187 -1.183602 2.3660e-01
# ---

# Extract F-statistic for weak instrument test
# For one instrument, F-stat is the square of the t-stat
t_stat <- first_stage$coeftable["invitation", "t value"]
f_stat <- t_stat^2
# F-statistic >> 10 indicates a strong instrument

# Visualize first stage
p1st <- df %>%
    group_by(invitation) %>%
    summarise(
        prop_daily_emails = mean(daily_emails),
        se = sqrt(prop_daily_emails * (1 - prop_daily_emails) / n()),
        .groups = "drop"
    ) %>%
    ggplot(aes(x = factor(invitation), y = prop_daily_emails)) +
    geom_col(fill = "#0072B2", alpha = 0.7) +
    geom_errorbar(
        aes(
            ymin = prop_daily_emails - 1.96 * se,
            ymax = prop_daily_emails + 1.96 * se
        ),
        width = 0.2
    ) +
    labs(
        title = "First stage: invitation increases daily email adoption",
        x = "Assigned to invitation",
        y = "Proportion receiving daily emails",
        caption = "Error bars show 95% confidence intervals"
    ) +
    scale_x_discrete(labels = c(
        "0" = "No invitation\n(Control)",
        "1" = "Received invitation\n(Treatment)"
    )) +
    scale_y_continuous(
        labels = scales::percent_format(accuracy = 1),
        limits = c(0, 0.5)
    ) +
    theme_minimal(base_size = 14) +
    theme(
        plot.title = element_text(face = "bold", hjust = 0.5),
        panel.grid.minor = element_blank(),
        plot.background = element_rect(fill = "white", color = NA),
        panel.background = element_rect(fill = "white", color = NA)
    )

ggsave("images/iv_email_firststage.png", plot = p1st, width = 8, height = 6, dpi = 300)

#########################################
# Estimating the ITT (Reduced Form)
#########################################

# ITT / Reduced form: effect of invitation assignment on revenue
itt_model <- feols(
    revenue ~ invitation + tenure_months +
        past_purchase + engagement_score,
    data = df, vcov = "hetero"
)

summary(itt_model)
#                   Estimate Std. Error  t value   Pr(>|t|)
# (Intercept)      99.452540   1.537228 64.69601  < 2.2e-16 ***
# invitation        4.211728   0.745718  5.64789 1.6687e-08 *** <---
# tenure_months     0.486786   0.031821 15.29743  < 2.2e-16 ***
# past_purchase     0.207762   0.003786 54.87064  < 2.2e-16 ***
# engagement_score  0.306029   0.018628 16.42873  < 2.2e-16 ***
# ---

# Extract ITT estimate (effect of being offered the invitation on revenue)
itt_estimate <- coef(itt_model)["invitation"]

#########################################
# Estimating the LATE with IV/2SLS
#########################################

# LATE estimation using 2SLS
# Syntax: outcome ~ treatment + covariates | instrument + covariates
iv_model <- ivreg(
    revenue ~ daily_emails + tenure_months +
        past_purchase + engagement_score |
        invitation + tenure_months +
            past_purchase + engagement_score,
    data = df
)

summary(iv_model, vcov = sandwich, diagnostics = TRUE)
#                   Estimate Std. Error t value Pr(>|t|)
# (Intercept)      98.326092   1.573156  62.502  < 2e-16 ***
# daily_emails     13.610909   2.366146   5.752 9.06e-09 *** <---
# tenure_months     0.497459   0.031271  15.908  < 2e-16 ***
# past_purchase     0.207631   0.003719  55.825  < 2e-16 ***
# engagement_score  0.309036   0.018278  16.908  < 2e-16 ***

# Extract LATE estimate (effect of daily emails for compliers)
# True effect in simulation: 15
# LATE / ITT ratio approximately equals 1 / first_stage_effect
late_estimate <- coef(iv_model)["daily_emails"]

# Robustness check: does adding covariates change the estimate?
# We expect similar results if assignment is truly random
iv_simple <- ivreg(revenue ~ daily_emails | invitation,
    data = df
)
summary(iv_simple)
#              Estimate Std. Error t value Pr(>|t|)
# (Intercept)  167.8467     0.7219 232.492   <2e-16 ***
# daily_emails  12.9552     2.7719   4.674    3e-06 *** <---
# ---

simple_late <- coef(iv_simple)["daily_emails"]
# Simple LATE should be similar if assignment is truly random

# Placebo test: does the instrument affect pre-treatment variables?
# We expect NO effect on past purchases
placebo_test <- feols(past_purchase ~ invitation,
    data = df, vcov = "hetero"
)

# Placebo test: coefficient for invitation should be insignificant (p > 0.05)
summary(placebo_test)
# Standard-errors: Heteroskedasticity-robust
#             Estimate Std. Error    t value  Pr(>|t|)
# (Intercept) 202.6820    1.41599 143.138134 < 2.2e-16 ***
# invitation   -1.1116    1.98321  -0.560504   0.57515 <---
# ---

#########################################
# As-treated bias demonstration
# This shows why naive comparisons are misleading
#########################################

# As-treated analysis:
# Naively compare those who actually took the treatment vs. those who didn't.
# This ignores the random assignment and reintroduces selection bias.
as_treated <- feols(
    revenue ~ daily_emails + tenure_months +
        past_purchase + engagement_score,
    data = df, vcov = "hetero"
)

summary(as_treated)
#                   Estimate Std. Error t value  Pr(>|t|)
# (Intercept)      97.104454   1.473392 65.9054 < 2.2e-16 ***
# daily_emails     18.744423   0.909359 20.6128 < 2.2e-16 *** <---
# tenure_months     0.501997   0.031175 16.1025 < 2.2e-16 ***
# past_purchase     0.207627   0.003715 55.8940 < 2.2e-16 ***
# engagement_score  0.309796   0.018223 17.0005 < 2.2e-16 ***
# ---

# Extract confidence interval
confint(as_treated, level = 0.95)["daily_emails", ]
# daily_emails 16.9619 20.52695
confint(iv_model, level = 0.95)["daily_emails", ]
# daily_emails 8.97082 18.25100

# Extract estimates for comparison
naive_estimate <- coef(as_treated)["daily_emails"]

# Calculate bias
# The naive estimate is biased upward because the unobservable_variable
# (which the analyst cannot see) drives both:
#   1. Who opts into daily emails (enthusiastic users more likely)
#   2. Who spends more money (enthusiastic users spend more regardless)
# IV works because the invitation is RANDOMLY assigned, so it's independent
# of the unobservable_variable.
bias_from_truth <- naive_estimate - true_LATE
percent_bias <- (naive_estimate - true_LATE) / true_LATE * 100
percent_bias

# Compare estimates:
# - True LATE (from DGP): 15
# - As-treated (naive): ~18.7 (biased upward by ~25%)
# - IV/2SLS (LATE): ~13.6 (close to true effect)
