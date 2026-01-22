##########################################################
## Everyday Causal Inference: How to estimate, test, and explain impacts with R and Python
## www.everydaycausal.com
## Copyright © 2025 by Robson Tigre. All rights reserved.
## You may read, share, and cite for learning purposes, provided you credit the source.
## It should not be used to create competing educational or commercial products
##########################################################
## Code for Chapter 8 - Regression discontinuity design: Cutoffs as natural experiments
## Created: Jan 13, 2026
## Last modified: Jan 19, 2026
##########################################################

# -----------------------------------------------------------------------------
# 1. Setup and Packages
# -----------------------------------------------------------------------------
# install.packages(c("tidyverse", "rdrobust", "rddensity", "lpdensity"))

library(tidyverse) # Data manipulation and visualization
library(rdrobust) # RDD estimation and plotting
library(rddensity) # Density tests for manipulation
library(lpdensity) # Local polynomial density estimation
rm(list = ls())
# Avoid scientific notation
options(scipen = 999)

# -----------------------------------------------------------------------------
# 2. Engineering the Data: A Case of Hidden Bias
# -----------------------------------------------------------------------------
# We're simulating a scenario where "Naive" analysis fails.
#
# The setup:
# - E-commerce platform with a loyalty program.
# - Customers with > 50 points get "Free Shipping" (Treatment).
# - Unobserved "Engagement" (U) drives both Loyalty (X) and Spending (Y).
#
# This creates a classic selection problem:
# High-engagement users are more likely to have free shipping AND spend more anyway.
# A simple comparison will vastly overstate the effect of free shipping.
#
# Additionally, we bake in a non-linear relationship between Loyalty and Spending.
# This ensures that even "OLS with controls" will likely fail (due to misspecification),
# allowing RDD to shine as the robust alternative.
set.seed(2024) # For reproducibility (matches appendix in regression-discontinuity.qmd)
n <- 20000

# Step 1: The Confounder (U)
# U = Customer Engagement. We can't see this, but it drives everything.
U <- rnorm(n, mean = 0, sd = 1)

# Step 2: Observed Covariates
# We observe Age and History.
# They are correlated with U, but continuous at the cutoff.
# We'll use them later to show how they improve precision in RDD.

age <- 35 + 2 * U + rnorm(n, 0, 5)
age <- round(pmax(18, age))

history <- 300 + 80 * U + rnorm(n, 0, 50)
history <- pmax(0, history)

# Step 3: Running Variable (Loyalty Score)
# More engaged users (high U) naturally have higher loyalty scores.
x_star <- 50 + 15 * U + rnorm(n, 0, 8)
x <- pmin(pmax(x_star, 0), 100)

# Step 4: Treatment Assignment (Sharp RDD)
treatment <- as.numeric(x >= 50)

# Step 5: Outcome (Future Spending)
# Here's where we mix the ingredients:
# 1. Base spending (intercept)
# 2. A CUBIC relationship with X (concave then convex). This non-linearity is tricky for OLS.
# 3. The True Treatment Effect (250).
# 4. The Bias: "60 * U". Since U is higher for treated units, this inflates differences.
# 5. Some noise.

y_noise <- rnorm(n, 0, 70)

y <- 600 +
    0.003 * (x - 50)^3 + # The non-linear "S-curve"
    250 * treatment + # The signal we want to recover (Truth = 250)
    60 * U + # The selection bias
    0.4 * history + # Covariate influence
    y_noise

y <- pmax(0, y)

# Step 6: Assemble and Save Data
data <- data.frame(
    x = x,
    y = y,
    treatment = treatment,
    age = age,
    history = history
)

# Save the data to CSV
# write.csv(data, "/Users/robsontigre/Desktop/everyday-ci/data/sharp_rdd_data_example.csv", row.names = FALSE)

# -----------------------------------------------------------------------------
# 3. Analysis: Recovering the Causal Effect
# -----------------------------------------------------------------------------
# Load pre-generated data (or use the data frame we just created):
data <- read.csv("/Users/robsontigre/Desktop/everyday-ci/data/sharp_rdd_data_example.csv")

skimr::skim(data)

# 3.1 Validation Checks
# Before running RDD, we must verify our assumptions.

# CHECK 1: Manipulation
# Are people "schooling" the system to get just above 50?
# A jump in density at 50 would suggest manipulation.
# We look for a p-value > 0.05 (fail to reject null of continuity).

density_test <- rddensity(data$x, c = 50)
summary(density_test)
# Interpreting the output:
# If p > 0.05, we have evidence that the running variable is NOT manipulated at the cutoff.

rdplotdensity(density_test, data$x,
    title = "Density plotting for manipulation testing",
    xlabel = "Loyalty score over the last 3 months",
    ylabel = "Density"
)

# CHECK 2: Covariate Balance
# Pre-treatment characteristics (Age, History) should not jump at the cutoff.
# We treat them as outcomes in an RDD model to verify this.

# Age should be smooth at the cutoff (p > 0.05)
summary(rdrobust(data$age, data$x, c = 50))

# Purchase history should be smooth at the cutoff (p > 0.05)
summary(rdrobust(data$history, data$x, c = 50))

# 3.2 The Identification Challenge
# Our True Effect is 250.
# Let's see how different methods perform in recovering this.

# --- Naive Comparison ---
# Just regress Y on Treatment. Ignores all confounders.
# This will be heavily biased upward because treated units have higher U.
naive <- lm(y ~ treatment, data = data)
summary(naive)
# Expected: Coefficient >> 250 (selection bias inflates the estimate)

# --- OLS with Covariates ---
# Control for running variable and observed covariates.
# Still biased due to: (1) non-linear relationship with X, (2) unobserved U.
ols_with_covs <- lm(y ~ treatment + x + age + history, data = data)
summary(ols_with_covs)
# Expected: Coefficient != 250 (functional form misspecification)

# ---  RDD (The Solution) ---
# We don't guess the functional form globally. We just compare the edges.
# Y = f(x) + tau*D + e ... but we estimate locally.

rdd_result <- rdrobust(data$y, data$x, c = 50)
summary(rdd_result)

# The RDD coefficient should be much closer to 250.
# It handles the selection bias AND the non-linear functional form.

# --- RDD with Covariates ---
# We add Age and History.
# Since they are balanced, they shouldn't change the coefficient much.
# But they should soak up residual variance, shrinking the Standard Error.

rdd_result_cov <- rdrobust(data$y, data$x,
    c = 50,
    covs = data[, c("age", "history")]
)
summary(rdd_result_cov)

# -----------------------------------------------------------------------------
# 3.3 Robustness Checks: Kernels and Polynomials
# -----------------------------------------------------------------------------
# These exercises verify that our main estimate is not sensitive to
# arbitrary modeling choices. A robust finding should hold across specifications.

# --- Different Kernels ---
# The kernel determines how we weight observations based on their distance
# from the cutoff. Observations closer to the cutoff get more weight.
# The shape of the weighting function shouldn't change our conclusions.

# Uniform kernel - gives equal weight to all obs within bandwidth
rdd_uniform <- rdrobust(data$y, data$x, c = 50, kernel = "uniform")
summary(rdd_uniform)

# --- Different Polynomial Orders ---
# The polynomial order determines how flexible we allow the relationship
# between the running variable and outcome to be on each side of the cutoff.
# Higher orders are more flexible but risk overfitting near boundaries.

# Quadratic (p = 2) - more flexible, allows for curvature
rdd_quadratic <- rdrobust(data$y, data$x, c = 50, p = 2)
summary(rdd_quadratic)

# --- Extract Results for Table ---
# Helper function to extract key results from rdrobust output
extract_rdd_results <- function(rdd_obj, label) {
    coef <- rdd_obj$coef[1] # Conventional estimate
    ci_l <- rdd_obj$ci[3, 1] # Robust CI lower
    ci_u <- rdd_obj$ci[3, 2] # Robust CI upper
    bw <- rdd_obj$bws[1, 1] # Bandwidth (left = right for symmetric)
    se <- rdd_obj$se[3] # Robust SE
    pval <- rdd_obj$pv[3] # Robust p-value

    data.frame(
        Specification = label,
        Estimate = round(coef, 1),
        CI_lower = round(ci_l, 1),
        CI_upper = round(ci_u, 1),
        Bandwidth = round(bw, 2),
        SE = round(se, 1),
        P_value = round(pval, 4)
    )
}

# Create results table
results_table <- rbind(
    extract_rdd_results(rdd_result, "Main (linear, triangular)"),
    extract_rdd_results(rdd_result_cov, "With covariates"),
    extract_rdd_results(rdd_uniform, "Uniform kernel"),
    extract_rdd_results(rdd_quadratic, "Quadratic polynomial")
)

print(results_table)

# -----------------------------------------------------------------------------
# 3.4 Visualization: Raw Data vs Binned Data
# -----------------------------------------------------------------------------
# Binning helps us see the pattern by reducing noise.
# We group observations into bins and calculate local averages.
# Important: The regression is still run on the RAW data, not the bins!

# Subset data for plotting clarity (zoom in around cutoff)
subset_data <- data %>% filter(x > 40 & x < 60)

# --- Raw Data Scatter Plot ---
# This shows the individual data points - notice how noisy it is!
raw_scatter <- ggplot(subset_data, aes(x = x, y = y, color = factor(treatment))) +
    geom_point(alpha = 0.3, size = 1.65) +
    geom_vline(xintercept = 50, linetype = "dashed", color = "#F18F01", linewidth = 1.2) +
    scale_color_manual(
        values = c("0" = "#8C8C8C", "1" = "#5C5C5C"),
        labels = c("Control", "Treated"),
        name = "Group"
    ) +
    scale_x_continuous(breaks = seq(40, 60, by = 2)) +
    labs(
        title = "Raw data: Individual observations",
        subtitle = "Each point is a customer; hard to see the pattern through the noise",
        x = "Loyalty score",
        y = "Future spending (R$)"
    ) +
    theme_minimal(base_size = 14) +
    theme(
        plot.title = element_text(face = "bold"),
        plot.background = element_rect(fill = "white", color = NA),
        panel.background = element_rect(fill = "white", color = NA),
        legend.position = "bottom",
        axis.title = element_text(size = 14 * 1.2)
    )

print(raw_scatter)
ggsave("images/rdd_raw_scatter.png", plot = raw_scatter, width = 10, height = 7, dpi = 300)

# 3.5 Visualization (binned plot)

# Define bandwidth boundaries for visualization
bw_min <- 45
bw_max <- 55

# Calculate y-coordinates for the effect arrow at x = 50
# We use a simple linear fit on the subset to place the arrow visually
y_left <- predict(lm(y ~ x, data = subset_data, subset = x < 50), newdata = data.frame(x = 50))
y_right <- predict(lm(y ~ x, data = subset_data, subset = x >= 50), newdata = data.frame(x = 50))

# Use rdplot to bin data appropriately and fit local polynomials
rdplot_base <- rdplot(
    y = subset_data$y, x = subset_data$x, c = 50, ci = 95,
    title = "RDD plot: Free shipping effect on future spending",
    y.label = "Future spending (R$)",
    x.label = "Loyalty score"
)$rdplot

# Add bandwidth shading and styling
rdplot_ <- rdplot_base +
    # Bandwidth shading
    annotate("rect",
        xmin = bw_min, xmax = bw_max, ymin = -Inf, ymax = Inf,
        fill = "grey90", alpha = 0.4
    ) +
    # Bandwidth boundary lines
    geom_vline(
        xintercept = bw_min, linetype = "dashed",
        color = "grey50", linewidth = 0.8, alpha = 0.6
    ) +
    geom_vline(
        xintercept = bw_max, linetype = "dashed",
        color = "grey50", linewidth = 0.8, alpha = 0.6
    ) +
    # Cutoff line styling
    geom_vline(
        xintercept = 50, linetype = "dashed",
        color = "#F18F01", linewidth = 1.2, alpha = 0.8
    ) +
    # Effect arrow and label
    annotate("segment",
        x = 50, xend = 50, y = y_left, yend = y_right,
        arrow = arrow(length = unit(0.3, "cm"), ends = "both"),
        color = "#2E86AB", linewidth = 1.5
    ) +
    annotate("text",
        x = 51, y = mean(c(y_left, y_right)),
        label = paste("Effect ~ R$250"),
        color = "#2E86AB", fontface = "bold", hjust = 0, size = 5
    ) +
    theme(
        plot.title = element_text(size = 20, face = "bold"),
        axis.title.x = element_text(size = 16 * 1.2),
        axis.title.y = element_text(size = 16 * 1.2),
        axis.text.x = element_text(size = 14),
        axis.text.y = element_text(size = 14),
        plot.background = element_rect(fill = "white", color = NA),
        panel.background = element_rect(fill = "white", color = NA),
        panel.grid.major = element_line(color = "grey90", linewidth = 0.5),
        panel.grid.minor = element_blank()
    )

# Display and save plot
print(rdplot_)
ggsave("images/sharp_rdd_plot.png", plot = rdplot_, width = 12, height = 9, dpi = 300)

# -----------------------------------------------------------------------------
# 4. Fuzzy RDD: Data Generation
# -----------------------------------------------------------------------------
# Fuzzy RDD adds a second source of selection: who TAKES treatment once eligible.
# Unobserved confounder U ("business ambition") drives credit score, loan uptake,
# and revenue, creating selection bias that naive comparisons cannot handle.

set.seed(2025)
n_fuzzy <- 15000

# Unobs. confounder: high-U businesses have better finances, more likely to take loans, higher revenue
U <- rnorm(n_fuzzy, mean = 0, sd = 1)

# Running variable: credit score (correlated with U)
x <- 650 + 40 * U + rnorm(n_fuzzy, 0, 60)
x <- pmin(pmax(x, 400), 900) # Bound between 400 and 900

# Observed covariates (balanced at cutoff, correlated with U)
business_age <- pmax(round(8 + 2 * U + rnorm(n_fuzzy, 0, 3)), 1)
employees <- pmax(round(12 + 3 * U + rpois(n_fuzzy, lambda = 5)), 1)

# Eligibility (instrument Z): deterministic at cutoff
Z <- as.numeric(x >= 650)

# Step 5: Treatment Uptake (D) - the fuzzy part
# Probability depends on eligibility AND unobserved ambition (U)
# U affects both uptake and outcomes, creating selection bias

prob_below <- plogis(-1.5 + 0.8 * U) # ~15-30% depending on U
prob_above <- plogis(1.2 + 0.6 * U) # ~70-90% depending on U

prob_treatment <- ifelse(Z == 1, prob_above, prob_below)
D <- rbinom(n_fuzzy, 1, prob_treatment)

# Expected treatment rates: ~20% below cutoff, ~80% above

# Outcome: TRUE EFFECT = R$5,000; U also affects revenue (selection bias)
treatment_effect <- 5000

y <- 25000 +
    15 * (x - 650) + # Credit score effect
    treatment_effect * D + # TRUE EFFECT (what we want to recover)
    3000 * U + # Selection bias (U affects revenue directly)
    100 * business_age + # Older businesses earn slightly more
    50 * employees + # More employees = more capacity
    rnorm(n_fuzzy, 0, 4000) # Noise

y <- pmax(y, 0) # Revenue can't be negative

# Assemble and save
data_fuzzy <- data.frame(
    x = x, y = y, D = D, Z = Z,
    business_age = business_age,
    employees = employees
)

# Save the data
# write.csv(data_fuzzy, "/Users/robsontigre/Desktop/everyday-ci/data/rdd_fuzzy_fintech.csv", row.names = FALSE)

# -----------------------------------------------------------------------------
# 4.1 Demonstrating the Selection Problem
# -----------------------------------------------------------------------------

# Load data (or use the data frame we just created)
data_fuzzy <- read.csv("/Users/robsontigre/Desktop/everyday-ci/data/rdd_fuzzy_fintech.csv")

# Before running Fuzzy RDD, let's see why naive comparisons fail.

# Naive comparison: Just compare loan-takers to non-loan-takers
naive_fuzzy <- lm(y ~ D, data = data_fuzzy)
summary(naive_fuzzy)

# OLS with controls: still biased (U is unobserved)
ols_fuzzy <- lm(y ~ D + x + business_age + employees, data = data_fuzzy)
summary(ols_fuzzy)

# -----------------------------------------------------------------------------
# 4.2 Validation Checks
# -----------------------------------------------------------------------------

# Manipulation test: p > 0.05 means no evidence of gaming
density_test_fuzzy <- rddensity(data_fuzzy$x, c = 650)
summary(density_test_fuzzy)

rdplotdensity(density_test_fuzzy, data_fuzzy$x,
    title = "Density test: credit score distribution",
    xlabel = "Credit score", ylabel = "Density"
)

# Covariate balance: covariates should not jump at cutoff
summary(rdrobust(data_fuzzy$business_age, data_fuzzy$x, c = 650))
summary(rdrobust(data_fuzzy$employees, data_fuzzy$x, c = 650))

# First stage: eligibility must strongly predict treatment
first_stage <- rdrobust(data_fuzzy$D, data_fuzzy$x, c = 650)
summary(first_stage)
# We need: (1) significant coefficient, (2) large jump in probability

# Visualize the first stage
# Calculate predictions for the arrow at cutoff
# We use loess to match the visual smoother; surface = "direct" allows extrapolation to x=650
loess_left <- loess(D ~ x, data = filter(data_fuzzy, x < 650), control = loess.control(surface = "direct"))
loess_right <- loess(D ~ x, data = filter(data_fuzzy, x >= 650), control = loess.control(surface = "direct"))
y_prob_left <- predict(loess_left, newdata = data.frame(x = 650))
y_prob_right <- predict(loess_right, newdata = data.frame(x = 650))

first_stage_plot <- ggplot(data_fuzzy, aes(x = x, y = D)) +
    # Jitter points vertically to show density at 0 and 1
    geom_point(position = position_jitter(width = 0, height = 0.01), alpha = 0.1, size = 0.6, color = "gray50") +
    geom_smooth(
        data = filter(data_fuzzy, x < 650),
        method = "loess", se = TRUE, color = "#2E86AB", linewidth = 1.2, fill = "#2E86AB", alpha = 0.2
    ) +
    geom_smooth(
        data = filter(data_fuzzy, x >= 650),
        method = "loess", se = TRUE, color = "#2E86AB", linewidth = 1.2, fill = "#2E86AB", alpha = 0.2
    ) +
    geom_vline(xintercept = 650, linetype = "dashed", color = "#F18F01", linewidth = 1.2) +
    # Jump arrow
    annotate("segment",
        x = 650, xend = 650, y = y_prob_left, yend = y_prob_right,
        arrow = arrow(length = unit(0.3, "cm"), ends = "both"),
        color = "#74d5ff", linewidth = 1.2, alpha = 0.7
    ) +
    # Jump Label
    annotate("text",
        x = 660, y = mean(c(y_prob_left, y_prob_right)),
        label = paste0("Jump ~ ", round((y_prob_right - y_prob_left) * 100, 0), "pp"),
        color = "#2E86AB", fontface = "bold", hjust = 0, size = 5
    ) +
    # Arrow 1: Control group who got loan (Always-takers)
    annotate("curve",
        x = 600, y = 0.8, xend = 630, yend = 0.98,
        arrow = arrow(length = unit(0.2, "cm")), curvature = -0.2, color = "gray30"
    ) +
    annotate("text",
        x = 570, y = 0.75,
        label = "These businesses in the\ncontrol group received a loan",
        hjust = 0.5, vjust = 1, size = 5, color = "gray30", lineheight = 0.9
    ) +
    # Arrow 2: Treatment group who didn't get loan (Never-takers)
    annotate("curve",
        x = 700, y = 0.2, xend = 670, yend = 0.02,
        arrow = arrow(length = unit(0.2, "cm")), curvature = -0.2, color = "gray30"
    ) +
    annotate("text",
        x = 740, y = 0.22,
        label = "These businesses in the treatment\ngroup did not receive a loan",
        hjust = 0.5, vjust = 0, size = 5, color = "gray30", lineheight = 0.9
    ) +
    scale_y_continuous(labels = scales::percent_format(), limits = c(-0.01, 1.01), breaks = seq(0, 1, 0.2)) +
    labs(
        title = "First stage: loan receipt probability by credit score",
        subtitle = "The jump at the cutoff is the variation Fuzzy RDD exploits",
        x = "Credit score",
        y = "Probability of getting a loan"
    ) +
    theme_minimal(base_size = 14) +
    theme(
        plot.title = element_text(face = "bold"),
        plot.subtitle = element_text(color = "gray40"),
        axis.title = element_text(size = 14 * 1.3),
        panel.grid.minor = element_blank(),
        plot.background = element_rect(fill = "white", color = NA),
        panel.background = element_rect(fill = "white", color = NA)
    )

print(first_stage_plot)
ggsave("images/rdd_fuzzy_first_stage.png", plot = first_stage_plot, width = 11, height = 7, dpi = 300)

# -----------------------------------------------------------------------------
# 4.3 Estimating the LATE (Local Average Treatment Effect)
# -----------------------------------------------------------------------------
# The `fuzzy` argument tells rdrobust to use the cutoff as an instrument

# Main specification
fuzzy_result <- rdrobust(
    y = data_fuzzy$y, x = data_fuzzy$x, c = 650,
    fuzzy = data_fuzzy$D
)
summary(fuzzy_result)

# With covariates (reduces SE without changing estimate)
fuzzy_result_cov <- rdrobust(data_fuzzy$y, data_fuzzy$x,
    c = 650,
    fuzzy = data_fuzzy$D,
    covs = data_fuzzy[, c("business_age", "employees")]
)
summary(fuzzy_result_cov)

# Robustness checks
fuzzy_uniform <- rdrobust(data_fuzzy$y, data_fuzzy$x,
    c = 650,
    fuzzy = data_fuzzy$D, kernel = "uniform"
)
summary(fuzzy_uniform)

fuzzy_quadratic <- rdrobust(data_fuzzy$y, data_fuzzy$x,
    c = 650,
    fuzzy = data_fuzzy$D, p = 2
)
summary(fuzzy_quadratic)

# -----------------------------------------------------------------------------
# 4.4 Visualization: The Reduced Form Jump
# -----------------------------------------------------------------------------
# rdplot shows reduced form (Y jump); LATE = reduced form / first stage

subset_data_fuzzy <- data_fuzzy %>% filter(x > 550 & x < 750)

# Get rdplot base
rdplot_fuzzy_base <- rdplot(
    y = subset_data_fuzzy$y, x = subset_data_fuzzy$x, c = 650,
    title = "Fuzzy RDD: loan effect on secondary revenue",
    y.label = "Secondary revenue (R$)",
    x.label = "Credit score"
)$rdplot

# Add styling
rdplot_fuzzy <- rdplot_fuzzy_base +
    geom_vline(xintercept = 650, linetype = "dashed", color = "#F18F01", linewidth = 1.2, alpha = 0.8) +
    theme(
        plot.title = element_text(size = 18, face = "bold"),
        axis.title = element_text(size = 14),
        axis.text = element_text(size = 12),
        plot.background = element_rect(fill = "white", color = NA),
        panel.background = element_rect(fill = "white", color = NA),
        panel.grid.major = element_line(color = "grey90", linewidth = 0.5),
        panel.grid.minor = element_blank()
    )

print(rdplot_fuzzy)
ggsave("images/rdd_fuzzy_fintech_result.png", plot = rdplot_fuzzy, width = 10, height = 7, dpi = 300)
