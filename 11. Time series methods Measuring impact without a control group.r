##########################################################
## Everyday Causal Inference: How to estimate, test, and explain impacts with R and Python
## www.everydaycausal.com
## Copyright © 2025 by Robson Tigre. All rights reserved.
## You may read, share, and cite for learning purposes, provided you credit the source.
## It should not be used to create competing educational or commercial products
##########################################################
## Code for 11. Time series methods: Measuring impact without a control group
## Created: Jan 12, 2026
## Last modified: Jan 29, 2026
##########################################################

# Setup -------------------------------------------------------------------
# Install packages if needed:
# install.packages("devtools", dependencies = TRUE)
# install.packages("tidybayes", dependencies = TRUE)
# devtools::install_github("FMenchetti/CausalArima", dependencies = TRUE)

library(tidyverse)
library(forecast)
library(CausalImpact)
library(CausalArima)
library(gridExtra)

# ------------------------------------------------------------------------
# Book-wide Theme and Color Palette
# ------------------------------------------------------------------------

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
                face = "bold",
                size = base_size * 1.3,
                color = "grey20",
                margin = margin(b = 5)
            ),
            plot.subtitle = element_text(
                size = base_size * 0.9,
                color = "grey40",
                margin = margin(b = 15)
            ),
            plot.caption = element_text(
                size = base_size * 0.7,
                color = "grey50",
                hjust = 0,
                margin = margin(t = 10)
            ),
            axis.title = element_text(
                size = base_size,
                color = "grey30"
            ),
            axis.text = element_text(
                size = base_size * 0.85,
                color = "grey40"
            ),
            panel.grid.major = element_line(color = "grey90", linewidth = 0.5),
            panel.grid.minor = element_blank(),
            legend.position = "bottom",
            legend.title = element_text(size = base_size * 0.9, face = "bold"),
            legend.text = element_text(size = base_size * 0.85),
            plot.margin = margin(20, 20, 20, 20)
        )
}

# ------------------------------------------------------------------------
# Data Generating Process (DGP)
# ------------------------------------------------------------------------
# DGP3: High-Power Design (validated for R/Python consistency)
# - True effect: +20 units step change
# - x1: Stationary AR(1) with mean 55, NO TREND (avoids false positives)
# - Seasonality: Day-of-week pattern (symmetric around midweek)
# - Noise: Low (sd=2) for high statistical power
# ------------------------------------------------------------------------

set.seed(42) # Fixed seed for reproducibility

# Define the total number of days in this analysis
total_days <- 120

# Create a sequence of dates for the time series
dates <- seq.Date(from = as.Date("2024-01-01"), by = "day", length.out = total_days)

# Generate STATIONARY AR(1) for x1 - strong predictor, NO TREND
# x1_t = 55 + 0.75 * (x1_{t-1} - 55) + epsilon
# Note: x1 is kept stationary (no trend!) to serve as a valid covariate
# that is NOT affected by the intervention
x1 <- numeric(total_days)
x1[1] <- 55
for (t in 2:total_days) {
    x1[t] <- 55 + 0.75 * (x1[t - 1] - 55) + rnorm(1, sd = 5)
}

# Extract the day of the week for each date (to create day-of-week seasonality)
day_of_week <- weekdays(dates)

# Assign seasonal effects for the days of the week
# Symmetric pattern around midweek for clear weekly seasonality
day_of_week_effect <- case_when(
    day_of_week == "Monday" ~ -10,
    day_of_week == "Tuesday" ~ -5,
    day_of_week == "Wednesday" ~ 0,
    day_of_week == "Thursday" ~ 0,
    day_of_week == "Friday" ~ 10,
    day_of_week == "Saturday" ~ 20,
    day_of_week == "Sunday" ~ 15,
    TRUE ~ 0
)

# Baseline values for y before intervention
# Note: y depends on x1 (strong relationship: 0.6 coefficient) and day-of-week seasonality
# Low noise (sd=2) ensures high power to detect the effect
y <- 100 + 0.6 * x1 + day_of_week_effect + rnorm(total_days, sd = 2)

# Define the intervention timepoint (day when the intervention occurs)
intervention_time <- 85

# TRUE EFFECT: Increase the response variable y by 20 units after the intervention
TRUE_EFFECT <- 20
y[dates >= dates[intervention_time]] <- y[dates >= dates[intervention_time]] + TRUE_EFFECT

# Build the dataset
data <- data.frame(
    date = as.Date(dates),
    y = as.numeric(y),
    x1 = as.numeric(x1),
    day_of_week = day_of_week
)

# Save the campaign data (uncomment to regenerate)
# write.csv(data, "data/time_series_campaign.csv", row.names = FALSE)

# Read the campaign data (ensures R and Python use identical data)
data <- read.csv("data/time_series_campaign.csv", colClasses = c(date = "Date"))

str(data)
cat("True effect:", TRUE_EFFECT, "units\n")
cat("Intervention at day:", intervention_time, "\n")

# ------------------------------------------------------------------------
# Main result: Causal ARIMA
# ------------------------------------------------------------------------

# Create time series object with weekly seasonality
y_ts <- ts(data$y, frequency = 7)

# Define pre and post periods
pre_period <- 1:(intervention_time - 1) # Pre-Period: Days 1 to 84 (before the intervention).
post_period <- intervention_time:length(y_ts) # Post-Period: Days 85 to 120 (after the intervention).

# Set up parameters for CausalArima
intervention_date <- data[min(post_period), "date"] # ->  "2024-03-15"
all_dates <- data$date

# Fit the CausalArima model
ce <- CausalArima(
    y = y_ts,
    dates = all_dates,
    int.date = intervention_date,
    nboot = 1000 # Bootstrap iterations for confidence intervals
)

# Display results
summary_model_CA <- impact(ce)
print(summary_model_CA$arima) # ARIMA model details
print(summary_model_CA$impact_norm) # Normalized impact

# Visualize the counterfactual forecast and save to file
forecasted <- plot(ce, type = "forecast") + theme(legend.position = "bottom")
ggsave("images/carima_forecast_plot.png", plot = forecasted, width = 10, height = 6, dpi = 300)

# Plot impact and cumulative impact
impact_p <- plot(ce, type = "impact")
combined_plot <- grid.arrange(impact_p$plot, impact_p$cumulative_plot)
ggsave("images/impact_combined_plot.png", plot = combined_plot, width = 10, height = 6)

# ------------------------------------------------------------------------
# Robustness test 1 CausalArima: Placebo test (no pre-intervention effects)
# ------------------------------------------------------------------------
# WHY THIS TEST MATTERS:
# CausalArima estimates the causal effect by comparing actual post-intervention
# values to a counterfactual forecast. But how do we know the model isn't just
# picking up noise, pre-existing trends, or seasonal patterns?
#
# The placebo test answers: "If we pretend the intervention happened earlier
# (when it didn't), does the model falsely detect an effect?"
#
# If the model finds a significant effect at a fake intervention date, it means:
# - The model may be overfitting to noise
# - There might be unmodeled seasonality or trends
# - We can't trust the main result
#
# Expected result: NO significant effect at the placebo date.

# Placebo test: pretend intervention happened at day 60 (well before actual intervention at 85)
# Note: Day 60 chosen to create a 20-day post-period within pre-intervention data
placebo_intervention_time <- 60
data_placebo <- data[1:80, ] # Placebo dataset from periods 1 to 80

# Create placebo time series
y_ts_placebo <- ts(data_placebo$y, frequency = 7)

# Define placebo periods
pre_period_placebo <- 1:(placebo_intervention_time - 1)
post_period_placebo <- placebo_intervention_time:length(y_ts_placebo)

# Run placebo analysis
intervention_date_placebo <- data_placebo[min(post_period_placebo), "date"]
all_dates_placebo <- data_placebo$date

ce_placebo <- CausalArima(
    y = y_ts_placebo,
    dates = all_dates_placebo,
    int.date = intervention_date_placebo,
    nboot = 1000
)

# Check results - should show no significant effect
summary_placebo <- impact(ce_placebo)
print(summary_placebo$impact_norm)

# Visualize the placebo counterfactual forecast
forecasted_placebo <- plot(ce_placebo, type = "forecast")
print(forecasted_placebo)

# Plot placebo impact
impact_placebo_CA <- plot(ce_placebo, type = "impact")
combined_placebo_plot <- grid.arrange(impact_placebo_CA$plot, impact_placebo_CA$cumulative_plot)
ggsave("images/impact_combined_placebo_plot.png", plot = combined_placebo_plot, width = 10, height = 6)


# ------------------------------------------------------------------------
# Robustness test 2 CausalArima: Residual diagnostics
# ------------------------------------------------------------------------
# WHY THIS TEST MATTERS:
# CausalArima builds the counterfactual by forecasting y using an ARIMA model
# fitted on pre-intervention data. If the ARIMA model is misspecified (wrong
# order, missing seasonality, structural breaks), the forecast will be biased
# — and so will our causal estimate.
#
# What we check:
# 1. No residual autocorrelation (Ljung-Box test) — if residuals are correlated,
#    the model missed some predictable structure
# 2. Residuals look like white noise (ACF/PACF plots)
# 3. Approximate normality — needed for valid confidence intervals

# Extract the fitted ARIMA model from the CausalArima object
arima_model <- ce$model

# 1. Ljung-Box test for autocorrelation in residuals
# H0: residuals are independently distributed (no autocorrelation)
# We want a HIGH p-value (> 0.05) to fail to reject H0
# Interpretation: p-value > 0.05 -> no significant autocorrelation (good!)
#                 p-value < 0.05 -> WARNING: residuals show autocorrelation,
#                                   model may be misspecified
ljung_box <- Box.test(residuals(arima_model), lag = 10, type = "Ljung-Box")
print(ljung_box)

# 2. Visual check: ACF and PACF of residuals
# Good model: no significant spikes beyond lag 0
png("images/residual_acf_pacf.png", width = 12, height = 4, units = "in", res = 300)
par(mfrow = c(1, 2))
acf(residuals(arima_model), main = "ACF of Residuals")
pacf(residuals(arima_model), main = "PACF of Residuals")
par(mfrow = c(1, 1))
dev.off()

# 3. Normality check (Shapiro-Wilk test)
# H0: residuals are normally distributed
# We want a HIGH p-value (> 0.05) to fail to reject H0
# Interpretation: p-value > 0.05 -> residuals appear normally distributed (good!)
#                 p-value < 0.05 -> WARNING: residuals may not be normal,
#                                   confidence intervals may be unreliable
shapiro_test <- shapiro.test(residuals(arima_model))
print(shapiro_test)

# 4. Residual plot over time (check for patterns)
plot(residuals(arima_model),
    main = "Residuals Over Time",
    ylab = "Residuals", xlab = "Time"
)
abline(h = 0, col = book_colors$muted, lty = 2)


# ------------------------------------------------------------------------
# Main result: CausalImpact
# ------------------------------------------------------------------------

# Define pre-intervention and post-intervention periods
# Note: total_days already set to 120 in DGP section

pre_period_ci <- c(1, intervention_time - 1) # Pre-Period: Days 1 to 84
post_period_ci <- c(intervention_time, total_days) # Post-Period: Days 85 to 120

# Run the CausalImpact analysis with seasonal components for day-of-week seasonality
impact_ci <- CausalImpact(
    data = cbind(y = data$y, x1 = data$x1),
    pre.period = pre_period_ci,
    post.period = post_period_ci,
    model.args = list(
        niter = 5000, # number of MCMC draws
        nseasons = 7, # Capture weekly seasonality
        season.duration = 1 # Seasonality repeats every 7 days
    )
)

# Display a summary of the results
summary(impact_ci)
summary(impact_ci, "report") # descriptive report

# Plot the impact and save to file
png("images/causalimpact_main_plot.png", width = 10, height = 8, units = "in", res = 300)
plot(impact_ci)
dev.off()

# Point estimates and credible intervals
summary(impact_ci, "summary")

# ------------------------------------------------------------------------
# Robustness test 1 CausalImpact: Placebo test (no pre-intervention effects)
# ------------------------------------------------------------------------
# WHY THIS TEST MATTERS:
# CausalImpact uses a Bayesian structural time series model with covariates
# to predict what y would have been without the intervention. Like CausalArima,
# we need to verify the model isn't detecting spurious effects.
#
# The placebo test asks: "If we pretend the intervention happened earlier
# (when it didn't), does CausalImpact falsely detect an effect?"
#
# A significant effect at a fake intervention date would suggest:
# - The model is picking up noise or unmodeled trends
# - The covariate relationship may be unstable
# - We should be cautious about the main result
#
# Expected result: NO significant effect at the placebo date.

# Placebo test using data up to day 80, pretend intervention at day 60
data_placebo_ci <- data[1:80, ]
# placebo_intervention_time already set to 60 above
pre_period_placebo_ci <- c(1, placebo_intervention_time - 1) # Pre-Period: Days 1 to 59
post_period_placebo_ci <- c(placebo_intervention_time, 80) # Post-Period: Days 60 to 80

impact_placebo_ci <- CausalImpact(
    data = cbind(y = data_placebo_ci$y, x1 = data_placebo_ci$x1),
    pre.period = pre_period_placebo_ci,
    post.period = post_period_placebo_ci,
    model.args = list(
        niter = 5000,
        nseasons = 7,
        season.duration = 1
    )
)

# Should show no significant effect
summary(impact_placebo_ci, "report")

# Plot and save placebo results
png("images/causalimpact_placebo_plot.png", width = 10, height = 8, units = "in", res = 300)
plot(impact_placebo_ci)
dev.off()

# ------------------------------------------------------------------------
# Robustness test 2 CausalImpact: No effect on auxiliary series
# ------------------------------------------------------------------------
# WHY THIS TEST MATTERS:
# CausalImpact uses auxiliary covariates (here, x1) to predict what y would
# have been without the intervention. This synthetic control approach assumes
# that the covariates themselves are NOT affected by the treatment.
#
# If x1 was contaminated by the intervention (e.g., a marketing campaign that
# also changed competitor behavior), then the counterfactual is biased — we'd
# be using post-treatment x1 values that already "baked in" the intervention.
#
# This test checks: Does x1 show a structural break at the intervention date?
# We use CausalArima (which needs no covariates) to test for a break in x1.
#
# Expected result: NO significant effect on x1.
#
# NOTE: This test is specific to CausalImpact because it relies on covariates.
# CausalArima doesn't use covariates, so this test doesn't apply to it.

ce_aux <- CausalArima(
    y = ts(data$x1, frequency = 7),
    dates = all_dates,
    int.date = intervention_date,
    nboot = 1000
)

# Visualize: the forecast should track actual x1 closely in post-period
forecasted_aux <- plot(ce_aux, type = "forecast")
print(forecasted_aux)

# Check impact estimate — should be near zero and not statistically significant
summary_aux <- impact(ce_aux)
print(summary_aux$impact_norm)
# Interpretation: If the confidence interval includes zero, x1 was likely
# unaffected by the intervention (good!). If significant, CausalImpact
# results may be biased.

# ------------------------------------------------------------------------
# Sensitivity analysis CausalImpact: Estimate without auxiliary series
# ------------------------------------------------------------------------
# WHY THIS TEST MATTERS:
# CausalImpact can run with or without covariates. When we include x1, we're
# assuming it helps predict y and improves counterfactual accuracy.
#
# This sensitivity check asks: "How much does our estimate depend on x1?"
#
# If results are similar with and without x1:
# - The estimate is robust to covariate choice
# - x1 may not be adding much predictive power
#
# If results differ substantially:
# - The estimate is sensitive to the covariate
# - We should carefully validate that x1 is a good predictor and unaffected
#
# Neither outcome is inherently "bad" — it's about understanding what drives
# your estimate and how confident you should be in it.

# Run CausalImpact without auxiliary series
impact_no_aux <- CausalImpact(
    data = data$y,
    pre.period = pre_period_ci,
    post.period = post_period_ci,
    model.args = list(niter = 5000, nseasons = 7, season.duration = 1)
)

# Compare results
print("With auxiliary series:")
print(summary(impact_ci)$AbsEffect)
print("Without auxiliary series:")
print(summary(impact_no_aux)$AbsEffect)

# ------------------------------------------------------------------------
# Appendix: C-ARIMA with external regressors (xreg)
# ------------------------------------------------------------------------
# WHY THIS MATTERS:
# C-ARIMA can also incorporate auxiliary series via the xreg argument.
# This lets ARIMA use x1 as an external predictor while retaining its
# frequentist bootstrap inference framework.
#
# Comparing: C-ARIMA (no X) vs C-ARIMA (with X) vs CausalImpact (with X)
# helps us understand how much auxiliary data improves each method.

# C-ARIMA WITHOUT external regressors (baseline — same as main result)
ce_no_x <- CausalArima(
    y = y_ts,
    dates = all_dates,
    int.date = intervention_date,
    nboot = 1000
)
result_no_x <- impact(ce_no_x)

# C-ARIMA WITH external regressors
# Pass x1 as a matrix via the xreg argument.
# The function uses x1 in the pre-period to learn the relationship,
# then uses post-period x1 values for counterfactual forecasting.
ce_with_x <- CausalArima(
    y = y_ts,
    dates = all_dates,
    int.date = intervention_date,
    xreg = as.matrix(data$x1),
    nboot = 1000
)
result_with_x <- impact(ce_with_x)

# Compare estimates
print("C-ARIMA without X:")
print(result_no_x$impact_norm)
print("C-ARIMA with X (x1):")
print(result_with_x$impact_norm)

# ------------------------------------------------------------------------
# Descriptive plots
# ------------------------------------------------------------------------

# Plot the y time series over time
ggplot(data, aes(x = date, y = y)) +
    geom_line(color = book_colors$primary, linewidth = 1.2) +
    geom_vline(xintercept = data$date[intervention_time], linetype = "dashed", color = book_colors$accent, linewidth = 1) +
    labs(
        title = paste0("Time series y with intervention effect from day ", intervention_time, " onwards"),
        x = "Day",
        y = "y = Daily sales in R$ MM"
    ) +
    theme_book() +
    theme(axis.title = element_text(size = rel(1.3)))

ggsave("images/time_series_y.png", width = 12, height = 8, dpi = 300)

# Plot the auxiliary time series x1 over time
ggplot(data, aes(x = date, y = x1)) +
    geom_line(color = book_colors$secondary, linewidth = 1.2) +
    geom_vline(xintercept = data$date[intervention_time], linetype = "dashed", color = book_colors$accent, linewidth = 1) +
    labs(
        title = expression("Auxiliary time series " ~ X[1] ~ " unaffected by the intervention"),
        x = "Day",
        y = expression(X[1])
    ) +
    theme_book() +
    theme(axis.title = element_text(size = rel(1.3)))

ggsave("images/time_series_x1.png", width = 12, height = 8, dpi = 300)

# Calculate average y by day of week, to show seasonality
avg_y_by_day <- data %>%
    group_by(day_of_week) %>%
    summarize(avg_y = mean(y))

avg_y_by_day$day_of_week <- factor(avg_y_by_day$day_of_week, levels = c(
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
))

# Plot average y by day of week
ggplot(avg_y_by_day, aes(x = day_of_week, y = avg_y)) +
    geom_bar(stat = "identity", fill = book_colors$primary, alpha = 0.8) +
    geom_text(
        aes(label = paste0("R$ ", scales::number(avg_y, accuracy = 1), " MM")),
        vjust = -0.25,
        size = 5,
        color = book_colors$dark_gray
    ) +
    labs(
        title = "Average daily sales in R$ MM, by day of the week",
        x = "Day of the week",
        y = "Average y, daily sales in R$ MM"
    ) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.1))) +
    theme_book() +
    theme(
        axis.text.x = element_text(angle = 25, hjust = 1)
    )

ggsave("images/day_of_week_avg.png", width = 10, height = 8, dpi = 300)
