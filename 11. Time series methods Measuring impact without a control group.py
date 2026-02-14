##########################################################
## Everyday Causal Inference: How to estimate, test, and explain impacts with R and Python
## www.everydaycausal.com
## Copyright © 2025 by Robson Tigre. All rights reserved.
## You may read, share, and cite for learning purposes, provided you credit the source.
## It should not be used to create competing educational or commercial products
##########################################################
## Code for 11. Time series methods: Measuring impact without a control group
## Created: Jan 29, 2026
## Last modified: Jan 29, 2026
##########################################################

# Setup -------------------------------------------------------------------
# Install packages if needed:
# pip install git+https://github.com/RobsonTigre/pycausalarima.git
# pip install causalimpact

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for script execution
import matplotlib.pyplot as plt
from pycausalarima import CausalArima
from causalimpact import CausalImpact
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import shapiro
import warnings
warnings.filterwarnings('ignore')  # Suppress convergence warnings for cleaner output

# ------------------------------------------------------------------------
# Book-wide Theme and Color Palette
# ------------------------------------------------------------------------

# Define a consistent color palette (colorblind-friendly)
BOOK_COLORS = {
    "primary": "#2E86AB",    # Steel blue - main data
    "secondary": "#A23B72",  # Magenta - secondary data
    "accent": "#F18F01",     # Orange - highlights/warnings
    "success": "#C73E1D",    # Red-orange - thresholds/targets
    "muted": "#6C757D",      # Gray - reference lines
    "light_gray": "#E5E5E5", # Light gray - backgrounds (grey90 equivalent)
    "dark_gray": "#4D4D4D",  # Dark gray - text (grey30 equivalent)
}

def apply_book_style(ax, base_size=14):
    """Apply consistent book styling to a matplotlib axes."""
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(BOOK_COLORS["muted"])
    ax.spines['bottom'].set_color(BOOK_COLORS["muted"])

    # Set tick and label styling
    ax.tick_params(colors=BOOK_COLORS["dark_gray"], labelsize=base_size * 0.85)
    ax.xaxis.label.set_color(BOOK_COLORS["dark_gray"])
    ax.yaxis.label.set_color(BOOK_COLORS["dark_gray"])
    ax.xaxis.label.set_fontsize(base_size)
    ax.yaxis.label.set_fontsize(base_size)

    # Title styling
    ax.title.set_fontsize(base_size * 1.3)
    ax.title.set_fontweight("bold")
    ax.title.set_color("#333333")  # grey20 equivalent

# ------------------------------------------------------------------------
# Data Generating Process (DGP)
# ------------------------------------------------------------------------
# DGP3: High-Power Design (validated for R/Python consistency)
# - True effect: +20 units step change
# - x1: Stationary AR(1) with mean 55, NO TREND (avoids false positives)
# - Seasonality: Day-of-week pattern (symmetric around midweek)
# - Noise: Low (sd=2) for high statistical power
# ------------------------------------------------------------------------

np.random.seed(42)  # Fixed seed for reproducibility

# Define the total number of days in this analysis
total_days = 120

# Create a sequence of dates for the time series
dates = pd.date_range(start="2024-01-01", periods=total_days, freq="D")

# Generate STATIONARY AR(1) for x1 - strong predictor, NO TREND
# x1_t = 55 + 0.75 * (x1_{t-1} - 55) + epsilon
# Note: x1 is kept stationary (no trend!) to serve as a valid covariate
# that is NOT affected by the intervention
x1 = np.zeros(total_days)
x1[0] = 55
for t in range(1, total_days):
    x1[t] = 55 + 0.75 * (x1[t-1] - 55) + np.random.normal(0, 5)

# Extract the day of the week for each date (to create day-of-week seasonality)
day_of_week = dates.day_name()

# Assign seasonal effects for the days of the week
# Symmetric pattern around midweek for clear weekly seasonality
day_effects = {
    "Monday": -10, "Tuesday": -5, "Wednesday": 0, "Thursday": 0,
    "Friday": 10, "Saturday": 20, "Sunday": 15
}
day_of_week_effect = np.array([day_effects[d] for d in day_of_week])

# Baseline values for y before intervention
# Note: y depends on x1 (strong relationship: 0.6 coefficient) and day-of-week seasonality
# Low noise (sd=2) ensures high power to detect the effect
y = 100 + 0.6 * x1 + day_of_week_effect + np.random.normal(0, 2, total_days)

# Define the intervention timepoint (day when the intervention occurs)
intervention_time = 85

# TRUE EFFECT: Increase the response variable y by 20 units after the intervention
TRUE_EFFECT = 20
y[intervention_time - 1:] += TRUE_EFFECT

# Build the dataset
data = pd.DataFrame({
    "date": dates,
    "y": y,
    "x1": x1,
    "day_of_week": day_of_week
})

# Save the campaign data (uncomment to regenerate)
# data.to_csv("data/time_series_campaign.csv", index=False)

# Read the campaign data (ensures R and Python use identical data)
data = pd.read_csv("data/time_series_campaign.csv", parse_dates=["date"])
print(f"True effect: {TRUE_EFFECT} units")
print(f"Intervention at day: {intervention_time}")

# ------------------------------------------------------------------------
# Main result: Causal ARIMA
# ------------------------------------------------------------------------

# Set up parameters for CausalArima
intervention_date = data["date"].iloc[intervention_time - 1]  # -> "2024-03-15"
all_dates = data["date"]

# Fit the CausalArima model
# Note: seasonal_order=(0,0,0,7) enables weekly seasonality detection with auto ARIMA
# This is analogous to R's ts(data$y, frequency = 7)
ca = CausalArima(
    y=data["y"].values,
    dates=all_dates,
    intervention_date=intervention_date,
    seasonal_order=(0, 0, 0, 7),  # Weekly seasonality
    n_boot=1000  # Bootstrap iterations for confidence intervals
)
result = ca.fit()

# Display results
# ca.summary() returns normalized impact (analogous to impact(ce)$impact_norm in R)
summary_ca = ca.summary()

# Visualize the counterfactual forecast and impact plots
# Note: Using try/except to handle potential plotting issues in the library
try:
    forecast_fig = ca.plot(type="forecast")
    forecast_fig.savefig("images/causal_arima_forecast_py.png", dpi=300, bbox_inches="tight")
    plt.close(forecast_fig)
    # Saved: images/causal_arima_forecast_py.png
except Exception as e:
    pass  # Forecast plot could not be generated, using fallback
    # Fallback: create manual forecast plot using result data
    res = ca.result_
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(res.dates, res.y, color=BOOK_COLORS["primary"], linewidth=1.5, label="Observed")
    ax.plot(res.dates[res.n_pre:], res.forecast, color=BOOK_COLORS["accent"], linestyle="--", linewidth=1.5, label="Counterfactual")
    ax.axvline(x=res.dates[res.n_pre], color=BOOK_COLORS["accent"], linestyle=":", linewidth=1.5)
    ax.set_xlabel("Date", fontsize=14)
    ax.set_ylabel("y = Daily sales in R$ MM", fontsize=14)
    ax.set_title("Causal ARIMA: Observed vs Counterfactual Forecast", fontsize=16, fontweight="bold")
    ax.legend()
    apply_book_style(ax)
    plt.tight_layout()
    plt.savefig("images/causal_arima_forecast_py.png", dpi=300, bbox_inches="tight")
    plt.close()
    # Saved (fallback): images/causal_arima_forecast_py.png

try:
    # Plot impact and cumulative impact
    # Note: plot(type="impact") returns dict with "plot" and "cumulative_plot" keys
    # This matches R's plot(ce, type="impact") which returns $plot and $cumulative_plot
    impact_plots = ca.plot(type="impact")

    # Save individual plots
    impact_plots["plot"].savefig("images/impact_plot_py.png", dpi=300, bbox_inches="tight")
    plt.close(impact_plots["plot"])

    impact_plots["cumulative_plot"].savefig("images/cumulative_impact_plot_py.png", dpi=300, bbox_inches="tight")
    plt.close(impact_plots["cumulative_plot"])
    # Saved: images/impact_plot_py.png, images/cumulative_impact_plot_py.png
except Exception as e:
    pass  # Impact plots could not be generated, using fallback
    # Fallback: create manual impact plots
    import numpy as np
    res = ca.result_
    dates_post = res.dates[res.n_pre:]
    impact = res.causal_effect
    cumulative_impact = np.cumsum(impact)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(dates_post, impact, color=BOOK_COLORS["primary"], linewidth=1.5)
    axes[0].axhline(y=0, color=BOOK_COLORS["muted"], linestyle="--", linewidth=1)
    axes[0].fill_between(dates_post, 0, impact, alpha=0.3, color=BOOK_COLORS["primary"])
    axes[0].set_ylabel("Point impact", fontsize=12)
    axes[0].set_title("Impact and Cumulative Impact", fontsize=14, fontweight="bold")

    axes[1].plot(dates_post, cumulative_impact, color=BOOK_COLORS["secondary"], linewidth=1.5)
    axes[1].axhline(y=0, color=BOOK_COLORS["muted"], linestyle="--", linewidth=1)
    axes[1].fill_between(dates_post, 0, cumulative_impact, alpha=0.3, color=BOOK_COLORS["secondary"])
    axes[1].set_xlabel("Date", fontsize=12)
    axes[1].set_ylabel("Cumulative impact", fontsize=12)

    plt.tight_layout()
    plt.savefig("images/impact_combined_plot_py.png", dpi=300, bbox_inches="tight")
    plt.close()
    # Saved (fallback): images/impact_combined_plot_py.png

# ------------------------------------------------------------------------
# Robustness test Causal ARIMA: no pre-intervention effects
# ------------------------------------------------------------------------

# Placebo test: pretend intervention happened at day 60 (well before actual intervention at 85)
# Note: Day 60 chosen to create a 20-day post-period within pre-intervention data
placebo_intervention_time = 60
data_placebo = data.iloc[:80].copy()  # Placebo dataset from periods 1 to 80

# Set up placebo parameters
intervention_date_placebo = data_placebo["date"].iloc[placebo_intervention_time - 1]
all_dates_placebo = data_placebo["date"]

# Run placebo analysis
ca_placebo = CausalArima(
    y=data_placebo["y"].values,
    dates=all_dates_placebo,
    intervention_date=intervention_date_placebo,
    seasonal_order=(0, 0, 0, 7),
    n_boot=1000
)
ca_placebo.fit()

# Check results - should show no significant effect
summary_placebo = ca_placebo.summary()

# Visualize the placebo counterfactual forecast
try:
    forecast_placebo_fig = ca_placebo.plot(type="forecast")
    forecast_placebo_fig.savefig("images/causal_arima_placebo_forecast_py.png", dpi=300, bbox_inches="tight")
    plt.close(forecast_placebo_fig)
    # Saved: images/causal_arima_placebo_forecast_py.png
except Exception as e:
    pass  # Placebo forecast plot could not be generated

# Plot placebo impact
try:
    impact_placebo_plots = ca_placebo.plot(type="impact")
    impact_placebo_plots["plot"].savefig("images/impact_placebo_plot_py.png", dpi=300, bbox_inches="tight")
    plt.close(impact_placebo_plots["plot"])
    impact_placebo_plots["cumulative_plot"].savefig("images/cumulative_impact_placebo_plot_py.png", dpi=300, bbox_inches="tight")
    plt.close(impact_placebo_plots["cumulative_plot"])
    # Saved: images/impact_placebo_plot_py.png, images/cumulative_impact_placebo_plot_py.png
except Exception as e:
    pass  # Placebo impact plots could not be generated

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

# Extract standardized residuals (Kalman filter innovations) from the fitted model
# Note: For state-space models like SARIMAX, the standardized forecast errors
# are the proper residuals for diagnostic tests. These are accessed via the
# underlying statsmodels results object (arima_res_).
residuals_arima = ca.model_.arima_res_.standardized_forecasts_error[0]

# 1. Ljung-Box test for autocorrelation in residuals
# H0: residuals are independently distributed (no autocorrelation)
# We want a HIGH p-value (> 0.05) to fail to reject H0
# Interpretation: p-value > 0.05 -> no significant autocorrelation (good!)
#                 p-value < 0.05 -> WARNING: residuals show autocorrelation,
#                                   model may be misspecified
ljung_box_results = acorr_ljungbox(residuals_arima, lags=[10], return_df=True)
# Results stored in ljung_box_results DataFrame with columns: lb_stat, lb_pvalue

# 2. Visual check: ACF and PACF of residuals
# Good model: no significant spikes beyond lag 0
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(residuals_arima, ax=axes[0], title="ACF of Residuals")
plot_pacf(residuals_arima, ax=axes[1], title="PACF of Residuals")
plt.tight_layout()
plt.savefig("images/residual_acf_pacf_py.png", dpi=300, bbox_inches="tight")
plt.close()

# 3. Normality check (Shapiro-Wilk test)
# H0: residuals are normally distributed
# We want a HIGH p-value (> 0.05) to fail to reject H0
# Interpretation: p-value > 0.05 -> residuals appear normally distributed (good!)
#                 p-value < 0.05 -> WARNING: residuals may not be normal,
#                                   confidence intervals may be unreliable
shapiro_stat, shapiro_pvalue = shapiro(residuals_arima)
# Results: shapiro_stat = test statistic, shapiro_pvalue = p-value

# 4. Residual plot over time (check for patterns)
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(residuals_arima, color=BOOK_COLORS["primary"], linewidth=1)
ax.axhline(y=0, color=BOOK_COLORS["muted"], linestyle="--", linewidth=1)
ax.set_title("Residuals Over Time", fontsize=14, fontweight="bold")
ax.set_xlabel("Time")
ax.set_ylabel("Residuals")
apply_book_style(ax)
plt.tight_layout()
plt.savefig("images/residuals_over_time_py.png", dpi=300, bbox_inches="tight")
plt.close()

# ------------------------------------------------------------------------
# Main result: CausalImpact
# ------------------------------------------------------------------------

# Define pre-intervention and post-intervention periods
# Note: total_days already set to 120 in DGP section
pre_period_ci = [0, intervention_time - 2]  # Pre-Period: Days 0 to 83 (1 to 84 in 1-based)
post_period_ci = [intervention_time - 1, total_days - 1]  # Post-Period: Days 84 to 119 (85 to 120 in 1-based)

# Prepare data for CausalImpact (outcome y and auxiliary series x1)
ci_data = data[["y", "x1"]].copy()
ci_data.index = pd.RangeIndex(start=0, stop=len(ci_data))

# Run the CausalImpact analysis
impact_ci = CausalImpact(ci_data, pre_period_ci, post_period_ci)

# Display a summary of the results
summary_ci = impact_ci.summary()
report_ci = impact_ci.summary(output="report")

# Plot the impact
impact_ci.plot()
plt.savefig("images/causalimpact_main_py.png", dpi=300, bbox_inches="tight")
plt.close()

# ------------------------------------------------------------------------
# Robustness test 1 CausalImpact: no pre-intervention effects
# ------------------------------------------------------------------------

# Placebo test using data up to day 80, pretend intervention at day 60
ci_data_placebo = data.iloc[:80][["y", "x1"]].copy()
ci_data_placebo.index = pd.RangeIndex(start=0, stop=len(ci_data_placebo))

pre_period_placebo_ci = [0, placebo_intervention_time - 2]  # Pre-Period: Days 0 to 58 (1 to 59 in 1-based)
post_period_placebo_ci = [placebo_intervention_time - 1, 79]  # Post-Period: Days 59 to 79 (60 to 80 in 1-based)

impact_placebo_ci = CausalImpact(ci_data_placebo, pre_period_placebo_ci, post_period_placebo_ci)

# Should show no significant effect
report_placebo_ci = impact_placebo_ci.summary(output="report")

impact_placebo_ci.plot()
plt.savefig("images/causalimpact_placebo_py.png", dpi=300, bbox_inches="tight")
plt.close()

# ------------------------------------------------------------------------
# Robustness test 2 CausalImpact: no effect in the auxiliary series
# ------------------------------------------------------------------------

# Test if the auxiliary series x1 was affected by the intervention
# (it should NOT be affected for valid inference)
ca_aux = CausalArima(
    y=data["x1"].values,
    dates=all_dates,
    intervention_date=intervention_date,
    seasonal_order=(0, 0, 0, 7),
    n_boot=1000
)
ca_aux.fit()

# Visualize auxiliary series test
try:
    forecast_aux_fig = ca_aux.plot(type="forecast")
    forecast_aux_fig.savefig("images/auxiliary_series_test_py.png", dpi=300, bbox_inches="tight")
    plt.close(forecast_aux_fig)
    # Saved: images/auxiliary_series_test_py.png
except Exception as e:
    pass  # Auxiliary series plot could not be generated

# Check results - should show no significant effect on x1
summary_aux = ca_aux.summary()

# Sensitivity analysis: Run CausalImpact without auxiliary series
ci_data_no_aux = data[["y"]].copy()
ci_data_no_aux.index = pd.RangeIndex(start=0, stop=len(ci_data_no_aux))

impact_no_aux = CausalImpact(ci_data_no_aux, pre_period_ci, post_period_ci)

# Compare results: with vs without auxiliary series
summary_with_aux = impact_ci.summary()
summary_without_aux = impact_no_aux.summary()

# ------------------------------------------------------------------------
# Appendix: C-ARIMA with external regressors (xreg)
# ------------------------------------------------------------------------
# C-ARIMA can also incorporate auxiliary series via the xreg argument.
# This lets ARIMA use x1 as an external predictor while retaining its
# frequentist bootstrap inference framework.
#
# Comparing: C-ARIMA (no X) vs C-ARIMA (with X) vs CausalImpact (with X)
# helps us understand how much auxiliary data improves each method.

# C-ARIMA WITHOUT external regressors (baseline — same as main result)
ca_no_x = CausalArima(
    y=data["y"].values,
    dates=data["date"],
    intervention_date=intervention_date,
    seasonal_order=(0, 0, 0, 7),
    n_boot=1000
)
ca_no_x.fit()

# C-ARIMA WITH external regressors
# Pass x1 as a 2D array via the xreg argument.
# The function uses x1 in the pre-period to learn the relationship,
# then uses post-period x1 values for counterfactual forecasting.
ca_with_x = CausalArima(
    y=data["y"].values,
    dates=data["date"],
    intervention_date=intervention_date,
    xreg=data[["x1"]].values,  # 2D array, even for one regressor
    seasonal_order=(0, 0, 0, 7),
    n_boot=1000
)
ca_with_x.fit()

# Compare estimates
print("C-ARIMA without X:")
ca_no_x.summary()
print("C-ARIMA with X (x1):")
ca_with_x.summary()

# ------------------------------------------------------------------------
# Descriptive plots
# ------------------------------------------------------------------------

# Plot the y time series over time
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(data["date"], data["y"], color=BOOK_COLORS["primary"], linewidth=1.2)
ax.axvline(x=data["date"].iloc[intervention_time - 1], linestyle="--", color=BOOK_COLORS["accent"], linewidth=1)
ax.set_xlabel("Day")
ax.set_ylabel("y = Daily sales in R$ MM")
ax.set_title(f"Time series y with intervention effect from day {intervention_time} onwards")
apply_book_style(ax)
plt.tight_layout()
plt.savefig("images/time_series_y_py.png", dpi=300, bbox_inches="tight")
plt.close()

# Plot the auxiliary time series x1 over time
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(data["date"], data["x1"], color=BOOK_COLORS["secondary"], linewidth=1.2)
ax.axvline(x=data["date"].iloc[intervention_time - 1], linestyle="--", color=BOOK_COLORS["accent"], linewidth=1)
ax.set_xlabel("Day")
ax.set_ylabel(r"$X_1$")
ax.set_title(r"Auxiliary time series $X_1$ unaffected by the intervention")
apply_book_style(ax)
plt.tight_layout()
plt.savefig("images/time_series_x1_py.png", dpi=300, bbox_inches="tight")
plt.close()

# Calculate average y by day of week, to show seasonality
avg_y_by_day = data.groupby("day_of_week")["y"].mean().reset_index()
avg_y_by_day.columns = ["day_of_week", "avg_y"]

# Order days of week
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
avg_y_by_day["day_of_week"] = pd.Categorical(avg_y_by_day["day_of_week"], categories=day_order, ordered=True)
avg_y_by_day = avg_y_by_day.sort_values("day_of_week")

# Plot average y by day of week
fig, ax = plt.subplots(figsize=(10, 8))
bars = ax.bar(avg_y_by_day["day_of_week"], avg_y_by_day["avg_y"], color=BOOK_COLORS["primary"], alpha=0.8)

# Add value labels on bars
for bar, val in zip(bars, avg_y_by_day["avg_y"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"R$ {val:.0f} MM", ha="center", va="bottom", fontsize=12, color=BOOK_COLORS["dark_gray"])

ax.set_xlabel("Day of the week")
ax.set_ylabel("Average y, daily sales in R$ MM")
ax.set_title("Average daily sales in R$ MM, by day of the week")
apply_book_style(ax)
ax.tick_params(axis="x", rotation=25)
ax.set_ylim(0, avg_y_by_day["avg_y"].max() * 1.15)
plt.tight_layout()
plt.savefig("images/day_of_week_avg_py.png", dpi=300, bbox_inches="tight")
plt.close()
