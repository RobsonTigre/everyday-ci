##########################################################
## Everyday Causal Inference: How to estimate, test, and explain impacts with R and Python
## www.everydaycausal.com
## Copyright © 2025 by Robson Tigre. All rights reserved.
## You may read, share, and cite for learning purposes, provided you credit the source.
## It should not be used to create competing educational or commercial products
##########################################################
## Code for Chapter 8 - Regression discontinuity design: Cutoffs as natural experiments
## Created: Jan 13, 2026
## Last modified: Jan 22, 2026
##########################################################

# -----------------------------------------------------------------------------
# 1. Setup and Packages
# -----------------------------------------------------------------------------
# pip install numpy pandas matplotlib rdrobust rddensity statsmodels scipy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from rdrobust import rdrobust, rdplot
from rddensity import rddensity, rdplotdensity
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.stats import binom

# Set options to avoid scientific notation
np.set_printoptions(suppress=True)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

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

np.random.seed(2024)  # For reproducibility (matches appendix in regression-discontinuity.qmd)
n = 20000

# Step 1: The Confounder (U)
# U = Customer Engagement. We can't see this, but it drives everything.
U = np.random.normal(0, 1, n)

# Step 2: Observed Covariates
# We observe Age and History.
# They are correlated with U, but continuous at the cutoff.
# We'll use them later to show how they improve precision in RDD.

age = 35 + 2 * U + np.random.normal(0, 5, n)
age = np.round(np.maximum(18, age))

history = 300 + 80 * U + np.random.normal(0, 50, n)
history = np.maximum(0, history)

# Step 3: Running Variable (Loyalty Score)
# More engaged users (high U) naturally have higher loyalty scores.
x_star = 50 + 15 * U + np.random.normal(0, 8, n)
x = np.clip(x_star, 0, 100)

# Step 4: Treatment Assignment (Sharp RDD)
treatment = (x >= 50).astype(int)

# Step 5: Outcome (Future Spending)
# Here's where we mix the ingredients:
# 1. Base spending (intercept)
# 2. A CUBIC relationship with X (concave then convex). This non-linearity is tricky for OLS.
# 3. The True Treatment Effect (250).
# 4. The Bias: "60 * U". Since U is higher for treated units, this inflates differences.
# 5. Some noise.

y_noise = np.random.normal(0, 70, n)

y = (600 +
     0.003 * (x - 50)**3 +      # The non-linear "S-curve"
     250 * treatment +           # The signal we want to recover (Truth = 250)
     60 * U +                    # The selection bias
     0.4 * history +             # Covariate influence
     y_noise)

y = np.maximum(0, y)

# Step 6: Assemble and Save Data
data = pd.DataFrame({
    'x': x,
    'y': y,
    'treatment': treatment,
    'age': age,
    'history': history
})

# Save the data to CSV
# data.to_csv("/Users/robsontigre/Desktop/everyday-ci/data/sharp_rdd_data_example.csv", index=False)

# -----------------------------------------------------------------------------
# 3. Analysis: Recovering the Causal Effect
# -----------------------------------------------------------------------------
# Load pre-generated data (or use the data frame we just created):
data = pd.read_csv("/Users/robsontigre/Desktop/everyday-ci/data/sharp_rdd_data_example.csv")

print(data.describe())
print(data.info())

# 3.1 Validation Checks
# Before running RDD, we must verify our assumptions.

# CHECK 1: Manipulation
# Are people "schooling" the system to get just above 50?
# A jump in density at 50 would suggest manipulation.
# We look for a p-value > 0.05 (fail to reject null of continuity).

density_test = rddensity(X=data['x'], c=50)
print(density_test)
# Interpreting the output:
# If p > 0.05, we have evidence that the running variable is NOT manipulated at the cutoff.

rdplotdensity(density_test, data['x'],
              title="Density plotting for manipulation testing",
              xlabel="Loyalty score over the last 3 months",
              ylabel="Density")
plt.savefig("images/rdd_density_test_py.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# CHECK 2: Covariate Balance
# Pre-treatment characteristics (Age, History) should not jump at the cutoff.
# We treat them as outcomes in an RDD model to verify this.

# Age should be smooth at the cutoff (p > 0.05)
print("Covariate balance test for age:")
print(rdrobust(y=data['age'], x=data['x'], c=50))

# Purchase history should be smooth at the cutoff (p > 0.05)
print("Covariate balance test for history:")
print(rdrobust(y=data['history'], x=data['x'], c=50))

# 3.2 The Identification Challenge
# Our True Effect is 250.
# Let's see how different methods perform in recovering this.

# --- Naive Comparison ---
# Just regress Y on Treatment. Ignores all confounders.
# This will be heavily biased upward because treated units have higher U.
naive = smf.ols("y ~ treatment", data=data).fit()
print("Naive comparison:")
print(naive.summary())
# Expected: Coefficient >> 250 (selection bias inflates the estimate)

# --- OLS with Covariates ---
# Control for running variable and observed covariates.
# Still biased due to: (1) non-linear relationship with X, (2) unobserved U.
ols_with_covs = smf.ols("y ~ treatment + x + age + history", data=data).fit()
print("OLS with covariates:")
print(ols_with_covs.summary())
# Expected: Coefficient != 250 (functional form misspecification)

# ---  RDD (The Solution) ---
# We don't guess the functional form globally. We just compare the edges.
# Y = f(x) + tau*D + e ... but we estimate locally.

rdd_result = rdrobust(y=data['y'], x=data['x'], c=50)
print("RDD baseline result:")
print(rdd_result)

# The RDD coefficient should be much closer to 250.
# It handles the selection bias AND the non-linear functional form.

# --- RDD with Covariates ---
# We add Age and History.
# Since they are balanced, they shouldn't change the coefficient much.
# But they should soak up residual variance, shrinking the Standard Error.

covariates = data[['age', 'history']].values
rdd_result_cov = rdrobust(
    y=data['y'],
    x=data['x'],
    c=50,
    covs=covariates
)
print("RDD with covariates:")
print(rdd_result_cov)

# -----------------------------------------------------------------------------
# 3.3 Robustness Checks: Kernels and Polynomials
# -----------------------------------------------------------------------------
# These exercises verify that our main estimate is not sensitive to
# arbitrary modeling choices. A robust finding should hold across specifications.

# --- Different Kernels ---
# The kernel determines how we weight observations based on their distance
# from the cutoff. Observations closer to the cutoff get more weight.
# The shape of the weighting function shouldn't change our conclusions.

# Triangular kernel (default in rdrobust) - gives more weight to obs near cutoff
rdd_triangular = rdrobust(y=data['y'], x=data['x'], c=50, kernel='triangular')
print("RDD with triangular kernel:")
print(rdd_triangular)

# Uniform kernel - gives equal weight to all obs within bandwidth
rdd_uniform = rdrobust(y=data['y'], x=data['x'], c=50, kernel='uniform')
print("RDD with uniform kernel:")
print(rdd_uniform)

# --- Different Polynomial Orders ---
# The polynomial order determines how flexible we allow the relationship
# between the running variable and outcome to be on each side of the cutoff.
# Higher orders are more flexible but risk overfitting near boundaries.

# Quadratic (p = 2) - more flexible, allows for curvature
rdd_quadratic = rdrobust(y=data['y'], x=data['x'], c=50, p=2)
print("RDD with quadratic polynomial:")
print(rdd_quadratic)

# --- Extract Results for Table ---
# Helper function to extract key results from rdrobust output
def extract_rdd_results(rdd_obj, label):
    """Extract key results from rdrobust output for comparison table."""
    # Note: Python rdrobust returns DataFrames, access with .iloc
    coef = rdd_obj.coef.iloc[0, 0]       # Conventional estimate
    ci_l = rdd_obj.ci.iloc[2, 0]         # Robust CI lower
    ci_u = rdd_obj.ci.iloc[2, 1]         # Robust CI upper
    bw = rdd_obj.bws.iloc[0, 0]          # Bandwidth (left = right for symmetric)
    se = rdd_obj.se.iloc[2, 0]           # Robust SE
    pval = rdd_obj.pv.iloc[2, 0]         # Robust p-value

    return {
        'Specification': label,
        'Estimate': round(coef, 1),
        'CI_lower': round(ci_l, 1),
        'CI_upper': round(ci_u, 1),
        'Bandwidth': round(bw, 2),
        'SE': round(se, 1),
        'P_value': round(pval, 4)
    }

# Create results table
results_list = [
    extract_rdd_results(rdd_result, "Main (linear, triangular)"),
    extract_rdd_results(rdd_result_cov, "With covariates"),
    extract_rdd_results(rdd_uniform, "Uniform kernel"),
    extract_rdd_results(rdd_quadratic, "Quadratic polynomial")
]

results_table = pd.DataFrame(results_list)
print("\nRDD Results Comparison Table:")
print(results_table.to_string(index=False))

# -----------------------------------------------------------------------------
# 3.4 Visualization: Raw Data vs Binned Data
# -----------------------------------------------------------------------------
# Binning helps us see the pattern by reducing noise.
# We group observations into bins and calculate local averages.
# Important: The regression is still run on the RAW data, not the bins!

# Subset data for plotting clarity (zoom in around cutoff)
subset_data = data[(data['x'] > 40) & (data['x'] < 60)].copy()

# --- Raw Data Scatter Plot ---
# This shows the individual data points - notice how noisy it is!
fig, ax = plt.subplots(figsize=(10, 7))

# Separate control and treated groups
control = subset_data[subset_data['treatment'] == 0]
treated = subset_data[subset_data['treatment'] == 1]

ax.scatter(control['x'], control['y'], alpha=0.3, s=15, c='#8C8C8C', label='Control')
ax.scatter(treated['x'], treated['y'], alpha=0.3, s=15, c='#5C5C5C', label='Treated')
ax.axvline(x=50, linestyle='--', color='#F18F01', linewidth=1.2)

ax.set_xlabel('Loyalty score', fontsize=14 * 1.2)
ax.set_ylabel('Future spending (R$)', fontsize=14 * 1.2)
ax.set_title('Raw data: Individual observations', fontsize=16, fontweight='bold')
ax.set_xticks(np.arange(40, 62, 2))
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("images/rdd_raw_scatter_py.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# 3.5 Visualization (binned plot)

# Define bandwidth boundaries for visualization
bw_min = 45
bw_max = 55

# Use rdplot to bin data appropriately and fit local polynomials
rdplot(
    y=subset_data['y'],
    x=subset_data['x'],
    c=50,
    ci=95,
    title="RDD plot: Free shipping effect on future spending",
    y_label="Future spending (R$)",
    x_label="Loyalty score"
)
plt.savefig("images/sharp_rdd_plot_py.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# -----------------------------------------------------------------------------
# 4. Fuzzy RDD: Data Generation
# -----------------------------------------------------------------------------
# Fuzzy RDD adds a second source of selection: who TAKES treatment once eligible.
# Unobserved confounder U ("business ambition") drives credit score, loan uptake,
# and revenue, creating selection bias that naive comparisons cannot handle.

np.random.seed(2025)
n_fuzzy = 15000

# Unobs. confounder: high-U businesses have better finances, more likely to take loans, higher revenue
U = np.random.normal(0, 1, n_fuzzy)

# Running variable: credit score (correlated with U)
x = 650 + 40 * U + np.random.normal(0, 60, n_fuzzy)
x = np.clip(x, 400, 900)  # Bound between 400 and 900

# Observed covariates (balanced at cutoff, correlated with U)
business_age = np.maximum(np.round(8 + 2 * U + np.random.normal(0, 3, n_fuzzy)), 1)
employees = np.maximum(np.round(12 + 3 * U + np.random.poisson(5, n_fuzzy)), 1)

# Eligibility (instrument Z): deterministic at cutoff
Z = (x >= 650).astype(int)

# Step 5: Treatment Uptake (D) - the fuzzy part
# Probability depends on eligibility AND unobserved ambition (U)
# U affects both uptake and outcomes, creating selection bias

def expit(x):
    """Logistic function (inverse logit)."""
    return 1 / (1 + np.exp(-x))

prob_below = expit(-1.5 + 0.8 * U)  # ~15-30% depending on U
prob_above = expit(1.2 + 0.6 * U)   # ~70-90% depending on U

prob_treatment = np.where(Z == 1, prob_above, prob_below)
D = np.random.binomial(1, prob_treatment)

# Expected treatment rates: ~20% below cutoff, ~80% above

# Outcome: TRUE EFFECT = R$5,000; U also affects revenue (selection bias)
treatment_effect = 5000

y = (25000 +
     15 * (x - 650) +               # Credit score effect
     treatment_effect * D +          # TRUE EFFECT (what we want to recover)
     3000 * U +                      # Selection bias (U affects revenue directly)
     100 * business_age +            # Older businesses earn slightly more
     50 * employees +                # More employees = more capacity
     np.random.normal(0, 4000, n_fuzzy))  # Noise

y = np.maximum(y, 0)  # Revenue can't be negative

# Assemble and save
data_fuzzy = pd.DataFrame({
    'x': x,
    'y': y,
    'D': D,
    'Z': Z,
    'business_age': business_age,
    'employees': employees
})

# Save the data
# data_fuzzy.to_csv("/Users/robsontigre/Desktop/everyday-ci/data/rdd_fuzzy_fintech.csv", index=False)

# -----------------------------------------------------------------------------
# 4.1 Demonstrating the Selection Problem
# -----------------------------------------------------------------------------

# Load data (or use the data frame we just created)
data_fuzzy = pd.read_csv("/Users/robsontigre/Desktop/everyday-ci/data/rdd_fuzzy_fintech.csv")

# Before running Fuzzy RDD, let's see why naive comparisons fail.

# Naive comparison: Just compare loan-takers to non-loan-takers
naive_fuzzy = smf.ols("y ~ D", data=data_fuzzy).fit()
print("Naive comparison (Fuzzy):")
print(naive_fuzzy.summary())

# OLS with controls: still biased (U is unobserved)
ols_fuzzy = smf.ols("y ~ D + x + business_age + employees", data=data_fuzzy).fit()
print("OLS with controls (Fuzzy):")
print(ols_fuzzy.summary())

# -----------------------------------------------------------------------------
# 4.2 Validation Checks
# -----------------------------------------------------------------------------

# Manipulation test: p > 0.05 means no evidence of gaming
density_test_fuzzy = rddensity(X=data_fuzzy['x'], c=650)
print("Density test for Fuzzy RDD:")
print(density_test_fuzzy)

rdplotdensity(density_test_fuzzy, data_fuzzy['x'],
              title="Density test: credit score distribution",
              xlabel="Credit score",
              ylabel="Density")
plt.savefig("images/rdd_density_fuzzy_py.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# Covariate balance: covariates should not jump at cutoff
print("Covariate balance - business_age:")
print(rdrobust(y=data_fuzzy['business_age'], x=data_fuzzy['x'], c=650))

print("Covariate balance - employees:")
print(rdrobust(y=data_fuzzy['employees'], x=data_fuzzy['x'], c=650))

# First stage: eligibility must strongly predict treatment
first_stage = rdrobust(y=data_fuzzy['D'], x=data_fuzzy['x'], c=650)
print("First stage (eligibility -> treatment):")
print(first_stage)
# We need: (1) significant coefficient, (2) large jump in probability

# Visualize the first stage
fig, ax = plt.subplots(figsize=(10, 7))

# Scatter plot with jitter for binary outcome
ax.scatter(data_fuzzy['x'], data_fuzzy['D'], alpha=0.15, s=8, color='gray')

# Fit smooth lines on either side using LOWESS
below_data = data_fuzzy[data_fuzzy['x'] < 650].sort_values('x')
above_data = data_fuzzy[data_fuzzy['x'] >= 650].sort_values('x')

lowess_below = lowess(below_data['D'], below_data['x'], frac=0.3)
lowess_above = lowess(above_data['D'], above_data['x'], frac=0.3)

ax.plot(lowess_below[:, 0], lowess_below[:, 1], color='#2E86AB', linewidth=1.2)
ax.fill_between(lowess_below[:, 0],
                lowess_below[:, 1] - 0.05,
                lowess_below[:, 1] + 0.05,
                color='#2E86AB', alpha=0.2)

ax.plot(lowess_above[:, 0], lowess_above[:, 1], color='#2E86AB', linewidth=1.2)
ax.fill_between(lowess_above[:, 0],
                lowess_above[:, 1] - 0.05,
                lowess_above[:, 1] + 0.05,
                color='#2E86AB', alpha=0.2)

ax.axvline(x=650, linestyle='--', color='#F18F01', linewidth=1.2)
ax.set_ylim(0, 1)
ax.set_xlabel('Credit score', fontsize=14 * 1.3)
ax.set_ylabel('Probability of getting a loan', fontsize=14 * 1.3)
ax.set_title('First stage: loan receipt probability by credit score', fontsize=16, fontweight='bold')
ax.text(0.5, -0.12, 'The jump at the cutoff is the variation Fuzzy RDD exploits',
        transform=ax.transAxes, ha='center', fontsize=11, color='gray')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("images/rdd_fuzzy_first_stage_py.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# -----------------------------------------------------------------------------
# 4.3 Estimating the LATE (Local Average Treatment Effect)
# -----------------------------------------------------------------------------
# The `fuzzy` argument tells rdrobust to use the cutoff as an instrument

# Main specification
fuzzy_result = rdrobust(
    y=data_fuzzy['y'],
    x=data_fuzzy['x'],
    c=650,
    fuzzy=data_fuzzy['D'].values
)
print("Fuzzy RDD result:")
print(fuzzy_result)

# With covariates (reduces SE without changing estimate)
covariates_fuzzy = data_fuzzy[['business_age', 'employees']].values
fuzzy_result_cov = rdrobust(
    y=data_fuzzy['y'],
    x=data_fuzzy['x'],
    c=650,
    fuzzy=data_fuzzy['D'].values,
    covs=covariates_fuzzy
)
print("Fuzzy RDD with covariates:")
print(fuzzy_result_cov)

# Robustness checks
fuzzy_uniform = rdrobust(
    y=data_fuzzy['y'],
    x=data_fuzzy['x'],
    c=650,
    fuzzy=data_fuzzy['D'].values,
    kernel='uniform'
)
print("Fuzzy RDD with uniform kernel:")
print(fuzzy_uniform)

fuzzy_quadratic = rdrobust(
    y=data_fuzzy['y'],
    x=data_fuzzy['x'],
    c=650,
    fuzzy=data_fuzzy['D'].values,
    p=2
)
print("Fuzzy RDD with quadratic polynomial:")
print(fuzzy_quadratic)

# -----------------------------------------------------------------------------
# 4.4 Visualization: The Reduced Form Jump
# -----------------------------------------------------------------------------
# rdplot shows reduced form (Y jump); LATE = reduced form / first stage

subset_data_fuzzy = data_fuzzy[(data_fuzzy['x'] > 550) & (data_fuzzy['x'] < 750)].copy()

# Get rdplot base
rdplot(
    y=subset_data_fuzzy['y'],
    x=subset_data_fuzzy['x'],
    c=650,
    title="Fuzzy RDD: loan effect on secondary revenue",
    y_label="Secondary revenue (R$)",
    x_label="Credit score"
)
plt.savefig("images/rdd_fuzzy_fintech_result_py.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print("\n" + "="*60)
print("Analysis complete!")
print("="*60)
print("\nKey findings:")
print(f"- Sharp RDD true effect: 250 | Estimated: ~{round(rdd_result.coef.iloc[0, 0])}")
print(f"- Fuzzy RDD true effect: 5000 | Estimated: ~{round(fuzzy_result.coef.iloc[0, 0])}")
print("\nImages saved to images/ directory with '_py' suffix.")
