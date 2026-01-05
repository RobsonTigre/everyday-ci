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

# If you haven't already, run this in your terminal to install the packages:
# pip install numpy pandas statsmodels linearmodels matplotlib scipy

import numpy as np
import pandas as pd
from linearmodels.iv import IV2SLS
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.special import expit  # logistic function
import matplotlib.pyplot as plt

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
np.random.seed(42)

# Simulate data
n = 10000

# Random assignment to invitation (instrument Z)
# This is the KEY: random assignment ensures independence from U
Z = np.random.binomial(1, 0.5, n)

# Observed covariates (pre-treatment characteristics)
tenure_months = np.maximum(np.random.normal(24, 12, n), 1)
past_purchase = np.maximum(np.random.normal(200, 100, n), 0)
engagement_score = np.maximum(np.random.normal(50, 20, n), 0)

#########################################
# THE UNOBSERVABLE VARIABLE
# This is the source of selection bias. Think of it as "user enthusiasm"
# or "deal-seeking propensity" - something that drives both:
#   1. Willingness to opt into daily emails
#   2. Baseline spending behavior
#
# The data scientist CANNOT observe this variable. It's in the error term.
#########################################
unobservable_variable = np.random.standard_normal(n)

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
# Baseline (Z=0): expit(-2.9) ≈ 0.05 (5% always-takers)
# Invited (Z=1): expit(-2.9 + 2.3) = expit(-0.6) ≈ 0.35 + effect of U
# The 0.5 coefficient on U means enthusiastic users are more likely to opt-in
logit_prob = -2.9 + 2.3 * Z + 0.5 * unobservable_variable
prob_daily_email = expit(logit_prob)
D = np.random.binomial(1, prob_daily_email)

# Baseline revenue (depends on customer characteristics)
baseline_revenue = (100 + 
                    0.5 * tenure_months + 
                    0.2 * past_purchase + 
                    0.3 * engagement_score +
                    np.random.normal(0, 30, n))

# True treatment effect for those who get daily emails
true_LATE = 15

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
revenue = baseline_revenue + D * true_LATE + 8 * unobservable_variable + np.random.normal(0, 20, n)

# Create dataframe (note: unobservable_variable is NOT included - analyst can't see it)
df = pd.DataFrame({
    'customer_id': range(1, n + 1),
    'invitation': Z,
    'daily_emails': D,
    'revenue': revenue,
    'tenure_months': tenure_months,
    'past_purchase': past_purchase,
    'engagement_score': engagement_score
})

# Save for later use
# df.to_csv('data/email_frequency_experiment.csv', index=False)

print(df.head())

# Check compliance rates
# Proportion opted in (invited, Z=1): ~36%
# Proportion opted in (control, Z=0): ~5.5%
# First stage (difference): ~31 percentage points

#########################################
# Checking the First Stage (Relevance Assumption)
#########################################

# Read data from csv
df = pd.read_csv("/Users/robsontigre/Desktop/everyday-ci/data/daily_communications.csv")

# First stage: effect of invitation on daily email adoption
X_first = df[['invitation', 'tenure_months', 'past_purchase', 'engagement_score']]
X_first = sm.add_constant(X_first)
y_first = df['daily_emails']

first_stage = sm.OLS(y_first, X_first).fit(cov_type='HC3')
print("\nFirst Stage Results:")
print(first_stage.summary())

# F-statistic >> 10 indicates a strong instrument
t_stat = first_stage.tvalues['invitation']
f_stat = t_stat**2
print(f"\nFirst-stage F-statistic: {f_stat:.2f}")

# Visualize first stage
first_stage_data = df.groupby('invitation')['daily_emails'].agg(['mean', 'sem'])
first_stage_data['ci_lower'] = first_stage_data['mean'] - 1.96 * first_stage_data['sem']
first_stage_data['ci_upper'] = first_stage_data['mean'] + 1.96 * first_stage_data['sem']

fig, ax = plt.subplots(figsize=(8, 6))
x_pos = [0, 1]
ax.bar(x_pos, first_stage_data['mean'], alpha=0.7, color='#0072B2')
ax.errorbar(x_pos, first_stage_data['mean'], 
            yerr=[first_stage_data['mean'] - first_stage_data['ci_lower'],
                  first_stage_data['ci_upper'] - first_stage_data['mean']],
            fmt='none', color='black', capsize=5)

ax.set_xlabel('Assigned to invitation', fontsize=12)
ax.set_ylabel('Proportion receiving daily emails', fontsize=12)
ax.set_title('First stage: invitation increases daily email adoption', 
             fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(['No invitation\n(Control)', 'Received invitation\n(Treatment)'])
ax.set_ylim(0, 0.5)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
ax.grid(True, alpha=0.3, axis='y')
ax.set_facecolor('white')
fig.patch.set_facecolor('white')
plt.figtext(0.5, 0.01, 'Error bars show 95% confidence intervals', 
            ha='center', fontsize=10, style='italic')
plt.tight_layout()
# plt.savefig('images/iv_email_firststage.png', dpi=300, bbox_inches='tight') 
plt.close()

#########################################
# Estimating the ITT (Reduced Form)
#########################################

# ITT / Reduced form: effect of invitation assignment on revenue
X_itt = df[['invitation', 'tenure_months', 'past_purchase', 'engagement_score']]
X_itt = sm.add_constant(X_itt)
y_itt = df['revenue']

itt_model = sm.OLS(y_itt, X_itt).fit(cov_type='HC3')
print("\nITT (Reduced Form) Results:")
print(itt_model.summary())

# Extract ITT estimate (effect of being offered the invitation on revenue)
itt_estimate = itt_model.params['invitation']

#########################################
# Estimating the LATE with IV/2SLS
#########################################

# LATE estimation using 2SLS
# dependent variable
dependent = df[['revenue']]

# exogenous regressors (covariates that go in both stages)
exog = sm.add_constant(df[['tenure_months', 'past_purchase', 'engagement_score']])

# endogenous variable (treatment that needs to be instrumented)
endog_var = df[['daily_emails']]

# instruments (the excluded instrument, only in first stage)
instruments = df[['invitation']]

# Fit IV model
iv_model = IV2SLS(dependent, exog, endog_var, instruments).fit(cov_type='robust')
print("\nLATE (IV/2SLS) Results:")
print(iv_model.summary)

# Extract LATE estimate (effect of daily emails for compliers)
# True effect in simulation: 15
# LATE / ITT ratio approximately equals 1 / first_stage_effect
late_estimate = iv_model.params['daily_emails']

# Robustness check: does adding covariates change the estimate?
# We expect similar results if assignment is truly random
# Simple IV: revenue ~ daily_emails (instrumented by invitation)
iv_simple = IV2SLS(dependent, pd.DataFrame({'const': np.ones(len(df))}), endog_var, instruments).fit(cov_type='robust')
simple_late = iv_simple.params['daily_emails']
# Simple LATE should be similar if assignment is truly random

# Placebo test: does the instrument affect pre-treatment variables?
# We expect NO effect on past purchases
placebo_model = sm.OLS(df['past_purchase'], sm.add_constant(df[['invitation']])).fit(cov_type='HC3')

# Placebo test: coefficient for invitation should be insignificant (p > 0.05)
print("\nPlacebo Test:")
print(placebo_model.summary())

#########################################
# As-treated bias demonstration
# This shows why naive comparisons are misleading
#########################################

# As-treated analysis:
# Naively compare those who actually took the treatment vs. those who didn't.
# This ignores the random assignment and reintroduces selection bias.
X_pp = df[['daily_emails', 'tenure_months', 'past_purchase', 'engagement_score']]
X_pp = sm.add_constant(X_pp)
y_pp = df['revenue']

as_treated = sm.OLS(y_pp, X_pp).fit(cov_type='HC3')
print("\nAs-Treated (Naive) Results:")
print(as_treated.summary())

# Extract estimates for comparison
naive_estimate = as_treated.params['daily_emails']

# Calculate bias
# The naive estimate is biased upward because the unobservable_variable
# (which the analyst cannot see) drives both:
#   1. Who opts into daily emails (enthusiastic users more likely)
#   2. Who spends more money (enthusiastic users spend more regardless)
# IV works because the invitation is RANDOMLY assigned, so it's independent
# of the unobservable_variable.
bias_from_truth = naive_estimate - true_LATE
percent_bias = (naive_estimate - true_LATE) / true_LATE * 100

# Compare estimates:
# - True LATE (from DGP): 15
# - As-treated (naive): ~18.7 (biased upward by ~25%)
# - IV/2SLS (LATE): ~13.6 (close to true effect)
