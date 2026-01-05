##########################################################
## Everyday Causal Inference: How to estimate, test, and explain impacts with R and Python 
## www.everydaycausal.com
## Copyright © 2025 by Robson Tigre. All rights reserved. 
## You may read, share, and cite for learning purposes, provided you credit the source.
## It should not be used to create competing educational or commercial products
##########################################################
## Code for Chapter 3 - Planning and designing credible causal analyses
## Created: Dec 18, 2025
## Last modified: Dec 20, 2025
##########################################################

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import ttest_ind

np.random.seed(123)
n = 5000

# Simulate covariates
user_tenure = np.random.choice(range(1, 37), n) # months on platform
prior_purchases = np.random.poisson(lam=5, size=n)
pre_profit = np.maximum(0, np.random.lognormal(mean=3, sigma=0.5, size=n)) # past 30 days profit

# Treatment assignment
treatment = np.random.binomial(n=1, p=0.5, size=n)

# Potential outcome: base + treatment effect + noise
# Potential outcome: base + treatment effect + noise
# We weight pre_profit heavily (0.8) to model persistence, plus some lift from tenure/purchases
base_rev = 2 + 0.2 * prior_purchases + 0.1 * user_tenure + 0.8 * pre_profit
effect = 2.5 # targeting ~10% lift over the ~23 baseline

# profit_30d is related to features and prior_purchases/pre_profit too
profit_30d = base_rev + treatment * effect + np.random.normal(0, 5, n)

df = pd.DataFrame({
    'user_id': range(1, n + 1),
    'treatment': treatment, 
    'pre_profit': pre_profit, 
    'prior_purchases': prior_purchases,
    'user_tenure': user_tenure, 
    'profit_30d': profit_30d
})

# Save the simulated data
# df.to_csv('/Users/robsontigre/Desktop/everyday-ci/data/personalized_feed_experiment.csv', index=False)

# read the CSV file
df = pd.read_csv('/Users/robsontigre/Desktop/everyday-ci/data/personalized_feed_experiment.csv')

treatment_col = 'treatment'
g1 = df[df[treatment_col] == 0]
g2 = df[df[treatment_col] == 1]

# balance table between treatment and control groups with t-tests for all variables
balance_table = pd.DataFrame([
    {
        'variable': col,
        'mean_control': g1[col].dropna().mean(),
        'mean_treat': g2[col].dropna().mean(),
        't-stat': ttest_ind(g1[col].dropna(), g2[col].dropna()).statistic,
        'p-value': ttest_ind(g1[col].dropna(), g2[col].dropna()).pvalue
    }
    for col in df.select_dtypes(include='number').columns if col != treatment_col
])
print(balance_table)

#           variable  mean_control   mean_treat    t-stat       p-value
# 0          user_id   2516.133177  2484.189620  0.782170  4.341519e-01
# 1       pre_profit     22.559131    22.493381  0.197934  8.431046e-01
# 2  prior_purchases      4.983549     5.006539 -0.365745  7.145705e-01
# 3      user_tenure     18.770466    18.275439  1.691283  9.084507e-02
# 4       profit_30d     22.846601    25.189634 -7.801984  7.368280e-15

main = smf.ols('profit_30d ~ treatment', data=df).fit()
print(main.summary())

# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept     22.8466      0.210    108.747      0.000      22.435      23.258
# treatment      2.3430      0.300      7.802      0.000       1.754       2.932
# ==============================================================================

# controlling for user history: features do not account for the treatment effect
robustness1 = smf.ols(
    'profit_30d ~ treatment + user_tenure + prior_purchases + pre_profit',
    data=df
).fit()
print(robustness1.summary())

# ===================================================================================
#                       coef    std err          t      P>|t|      [0.025      0.975]
# -----------------------------------------------------------------------------------
# Intercept           1.9655      0.265      7.412      0.000       1.446       2.485
# treatment           2.4419      0.141     17.358      0.000       2.166       2.718
# user_tenure         0.1035      0.007     15.219      0.000       0.090       0.117
# prior_purchases     0.2013      0.032      6.355      0.000       0.139       0.263
# pre_profit          0.7951      0.006    132.681      0.000       0.783       0.807
# ==============================================================================

# pre-experiment profit: no effect before the treatment
robustness2 = smf.ols('pre_profit ~ treatment', data=df).fit()
print(robustness2.summary())

# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept     22.5591      0.232     97.077      0.000      22.104      23.015
# treatment     -0.0657      0.332     -0.198      0.843      -0.717       0.585
# ==============================================================================