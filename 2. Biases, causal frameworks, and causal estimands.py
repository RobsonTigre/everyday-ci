##########################################################
## Everyday Causal Inference: How to estimate, test, and explain impacts with R and Python
## www.everydaycausal.com
## Copyright © 2025 by Robson Tigre. All rights reserved.
## You may read, share, and cite for learning purposes, provided you credit the source.
## It should not be used to create competing educational or commercial products
##########################################################
## Code for Chapter 2 - Biases, causal frameworks, and causal estimands
## Created: Dec 18, 2025
## Last modified: Dec 19, 2025
##########################################################

import pandas as pd # Data manipulation
import numpy as np # Mathematical computing in Python
import statsmodels.formula.api as smf # Linear regression

np.random.seed(42)
n = 800

# Engagement: drives everything else
engagement = np.random.normal(loc=8, scale=3, size=n) # user time on app, in minutes

# Recommendations shown: strongly correlated with engagement (more time = more shown)
# Note: lambda must be non-negative. In this simulation parameters are chosen so it stays positive.
recommendations_shown = np.random.poisson(lam=2 + 1.5 * engagement)

# Total R$ spent: strongly related to engagement, weakly of recommendations
total_spent = np.random.normal(
    loc=40 + 15 * engagement + 0.5 * recommendations_shown, # main effect = engagement!
    scale=25,
    size=n
)
total_spent = np.maximum(total_spent, 0)

df = pd.DataFrame({
    'engagement': engagement,
    'recommendations_shown': recommendations_shown,
    'total_spent': total_spent
})

# Save the data as CSV
# df.to_csv("/Users/robsontigre/Desktop/everyday-ci/data/recommend_spend.csv", index=False)

import pandas as pd # Data manipulation
import statsmodels.formula.api as smf # Linear regression

# Read the CSV file
df = pd.read_csv("/Users/robsontigre/Desktop/everyday-ci/data/recommend_spend.csv")

# Run the regression
simple_regression = smf.ols('total_spent ~ recommendations_shown', data=df).fit()
# Summarize the results
print(simple_regression.summary())

# =========================================================================================
#                             coef    std err          t      P>|t|      [0.025      0.975]
# -----------------------------------------------------------------------------------------
# Intercept                80.0336      3.432     23.317      0.000      73.296      86.771
# recommendations_shown     6.1177      0.229     26.752      0.000       5.669       6.567
# ==============================================================================

# Run the multiple regression
multiple_regression = smf.ols('total_spent ~ recommendations_shown + engagement', data=df).fit()

# Summarize the results
print(multiple_regression.summary())

# =========================================================================================
#                             coef    std err          t      P>|t|      [0.025      0.975]
# -----------------------------------------------------------------------------------------
# Intercept                42.7209      2.564     16.662      0.000      37.688      47.754
# recommendations_shown     0.6061      0.230      2.630      0.009       0.154       1.058
# engagement               14.4854      0.455     31.807      0.000      13.591      15.379
# ==============================================================================