##########################################################
## Everyday Causal Inference: How to estimate, test, and explain impacts with R and Python 
## www.everydaycausal.com
## Copyright © 2025 by Robson Tigre. All rights reserved. 
## You may read, share, and cite for learning purposes, provided you credit the source.
## It should not be used to create competing educational or commercial products
##########################################################
## Code for Chapter 1 - Data, statistical models, and ‘what is causality’
## Created: Dec 18, 2025
## Last modified: Dec 19, 2025
##########################################################

#########################################
# Creating advertising data and example
#########################################
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

np.random.seed(123)
n = 100
holiday = np.random.binomial(1, 0.3, n)
ad_spend = np.random.normal(500 + 100 * holiday, 80, n)
ad_spend = np.maximum(ad_spend, 0)
sales = 50 + 0.4 * ad_spend + 20 * holiday + np.random.normal(0, 50, n)
df_ads = pd.DataFrame({'sales': sales, 'ad_spend': ad_spend, 'holiday': holiday})
# df_ads.to_csv('data/advertising_data.csv', index=False)

# read the data from csv
df = pd.read_csv("/Users/robsontigre/Desktop/everyday-ci/data/advertising_data.csv")

# fit and summarize simple regression model
simple_regression = smf.ols('sales ~ ad_spend', data=df).fit()
print(simple_regression.summary())

# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept     52.8838     27.621      1.915      0.058      -1.928     107.696
# ad_spend       0.4070      0.052      7.849      0.000       0.304       0.510
# ==============================================================================

# fit and summarize multiple regression model
multiple_regression = smf.ols('sales ~ ad_spend + holiday', data=df).fit()
print(multiple_regression.summary())

# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept     77.2284     30.728      2.513      0.014      16.241     138.215
# ad_spend       0.3487      0.061      5.686      0.000       0.227       0.470
# holiday       21.4876     12.379      1.736      0.086      -3.081      46.056
# ==============================================================================

#########################################
# Creating tricky coefficients and example
#########################################
np.random.seed(42)
n = 1000
is_ios = np.random.binomial(1, 0.6, n)
user_segment = np.random.choice(['Casual', 'Power', 'Business'], n, p=[0.5, 0.3, 0.2])
account_age = np.random.uniform(0, 5, n)
new_ui = np.random.binomial(1, 0.5, n)
y = (10 + 2 * new_ui + 1 * is_ios).astype(float)
y += np.where(user_segment == 'Power', 5, 0)
y += np.where(user_segment == 'Business', 8, 0)
y += 1.5 * (new_ui * is_ios) + 2 * account_age - 0.3 * (account_age**2)
y += np.random.normal(0, 1, n)
df_ui = pd.DataFrame({'time_on_app': y, 'new_ui': new_ui, 'is_ios': is_ios, 
                      'user_segment': user_segment, 'account_age': account_age})
# df_ui.to_csv('data/tricky_coefficients.csv', index=False)

df = pd.read_csv("/Users/robsontigre/Desktop/everyday-ci/data/tricky_coefficients.csv")
m1 = smf.ols('time_on_app ~ new_ui', data=df).fit()
print(m1.summary())

# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept     16.3383      0.166     98.563      0.000      16.013      16.664
# new_ui         2.8031      0.230     12.171      0.000       2.351       3.255
# ==============================================================================

m2 = smf.ols('time_on_app ~ new_ui + C(user_segment, Treatment(reference="Casual"))', 
        data=df).fit()
print(m2.summary())

# ==============================================================================================================================
#                                                                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------------------------------------------------------
# Intercept                                                     13.2192      0.092    143.035      0.000      13.038      13.401
# C(user_segment, Treatment(reference="Casual"))[T.Business]     7.9503      0.137     58.187      0.000       7.682       8.218
# C(user_segment, Treatment(reference="Casual"))[T.Power]        4.8665      0.121     40.344      0.000       4.630       5.103
# new_ui                                                         2.8692      0.104     27.506      0.000       2.664       3.074
# ==============================================================================

m3 = smf.ols('time_on_app ~ new_ui * is_ios', data=df).fit()
print(m3.summary())

# =================================================================================
#                     coef    std err          t      P>|t|      [0.025      0.975]
# ---------------------------------------------------------------------------------
# Intercept        15.8250      0.259     61.127      0.000      15.317      16.333
# new_ui            1.7079      0.363      4.708      0.000       0.996       2.420
# is_ios            0.8331      0.330      2.526      0.012       0.186       1.480
# new_ui:is_ios     1.7227      0.460      3.747      0.000       0.821       2.625
# ==============================================================================

# I() allows arithmetic inside the formula
m4 = smf.ols('time_on_app ~ new_ui + account_age + I(account_age**2)', data=df).fit()
print(m4.summary())

# =======================================================================================
#                           coef    std err          t      P>|t|      [0.025      0.975]
# ---------------------------------------------------------------------------------------
# Intercept              14.2326      0.361     39.446      0.000      13.525      14.941
# new_ui                  2.8539      0.226     12.650      0.000       2.411       3.297
# account_age             1.6233      0.313      5.192      0.000       1.010       2.237
# I(account_age ** 2)    -0.2386      0.061     -3.939      0.000      -0.357      -0.120
# ==============================================================================

#########################################
# Creating email campaign and example
#########################################
np.random.seed(42)
n = 1000
discount_email = np.concatenate([np.zeros(n//2), np.ones(n//2)])
amount_spent = 100 + 5 * discount_email + np.random.normal(0, 10, n)
df_email = pd.DataFrame({'discount_email': discount_email, 'amount_spent': amount_spent})
# df_email.to_csv('data/email_campaign.csv', index=False)

# read the data from csv
df = pd.read_csv("/Users/robsontigre/Desktop/everyday-ci/data/email_campaign.csv")

# fit and summarize simple regression model
model = smf.ols('amount_spent ~ discount_email', data=df).fit()
print(model.summary())

# ==================================================================================
#                      coef    std err          t      P>|t|      [0.025      0.975]
# ----------------------------------------------------------------------------------
# Intercept         99.6995      0.449    222.265      0.000      98.819     100.580
# discount_email     5.0844      0.634      8.015      0.000       3.840       6.329
# ==============================================================================

##################################
## Appendix 1.A: defining key statistics concepts {.appendix .unnumbered #sec-appendix-1A}
##################################

# read the data from csv
df = pd.read_csv("/Users/robsontigre/Desktop/everyday-ci/data/advertising_data.csv")
# calculate mean sales
mean_sales = df['sales'].mean()
print(mean_sales) # R$266.44

print(df[df["holiday"] == 1]["sales"].mean()) # R$309.04
print(df[df["holiday"] == 0]["sales"].mean()) # R$249.03

## Python code
print(df["sales"].var()) # R$3635.8
print(df["sales"].std()) # R$60.3
