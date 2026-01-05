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

library(tidyverse)
library(RCT)

set.seed(123)
n <- 5000

# Simulate covariates
user_tenure <- sample(1:36, n, replace = TRUE) # months on platform
prior_purchases <- rpois(n, lambda = 5)
pre_profit <- pmax(0, rlnorm(n, meanlog = 3, sdlog = 0.5)) # past 30 days profit (R$, truncated at 0)

# Treatment assignment
treatment <- rbinom(n, 1, 0.5)

# Potential outcome: base + treatment effect + noise
# We weight pre_profit heavily (0.8) to model persistence, plus some lift from tenure/purchases
base_rev <- 2 + 0.2 * prior_purchases + 0.1 * user_tenure + 0.8 * pre_profit
effect <- 2.5 # targeting ~10% lift over the ~23 baseline

# profit_30d is related to features and prior_purchases/pre_profit too
profit_30d <- base_rev + treatment * effect + rnorm(n, 0, 5)

df <- data.frame(
    user_id = 1:n,
    treatment,
    pre_profit,
    prior_purchases,
    user_tenure,
    profit_30d
)

# Save the simulated data
# write.csv(df, "/Users/robsontigre/Desktop/everyday-ci/data/personalized_feed_experiment.csv", row.names = FALSE)

options(scipen = 999)

df <- read.csv("/Users/robsontigre/Desktop/everyday-ci/data/personalized_feed_experiment.csv")

# balance table between treatment and control groups with t-tests for all variables
balance_table <- balance_table(
    data = df,
    treatment = "treatment"
)
print(balance_table)

#   variables1      Media_control1 Media_trat1 p_value1
#   <chr>                    <dbl>       <dbl>    <dbl>
# 1 pre_profit               22.6        22.5  8.43e- 1
# 2 prior_purchases           4.98        5.01 7.14e- 1
# 3 profit_30d               22.8        25.2  7.70e-15
# 4 user_id                2516.       2484.   4.34e- 1
# 5 user_tenure              18.8        18.3  9.09e- 2

# main result
main <- lm(profit_30d ~ treatment, data = df)
summary(main)
# Coefficients:
#             Estimate Std. Error t value             Pr(>|t|)
# (Intercept)  22.8466     0.2101 108.747 < 0.0000000000000002 ***
# treatment     2.3430     0.3003   7.802  0.00000000000000737 ***
# ---

confint(main) # get confidence intervals

# > confint(main) # get confidence intervals
#                 2.5 %    97.5 %
# (Intercept) 22.434733 23.258470
# treatment    1.754289  2.931777


# controlling for user history: features do not account for the treatment effect
robustness1 <- lm(profit_30d ~ treatment + user_tenure + prior_purchases + pre_profit, data = df)
summary(robustness1)

# Coefficients:
#                 Estimate Std. Error t value             Pr(>|t|)
# (Intercept)     1.965472   0.265189   7.412    0.000000000000146 ***
# treatment       2.441896   0.140682  17.358 < 0.0000000000000002 ***
# user_tenure     0.103457   0.006798  15.219 < 0.0000000000000002 ***
# prior_purchases 0.201256   0.031669   6.355    0.000000000227023 ***
# pre_profit      0.795076   0.005992 132.681 < 0.0000000000000002 ***
# ---

# pre-experiment profit: no effect before the treatment
robustness2 <- lm(pre_profit ~ treatment, data = df)
summary(robustness2)

# Coefficients:
#             Estimate Std. Error t value            Pr(>|t|)
# (Intercept) 22.55913    0.23238  97.077 <0.0000000000000002 ***
# treatment   -0.06575    0.33218  -0.198               0.843
# ---
