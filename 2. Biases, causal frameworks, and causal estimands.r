##########################################################
## Everyday Causal Inference: How to estimate, test, and explain impacts with R and Python
## Copyright © 2025 by Robson Tigre. All rights reserved.
## You may read, share, and cite for learning purposes, provided you credit the source.
## It should not be used to create competing educational or commercial products
##########################################################
## Code for Chapter 2 - Biases, causal frameworks, and causal estimands
## Created: Dec 18, 2025
## Last modified: Dec 19, 2025
##########################################################

library(ggplot2)
library(broom)

set.seed(42)
n <- 800

# Engagement: drives everything else
engagement <- rnorm(n, mean = 8, sd = 3) # user time on app, in minutes

# Recommendations shown: strongly correlated with engagement (more time = more shown)
recommendations_shown <- rpois(n, lambda = 2 + 1.5 * engagement)

# Total R$ spent: strongly related to engagement, weakly of recommendations
total_spent <- rnorm(
    n,
    mean = 40 + 15 * engagement + 0.5 * recommendations_shown, # main effect = engagement!
    sd = 25
)
total_spent <- pmax(total_spent, 0)

df <- data.frame(
    engagement = engagement,
    recommendations_shown = recommendations_shown,
    total_spent = total_spent
)

# Save the data as CSV
# write.csv(df, "/Users/robsontigre/Desktop/everyday-ci/data/recommend_spend.csv", row.names = FALSE)


# Read the CSV file
df <- read.csv("/Users/robsontigre/Desktop/everyday-ci/data/recommend_spend.csv")

# Run the regression
simple_regression <- lm(total_spent ~ recommendations_shown, data = df)
# Summarize the results
summary(simple_regression)

# Coefficients:
#                       Estimate Std. Error t value Pr(>|t|)
# (Intercept)            80.0336     3.4325   23.32   <2e-16 ***
# recommendations_shown   6.1177     0.2287   26.75   <2e-16 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1


# Run the multiple regression
multiple_regression <- lm(total_spent ~ recommendations_shown + engagement, data = df)

# Summarize the results
summary(multiple_regression)

# Coefficients:
#                       Estimate Std. Error t value Pr(>|t|)
# (Intercept)            42.7209     2.5640   16.66  < 2e-16 ***
# recommendations_shown   0.6061     0.2304    2.63  0.00869 **
# engagement             14.4854     0.4554   31.81  < 2e-16 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
