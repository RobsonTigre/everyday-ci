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
set.seed(123)
n <- 100
holiday <- rbinom(n, size = 1, prob = 0.3)
ad_spend <- rnorm(n, mean = 500 + 100 * holiday, sd = 80)
ad_spend <- pmax(ad_spend, 0)
sales <- 50 + 0.4 * ad_spend + 20 * holiday + rnorm(n, mean = 0, sd = 50)
df_ads <- data.frame(sales, ad_spend, holiday)
# write.csv(df_ads, "data/advertising_data.csv", row.names = FALSE)

# regression analysis
df <- read.csv("/Users/robsontigre/Desktop/everyday-ci/data/advertising_data.csv")

# fit and summarize simple regression model
simple_regression <- lm(sales ~ ad_spend, data = df)
summary(simple_regression)

# Coefficients:
#             Estimate Std. Error t value Pr(>|t|)
# (Intercept) 52.88382   27.62051   1.915   0.0585 .
# ad_spend     0.40700    0.05186   7.849 5.33e-12 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

# fit and summarize multiple regression model
multiple_regression <- lm(sales ~ ad_spend + holiday, data = df)
summary(multiple_regression)

# Coefficients:
#             Estimate Std. Error t value Pr(>|t|)
# (Intercept) 77.22839   30.72825   2.513   0.0136 *
# ad_spend     0.34873    0.06133   5.686 1.37e-07 ***
# holiday     21.48756   12.37870   1.736   0.0858 .
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1


#########################################
# Creating tricky coefficients and example
#########################################
set.seed(42)
n <- 1000
is_ios <- rbinom(n, 1, 0.6)
user_segment <- sample(c("Casual", "Power", "Business"), n, replace = TRUE, prob = c(0.5, 0.3, 0.2))
account_age <- runif(n, 0, 5)
new_ui <- rbinom(n, 1, 0.5)
y <- 10 + 2 * new_ui + 1 * is_ios
y <- y + ifelse(user_segment == "Power", 5, 0) + ifelse(user_segment == "Business", 8, 0)
y <- y + 1.5 * (new_ui * is_ios) + 2 * account_age - 0.3 * (account_age^2) + rnorm(n, 0, 1)
df_ui <- data.frame(time_on_app = y, new_ui, is_ios, user_segment, account_age)
# write.csv(df_ui, "/Users/robsontigre/Desktop/everyday-ci/data/tricky_coefficients.csv", row.names = FALSE)

df <- read.csv("/Users/robsontigre/Desktop/everyday-ci/data/tricky_coefficients.csv")
m1 <- lm(time_on_app ~ new_ui, data = df)
summary(m1)

# Coefficients:
#             Estimate Std. Error t value Pr(>|t|)
# (Intercept)  16.3383     0.1658   98.56   <2e-16 ***
# new_ui        2.8031     0.2303   12.17   <2e-16 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

df$user_segment <- relevel(factor(df$user_segment), ref = "Casual")
m2 <- lm(time_on_app ~ new_ui + user_segment, data = df)
summary(m2)

#                      Estimate Std. Error t value Pr(>|t|)
# (Intercept)          13.21915    0.09242  143.03   <2e-16 ***
# new_ui                2.86917    0.10431   27.51   <2e-16 ***
# user_segmentBusiness  7.95033    0.13663   58.19   <2e-16 ***
# user_segmentPower     4.86654    0.12062   40.34   <2e-16 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

m3 <- lm(time_on_app ~ new_ui * is_ios, data = df)
summary(m3)

# Coefficients:
#               Estimate Std. Error t value Pr(>|t|)
# (Intercept)    15.8250     0.2589  61.127  < 2e-16 ***
# new_ui          1.7079     0.3628   4.708 2.86e-06 ***
# is_ios          0.8331     0.3298   2.526 0.011692 *
# new_ui:is_ios   1.7227     0.4598   3.747 0.000189 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

m4 <- lm(time_on_app ~ new_ui + account_age + I(account_age^2), data = df)
summary(m4)

# Coefficients:
#                  Estimate Std. Error t value Pr(>|t|)
# (Intercept)      14.23257    0.36082  39.446  < 2e-16 ***
# new_ui            2.85386    0.22560  12.650  < 2e-16 ***
# account_age       1.62328    0.31263   5.192 2.52e-07 ***
# I(account_age^2) -0.23856    0.06057  -3.939 8.76e-05 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1


#########################################
# Creating email campaign and example
#########################################
set.seed(42)
n <- 1000
discount_email <- c(rep(0, n / 2), rep(1, n / 2))
amount_spent <- 100 + 5 * discount_email + rnorm(n, mean = 0, sd = 10)
df_email <- data.frame(discount_email, amount_spent)
# write.csv(df_email, "/Users/robsontigre/Desktop/everyday-ci/data/email_campaign.csv", row.names = FALSE)

df <- read.csv("/Users/robsontigre/Desktop/everyday-ci/data/email_campaign.csv")

# fit the model
model <- lm(amount_spent ~ discount_email, data = df)
summary(model)
confint(model)

# Coefficients:
#                Estimate Std. Error t value Pr(>|t|)
# (Intercept)     99.6995     0.4486 222.265  < 2e-16 ***
# discount_email   5.0844     0.6344   8.015 3.06e-15 ***
# ---

# > confint(model)
#                    2.5 %     97.5 %
# (Intercept)    98.819305 100.579770
# discount_email  3.839599   6.329272

##################################
## Appendix 1.A: defining key statistics concepts {.appendix .unnumbered #sec-appendix-1A}
##################################

# read the data from csv
df <- read.csv("/Users/robsontigre/Desktop/everyday-ci/data/advertising_data.csv")
# calculate mean sales
mean_sales <- mean(df$sales)
print(mean_sales) # R$ 266.44

mean(df$sales[df$holiday == 1]) # R$309.04
mean(df$sales[df$holiday == 0]) # R$249.03

## R code
var(df$sales) # R$3635.8
sd(df$sales) # R$60.3

cor(df)
