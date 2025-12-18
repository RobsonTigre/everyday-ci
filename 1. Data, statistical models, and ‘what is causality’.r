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
write.csv(df_ads, "/Users/robsontigre/Desktop/everyday-ci/data/advertising_data.csv", row.names = FALSE)


[include code to generate the figure here]

#########################################
# Creating email campaign and example
#########################################
set.seed(42)
n <- 1000
discount_email <- c(rep(0, n / 2), rep(1, n / 2))
amount_spent <- 100 + 5 * discount_email + rnorm(n, mean = 0, sd = 10)
df_email <- data.frame(discount_email, amount_spent)
write.csv(df_email, "/Users/robsontigre/Desktop/everyday-ci/data/email_campaign.csv", row.names = FALSE)

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
write.csv(df_ui, "/Users/robsontigre/Desktop/everyday-ci/data/tricky_coefficients.csv", row.names = FALSE)
