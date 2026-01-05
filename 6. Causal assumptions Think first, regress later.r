##########################################################
## Everyday Causal Inference: How to estimate, test, and explain impacts with R and Python
## www.everydaycausal.com
## Copyright © 2025 by Robson Tigre. All rights reserved.
## You may read, share, and cite for learning purposes, provided you credit the source.
## It should not be used to create competing educational or commercial products
##########################################################
## Code for Chapter 6 - Causal assumptions: Think first, regress later
## Created: Dec 25, 2025
## Last modified: Jan 02, 2026
##########################################################

library(tidyverse)

#########################################
# Book-wide Theme and Color Palette
#########################################
book_colors <- list(
    primary = "#2E86AB", # Steel blue - main data
    secondary = "#A23B72", # Magenta - secondary data
    accent = "#F18F01", # Orange - highlights/warnings
    success = "#C73E1D", # Red-orange - thresholds/targets
    muted = "#6C757D", # Gray - reference lines
    light_gray = "grey90", # Light gray - backgrounds
    dark_gray = "grey30" # Dark gray - text
)

theme_book <- function(base_size = 14) {
    theme_minimal(base_size = base_size) +
        theme(
            plot.title = element_text(face = "bold", size = base_size * 1.3, color = "grey20", margin = margin(b = 5)),
            plot.subtitle = element_text(size = base_size * 0.9, color = "grey40", margin = margin(b = 15)),
            plot.caption = element_text(size = base_size * 0.7, color = "grey50", hjust = 0, margin = margin(t = 10)),
            axis.title = element_text(size = base_size, color = "grey30"),
            axis.text = element_text(size = base_size * 0.85, color = "grey40"),
            panel.grid.major = element_line(color = "grey90", linewidth = 0.5),
            panel.grid.minor = element_blank(),
            legend.position = "bottom",
            legend.title = element_text(size = base_size * 0.9, face = "bold"),
            legend.text = element_text(size = base_size * 0.85),
            plot.margin = margin(20, 20, 20, 20)
        )
}

#########################################
# Results of adjusting for good and bad controls
#########################################

# Simulation: Good and Bad Controls
set.seed(42)
n <- 5000 # Increased N for stability

# 1. COVARIATES
user_activity <- rnorm(n, 50, 10) # Confounder
age <- rnorm(n, 30, 5) # Outcome Predictor
marketing <- rbinom(n, 1, 0.5) # Treatment Predictor
shoe_size <- runif(n, 0, 12) # Noise

# 2. TREATMENT (Premium)
# Caused by Activity (Strong Confounding) and Marketing (Strong Treatment Predictor)
# Intercept tuned to keep Premium probability balanced
prob_prem <- plogis(-6.5 + 0.08 * user_activity + 5.0 * marketing)
premium <- rbinom(n, 1, prob_prem)

# 3. INTERMEDIATES
# Mediator: Ad-Free Experience (Premium users get it)
ad_free <- ifelse(premium == 1, 1, rbinom(n, 1, 0.1))

# 4. OUTCOME (Engagement)
# TRUE EFFECT of Premium = 9.2 (2 direct + 0.9*8 via Ad-Free)
# Also affected by Activity (Strong Confounding) and Age (Strong Outcome Predictor)
# Intercept tuned to ensure positive engagement values for collider logic
base_eng <- 20 + 2.0 * user_activity - 3.0 * age
engagement <- base_eng + 2 * premium + 8 * ad_free + rnorm(n, 0, 3.5)

# 5. COLLIDER (Support Ticket)
# Caused by Premium AND Engagement
# Tuned to induce substantial bias (Berkson's Paradox)
prob_tick <- plogis(-11 + 2.0 * premium + 0.2 * engagement)
ticket <- rbinom(n, 1, prob_tick)

df <- data.frame(engagement, premium, user_activity, ad_free, ticket, age, marketing, shoe_size)

# Saving data as CSV
write.csv(df, "/Users/robsontigre/Desktop/everyday-ci/data/engagement.csv", row.names = FALSE)

# Reading the data
df <- read.csv("/Users/robsontigre/Desktop/everyday-ci/data/engagement.csv")

# Helper function to extract estimate and SE
get_coef <- function(m) summary(m)$coefficients["premium", c("Estimate", "Std. Error")]

# Case 1: Naive (No Controls)
m1 <- lm(engagement ~ premium, data = df)

# Case 2: Good Control (Confounder)
m2 <- lm(engagement ~ premium + user_activity, data = df)

# Case 3: Bad Control (Mediator)
m3 <- lm(engagement ~ premium + user_activity + ad_free, data = df)

# Case 4: Bad Control (Collider)
m4 <- lm(engagement ~ premium + user_activity + ticket, data = df)

# Case 5: Precision Boost (Outcome Pred)
m5 <- lm(engagement ~ premium + user_activity + age, data = df)

# Case 6: Variance Inflation (Treatment Pred)
m6 <- lm(engagement ~ premium + user_activity + marketing, data = df)

# Case 7: Noise (Neutral)
m7 <- lm(engagement ~ premium + user_activity + shoe_size, data = df)

# Print comparison
models <- list(m1, m2, m3, m4, m5, m6, m7)
truth <- 9.2

# Extract coefficients (Estimate and SE) for all models
coefs <- t(sapply(models, get_coef))

results <- data.frame(
    Model = c(
        "1. Naive", "2. Good (Confounder)", "3. Bad (Mediator)",
        "4. Bad (Collider)", "5. Good (Outcome predictor)", "6. Bad (Treat predictor)",
        "7. Neutral (Noise)"
    ),
    Estimate = coefs[, "Estimate"],
    SE = coefs[, "Std. Error"],
    Truth = truth
)

# Bias and percentage of bias
results$Bias <- results$Estimate - results$Truth
results$Bias_Pct <- round(100 * results$Bias / results$Truth, 1)

# SE change relative to Good Confounder (model 2) as baseline
baseline_se <- results$SE[2]
results$SE_Change_Pct <- round(100 * (results$SE - baseline_se) / baseline_se, 1)

print(results, digits = 3)

#########################################
# OLS misspecification: When linearity assumptions fail
#########################################

# Simulate nonlinear relationship and show misspecification bias
set.seed(123)
n <- 500

# Continuous "intensity": emails per week (ranging 0-10)
emails <- runif(n, 0, 10)

# True outcome: inverted U-shaped (quadratic) relationship
# Revenue peaks around 5 emails/week, then declines (email fatigue)
revenue <- 50 + 8 * emails - 0.8 * emails^2 + rnorm(n, 0, 5)

df <- data.frame(emails, revenue)

# Save to CSV for reproducibility
write.csv(df, "/Users/robsontigre/Desktop/everyday-ci/data/email_frequency.csv", row.names = FALSE)

# Read data
df <- read.csv("/Users/robsontigre/Desktop/everyday-ci/data/email_frequency.csv")

# Misspecified model: linear only
model_linear <- lm(revenue ~ emails, data = df)

# Correct model: includes quadratic term
model_quadratic <- lm(revenue ~ emails + I(emails^2), data = df)

summary(model_linear)
summary(model_quadratic)

# Compare estimates
cat("Linear model (misspecified):\n")
cat(sprintf("  Slope estimate: %.2f (revenue per email)\n", coef(model_linear)["emails"]))

cat("\nQuadratic model (correct):\n")
cat(sprintf("  Linear term: %.2f\n", coef(model_quadratic)["emails"]))
cat(sprintf("  Quadratic term: %.2f\n", coef(model_quadratic)["I(emails^2)"]))

# Visualization using geom_smooth() with book-wide aesthetics
p_nonlinear <- ggplot(df, aes(x = emails, y = revenue)) +
    geom_point(alpha = 0.3, color = book_colors$muted) +
    geom_smooth(
        method = "lm", formula = y ~ x,
        aes(color = "Linear (misspecified)"),
        se = FALSE, linewidth = 1.2
    ) +
    geom_smooth(
        method = "lm", formula = y ~ poly(x, 2),
        aes(color = "Quadratic (correct)"),
        se = FALSE, linewidth = 1.2
    ) +
    # Annotations that teach
    annotate("label",
        x = 8, y = 68,
        label = "Linear model:\nSlope ≈ 0\n(misleading!)",
        color = book_colors$secondary,
        fill = "white",
        size = 6
    ) +
    annotate("label",
        x = 5, y = 55,
        label = "True peak:\n~5 emails/week",
        color = book_colors$primary,
        fill = "white",
        size = 6
    ) +
    scale_color_manual(
        values = c(
            "Linear (misspecified)" = book_colors$secondary,
            "Quadratic (correct)" = book_colors$primary
        ),
        name = "Model"
    ) +
    labs(
        title = "OLS misspecification: linear model misses the true relationship",
        subtitle = "True relationship is inverted-U; linear model averages to near-zero slope",
        x = "Emails per week",
        y = "Weekly revenue (R$)",
        caption = "Takeaway: When the true relationship is nonlinear, a linear model can be severely misleading."
    ) +
    theme_book() +
    theme(
        legend.position = "bottom",
        axis.title = element_text(size = 17)
    )

# save figure
ggsave("/Users/robsontigre/Desktop/everyday-causal-inference/images/ols_misspecification.png", p_nonlinear, width = 10, height = 7, dpi = 300)


#########################################
# Simpson's Paradox: When confounding assumptions fail
#########################################

# Simulate Simpson's Paradox with CONTINUOUS X variable
# Example: Advertising spend vs. sales, confounded by store location
set.seed(42)
n <- 200

# Confounder: Store location (high-traffic vs low-traffic)
# High-traffic stores: prime locations with natural foot traffic, spend less on ads
# Low-traffic stores: struggling locations, must spend heavily on ads to attract customers
location <- sample(c("high_traffic", "low_traffic"), n, replace = TRUE, prob = c(0.5, 0.5))

# X variable (continuous): Advertising spend per week (R$1000s)
# Low-traffic stores: spend MORE on ads (mean = 8) - they need it to survive
# High-traffic stores: spend LESS on ads (mean = 2) - customers find them naturally
ad_spend <- ifelse(location == "low_traffic",
    rnorm(n, mean = 8, sd = 1.5),
    rnorm(n, mean = 2, sd = 1)
)
ad_spend <- pmax(ad_spend, 0.5) # Floor at 0.5

# Outcome: Weekly sales (R$1000s)
# Both location types benefit from ads (+3 sales per R$1000 ad spend)
# But high-traffic stores have HIGHER baseline sales (70 vs 35)
sales <- ifelse(location == "high_traffic", 70, 35) +
    3 * ad_spend +
    rnorm(n, 0, 4)

df <- data.frame(location, ad_spend, sales)

# Save data as CSV
write.csv(df, "/Users/robsontigre/Desktop/everyday-ci/data/simpsons_paradox.csv", row.names = FALSE)

# Read data
df <- read.csv("/Users/robsontigre/Desktop/everyday-ci/data/simpsons_paradox.csv")

# Aggregate analysis (wrong - shows NEGATIVE relationship!)
model_aggregate <- lm(sales ~ ad_spend, data = df)
cat(sprintf("Aggregate slope: %.2f (appears negative!)\n", coef(model_aggregate)["ad_spend"]))

# Stratified analysis (correct - shows POSITIVE effect in each group)
model_high <- lm(sales ~ ad_spend, data = df[df$location == "high_traffic", ])
model_low <- lm(sales ~ ad_spend, data = df[df$location == "low_traffic", ])
cat(sprintf("High-traffic stores slope: %.2f\n", coef(model_high)["ad_spend"]))
cat(sprintf("Low-traffic stores slope: %.2f\n", coef(model_low)["ad_spend"]))

# Regression with location covariate (correct)
model_correct <- lm(sales ~ ad_spend + location, data = df)
summary(model_correct)

# Visualization: Regression lines by group vs aggregate
p_simpson <- ggplot(df, aes(x = ad_spend, y = sales)) +
    # Points colored by location
    geom_point(aes(color = location), alpha = 0.6, size = 2.5) +
    # Aggregate regression line (wrong - ignores confounding)
    geom_smooth(
        method = "lm", formula = y ~ x,
        aes(linetype = "Aggregate (wrong)"),
        color = book_colors$accent,
        se = FALSE, linewidth = 1.5
    ) +
    # Stratified regression lines (correct)
    geom_smooth(
        aes(color = location, linetype = "Stratified (correct)"),
        method = "lm", formula = y ~ x,
        se = FALSE, linewidth = 1.2
    ) +
    # Coefficient annotations
    annotate("label",
        x = 5, y = 70,
        label = sprintf("Aggregate slope: %.2f", coef(model_aggregate)["ad_spend"]),
        color = book_colors$accent,
        fill = "white",
        size = 5,
        fontface = "bold"
    ) +
    annotate("label",
        x = 1.5, y = 68,
        label = sprintf("High-traffic: +%.2f", coef(model_high)["ad_spend"]),
        color = book_colors$primary,
        fill = "white",
        size = 5,
        fontface = "bold"
    ) +
    annotate("label",
        x = 9.5, y = 52,
        label = sprintf("Low-traffic: +%.2f", coef(model_low)["ad_spend"]),
        color = book_colors$secondary,
        fill = "white",
        size = 5,
        fontface = "bold"
    ) +
    scale_color_manual(
        values = c("low_traffic" = book_colors$secondary, "high_traffic" = book_colors$primary),
        name = "Store location",
        labels = c("High-traffic", "Low-traffic")
    ) +
    scale_linetype_manual(
        values = c("Aggregate (wrong)" = "dashed", "Stratified (correct)" = "solid"),
        name = "Analysis"
    ) +
    labs(
        title = "Simpson's paradox: aggregate trend reverses the true effect",
        subtitle = "Aggregate analysis (dashed) shows negative slope; stratified analysis (solid) reveals true positive effect",
        x = "Ad spend (R$1000s per week)",
        y = "Weekly sales (R$1000s)",
        caption = "Takeaway: Ignoring confounders (store location) can completely reverse your conclusions."
    ) +
    theme_book() +
    theme(
        legend.position = "bottom",
        legend.box = "horizontal"
    ) +
    guides(
        color = guide_legend(order = 1),
        linetype = guide_legend(order = 2)
    )

# Save figure
ggsave("/Users/robsontigre/Desktop/everyday-causal-inference/images/simpsons_paradox.png",
    p_simpson,
    width = 10, height = 7, dpi = 300
)

#########################################
# Common Support / Extrapolation Danger
#########################################

set.seed(2026)
n <- 1000

# 1. COVARIATE: Net worth (R$, in thousands)
# Range from 50K to 800K
net_worth <- runif(n, min = 50, max = 800)

# 2. TREATMENT ASSIGNMENT: Premium account
# Probability of premium increases steeply with net worth
# - Below 100K: almost no one has premium
# - 100K-500K: overlap zone (some have, some don't)
# - Above 500K: almost everyone has premium
prob_premium <- plogis(-8 + 0.025 * net_worth)
premium <- rbinom(n, 1, prob_premium)

# 3. POTENTIAL OUTCOMES
# Simple linear relationship: returns increase with net worth
# The key issue is that we have NO CONTROL DATA to learn from above 500K.

# True treatment effect (constant)
tau <- 5

# Y(0): Untreated potential outcome (linear relationship)
y0_true <- 5 + 0.008 * net_worth + rnorm(n, 0, 1.5)

# Y(1): Treated potential outcome
y1_true <- y0_true + tau + rnorm(n, 0, 0.5)

# 4. OBSERVED OUTCOME
returns <- ifelse(premium == 1, y1_true, y0_true)

df_overlap <- data.frame(
    net_worth = net_worth,
    premium = factor(premium, levels = c(0, 1), labels = c("Non-Premium", "Premium")),
    returns = returns
)

# Save data
write.csv(df_overlap, "/Users/robsontigre/Desktop/everyday-ci/data/overlap_demo.csv", row.names = FALSE)

# Read the data
df_overlap <- read.csv("/Users/robsontigre/Desktop/everyday-ci/data/overlap_demo.csv")

# 5. FIT THE LINEAR MODEL
# The model learns Y = alpha + tau*D + beta*X from all available data
# Key issue: it must extrapolate for wealthy customers where no control data exists
model_overlap <- lm(returns ~ premium + net_worth, data = df_overlap)
summary(model_overlap)

# Extract coefficients
alpha_hat <- coef(model_overlap)["(Intercept)"]
tau_hat <- coef(model_overlap)["premiumPremium"]
beta_hat <- coef(model_overlap)["net_worth"]

cat(sprintf("Estimated treatment effect: %.2f (True: %.2f)\n", tau_hat, tau))
cat(sprintf("Estimated slope (beta): %.4f\n", beta_hat))

# 6. CREATE PREDICTION LINES
# Grid of net worth values for smooth prediction lines
x_grid <- seq(50, 800, length.out = 200)

# Model's predicted counterfactual for control (Y_hat(0))
y_hat_0 <- alpha_hat + beta_hat * x_grid

# Model's predicted treated outcome (Y_hat(1))
y_hat_1 <- alpha_hat + tau_hat + beta_hat * x_grid

pred_df <- data.frame(
    net_worth = x_grid,
    y_hat_0 = y_hat_0,
    y_hat_1 = y_hat_1
)

# 7. VISUALIZATION
p_overlap <- ggplot() +
    # Shaded overlap zone (where we have both treated and control data)
    annotate("rect",
        xmin = 50, xmax = 500,
        ymin = -Inf, ymax = Inf,
        fill = "#90EE90", alpha = 0.15 # Light pastel green
    ) +
    # Shaded danger zone (no control data above 500K)
    annotate("rect",
        xmin = 500, xmax = 800,
        ymin = -Inf, ymax = Inf,
        fill = book_colors$accent, alpha = 0.1
    ) +
    # Scatter points (observed data)
    geom_point(
        data = df_overlap,
        aes(x = net_worth, y = returns, color = premium),
        alpha = 0.5, size = 2
    ) +
    # Model's predicted control line (solid in overlap, dashed in extrapolation)
    geom_line(
        data = pred_df %>% filter(net_worth <= 500),
        aes(x = net_worth, y = y_hat_0),
        color = book_colors$accent, linewidth = 1.2
    ) +
    geom_line(
        data = pred_df %>% filter(net_worth >= 500),
        aes(x = net_worth, y = y_hat_0, linetype = "Model's extrapolation"),
        color = book_colors$accent, linewidth = 1.2
    ) +
    # Model's predicted treated line
    geom_line(
        data = pred_df,
        aes(x = net_worth, y = y_hat_1),
        color = book_colors$primary, linewidth = 1.2
    ) +
    # Vertical line marking the boundary
    geom_vline(xintercept = 500, linetype = "dotted", color = book_colors$muted, linewidth = 0.8) +
    # Annotations
    annotate("label",
        x = 650, y = 2,
        label = "Danger zone\nNo control data here",
        color = book_colors$accent,
        fill = "#FEF1E0", # Light orange to match danger zone shade
        size = 4,
        fontface = "italic"
    ) +
    annotate("label",
        x = 275, y = 2,
        label = "Overlap zone\nBoth groups observed",
        color = "#2E8B57",
        fill = "#E8FCE8", # Light mint green to match overlap zone shade
        size = 4,
        fontface = "italic"
    ) +
    # Scales
    scale_color_manual(
        values = c("Non-Premium" = book_colors$accent, "Premium" = book_colors$primary),
        name = "Account type"
    ) +
    scale_linetype_manual(
        values = c("Model's extrapolation" = "dashed"),
        name = ""
    ) +
    labs(
        title = "Lack of overlap: when regression becomes a guess",
        subtitle = "Above R$500K, no control data exists. The model must extrapolate, and you should be skeptical.",
        x = "Net worth (R$ thousands)",
        y = "Investment returns (%)",
        caption = "Orange points = Non-Premium (control). Blue points = Premium (treated).\nDashed orange line = model's extrapolation into the danger zone."
    ) +
    theme_book() +
    theme(
        legend.position = "bottom",
        legend.box = "horizontal"
    ) +
    guides(
        color = guide_legend(order = 1),
        linetype = guide_legend(order = 2)
    )

# Save figure
ggsave("/Users/robsontigre/Desktop/everyday-causal-inference/images/common_support.png",
    p_overlap,
    width = 11, height = 8, dpi = 300
)
