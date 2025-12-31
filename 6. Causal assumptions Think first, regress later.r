##########################################################
## Everyday Causal Inference: How to estimate, test, and explain impacts with R and Python
## Copyright © 2025 by Robson Tigre. All rights reserved.
## You may read, share, and cite for learning purposes, provided you credit the source.
## It should not be used to create competing educational or commercial products
##########################################################
## Code for Chapter 6 - Causal assumptions: Think first, regress later
## Created: Dec 25, 2025
## Last modified: Dec 31, 2025
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
# write.csv(df, "/Users/robsontigre/Desktop/everyday-ci/data/engagement.csv", row.names = FALSE)

# --- RUNNING THE REGRESSIONS ---

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
# write.csv(df, "/Users/robsontigre/Desktop/everyday-ci/data/email_frequency.csv", row.names = FALSE)

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
        size = 6,
        label.size = 0
    ) +
    annotate("label",
        x = 5, y = 55,
        label = "True peak:\n~5 emails/week",
        color = book_colors$primary,
        fill = "white",
        size = 6,
        label.size = 0
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
ggsave("/Users/robsontigre/Desktop/everyday-causal-inference/images/non_linearity.png", p_nonlinear, width = 10, height = 7, dpi = 300)


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
        fontface = "bold",
        label.size = 0
    ) +
    annotate("label",
        x = 1.5, y = 68,
        label = sprintf("High-traffic: +%.2f", coef(model_high)["ad_spend"]),
        color = book_colors$primary,
        fill = "white",
        size = 5,
        fontface = "bold",
        label.size = 0
    ) +
    annotate("label",
        x = 9.5, y = 52,
        label = sprintf("Low-traffic: +%.2f", coef(model_low)["ad_spend"]),
        color = book_colors$secondary,
        fill = "white",
        size = 5,
        fontface = "bold",
        label.size = 0
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
