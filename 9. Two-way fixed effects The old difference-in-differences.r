##########################################################
## Everyday Causal Inference: How to estimate, test, and explain impacts with R and Python
## www.everydaycausal.com
## Copyright © 2025 by Robson Tigre. All rights reserved.
## You may read, share, and cite for learning purposes, provided you credit the source.
## It should not be used to create competing educational or commercial products
##########################################################
## Code for 9 Two-way fixed effects: The old difference-in-differences
## Created: Jan 23, 2026
## Last modified: Jan 23, 2026
##########################################################

# ==========================================================
# SETUP
# ==========================================================

library(tidyverse)
library(fixest)
library(skimr)
library(plm)

rm(list = ls())
gc()

# ==========================================================
# Book-wide Theme and Color Palette
# ==========================================================
# Define a consistent color palette (colorblind-friendly)
book_colors <- list(
    primary = "#2E86AB", # Steel blue - main data
    secondary = "#A23B72", # Magenta - secondary data
    accent = "#F18F01", # Orange - highlights/warnings
    success = "#C73E1D", # Red-orange - thresholds/targets
    muted = "#6C757D", # Gray - reference lines
    light_gray = "grey90", # Light gray - backgrounds
    dark_gray = "grey30" # Dark gray - text
)

# Unified theme for all plots
theme_book <- function(base_size = 14) {
    theme_minimal(base_size = base_size) +
        theme(
            # Title styling
            plot.title = element_text(
                face = "bold",
                size = base_size * 1.3,
                color = "grey20",
                margin = margin(b = 5)
            ),
            plot.subtitle = element_text(
                size = base_size * 0.9,
                color = "grey40",
                margin = margin(b = 15)
            ),
            plot.caption = element_text(
                size = base_size * 0.7,
                color = "grey50",
                hjust = 0,
                margin = margin(t = 10)
            ),
            # Axis styling
            axis.title = element_text(
                size = base_size,
                color = "grey30"
            ),
            axis.text = element_text(
                size = base_size * 0.85,
                color = "grey40"
            ),
            # Grid styling - clean and subtle
            panel.grid.major = element_line(color = "grey90", linewidth = 0.5),
            panel.grid.minor = element_blank(),
            # Legend styling
            legend.position = "bottom",
            legend.title = element_text(size = base_size * 0.9, face = "bold"),
            legend.text = element_text(size = base_size * 0.85),
            # Plot margins
            plot.margin = margin(20, 20, 20, 20)
        )
}


# ==========================================================
# DATA GENERATING PROCESS (DGP)
# ==========================================================
# Simulating an OOH (Out-of-Home) marketing campaign in the
# Brazilian state of Paraiba (PB), with Pernambuco (PE) as control

set.seed(789)
n_stores <- 100 # Number of stores
n_periods <- 10 # Number of periods

# Creating a balanced panel with store fixed effects and serial correlation
data <- expand_grid(
    store = 1:n_stores,
    period = 1:n_periods
) |>
    mutate(
        # States: 60% in Pernambuco (control) and 40% in Paraiba (treatment)
        state = ifelse(store <= 0.6 * n_stores, "PE", "PB"),
        # Treatment indicator (1 for PB stores)
        treatment = ifelse(state == "PB", 1, 0),
        # Store-level fixed effect
        store_effect = rnorm(n_stores, mean = 2, sd = 1)[store]
    ) |>
    # Simulate AR(1) errors for each store (serial correlation)
    group_by(store) |>
    mutate(
        error = arima.sim(
            model = list(ar = 0.6),
            n = n_periods,
            innov = rnorm(n_periods, sd = 2)
        )
    ) |>
    ungroup() |>
    mutate(
        # Baseline sales (in millions of R$) with time trend and store effects
        sales_base = ((1 + (period / 20)) *
            rnorm(n(), mean = ifelse(state == "PE", 12, 10), sd = 2)) +
            store_effect + error,
        # Apply treatment effects with varying intensity over time
        sales = case_when(
            treatment == 1 & period == 7 ~ sales_base * 1.10, # 10% lift
            treatment == 1 & period == 8 ~ sales_base * 1.25, # 25% lift
            treatment == 1 & period == 9 ~ sales_base * 1.20, # 20% lift
            treatment == 1 & period == 10 ~ sales_base * 1.15, # 15% lift
            TRUE ~ sales_base
        ),
        # Dummy for treated units in post-treatment periods
        treated_period = if_else(treatment == 1 & period >= 7, 1, 0),
        # Relative period (event time): treatment starts at period 7
        relative_period = period - 7
    ) |>
    arrange(store, period) |>
    select(store, period, state, treatment, sales, treated_period, relative_period)

# Save the dataset
# write.csv(data, "/Users/robsontigre/Desktop/everyday-ci/data/did-twfe-ooh.csv", row.names = FALSE)
# Data saved to the specified path


# ==========================================================
# DESCRIPTIVE STATISTICS
# ==========================================================
data <- read.csv("/Users/robsontigre/Desktop/everyday-ci/data/did-twfe-ooh.csv")

# Quick summary of the dataset structure and distributions
skim(data)

# Check data structure
str(data)


# ==========================================================
# TESTING FOR SERIAL CORRELATION
# ==========================================================
# Wooldridge's test for serial correlation in panel data
# H0: No serial correlation in the idiosyncratic errors
# Since we simulated AR(1) errors, we expect to reject H0

pwartest(sales ~ period, data = data)

# Interpretation: A significant p-value indicates the presence of
# autocorrelation in the residuals, which justifies our use of
# clustered standard errors at the store level.


# ==========================================================
# EXPLORATORY DATA ANALYSIS
# Figure: raw-data-chap-9.png
# ==========================================================

# Average sales by group over time
avg_sales <- data |>
    group_by(period, state) |>
    summarise(avg_sales = mean(sales), .groups = "drop")

p_raw_data <- ggplot(avg_sales, aes(x = period, y = avg_sales, color = state)) +
    geom_line(linewidth = 1.2) +
    geom_point(size = 3) +
    geom_vline(xintercept = 7, linetype = "dashed", color = book_colors$muted, linewidth = 0.8) +
    scale_color_manual(
        values = c("PE" = book_colors$secondary, "PB" = book_colors$primary),
        labels = c("PE" = "Control (PE)", "PB" = "Treatment (PB)"),
        name = "Group"
    ) +
    # Annotation for treatment start
    annotate("label",
        x = 7, y = min(avg_sales$avg_sales) * 1.2,
        label = "OOH campaign\nstarts",
        color = book_colors$muted,
        fill = "white",
        size = 4,
        label.size = 0
    ) +
    labs(
        title = "Sales trajectories before and after the OOH campaign",
        subtitle = "Stores in Paraiba (treatment) vs Pernambuco (control)",
        x = "Period",
        y = "Average sales (millions R$)",
        caption = "Takeaway: Both groups follow similar trends pre-treatment, then diverge after the campaign starts."
    ) +
    scale_x_continuous(breaks = 1:10) +
    theme_book() +
    theme(
        legend.position = "bottom",
        axis.title = element_text(size = 17)
    )

print(p_raw_data)
ggsave("images/raw-data-chap.png", p_raw_data, width = 10, height = 7, dpi = 300)


# ==========================================================
# AGGREGATED DIFFERENCE-IN-DIFFERENCES MODEL
# ==========================================================
# Model: Y_it = gamma_i + lambda_t + delta * D_it + epsilon_it
# Where:
#   - gamma_i: store fixed effects
#   - lambda_t: period fixed effects
#   - D_it: treatment indicator (treated_period)
#   - delta: average treatment effect (ATT)

agg_did_model <- feols(
    sales ~ treated_period | store + period,
    cluster = ~store,
    data = data
)

summary(agg_did_model)

# Interpretation: The coefficient on treated_period represents the average
# increase in sales (in millions R$) for stores in Paraiba after the OOH
# campaign started, compared to stores in Pernambuco.


# ==========================================================
# EVENT STUDY (DYNAMIC DIFFERENCE-IN-DIFFERENCES)
# ==========================================================
# Model: Y_it = gamma_i + lambda_t + sum_k(delta_k * D_kit) + epsilon_it
# Where:
#   - delta_k: treatment effect at event time k
#   - Reference period: k = -1 (one period before treatment)

event_study_model <- feols(
    sales ~ i(relative_period, treatment, ref = -1) | store + period,
    cluster = ~store,
    data = data
)

summary(event_study_model)

# Joint F-test for pre-treatment coefficients (leads)
# H0: All pre-treatment effects are jointly zero
wald_test <- wald(event_study_model, "relative_period::-[2-6]")
print(wald_test)


# ==========================================================
# EVENT STUDY PLOT
# Figure: es-plot-chap-9.png
# ==========================================================

# Extract coefficients and confidence intervals
coef_df <- broom::tidy(event_study_model, conf.int = TRUE) |>
    filter(str_detect(term, "relative_period")) |>
    mutate(
        relative_period = as.numeric(str_extract(term, "-?\\d+")),
        type = ifelse(relative_period < 0, "Pre-treatment", "Post-treatment")
    )

# Add the reference period (k = -1) with zero effect
coef_df <- bind_rows(
    coef_df,
    tibble(
        term = "relative_period::-1:treatment",
        estimate = 0,
        std.error = 0,
        conf.low = 0,
        conf.high = 0,
        relative_period = -1,
        type = "Pre-treatment"
    )
) |>
    arrange(relative_period)

# Identify max effect for annotation
max_effect_row <- coef_df |> filter(estimate == max(estimate))

# Plot
p_event_study <- ggplot(coef_df, aes(x = relative_period, y = estimate)) +
    # Reference lines
    geom_hline(yintercept = 0, linetype = "dashed", color = book_colors$muted, linewidth = 0.8) +
    geom_vline(xintercept = 0, linetype = "dashed", color = book_colors$muted, linewidth = 0.8) +
    # Shade pre-treatment region
    annotate("rect",
        xmin = -6.5, xmax = 0,
        ymin = min(coef_df$conf.low) - 0.5, ymax = max(coef_df$conf.high) + 0.5,
        fill = book_colors$light_gray, alpha = 0.5
    ) +
    # Error bars and points
    geom_errorbar(
        aes(ymin = conf.low, ymax = conf.high),
        width = 0.2,
        color = book_colors$primary,
        linewidth = 0.8
    ) +
    geom_point(size = 3, color = book_colors$primary) +
    # Annotations
    annotate("text",
        x = -3.5, y = max(coef_df$conf.high) * 0.7,
        label = "Pre-treatment\n(should be ~0)",
        color = book_colors$muted,
        size = 5.5,
        fontface = "italic"
    ) +
    annotate("text",
        x = 2.5, y = max(coef_df$conf.high) * 0.95,
        label = "Post-treatment\n(treatment effect)",
        color = book_colors$primary,
        size = 5.5,
        fontface = "italic"
    ) +
    labs(
        title = "Event study: effect of OOH campaign on sales over time",
        subtitle = "Reference period: k = -1 (one period before treatment)",
        x = "Periods relative to treatment",
        y = "Estimated effect (millions R$)",
        caption = "Takeaway: No significant pre-trends; effect peaks at k=1, then gradually declines."
    ) +
    scale_x_continuous(breaks = seq(-6, 3, 1)) +
    theme_book() +
    theme(
        axis.title = element_text(size = 17)
    )

print(p_event_study)
ggsave("images/es-plot-chap-9.png", p_event_study, width = 10, height = 7, dpi = 300)

# ==========================================================
# INTERPRETING THE RESULTS
# ==========================================================
# AGGREGATED DiD ESTIMATE:
# - The coefficient on 'treated_period' gives the average treatment effect (ATT)
#   in millions of R$. Multiply by 4 for the total effect over post-treatment periods.
#
# EVENT STUDY ESTIMATES:
# - Pre-treatment effects (relative_period < 0) should be close to zero if
#   parallel trends hold. Significant pre-trends would cast doubt on the design.
# - Post-treatment effects (relative_period >= 0) show the dynamic treatment
#   effects over time. Inspect the pattern: peak effect, decay, persistence.
