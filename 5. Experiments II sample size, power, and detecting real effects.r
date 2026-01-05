##########################################################
## Everyday Causal Inference: How to estimate, test, and explain impacts with R and Python
## www.everydaycausal.com
## Copyright © 2025 by Robson Tigre. All rights reserved.
## You may read, share, and cite for learning purposes, provided you credit the source.
## It should not be used to create competing educational or commercial products
##########################################################
## Code for Chapter 5 - Experiments II: sample size, power, and detecting real effects
## Created: Dec 25, 2025
## Last modified: Dec 29, 2025
##########################################################

#########################################
# Package Setup
#########################################
# If you haven't already, run this once to install the packages
# install.packages("tidyverse")
# install.packages("pwr")
# install.packages("patchwork")
# install.packages("scales")
# install.packages("gridExtra")

# You must run the lines below at the start of every new R session.
library(tidyverse) # The "Swiss army knife": loads dplyr, ggplot2, readr, etc.
library(pwr) # Power analysis
library(patchwork) # Combining plots
library(scales) # For formatting
library(gridExtra) # For grid.arrange


#########################################
# Book-wide Theme and Color Palette
#########################################
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


#########################################
# Section 5.1: Hypothesis testing illustration
# Figure: hypothesis-test-panels.png
#########################################
set.seed(123)

# Simulate "Parallel Universes" (The Null Hypothesis)
sim_data <- data.frame(effect = rnorm(10000, mean = 0, sd = 7))

# Calculate critical values for alpha = 0.05 (two-tailed)
critical_val_upper <- quantile(sim_data$effect, 0.975)
critical_val_lower <- quantile(sim_data$effect, 0.025)

# The observed result from the A/B test
observed_effect <- 20

# Create a helper column for fill color
sim_data <- sim_data %>%
    mutate(region = case_when(
        effect > critical_val_upper | effect < critical_val_lower ~ "Rejection region",
        TRUE ~ "Fail to reject"
    ))

# Panel 1: The null distribution
p1 <- ggplot(sim_data, aes(x = effect)) +
    geom_histogram(bins = 60, fill = book_colors$light_gray, color = "white") +
    geom_vline(xintercept = 0, linetype = "dashed", color = book_colors$muted, linewidth = 0.8) +
    annotate("text",
        x = 0, y = 800,
        label = "If H₀ is true,\neffects cluster here",
        color = book_colors$dark_gray,
        size = 4,
        vjust = 0
    ) +
    labs(
        title = "Panel A: The null distribution",
        x = "Estimated effect (R$)",
        y = "Frequency"
    ) +
    scale_x_continuous(limits = c(-30, 30)) +
    theme_book() +
    theme(panel.grid.major.x = element_blank())

# Panel 2: Adding rejection regions
p2 <- ggplot(sim_data, aes(x = effect, fill = region)) +
    geom_histogram(bins = 60, color = "white") +
    geom_vline(xintercept = 0, linetype = "dashed", color = book_colors$muted) +
    geom_vline(
        xintercept = c(critical_val_lower, critical_val_upper),
        linetype = "dashed",
        color = book_colors$dark_gray
    ) +
    scale_fill_manual(
        values = c("Rejection region" = book_colors$success, "Fail to reject" = book_colors$light_gray),
        name = ""
    ) +
    annotate("text",
        x = -22, y = 400,
        label = "Reject H₀\n(2.5%)",
        color = book_colors$success,
        size = 3.5,
        fontface = "bold"
    ) +
    annotate("text",
        x = 22, y = 400,
        label = "Reject H₀\n(2.5%)",
        color = book_colors$success,
        size = 3.5,
        fontface = "bold"
    ) +
    labs(
        title = "Panel B: Setting α = 0.05",
        x = "Estimated effect (R$)",
        y = "Frequency"
    ) +
    scale_x_continuous(limits = c(-30, 30)) +
    guides(fill = "none") +
    theme_book() +
    theme(panel.grid.major.x = element_blank())

# Panel 3: Adding the observed result
p3 <- ggplot(sim_data, aes(x = effect, fill = region)) +
    geom_histogram(bins = 60, color = "white", alpha = 0.5) +
    scale_fill_manual(
        values = c("Rejection region" = book_colors$success, "Fail to reject" = book_colors$light_gray)
    ) +
    geom_vline(
        xintercept = c(critical_val_lower, critical_val_upper),
        linetype = "dashed",
        color = book_colors$dark_gray
    ) +
    # Add the observed arrow/point
    annotate("segment",
        x = observed_effect, xend = observed_effect,
        y = 180, yend = 5,
        arrow = arrow(length = unit(0.3, "cm"), type = "closed"),
        color = book_colors$primary,
        linewidth = 1.5
    ) +
    annotate("label",
        x = observed_effect, y = 250,
        label = paste0("Observed:\n+R$ ", observed_effect),
        color = book_colors$primary,
        fill = "white",
        fontface = "bold",
        size = 4.5,
        label.size = 0
    ) +
    annotate("text",
        x = observed_effect, y = 100,
        label = "p < 0.05\nReject H₀",
        color = book_colors$primary,
        size = 3.5,
        fontface = "italic"
    ) +
    labs(
        title = "Panel C: Our result",
        x = "Estimated effect (R$)",
        y = "Frequency"
    ) +
    scale_x_continuous(limits = c(-30, 30)) +
    guides(fill = "none") +
    theme_book() +
    theme(panel.grid.major.x = element_blank())

combined_plot <- p1 + p2 + p3 + plot_layout(ncol = 3)

print(combined_plot)

# ggsave("images/hypothesis-test-panels.png", combined_plot, width = 18, height = 6, dpi = 600)
# ggsave("images/hypothesis-test-panels.pdf", combined_plot, width = 18, height = 6)


#########################################
# Section 5.3: Basic Power Analysis Example
# Recommendation system A/B test
#########################################
# Historical means and pooled standard deviation
mean_control <- 150
mean_treatment <- 165
sd_pooled <- 75

# Compute effect size (Cohen's d)
effect_size <- abs(mean_treatment - mean_control) / sd_pooled

# Power analysis for two-sample t-test (two-sided)
power_result <- pwr.t.test(
    d = effect_size,
    power = 0.80,
    sig.level = 0.05,
    type = "two.sample",
    alternative = "two.sided"
)

# Required sample size per group
required_n <- ceiling(power_result$n)
print(paste("Effect size (Cohen's d):", effect_size))
print(paste("Required sample size per group:", required_n))


#########################################
# Section 5.3: Power vs Sample Size Curve
# Figure: power_vs_sample_size.png
#########################################
# Define parameters
effect_d <- 0.2
alpha_level <- 0.05
n_range <- seq(10, 600, by = 10)

# Calculate power for each sample size
power_values <- sapply(n_range, function(n) {
    pwr.t.test(n = n, d = effect_d, sig.level = alpha_level, type = "two.sample")$power
})

# Find the minimum n for 80% power
min_n_80 <- n_range[which(power_values >= 0.80)[1]]

# Create data frame for plotting
power_df <- data.frame(n = n_range, power = power_values)

# Create the plot with enhanced didactics
p_power_vs_n <- ggplot(power_df, aes(x = n, y = power)) +
    # Main curve
    geom_line(color = book_colors$primary, linewidth = 1.5) +
    # Reference lines with clear meaning
    geom_hline(
        yintercept = 0.80,
        linetype = "dashed",
        color = book_colors$success,
        linewidth = 0.8
    ) +
    geom_vline(
        xintercept = min_n_80,
        linetype = "dashed",
        color = book_colors$success,
        linewidth = 0.8
    ) +
    # Highlight the intersection point
    geom_point(
        data = data.frame(x = min_n_80, y = 0.80),
        aes(x = x, y = y),
        color = book_colors$success,
        size = 4
    ) +
    # Annotations that teach
    annotate("label",
        x = min_n_80 + 30, y = 0.72,
        label = paste0("You need at least\n", min_n_80, " per group"),
        color = book_colors$success,
        fill = "white",
        size = 5,
        hjust = 0,
        label.size = 0
    ) +
    annotate("text",
        x = 580, y = 0.83,
        label = "Target: 80% power",
        color = book_colors$success,
        size = 5,
        hjust = 1,
        fontface = "italic"
    ) +
    # Shade the "underpowered" region
    annotate("rect",
        xmin = 0, xmax = min_n_80,
        ymin = 0, ymax = 0.80,
        fill = book_colors$accent,
        alpha = 0.1
    ) +
    annotate("text",
        x = min_n_80 / 1.5, y = 0.4,
        label = "Underpowered\nzone",
        color = book_colors$accent,
        size = 5,
        fontface = "italic"
    ) +
    # Labels
    labs(
        x = "Sample size per group",
        y = "Statistical power",
        title = "Power increases with sample size, but with diminishing returns",
        subtitle = paste0("Effect size d = ", effect_d, ", significance level α = ", alpha_level),
        caption = "Takeaway: After reaching 80% power, adding more participants yields marginal gains."
    ) +
    scale_y_continuous(
        limits = c(0, 1),
        breaks = seq(0, 1, 0.2),
        labels = scales::percent_format(accuracy = 1)
    ) +
    scale_x_continuous(breaks = seq(0, 600, 100)) +
    theme_book() +
    theme(axis.title = element_text(size = 17)) # 20% larger than base (14 * 1.2)

print(p_power_vs_n)
# ggsave("images/power_vs_sample_size.png", p_power_vs_n, width = 10, height = 7, dpi = 300)


#########################################
# Section 5.4.1: Effect Size vs Required Sample Size
# Figure: effect_size_vs_n_python.png
#########################################
# Define a sequence of effect sizes to test
effect_sizes <- seq(0.1, 0.5, 0.05)

# Calculate the required sample size for each effect size
sample_sizes <- sapply(effect_sizes, function(d) {
    ceiling(pwr.t.test(
        d = d,
        power = 0.80,
        sig.level = 0.05,
        type = "two.sample"
    )$n)
})

# Create data frame
effect_df <- data.frame(
    effect_size = effect_sizes,
    sample_size = sample_sizes
)

# Create ggplot version for consistency
p_effect_vs_n <- ggplot(effect_df, aes(x = effect_size, y = sample_size)) +
    geom_line(color = book_colors$primary, linewidth = 1.2) +
    geom_point(color = book_colors$primary, size = 3) +
    # Highlight key benchmarks
    geom_hline(
        yintercept = effect_df$sample_size[effect_df$effect_size == 0.2],
        linetype = "dotted",
        color = book_colors$muted
    ) +
    annotate("label",
        x = 0.2, y = effect_df$sample_size[effect_df$effect_size == 0.2] + 150,
        label = paste0("Small effect (d=0.2):\n", effect_df$sample_size[effect_df$effect_size == 0.2], " per group"),
        size = 4,
        fill = "white",
        label.size = 0
    ) +
    annotate("label",
        x = 0.48, y = effect_df$sample_size[effect_df$effect_size == 0.5] + 150,
        label = paste0("Large effect (d=0.5):\nOnly ", effect_df$sample_size[effect_df$effect_size == 0.5], " per group"),
        size = 4,
        fill = "white",
        label.size = 0
    ) +
    labs(
        x = "Effect size (Cohen's d)",
        y = "Required sample size per group",
        title = "Smaller effects require exponentially larger samples",
        subtitle = "Target power = 80%, significance level α = 0.05",
        caption = "Takeaway: Detecting a d=0.2 effect needs 16x more participants than d=0.8."
    ) +
    scale_x_continuous(breaks = seq(0.1, 0.5, 0.1)) +
    scale_y_continuous(labels = scales::comma) +
    theme_book() +
    theme(axis.title = element_text(size = 17)) # 20% larger than base (14 * 1.2)

print(p_effect_vs_n)
# ggsave("images/effect_size_vs_n_python.png", p_effect_vs_n, width = 10, height = 7, dpi = 300)


#########################################
# Section 5.4.2: Power Curves for Different Significance Levels
# Figure: power_curves_alpha_python.png
#########################################
# Define parameters
alpha_levels <- c(0.01, 0.05, 0.10)
n_seq <- seq(10, 200, length.out = 100)

# Calculate power for each alpha level
power_data <- expand.grid(n = n_seq, alpha = alpha_levels) %>%
    mutate(
        power = mapply(function(n, alpha) {
            pwr.t.test(n = n, d = 0.3, sig.level = alpha, type = "two.sample")$power
        }, n, alpha),
        alpha_label = factor(
            paste0("α = ", alpha),
            levels = c("α = 0.01", "α = 0.05", "α = 0.1")
        )
    )

# Create the plot
p_alpha_curves <- ggplot(power_data, aes(x = n, y = power, color = alpha_label)) +
    geom_line(linewidth = 1.2) +
    geom_hline(yintercept = 0.80, linetype = "dashed", color = book_colors$muted) +
    annotate("text",
        x = 100, y = 0.83,
        label = "80% power",
        color = book_colors$muted,
        size = 5,
        hjust = 1,
        fontface = "italic"
    ) +
    # Color-coded annotations
    annotate("text",
        x = 170, y = 0.95,
        label = "More lenient\n(more false positives)",
        color = "#2CA02C",
        size = 5,
        fontface = "italic"
    ) +
    annotate("text",
        x = 170, y = 0.45,
        label = "More stringent\n(fewer false positives)",
        color = "#D62728",
        size = 5,
        fontface = "italic"
    ) +
    scale_color_manual(
        values = c("α = 0.01" = "#D62728", "α = 0.05" = "#1F77B4", "α = 0.1" = "#2CA02C"),
        name = "Significance level"
    ) +
    labs(
        x = "Sample size per group",
        y = "Statistical power",
        title = "The tradeoff: stricter significance requires larger samples",
        subtitle = "Effect size d = 0.3",
        caption = "Takeaway: Lower α protects against false positives but requires more data to detect real effects."
    ) +
    scale_y_continuous(
        limits = c(0, 1),
        breaks = seq(0, 1, 0.2),
        labels = scales::percent_format(accuracy = 1)
    ) +
    theme_book() +
    theme(
        legend.position = "bottom",
        axis.title = element_text(size = 17) # 20% larger than base (14 * 1.2)
    )

print(p_alpha_curves)
# ggsave("images/power_curves_alpha_python.png", p_alpha_curves, width = 10, height = 7, dpi = 300)


#########################################
# Section 5.4.3: Power analysis - allocation ratio
# Figure: power-allocation-ratio.png
#########################################
total_n <- 1000
effect_size <- 0.2
allocation_ratios <- seq(0.1, 0.9, by = 0.1)

# Calculate power for different allocation ratios
powers <- sapply(allocation_ratios, function(ratio) {
    n1 <- total_n * ratio
    n2 <- total_n * (1 - ratio)
    pwr.t2n.test(n1 = n1, n2 = n2, d = effect_size, sig.level = 0.05)$power
})

# Find maximum power
max_power <- max(powers)
max_ratio <- allocation_ratios[which.max(powers)]

# Create data frame
allocation_df <- data.frame(
    ratio = allocation_ratios,
    power = powers
)

# Create ggplot version
p_allocation <- ggplot(allocation_df, aes(x = ratio, y = power)) +
    geom_line(color = book_colors$primary, linewidth = 1.2) +
    geom_point(color = book_colors$primary, size = 3) +
    # Highlight optimal point
    geom_point(
        data = allocation_df %>% filter(ratio == 0.5),
        aes(x = ratio, y = power),
        color = book_colors$success,
        size = 6,
        shape = 21,
        fill = "white",
        stroke = 2
    ) +
    geom_vline(xintercept = 0.5, linetype = "dashed", color = book_colors$success, linewidth = 0.8) +
    geom_hline(yintercept = max_power, linetype = "dotted", color = book_colors$muted) +
    annotate("label",
        x = 0.5, y = max_power + 0.06,
        label = paste0("Optimal: 50/50 split\nPower = ", scales::percent(max_power, accuracy = 0.1)),
        color = book_colors$success,
        fill = "white",
        size = 5,
        fontface = "bold",
        label.size = 0
    ) +
    annotate("text",
        x = 0.2, y = powers[1] + 0.03,
        label = "10% treatment\n= low power",
        color = book_colors$muted,
        size = 5
    ) +
    annotate("text",
        x = 0.8, y = powers[9] + 0.03,
        label = "90% treatment\n= low power",
        color = book_colors$muted,
        size = 5
    ) +
    labs(
        x = "Proportion of N allocated to treatment",
        y = "Statistical power",
        title = "Equal allocation (50/50) maximizes statistical power",
        subtitle = paste0("For this example, total N = ", total_n, ", effect size d = ", effect_size),
        caption = "Takeaway: Deviating from 50/50 reduces power; compensate with larger total samples."
    ) +
    scale_x_continuous(breaks = seq(0.1, 0.9, 0.1), labels = scales::percent_format(accuracy = 1)) +
    scale_y_continuous(
        limits = c(0.35, 1),
        labels = scales::percent_format(accuracy = 1)
    ) +
    theme_book() +
    theme(axis.title = element_text(size = 17)) # 20% larger than base (14 * 1.2)


print(p_allocation)
# ggsave("images/power-allocation-ratio.png", p_allocation, width = 10, height = 7, dpi = 300)


#########################################
# Section 5.4.4: Outcome variance comparison
# Figure: variance_comparison_r.png
#########################################
set.seed(123)

# Parameters for both scenarios
mean_diff <- 20
n <- 10000

# High variance example
mean1_high_var <- 100
mean2_high_var <- mean1_high_var + mean_diff
sd_high_var <- 20

group1_high_var <- rnorm(n, mean = mean1_high_var, sd = sd_high_var)
group2_high_var <- rnorm(n, mean = mean2_high_var, sd = sd_high_var)

# Low variance example
mean1_low_var <- 100
mean2_low_var <- mean1_low_var + mean_diff
sd_low_var <- 5

group1_low_var <- rnorm(n, mean = mean1_low_var, sd = sd_low_var)
group2_low_var <- rnorm(n, mean = mean2_low_var, sd = sd_low_var)

# Combine data into dataframes
data_high_var <- data.frame(
    value = c(group1_high_var, group2_high_var),
    group = factor(rep(c("Control", "Treatment"), each = n))
)

data_low_var <- data.frame(
    value = c(group1_low_var, group2_low_var),
    group = factor(rep(c("Control", "Treatment"), each = n))
)

# Plot high variance example
p1_var <- ggplot(data_high_var, aes(x = value, fill = group)) +
    geom_density(alpha = 0.6) +
    geom_vline(xintercept = mean1_high_var, linetype = "dashed", color = book_colors$secondary, linewidth = 0.8) +
    geom_vline(xintercept = mean2_high_var, linetype = "dashed", color = book_colors$primary, linewidth = 0.8) +
    scale_fill_manual(
        values = c("Control" = book_colors$secondary, "Treatment" = book_colors$primary),
        name = ""
    ) +
    annotate("label",
        x = mean1_high_var, y = 0.026,
        label = paste0("μ = R$", mean1_high_var),
        color = book_colors$secondary,
        fill = "white",
        size = 5,
        label.size = 0
    ) +
    annotate("label",
        x = mean2_high_var, y = 0.021,
        label = paste0("μ = R$", mean2_high_var),
        color = book_colors$primary,
        fill = "white",
        size = 5,
        label.size = 0
    ) +
    annotate("text",
        x = 150, y = 0.035,
        label = paste0("SD = R$", sd_high_var, "\n(high noise)"),
        color = book_colors$muted,
        size = 5,
        fontface = "italic"
    ) +
    labs(
        title = "High variance: harder to see the R$20 difference",
        x = "Outcome (R$)",
        y = "Density"
    ) +
    scale_y_continuous(limits = c(0, 0.045)) +
    scale_x_continuous(limits = c(20, 200)) +
    theme_book() +
    theme(legend.position = "bottom")

# Plot low variance example
p2_var <- ggplot(data_low_var, aes(x = value, fill = group)) +
    geom_density(alpha = 0.6) +
    geom_vline(xintercept = mean1_low_var, linetype = "dashed", color = book_colors$secondary, linewidth = 0.8) +
    geom_vline(xintercept = mean2_low_var, linetype = "dashed", color = book_colors$primary, linewidth = 0.8) +
    scale_fill_manual(
        values = c("Control" = book_colors$secondary, "Treatment" = book_colors$primary),
        name = ""
    ) +
    annotate("label",
        x = mean1_low_var, y = 0.06,
        label = paste0("μ = R$", mean1_low_var),
        color = book_colors$secondary,
        fill = "white",
        size = 5,
        label.size = 0
    ) +
    annotate("label",
        x = mean2_low_var, y = 0.06,
        label = paste0("μ = R$", mean2_low_var),
        color = book_colors$primary,
        fill = "white",
        size = 5,
        label.size = 0
    ) +
    annotate("text",
        x = 135, y = 0.07,
        label = paste0("SD = R$", sd_low_var, "\n(low noise)"),
        color = book_colors$muted,
        size = 5,
        fontface = "italic"
    ) +
    labs(
        title = "Low variance: the R$20 difference is obvious",
        x = "Outcome (R$)",
        y = "Density"
    ) +
    scale_y_continuous(limits = c(0, 0.085)) +
    scale_x_continuous(limits = c(80, 150)) +
    theme_book() +
    theme(legend.position = "bottom")

# Combine plots side by side
combined_var <- p1_var + p2_var +
    plot_annotation(
        title = "The same R$20 effect, but variance determines detectability",
        subtitle = "Both plots show a treatment that increases spending by R$20",
        caption = "Takeaway: Reducing outcome variance has the same effect as increasing sample size.",
        theme = theme(
            plot.title = element_text(face = "bold", size = 16),
            plot.subtitle = element_text(color = "grey40", size = 12),
            plot.caption = element_text(color = "grey50", size = 10, hjust = 0)
        )
    )

print(combined_var)
# ggsave("images/variance_comparison_r.png", plot = combined_var, width = 14, height = 7, dpi = 300)


#########################################
# Section 5.4.4: Power vs Outcome Variance (Simulation)
# Figure: power_vs_variance.png
#########################################
set.seed(42)

# Simulation function to estimate power empirically
simulate_power <- function(n_total, sd_outcome, prop_treatment = 0.5,
                           true_effect_percent = 5, n_simulations = 1000,
                           significance_level = 0.05) {
    base_value <- 100
    true_effect <- base_value * (true_effect_percent / 100)
    significant_tests <- 0

    for (i in 1:n_simulations) {
        n_control <- round(n_total * (1 - prop_treatment))
        control_data <- rnorm(n_control, mean = base_value, sd = sd_outcome)
        n_treatment <- n_total - n_control
        treatment_data <- rnorm(n_treatment, mean = base_value + true_effect, sd = sd_outcome)
        data <- data.frame(
            outcome = c(control_data, treatment_data),
            treatment = c(rep(0, n_control), rep(1, n_treatment))
        )
        test_result <- t.test(outcome ~ treatment, data = data)
        if (test_result$p.value < significance_level) {
            significant_tests <- significant_tests + 1
        }
    }
    return(significant_tests / n_simulations)
}

# Test different standard deviation levels
sds <- seq(15, 50, by = 5)
power_by_sd <- sapply(sds, function(sd) {
    simulate_power(n_total = 800, sd_outcome = sd, prop_treatment = 0.5)
})

# Create the plot
variance_df <- data.frame(sd = sds, power = power_by_sd)

p_variance <- ggplot(variance_df, aes(x = sd, y = power)) +
    geom_line(color = book_colors$primary, linewidth = 1.2) +
    geom_point(color = book_colors$primary, size = 3) +
    geom_hline(yintercept = 0.8, linetype = "dashed", color = book_colors$success) +
    # Shade regions by interpretability
    annotate("rect",
        xmin = 15, xmax = 25,
        ymin = 0, ymax = 1,
        fill = "#2CA02C", alpha = 0.1
    ) +
    annotate("rect",
        xmin = 35, xmax = 50,
        ymin = 0, ymax = 1,
        fill = "#D62728", alpha = 0.1
    ) +
    annotate("text",
        x = 20, y = 0.15,
        label = "Low variance\n(easy to detect)",
        color = "#2CA02C",
        size = 6,
        fontface = "italic"
    ) +
    annotate("text",
        x = 42.5, y = 0.15,
        label = "High variance\n(hard to detect)",
        color = "#D62728",
        size = 6,
        fontface = "italic"
    ) +
    annotate("text",
        x = 49, y = 0.83,
        label = "80% power",
        color = book_colors$success,
        size = 5,
        hjust = 1,
        fontface = "italic"
    ) +
    labs(
        x = "Standard deviation of outcome",
        y = "Statistical power",
        title = "Higher variance drowns out the signal",
        subtitle = "Simulation: N = 800, 5% true effect, α = 0.05, 50/50 allocation",
        caption = "Takeaway: Reducing variance is often more cost-effective than increasing sample size."
    ) +
    scale_y_continuous(
        limits = c(0, 1),
        labels = scales::percent_format(accuracy = 1)
    ) +
    theme_book() +
    theme(axis.title = element_text(size = 17)) # 20% larger than base (14 * 1.2)

print(p_variance)
# ggsave("images/power_vs_variance.png", p_variance, width = 10, height = 7, dpi = 300)


#########################################
# Section 5.4.5: Compliance effect
# Figure: compliance_effect_r.png and compliance_dillution_r.png
#########################################
set.seed(123)

# Sample sizes
n_control <- 1000
n_treatment <- 1000

# Proportion of compliers in the treatment group
prop_compliers <- 0.1

# Generate outcome variable for control group
y_control <- rnorm(n_control, mean = 50, sd = 10)

# Generate outcome variable for non-compliers (same baseline as control)
# Note: Non-compliers must have the same mean as control to satisfy the
# exclusion restriction — assignment only affects Y through actual treatment.
y_non_compliers <- rnorm(n_treatment * (1 - prop_compliers), mean = 50, sd = 10)

# Generate outcome variable for compliers (shifted significantly)
y_compliers <- rnorm(n_treatment * prop_compliers, mean = 75, sd = 10)

# Combine into a data frame
data_compliance <- data.frame(
    y = c(y_control, y_non_compliers, y_compliers),
    group = factor(
        c(
            rep("Control", n_control),
            rep("Non-compliers", length(y_non_compliers)),
            rep("Compliers", length(y_compliers))
        ),
        levels = c("Control", "Non-compliers", "Compliers")
    )
)


# Estimate LATE (don't worry about that... you will learn about it later)

# Create the variables needed for IV estimation:
# Z = treatment assignment (instrument) - 0 for control, 1 for treatment
# D = actual treatment received (endogenous) - 1 only for compliers
# Y = outcome

data_iv <- data.frame(
    y = c(y_control, y_non_compliers, y_compliers),
    z = c(rep(0, n_control), rep(1, n_treatment)), # Treatment assignment
    d = c(rep(0, n_control), rep(0, length(y_non_compliers)), rep(1, length(y_compliers))) # Actual treatment
)

first_stage <- lm(d ~ z, data = data_iv)
data_iv$d_hat <- fitted(first_stage)

# Second stage: Y ~ D_hat
second_stage <- lm(y ~ d_hat, data = data_iv)
summary(second_stage)

# Calculate means
mean_control_comp <- mean(y_control)
mean_non_compliers <- mean(y_non_compliers)
mean_compliers <- mean(y_compliers)

# Plot 1: Distributions of control, non-compliers, and compliers
p1_comp <- ggplot(data_compliance, aes(x = y, fill = group)) +
    geom_density(alpha = 0.5) +
    geom_vline(xintercept = mean_control_comp, linetype = "dashed", color = book_colors$secondary, linewidth = 0.8) +
    geom_vline(xintercept = mean_non_compliers, linetype = "dashed", color = book_colors$muted, linewidth = 0.8) +
    geom_vline(xintercept = mean_compliers, linetype = "dashed", color = book_colors$primary, linewidth = 0.8) +
    scale_fill_manual(
        values = c(
            "Control" = book_colors$secondary,
            "Non-compliers" = book_colors$muted,
            "Compliers" = book_colors$primary
        ),
        name = "Group"
    ) +
    annotate("label",
        x = mean_control_comp - 6, y = 0.046,
        label = paste0("Control\nμ = R$", round(mean_control_comp, 2)),
        color = book_colors$secondary,
        fill = "white",
        size = 5,
        label.size = 0
    ) +
    annotate("label",
        x = mean_non_compliers + 6, y = 0.042,
        label = paste0("Non-compliers\nμ = R$", round(mean_non_compliers, 2)),
        color = book_colors$muted,
        fill = "white",
        size = 5,
        label.size = 0
    ) +
    annotate("label",
        x = mean_compliers + 6, y = 0.037,
        label = paste0("Compliers\nμ = R$", round(mean_compliers, 2)),
        color = book_colors$primary,
        fill = "white",
        size = 5,
        label.size = 0
    ) +
    # Show the LATE
    annotate("segment",
        x = mean_control_comp, xend = mean_compliers,
        y = 0.055, yend = 0.055,
        arrow = arrow(ends = "both", length = unit(0.15, "cm")),
        color = book_colors$accent, linewidth = 1
    ) +
    annotate("text",
        x = (mean_control_comp + mean_compliers) / 2, y = 0.058,
        label = paste0("Effect on those who used the treatment (LATE): R$", round(mean_compliers - mean_control_comp, 2)),
        color = book_colors$accent,
        size = 5,
        fontface = "bold"
    ) +
    labs(
        title = "The true effect is visible only among compliers",
        subtitle = paste0("Scenario: Only ", prop_compliers * 100, "% of treatment group actually uses the coupon"),
        x = "Outcome (R$)",
        y = "Density",
        caption = "LATE = Local Average Treatment Effect (effect on those who actually comply)"
    ) +
    theme_book() +
    theme(legend.position = "bottom", axis.title = element_text(size = 17))

print(p1_comp)
# ggsave("images/compliance_effect_r.png", p1_comp, width = 13, height = 8, dpi = 300)

# Create combined treatment group
y_treated <- c(y_non_compliers, y_compliers)
mean_treated_comp <- mean(y_treated)

data_combined_comp <- data.frame(
    y = c(y_control, y_treated),
    group = factor(
        c(rep("Control", n_control), rep("Treatment (all assigned)", length(y_treated))),
        levels = c("Control", "Treatment (all assigned)")
    )
)

# Plot 2: Combined treatment vs control (ITT)
p2_comp <- ggplot(data_combined_comp, aes(x = y, fill = group)) +
    geom_density(alpha = 0.5) +
    geom_vline(xintercept = mean_control_comp, linetype = "dashed", color = book_colors$secondary, linewidth = 0.8) +
    geom_vline(xintercept = mean_treated_comp, linetype = "dashed", color = book_colors$primary, linewidth = 0.8) +
    scale_fill_manual(
        values = c(
            "Control" = book_colors$secondary,
            "Treatment (all assigned)" = book_colors$primary
        ),
        name = "Group"
    ) +
    annotate("label",
        x = mean_control_comp - 5, y = 0.04,
        label = paste0("Control\nμ = R$", round(mean_control_comp, 0)),
        color = book_colors$secondary,
        fill = "white",
        size = 5,
        label.size = 0
    ) +
    annotate("label",
        x = mean_treated_comp + 5, y = 0.037,
        label = paste0("Treatment\nμ = R$", round(mean_treated_comp, 0)),
        color = book_colors$primary,
        fill = "white",
        size = 5,
        label.size = 0
    ) +
    # Show the diluted effect
    annotate("segment",
        x = mean_control_comp, xend = mean_treated_comp,
        y = 0.045, yend = 0.045,
        arrow = arrow(ends = "both", length = unit(0.15, "cm")),
        color = book_colors$accent, linewidth = 1
    ) +
    annotate("text",
        x = (mean_control_comp + mean_treated_comp) / 2, y = 0.048,
        label = paste0("Diluted effect (ITT): R$", round(mean_treated_comp - mean_control_comp, 0)),
        color = book_colors$accent,
        size = 6,
        fontface = "bold"
    ) +
    annotate("text",
        x = 80, y = 0.04,
        label = paste0(
            "Effect diluted from R$", round(mean_compliers - mean_control_comp, 0),
            " to R$", round(mean_treated_comp - mean_control_comp, 0),
            "\n(", round((mean_treated_comp - mean_control_comp) / (mean_compliers - mean_control_comp) * 100, 0), "% of true effect)"
        ),
        color = book_colors$muted,
        size = 6,
        fontface = "italic"
    ) +
    labs(
        title = "Non-compliance dilutes the observed effect",
        subtitle = "What a standard A/B test (intent-to-treat) actually measures",
        x = "Outcome (R$)",
        y = "Density",
        caption = "ITT = Intent-to-Treat effect. Mix of compliers (10%) and non-compliers (90%) hides the true signal."
    ) +
    theme_book() +
    theme(legend.position = "bottom", axis.title = element_text(size = 17))

print(p2_comp)
# ggsave("images/compliance_dillution_r.png", p2_comp, width = 12, height = 8, dpi = 300)


#########################################
# Section 5.5: Peeking Simulation - P-value Trajectories
# Figure: pvalue-trajectories.png and peeking-false-positive.png
#########################################
set.seed(123)

# Parameters for the simulation
n_experiments <- 100
max_sample_size <- 500
check_interval <- 20
significance_level <- 0.05

# Storage for results
pvalue_trajectories <- list()

# Run simulations
for (exp in 1:n_experiments) {
    control_full <- rnorm(max_sample_size, mean = 100, sd = 20)
    treatment_full <- rnorm(max_sample_size, mean = 100, sd = 20)
    check_points <- seq(check_interval, max_sample_size, by = check_interval)
    pvalues <- numeric(length(check_points))

    for (i in seq_along(check_points)) {
        n <- check_points[i]
        test_result <- t.test(control_full[1:n], treatment_full[1:n])
        pvalues[i] <- test_result$p.value
    }

    pvalue_trajectories[[exp]] <- data.frame(
        sample_size = check_points,
        pvalue = pvalues,
        experiment = exp
    )
}

# Combine all trajectories
all_trajectories <- do.call(rbind, pvalue_trajectories)

# Identify which experiments ever crossed the threshold
crossing_exps <- all_trajectories %>%
    group_by(experiment) %>%
    summarize(ever_significant = any(pvalue < 0.05)) %>%
    filter(ever_significant) %>%
    pull(experiment)

non_crossing_exps <- setdiff(1:n_experiments, crossing_exps)

# Select 3 representative experiments for the plot:
# 2 that crossed EARLY (small sample size) + 1 that crossed LATE
# This illustrates both the volatility at small n and the any-stage risk

# Find the first crossing point for each experiment that ever crossed
first_crossing <- all_trajectories %>%
    filter(experiment %in% crossing_exps, pvalue < 0.05) %>%
    group_by(experiment) %>%
    summarize(first_cross_n = min(sample_size)) %>%
    arrange(first_cross_n)

# Select 2 early crossers (first 40% of sample sizes) and 1 late crosser (last 40%)
early_threshold <- quantile(first_crossing$first_cross_n, 0.4)
late_threshold <- quantile(first_crossing$first_cross_n, 0.6)

early_crossers <- first_crossing %>%
    filter(first_cross_n <= early_threshold) %>%
    pull(experiment)
late_crossers <- first_crossing %>%
    filter(first_cross_n >= late_threshold) %>%
    pull(experiment)

set.seed(42) # For reproducibility
selected_early <- if (length(early_crossers) >= 2) sample(early_crossers, 2) else early_crossers
selected_late <- if (length(late_crossers) >= 1) sample(late_crossers, 1) else tail(first_crossing$experiment, 1)
selected_experiments <- c(selected_early, selected_late)

# Filter to only selected experiments for the plot
plot_trajectories <- all_trajectories %>%
    filter(experiment %in% selected_experiments) %>%
    mutate(
        experiment_label = case_when(
            experiment == selected_experiments[1] ~ "Early cross #1",
            experiment == selected_experiments[2] ~ "Early cross #2",
            experiment == selected_experiments[3] ~ "Late cross"
        ),
        experiment_label = factor(experiment_label, levels = c("Early cross #1", "Early cross #2", "Late cross"))
    )

# Define distinct, highly distinguishable colors for each trajectory
trajectory_colors <- c(
    "Early cross #1" = "#E64A19", # Deep orange
    "Early cross #2" = "#00897B", # Teal
    "Late cross" = "#7B1FA2" # Deep purple
)

# Plot 1: P-value trajectories (3 representative experiments with distinct colors)
p_trajectories <- ggplot(plot_trajectories, aes(x = sample_size, y = pvalue, group = experiment)) +
    # Red shaded area for false positive zone
    annotate("rect",
        xmin = -Inf, xmax = Inf,
        ymin = 0, ymax = 0.05,
        fill = "#D62728", alpha = 0.15
    ) +
    geom_line(aes(color = experiment_label), linewidth = 1.4, alpha = 0.7) +
    geom_point(aes(color = experiment_label), size = 2.5, alpha = 0.5) +
    geom_hline(yintercept = 0.05, linetype = "dashed", color = "#D62728", linewidth = 1) +
    scale_color_manual(
        values = trajectory_colors,
        name = "Trajectory"
    ) +
    annotate("label",
        x = max_sample_size, y = 0.10,
        label = "α = 0.05 threshold",
        color = "#D62728",
        fill = "white",
        size = 5,
        hjust = 1,
        label.size = 0
    ) +
    annotate("text",
        x = 300, y = 0.00,
        label = "Zone of false positives",
        color = "#D62728",
        size = 5,
        fontface = "bold",
        hjust = 0
    ) +
    labs(
        title = "This is statistics: p-values wander even with no true effect",
        subtitle = "Here are 3 simulated A/A tests out of the 1000 I run: \n 2 cross early (volatile small samples), 1 crosses late (any-stage risk)",
        x = "Sample size per group",
        y = "P-value",
        caption = paste0("In full simulation: ", length(crossing_exps), " of ", n_experiments, " experiments crossed the threshold (", n_experiments - length(crossing_exps), " never crossed)\nTakeaway: Peeking can fool you at any stage — the cumulative risk grows with each look.")
    ) +
    scale_y_continuous(limits = c(0, 1)) +
    theme_book() +
    theme(
        legend.position = "top",
        axis.title = element_text(size = 17) # 20% larger than base (14 * 1.2)
    )

print(p_trajectories)
# ggsave("images/pvalue-trajectories.png", p_trajectories, width = 10, height = 7, dpi = 300)

# Calculate cumulative false positive rate
check_points <- seq(check_interval, max_sample_size, by = check_interval)
cumulative_fp <- numeric(length(check_points))

for (i in seq_along(check_points)) {
    fp_count <- 0
    for (exp in 1:n_experiments) {
        traj <- pvalue_trajectories[[exp]]
        if (any(traj$pvalue[1:i] < significance_level)) {
            fp_count <- fp_count + 1
        }
    }
    cumulative_fp[i] <- fp_count / n_experiments
}

fp_df <- data.frame(
    sample_size = check_points,
    fp_rate = cumulative_fp
)

final_fp_rate <- tail(fp_df$fp_rate, 1)

# Plot 2: Cumulative false positive rate
p_fp <- ggplot(fp_df, aes(x = sample_size, y = fp_rate)) +
    # Shade the nominal 5% area in gray
    annotate("rect",
        xmin = -Inf, xmax = Inf,
        ymin = 0, ymax = 0.05,
        fill = book_colors$light_gray, alpha = 0.5
    ) +
    # Shade the "inflation" area
    geom_ribbon(aes(ymin = 0.05, ymax = fp_rate), fill = book_colors$accent, alpha = 0.2) +
    geom_line(color = book_colors$accent, linewidth = 1.5) +
    geom_point(color = book_colors$accent, size = 2) +
    geom_hline(yintercept = 0.05, linetype = "dashed", color = book_colors$muted, linewidth = 0.8) +
    annotate("text",
        x = 100, y = 0.02,
        label = "Nominal 5% (if you don't peek)",
        color = book_colors$muted,
        size = 6,
        hjust = 0
    ) +
    annotate("label",
        x = max_sample_size - 20, y = final_fp_rate - 0.02,
        label = paste0(
            "Actual: ", scales::percent(final_fp_rate, accuracy = 1), "\n(",
            round(final_fp_rate / 0.05, 1), "x inflation)"
        ),
        color = book_colors$accent,
        fill = "white",
        size = 6,
        fontface = "bold",
        hjust = 1,
        label.size = 0
    ) +
    labs(
        title = "The rate of false positives grows with each peek",
        subtitle = paste0("Simulation of ", n_experiments, " A/A tests with ", max_sample_size / check_interval, " peeks each"),
        x = "Sample size per group",
        y = "Cumulative false positive rate",
        caption = "Takeaway: Pre-commit to your sample size, or use sequential testing methods that control for peeking."
    ) +
    scale_y_continuous(
        limits = c(0, 0.35),
        labels = scales::percent_format(accuracy = 1)
    ) +
    theme_book() +
    theme(axis.title = element_text(size = 17)) # 20% larger than base (14 * 1.2)

print(p_fp)
# ggsave("images/peeking-false-positive.png", p_fp, width = 10, height = 7, dpi = 300)


#########################################
# Section 5.5: Ignoring multiple testing
# Figure: multiple-comparisons.png
#########################################
# Set up the range for the number of metrics (hypotheses tested)
n_metrics <- 1:50
alpha <- 0.05

# Calculate the probability of at least one false positive
prob_fp <- 1 - (1 - alpha)^n_metrics

# Create a data frame for plotting
multiple_df <- data.frame(
    n_metrics = n_metrics,
    prob_fp = prob_fp
)

# Key points to highlight
key_points <- data.frame(
    n = c(10, 20, 50),
    prob = 1 - (1 - alpha)^c(10, 20, 50)
)

# Generate the plot
p_multiple <- ggplot(multiple_df, aes(x = n_metrics, y = prob_fp)) +
    # Shade the nominal 5% area in gray
    annotate("rect",
        xmin = -Inf, xmax = Inf,
        ymin = 0, ymax = 0.05,
        fill = book_colors$light_gray, alpha = 0.5
    ) +
    geom_ribbon(aes(ymin = 0.05, ymax = prob_fp), fill = book_colors$accent, alpha = 0.2) +
    geom_line(color = book_colors$accent, linewidth = 1.2) +
    geom_hline(yintercept = 0.05, linetype = "dashed", color = book_colors$muted) +
    geom_point(data = key_points, aes(x = n, y = prob), color = book_colors$accent, size = 4) +
    annotate("label",
        x = 10, y = key_points$prob[1] + 0.08,
        label = paste0("10 metrics:\n", scales::percent(key_points$prob[1], accuracy = 1)),
        fill = "white",
        size = 5,
        label.size = 0
    ) +
    annotate("label",
        x = 20, y = key_points$prob[2] + 0.08,
        label = paste0("20 metrics:\n", scales::percent(key_points$prob[2], accuracy = 1)),
        fill = "white",
        size = 5,
        label.size = 0
    ) +
    annotate("label",
        x = 48, y = key_points$prob[3] - 0.09,
        label = paste0("50 metrics:\n", scales::percent(key_points$prob[3], accuracy = 1)),
        fill = "white",
        size = 5,
        label.size = 0
    ) +
    annotate("text",
        x = 5, y = 0.02,
        label = "Nominal α = 5%",
        color = book_colors$muted,
        size = 5,
        fontface = "italic"
    ) +
    scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(0, 1)) +
    labs(
        title = "The multiple comparisons self deception",
        subtitle = "Probability of finding at least one 'significant' result by chance alone",
        x = "Number of independent metrics tested",
        y = "Probability of ≥ 1 false positive",
        caption = "Takeaway: Testing 20 metrics at α = 0.05 gives you a 64% chance of a spurious 'discovery'."
    ) +
    theme_book() +
    theme(axis.title = element_text(size = 17)) # 20% larger than base (14 * 1.2)

print(p_multiple)
# ggsave("images/multiple-comparisons.png", plot = p_multiple, width = 10, height = 7, dpi = 300)
