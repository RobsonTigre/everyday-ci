##########################################################
## Everyday Causal Inference: How to estimate, test, and explain impacts with R and Python
## Copyright © 2025 by Robson Tigre. All rights reserved.
## You may read, share, and cite for learning purposes, provided you credit the source.
## It should not be used to create competing educational or commercial products
##########################################################
## Code for Chapter 4 - Experiments I: designing and running reliable experiments
## Created: Dec 22, 2025
## Last modified: Dec 22, 2025
##########################################################

#########################################
# Ignoring multiple testing
#########################################
library(ggplot2)
library(scales)
library(dplyr)

# Set up the range for the number of metrics (hypotheses tested)
n_metrics <- 1:50
alpha <- 0.05

# Calculate the probability of at least one false positive
# Formula: P(FP >= 1) = 1 - (1 - alpha)^n
prob_fp <- 1 - (1 - alpha)^n_metrics

# Create a data frame for plotting
data <- data.frame(
    n_metrics = n_metrics,
    prob_fp = prob_fp
)

# Generate the plot
p <- ggplot(data, aes(x = n_metrics, y = prob_fp)) +
    geom_line(color = "#E69F00", size = 1.2) +
    geom_hline(yintercept = 0.64, linetype = "dashed", color = "gray50") +
    annotate("text", x = 25, y = 0.60, label = "64% chance with 20 metrics", color = "gray30") +
    scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(0, 1)) +
    labs(
        title = "The Multiple Comparison Trap",
        subtitle = "Probability of finding at least one 'significant' result by chance (False Positive)",
        x = "Number of independent metrics tested",
        y = "Probability of ≥ 1 False Positive"
    ) +
    theme_minimal(base_size = 14) +
    theme(
        panel.grid.minor = element_blank(),
        plot.title = element_text(face = "bold"),
        plot.subtitle = element_text(color = "gray30")
    )

# Save the plot
ggsave("images/multiple-comparisons.png", plot = p, width = 8, height = 5, dpi = 300)
