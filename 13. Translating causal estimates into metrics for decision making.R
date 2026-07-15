##########################################################
## Everyday Causal Inference: How to estimate, test, and explain impacts with R and Python
## www.everydaycausal.com
## Copyright (c) 2025 by Robson Tigre. All rights reserved.
## You may read, run, adapt, and cite this code, provided you credit the source.
## It should not be used to create competing educational or commercial products
##########################################################
## Code for Chapter 13 - Translating causal estimates into metrics of business value
## Created: Mar 06, 2026
## Last modified: 2026-05-07
##########################################################

# ==========================================================
# SETUP
# ==========================================================
# install once: install.packages(c("tidyverse", "scales"))
library(tidyverse)
library(scales)

# Book-wide color palette (colorblind-friendly, consistent across all chapters)
book_colors <- list(
    primary    = "#2E86AB", # Steel blue
    secondary  = "#A23B72", # Magenta
    accent     = "#F18F01", # Orange
    success    = "#C73E1D", # Red-orange
    muted      = "#6C757D", # Gray
    light_gray = "grey90",
    dark_gray  = "grey30"
)

theme_book <- function(base_size = 14) {
    theme_minimal(base_size = base_size) +
        theme(
            plot.title = element_text(
                face = "bold", size = base_size * 1.3,
                color = "grey20", margin = margin(b = 5)
            ),
            plot.subtitle = element_text(
                size = base_size * 0.9, color = "grey40",
                margin = margin(b = 15)
            ),
            plot.caption = element_text(
                size = base_size * 0.7, color = "grey50",
                hjust = 0, margin = margin(t = 10)
            ),
            axis.title = element_text(size = base_size, color = "grey30"),
            axis.text = element_text(size = base_size * 0.85, color = "grey40"),
            panel.grid.major = element_line(color = "grey90", linewidth = 0.5),
            panel.grid.minor = element_blank(),
            legend.position = "bottom",
            plot.margin = margin(20, 20, 20, 20)
        )
}


# ==========================================================
# BLOCK 1: SCALING -- FROM STATISTICAL UNITS TO FINANCIAL UNITS
# (Chapter section: "From statistical units to financial units")
# ==========================================================
# Multiply per-user effect by the eligible user base, then by the projection
# horizon, to get the naive starting number that the rest of the pipeline
# discounts down. R$2.34 x 5M MAU x 12 months ~= R$140M annual.

ate <- 2.34 # R$ per user per 30 days (from Chapter 3)
mau <- 5e6 # monthly active users
monthly_impact <- ate * mau
annual_naive <- monthly_impact * 12

cat(sprintf("Per-user effect: R$%.2f/month\n", ate))
cat(sprintf("MAU: %s\n", format(mau, big.mark = ",")))
cat(sprintf(
    "Projected monthly impact: R$%s\n",
    format(monthly_impact, big.mark = ",")
))
cat(sprintf(
    "Naive annual projection: R$%s\n",
    format(annual_naive, big.mark = ",")
))


# ==========================================================
# BLOCK 2: CLV UPLIFT WITH MONTHLY DECAY (headline calculation, r = 0)
# (Chapter section: "Decay over time: customer lifetime value uplift")
# ==========================================================
# Decay only, no time-value-of-money discount. Using r = 0.01 (illustrative
# monthly WACC, ~12% annualized) lowers the result to ~R$16.10; over a 12-month
# horizon that difference is small next to the decay assumption itself, so the
# chapter shows the headline at r = 0 for clarity.

ate_month_0 <- 2.34 # R$ per user, month 0
effect_persistence <- 0.90 # monthly persistence; effect decays 10% per period
horizon <- 12 # months

months <- 0:(horizon - 1)
effects <- ate_month_0 * effect_persistence^months
clv_uplift <- sum(effects)

cat(sprintf("Month 0 effect: R$%.2f\n", effects[1]))
cat(sprintf("Month 6 effect: R$%.2f\n", effects[7]))
cat(sprintf("Month 11 effect: R$%.2f\n", effects[12]))
cat(sprintf("CLV uplift (12 months): R$%.2f per user\n", clv_uplift))
cat(sprintf("Without decay: R$%.2f per user\n", ate_month_0 * 12))


# ==========================================================
# FIGURE: clv-decay-vs-naive.png   (@fig-clv-decay)
# (Chapter section: "Decay over time: customer lifetime value uplift")
# ==========================================================
# Shows the optimism-bias gap between the naive flat path (R$2.34/month) and
# the decayed path (lambda = 0.90). The shaded ribbon is the cumulative
# overstatement that the naive forecast would build into the board deck.

naive_path <- rep(ate_month_0, horizon)
cum_decayed <- cumsum(effects)
cum_naive <- cumsum(naive_path)

clv_df <- tibble(
    month  = rep(months, 2),
    effect = c(effects, naive_path),
    type   = rep(c("With 10% monthly decay", "Naive (no decay)"), each = horizon)
)

p_clv <- ggplot(clv_df, aes(x = month, y = effect, color = type, linetype = type)) +
    geom_ribbon(
        data = tibble(month = months, ymin = effects, ymax = naive_path),
        aes(x = month, ymin = ymin, ymax = ymax),
        inherit.aes = FALSE,
        fill = book_colors$accent, alpha = 0.15
    ) +
    geom_line(linewidth = 1.2) +
    geom_point(size = 2.5) +
    annotate("text",
        x = 7, y = (ate_month_0 + effects[8]) / 2,
        label = "Optimism\nbias", color = book_colors$accent,
        fontface = "italic", size = 4.5
    ) +
    annotate("text",
        x = 11, y = naive_path[12] + 0.15,
        label = sprintf("R$%.2f", ate_month_0),
        color = book_colors$muted, size = 3.5
    ) +
    annotate("text",
        x = 11, y = effects[12] - 0.15,
        label = sprintf("R$%.2f", effects[12]),
        color = book_colors$primary, size = 3.5
    ) +
    scale_color_manual(values = c(
        "Naive (no decay)" = book_colors$muted,
        "With 10% monthly decay" = book_colors$primary
    )) +
    scale_linetype_manual(values = c(
        "Naive (no decay)" = "dashed",
        "With 10% monthly decay" = "solid"
    )) +
    scale_x_continuous(breaks = 0:11, labels = 1:12) +
    scale_y_continuous(labels = function(x) paste0("R$", formatC(x, format = "f", digits = 2))) +
    labs(
        x = "Month",
        y = "Monthly effect per user",
        title = "Optimism bias: naive vs. decayed effect projection",
        subtitle = sprintf(
            "CLV uplift: R$%.2f (with decay) vs. R$%.2f (naive) over 12 months",
            tail(cum_decayed, 1), tail(cum_naive, 1)
        ),
        color = NULL, linetype = NULL
    ) +
    theme_book() +
    theme(
        legend.position = c(0.75, 0.85),
        legend.background = element_rect(fill = "white", color = NA)
    )

ggsave("images/clv-decay-vs-naive.png", p_clv,
    width = 9, height = 5.5, dpi = 300, bg = "white"
)
cat("Saved: images/clv-decay-vs-naive.png\n")


# ==========================================================
# BLOCK 3: ROI -- PERSONALIZED FEED
# (Chapter section: "Return on investment (ROI)")
# ==========================================================
# feed_12m_profit is the 12-month cumulative profit (post-decay,
# pre-investment) -- NOT an annual run-rate. profit_30d from @sec-protocol is
# already net of variable costs; the code below only subtracts the fixed
# investment to get net profit.
clv_per_user <- sum(2.34 * 0.90^(0:11)) # ~R$17/user (decay only; omitting the ~1% monthly discount overstates by ~4%)
feed_12m_profit <- clv_per_user * 5e6 # ~R$84M: 12-month cumulative profit (variable costs already netted)
feed_investment <- 4e6 # R$4M fixed investment (build + maintain)
feed_roi <- (feed_12m_profit - feed_investment) / feed_investment # = Incremental Net Profit / Investment

cat(sprintf("\nCLV uplift per user: R$%.2f\n", clv_per_user))
cat(sprintf(
    "12-month profit (post-decay): R$%sM\n",
    format(round(feed_12m_profit / 1e6), big.mark = ",")
))
cat(sprintf(
    "Incremental net profit: R$%sM\n",
    format(round((feed_12m_profit - feed_investment) / 1e6), big.mark = ",")
))
cat(sprintf("ROI: %.0fx annual return\n", feed_roi))


# ==========================================================
# FIGURE: diminishing-returns-saturation.png   (@fig-diminishing-returns)
# (Chapter section: "Diminishing returns")
# ==========================================================
# Illustrative ad-spend saturation curve: revenue grows logarithmically with
# spend, so each additional Real buys less revenue than the one before.

set.seed(42)
spend_levels <- seq(100, 5000, by = 100) # R$ thousands
true_revenue <- 800 * log(spend_levels / 100 + 1) + rnorm(length(spend_levels), 0, 50)
ad_data <- tibble(spend = spend_levels, revenue = true_revenue)

p_sat <- ggplot(ad_data, aes(x = spend, y = revenue)) +
    geom_point(color = book_colors$primary, alpha = 0.6, size = 2) +
    geom_smooth(
        method = "lm", formula = y ~ log(x),
        color = book_colors$secondary, linewidth = 1.2, se = FALSE
    ) +
    annotate("text",
        x = 2300, y = max(true_revenue) * 0.22,
        label = "Each extra Real buys\nless revenue\nthan the one before",
        color = book_colors$accent,
        hjust = 0, fontface = "italic", size = 4.5
    ) +
    scale_x_continuous(
        breaks = c(500, 1000, 1500, 2500, 5000),
        labels = c("R$0.5M", "R$1M", "R$1.5M", "R$2.5M", "R$5M")
    ) +
    labs(
        x = "Cumulative ad spend",
        y = "Incremental revenue (R$ thousands)",
        title = "The saturation curve: more spend, less bang per buck",
        caption = "Simulated data for illustration"
    ) +
    theme_book()

ggsave("images/diminishing-returns-saturation.png", p_sat,
    width = 9, height = 5.5, dpi = 300, bg = "white"
)
cat("Saved: images/diminishing-returns-saturation.png\n")


# ==========================================================
# FIGURE: diminishing-returns-squeeze.png   (@fig-squeeze)
# (Chapter section: "Diminishing returns")
# ==========================================================
# Marginal revenue falls while marginal cost rises -- the margin between the
# two curves shrinks and eventually inverts. Crossover marks where marginal
# ROI turns negative.

set.seed(42)
n_sq <- 500
spend_sq <- seq(0.05, 2.0, length.out = n_sq) # R$ millions
mr_sq <- 8.0 * exp(-2.0 * spend_sq) + 0.4
mc_sq <- 1.2 + 3.5 * spend_sq^2.2

squeeze_df <- tibble(
    spend = spend_sq,
    marginal_revenue = mr_sq,
    marginal_cost = mc_sq
)

# Find crossover by linear interpolation
diff_val <- mr_sq - mc_sq
cross_idx <- which(diff(sign(diff_val)) != 0)[1]
x1 <- spend_sq[cross_idx]
x2 <- spend_sq[cross_idx + 1]
y1 <- diff_val[cross_idx]
y2 <- diff_val[cross_idx + 1]
cross_x <- x1 - y1 * (x2 - x1) / (y2 - y1)
cross_y <- 8.0 * exp(-2.0 * cross_x) + 0.4

shade_df <- squeeze_df |> filter(marginal_revenue >= marginal_cost)

healthy_x <- 0.20
healthy_mr <- 8.0 * exp(-2.0 * healthy_x) + 0.4
healthy_mc <- 1.2 + 3.5 * healthy_x^2.2

neg_label_x <- 0.80
neg_label_y <- 8.6

p_squeeze <- ggplot(squeeze_df, aes(x = spend)) +
    geom_ribbon(
        data = shade_df,
        aes(ymin = marginal_cost, ymax = marginal_revenue),
        fill = book_colors$primary, alpha = 0.08
    ) +
    geom_line(aes(y = marginal_revenue, color = "Marginal revenue"), linewidth = 1.2) +
    geom_line(aes(y = marginal_cost, color = "Marginal cost"), linewidth = 1.2) +
    annotate("point",
        x = cross_x, y = cross_y,
        size = 3.5, color = book_colors$accent, shape = 16
    ) +
    annotate("segment",
        x = neg_label_x + 0.02, xend = cross_x + 0.03,
        y = 7.4, yend = cross_y + 0.18,
        color = book_colors$accent, linewidth = 0.5,
        arrow = arrow(length = unit(0.08, "inches"), type = "open")
    ) +
    annotate("text",
        x = neg_label_x, y = neg_label_y,
        label = "Point at which\nmarginal ROI\nturns negative",
        size = 4.0, fontface = "bold",
        color = book_colors$accent,
        hjust = 0, vjust = 0.5
    ) +
    annotate("text",
        x = healthy_x + 0.05, y = 2.8,
        label = "Healthy\nmargin",
        size = 3.5, fontface = "bold",
        color = book_colors$primary,
        hjust = 0, vjust = 0.5
    ) +
    annotate("segment",
        x = healthy_x, xend = healthy_x,
        y = healthy_mc + 0.15, yend = healthy_mr - 0.15,
        color = book_colors$primary, linewidth = 0.6,
        arrow = arrow(
            ends = "both", length = unit(0.08, "inches"),
            type = "open"
        )
    ) +
    scale_color_manual(values = c(
        "Marginal revenue" = book_colors$primary,
        "Marginal cost" = book_colors$secondary
    )) +
    scale_x_continuous(
        breaks = c(0.1, 0.5, 1.0, 1.5, 2.0),
        labels = paste0("R$", c(0.1, 0.5, 1.0, 1.5, 2.0), "M"),
        expand = expansion(mult = c(0.02, 0.04))
    ) +
    scale_y_continuous(expand = expansion(mult = c(0.02, 0.08))) +
    coord_cartesian(ylim = c(0, 10)) +
    labs(
        title = "The double squeeze: revenue falls, costs rise",
        subtitle = "At scale, the margin between marginal revenue and marginal cost narrows — then inverts",
        caption = "Simulated data for illustration",
        x = "Cumulative ad spend",
        y = "R$ per additional R$1K spent",
        color = NULL
    ) +
    theme_book() +
    theme(legend.title = element_blank())

ggsave("images/diminishing-returns-squeeze.png", p_squeeze,
    width = 9, height = 5.5, dpi = 300, bg = "white"
)
cat(sprintf("Saved: images/diminishing-returns-squeeze.png  (crossover at R$%.2fM)\n", cross_x))


# ==========================================================
# FIGURE: waterfall-feed.png   (@fig-waterfall-feed)
# (Chapter section: "How much of the impact survives rollout")
# ==========================================================
# Numbers match the chapter waterfall table:
#   Naive: R$2.34 x 5M x 12 = R$140M
#   Decay (-40%):              140 x 0.60 = 84
#   Adoption ramp (-10% of 84): 84 x 0.90 = 75.6 ~= 76
#   Representativeness (-5% of 76): 76 x 0.95 = 72.2 ~= 72
#   Cannibal./SUTVA (-3% of 72):    72 x 0.97 = 69.8 ~= 70

build_waterfall <- function(labels, totals, title, subtitle,
                            unit_label = "R$ millions",
                            label_fn = function(x) paste0("R$", round(x), "M")) {
    n <- length(labels)
    deductions <- c(0, diff(totals))
    ymin <- numeric(n)
    ymax <- numeric(n)
    for (i in seq_len(n)) {
        if (i == 1 || i == n) {
            ymin[i] <- 0
            ymax[i] <- totals[i]
        } else {
            ymin[i] <- totals[i]
            ymax[i] <- totals[i - 1]
        }
    }
    bar_type <- rep("deduction", n)
    bar_type[1] <- "start"
    bar_type[n] <- "end"

    df <- tibble(
        label = factor(labels, levels = labels),
        total = totals,
        deduction = deductions,
        ymin = ymin,
        ymax = ymax,
        bar_type = bar_type
    )

    connectors <- tibble(
        x    = seq(1.45, n - 0.55, by = 1),
        xend = seq(1.55, n - 0.45, by = 1),
        y    = head(totals, -1),
        yend = head(totals, -1)
    )
    connectors <- connectors[1:(n - 2), ]

    max_val <- max(totals)
    thin_threshold <- max_val * 0.08
    df <- df |>
        mutate(
            bar_label = ifelse(
                bar_type == "deduction",
                paste0("−", label_fn(abs(deduction))),
                label_fn(total)
            ),
            bar_height = abs(ymax - ymin),
            is_thin = bar_height < thin_threshold,
            label_y = case_when(
                bar_type != "deduction" ~ ymax / 2,
                is_thin ~ ymin - max_val * 0.03,
                TRUE ~ (ymin + ymax) / 2
            ),
            label_color = ifelse(is_thin, book_colors$dark_gray, "white")
        )

    ggplot(df) +
        geom_rect(
            aes(
                xmin = as.numeric(label) - 0.4,
                xmax = as.numeric(label) + 0.4,
                ymin = ymin, ymax = ymax,
                fill = bar_type
            ),
            color = "white", linewidth = 0.3
        ) +
        geom_segment(
            data = connectors,
            aes(x = x, xend = xend, y = y, yend = yend),
            color = book_colors$muted, linewidth = 0.5, linetype = "dotted"
        ) +
        geom_text(aes(x = as.numeric(label), y = label_y, label = bar_label),
            color = df$label_color, fontface = "bold", size = 3.8
        ) +
        scale_fill_manual(
            values = c(
                "start" = book_colors$primary,
                "deduction" = book_colors$success,
                "end" = book_colors$primary
            ),
            guide = "none"
        ) +
        scale_x_continuous(
            breaks = seq_len(n), labels = labels,
            expand = expansion(mult = c(0.02, 0.02))
        ) +
        scale_y_continuous(
            labels = function(x) paste0("R$", x, "M"),
            expand = expansion(mult = c(0, 0.05))
        ) +
        labs(title = title, subtitle = subtitle, x = NULL, y = unit_label) +
        theme_book() +
        theme(
            axis.text.x = element_text(angle = 25, hjust = 1, size = 11),
            panel.grid.major.x = element_blank()
        )
}

feed_labels <- c(
    "Naive\nprojection",
    "Effect\ndecay",
    "Adoption\nramp",
    "Represent-\nativeness",
    "Cannibal. /\nSUTVA",
    "Realistic\nprojection"
)
feed_totals <- c(140, 84, 76, 72, 70, 70)

p_feed <- build_waterfall(
    labels   = feed_labels,
    totals   = feed_totals,
    title    = "The projection waterfall: personalized feed",
    subtitle = "Each discount peels back a layer of optimism from the naive R$140M"
)

ggsave("images/waterfall-feed.png", p_feed,
    width = 10, height = 6, dpi = 300, bg = "white"
)
cat("Saved: images/waterfall-feed.png\n")


# ==========================================================
# BLOCK 4: SENSITIVITY ANALYSIS
# (Chapter section: "Sensitivity analysis")
# ==========================================================
# Sensitivity analysis: personalized feed ROI under different assumptions.
# profit_12m below is the 12-month cumulative profit (post-decay,
# pre-investment); ROI subtracts fixed investment in the numerator.
scenarios <- expand.grid(
    investment = c(4e6, 6e6, 8e6), # R$4M, R$6M, R$8M fixed build investment
    decay      = c(0.00, 0.10, 0.20),
    mau        = c(3e6, 5e6, 7e6) # 3M, 5M, 7M users
)

scenarios <- scenarios |>
    mutate(
        profit_12m = sapply(decay, function(d) sum(2.34 * (1 - d)^(0:11))) * mau,
        roi = (profit_12m - investment) / investment
    )

# Show a focused slice: vary investment and decay at 5M MAU
focus <- scenarios |>
    filter(mau == 5e6) |> # 5M MAU slice
    mutate(
        investment_label = paste0("R$", investment / 1e6, "M"),
        decay_label = paste0(decay * 100, "% decay"),
        roi_label = paste0(round(roi), "x")
    ) |>
    select(investment_label, decay_label, roi_label) |>
    pivot_wider(names_from = decay_label, values_from = roi_label)

cat("\nROI sensitivity (5M MAU, varying fixed investment and monthly decay):\n")
print(as.data.frame(focus), row.names = FALSE)


# ==========================================================
# BLOCK 5: PRE-EXPERIMENT ROI PROJECTION FROM MEI-CALIBRATED MDE
# (Chapter section: "From breakeven to experiment design")
# ==========================================================
# profit_12m_mde below is 12-month cumulative profit (post-decay,
# pre-investment), assuming the true effect equals the MEI (which is also the MDE here).
mde <- 0.80 # R$/user/month (stakeholder MEI used as target MDE)
mau <- 5e6 # 5M monthly active users
investment <- 4e6 # R$4M build + maintain
decay <- 0.10 # 10% monthly decay
horizon <- 12 # months

# Breakeven effect: minimum per-user-per-month lift that covers investment over decayed horizon
effective_months <- sum((1 - decay)^(0:(horizon - 1)))
breakeven_effect <- investment / (mau * effective_months)

# Projected 12-month cumulative profit if true effect = MEI (conservative)
profit_12m_mde <- sum(mde * (1 - decay)^(0:(horizon - 1))) * mau # with decay

roi_mde <- (profit_12m_mde - investment) / investment

cat(sprintf("\nBreakeven effect: R$%.2f/user/month\n", breakeven_effect))
cat(sprintf("MEI (= target MDE): R$%.2f/user/month\n", mde))
cat(sprintf("MEI to breakeven ratio: %.1fx above breakeven\n\n", mde / breakeven_effect))

# If the true effect equals the MEI (conservative), what does the project look like?
cat(sprintf(
    "  12-month profit (with decay): R$%sM\n",
    format(round(profit_12m_mde / 1e6), big.mark = ",")
))
cat(sprintf("  ROI: %.0fx\n", roi_mde))


# ==========================================================
# BLOCK 6: CONFIDENCE INTERVAL -> FINANCIAL RANGE
# (Chapter section: "The p-value trap and ROI confidence intervals")
# ==========================================================
# Plug-in method: re-run the financial formula at each CI bound. ROI is a
# strictly monotone function of the effect, so this preserves coverage.
feed_ci <- c(1.75, 2.93) # 95% CI from Chapter 3
clv_factor <- sum(0.90^(0:11)) # 12-month decay multiplier (~7.18)
feed_profit_12m <- feed_ci * 5e6 * clv_factor # 5M MAU
feed_roi_ci <- (feed_profit_12m - 4e6) / 4e6 # ROI at each bound

# Confidence interval -> financial range
cat(sprintf("Effect CI: [R$%.2f, R$%.2f] per user/month\n", feed_ci[1], feed_ci[2]))
cat(sprintf(
    "12-month profit range (with 10%% decay): [R$%.0fM, R$%.0fM]\n",
    feed_profit_12m[1] / 1e6, feed_profit_12m[2] / 1e6
))
cat(sprintf("ROI range against R$4M cost: %.0fx to %.0fx\n", feed_roi_ci[1], feed_roi_ci[2]))


# ==========================================================
# BLOCK 7: ESTIMATING lambda FROM HOLDOUT DATA
# (Chapter section: "Appendix 13.B -- Effect decay is predictable")
# ==========================================================
# Simulate a five-month holdout where the true effect-persistence factor is
# lambda = 0.88 and recover the estimate from a simple log-linear regression.
# This is the template you'd use with real holdout data -- replace the
# simulated values with your observed per-period effects.
set.seed(42)
effect_0 <- 2.34 # initial per-user effect (R$/month)
lambda_true <- 0.88 # true monthly effect persistence
months <- 0:4 # holdout measurements at months 0-4

# Simulate observed effects: true decay + measurement noise
log_effects <- log(effect_0) + months * log(lambda_true) +
    rnorm(5, mean = 0, sd = 0.05)

# Regress log(effect) on time to recover lambda
fit <- lm(log_effects ~ months)
slope <- coef(fit)["months"]
se <- summary(fit)$coefficients["months", "Std. Error"]

# Estimated lambda and 95% CI
lambda_hat <- exp(slope)
t_crit <- qt(0.975, df = length(months) - 2)
ci_lo <- exp(slope - t_crit * se)
ci_hi <- exp(slope + t_crit * se)

# Assumed lambda was 0.90 (the chapter's working assumption); compare to the estimate below
cat(sprintf(
    "Estimated lambda: %.2f  [95%% CI: %.2f, %.2f]\n",
    lambda_hat, ci_lo, ci_hi
))
