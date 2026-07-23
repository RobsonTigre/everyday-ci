##########################################################
## Everyday Causal Inference: How to estimate, test, and explain impacts with R and Python
## www.everydaycausal.com
## Copyright © 2025 by Robson Tigre. All rights reserved.
## You may read, run, adapt, and cite this code, provided you credit the source.
## It should not be used to create competing educational or commercial products
##########################################################
## Code for 12. Heterogeneous treatment effects: Different people, different reactions
## Created: Jun 07, 2026
## Last modified: Jul 18, 2026
##########################################################

# ---- Libraries --------------------------------------------------------------
# If you haven't already, run this once to install the packages:
# install.packages(c("grf", "ggplot2", "policytree", "patchwork"))

suppressPackageStartupMessages({ library(grf); library(ggplot2); library(policytree); library(patchwork) })

## ---- book-wide theme and color palette (from .agent/rules/plots-aesthetics.md) ----
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
                size = base_size * 1.04, # 20% smaller than the prior 1.3 so single-line titles fit the width
                color = "grey20",
                hjust = 0.5, # center the title
                margin = margin(b = 5)
            ),
            plot.title.position = "plot", # center over the full figure width, not just the panel
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

## ---- config (single source of truth) ----
## seed: fixed so set.seed() below regenerates the committed CSV byte-for-byte and
## every number the chapter quotes reproduces run for run (the later analysis
## set.seed() calls use it too). The chapter's own display blocks stay seedless.
cfg <- list(seed = 20260606, n = 20000, coupon_cost = 5,
            budget_share = 0.20, n_trees = 2000)
set.seed(cfg$seed)

## ---- covariates ----
n          <- cfg$n
recency    <- runif(n, 0, 1)            # 0 = just purchased, 1 = long lapsed
frequency  <- rpois(n, lambda = 3)
disc_sens  <- rbeta(n, 2, 2)            # discount sensitivity in [0, 1]
past_spend <- rgamma(n, shape = 2, rate = 0.02)

## ---- randomized coupon (clean identification) ----
D <- rbinom(n, 1, 0.5)

## ---- known heterogeneous effect on incremental profit before coupon cost (R$) ----
true_cate <- 12 * disc_sens * (recency > 0.6) * (frequency >= 3) - 1
y0 <- 0.02 * past_spend + rnorm(n, 0, 5)
y  <- y0 + D * true_cate

df <- data.frame(user_id = seq_len(n), D, recency, frequency, disc_sens,
                 past_spend, true_cate, observed_profit = y)
dir.create("data", showWarnings = FALSE)
write.csv(df, "data/coupon_allocation_experiment.csv", row.names = FALSE)
cat(sprintf("True ATE = %.3f; true CATE range [%.2f, %.2f]\n",
            mean(true_cate), min(true_cate), max(true_cate)))

## ---- estimators ----
X  <- as.matrix(df[, c("recency", "frequency", "disc_sens", "past_spend")])
cf <- causal_forest(X, df$observed_profit, df$D, num.trees = cfg$n_trees, seed = cfg$seed)
df$cate_forest <- predict(cf)$predictions
ols <- lm(observed_profit ~ D * (recency + frequency + disc_sens + past_spend), data = df)
nd1 <- df; nd1$D <- 1; nd0 <- df; nd0$D <- 0
df$cate_ols <- predict(ols, nd1) - predict(ols, nd0)

## ---- headline experiment result + how well each estimator recovers the truth ----
## The chapter's opening table quotes the difference-in-means ATE and its 95% CI,
## and the CATE-vs-truth figure quotes each estimator's correlation with true_cate.
## Deterministic reads -- no rng consumed.
ate_hat <- mean(df$observed_profit[df$D == 1]) - mean(df$observed_profit[df$D == 0])
ate_se  <- sqrt(var(df$observed_profit[df$D == 1]) / sum(df$D == 1) +
                var(df$observed_profit[df$D == 0]) / sum(df$D == 0))
cat(sprintf("Estimated ATE (difference in means) = %.3f; 95%% CI [%.3f, %.3f]\n",
            ate_hat, ate_hat - 1.96 * ate_se, ate_hat + 1.96 * ate_se))
cat(sprintf("forest corr with truth = %.3f; ols corr with truth = %.3f\n",
            cor(df$cate_forest, df$true_cate), cor(df$cate_ols, df$true_cate)))

## ---- figure 1: forest recovers what OLS averages away ----
dir.create("images", showWarnings = FALSE)
fig1 <- rbind(data.frame(true = df$true_cate, est = df$cate_forest, method = "Causal forest"),
              data.frame(true = df$true_cate, est = df$cate_ols,    method = "OLS interaction"))
p1 <- ggplot(fig1, aes(true, est)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = book_colors$muted) +
  geom_point(alpha = 0.15, size = 0.5, colour = book_colors$primary) + facet_wrap(~method) +
  labs(x = "True CATE (R$)", y = "Estimated CATE (R$)",
       title = "Who benefits: forest recovers heterogeneity OLS averages away") +
  theme_book()
ggsave("images/heterogeneous-coupon-cate.png", p1, width = 8, height = 4, dpi = 300)

## ---- figure 2: cumulative net incremental profit vs random under budget ----
gain_curve <- function(score) {
  ord <- order(score, decreasing = TRUE)
  net <- df$true_cate[ord] - cfg$coupon_cost
  data.frame(k = seq_along(net) / length(net), cum = cumsum(net))
}
fc <- gain_curve(df$cate_forest); fc$policy <- "Target by forest CATE"
rc <- gain_curve(runif(n));       rc$policy <- "Random"
p2 <- ggplot(rbind(fc, rc), aes(k, cum, colour = policy)) +
  geom_line(linewidth = 0.8) +
  geom_vline(xintercept = cfg$budget_share, linetype = "dashed", colour = book_colors$accent) +
  scale_colour_manual(values = c("Target by forest CATE" = book_colors$primary,
                                 "Random" = book_colors$secondary)) +
  labs(x = "Share of users given a coupon", y = "Cumulative net incremental profit (R$)",
       colour = NULL, title = "Targeting by CATE beats random under a fixed budget") +
  theme_book()
ggsave("images/coupon-targeting-gains.png", p2, width = 7, height = 4.5, dpi = 300)

## ---- figure 3: distribution of estimated CATE with positive-EV region ----
ev_cut <- cfg$coupon_cost
mu     <- mean(df$cate_forest)
p3 <- ggplot(df, aes(cate_forest)) +
  annotate("rect", xmin = ev_cut, xmax = max(df$cate_forest), ymin = 0, ymax = Inf,
           fill = book_colors$accent, alpha = 0.12) +
  geom_histogram(bins = 60, fill = book_colors$muted, colour = "white", linewidth = 0.1) +
  geom_vline(xintercept = mu, linetype = "dashed", colour = book_colors$dark_gray) +
  geom_vline(xintercept = ev_cut, linetype = "dotted", colour = book_colors$accent) +
  annotate("text", x = ev_cut, y = Inf, label = " positive EV (CATE > cost)",
           hjust = 0, vjust = 1.6, colour = book_colors$accent, size = 3.3) +
  annotate("text", x = mu, y = Inf, label = "mean ", hjust = 1, vjust = 1.6,
           colour = book_colors$dark_gray, size = 3.3) +
  labs(x = "Estimated CATE (R$)", y = "Users",
       title = "Most users sit below the coupon cost; a profitable tail clears it") +
  theme_book()
ggsave("images/coupon-cate-histogram.png", p3, width = 7, height = 4.5, dpi = 300)

## ---- figure 4: forest variable importance ----
vi  <- variable_importance(cf)
vdf <- data.frame(feature = colnames(X), importance = as.numeric(vi))
vdf <- vdf[order(vdf$importance), ]
vdf$feature <- factor(vdf$feature, levels = vdf$feature)
p4 <- ggplot(vdf, aes(importance, feature)) +
  geom_col(fill = book_colors$primary) +
  labs(x = "Depth-weighted split frequency", y = NULL,
       title = "What the forest splits on: variable importance") +
  theme_book()
ggsave("images/coupon-variable-importance.png", p4, width = 7, height = 4, dpi = 300)

## ---- figure 5: estimated CATE vs discount sensitivity ----
p5 <- ggplot(df, aes(disc_sens, cate_forest)) +
  geom_point(alpha = 0.10, size = 0.5, colour = book_colors$muted) +
  geom_smooth(method = "loess", formula = y ~ x, se = TRUE,
              colour = book_colors$primary, linewidth = 1) +
  geom_hline(yintercept = cfg$coupon_cost, linetype = "dotted", colour = book_colors$accent) +
  labs(x = "Discount sensitivity", y = "Estimated CATE (R$)",
       title = "Estimated benefit rises with discount sensitivity") +
  theme_book()
ggsave("images/coupon-cate-vs-feature.png", p5, width = 7, height = 4.5, dpi = 300)

## ---- figure: how the forest finds "users like Maria" (adaptive similarity weights) ----
## get_forest_weights returns each training user's weight in the locally weighted
## regression behind one target's CATE -- i.e. how often they share a leaf with the
## target. It is the forest's data-learned answer to "who counts as similar?".
maria   <- which(df$recency > 0.6 & df$frequency >= 3 & df$disc_sens > 0.8)[1]
w_maria <- as.numeric(get_forest_weights(cf, newdata = X[maria, , drop = FALSE]))
wdf  <- data.frame(recency = df$recency, frequency = df$frequency, w = w_maria)
ntop <- round(0.02 * n)                                  # show the top ~2% by weight
top  <- wdf[order(-wdf$w), ][seq_len(ntop), ]
p_w <- ggplot(wdf, aes(recency, frequency)) +
  geom_jitter(height = 0.18, width = 0, colour = book_colors$light_gray, size = 0.5, alpha = 0.5) +
  geom_jitter(data = top, aes(size = w), height = 0.18, width = 0,
              colour = book_colors$primary, alpha = 0.55) +
  geom_vline(xintercept = 0.6, linetype = "dotted", colour = book_colors$muted) +
  geom_hline(yintercept = 3,   linetype = "dotted", colour = book_colors$muted) +
  annotate("point", x = df$recency[maria], y = df$frequency[maria],
           shape = 8, size = 4, stroke = 1.2, colour = book_colors$accent) +
  annotate("text", x = df$recency[maria], y = df$frequency[maria], label = "  Maria",
           hjust = 0, vjust = 0.5, colour = book_colors$accent, size = 3.4, fontface = "bold") +
  scale_size_continuous(range = c(0.3, 3.5), guide = "none") +
  labs(x = "Recency (1 = long lapsed)", y = "Frequency (purchases)",
       title = "How the forest finds \"users like Maria\": similarity weights") +
  theme_book()
ggsave("images/coupon-forest-weights.png", p_w, width = 7, height = 4.5, dpi = 300)

## ============================================================================
## Intuition figures for the "Why trees and forests help" section
## ============================================================================

## ---- figure A: how a single decision tree is built (illustrative outcome tree) ----
## A plain prediction tree, BEFORE the treatment-effect twist: it predicts an OBSERVABLE
## outcome -- here a user's PROFIT LEVEL -- by recursively cutting the recency x frequency
## plane and labelling each final box with its members' average. Illustrative data so the
## build steps read cleanly. Colour marks profit LEVEL on a neutral (magenta) ramp, leaving
## the blue/orange scale to mean treatment EFFECT from the next figures on. High profit sits
## among active (low recency), frequent buyers -- the TOP-LEFT corner -- a DIFFERENT corner
## from where the coupon effect lives (lapsed + frequent), so the figure never implies that
## "profit level = coupon effect". The causal tree reuses this same partitioning machinery
## but changes only what each leaf estimates.
set.seed(2026)
nb     <- 110
b_rec  <- runif(nb, 0, 1)
b_freq <- pmin(rpois(nb, 3), 6)
b_fj   <- b_freq + runif(nb, -0.18, 0.18)                 # vertical jitter for integer counts
# illustrative profit level: higher for active (low recency), frequent buyers (top-left)
b_prof <- 6 * (1 - b_rec) + 0.7 * b_freq + rnorm(nb, 0, 0.8)
s_rec <- 0.5; s_freq <- 2.5                               # the two splits, in order
steps <- c("1. One group", "2. First split (recency)",
           "3. Second split (frequency)", "4. Leaves: predict each box")
bdf    <- data.frame(recency = b_rec, fj = b_fj, prof = b_prof)
ptsA  <- do.call(rbind, lapply(steps, function(s) transform(bdf, step = s)))
ptsA$step <- factor(ptsA$step, levels = steps)
segA <- rbind(
  data.frame(x = s_rec, xend = s_rec, y = -0.5,  yend = 6.5, step = steps[2]),
  data.frame(x = s_rec, xend = s_rec, y = -0.5,  yend = 6.5, step = steps[3]),
  data.frame(x = -0.02, xend = 1.02, y = s_freq, yend = s_freq, step = steps[3]),
  data.frame(x = s_rec, xend = s_rec, y = -0.5,  yend = 6.5, step = steps[4]),
  data.frame(x = -0.02, xend = 1.02, y = s_freq, yend = s_freq, step = steps[4]))
segA$step <- factor(segA$step, levels = steps)
# leaf means computed ONCE (rec<split?, freq<split?) and reused by BOTH the plane
# labels and the rule tree, so the two views of the same tree can never disagree.
lf <- expand.grid(rl = c(TRUE, FALSE), fl = c(TRUE, FALSE))   # rec<split?, freq<split?
leaf_tbl <- do.call(rbind, lapply(seq_len(nrow(lf)), function(i) {
  rl <- lf$rl[i]; fl <- lf$fl[i]
  inb <- ((b_rec < s_rec) == rl) & ((b_freq < s_freq) == fl)
  if (sum(inb) == 0) return(NULL)
  data.frame(rl = rl, fl = fl, prof = mean(b_prof[inb]),
             lab = sprintf("R$%.1f", mean(b_prof[inb])))
}))
prof_lim <- range(b_prof)                                     # shared colour limits
# panel 4: drop each leaf's average profit into its box
lf_lab <- transform(leaf_tbl,
  cx   = ifelse(rl, (-0.02 + s_rec) / 2, (s_rec + 1.02) / 2),
  cy   = ifelse(fl, (-0.5  + s_freq) / 2, (s_freq + 6.5) / 2),
  step = factor(steps[4], levels = steps))
# keyed markers: "A" = the recency cut, "B" = the frequency cut. Letters (not the
# panel numbers) so each cut on the plane shares a tag with its node on the tree.
keyA <- data.frame(x = c(s_rec, 0.96), y = c(6.2, s_freq), tag = c("1", "2"),
                   step = factor(steps[4], levels = steps))
pA <- ggplot() +
  geom_point(data = ptsA, aes(recency, fj, colour = prof), size = 1.3, alpha = 0.9) +
  geom_segment(data = segA, aes(x = x, xend = xend, y = y, yend = yend),
               colour = book_colors$dark_gray, linewidth = 0.6, inherit.aes = FALSE) +
  geom_label(data = lf_lab, aes(cx, cy, label = lab), size = 3.0, colour = book_colors$dark_gray,
             linewidth = 0, fill = "white", alpha = 0.85, inherit.aes = FALSE) +
  geom_point(data = keyA, aes(x, y), shape = 21, size = 5, stroke = 0.5,
             fill = "white", colour = book_colors$dark_gray, inherit.aes = FALSE) +
  geom_text(data = keyA, aes(x, y, label = tag), size = 2.9, fontface = "bold",
            colour = book_colors$dark_gray, inherit.aes = FALSE) +
  facet_wrap(~step, ncol = 2) +
  scale_colour_gradient(low = "grey80", high = book_colors$secondary, name = "Profit (R$)",
                        limits = prof_lim) +
  coord_cartesian(xlim = c(-0.02, 1.02), ylim = c(-0.5, 6.5)) +
  labs(x = "Recency (1 = long lapsed)", y = "Frequency (purchases)",
       subtitle = "Panel A: the mechanics of the cuts") +
  theme_book()

## the SAME tree, drawn as the if/then rules it stores: each internal node is one
## cut (keyed A/B to the plane), each leaf the average profit of the users in it.
getval <- function(rl, fl, col) leaf_tbl[[col]][leaf_tbl$rl == rl & leaf_tbl$fl == fl]
intern <- data.frame(
  x   = c(0.50, 0.27, 0.73), y = c(0.92, 0.56, 0.56),
  lab = c("① recency <= 0.50?", "② frequency <= 2.5?",
          "② frequency <= 2.5?"))   # circled-number key (①/②) ties each node to its cut
leaves <- data.frame(
  x  = c(0.13, 0.41, 0.59, 0.87), y = rep(0.18, 4),
  rl = c(TRUE, TRUE, FALSE, FALSE), fl = c(TRUE, FALSE, TRUE, FALSE))
leaves$lab    <- mapply(getval, leaves$rl, leaves$fl, MoreArgs = list(col = "lab"))
leaves$prof   <- mapply(getval, leaves$rl, leaves$fl, MoreArgs = list(col = "prof"))
leaves$txtcol <- ifelse(leaves$prof > mean(prof_lim), "white", "grey20")
edges <- data.frame(
  x    = c(0.50, 0.50, 0.27, 0.27, 0.73, 0.73),
  y    = c(0.92, 0.92, 0.56, 0.56, 0.56, 0.56),
  xend = c(0.27, 0.73, 0.13, 0.41, 0.59, 0.87),
  yend = c(0.56, 0.56, 0.18, 0.18, 0.18, 0.18),
  br   = c("yes", "no", "yes", "no", "yes", "no"))
edges$mx <- (edges$x + edges$xend) / 2; edges$my <- (edges$y + edges$yend) / 2
pTree <- ggplot() +
  geom_segment(data = edges, aes(x = x, y = y, xend = xend, yend = yend),
               colour = book_colors$muted, linewidth = 0.5) +
  geom_label(data = edges, aes(mx, my, label = br), size = 3.3, colour = book_colors$muted,
             linewidth = 0, fill = "white") +
  geom_tile(data = leaves, aes(x, y, fill = prof), width = 0.15, height = 0.13,
            colour = book_colors$muted, linewidth = 0.4) +   # narrower: open a gap so the two middle leaves read as separate
  geom_text(data = leaves, aes(x, y, label = lab, colour = txtcol), size = 2.9) +
  geom_label(data = intern, aes(x, y, label = lab), size = 2.8, linewidth = 0.2,
             colour = "grey20", fill = book_colors$light_gray) +
  scale_fill_gradient(low = "grey80", high = book_colors$secondary, limits = prof_lim,
                      guide = "none") +
  scale_colour_identity() +
  coord_cartesian(xlim = c(0, 1), ylim = c(0.06, 1.0), clip = "off") +
  labs(subtitle = "Panel B: the same splits as if/then rules") +
  theme_void(base_size = 14) +
  theme(plot.subtitle = element_text(size = 14 * 0.9, colour = "grey40", hjust = 0.5,
                                     margin = margin(b = 10)),
        plot.margin = margin(20, 18, 20, 8))

treeBuild <- (pA / pTree) + plot_layout(heights = c(1.5, 1)) +   # stacked: plane on top, tree below (each full width)
  plot_annotation(title = "How a single decision tree is built",
                  theme = theme(plot.title = element_text(face = "bold", hjust = 0.5,
                                  size = 14 * 1.04, colour = "grey20", margin = margin(b = 6))))
ggsave("images/coupon-tree-build.png", treeBuild, width = 8.5, height = 9.5, dpi = 300)

## ---- figure B: honesty / sample splitting (real coupon data) ----
## The structural fix for "grading your own exam": one sample CHOOSES the boxes, a
## DIFFERENT sample MEASURES the coupon effect inside them. Coarse depth-2 boxes keep each
## box large, so the held-out effects are clean. Boxes are chosen by a transparent greedy
## search for the cut that makes the treated-minus-control coupon effect differ most between
## the two sides -- the causal-tree split criterion itself, run on the structure sample only.
# NOTE: a distinct seed -- reusing cfg$seed would redraw recency (the DGP's first runif)
# and make the split exactly recency < 0.5, confounding the structure/estimate halves.
set.seed(2027)
is_struct <- runif(n) < 0.5
struct <- df[is_struct, ]; estim <- df[!is_struct, ]
best_cut <- function(d, var, grid) {
  eff <- function(g) mean(g$observed_profit[g$D == 1]) - mean(g$observed_profit[g$D == 0])
  sc <- sapply(grid, function(c) {
    da <- d[d[[var]] < c, ]; db <- d[d[[var]] >= c, ]
    if (min(sum(da$D == 1), sum(da$D == 0), sum(db$D == 1), sum(db$D == 0)) < 30) return(-Inf)
    abs(eff(da) - eff(db))                       # split where the effect differs most
  })
  grid[which.max(sc)]
}
c_rec <- best_cut(struct, "recency", seq(0.3, 0.8, by = 0.05))     # strongest first split
c_fhi <- best_cut(struct[struct$recency >= c_rec, ], "frequency", c(1.5, 2.5, 3.5, 4.5))
c_flo <- best_cut(struct[struct$recency <  c_rec, ], "frequency", c(1.5, 2.5, 3.5, 4.5))
ytop  <- max(df$frequency) + 0.5
bx <- rbind(
  data.frame(xmin = -0.02, xmax = c_rec, ymin = -0.5,  ymax = c_flo),   # rec<cut, freq<cut
  data.frame(xmin = -0.02, xmax = c_rec, ymin = c_flo, ymax = ytop),    # rec<cut, freq>=cut
  data.frame(xmin = c_rec, xmax = 1.02, ymin = -0.5,  ymax = c_fhi),    # rec>=cut, freq<cut
  data.frame(xmin = c_rec, xmax = 1.02, ymin = c_fhi, ymax = ytop))     # rec>=cut, freq>=cut
bx$eff <- sapply(seq_len(nrow(bx)), function(i) {
  g <- estim[estim$recency >= bx$xmin[i] & estim$recency < bx$xmax[i] &
             estim$frequency >= bx$ymin[i] & estim$frequency < bx$ymax[i], ]
  mean(g$observed_profit[g$D == 1]) - mean(g$observed_profit[g$D == 0])
})
bx$lab <- sprintf("R$%.1f", bx$eff)
bx$cx  <- (bx$xmin + bx$xmax) / 2
bx$cy  <- pmin(pmax((bx$ymin + bx$ymax) / 2, 0.4), max(df$frequency))
panB <- c("Structure sample: choose the boxes", "Estimate sample: measure the effect")
ppB  <- rbind(transform(struct[, c("recency", "frequency")], panel = panB[1]),
              transform(estim[,  c("recency", "frequency")], panel = panB[2]))
ppB$panel <- factor(ppB$panel, levels = panB)
set.seed(7); ppB$fj <- ppB$frequency + runif(nrow(ppB), -0.2, 0.2)
linesB <- rbind(
  data.frame(x = c_rec, xend = c_rec, y = -0.5,  yend = ytop),          # recency split
  data.frame(x = -0.02, xend = c_rec, y = c_flo, yend = c_flo),         # left frequency split
  data.frame(x = c_rec, xend = 1.02, y = c_fhi, yend = c_fhi))          # right frequency split
linesB <- do.call(rbind, lapply(panB, function(p) transform(linesB, panel = p)))
linesB$panel <- factor(linesB$panel, levels = panB)
bx$panel <- factor(panB[2], levels = panB)                              # labels on estimate panel
# points are neutral (the panel title says which sample); the held-out effect is shown by
# colouring each estimate-panel box on the same blue/orange scale as the forest figure.
pB <- ggplot() +
  geom_rect(data = bx, aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, fill = eff),
            alpha = 0.55, inherit.aes = FALSE) +
  geom_point(data = ppB, aes(recency, fj), colour = book_colors$muted, size = 0.5, alpha = 0.30) +
  geom_segment(data = linesB, aes(x = x, xend = xend, y = y, yend = yend),
               colour = book_colors$dark_gray, linewidth = 0.6, inherit.aes = FALSE) +
  geom_label(data = bx, aes(cx, cy, label = lab), size = 3.0, colour = book_colors$dark_gray,
             linewidth = 0, fill = "white", alpha = 0.85, inherit.aes = FALSE) +
  facet_wrap(~panel) +
  scale_fill_gradient2(low = book_colors$accent, mid = "grey95", high = book_colors$primary,
                       midpoint = 0, name = "Effect (R$)") +
  labs(x = "Recency (1 = long lapsed)", y = "Frequency (purchases)",
       title = "Honesty: choose the boxes on one sample, measure effects on another") +
  theme_book()
ggsave("images/coupon-honesty-split.png", pB, width = 8.5, height = 4.5, dpi = 300)

## ---- figure C: from one tree to a forest (illustrative) ----
## Each tree carves the plane into a few blocky regions and averages the effect inside
## each; on a random subsample with a random subset of features, every tree drops its
## boundaries somewhere different. Averaging many DECORRELATED trees smooths the noise
## into the underlying effect surface. The surface mirrors the chapter's planted gate
## (a clear coupon payoff for lapsed, frequent buyers).
set.seed(2028)
gx <- seq(0, 1, length.out = 60); gy <- seq(0, 6, length.out = 60)
gd <- expand.grid(recency = gx, frequency = gy)
true_surf <- ifelse(gd$recency > 0.6 & gd$frequency >= 3, 5, -1)
one_tree <- function() {
  rcuts <- sort(runif(sample(2:3, 1), 0.2, 0.9))
  fcuts <- sort(runif(sample(2:3, 1), 1, 5))
  blk   <- interaction(findInterval(gd$recency, rcuts), findInterval(gd$frequency, fcuts))
  as.numeric(tapply(true_surf, blk, mean)[as.character(blk)]) + rnorm(nrow(gd), 0, 1.2)
}
ntree  <- 5
trees  <- replicate(ntree, one_tree())
forest <- rowMeans(replicate(200, one_tree()))
panC <- c(paste("Tree", 1:ntree), "Forest (average of 200)")
bigC <- rbind(
  do.call(rbind, lapply(seq_len(ntree), function(i) data.frame(gd, val = trees[, i], panel = panC[i]))),
  data.frame(gd, val = forest, panel = panC[ntree + 1]))
bigC$panel <- factor(bigC$panel, levels = panC)
pC <- ggplot(bigC, aes(recency, frequency, fill = val)) +
  geom_raster() +
  facet_wrap(~panel, ncol = 3) +
  scale_fill_gradient2(low = book_colors$accent, mid = "grey95", high = book_colors$primary,
                       midpoint = 0, name = "Effect (R$)") +
  labs(x = "Recency (1 = long lapsed)", y = "Frequency (purchases)",
       title = "From one tree to a forest: averaging decorrelated trees smooths the noise") +
  theme_book()
ggsave("images/coupon-tree-to-forest.png", pC, width = 8.5, height = 5.5, dpi = 300)

## ---- figure 6: depth-2 policy tree (cost-aware reward) ----
## reward matrix: action 1 = no coupon (reward 0); action 2 = coupon (out-of-bag CATE
## net of cost). This simple matrix generates the DISPLAYED tree and its reported
## numbers (treated share, model-implied and truth-scored values printed below).
## Production alternative (NOT the displayed tree's input): doubly robust scores.
## double_robust_scores(cf) returns one AIPW score per action; keep BOTH columns and
## subtract the coupon cost from the coupon action only:
##   Gamma_dr <- double_robust_scores(cf)              # columns: control, treated
##   Gamma_dr[, 2] <- Gamma_dr[, 2] - cfg$coupon_cost  # cost hits the coupon action only
## Do not zero the control column or subtract the cost from both actions, and do not
## expect the alternative to reproduce the simple matrix's numbers exactly.
Gamma <- cbind("No coupon" = 0, "Coupon" = df$cate_forest - cfg$coupon_cost)
ptree <- policy_tree(X, Gamma, depth = 2, min.node.size = 1000)  # broad, stable leaves

## Custom base-R renderer instead of plot.policy_tree's built-in DiagrammeR
## widget: this needs an exact-resolution static PNG in book_colors matching
## every other chapter figure, which plot.policy_tree doesn't expose. Layout
## (x by leaf order, y by depth) follows policytree's sequential node
## numbering, so positions must track the actual left/right children.
plot_policy_tree_base <- function(pt, file, title) {
  nodes <- pt$nodes; feat <- pt$columns; acts <- pt$action.names
  xpos <- numeric(length(nodes)); ypos <- numeric(length(nodes)); leaf_n <- 0
  set_depth <- function(i, d) {
    ypos[i] <<- d; nd <- nodes[[i]]
    if (!isTRUE(nd$is_leaf)) { set_depth(nd$left_child, d + 1); set_depth(nd$right_child, d + 1) }
  }
  assign_x <- function(i) {
    nd <- nodes[[i]]
    if (isTRUE(nd$is_leaf)) { leaf_n <<- leaf_n + 1; xpos[i] <<- leaf_n }
    else { assign_x(nd$left_child); assign_x(nd$right_child)
           xpos[i] <<- (xpos[nd$left_child] + xpos[nd$right_child]) / 2 }
  }
  set_depth(1, 0); assign_x(1)
  maxd <- max(ypos)
  X <- if (leaf_n > 1) 0.12 + 0.76 * (xpos - 1) / (leaf_n - 1) else rep(0.5, length(nodes))
  Y <- if (maxd > 0)   0.90 - 0.78 * ypos / maxd            else rep(0.5, length(nodes))
  png(file, width = 2750, height = 1550, res = 300)  # 2.5x the 1100x620@120 canvas: same 9.17x5.17in, crisper
  op <- par(mar = c(0.5, 0.5, 2.5, 0.5)); on.exit({ par(op); dev.off() })
  plot(NA, xlim = c(0, 1), ylim = c(0, 1), axes = FALSE, xlab = "", ylab = "", main = title)
  draw_branch_label <- function(x, y, lab) {
    branch_cex <- 0.95
    w <- strwidth(lab, cex = branch_cex)
    h <- strheight(lab, cex = branch_cex)
    rect(x - w / 2 - 0.01, y - h / 2 - 0.008,
         x + w / 2 + 0.01, y + h / 2 + 0.008,
         col = "white", border = NA)
    text(x, y, lab, cex = branch_cex, col = book_colors$dark_gray)
  }
  for (i in seq_along(nodes)) {
    nd <- nodes[[i]]; if (isTRUE(nd$is_leaf)) next
    for (ch in c(nd$left_child, nd$right_child))
      lines(c(X[i], X[ch]), c(Y[i] - 0.045, Y[ch] + 0.05), col = book_colors$muted, lwd = 1.4)
    draw_branch_label(mean(c(X[i], X[nd$left_child])),  mean(c(Y[i], Y[nd$left_child])),  "yes")
    draw_branch_label(mean(c(X[i], X[nd$right_child])), mean(c(Y[i], Y[nd$right_child])), "no")
  }
  for (i in seq_along(nodes)) {
    nd <- nodes[[i]]
    if (isTRUE(nd$is_leaf)) {
      lab <- acts[nd$action]; col <- if (nd$action == 2) book_colors$primary else book_colors$muted
      rect(X[i] - 0.105, Y[i] - 0.058, X[i] + 0.105, Y[i] + 0.058, col = col, border = NA)
      text(X[i], Y[i], lab, col = "white", font = 2, cex = 1.05)
    } else {
      split_label <- if (abs(nd$split_value - round(nd$split_value)) < 1e-8) {
        sprintf("%.0f", nd$split_value)
      } else {
        sprintf("%.2f", nd$split_value)
      }
      lab <- sprintf("%s <= %s?", feat[nd$split_variable], split_label)
      rect(X[i] - 0.145, Y[i] - 0.052, X[i] + 0.145, Y[i] + 0.052, col = book_colors$light_gray, border = book_colors$muted)
      text(X[i], Y[i], lab, cex = 1.0, col = book_colors$dark_gray)
    }
  }
}
plot_policy_tree_base(ptree, "images/coupon-policy-tree.png",
                      "Depth-2 coupon policy")

## The displayed tree's two valuations, never collapsed into one claim: the
## model-implied value scores the allocation with the same OOB CATEs the tree was
## learned from; the truth-scored value grades that same allocation against the
## planted true_cate (a simulation-only luxury).
tree_action <- predict(ptree, X)   # 1 = no coupon, 2 = coupon
tree_coupon <- tree_action == 2
cat(sprintf("[policy tree] treated share = %.1f%%; model-implied net value (OOB CATE - cost) = R$%.0f; truth-scored net value (true CATE - cost) = R$%.0f\n",
            100 * mean(tree_coupon),
            sum((df$cate_forest - cfg$coupon_cost)[tree_coupon]),
            sum((df$true_cate  - cfg$coupon_cost)[tree_coupon])))

## ---- figure 6b: depth-3 policy tree (same rewards, one level deeper) ----
## Exact tree search cost grows steeply with depth (Sverdrup et al. 2020, JOSS),
## so an exact depth-3 search at n = 20,000 is computationally impractical -- not
## attempted here. hybrid_policy_tree (policytree 1.2.4) runs the exact depth-2
## lookahead one level at a time; it is NOT guaranteed globally optimal at depth 3.
## On this data it lands on the planted three-gate rule in under a minute.
ptree3 <- hybrid_policy_tree(X, Gamma, depth = 3, search.depth = 2,
                             min.node.size = 1000)
print(ptree3)
tree3_coupon <- predict(ptree3, X) == 2
cat(sprintf("[policy tree depth 3] treated share = %.1f%%; model-implied net value (OOB CATE - cost) = R$%.0f; truth-scored net value (true CATE - cost) = R$%.0f\n",
            100 * mean(tree3_coupon),
            sum((df$cate_forest - cfg$coupon_cost)[tree3_coupon]),
            sum((df$true_cate  - cfg$coupon_cost)[tree3_coupon])))

## Truth ceiling for ANY allocation: treat exactly the users whose true effect
## exceeds the R$5 cost (the finite-sample truth optimum under unconstrained
## binary allocation). On this data that predicate equals the planted three-gate
## rule recency > 0.6 & frequency >= 3 & disc_sens > 0.5, row for row.
oracle_coupon <- df$true_cate - cfg$coupon_cost > 0
cat(sprintf("[truth ceiling] treated share = %.1f%%; truth-scored net value (true CATE - cost) = R$%.0f\n",
            100 * mean(oracle_coupon),
            sum((df$true_cate - cfg$coupon_cost)[oracle_coupon])))

plot_policy_tree_base(ptree3, "images/coupon-policy-tree-depth3.png",
                      "Depth-3 coupon policy")
cat("Wrote data/coupon_allocation_experiment.csv and 11 figures (TOC curve written after the RATE block below).\n")

## ---- validation diagnostics: numbers behind the chapter's trust table ----
## Each block backs one row of the "diagnostic / what we want / coupon result /
## decision" table in the chapter's validation section.

# (1) calibration: differential coefficient ~ 1 with a small p-value says the
# claimed spread of effects is real, not inflated
cat("\n[calibration]\n"); print(test_calibration(cf))

# (2) best linear projection: which features move the CATE, with SEs.
# Raw coefficients are per unit of each feature, so they can't be ranked
# across features (recency spans 0-1, frequency counts purchases). The
# second pass projects on scale(X) -- per-1-SD coefficients on a common
# scale -- while the forest itself stays fitted on the raw features.
cat("\n[best linear projection]\n"); print(best_linear_projection(cf, X))
X_blp_std <- scale(X)
cat("\n[best linear projection, per 1 SD]\n")
print(best_linear_projection(cf, X_blp_std))

# (3) RATE: does ranking users by the forest's CATE have targeting value?
# grf doc WARNING: priorities "should be constructed independently from the
# evaluation forest training data" -- rank-forest on one half, evaluate on the
# other (the same-forest OOB shortcut + symmetric CI is anti-conservative;
# see grf's rate_cv vignette).
set.seed(cfg$seed)
idx_tr   <- sample(nrow(df), nrow(df) / 2)
cf_rank  <- causal_forest(X[idx_tr, ], df$observed_profit[idx_tr], df$D[idx_tr],
                          num.trees = cfg$n_trees, seed = cfg$seed)
priority <- predict(cf_rank, X[-idx_tr, ])$predictions
cf_eval  <- causal_forest(X[-idx_tr, ], df$observed_profit[-idx_tr], df$D[-idx_tr],
                          num.trees = cfg$n_trees, seed = cfg$seed)
rate <- rank_average_treatment_effect(cf_eval, priority)
cat(sprintf("\n[RATE, train/eval split] AUTOC = %.3f +/- %.3f (95%% CI)\n",
            rate$estimate, 1.96 * rate$std.err))

## ---- figure: TOC curve (the continuous version of GATES) ----
## A fine q-grid on the same train/eval split; rate$TOC has columns estimate, std.err, q.
## TOC(q) = extra uplift over the overall ATE among the top-q ranked users.
toc <- rank_average_treatment_effect(cf_eval, priority,
                                     q = seq(0.05, 1, length.out = 100))$TOC
toc$lo <- toc$estimate - 1.96 * toc$std.err
toc$hi <- toc$estimate + 1.96 * toc$std.err
p_toc <- ggplot(toc, aes(q, estimate)) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = book_colors$muted) +
  geom_ribbon(aes(ymin = lo, ymax = hi), fill = book_colors$primary, alpha = 0.15) +
  geom_line(colour = book_colors$primary, linewidth = 0.9) +
  labs(x = "Treated fraction (q), ranked by predicted CATE",
       y = "Uplift vs. overall ATE (R$)",
       title = "Do the top-ranked users really gain more? The TOC curve") +
  theme_book()
ggsave("images/coupon-toc-curve.png", p_toc, width = 7, height = 4.5, dpi = 300)
cat("Wrote images/coupon-toc-curve.png.\n")

# (4) GATES: realized treated-minus-control uplift by predicted-CATE quintile.
# grf predictions are out-of-bag, so in-sample bucketing is valid here
# (econml's effect(X) is not OOB -- see the Python script's held-out version).
qg <- cut(df$cate_forest, quantile(df$cate_forest, 0:5 / 5),
          include.lowest = TRUE, labels = FALSE)
gates <- sapply(1:5, function(q) {
  g <- df[qg == q, ]
  mean(g$observed_profit[g$D == 1]) - mean(g$observed_profit[g$D == 0])
})
cat("\n[GATES] uplift by predicted-CATE quintile (1 = lowest):\n")
print(round(gates, 2))

# targeting table: model-implied value of the four policies in the chapter's
# "Rank, cut, and compare" table. Each policy is scored by the model's own
# estimates (cate_forest - cost) -- the "what the model expects" numbers; the
# gain-curve figure scores the same ranking against the planted truth instead.
set.seed(cfg$seed)
ev_est   <- df$cate_forest - cfg$coupon_cost
n_budget <- floor(cfg$budget_share * n)
cat(sprintf("\n[targeting table] blanket = %.0f; random %.0f%% = %.0f; top %.0f%% by CATE = %.0f; positive-EV only = %.0f (%.1f%% of users)\n",
            sum(ev_est), 100 * cfg$budget_share, sum(sample(ev_est, n_budget)),
            100 * cfg$budget_share, sum(sort(ev_est, decreasing = TRUE)[seq_len(n_budget)]),
            sum(ev_est[ev_est > 0]), 100 * mean(ev_est > 0)))

# (5) overlap: estimated propensities should sit far from 0 and 1
cat(sprintf("\n[overlap] propensity (W.hat) range: [%.3f, %.3f]\n",
            min(cf$W.hat), max(cf$W.hat)))

# (6) placebo: permute the treatment label and refit -- the apparent
# heterogeneity should collapse toward zero
set.seed(cfg$seed)
cf_pl <- causal_forest(X, df$observed_profit, sample(df$D),
                       num.trees = cfg$n_trees, seed = cfg$seed)
cat(sprintf("\n[placebo] SD of predictions = %.3f (real forest: %.3f)\n",
            sd(predict(cf_pl)$predictions), sd(df$cate_forest)))
print(test_calibration(cf_pl))

## ---- power check: "no heterogeneity detected" is usually a power problem ----
## The planted spread here is loud: SD(true_cate) = 2.8, ~7x the true ATE (0.38).
## (a) Even a 2,000-user subsample still detects it. (b) Shrink the deviations around
## the SAME ATE (same noise, same draws) to ~half the ATE's size and the full
## 20,000-user experiment still pins the ATE but sees no heterogeneity at all.
## These two runs back the numbers quoted in the chapter's power subsection.
set.seed(cfg$seed)
idx_small <- sample(n, 2000)
cf_small  <- causal_forest(X[idx_small, ], df$observed_profit[idx_small],
                           df$D[idx_small], num.trees = cfg$n_trees, seed = cfg$seed)
cat(sprintf("\n[loud world, n = 2000] corr with truth = %.3f\n",
            cor(predict(cf_small)$predictions, df$true_cate[idx_small])))
print(test_calibration(cf_small))

y_base   <- df$observed_profit - df$D * df$true_cate  # outcome stripped of treatment effects
ate_true <- mean(df$true_cate)
tau_q    <- ate_true + (df$true_cate - ate_true) / 16    # quiet world: SD = 0.18, ~half the ATE
y_q      <- y_base + df$D * tau_q
cf_q     <- causal_forest(X, y_q, df$D, num.trees = cfg$n_trees, seed = cfg$seed)
tt_q     <- t.test(y_q[df$D == 1], y_q[df$D == 0])
cat(sprintf("\n[quiet world (SD = %.2f, %.2fx ATE), n = %d] ATE t = %.2f; corr with truth = %.3f\n",
            sd(tau_q), sd(tau_q) / ate_true, n, tt_q$statistic,
            cor(predict(cf_q)$predictions, tau_q)))
print(test_calibration(cf_q))
