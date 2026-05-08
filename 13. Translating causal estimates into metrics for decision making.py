##########################################################
## Everyday Causal Inference: How to estimate, test, and explain impacts with R and Python
## www.everydaycausal.com
## Copyright (c) 2025 by Robson Tigre. All rights reserved.
## You may read, share, and cite for learning purposes, provided you credit the source.
## It should not be used to create competing educational or commercial products
##########################################################
## Code for Chapter 13 - Translating causal estimates into metrics of business value
## Created: Mar 06, 2026
## Last modified: 2026-05-07
##########################################################

# ==========================================================
# SETUP
# ==========================================================
# pip install numpy pandas matplotlib scipy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import t as t_dist

# Book color palette (colorblind-friendly, consistent across all chapters)
book_colors = {
    "primary":    "#2E86AB",   # Steel blue
    "secondary":  "#A23B72",   # Magenta
    "accent":     "#F18F01",   # Orange
    "success":    "#C73E1D",   # Red-orange
    "muted":      "#6C757D",   # Gray
    "light_gray": "#E5E5E5",
    "dark_gray":  "#4D4D4D",
}


# ==========================================================
# BLOCK 1: SCALING -- FROM STATISTICAL UNITS TO FINANCIAL UNITS
# (Chapter section: "From statistical units to financial units")
# ==========================================================
# Multiply per-user effect by the eligible user base, then by the projection
# horizon, to get the naive starting number that the rest of the pipeline
# discounts down. R$2.34 x 5M MAU x 12 months ~= R$140M annual.

ate = 2.34       # R$ per user per 30 days (from Chapter 3)
mau = 5_000_000  # monthly active users
monthly_impact = ate * mau
annual_naive = monthly_impact * 12

print(f"Per-user effect: R${ate:.2f}/month")
print(f"MAU: {mau:,.0f}")
print(f"Projected monthly impact: R${monthly_impact:,.0f}")
print(f"Naive annual projection: R${annual_naive:,.0f}")


# ==========================================================
# BLOCK 2: CLV UPLIFT WITH MONTHLY DECAY (headline calculation, r = 0)
# (Chapter section: "Decay over time: customer lifetime value uplift")
# ==========================================================
# Decay only, no time-value-of-money discount. Using r = 0.01 (illustrative
# monthly WACC, ~12% annualized) lowers the result to ~R$16.10; over a 12-month
# horizon that difference is small next to the decay assumption itself, so the
# chapter shows the headline at r = 0 for clarity.

ate_month_0 = 2.34          # R$ per user, month 0
effect_persistence = 0.90   # monthly persistence; effect decays 10% per period
horizon = 12                # months

months = np.arange(horizon)
effects = ate_month_0 * effect_persistence ** months
clv_uplift = effects.sum()

print(f"\nMonth 0 effect: R${effects[0]:.2f}")
print(f"Month 6 effect: R${effects[6]:.2f}")
print(f"Month 11 effect: R${effects[11]:.2f}")
print(f"CLV uplift (12 months): R${clv_uplift:.2f} per user")
print(f"Without decay: R${ate_month_0 * 12:.2f} per user")


# ==========================================================
# FIGURE: clv-decay-vs-naive.png   (@fig-clv-decay)
# (Chapter section: "Decay over time: customer lifetime value uplift")
# ==========================================================
# Shows the optimism-bias gap between the naive flat path (R$2.34/month) and
# the decayed path (lambda = 0.90). The shaded ribbon is the cumulative
# overstatement that the naive forecast would build into the board deck.

naive_path = np.full(horizon, ate_month_0)
cum_decayed = effects.cumsum()
cum_naive = naive_path.cumsum()

fig, ax = plt.subplots(figsize=(9, 5.5))

ax.fill_between(months, effects, naive_path,
                color=book_colors["accent"], alpha=0.15)
ax.plot(months, naive_path, color=book_colors["muted"], linestyle="--",
        linewidth=2.0, label="Naive (no decay)")
ax.plot(months, effects, color=book_colors["primary"], linewidth=2.0,
        label="With 10% monthly decay")
ax.scatter(months, naive_path, color=book_colors["muted"], s=35, zorder=3)
ax.scatter(months, effects, color=book_colors["primary"], s=35, zorder=3)

ax.text(7, (ate_month_0 + effects[7]) / 2, "Optimism\nbias",
        color=book_colors["accent"], style="italic", fontsize=12,
        ha="center", va="center")
ax.text(11, naive_path[11] + 0.15, f"R${ate_month_0:.2f}",
        color=book_colors["muted"], fontsize=10, ha="center")
ax.text(11, effects[11] - 0.15, f"R${effects[11]:.2f}",
        color=book_colors["primary"], fontsize=10, ha="center")

ax.set_xticks(months)
ax.set_xticklabels(range(1, horizon + 1))
ax.set_xlabel("Month", fontsize=12, color="#4D4D4D")
ax.set_ylabel("Monthly effect per user", fontsize=12, color="#4D4D4D")
ax.set_title("Optimism bias: naive vs. decayed effect projection",
             fontweight="bold", fontsize=15, color="#333333", loc="left")
ax.text(0, 1.02,
        f"CLV uplift: R${cum_decayed[-1]:.2f} (with decay) vs. "
        f"R${cum_naive[-1]:.2f} (naive) over 12 months",
        transform=ax.transAxes, fontsize=11, color="#666666")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"R${x:.2f}"))
ax.grid(True, color="#E5E5E5", linewidth=0.5)
ax.set_axisbelow(True)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none")

plt.tight_layout()
plt.savefig("images/clv-decay-vs-naive.png", dpi=300, bbox_inches="tight",
            facecolor="white")
plt.close(fig)
print("Saved: images/clv-decay-vs-naive.png")


# ==========================================================
# BLOCK 3: ROI -- PERSONALIZED FEED
# (Chapter section: "Return on investment (ROI)")
# ==========================================================
# feed_12m_profit below is the 12-month cumulative profit (post-decay,
# pre-investment) -- NOT an annual run-rate. profit_30d from @sec-protocol is
# already net of variable costs; the code below only subtracts the fixed
# investment to get net profit.
clv_per_user    = sum(2.34 * 0.90**t for t in range(12))  # ~R$17/user (decay only; discount omitted -- overstates by ~4%)
feed_12m_profit = clv_per_user * 5e6   # ~R$84M: 12-month cumulative profit (variable costs already netted)
feed_investment = 4e6                  # R$4M fixed investment (build + maintain)
feed_roi        = (feed_12m_profit - feed_investment) / feed_investment  # = Incremental Net Profit / Investment

print(f"\nCLV uplift per user: R${clv_per_user:.2f}")
print(f"12-month profit (post-decay): R${feed_12m_profit/1e6:,.0f}M")
print(f"Incremental net profit: R${(feed_12m_profit - feed_investment)/1e6:,.0f}M")
print(f"ROI: {feed_roi:.0f}x annual return")


# ==========================================================
# FIGURE: diminishing-returns-saturation.png   (@fig-diminishing-returns)
# (Chapter section: "Diminishing returns")
# ==========================================================
# Illustrative ad-spend saturation curve: revenue grows logarithmically with
# spend, so each additional Real buys less revenue than the one before.

np.random.seed(42)
spend_levels = np.arange(100, 5100, 100)  # R$ thousands
true_revenue = 800 * np.log(spend_levels / 100 + 1) + np.random.normal(0, 50, len(spend_levels))

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.scatter(spend_levels, true_revenue, color=book_colors["primary"],
           alpha=0.6, s=30)

# Fit and plot log curve (matches R's geom_smooth(formula = y ~ log(x)))
log_spend = np.log(spend_levels)
coeffs = np.polyfit(log_spend, true_revenue, 1)
fitted = np.polyval(coeffs, log_spend)
ax.plot(spend_levels, fitted, color=book_colors["secondary"], linewidth=2.0)

ax.text(2300, max(true_revenue) * 0.22,
        "Each extra Real buys\nless revenue\nthan the one before",
        color=book_colors["accent"], style="italic", fontsize=12, ha="left")
ax.set_xticks([500, 1000, 1500, 2500, 5000])
ax.set_xticklabels(["R$0.5M", "R$1M", "R$1.5M", "R$2.5M", "R$5M"])
ax.set_xlabel("Cumulative ad spend", fontsize=12, color="#4D4D4D")
ax.set_ylabel("Incremental revenue (R$ thousands)", fontsize=12, color="#4D4D4D")
ax.set_title("The saturation curve: more spend, less bang per buck",
             fontweight="bold", fontsize=15, color="#333333", loc="left")
ax.text(0, -0.15, "Simulated data for illustration",
        transform=ax.transAxes, fontsize=9, color="#888888")
ax.grid(True, color="#E5E5E5", linewidth=0.5)
ax.set_axisbelow(True)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig("images/diminishing-returns-saturation.png", dpi=300,
            bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Saved: images/diminishing-returns-saturation.png")


# ==========================================================
# FIGURE: diminishing-returns-squeeze.png   (@fig-squeeze)
# (Chapter section: "Diminishing returns")
# ==========================================================
# Marginal revenue falls while marginal cost rises -- the margin between the
# two curves shrinks and eventually inverts. Crossover marks where marginal
# ROI turns negative.

np.random.seed(42)
n_sq = 500
spend_sq = np.linspace(0.05, 2.0, n_sq)   # R$ millions
mr_sq = 8.0 * np.exp(-2.0 * spend_sq) + 0.4
mc_sq = 1.2 + 3.5 * spend_sq ** 2.2

# Find crossover by linear interpolation on the difference series
diff_val = mr_sq - mc_sq
sign_change = np.where(np.diff(np.sign(diff_val)) != 0)[0]
cross_idx = sign_change[0]
x1, x2 = spend_sq[cross_idx], spend_sq[cross_idx + 1]
y1, y2 = diff_val[cross_idx], diff_val[cross_idx + 1]
cross_x = x1 - y1 * (x2 - x1) / (y2 - y1)
cross_y = 8.0 * np.exp(-2.0 * cross_x) + 0.4

healthy_x  = 0.20
healthy_mr = 8.0 * np.exp(-2.0 * healthy_x) + 0.4
healthy_mc = 1.2 + 3.5 * healthy_x ** 2.2

neg_label_x = 0.80
neg_label_y = 8.6

fig, ax = plt.subplots(figsize=(9, 5.5))

# Shaded margin where MR > MC
mask = mr_sq >= mc_sq
ax.fill_between(spend_sq[mask], mc_sq[mask], mr_sq[mask],
                color=book_colors["primary"], alpha=0.08)

ax.plot(spend_sq, mr_sq, color=book_colors["primary"], linewidth=2.0,
        label="Marginal revenue")
ax.plot(spend_sq, mc_sq, color=book_colors["secondary"], linewidth=2.0,
        label="Marginal cost")

# Crossover dot
ax.scatter([cross_x], [cross_y], color=book_colors["accent"], s=60, zorder=4)

# Leader line from the relabeled crossover annotation down to the dot
ax.annotate("", xy=(cross_x + 0.03, cross_y + 0.18),
            xytext=(neg_label_x + 0.02, 7.4),
            arrowprops=dict(arrowstyle="->", color=book_colors["accent"],
                            lw=0.8))
ax.text(neg_label_x, neg_label_y,
        "Point at which\nmarginal ROI\nturns negative",
        color=book_colors["accent"], fontweight="bold", fontsize=11,
        ha="left", va="center")

# Healthy-margin label and double-headed arrow
ax.text(healthy_x + 0.05, 2.8, "Healthy\nmargin",
        color=book_colors["primary"], fontweight="bold", fontsize=10,
        ha="left", va="center")
ax.annotate("", xy=(healthy_x, healthy_mr - 0.15),
            xytext=(healthy_x, healthy_mc + 0.15),
            arrowprops=dict(arrowstyle="<->", color=book_colors["primary"],
                            lw=0.9))

ax.set_xticks([0.1, 0.5, 1.0, 1.5, 2.0])
ax.set_xticklabels([f"R${v}M" for v in (0.1, 0.5, 1.0, 1.5, 2.0)])
ax.set_xlabel("Cumulative ad spend", fontsize=12, color="#4D4D4D")
ax.set_ylabel("R$ per additional R$1K spent", fontsize=12, color="#4D4D4D")
ax.set_title("The double squeeze: revenue falls, costs rise",
             fontweight="bold", fontsize=15, color="#333333", loc="left")
ax.text(0, 1.02,
        "At scale, the margin between marginal revenue and marginal cost narrows — then inverts",
        transform=ax.transAxes, fontsize=11, color="#666666")
ax.text(0, -0.15, "Simulated data for illustration",
        transform=ax.transAxes, fontsize=9, color="#888888")
ax.set_ylim(0, 10)
ax.grid(True, color="#E5E5E5", linewidth=0.5)
ax.set_axisbelow(True)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none")

plt.tight_layout()
plt.savefig("images/diminishing-returns-squeeze.png", dpi=300,
            bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: images/diminishing-returns-squeeze.png  (crossover at R${cross_x:.2f}M)")


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

feed_labels = ["Naive\nprojection",
               "Effect\ndecay",
               "Adoption\nramp",
               "Represent-\nativeness",
               "Cannibal. /\nSUTVA",
               "Realistic\nprojection"]
feed_totals = [140, 84, 76, 72, 70, 70]


def build_waterfall(labels, totals, title, subtitle, unit_label="R$ millions"):
    n = len(labels)
    deductions = [0] + [totals[i] - totals[i - 1] for i in range(1, n)]

    bar_type = ["deduction"] * n
    bar_type[0] = "start"
    bar_type[-1] = "end"

    fill_color = {
        "start":     book_colors["primary"],
        "deduction": book_colors["success"],
        "end":       book_colors["primary"],
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    max_val = max(totals)
    thin_threshold = max_val * 0.08

    for i in range(n):
        if i == 0 or i == n - 1:
            ymin, ymax = 0, totals[i]
        else:
            ymin, ymax = totals[i], totals[i - 1]

        ax.add_patch(Rectangle((i + 1 - 0.4, ymin), 0.8, ymax - ymin,
                               facecolor=fill_color[bar_type[i]],
                               edgecolor="white", linewidth=0.5))

        bar_height = abs(ymax - ymin)
        is_thin = bar_height < thin_threshold

        if bar_type[i] == "deduction":
            label_str = f"−R${round(abs(deductions[i]))}M"
            label_y = ymin - max_val * 0.03 if is_thin else (ymin + ymax) / 2
        else:
            label_str = f"R${round(totals[i])}M"
            label_y = ymax / 2

        label_color = book_colors["dark_gray"] if is_thin else "white"
        ax.text(i + 1, label_y, label_str, color=label_color,
                fontweight="bold", fontsize=10, ha="center", va="center")

    # Connector lines (skip the last connector before the final total bar)
    for i in range(n - 2):
        ax.plot([i + 1 + 0.45, i + 2 - 0.45], [totals[i], totals[i]],
                color=book_colors["muted"], linewidth=0.8, linestyle=":")

    ax.set_xticks(range(1, n + 1))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel(unit_label, fontsize=12, color="#4D4D4D")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"R${int(x)}M"))
    ax.set_title(title, fontweight="bold", fontsize=15, color="#333333",
                 loc="left")
    ax.text(0, 1.02, subtitle, transform=ax.transAxes,
            fontsize=11, color="#666666")
    ax.set_xlim(0.5, n + 0.5)
    ax.set_ylim(0, max_val * 1.05)
    ax.grid(True, axis="y", color="#E5E5E5", linewidth=0.5)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    return fig


fig = build_waterfall(
    labels   = feed_labels,
    totals   = feed_totals,
    title    = "The projection waterfall: personalized feed",
    subtitle = "Each discount peels back a layer of optimism from the naive R$140M",
)
plt.tight_layout()
plt.savefig("images/waterfall-feed.png", dpi=300, bbox_inches="tight",
            facecolor="white")
plt.close(fig)
print("Saved: images/waterfall-feed.png")


# ==========================================================
# BLOCK 4: SENSITIVITY ANALYSIS
# (Chapter section: "Sensitivity analysis")
# ==========================================================
# profit_12m below is the 12-month cumulative profit (post-decay,
# pre-investment); ROI subtracts fixed investment in the numerator.
investments = [4e6, 6e6, 8e6]
decays = [0.00, 0.10, 0.20]

rows = []
for investment in investments:
    row = {"Investment": f"R${investment/1e6:.0f}M"}
    for decay in decays:
        profit_12m = sum(2.34 * (1 - decay)**t for t in range(12)) * 5e6
        roi = (profit_12m - investment) / investment
        row[f"{decay*100:.0f}% decay"] = f"{roi:.0f}x"
    rows.append(row)

print("\nROI sensitivity (5M MAU, varying fixed investment and monthly decay):")
print(pd.DataFrame(rows).to_string(index=False))


# ==========================================================
# BLOCK 5: PRE-EXPERIMENT ROI PROJECTION FROM MEI-CALIBRATED MDE
# (Chapter section: "From breakeven to experiment design")
# ==========================================================
# profit_12m_mde below is 12-month cumulative profit (post-decay,
# pre-investment), assuming the true effect equals the MEI (which is also the MDE here).
mde        = 0.80         # R$/user/month (stakeholder MEI used as target MDE)
mau        = 5e6          # 5M monthly active users
investment = 4e6          # R$4M build + maintain
decay      = 0.10         # 10% monthly decay
horizon    = 12           # months

# Breakeven effect: minimum per-user-per-month lift that covers investment over decayed horizon
effective_months = sum((1 - decay) ** t for t in range(horizon))
breakeven_effect = investment / (mau * effective_months)

# Projected 12-month cumulative profit if true effect = MEI (conservative)
profit_12m_mde = sum(mde * (1 - decay) ** t for t in range(horizon)) * mau

roi_mde = (profit_12m_mde - investment) / investment

print(f"\nBreakeven effect: R${breakeven_effect:.2f}/user/month")
print(f"MEI (= target MDE): R${mde:.2f}/user/month")
print(f"MEI to breakeven ratio: {mde / breakeven_effect:.1f}x above breakeven\n")

# If the true effect equals the MEI (conservative), what does the project look like?
print(f"  12-month profit (with decay): R${profit_12m_mde / 1e6:,.0f}M")
print(f"  ROI: {roi_mde:.0f}x")


# ==========================================================
# BLOCK 6: CONFIDENCE INTERVAL -> FINANCIAL RANGE
# (Chapter section: "The p-value trap and ROI confidence intervals")
# ==========================================================
# Plug-in method: re-run the financial formula at each CI bound. ROI is a
# strictly monotone function of the effect, so this preserves coverage.
feed_ci         = [1.75, 2.93]                       # 95% CI from Chapter 3
clv_factor      = sum(0.90 ** t for t in range(12))  # 12-month decay multiplier (~7.18)
feed_profit_12m = [x * 5e6 * clv_factor for x in feed_ci]   # 5M MAU
feed_roi_ci     = [(a - 4e6) / 4e6 for a in feed_profit_12m]  # ROI at each bound

# Confidence interval -> financial range
print(f"Effect CI: [R${feed_ci[0]:.2f}, R${feed_ci[1]:.2f}] per user/month")
print(f"12-month profit range (with 10% decay): "
      f"[R${feed_profit_12m[0]/1e6:.0f}M, R${feed_profit_12m[1]/1e6:.0f}M]")
print(f"ROI range against R$4M cost: "
      f"{feed_roi_ci[0]:.0f}x to {feed_roi_ci[1]:.0f}x")


# ==========================================================
# BLOCK 7: ESTIMATING lambda FROM HOLDOUT DATA
# (Chapter section: "Appendix 13.B -- Effect decay is predictable")
# ==========================================================
# Simulate a five-month holdout where the true effect-persistence factor is
# lambda = 0.88 and recover the estimate from a simple log-linear regression.
# This is the template you'd use with real holdout data -- replace the
# simulated values with your observed per-period effects.
np.random.seed(42)
effect_0    = 2.34     # initial per-user effect (R$/month)
lambda_true = 0.88     # true monthly effect persistence
months      = np.arange(5)  # holdout measurements at months 0-4

# Simulate observed effects: true decay + measurement noise
log_effects = (np.log(effect_0)
               + months * np.log(lambda_true)
               + np.random.normal(0, 0.05, 5))

# Regress log(effect) on time to recover lambda
X = np.column_stack([np.ones(5), months])
beta = np.linalg.lstsq(X, log_effects, rcond=None)[0]
residuals = log_effects - X @ beta
mse = np.sum(residuals ** 2) / (5 - 2)
se_slope = np.sqrt(mse * np.linalg.inv(X.T @ X)[1, 1])

# Estimated lambda and 95% CI
lambda_hat = np.exp(beta[1])
t_crit = t_dist.ppf(0.975, df=len(months) - 2)
ci_lo = np.exp(beta[1] - t_crit * se_slope)
ci_hi = np.exp(beta[1] + t_crit * se_slope)

# Assumed lambda was 0.90 (the chapter's working assumption); compare to the estimate below
print(f"Estimated lambda: {lambda_hat:.2f}  [95% CI: {ci_lo:.2f}, {ci_hi:.2f}]")
