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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.special import expit  # logistic function

#########################################
# Book-wide Theme and Color Palette
#########################################
book_colors = {
    "primary": "#2E86AB",    # Steel blue - main data
    "secondary": "#A23B72",  # Magenta - secondary data
    "accent": "#F18F01",     # Orange - highlights/warnings
    "success": "#C73E1D",    # Red-orange - thresholds/targets
    "muted": "#6C757D",      # Gray - reference lines
    "light_gray": "#E5E5E5", # Light gray - backgrounds
    "dark_gray": "#4D4D4D"   # Dark gray - text
}

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.titleweight": "bold",
    "axes.labelsize": 14,
    "axes.labelcolor": "#4D4D4D",
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 20,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.color": "#E5E5E5"
})

#########################################
# Results of adjusting for good and bad controls
#########################################

# Simulation: Good and Bad Controls
np.random.seed(42)
n = 5000  # Increased N for stability

# 1. COVARIATES
user_activity = np.random.normal(50, 10, n)  # Confounder
age = np.random.normal(30, 5, n)  # Outcome Predictor
marketing = np.random.binomial(1, 0.5, n)  # Treatment Predictor
shoe_size = np.random.uniform(0, 12, n)  # Noise

# 2. TREATMENT (Premium)
# Caused by Activity (Strong Confounding) and Marketing (Strong Treatment Predictor)
# Intercept tuned to keep Premium probability balanced
prob_prem = expit(-6.5 + 0.08 * user_activity + 5.0 * marketing)
premium = np.random.binomial(1, prob_prem)

# 3. INTERMEDIATES
# Mediator: Ad-Free Experience (Premium users get it)
ad_free = np.where(premium == 1, 1, np.random.binomial(1, 0.1, n))

# 4. OUTCOME (Engagement)
# TRUE EFFECT of Premium = 9.2 (2 direct + 0.9*8 via Ad-Free)
# Also affected by Activity (Strong Confounding) and Age (Strong Outcome Predictor)
# Intercept tuned to ensure positive engagement values for collider logic
base_eng = 20 + 2.0 * user_activity - 3.0 * age
engagement = base_eng + 2 * premium + 8 * ad_free + np.random.normal(0, 3.5, n)

# 5. COLLIDER (Support Ticket)
# Caused by Premium AND Engagement
# Tuned to induce substantial bias (Berkson's Paradox)
prob_tick = expit(-11 + 2.0 * premium + 0.2 * engagement)
ticket = np.random.binomial(1, prob_tick)

df = pd.DataFrame({
    "engagement": engagement,
    "premium": premium,
    "user_activity": user_activity,
    "ad_free": ad_free,
    "ticket": ticket,
    "age": age,
    "marketing": marketing,
    "shoe_size": shoe_size
})

# Saving data as CSV
# df.to_csv("/Users/robsontigre/Desktop/everyday-ci/data/engagement.csv", index=False)

# --- RUNNING THE REGRESSIONS ---

# Reading the data
df = pd.read_csv("/Users/robsontigre/Desktop/everyday-ci/data/engagement.csv")


# Helper function to run OLS and extract estimate and SE for 'premium'
def get_coef(formula_vars, data):
    """Run OLS and return (estimate, SE) for 'premium' coefficient."""
    y = data["engagement"]
    X = sm.add_constant(data[formula_vars])
    model = sm.OLS(y, X).fit()
    return model.params["premium"], model.bse["premium"]


# Case 1: Naive (No Controls)
est1, se1 = get_coef(["premium"], df)

# Case 2: Good Control (Confounder)
est2, se2 = get_coef(["premium", "user_activity"], df)

# Case 3: Bad Control (Mediator)
est3, se3 = get_coef(["premium", "user_activity", "ad_free"], df)

# Case 4: Bad Control (Collider)
est4, se4 = get_coef(["premium", "user_activity", "ticket"], df)

# Case 5: Precision Boost (Outcome Pred)
est5, se5 = get_coef(["premium", "user_activity", "age"], df)

# Case 6: Variance Inflation (Treatment Pred)
est6, se6 = get_coef(["premium", "user_activity", "marketing"], df)

# Case 7: Noise (Neutral)
est7, se7 = get_coef(["premium", "user_activity", "shoe_size"], df)

# Print comparison
truth = 9.2

results = pd.DataFrame({
    "Model": [
        "1. Naive", "2. Good (Confounder)", "3. Bad (Mediator)",
        "4. Bad (Collider)", "5. Good (Outcome predictor)", "6. Bad (Treat predictor)",
        "7. Neutral (Noise)"
    ],
    "Estimate": [est1, est2, est3, est4, est5, est6, est7],
    "SE": [se1, se2, se3, se4, se5, se6, se7],
    "Truth": truth
})

# Bias and percentage of bias
results["Bias"] = results["Estimate"] - results["Truth"]
results["Bias_Pct"] = (100 * results["Bias"] / results["Truth"]).round(1)

# SE change relative to Good Confounder (model 2) as baseline
baseline_se = results["SE"].iloc[1]
results["SE_Change_Pct"] = (100 * (results["SE"] - baseline_se) / baseline_se).round(1)

print(results.to_string(index=False))

#########################################
# OLS misspecification: When linearity assumptions fail
#########################################

# Simulate nonlinear relationship and show misspecification bias
np.random.seed(123)
n = 500

# Continuous "intensity": emails per week (ranging 0-10)
emails = np.random.uniform(0, 10, n)

# True outcome: inverted U-shaped (quadratic) relationship
# Revenue peaks around 5 emails/week, then declines (email fatigue)
revenue = 50 + 8 * emails - 0.8 * emails**2 + np.random.normal(0, 5, n)

df_email = pd.DataFrame({"emails": emails, "revenue": revenue})

# Save to CSV for reproducibility
# df_email.to_csv("/Users/robsontigre/Desktop/everyday-ci/data/email_frequency.csv", index=False)

# Reading the data
df_email = pd.read_csv("/Users/robsontigre/Desktop/everyday-ci/data/email_frequency.csv")

# Misspecified model: linear only
X_linear = sm.add_constant(df_email["emails"])
model_linear = sm.OLS(df_email["revenue"], X_linear).fit()

# Correct model: includes quadratic term
df_email["emails_sq"] = df_email["emails"] ** 2
X_quad = sm.add_constant(df_email[["emails", "emails_sq"]])
model_quadratic = sm.OLS(df_email["revenue"], X_quad).fit()

print(model_linear.summary())
print(model_quadratic.summary())

# Compare estimates
print("Linear model (misspecified):")
print(f"  Slope estimate: {model_linear.params['emails']:.2f} (revenue per email)")

print("\nQuadratic model (correct):")
print(f"  Linear term: {model_quadratic.params['emails']:.2f}")
print(f"  Quadratic term: {model_quadratic.params['emails_sq']:.2f}")

# Visualization using book-wide aesthetics
fig, ax = plt.subplots(figsize=(10, 7))

# Scatter points
ax.scatter(df_email["emails"], df_email["revenue"], alpha=0.3, color=book_colors["muted"], s=30)

# Fitted values for plotting
x_plot = np.linspace(0, 10, 100)
y_linear = model_linear.params["const"] + model_linear.params["emails"] * x_plot
y_quad = (model_quadratic.params["const"] + 
          model_quadratic.params["emails"] * x_plot + 
          model_quadratic.params["emails_sq"] * x_plot**2)

ax.plot(x_plot, y_linear, color=book_colors["secondary"], linewidth=1.5, label="Linear (misspecified)")
ax.plot(x_plot, y_quad, color=book_colors["primary"], linewidth=1.5, label="Quadratic (correct)")

# Annotations
ax.annotate("Linear model:\nSlope ≈ 0\n(misleading!)", xy=(8, 68), fontsize=12, 
            color=book_colors["secondary"], ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none"))
ax.annotate("True peak:\n~5 emails/week", xy=(5, 55), fontsize=12, 
            color=book_colors["primary"], ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none"))

ax.set_xlabel("Emails per week", fontsize=17)
ax.set_ylabel("Weekly revenue (R$)", fontsize=17)
ax.set_title("OLS misspecification: linear model misses the true relationship", fontweight="bold")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
ax.text(0.5, -0.22, "Takeaway: When the true relationship is nonlinear, a linear model can be severely misleading.",
        transform=ax.transAxes, ha="center", fontsize=10, color="grey")

# plt.tight_layout()
# plt.savefig("/Users/robsontigre/Desktop/everyday-causal-inference/images/non_linearity.png", dpi=300, bbox_inches="tight")
# plt.close()


#########################################
# Simpson's Paradox: When confounding assumptions fail
#########################################

# Simulate Simpson's Paradox with CONTINUOUS X variable
# Example: Advertising spend vs. sales, confounded by store location
np.random.seed(42)
n = 200

# Confounder: Store location (high-traffic vs low-traffic)
# High-traffic stores: prime locations with natural foot traffic, spend less on ads
# Low-traffic stores: struggling locations, must spend heavily on ads to attract customers
location = np.random.choice(["high_traffic", "low_traffic"], n, p=[0.5, 0.5])

# X variable (continuous): Advertising spend per week (R$1000s)
# Low-traffic stores: spend MORE on ads (mean = 8) - they need it to survive
# High-traffic stores: spend LESS on ads (mean = 2) - customers find them naturally
ad_spend = np.where(
    location == "low_traffic",
    np.random.normal(8, 1.5, n),
    np.random.normal(2, 1, n)
)
ad_spend = np.maximum(ad_spend, 0.5)  # Floor at 0.5

# Outcome: Weekly sales (R$1000s)
# Both location types benefit from ads (+3 sales per R$1000 ad spend)
# But high-traffic stores have HIGHER baseline sales (70 vs 35)
sales = np.where(location == "high_traffic", 70, 35) + 3 * ad_spend + np.random.normal(0, 4, n)

df_simpson = pd.DataFrame({"location": location, "ad_spend": ad_spend, "sales": sales})

# Save data as CSV
# df_simpson.to_csv("/Users/robsontigre/Desktop/everyday-ci/data/simpsons_paradox.csv", index=False)

# Reading the data
df_simpson = pd.read_csv("/Users/robsontigre/Desktop/everyday-ci/data/simpsons_paradox.csv")

# Aggregate analysis (wrong - shows NEGATIVE relationship!)
X_agg = sm.add_constant(df_simpson["ad_spend"])
model_aggregate = sm.OLS(df_simpson["sales"], X_agg).fit()
print(f"Aggregate slope: {model_aggregate.params['ad_spend']:.2f} (appears negative!)")

# Stratified analysis (correct - shows POSITIVE effect in each group)
df_high = df_simpson[df_simpson["location"] == "high_traffic"]
df_low = df_simpson[df_simpson["location"] == "low_traffic"]

X_high = sm.add_constant(df_high["ad_spend"])
model_high = sm.OLS(df_high["sales"], X_high).fit()

X_low = sm.add_constant(df_low["ad_spend"])
model_low = sm.OLS(df_low["sales"], X_low).fit()

print(f"High-traffic stores slope: {model_high.params['ad_spend']:.2f}")
print(f"Low-traffic stores slope: {model_low.params['ad_spend']:.2f}")

# Regression with location covariate (correct)
df_simpson["location_dummy"] = (df_simpson["location"] == "low_traffic").astype(int)
X_correct = sm.add_constant(df_simpson[["ad_spend", "location_dummy"]])
model_correct = sm.OLS(df_simpson["sales"], X_correct).fit()
print(model_correct.summary())

# Visualization: Regression lines by group vs aggregate
fig, ax = plt.subplots(figsize=(10, 7))

# Points colored by location
colors_map = {"high_traffic": book_colors["primary"], "low_traffic": book_colors["secondary"]}
for loc in ["high_traffic", "low_traffic"]:
    subset = df_simpson[df_simpson["location"] == loc]
    ax.scatter(subset["ad_spend"], subset["sales"], alpha=0.6, s=60, 
               color=colors_map[loc], label=f"{'High' if loc == 'high_traffic' else 'Low'}-traffic")

# Aggregate regression line (wrong)
x_plot = np.linspace(0.5, 12, 100)
y_agg = model_aggregate.params["const"] + model_aggregate.params["ad_spend"] * x_plot
ax.plot(x_plot, y_agg, color=book_colors["accent"], linewidth=1.5, linestyle="--", label="Aggregate (wrong)")

# Stratified regression lines (correct)
y_high = model_high.params["const"] + model_high.params["ad_spend"] * x_plot
y_low = model_low.params["const"] + model_low.params["ad_spend"] * x_plot
ax.plot(x_plot, y_high, color=book_colors["primary"], linewidth=1.2, linestyle="-")
ax.plot(x_plot, y_low, color=book_colors["secondary"], linewidth=1.2, linestyle="-")

# Coefficient annotations
ax.annotate(f"Aggregate slope: {model_aggregate.params['ad_spend']:.2f}", 
            xy=(5, 70), fontsize=11, fontweight="bold", color=book_colors["accent"], ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none"))
ax.annotate(f"High-traffic: +{model_high.params['ad_spend']:.2f}", 
            xy=(1.5, 68), fontsize=11, fontweight="bold", color=book_colors["primary"], ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none"))
ax.annotate(f"Low-traffic: +{model_low.params['ad_spend']:.2f}", 
            xy=(9.5, 52), fontsize=11, fontweight="bold", color=book_colors["secondary"], ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none"))

ax.set_xlabel("Ad spend (R$1000s per week)")
ax.set_ylabel("Weekly sales (R$1000s)")
ax.set_title("Simpson's paradox: aggregate trend reverses the true effect", fontweight="bold")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
ax.text(0.5, -0.22, "Takeaway: Ignoring confounders (store location) can completely reverse your conclusions.",
        transform=ax.transAxes, ha="center", fontsize=10, color="grey")

plt.show()

# plt.tight_layout()
# plt.savefig("/Users/robsontigre/Desktop/everyday-causal-inference/images/simpsons_paradox.png", dpi=300, bbox_inches="tight")
# plt.close()
