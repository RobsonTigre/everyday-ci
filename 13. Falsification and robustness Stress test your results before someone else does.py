##########################################################
## Everyday Causal Inference: How to estimate, test, and explain impacts with R and Python
## www.everydaycausal.com
## Copyright (c) 2025 by Robson Tigre. All rights reserved.
## You may read, run, adapt, and cite this code, provided you credit the source.
## It should not be used to create competing educational or commercial products
##########################################################
## Code for Chapter 13 - Falsification and robustness: Stress test your results before someone else does
## Created: Aug 07, 2026
## Last modified: Aug 12, 2026
##########################################################

# ==========================================================
# SETUP
# ==========================================================

# If you haven't already, run this in your terminal to install the packages:
# pip install pandas numpy matplotlib networkx statsmodels PySensemakr scikit-learn scipy
# (or use "pip3"; from the repository root, "pip install -r requirements.txt"
# installs the exact pinned versions these results were produced with)

# You must run the lines below at the start of every new Python session.
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt                    # balance and overlap plots
import networkx as nx                              # DAG d-separation checks
import statsmodels.api as sm                       # formula-based binomial GLM
import statsmodels.formula.api as smf              # OLS with a formula interface
import sensemakr                                   # sensitivity analysis (PySensemakr)
from matplotlib.ticker import MaxNLocator
from scipy.stats import norm
from scipy.stats import t as student_t
from sensemakr.bias_functions import adjusted_estimate  # contour without plot()
from sklearn.neighbors import NearestNeighbors     # nearest-neighbour matching

# ==========================================================
# Book-wide Theme and Color Palette
# ==========================================================
# Define a consistent color palette (colorblind-friendly)
book_colors = {
    'primary': '#2E86AB',    # Steel blue - main data
    'secondary': '#A23B72',  # Magenta - secondary data
    'accent': '#F18F01',     # Orange - highlights/warnings
    'success': '#C73E1D',    # Red-orange - thresholds/targets
    'muted': '#6C757D',      # Gray - reference lines
    'light_gray': '#E5E5E5', # Light gray - backgrounds
    'dark_gray': '#4D4D4D'   # Dark gray - text
}

def setup_book_style():
    """Apply consistent styling to matplotlib plots"""
    plt.rcParams.update({
        'figure.figsize': (10, 7),
        'figure.dpi': 100,
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.titleweight': 'bold',
        'axes.labelsize': 14,
        'axes.labelcolor': '#4D4D4D',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.color': '#E5E5E5',
        'legend.fontsize': 11,
        'legend.frameon': False,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
    })

setup_book_style()

# ==========================================================
# Read the frozen dataset
# ==========================================================
# R generates and freezes data/free_shipping_banner.csv (see the companion
# R script's DGP block); Python only ever reads that same frozen file, so
# R and Python never diverge on different RNGs and produce different numbers.

df = pd.read_csv("data/free_shipping_banner.csv")

# ==========================================================
# Shared reporting helpers
# ==========================================================
# Every regression result in this script prints the estimate, the standard
# error, and a 95% Student-t interval built from the model's residual degrees
# of freedom. The R companion prints the identical schema, so the two
# languages can be compared line by line.

def report_regression(label, model, term="treated", covariance="HC1"):
    """Print a robust (HC1 or HC3) estimate with a Student-t interval."""
    estimate = model.params[term]
    se = model.bse[term]
    critical = student_t.ppf(0.975, model.df_resid)
    print(
        f"{label}: estimate = {estimate:.4f}; {covariance} SE = {se:.4f}; "
        f"95% Student-t CI = [{estimate - critical * se:.4f}, "
        f"{estimate + critical * se:.4f}]"
    )

def report_classical(label, model, term="treated"):
    """Print a classical (homoskedastic) estimate with a Student-t interval."""
    estimate = model.params[term]
    se = model.bse[term]
    critical = student_t.ppf(0.975, model.df_resid)
    print(
        f"{label}: estimate = {estimate:.4f}; classical SE = {se:.4f}; "
        f"95% Student-t CI = [{estimate - critical * se:.4f}, "
        f"{estimate + critical * se:.4f}]"
    )

def report_correlation(label, x, y):
    """Print a Pearson correlation with a 95% Fisher interval."""
    keep = x.notna() & y.notna()
    x, y = x[keep], y[keep]
    r = x.corr(y, method="pearson")
    delta = norm.ppf(0.975) / (len(x) - 3) ** 0.5
    lower = np.tanh(np.arctanh(r) - delta)
    upper = np.tanh(np.arctanh(r) + delta)
    print(
        f"{label}: Pearson r = {r:.4f}; "
        f"95% Fisher CI = [{lower:.4f}, {upper:.4f}]"
    )

CORE = "treated + prior_orders + prior_profit + sessions_pre"

# ==========================================================
# ANALYSIS STAGES: the raw comparison and the intermediate core model
# ==========================================================
# The two numbers the chapter opens with, and the simulated truths they are
# graded against. Both interval rows use classical (homoskedastic) standard
# errors; every diagnostic below uses HC1 unless it says otherwise.

report_classical("raw comparison", smf.ols("profit_30d ~ treated", df).fit())
report_classical("intermediate core controls",
                 smf.ols(f"profit_30d ~ {CORE}", df).fit())
# raw comparison: estimate = 7.8558; classical SE = 0.1373; 95% Student-t CI = [7.5867, 8.1249]
# intermediate core controls: estimate = 2.1866; classical SE = 0.1331; 95% Student-t CI = [1.9257, 2.4475]

# Simulated truths. true_effect_proxy is the per-user ground truth that ships
# in the public CSV; the latent confounder itself never does.
print("targeted-user ATT truth  =", df.loc[df.treated == 1, "true_effect_proxy"].mean())
print("whole-base ATE truth     =", df["true_effect_proxy"].mean())
print("untreated ATC truth      =", df.loc[df.treated == 0, "true_effect_proxy"].mean())
print("eligible-targeted truth  =",
      df.loc[(df.treated == 1) & (df.shipping_eligible == 1), "true_effect_proxy"].mean())
print("ineligible-targeted truth=",
      df.loc[(df.treated == 1) & (df.shipping_eligible == 0), "true_effect_proxy"].mean())
# 0.9877 targeted ATT; 0.8981 whole-base ATE; 0.8176 untreated ATC;
# 1.4982 eligible-targeted; 0.0000 ineligible-targeted

# ==========================================================
# PLACEBO IN TIME: a "fake effect" before the banner existed
# ==========================================================
# The banner cannot have changed anything before it launched. Swap the
# outcome for the pre-period profit and re-run the intermediate spec; the
# 'treated' coefficient should be ~0 if the comparison is clean.
# It is not: intermediate placebo-in-time = R$1.59 [R$1.32, R$1.86].

placebo_time = smf.ols(f"profit_pre ~ {CORE}", data=df).fit(cov_type="HC1")
report_regression("intermediate placebo in time", placebo_time)
# intermediate placebo in time: estimate = 1.5918; HC1 SE = 0.1366; 95% Student-t CI = [1.3239, 1.8596]

# ==========================================================
# NEGATIVE-CONTROL OUTCOMES: an effect where the banner cannot reach
# ==========================================================
# Swap the outcome for spend in categories the free-shipping offer does not
# apply to. The banner has no mechanism to move it, so treated should be ~0.
# It is not: intermediate negative-control = R$1.11 [R$0.95, R$1.28].

negctrl = smf.ols(
    f"noneligible_category_profit ~ {CORE}", data=df
).fit(cov_type="HC1")
report_regression("intermediate negative control", negctrl)
# intermediate negative control: estimate = 1.1148; HC1 SE = 0.0838; 95% Student-t CI = [0.9505, 1.2791]

# ==========================================================
# PLACEBO TREATMENTS: an effect from an assignment that did nothing
# ==========================================================
# Two checks: a deterministic null permutation (destroys targeting
# structure entirely) and a score-preserving fake exposure (keeps the
# selection pressure in the targeting score, but among users who were
# never actually treated). Both condition on the real 'treated' indicator.

LCG_MULTIPLIER = 104729
LCG_OFFSET = 12345
n = len(df)

# The fake label gets the same share of the untreated pool that real exposure
# has of the whole base: floor(14209 / 30000 * 15791) = 7479.
FAKE_EXPOSURE_N = int(np.floor(df["treated"].sum() / n * (df["treated"] == 0).sum()))
assert FAKE_EXPOSURE_N == 7479

# A deterministic null permutation, identical in R and Python.
i0 = np.arange(n, dtype=np.int64)
idx = (LCG_MULTIPLIER * i0 + LCG_OFFSET) % n
assert np.unique(idx).size == n
df["placebo_null"] = df["treated"].to_numpy()[idx]

# Preserve the score-based selection pattern among untreated users.
ordered = (
    df[df["treated"] == 0]
    .sort_values(["predicted_purchase_score", "user_id"],
                 ascending=[False, True], kind="mergesort")
)
df["fake_score_exposure"] = 0
df.loc[ordered.head(FAKE_EXPOSURE_N).index, "fake_score_exposure"] = 1
assert df["fake_score_exposure"].sum() == FAKE_EXPOSURE_N

null_fit = smf.ols(f"profit_30d ~ placebo_null + {CORE}", data=df).fit(cov_type="HC1")
score_fit = smf.ols(
    f"profit_30d ~ fake_score_exposure + {CORE}", data=df
).fit(cov_type="HC1")

report_regression("deterministic null", null_fit, "placebo_null")
report_regression("score-preserving fake exposure", score_fit, "fake_score_exposure")
# deterministic null: estimate = -0.0626; HC1 SE = 0.1197; 95% Student-t CI = [-0.2972, 0.1720]
# score-preserving fake exposure: estimate = 1.5868; HC1 SE = 0.1838; 95% Student-t CI = [1.2266, 1.9471]

# ==========================================================
# PLACEBO UNITS: groups the banner could not have helped
# ==========================================================
# shipping_eligible == 0 users cannot redeem free shipping, so their true
# banner effect is R$0.00 by construction. A design that removes selection
# should return roughly zero for their banner contrast.

ineligible = df[df["shipping_eligible"] == 0].copy()

# 1) The observational estimator on the placebo units.
#    Need NOT be zero: selection still operates among the ineligible.
obs_placebo = smf.ols(f"profit_30d ~ {CORE}", data=ineligible).fit(cov_type="HC1")
report_regression("ineligible observational contrast", obs_placebo)
# ineligible observational contrast: estimate = 1.1231; HC1 SE = 0.2146; 95% Student-t CI = [0.7025, 1.5437]

# 2) Permutation calibration: fake the assignment within a group
#    that cannot benefit, and see how often we reject the null.
rng = np.random.default_rng(42)
reject = []
for _ in range(1000):
    ineligible["fake"] = rng.permutation(ineligible["treated"].values)
    m = smf.ols(
        "profit_30d ~ fake + prior_orders + prior_profit + sessions_pre",
        data=ineligible,
    ).fit(cov_type="HC1")
    reject.append(abs(m.tvalues["fake"]) > student_t.ppf(0.975, m.df_resid))
print(np.mean(reject))   # 0.056: a different RNG stream, also near 0.05

# ==========================================================
# BALANCE AND OVERLAP: are we comparing comparable users?
# ==========================================================
# Standardized mean differences (SMDs) put covariate gaps on a common
# scale; propensity-score overlap asks whether every kind of user appears
# in both groups. Good observed balance is necessary, not sufficient --
# it says nothing about the latent shopping_intent no column captures.

PROPENSITY = (
    "treated ~ prior_orders + prior_profit + sessions_pre + "
    "predicted_purchase_score + tenure_days + shipping_eligible"
)
VARS = ["prior_orders", "prior_profit", "sessions_pre",
        "predicted_purchase_score", "tenure_days", "shipping_eligible"]

# Formula-based, unpenalized binomial GLM, matching the R propensity model.
ps_model = smf.glm(PROPENSITY, data=df,
                   family=sm.families.Binomial()).fit()
df["ps"] = ps_model.predict(df)

# Overlap: the estimated propensity distribution in each group.
fig, ax = plt.subplots()
for label, group in df.groupby("treated"):
    group["ps"].plot.density(ax=ax, label=f"treated={label}")
ax.set(xlabel="Estimated probability of targeting", ylabel="Density")
ax.legend()

treated_df, controls_df = df[df.treated == 1].copy(), df[df.treated == 0].copy()
match_idx = (NearestNeighbors(n_neighbors=1)
       .fit(controls_df[["ps"]])
       .kneighbors(treated_df[["ps"]], return_distance=False)
       .ravel())

# Keep each unique control once and weight it by its raw reuse count.
control_rows, reuse = np.unique(controls_df.iloc[match_idx].index, return_counts=True)
matched_controls = controls_df.loc[control_rows].copy()
matched_controls["match_weight"] = reuse
treated_df["match_weight"] = 1
assert len(matched_controls) == 6017
ess = reuse.sum() ** 2 / np.square(reuse).sum()  # 2727.325
print("unique matched controls =", len(matched_controls), "; control ESS =", round(ess, 3))

# Balance before and after matching. SMDs use the treated group's SD
# because the target is the ATT.
before_smd, after_smd = {}, {}
for col in VARS:
    treated_sd = treated_df[col].std(ddof=1)
    before_smd[col] = (treated_df[col].mean() - controls_df[col].mean()) / treated_sd
    matched_control_mean = np.average(
        matched_controls[col], weights=matched_controls["match_weight"]
    )
    after_smd[col] = (treated_df[col].mean() - matched_control_mean) / treated_sd

balance = pd.DataFrame({
    "Unmatched": before_smd,
    "Matched (ATT)": after_smd,
})
print(balance)
ax = balance.abs().plot.barh()
ax.axvline(0.10, color="black", linestyle="--")
ax.set(xlabel="Absolute standardized mean difference", ylabel="")
print("largest matched |SMD| =", round(max(balance["Matched (ATT)"].abs()), 6))
# largest matched |SMD| = 0.026281

# ==========================================================
# D-SEPARATION: the falsification you do not yet do
# ==========================================================
# A candidate DAG implies conditional independencies that can sometimes be
# checked in observed data. A failed check is evidence against that
# maintained graph-plus-model implication, not a full verdict on the story.

# networkx renamed d_separated to is_d_separator in 3.3; support both.
# Note: the `or` form matters — getattr(nx, "is_d_separator", nx.d_separated)
# evaluates the default eagerly and breaks on 3.5+, where d_separated is gone.
is_d_separator = getattr(nx, "is_d_separator", None) or nx.d_separated

# Candidate graph under test; verify its four implications before testing them.
roots = ["prior_orders", "prior_profit", "sessions_pre"]
g = nx.DiGraph()
g.add_edges_from(
    [(x, "predicted_purchase_score") for x in roots]
    + [(x, "treated") for x in roots]
    + [(x, "profit_30d") for x in roots]
    + [("predicted_purchase_score", "treated"),
       ("treated", "profit_30d")]
)
implications = [
    ("prior_orders _||_ prior_profit", "prior_orders", "prior_profit", set()),
    ("prior_orders _||_ sessions_pre", "prior_orders", "sessions_pre", set()),
    ("prior_profit _||_ sessions_pre", "prior_profit", "sessions_pre", set()),
    ("predicted_purchase_score _||_ profit_30d after intermediate conditioning",
     "predicted_purchase_score", "profit_30d", {"treated", *roots}),
]
for label, left, right, given in implications:
    assert is_d_separator(g, {left}, {right}, given), label

# Report the three marginal implications first.
report_correlation("prior_orders _||_ prior_profit",
                   df["prior_orders"], df["prior_profit"])
report_correlation("prior_orders _||_ sessions_pre",
                   df["prior_orders"], df["sessions_pre"])
report_correlation("prior_profit _||_ sessions_pre",
                   df["prior_profit"], df["sessions_pre"])

# Primary implication: residualize score and profit on exposure and core controls.
ctrl = CORE
r_score  = smf.ols(f"predicted_purchase_score ~ {ctrl}", data=df).fit().resid
r_profit = smf.ols(f"profit_30d ~ {ctrl}", data=df).fit().resid

report_correlation("predicted_purchase_score _||_ profit_30d after intermediate conditioning",
                   r_score, r_profit)
# prior_orders _||_ prior_profit: Pearson r = 0.7343; 95% Fisher CI = [0.7290, 0.7395]
# prior_orders _||_ sessions_pre: Pearson r = 0.2166; 95% Fisher CI = [0.2058, 0.2274]
# prior_profit _||_ sessions_pre: Pearson r = 0.3655; 95% Fisher CI = [0.3557, 0.3753]
# predicted_purchase_score _||_ profit_30d after intermediate conditioning: Pearson r = 0.1188; 95% Fisher CI = [0.1076, 0.1300]

# ==========================================================
# ALIGN THE FOCAL MODEL WITH THE TARGETED-USER ATT
# ==========================================================
# The intermediate model assumes one constant effect even though eligibility
# determines whether free shipping can matter. Add treated * eligibility,
# centered so the 'treated' coefficient reads as the targeted-population
# contrast: R$1.87 [R$1.60, R$2.13] with HC1.

treated_eligibility_share = df.loc[
    df["treated"] == 1, "shipping_eligible"
].mean()
df["eligibility_centered"] = (
    df["shipping_eligible"] - treated_eligibility_share
)
FOCAL_RHS = ("treated * eligibility_centered + "
             "prior_orders + prior_profit + sessions_pre")

focal = smf.ols(f"profit_30d ~ {FOCAL_RHS}", data=df).fit(cov_type="HC1")
report_regression("focal targeted-standardized contrast", focal)
# focal targeted-standardized contrast: estimate = 1.8657; HC1 SE = 0.1341; 95% Student-t CI = [1.6028, 2.1285]

raw = smf.ols(
    "profit_30d ~ treated * shipping_eligible "
    "+ prior_orders + prior_profit + sessions_pre",
    data=df,
).fit(cov_type="HC1")
# Both parameterizations are the same fit with different labels.
assert (focal.fittedvalues - raw.fittedvalues).abs().max() < 1e-9
df["ineligibility"] = 1 - df["shipping_eligible"]
raw_eligible = smf.ols(
    "profit_30d ~ treated * ineligibility "
    "+ prior_orders + prior_profit + sessions_pre",
    data=df,
).fit(cov_type="HC1")
report_regression("raw ineligible-user contrast", raw)
report_regression("raw treatment-by-eligibility interaction", raw,
                  "treated:shipping_eligible")
report_regression("raw eligible-user contrast", raw_eligible)
# raw ineligible-user contrast: estimate = 1.2491; HC1 SE = 0.2006; 95% Student-t CI = [0.8558, 1.6423]
# raw treatment-by-eligibility interaction: estimate = 0.9353; HC1 SE = 0.2444; 95% Student-t CI = [0.4563, 1.4143]
# raw eligible-user contrast: estimate = 2.1844; HC1 SE = 0.1638; 95% Student-t CI = [1.8634, 2.5054]

# ==========================================================
# FOCAL RECHECK: rerun every applicable diagnostic on the focal model
# ==========================================================
# The placebo-treatment rows add the placebo indicator as one more regressor
# on the focal right-hand side. Placebo-in-time and negative-control keep the
# focal specification and swap only the outcome. The ineligible placebo-unit
# recheck is the "raw ineligible-user contrast" printed just above: the same
# fitted model written with raw shipping_eligible instead of the centered
# version, which reads the contrast at eligibility zero.

report_regression("focal placebo in time",
                  smf.ols(f"profit_pre ~ {FOCAL_RHS}", df).fit(cov_type="HC1"))
report_regression("focal negative control",
                  smf.ols(f"noneligible_category_profit ~ {FOCAL_RHS}", df).fit(cov_type="HC1"))
report_regression("focal deterministic null",
                  smf.ols(f"profit_30d ~ placebo_null + {FOCAL_RHS}", df).fit(cov_type="HC1"),
                  "placebo_null")
report_regression("focal score-preserving fake exposure",
                  smf.ols(f"profit_30d ~ fake_score_exposure + {FOCAL_RHS}", df).fit(cov_type="HC1"),
                  "fake_score_exposure")
report_correlation("predicted_purchase_score _||_ profit_30d after focal conditioning",
                   smf.ols(f"predicted_purchase_score ~ {FOCAL_RHS}", df).fit().resid,
                   smf.ols(f"profit_30d ~ {FOCAL_RHS}", df).fit().resid)
# focal placebo in time: estimate = 1.6037; HC1 SE = 0.1384; 95% Student-t CI = [1.3325, 1.8750]
# focal negative control: estimate = 1.1267; HC1 SE = 0.0849; 95% Student-t CI = [0.9604, 1.2931]
# focal deterministic null: estimate = -0.0507; HC1 SE = 0.1187; 95% Student-t CI = [-0.2835, 0.1820]
# focal score-preserving fake exposure: estimate = 1.3766; HC1 SE = 0.1828; 95% Student-t CI = [1.0182, 1.7349]
# predicted_purchase_score _||_ profit_30d after focal conditioning: Pearson r = 0.1058; 95% Fisher CI = [0.0946, 0.1170]

# ==========================================================
# ROBUSTNESS GRID: six defensible specifications, one lever at a time
# ==========================================================
# Core controls, context covariates, the targeting score, common-support
# trimming, ATT weighting, and 1:1 nearest-neighbour matching -- read this
# grid for effect-size stability, never as an average.

core_rhs = FOCAL_RHS
ctx_rhs = core_rhs + " + tenure_days + C(loyalty_tier) + C(region) + C(device)"

adjusted = smf.ols(f"profit_30d ~ {core_rhs}", df).fit(cov_type="HC1")
context  = smf.ols(f"profit_30d ~ {ctx_rhs}", df).fit(cov_type="HC1")
score    = smf.ols(f"profit_30d ~ {ctx_rhs} + predicted_purchase_score", df).fit(cov_type="HC1")

# Trim to common support on the targeting score, then re-fit the core spec.
score_t, score_c = df.loc[df.treated == 1, "predicted_purchase_score"], \
                   df.loc[df.treated == 0, "predicted_purchase_score"]
cs_lo, cs_hi = max(score_t.min(), score_c.min()), min(score_t.max(), score_c.max())
df_trim = df[df.predicted_purchase_score.between(cs_lo, cs_hi)]
trimmed = smf.ols(f"profit_30d ~ {core_rhs}", df_trim).fit(cov_type="HC1")

# Formula-based unpenalized binomial GLM for IPW and matching.
ps_model2 = smf.glm(PROPENSITY, data=df,
                   family=sm.families.Binomial()).fit()
ps = ps_model2.predict(df)
wt = np.where(df["treated"] == 1, 1.0, ps / (1 - ps))
ipw = smf.wls(f"profit_30d ~ {core_rhs}", df, weights=wt).fit(cov_type="HC1")
ipw_placebo_time = smf.wls(
    f"profit_pre ~ {core_rhs}", df, weights=wt
).fit(cov_type="HC1")
ipw_negative_control = smf.wls(
    f"noneligible_category_profit ~ {core_rhs}", df, weights=wt
).fit(cov_type="HC1")

# Match with replacement; keep each control once with its raw reuse count.
df = df.assign(ps=ps)
t_df, c_df = df[df.treated == 1], df[df.treated == 0]
grid_idx = (NearestNeighbors(n_neighbors=1).fit(c_df[["ps"]])
       .kneighbors(t_df[["ps"]], return_distance=False).ravel())
control_rows2, reuse2 = np.unique(c_df.iloc[grid_idx].index, return_counts=True)
matched_controls2 = c_df.loc[control_rows2].copy()
matched_controls2["match_weight"] = reuse2
t_df = t_df.copy(); t_df["match_weight"] = 1
matched_grid = pd.concat([t_df, matched_controls2])
nn = smf.wls(f"profit_30d ~ {core_rhs}", matched_grid,
             weights=matched_grid["match_weight"]).fit(cov_type="HC3")

# Read the six targeted-standardized estimates; never duplicate reused controls.
for name, m in [("adjusted", adjusted), ("context", context),
                ("score", score), ("trimmed", trimmed), ("ipw", ipw),
                ("nn", nn)]:
    covariance = "HC3" if name == "nn" else "HC1"
    report_regression(name, m, "treated", covariance=covariance)
report_regression("IPW placebo in time", ipw_placebo_time)
report_regression("IPW noneligible-category placebo", ipw_negative_control)
print(
    "common-support trim: "
    f"{len(df) - len(df_trim)} rows removed; "
    f"{df.treated.sum() - df_trim.treated.sum()} treated removed; "
    f"targeted truth = {df_trim.loc[df_trim.treated == 1, 'true_effect_proxy'].mean():.6f}"
)
# adjusted estimates: 1.87, 1.86, 1.21, 1.86, 1.22, 1.08
# IPW placebo in time: estimate = 0.6296; HC1 SE = 0.1733; 95% Student-t CI = [0.2900, 0.9692]
# IPW noneligible-category placebo: estimate = 0.3902; HC1 SE = 0.1047; 95% Student-t CI = [0.1849, 0.5955]
# common-support trim: 18 rows removed; 10 treated removed; targeted truth = 0.987911

# Changing the population changes the question. Restricting the core model to
# shipping-eligible users is graded against the eligible-targeted truth
# (R$1.50), not the R$0.99 truth for all targeted users, so it is NOT part of
# the six-specification same-population grid above.
report_regression("eligible-only core",
                  smf.ols(f"profit_30d ~ {CORE}",
                          df[df.shipping_eligible == 1]).fit(cov_type="HC1"))
# eligible-only core: estimate = 2.2664; HC1 SE = 0.1707; 95% Student-t CI = [1.9317, 2.6010]

# ==========================================================
# SENSITIVITY TO UNOBSERVED CONFOUNDING
# ==========================================================
# The focal model still rests on the assumption that observed controls
# capture the selection the targeting system acted on. Sensitivity
# analysis bounds how strong a hidden confounder would have to be to
# overturn the result. PySensemakr installs as PySensemakr but imports as
# sensemakr.

model_sens = smf.ols(f"profit_30d ~ {FOCAL_RHS}", data=df).fit()

# Sensitivity of the `treated` coefficient, benchmarked against the
# strongest observed control, prior_profit.
sens = sensemakr.Sensemakr(
    model_sens, treatment="treated",
    benchmark_covariates="prior_profit", kd=[1]
)
sens.summary()
# RV = 0.077; RV-alpha = 0.067; treatment partial R2 = 0.0065
# formal 1x prior-profit bound: R$-1.19 [R$-1.44, R$-0.94]

# PySensemakr 0.0.8's own sens.plot() raises on matplotlib 3.10 and later
# (upstream issue nlapier2/PySensemakr#43), so draw the contour from the
# package's bias functions. R's plot(sens) still works and needs no equivalent.
# Each line is the treated coefficient after adjusting for a confounder of that
# strength; the red line is where the coefficient hits zero.
grid_x = np.linspace(0, 0.12, 60)
grid_y = np.linspace(0, 0.19, 60)
z = np.array([[adjusted_estimate(model=model_sens, treatment="treated",
                                 r2dz_x=x, r2yz_dx=y) for x in grid_x]
              for y in grid_y])

levels = MaxNLocator(9).tick_values(z.min(), z.max())
fig, ax = plt.subplots()
ax.clabel(ax.contour(grid_x, grid_y, z, colors="grey",
                     levels=levels[levels != 0]), fmt="%1.3g", fontsize=8)
ax.contour(grid_x, grid_y, z, colors="red", linestyles="dashed", levels=[0])
bound = sens.bounds.iloc[0]
ax.plot(bound["r2dz_x"], bound["r2yz_dx"], "s", color="red")
ax.annotate(f"{bound['bound_label']}: R${bound['adjusted_estimate']:.2f}",
            (bound["r2dz_x"], bound["r2yz_dx"]),
            textcoords="offset points", xytext=(6, 6), color="red", fontsize=8)
ax.set(xlabel="Partial R^2 of confounder with treatment",
       ylabel="Partial R^2 of confounder with outcome")
# unadjusted (0,0) = 1.865679; 1x prior_profit bound = -1.187655

# Closed form from the t-stat and df alone (classical SEs):
t_stat = model_sens.tvalues["treated"]; dfree = model_sens.df_resid
partial_r2 = t_stat**2 / (t_stat**2 + dfree)         # 0.0065
f_stat = t_stat / np.sqrt(dfree)
rv = 0.5 * (np.sqrt(f_stat**4 + 4 * f_stat**2) - f_stat**2)   # 0.077

# ==========================================================
# MECHANISM AND SPECIFICITY: does the effect show up only where it can?
# ==========================================================
# The mechanism predicts a larger contrast where free shipping can act
# than where it cannot. Fit the same focal model to eligible-category
# profit, noneligible-category profit, and their per-user difference.

df["category_difference"] = (
    df["eligible_category_profit"] - df["noneligible_category_profit"]
)

elig = smf.ols(f"eligible_category_profit ~ {FOCAL_RHS}", df).fit(cov_type="HC1")
nonelig = smf.ols(f"noneligible_category_profit ~ {FOCAL_RHS}", df).fit(cov_type="HC1")
difference = smf.ols(f"category_difference ~ {FOCAL_RHS}", df).fit(cov_type="HC1")

for name, model in [("eligible", elig), ("noneligible", nonelig),
                    ("difference", difference)]:
    report_regression(name, model)
# eligible: estimate = 1.4938; HC1 SE = 0.0824; 95% Student-t CI = [1.3324, 1.6552]
# noneligible: estimate = 1.1267; HC1 SE = 0.0849; 95% Student-t CI = [0.9604, 1.2931]
# difference: estimate = 0.3671; HC1 SE = 0.1109; 95% Student-t CI = [0.1497, 0.5845]

# ==========================================================
# HETEROGENEITY AS FALSIFICATION: leave-one-region-out
# ==========================================================
# Re-estimate the focal model after dropping each region in turn. If one
# deletion moves the estimate sharply, the aggregate depends on that slice.

full = smf.ols(f"profit_30d ~ {FOCAL_RHS}", df).fit().params["treated"]
for r in sorted(df["region"].unique()):
    m = smf.ols(f"profit_30d ~ {FOCAL_RHS}", df[df.region != r]).fit(cov_type="HC1")
    print(r, round(m.params["treated"], 2), "vs full", round(full, 2))
# range R$1.72-R$1.95; spread R$0.23

# ==========================================================
# APPENDIX: SPECIFICATION CURVE
# ==========================================================
# When the grid of defensible choices grows large, a specification curve
# plots the estimate across every pre-declared combination rather than
# selecting one preferred path.

spec_controls = ["prior_orders", "prior_profit", "sessions_pre",
            "predicted_purchase_score", "tenure_days"]

# Loop over every subset of controls: one estimate per defensible choice.
rows = []
for k in range(len(spec_controls) + 1):
    for combo in itertools.combinations(spec_controls, k):
        rhs = " + ".join(("treated",) + combo)
        est = smf.ols(f"profit_30d ~ {rhs}", df).fit().params["treated"]
        rows.append({"controls": combo, "treated": est})

curve = pd.DataFrame(rows).sort_values("treated")  # plot this: the full spread
