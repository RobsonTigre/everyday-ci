##########################################################
## Everyday Causal Inference: How to estimate, test, and explain impacts with R and Python
## www.everydaycausal.com
## Copyright © 2025 by Robson Tigre. All rights reserved.
## You may read, share, and cite for learning purposes, provided you credit the source.
## It should not be used to create competing educational or commercial products
##########################################################
## Code for 12. Heterogeneous treatment effects: Different people, different reactions
## Created: Jun 07, 2026
## Last modified: Jul 18, 2026
##########################################################

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from econml.dml import CausalForestDML

# book-wide aesthetic (mirrors .agent/rules/plots-aesthetics.md: book_colors + theme_book)
BOOK = {
    "primary": "#2E86AB",   # steel blue - main data
    "secondary": "#A23B72", # magenta - secondary data
    "accent": "#F18F01",    # orange - highlights / thresholds
    "success": "#C73E1D",   # red-orange - fitted lines / targets
    "muted": "#6C757D",     # gray - reference lines
    "light_gray": "#E5E5E5",
    "dark_gray": "#4D4D4D",
}
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.size": 12, "axes.titlesize": 12, "axes.titleweight": "bold",
    "figure.titlesize": 12.8, "figure.titleweight": "bold",
    "axes.labelsize": 12, "axes.labelcolor": "#4D4D4D",
    "xtick.color": "#666666", "ytick.color": "#666666",
    "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
    "axes.edgecolor": "#666666", "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#E5E5E5", "grid.linewidth": 0.5,
    "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight",
})
from mcf.optpolicy_functions import OptimalPolicy
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split   # seeded two-fold cross-fit
import statsmodels.formula.api as smf                   # formula OLS for the interaction baseline

# seed: fixed so every analysis number the chapter quotes reproduces run for run.
# The R companion's identical seed regenerates the shared CSV; here it seeds the
# forests, the cross-fit splits, and the random-policy draw. Chapter display blocks
# stay seedless.
CFG = dict(seed=20260606, coupon_cost=5, budget_share=0.20, n_estimators=2000)
rng = np.random.default_rng(CFG["seed"])

df = pd.read_csv("data/coupon_allocation_experiment.csv")  # identical data to the R side
X = df[["recency", "frequency", "disc_sens", "past_spend"]].values
D = df["D"].values
Y = df["observed_profit"].values

# causal forest (DML-style nuisance adjustment is internal to the estimator)
est = CausalForestDML(
    model_y=RandomForestRegressor(n_estimators=300, min_samples_leaf=20, random_state=CFG["seed"]),
    model_t=RandomForestClassifier(n_estimators=300, min_samples_leaf=20, random_state=CFG["seed"]),
    discrete_treatment=True, n_estimators=CFG["n_estimators"], random_state=CFG["seed"],
)
est.fit(Y, D, X=X, cache_values=True)   # cache_values exposes residuals_ for the overlap check below
df["cate_forest"] = est.effect(X)

# Inner causal forest: econml has no get_forest_weights, but the fitted forest exposes
# .apply() (per-tree leaf ids), which is all we need to reconstruct grf's similarity
# weights -- the share of trees in which a user lands in the target's leaf.
inner = est.model_final_.estimators_[0]
def forest_weights(target_idx):
    leaves = inner.apply(X)                  # (n, n_trees) leaf id per tree
    tgt    = inner.apply(X[[target_idx]])    # (1, n_trees)
    same   = (leaves == tgt)                 # (n, n_trees) shares a leaf with target?
    sizes  = same.sum(0); sizes[sizes == 0] = 1
    return (same / sizes).mean(1)            # (n,) weights, sum to ~1

# Small helper reused by the TOC figure and the power checks below.
def _forest_cate(y, d, x, x_eval):
    f = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=300, min_samples_leaf=20, random_state=CFG["seed"]),
        model_t=RandomForestClassifier(n_estimators=300, min_samples_leaf=20, random_state=CFG["seed"]),
        discrete_treatment=True, n_estimators=CFG["n_estimators"], random_state=CFG["seed"],
    )
    f.fit(y, d, X=x)
    return f.effect(x_eval)

# OLS interaction baseline (formula interface, the same model the chapter displays)
ols = smf.ols("observed_profit ~ D * (recency + frequency + disc_sens + past_spend)",
              data=df).fit()
df["cate_ols"] = ols.predict(df.assign(D=1)) - ols.predict(df.assign(D=0))

print(f"True ATE = {df['true_cate'].mean():.3f}; "
      f"forest corr = {np.corrcoef(df['cate_forest'], df['true_cate'])[0,1]:.3f}; "
      f"ols corr = {np.corrcoef(df['cate_ols'], df['true_cate'])[0,1]:.3f}")

# headline experiment result: the difference-in-means ATE + 95% CI the chapter's
# opening table quotes. Reads observed_profit and D only -- deterministic,
# consumes no rng.
yt_head = df.loc[df["D"] == 1, "observed_profit"]
yc_head = df.loc[df["D"] == 0, "observed_profit"]
ate_head = yt_head.mean() - yc_head.mean()
se_head = np.sqrt(yt_head.var(ddof=1) / len(yt_head) + yc_head.var(ddof=1) / len(yc_head))
print(f"Estimated ATE (difference in means) = {ate_head:.3f}; "
      f"95% CI [{ate_head - 1.96 * se_head:.3f}, {ate_head + 1.96 * se_head:.3f}]")

# Out-of-fold CATE for every decision input below (targeting table, gain curve,
# histogram, CATE-vs-feature, mcf policy scores). effect(X) on the forest's own
# training rows is in-sample -- econml has no out-of-bag mode -- so no row may
# receive a decision score from a forest trained on that row. Seeded two-fold
# cross-fit: each half is predicted only by the forest trained on the OTHER
# half, then both blocks are restored to the original row order. (grf's
# predict() is out-of-bag, which is why the R side needs no analog of this.)
idx_a, idx_b = train_test_split(np.arange(len(df)), test_size=0.5, shuffle=True,
                                random_state=CFG["seed"])
cate_xfit = np.empty(len(df))
cate_xfit[idx_b] = _forest_cate(Y[idx_a], D[idx_a], X[idx_a], X[idx_b])
cate_xfit[idx_a] = _forest_cate(Y[idx_b], D[idx_b], X[idx_b], X[idx_a])
df["cate_xfit"] = cate_xfit

# figure 1: who benefits
fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
for ax, col, name in zip(axes, ["cate_forest", "cate_ols"], ["Causal forest", "OLS interaction"]):
    ax.scatter(df["true_cate"], df[col], s=3, alpha=0.15, color=BOOK["primary"])
    lo, hi = df["true_cate"].min(), df["true_cate"].max()
    ax.plot([lo, hi], [lo, hi], "--", color=BOOK["muted"])
    ax.set(title=name, xlabel="True CATE (R$)")
axes[0].set_ylabel("Estimated CATE (R$)")
fig.suptitle("Who benefits: forest recovers heterogeneity OLS averages away")
fig.tight_layout(); fig.savefig("images/heterogeneous-coupon-cate-python.png", dpi=300); plt.close(fig)

# figure 2: cumulative net incremental profit vs random under budget
def gain_curve(score):
    order = np.argsort(-score)
    net = df["true_cate"].values[order] - CFG["coupon_cost"]
    return np.arange(1, len(net) + 1) / len(net), np.cumsum(net)
kf, cf = gain_curve(df["cate_xfit"].values)   # rank by out-of-fold CATE, score by truth
kr, cr = gain_curve(rng.random(len(df)))
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(kf, cf, label="Target by forest CATE", color=BOOK["primary"])
ax.plot(kr, cr, label="Random", color=BOOK["secondary"])
ax.axvline(CFG["budget_share"], ls="--", color=BOOK["accent"])
ax.set(xlabel="Share of users given a coupon", ylabel="Cumulative net incremental profit (R$)",
       title="Targeting by CATE beats random under a fixed budget")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=2)
fig.tight_layout(); fig.savefig("images/coupon-targeting-gains-python.png", dpi=300); plt.close(fig)

# figure 3: distribution of estimated CATE with positive-EV region
ev_cut = CFG["coupon_cost"]
mu = df["cate_xfit"].mean()
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.hist(df["cate_xfit"], bins=60, color=BOOK["muted"], edgecolor="white", linewidth=0.1)
ax.axvspan(ev_cut, df["cate_xfit"].max(), color=BOOK["accent"], alpha=0.12)
ax.axvline(mu, ls="--", color=BOOK["dark_gray"], linewidth=1)
ax.axvline(ev_cut, ls=":", color=BOOK["accent"], linewidth=1)
ax.text(ev_cut, ax.get_ylim()[1] * 0.93, " positive EV (CATE > cost)",
        color=BOOK["accent"], fontsize=9, ha="left", va="top")
ax.text(mu, ax.get_ylim()[1] * 0.93, "mean ", color=BOOK["dark_gray"], fontsize=9, ha="right", va="top")
ax.set(xlabel="Estimated CATE (R$)", ylabel="Users",
       title="Most users sit below the coupon cost; a profitable tail clears it")
fig.tight_layout(); fig.savefig("images/coupon-cate-histogram-python.png", dpi=300); plt.close(fig)

# figure 4: forest variable importance (est.feature_importances_ exists in econml 0.16.0).
# EconML importance = normalized treatment-effect heterogeneity created by each
# split, depth-weighted. This is NOT grf's depth-weighted split-frequency
# statistic, so this parity figure gets its own filename and labels; the
# chapter displays only the grf figure (coupon-variable-importance.png, R side).
features = ["recency", "frequency", "disc_sens", "past_spend"]
imp = np.asarray(est.feature_importances_).ravel()
order = np.argsort(imp)
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh([features[i] for i in order], imp[order], color=BOOK["primary"])
ax.set(xlabel="Normalized heterogeneity importance", ylabel=None,
       title="Heterogeneity importance in the Python forest")
fig.tight_layout(); fig.savefig("images/coupon-heterogeneity-importance-python.png", dpi=300); plt.close(fig)

# figure 5: estimated CATE vs discount sensitivity (binned mean trend)
ds = df["disc_sens"].values
ce = df["cate_xfit"].values
bins = np.linspace(ds.min(), ds.max(), 21)
idx = np.clip(np.digitize(ds, bins) - 1, 0, len(bins) - 2)
centers = 0.5 * (bins[:-1] + bins[1:])
binned = np.array([ce[idx == b].mean() if np.any(idx == b) else np.nan
                   for b in range(len(centers))])
# Band around each binned mean: uncertainty in the smoothed TREND of the
# estimates (parity with the R side's loess band). It is descriptive -- it does
# not price the estimation error inside each CATE, and it is not a confidence
# band for any user's multivariate CATE; the chapter caption says the same.
binned_se = np.array([ce[idx == b].std(ddof=1) / np.sqrt((idx == b).sum())
                      if np.any(idx == b) else np.nan for b in range(len(centers))])
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(ds, ce, s=3, alpha=0.10, color=BOOK["muted"])
ax.fill_between(centers, binned - 1.96 * binned_se, binned + 1.96 * binned_se,
                color=BOOK["primary"], alpha=0.15, linewidth=0)
ax.plot(centers, binned, color=BOOK["primary"], linewidth=2, label="Binned mean trend (95% band)")
ax.axhline(CFG["coupon_cost"], ls=":", color=BOOK["accent"])
ax.set(xlabel="Discount sensitivity", ylabel="Estimated CATE (R$)",
       title="Estimated benefit rises with discount sensitivity")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=1)
fig.tight_layout(); fig.savefig("images/coupon-cate-vs-feature-python.png", dpi=300); plt.close(fig)

# figure: how the forest finds "users like Maria" -- adaptive similarity weights.
# weight = share of trees in which a user lands in Maria's leaf (computed from .apply).
maria = int(np.where((df["recency"] > 0.6) & (df["frequency"] >= 3) & (df["disc_sens"] > 0.8))[0][0])
wm = forest_weights(maria)
rec, freq = df["recency"].values, df["frequency"].values.astype(float)
jit = rng.normal(0, 0.12, len(df))                       # vertical jitter for the integer count
top = np.argsort(-wm)[:round(0.02 * len(df))]            # top ~2% by weight
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(rec, freq + jit, s=4, color=BOOK["light_gray"], alpha=0.5, linewidths=0)
ax.scatter(rec[top], freq[top] + jit[top], s=6 + 1600 * wm[top], color=BOOK["primary"],
           alpha=0.55, linewidths=0)
ax.axvline(0.6, ls=":", color=BOOK["muted"]); ax.axhline(3, ls=":", color=BOOK["muted"])
ax.scatter([rec[maria]], [freq[maria]], marker="*", s=240, color=BOOK["accent"], zorder=5)
ax.text(rec[maria], freq[maria], "  Maria", color=BOOK["accent"], fontsize=9, va="center", fontweight="bold")
ax.set(xlabel="Recency (1 = long lapsed)", ylabel="Frequency (purchases)",
       title='How the forest finds "users like Maria": similarity weights')
fig.tight_layout(); fig.savefig("images/coupon-forest-weights-python.png", dpi=300); plt.close(fig)

# ============================================================================
# Intuition figures for the "Why trees and forests help" section
# ============================================================================

# figure A: how a single decision tree is built (illustrative ordinary outcome tree).
# A plain prediction tree predicts an OBSERVABLE outcome -- here a user's PROFIT LEVEL --
# by recursively cutting recency x frequency and labelling each box with its members'
# average, BEFORE the treatment-effect twist. Colour marks profit level on a neutral
# (magenta) ramp, leaving blue/orange to mean treatment EFFECT from the next figures on.
# High profit sits among active (low recency), frequent buyers (top-left) -- a different
# corner from where the coupon effect lives (lapsed + frequent).
rng_b = np.random.default_rng(2026)
nb = 110
b_rec = rng_b.uniform(0, 1, nb)
b_freq = np.minimum(rng_b.poisson(3, nb), 6)
b_fj = b_freq + rng_b.uniform(-0.18, 0.18, nb)
b_prof = 6 * (1 - b_rec) + 0.7 * b_freq + rng_b.normal(0, 0.8, nb)
s_rec, s_freq = 0.5, 2.5
prof_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("prof", ["#CCCCCC", BOOK["secondary"]])
steps = ["1. One group", "2. First split (recency)",
         "3. Second split (frequency)", "4. Leaves: predict each box"]
prof_lim = (b_prof.min(), b_prof.max())

# leaf mean computed ONCE and reused by both the plane labels and the rule tree,
# so the two views of the same tree can never disagree.
def leaf_mean(rl, fl):
    inb = ((b_rec < s_rec) == rl) & ((b_freq < s_freq) == fl)
    return b_prof[inb].mean() if inb.sum() else np.nan

# composite, stacked: the plane partition (top, 2x2) and the SAME tree as if/then rules (below)
fig = plt.figure(figsize=(8.5, 9.5))
outer = fig.add_gridspec(2, 1, height_ratios=[1.5, 1], hspace=0.22)
plane_gs = outer[0].subgridspec(2, 2, hspace=0.30, wspace=0.10)
axes = np.array([[fig.add_subplot(plane_gs[0, 0]), fig.add_subplot(plane_gs[0, 1])],
                 [fig.add_subplot(plane_gs[1, 0]), fig.add_subplot(plane_gs[1, 1])]])
sc = None
for k, (ax, title) in enumerate(zip(axes.ravel(), steps)):
    sc = ax.scatter(b_rec, b_fj, c=b_prof, cmap=prof_cmap, vmin=prof_lim[0], vmax=prof_lim[1],
                    s=18, alpha=0.9, linewidths=0)
    if k >= 1:
        ax.axvline(s_rec, color=BOOK["dark_gray"], lw=0.8)
    if k >= 2:
        ax.plot([-0.02, 1.02], [s_freq, s_freq], color=BOOK["dark_gray"], lw=0.8)
    if k == 3:                                   # label each leaf with its average profit
        for rl in (True, False):
            for fl in (True, False):
                m = leaf_mean(rl, fl)
                if np.isnan(m):
                    continue
                cx = (-0.02 + s_rec) / 2 if rl else (s_rec + 1.02) / 2
                cy = (-0.5 + s_freq) / 2 if fl else (s_freq + 6.5) / 2
                ax.text(cx, cy, f"R${m:.1f}", ha="center", va="center",
                        fontsize=9, color=BOOK["dark_gray"],
                        bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.85))
        # keyed markers: "A" = the recency cut, "B" = the frequency cut (letters, not
        # the panel numbers) so each cut shares a tag with its node on the tree.
        for mx, my, tag in [(s_rec, 6.2, "1"), (0.96, s_freq, "2")]:
            ax.text(mx, my, tag, ha="center", va="center", fontsize=8, fontweight="bold",
                    color=BOOK["dark_gray"],
                    bbox=dict(boxstyle="circle,pad=0.3", fc="white", ec=BOOK["dark_gray"], lw=0.6))
    ax.set_title(title, fontsize=10)
    ax.set(xlim=(-0.02, 1.02), ylim=(-0.5, 6.5))
    if k in (0, 1):
        ax.set_xticklabels([])
    if k in (1, 3):
        ax.set_yticklabels([])
for ax in (axes[1, 0], axes[1, 1]):
    ax.set_xlabel("Recency (1 = long lapsed)")
for ax in (axes[0, 0], axes[1, 0]):
    ax.set_ylabel("Frequency (purchases)")
fig.text(0.5, 0.93, "Panel A: the mechanics of the cuts", ha="center", fontsize=11, color="grey")
cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), location="bottom", fraction=0.05, pad=0.10)
cbar.set_label("Profit (R$)")

# the same tree as if/then rules: nodes are cuts (keyed A/B), leaves are avg profit
ax_tree = fig.add_subplot(outer[1]); ax_tree.axis("off")
ax_tree.set(xlim=(0, 1), ylim=(0.05, 1.0))
ax_tree.set_title("Panel B: the same splits as if/then rules", fontsize=11, color="grey")
intern = [(0.50, 0.92, "① recency <= 0.50?"),     # circled-number key (①/②) ties
          (0.27, 0.56, "② frequency <= 2.5?"),    # each node to its cut line in Panel A
          (0.73, 0.56, "② frequency <= 2.5?")]    # (DejaVu Sans includes U+2460/2461)
edges = [(0.50, 0.92, 0.27, 0.56, "yes"), (0.50, 0.92, 0.73, 0.56, "no"),
         (0.27, 0.56, 0.13, 0.18, "yes"), (0.27, 0.56, 0.41, 0.18, "no"),
         (0.73, 0.56, 0.59, 0.18, "yes"), (0.73, 0.56, 0.87, 0.18, "no")]
for x, y, xe, ye, br in edges:
    ax_tree.plot([x, xe], [y, ye], color=BOOK["muted"], lw=0.8, zorder=1)
    ax_tree.text((x + xe) / 2, (y + ye) / 2, br, fontsize=11, color=BOOK["muted"],
                 ha="center", va="center",
                 bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.9), zorder=2)
mid = (prof_lim[0] + prof_lim[1]) / 2
for x, rl, fl in [(0.13, True, True), (0.41, True, False), (0.59, False, True), (0.87, False, False)]:
    m = leaf_mean(rl, fl)
    fc = prof_cmap((m - prof_lim[0]) / (prof_lim[1] - prof_lim[0]))
    ax_tree.add_patch(plt.Rectangle((x - 0.075, 0.12), 0.15, 0.12, facecolor=fc,  # narrower: gap between the two middle leaves
                                    edgecolor=BOOK["muted"], lw=0.5, zorder=3))
    ax_tree.text(x, 0.18, f"R${m:.1f}", ha="center", va="center", fontsize=9, zorder=4,
                 color="white" if m > mid else "#222222")
for x, y, lab in intern:
    ax_tree.text(x, y, lab, ha="center", va="center", fontsize=8.5, color="#222222", zorder=4,
                 bbox=dict(boxstyle="round", fc=BOOK["light_gray"], ec=BOOK["muted"], lw=0.6))
fig.suptitle("How a single decision tree is built", fontweight="bold")
fig.savefig("images/coupon-tree-build-python.png", dpi=300, bbox_inches="tight"); plt.close(fig)

# figure B: honesty / sample splitting (real coupon data). One sample CHOOSES the boxes,
# a DIFFERENT sample MEASURES the coupon effect inside them. Boxes from a transparent greedy
# search for the cut that makes the treated-minus-control coupon effect differ most between
# the two sides -- the causal-tree split criterion itself, run on the structure sample only.
# recency comes from the CSV, so any generator is independent of it -- no need for the R
# side's distinct-seed guard.
rng_h = np.random.default_rng(2027)
is_struct = rng_h.random(len(df)) < 0.5
s, e = df[is_struct], df[~is_struct]

def best_cut(d, col, grid):
    def eff(g):
        return g["observed_profit"][g["D"] == 1].mean() - g["observed_profit"][g["D"] == 0].mean()
    best, bv = grid[0], -np.inf
    for c in grid:
        da, db = d[d[col] < c], d[d[col] >= c]
        if min((da["D"] == 1).sum(), (da["D"] == 0).sum(),
               (db["D"] == 1).sum(), (db["D"] == 0).sum()) < 30:
            continue
        v = abs(eff(da) - eff(db))               # split where the effect differs most
        if v > bv:
            bv, best = v, c
    return best

c_rec = best_cut(s, "recency", np.arange(0.3, 0.81, 0.05))
c_fhi = best_cut(s[s["recency"] >= c_rec], "frequency", [1.5, 2.5, 3.5, 4.5])
c_flo = best_cut(s[s["recency"] < c_rec], "frequency", [1.5, 2.5, 3.5, 4.5])
ytop = df["frequency"].max() + 0.5
boxes = [(-0.02, c_rec, -0.5, c_flo), (-0.02, c_rec, c_flo, ytop),
         (c_rec, 1.02, -0.5, c_fhi), (c_rec, 1.02, c_fhi, ytop)]

def box_eff(b):
    x0, x1, y0, y1 = b
    g = e[(e["recency"] >= x0) & (e["recency"] < x1) &
          (e["frequency"] >= y0) & (e["frequency"] < y1)]
    return g["observed_profit"][g["D"] == 1].mean() - g["observed_profit"][g["D"] == 0].mean()

effs = [box_eff(b) for b in boxes]
# points are neutral (the panel title says which sample); the held-out effect colours each
# estimate-panel box on the same blue/orange scale as the forest figure.
eff_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("eff", [BOOK["accent"], "#F2F2F2", BOOK["primary"]])
eff_lim = max(abs(min(effs)), abs(max(effs))) + 1
eff_norm = matplotlib.colors.TwoSlopeNorm(vmin=-eff_lim, vcenter=0, vmax=eff_lim)
panels = [("Structure sample: choose the boxes", s, False),
          ("Estimate sample: measure the effect", e, True)]
fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.5), sharex=True, sharey=True)
for ax, (title, d, show) in zip(axes, panels):
    if show:                                     # held-out effect colours each box
        for b, ef in zip(boxes, effs):
            ax.add_patch(plt.Rectangle((b[0], b[2]), b[1] - b[0], b[3] - b[2],
                                       color=eff_cmap(eff_norm(ef)), alpha=0.55, lw=0, zorder=0))
    fjit = d["frequency"] + rng_h.uniform(-0.2, 0.2, len(d))
    ax.scatter(d["recency"], fjit, s=3, color=BOOK["muted"], alpha=0.30, linewidths=0, zorder=1)
    ax.axvline(c_rec, color=BOOK["dark_gray"], lw=0.7)
    ax.plot([-0.02, c_rec], [c_flo, c_flo], color=BOOK["dark_gray"], lw=0.7)
    ax.plot([c_rec, 1.02], [c_fhi, c_fhi], color=BOOK["dark_gray"], lw=0.7)
    if show:
        for b, ef in zip(boxes, effs):
            cx = (b[0] + b[1]) / 2
            cy = min(max((b[2] + b[3]) / 2, 0.4), df["frequency"].max())
            ax.text(cx, cy, f"R${ef:.1f}", ha="center", va="center", fontsize=9,
                    color=BOOK["dark_gray"],
                    bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.85))
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Recency (1 = long lapsed)")
    ax.set(xlim=(-0.02, 1.02))
axes[0].set_ylabel("Frequency (purchases)")
fig.suptitle("Honesty: choose the boxes on one sample, measure effects on another")
fig.tight_layout(); fig.savefig("images/coupon-honesty-split-python.png", dpi=300); plt.close(fig)

# figure C: from one tree to a forest (illustrative). Each tree = a few random blocky
# cuts (a decorrelated partition); averaging many trees smooths the noise into the
# underlying effect surface (the chapter's planted gate for lapsed, frequent buyers).
rng_f = np.random.default_rng(2028)
gx = np.linspace(0, 1, 60)
gy = np.linspace(0, 6, 60)
RX, FY = np.meshgrid(gx, gy)
true_surf = np.where((RX > 0.6) & (FY >= 3), 5.0, -1.0)

def one_tree():
    rcuts = np.sort(rng_f.uniform(0.2, 0.9, rng_f.integers(2, 4)))
    fcuts = np.sort(rng_f.uniform(1, 5, rng_f.integers(2, 4)))
    rb, fb = np.digitize(RX, rcuts), np.digitize(FY, fcuts)
    out = np.zeros_like(true_surf)
    for i in np.unique(rb):
        for j in np.unique(fb):
            mask = (rb == i) & (fb == j)
            out[mask] = true_surf[mask].mean()
    return out + rng_f.normal(0, 1.2, out.shape)

ntree = 5
mats = [one_tree() for _ in range(ntree)] + [np.mean([one_tree() for _ in range(200)], axis=0)]
panC = [f"Tree {i + 1}" for i in range(ntree)] + ["Forest (average of 200)"]
cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "eff", [BOOK["accent"], "#F2F2F2", BOOK["primary"]])
norm = matplotlib.colors.TwoSlopeNorm(vmin=-4, vcenter=0, vmax=7)
fig, axes = plt.subplots(2, 3, figsize=(8.5, 5.5), sharex=True, sharey=True)
for ax, mat, title in zip(axes.ravel(), mats, panC):
    im = ax.pcolormesh(gx, gy, mat, cmap=cmap, norm=norm, shading="auto")
    ax.set_title(title, fontsize=10)
# one shared x-label, like the R facets' single axis title (three per-panel copies collide)
fig.supxlabel("Recency (1 = long lapsed)", fontsize=12, color=BOOK["dark_gray"])
for ax in axes[:, 0]:
    ax.set_ylabel("Frequency (purchases)")
fig.suptitle("From one tree to a forest: averaging decorrelated trees smooths the noise")
cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
cbar.set_label("Effect (R$)")
fig.savefig("images/coupon-tree-to-forest-python.png", dpi=300, bbox_inches="tight"); plt.close(fig)

# figure: TOC curve (the continuous version of GATES), built transparently like the
# held-out GATES below: priority from one half, evaluate uplift vs the overall ATE on
# the other (grf uses doubly-robust scores -- same downward shape). econml's DRTester
# can also draw TOC/QINI curves; it supplies the AUTOC number in the validation block.
rng_toc = np.random.default_rng(CFG["seed"])
tr_t = np.zeros(len(df), bool); tr_t[rng_toc.choice(len(df), len(df) // 2, replace=False)] = True
prio = _forest_cate(Y[tr_t], D[tr_t], X[tr_t], X[~tr_t])
Ye, De = Y[~tr_t], D[~tr_t]
ate_e = Ye[De == 1].mean() - Ye[De == 0].mean()
order = np.argsort(-prio); m = len(order)
qs = np.linspace(0.05, 1.0, 100); est_q, se_q = [], []
for q in qs:
    sel = order[:max(4, int(round(q * m)))]
    yt, dt = Ye[sel], De[sel]
    m1, m0 = yt[dt == 1], yt[dt == 0]
    est_q.append(m1.mean() - m0.mean() - ate_e)
    se_q.append(np.sqrt(m1.var(ddof=1) / len(m1) + m0.var(ddof=1) / len(m0)))
est_q, se_q = np.array(est_q), np.array(se_q)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.axhline(0, ls="--", color=BOOK["muted"])
ax.fill_between(qs, est_q - 1.96 * se_q, est_q + 1.96 * se_q, color=BOOK["primary"], alpha=0.15)
ax.plot(qs, est_q, color=BOOK["primary"], linewidth=1.6)
ax.set(xlabel="Treated fraction (q), ranked by predicted CATE", ylabel="Uplift vs. overall ATE (R$)",
       title="Do the top-ranked users really gain more? The TOC curve")
fig.tight_layout(); fig.savefig("images/coupon-toc-curve-python.png", dpi=300); plt.close(fig)

# figure 6 (policy tree): rendered R-side by policytree (exact optimal tree). Here we
# confirm parity with mcf's OptimalPolicy, which runs the same style of exhaustive
# (non-greedy) search but over a candidate split grid (pt_no_of_evalupoints per
# feature) -- close to, not guaranteed equal to, policytree's every-split optimum.
# (econml's PolicyTree is greedy and collapses to treating no one on this conjunctive
# rule, so it is not a faithful analog of R's policytree.) Both feed a cross-fitted
# CATE net of the R$5 cost as the per-action policy score: grf's out-of-bag CATEs on
# the R side, the two-fold out-of-fold vector here.
# Doubly robust production alternative (semantics only -- the R script shows the code):
#   no-coupon column = DR score for action 0; coupon column = DR score for action 1 - R$5.
# Keep BOTH action columns; do not zero the no-coupon column or subtract the cost from
# both, and do not use the DR alternative to regenerate the displayed tree's numbers.
pdf = df.copy()
pdf["score_nocoupon"], pdf["score_coupon"] = 0.0, pdf["cate_xfit"] - CFG["coupon_cost"]

def _solve_policy(depth, **extra):
    optp = OptimalPolicy(
        var_polscore_name=["score_nocoupon", "score_coupon"],
        var_x_name_ord=["recency", "frequency", "disc_sens", "past_spend"],
        gen_method="policy tree", pt_depth_tree_1=depth, pt_depth_tree_2=0,
        # mcf evaluates candidate splits on a grid of this many points per feature
        # (its default, made explicit): the search is exhaustive over that grid,
        # not over every continuous split value.
        pt_no_of_evalupoints=100,
        gen_outpath=f"/tmp/mcf_coupon_d{depth}", **extra,
    )
    return optp.solve(pdf, data_title="coupon")

def _solve_tree(depth):
    try:
        alloc, _, _ = _solve_policy(depth)
    except Exception as e:
        # Serial fallback ONLY for parallel-infrastructure failures (spawn/Ray/psutil/
        # pickling); anything else -- missing columns, bad score shapes, API misuse --
        # is a real bug and re-raises.
        msg = f"{type(e).__name__}: {e}".lower()
        if any(k in msg for k in ("parallel", "spawn", "ray", "psutil", "pickl", "fork",
                                  "operation not permitted")):  # EPERM from sandboxed process control
            print(f"mcf parallel run failed ({type(e).__name__}); retrying serially.")
            alloc, _, _ = _solve_policy(depth, _int_parallel_processing=False)
        else:
            raise
    return alloc["Policy Tree"].to_numpy() == 1

# The displayed tree's two valuations, never collapsed into one claim: the
# model-implied value uses the same out-of-fold CATEs the tree was learned from;
# the truth-scored value grades the same allocation against the planted true_cate.
coupon = _solve_tree(2)
xfit_v, true_v = df["cate_xfit"].to_numpy(), df["true_cate"].to_numpy()
print(f"[policy tree] treated share = {coupon.mean():.1%}; "
      f"model-implied net value (xfit CATE - cost) = R${(xfit_v[coupon] - CFG['coupon_cost']).sum():.0f}; "
      f"truth-scored net value (true CATE - cost) = R${(true_v[coupon] - CFG['coupon_cost']).sum():.0f}")

# Depth 3, same scores, run live. The grid search over the depth-3 tree space took
# roughly an hour at n = 20,000 on a laptop (~40-75 min across runs, mcf 0.6.0) vs ~20s at
# depth 2 -- that cost explosion with depth is the point the chapter makes. Tuning
# is NOT identical to the R side: mcf's data-dependent default minimum leaf size
# (~333 rows at this depth and n) vs policytree's min.node.size = 1000, so the two
# sides independently corroborate the same treatment region rather than replicate
# one optimization.
coupon3 = _solve_tree(3)
print(f"[policy tree depth 3] treated share = {coupon3.mean():.1%}; "
      f"model-implied net value (xfit CATE - cost) = R${(xfit_v[coupon3] - CFG['coupon_cost']).sum():.0f}; "
      f"truth-scored net value (true CATE - cost) = R${(true_v[coupon3] - CFG['coupon_cost']).sum():.0f}")

# Truth ceiling for ANY allocation: treat exactly the users whose true effect
# exceeds the R$5 cost (the finite-sample truth optimum under unconstrained binary
# allocation). On this data that predicate equals the planted three-gate rule
# recency > 0.6 & frequency >= 3 & disc_sens > 0.5, row for row.
oracle = true_v - CFG["coupon_cost"] > 0
print(f"[truth ceiling] treated share = {oracle.mean():.1%}; "
      f"truth-scored net value (true CATE - cost) = R${(true_v[oracle] - CFG['coupon_cost']).sum():.0f}")
print("Wrote 10 figures (policy tree rendered R-side; mcf confirms the rule).")

# power check: "no heterogeneity detected" is usually a power problem.
# The planted spread is loud (SD = 2.8 vs true ATE = 0.38, ~7x). (a) Even a 2,000-user
# subsample still detects it. (b) Shrink the deviations around the SAME ATE (same noise,
# same draws) to ~half the ATE's size and the full 20,000-user experiment still pins the
# ATE but sees no heterogeneity. Python parity for the chapter's power subsection
# (the headline calibration-test numbers quoted there come from grf's test_calibration).
from scipy import stats

# (_forest_cate is defined once near the top, reused here.)
rng_pc = np.random.default_rng(CFG["seed"])

# (a) loud heterogeneity, small sample
idx_s = rng_pc.choice(len(df), 2000, replace=False)
pred_s = _forest_cate(Y[idx_s], D[idx_s], X[idx_s], X[idx_s])
print(f"[loud world, n = 2000] corr with truth = "
      f"{np.corrcoef(pred_s, df['true_cate'].values[idx_s])[0, 1]:.3f}")

# (b) quiet heterogeneity, full sample: spread / 16 (~half the ATE)
y_base = Y - D * df["true_cate"].values
ate_true = df["true_cate"].mean()
tau_q = ate_true + (df["true_cate"].values - ate_true) / 16
y_q = y_base + D * tau_q
pred_q = _forest_cate(y_q, D, X, X)
tt_q = stats.ttest_ind(y_q[D == 1], y_q[D == 0])
print(f"[quiet world (SD = {tau_q.std():.2f}, {tau_q.std() / ate_true:.2f}x ATE), n = {len(df)}] "
      f"ATE t = {tt_q.statistic:.2f} (p = {tt_q.pvalue:.2g}); "
      f"corr with truth = {np.corrcoef(pred_q, tau_q)[0, 1]:.3f}")

# (c) honest GATES in the quiet world: fit on one half, bucket and evaluate uplift on the
# OTHER half. Bucketing on in-sample effect(X) would show a fake monotone gradient here
# (effect(X) is not out-of-bag, so prediction noise leaks into realized uplift); the
# held-out version reads flat, matching grf's test_calibration verdict.
tr = np.zeros(len(df), dtype=bool)
tr[rng_pc.choice(len(df), len(df) // 2, replace=False)] = True
pred_ho = _forest_cate(y_q[tr], D[tr], X[tr], X[~tr])
y_ho, d_ho = y_q[~tr], D[~tr]
grp = pd.qcut(pred_ho, 5, labels=False, duplicates="drop")
gates_q = pd.Series([y_ho[(grp == q) & (d_ho == 1)].mean() - y_ho[(grp == q) & (d_ho == 0)].mean()
                     for q in sorted(np.unique(grp))])
print("held-out GATES in the quiet world (uplift by predicted quintile, flat = no detectable ranking):")
print(gates_q.round(3).to_string())

# validation diagnostics (parity): Python backs every row of the chapter's trust
# table. Calibration: econml has no test_calibration, but the test is a single
# regression on the DML residuals, so we run it directly: regress the outcome residual
# on the treatment residual scaled by (a) the mean predicted CATE and (b) each user's
# deviation from it. The second coefficient is grf's `differential.forest.prediction`.
# BLP, RATE/AUTOC, overlap, and the stability-row calibration follow further down.
import statsmodels.api as sm
from sklearn.model_selection import train_test_split


def _forest_fit(y, d, x):
    """Same forest as _forest_cate, but returns the fitted object (for its residuals)."""
    f = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=300, min_samples_leaf=20, random_state=CFG["seed"]),
        model_t=RandomForestClassifier(n_estimators=300, min_samples_leaf=20, random_state=CFG["seed"]),
        discrete_treatment=True, n_estimators=CFG["n_estimators"], random_state=CFG["seed"],
    )
    f.fit(y, d, X=x, cache_values=True)   # cache_values exposes .residuals_
    return f


# Grade the forest on users it did NOT train on: effect(X) is in-sample, and scoring a
# forest on its own training users inflates the differential coefficient (1.29 in-sample
# vs 1.02 held out; grf's out-of-bag test_calibration reads 1.00 / 0.99 on the full sample).
tr_c, ev_c = train_test_split(np.arange(len(df)), test_size=0.5, random_state=CFG["seed"])
cf_tr = _forest_fit(Y[tr_c], D[tr_c], X[tr_c])              # train-half forest, reused by BLP/RATE below
tau_ev = cf_tr.effect(X[ev_c])                              # CATEs from the train half
cf_ev = _forest_fit(Y[ev_c], D[ev_c], X[ev_c])              # residuals from the eval half
y_res, d_res = cf_ev.residuals_[0].ravel(), cf_ev.residuals_[1].ravel()  # Y - m(X), D - e(X)

tau_bar = tau_ev.mean()
cal = sm.OLS(y_res, np.column_stack([tau_bar * d_res,                # grades the average
                                     (tau_ev - tau_bar) * d_res])    # grades the spread
             ).fit(cov_type="HC3")
print(f"\n[calibration, held out] mean coef = {cal.params[0]:.3f} (SE {cal.bse[0]:.3f}); "
      f"differential coef = {cal.params[1]:.3f} (SE {cal.bse[1]:.3f}, p = {cal.pvalues[1]:.2g})")

# Same calibration on the quiet world (y_q, built in the power check above): it should
# read flat. Backs the -0.22 quoted in the chapter's Python power-check tab; grf's
# out-of-bag test_calibration reads -0.12 (p = 0.77) there. Both say "nothing here".
tau_q_ev = _forest_cate(y_q[tr_c], D[tr_c], X[tr_c], X[ev_c])
cfq_ev = _forest_fit(y_q[ev_c], D[ev_c], X[ev_c])
yq_res, dq_res = cfq_ev.residuals_[0].ravel(), cfq_ev.residuals_[1].ravel()
tau_q_bar = tau_q_ev.mean()
cal_q = sm.OLS(yq_res, np.column_stack([tau_q_bar * dq_res,
                                        (tau_q_ev - tau_q_bar) * dq_res])).fit(cov_type="HC3")
print(f"[calibration, quiet world, held out] differential coef = {cal_q.params[1]:.3f} "
      f"(SE {cal_q.bse[1]:.3f}, p = {cal_q.pvalues[1]:.2g}) -- flat, as it should be")

# best linear projection (parity with grf::best_linear_projection): regress each
# user's DOUBLY ROBUST SCORE on their features (Semenova & Chernozhukov 2021).
# The DR score is the AIPW pseudo-outcome built from the same residuals as the
# calibration test: tau(X) + D_res / (e(1-e)) * (Y_res - tau(X) * D_res), with
# e(X) = D - D_res. Held out like everything econml-side: tau from the train
# half, residuals and scores on the eval half. Reads recency 4.98, disc_sens
# 3.39, frequency 0.62 (all p < 0.001), past_spend ~ 0 -- grf's drivers.
e_ev = D[ev_c] - d_res
gamma = tau_ev + d_res / (e_ev * (1 - e_ev)) * (y_res - tau_ev * d_res)
blp = sm.OLS(gamma, sm.add_constant(pd.DataFrame(X[ev_c], columns=features))
             ).fit(cov_type="HC3")
print("\n[best linear projection, held out] DR scores ~ features "
      "(grf, full sample OOB: recency 4.66, disc_sens 3.09, frequency 0.61, past_spend ~ 0):")
print(blp.summary2().tables[1].round(3))

# Standardized pass: raw coefficients are per unit of each feature (recency
# spans 0-1, frequency counts purchases), so they can't be ranked across
# features. Z-scoring the PROJECTION covariates puts every coefficient on a
# per-1-SD scale; the forest itself stays fitted on the raw features.
X_blp_std = pd.DataFrame(X[ev_c], columns=features)
X_blp_std = (X_blp_std - X_blp_std.mean()) / X_blp_std.std(ddof=1)
blp_std = sm.OLS(gamma, sm.add_constant(X_blp_std)).fit(cov_type="HC3")
print("\n[best linear projection, held out, per 1 SD]:")
print(blp_std.summary2().tables[1].round(3))

# RATE / AUTOC (parity with grf::rank_average_treatment_effect): econml ships
# this as DRTester.evaluate_uplift. Same discipline as grf's doc warning: the
# CATE model (cf_tr) is fitted on the train half only; nuisances and evaluation
# run on the eval half. metric="toc" integrates the TOC over a 5-95%
# treated-fraction grid, so it clips the very top slice grf includes -- the
# same verdict on a somewhat smaller number: 2.22 +/- 0.19 here vs grf's
# 2.70 [2.50, 2.91].
from econml.validate.drtester import DRTester
import warnings
tester = DRTester(
    model_regression=RandomForestRegressor(n_estimators=300, min_samples_leaf=20, random_state=CFG["seed"]),
    model_propensity=RandomForestClassifier(n_estimators=300, min_samples_leaf=20, random_state=CFG["seed"]),
    cate=cf_tr,
)
tester.fit_nuisance(X[ev_c], D[ev_c], Y[ev_c], X[tr_c], D[tr_c], Y[tr_c])
with warnings.catch_warnings():                    # bootstrap emits benign divide warnings
    warnings.simplefilter("ignore", RuntimeWarning)
    autoc = tester.evaluate_uplift(Xval=X[ev_c], Xtrain=X[tr_c], metric="toc")
print(f"\n[RATE, DRTester] AUTOC = {autoc.params[0]:.3f} +/- {1.96 * autoc.errs[0]:.3f} "
      f"(95% CI), p = {autoc.pvals[0]:.2g}")

# DRTester's built-in BLP test is the Chernozhukov et al. heterogeneity check --
# the one-call version of the hand-rolled calibration regression above (reads
# 1.01 here, vs 1.02 hand-rolled and grf's out-of-bag 0.99).
drblp = tester.evaluate_blp(Xval=X[ev_c], Xtrain=X[tr_c])
print(f"[calibration, DRTester one-call] heterogeneity coef = {drblp.params[0]:.3f} "
      f"(SE {drblp.errs[0]:.3f}, p = {drblp.pvals[0]:.2g})")

# overlap (parity with grf's W.hat range): the cross-fitted propensities already
# sit inside the main forest's cached residuals, e(X) = D - D_res. Reads
# [0.27, 0.74] here vs grf's [0.36, 0.65] -- wider (RF-classifier propensities
# are noisier than grf's regression forest) but the same far-from-0-and-1 call.
e_hat = D - est.residuals_[1].ravel()
print(f"[overlap] propensity e(X) range: [{e_hat.min():.3f}, {e_hat.max():.3f}]")

# stability row (parity with grf's n = 2,000 test_calibration): the held-out
# calibration regression on the small subsample. idx_s (from the power check)
# is already in random order, so its two halves are a clean split -- and no new
# rng_pc draws, which would shift the seeded blocks below. Reads 1.30 (t = 7.6)
# here vs grf's out-of-bag 1.10 (t = 11.6): heterogeneity still detected.
tr_s, ev_s = idx_s[:1000], idx_s[1000:]
tau_s = _forest_cate(Y[tr_s], D[tr_s], X[tr_s], X[ev_s])
cf_s = _forest_fit(Y[ev_s], D[ev_s], X[ev_s])
ys_res, ds_res = cf_s.residuals_[0].ravel(), cf_s.residuals_[1].ravel()
tau_s_bar = tau_s.mean()
cal_s = sm.OLS(ys_res, np.column_stack([tau_s_bar * ds_res,
                                        (tau_s - tau_s_bar) * ds_res])).fit(cov_type="HC3")
print(f"[calibration, n = 2000 subsample, held out] differential coef = {cal_s.params[1]:.3f} "
      f"(SE {cal_s.bse[1]:.3f}, t = {cal_s.tvalues[1]:.1f})")

# GATES, loud world, held-out: fit on one half, bucket and evaluate uplift on
# the OTHER half (effect(X) is not out-of-bag, so in-sample bucketing would
# leak prediction noise into the realized uplift).
tr_l = np.zeros(len(df), dtype=bool)
tr_l[rng_pc.choice(len(df), len(df) // 2, replace=False)] = True
pred_l = _forest_cate(Y[tr_l], D[tr_l], X[tr_l], X[~tr_l])
y_l, d_l = Y[~tr_l], D[~tr_l]
grp_l = pd.qcut(pred_l, 5, labels=False, duplicates="drop")
gates_l = pd.Series([y_l[(grp_l == q) & (d_l == 1)].mean() - y_l[(grp_l == q) & (d_l == 0)].mean()
                     for q in sorted(np.unique(grp_l))])
print("held-out GATES in the loud world (uplift by predicted quintile, 0 = lowest):")
print(gates_l.round(2).to_string())

# targeting table: model-implied value of the four policies in the chapter's
# "Rank, cut, and compare" table. Each policy is scored by the model's own
# out-of-fold estimates (cate_xfit - cost) -- the "what the model expects"
# numbers; the gain-curve figure scores the same ranking against the planted
# truth instead. (Own generator: rng_pc's stream feeds the placebo below.)
rng_tt = np.random.default_rng(CFG["seed"])
ev_est = (df["cate_xfit"] - CFG["coupon_cost"]).values
n_budget = int(CFG["budget_share"] * len(df))
print(f"[targeting table] blanket = {ev_est.sum():.0f}; "
      f"random {CFG['budget_share']:.0%} = {rng_tt.choice(ev_est, n_budget, replace=False).sum():.0f}; "
      f"top {CFG['budget_share']:.0%} by CATE = {np.sort(ev_est)[::-1][:n_budget].sum():.0f}; "
      f"positive-EV only = {ev_est[ev_est > 0].sum():.0f} ({(ev_est > 0).mean():.1%} of users)")

# placebo: permute the treatment label and refit -- the apparent heterogeneity
# should collapse toward zero.
D_pl = rng_pc.permutation(D)
pred_pl = _forest_cate(Y, D_pl, X, X)
print(f"placebo (permuted D): SD of predictions = {pred_pl.std():.3f} "
      f"(real forest: {df['cate_forest'].std():.3f}); "
      f"corr with truth = {np.corrcoef(pred_pl, df['true_cate'])[0, 1]:.3f}")

# held-out calibration on the permuted fit (backs the differential-coefficient
# half of the chapter's placebo row: grf's test_calibration reads 0.03, p = 0.43
# there). Same tr_c/ev_c split and two-column regression as the loud world above.
tau_pl = _forest_cate(Y[tr_c], D_pl[tr_c], X[tr_c], X[ev_c])
cfpl_ev = _forest_fit(Y[ev_c], D_pl[ev_c], X[ev_c])
ypl_res, dpl_res = cfpl_ev.residuals_[0].ravel(), cfpl_ev.residuals_[1].ravel()
tau_pl_bar = tau_pl.mean()
cal_pl = sm.OLS(ypl_res, np.column_stack([tau_pl_bar * dpl_res,
                                          (tau_pl - tau_pl_bar) * dpl_res])).fit(cov_type="HC3")
print(f"placebo calibration, held out: differential coef = {cal_pl.params[1]:.3f} "
      f"(SE {cal_pl.bse[1]:.3f}, p = {cal_pl.pvalues[1]:.2g}) -- flat, as it should be")
