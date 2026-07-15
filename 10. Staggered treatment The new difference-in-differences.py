##########################################################
## Everyday Causal Inference: How to estimate, test, and explain impacts with R and Python
## www.everydaycausal.com
## Copyright (c) 2025 by Robson Tigre. All rights reserved.
## You may read, run, adapt, and cite this code, provided you credit the source.
## It should not be used to create competing educational or commercial products
##########################################################
## Code for Chapter 10 - Staggered treatment: The new difference-in-differences
## Created: May 08, 2026
## Last modified: 2026-05-18
##########################################################

# If you haven't already, run this in your terminal to install the packages:
# pip install matplotlib numpy pandas seaborn diff-diff linearmodels

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from diff_diff import CallawaySantAnna
from linearmodels.panel import PanelOLS

warnings.filterwarnings("ignore")

# ---- Centralized configuration ----------------------------------------------
SEED = 64                       # matches the R DGP seed
N_BOOT = 1000                   # multiplier-bootstrap iterations (R `biters` default)
MIN_E, MAX_E = -6, 7            # event-study horizon shown in the chapter
DATA_MAIN = "data/staggered_did.csv"
DATA_COV = "data/staggered_did_cov.csv"
IMG_DIR = "images"

# True treatment-effect schedule used by the DGP, tau(g, k) = base + slope * k.
# These are DGP *constants* (not data generation) -- the realized `tau` column
# is dropped before the CSV is written, so the ground-truth ATT is recomputed
# here from `cohort` and `relative_period`.
TAU_SCHEDULE = pd.DataFrame(
    {
        "cohort": [5, 7, 9, 11],
        "base": [1.5, 1.8, 2.1, 2.4],
        "slope": [0.50, 0.40, 0.28, 0.16],
    }
)

# ---- Project theme + palette (mirrors .agent/rules/plots-aesthetics.md) ------
book_colors = {
    "primary": "#2E86AB",     # steel blue   - main data
    "secondary": "#A23B72",   # magenta      - secondary data
    "accent": "#F18F01",      # orange       - highlights / overall lines
    "success": "#C73E1D",     # red-orange   - thresholds / contrasts
    "muted": "#6C757D",       # gray         - reference lines
    "light_gray": "#E5E5E5",  # light gray   - backgrounds
    "dark_gray": "#4D4D4D",   # dark gray    - text
}

# Cohort series colors -- pastel blue / radish / sage / lavender, kept separate
# from book_colors so the cohort-trend figures stay visually consistent.
cohort_palette = {
    "Treated in period 5": "#5C8FAF",
    "Treated in period 7": "#D77A91",
    "Treated in period 9": "#5E9A78",
    "Treated in period 11": "#A07AC1",
    "Never treated": book_colors["muted"],
}
COHORT_ORDER = [
    "Never treated",
    "Treated in period 5",
    "Treated in period 7",
    "Treated in period 9",
    "Treated in period 11",
]


def style_axes(ax, title=None, subtitle=None, xlabel=None, ylabel=None):
    """Apply the book's minimal theme to a matplotlib Axes (theme_book analog)."""
    ax.grid(True, which="major", color=book_colors["light_gray"], linewidth=0.6)
    ax.grid(False, which="minor")
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if xlabel is not None:
        ax.set_xlabel(xlabel, color=book_colors["dark_gray"])
    if ylabel is not None:
        ax.set_ylabel(ylabel, color=book_colors["dark_gray"])
    # Title (bold) + optional subtitle (muted), stacked like ggplot.
    if title is not None and subtitle is not None:
        ax.set_title(
            f"{title}\n", fontweight="bold", color=book_colors["dark_gray"], loc="left"
        )
        ax.text(
            0.0, 1.02, subtitle, transform=ax.transAxes, fontsize="small",
            color=book_colors["muted"], va="bottom", ha="left",
        )
    elif title is not None:
        ax.set_title(title, fontweight="bold", color=book_colors["dark_gray"], loc="left")


def save_fig(fig, name):
    """Save a figure to images/<name>_py.png (the _py suffix protects R figures)."""
    path = f"{IMG_DIR}/{name}_py.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {path}")


def cohort_label(cohort):
    """Map a numeric cohort to its chapter label."""
    return "Never treated" if cohort == 0 else f"Treated in period {cohort}"


# ---- Estimation helpers -----------------------------------------------------
def fit_cs(data, covariates=None, control_group="never_treated"):
    """Fit Callaway-Sant'Anna (doubly robust, universal base period, seeded).

    Returns the CallawaySantAnnaResults object. `aggregate="all"` gives the
    simple / group / event-study aggregations in one fit; the calendar
    aggregation is computed separately by `calendar_aggregation()`.
    """
    cs = CallawaySantAnna(
        control_group=control_group,
        anticipation=0,
        estimation_method="dr",
        base_period="universal",
        n_bootstrap=N_BOOT,
        seed=SEED,
        cband=True,
        cluster="store_id",  # cluster the multiplier bootstrap at the store level
    )
    return cs.fit(
        data,
        outcome="sales",
        unit="store_id",
        time="period",
        first_treat="cohort",
        covariates=covariates,
        aggregate="all",
    )


def event_study_df(res, min_e=MIN_E, max_e=MAX_E):
    """Tidy the event-study aggregation into a DataFrame with uniform bands."""
    rows = []
    for e, d in res.event_study_effects.items():
        e = int(e)
        if e < min_e or e > max_e:
            continue
        if e == -1:  # universal base period: reference event time pinned to 0
            rows.append({"e": -1, "att": 0.0, "se": 0.0, "lo": 0.0, "hi": 0.0})
            continue
        lo, hi = d.get("cband_conf_int", d["conf_int"])
        rows.append(
            {"e": e, "att": float(d["effect"]), "se": float(d["se"]),
             "lo": float(lo), "hi": float(hi)}
        )
    return pd.DataFrame(rows).sort_values("e").reset_index(drop=True)


def dynamic_overall(res, min_e=MIN_E, max_e=MAX_E):
    """Average post-treatment event-study ATT (e >= 0) -- the R `dynamic` overall."""
    post = [
        float(d["effect"])
        for e, d in res.event_study_effects.items()
        if 0 <= int(e) <= max_e
    ]
    return float(np.mean(post))


def calendar_aggregation(res):
    """Calendar-time ATT, computed from group_time_effects.

    diff-diff has no native calendar aggregation, so we build it the same way
    R's `aggte(type="calendar")` does: for each calendar period t, the
    group-size-weighted mean of ATT(g, t) over groups already treated by t.

    Point estimates are exact. Standard errors use an independence
    approximation across the cells in a period (diff-diff does not expose the
    group-time covariance needed for the exact uniform-band variance); the
    resulting bands are indicative, slightly narrower than the exact ones.
    """
    gt = res.group_time_effects
    crit = float(res.cband_crit_value) if res.cband_crit_value else 1.96
    rows = []
    for t in sorted(int(x) for x in res.time_periods):
        cells = [(g, t) for g in res.groups if g <= t and (g, t) in gt]
        if not cells:
            continue
        w = np.array([gt[c]["n_treated"] for c in cells], dtype=float)
        w /= w.sum()
        eff = np.array([gt[c]["effect"] for c in cells], dtype=float)
        se = np.array([gt[c]["se"] for c in cells], dtype=float)
        att = float(w @ eff)
        se_t = float(np.sqrt(np.sum((w * se) ** 2)))
        rows.append({"period": t, "att": att, "se": se_t,
                     "lo": att - crit * se_t, "hi": att + crit * se_t})
    df = pd.DataFrame(rows)
    df.attrs["overall"] = float(df["att"].mean())
    return df


def true_overall_att(panel):
    """Ground-truth overall ATT, recomputed from the DGP's tau schedule."""
    treated = panel.query("cohort != 0 and period >= cohort").merge(
        TAU_SCHEDULE, on="cohort", how="left"
    )
    tau = treated["base"] + treated["slope"] * (treated["period"] - treated["cohort"])
    return float(tau.mean())


def twfe_overall(panel):
    """Naive static TWFE: sales ~ treated + pre_treatment_sales | store + period."""
    df = panel.set_index(["store_id", "period"])
    # pre_treatment_sales is store-invariant, so EntityEffects absorb it; R's
    # fixest drops it silently, drop_absorbed=True mirrors that behaviour.
    fit = PanelOLS.from_formula(
        "sales ~ treated + pre_treatment_sales + EntityEffects + TimeEffects",
        data=df, drop_absorbed=True,
    ).fit(cov_type="clustered", cluster_entity=True)
    return float(fit.params["treated"]), float(fit.std_errors["treated"])


def twfe_event_study(panel, min_e=MIN_E, max_e=MAX_E, with_holdout=False):
    """Naive TWFE event study via relative-time dummies, e = -1 omitted.

    with_holdout=False drops the never-treated stores -- the forbidden-
    comparisons setup, where already-treated cohorts become the implicit
    control. with_holdout=True keeps them: never-treated rows get rel = -1 so
    they fold into the omitted reference bin and act as pure clean controls."""
    es = panel.copy() if with_holdout else panel.query("cohort != 0").copy()
    es["rel"] = np.where(es["cohort"] == 0, -1, es["period"] - es["cohort"])
    dummies, names = [], []
    for e in sorted(es["rel"].unique()):
        if e == -1:  # omitted reference
            continue
        col = f"rel_m{abs(e)}" if e < 0 else f"rel_p{e}"
        es[col] = (es["rel"] == e).astype(int)
        dummies.append(col)
        names.append(e)
    df = es.set_index(["store_id", "period"])
    formula = (
        "sales ~ " + " + ".join(dummies)
        + " + pre_treatment_sales + EntityEffects + TimeEffects"
    )
    fit = PanelOLS.from_formula(formula, data=df, drop_absorbed=True).fit(
        cov_type="clustered", cluster_entity=True
    )
    rows = [{"e": -1, "att": 0.0, "se": np.nan, "lo": 0.0, "hi": 0.0, "src": "TWFE"}]
    for col, e in zip(dummies, names):
        if e < min_e or e > max_e:
            continue
        # Without a never-treated control the TWFE event study is
        # underidentified, so the most positive relative-time dummies are
        # dropped for collinearity (R's fixest drops rel::6 and rel::7
        # likewise); skip whatever did not survive.
        if col not in fit.params.index:
            continue
        est, se = float(fit.params[col]), float(fit.std_errors[col])
        rows.append({"e": e, "att": est, "se": se,
                     "lo": est - 1.96 * se, "hi": est + 1.96 * se, "src": "TWFE"})
    return pd.DataFrame(rows).sort_values("e").reset_index(drop=True)


###############################################################################
# SECTION 1 -- MAIN ANALYSIS AND FIGURES (reads data/staggered_did.csv)
###############################################################################
panel = pd.read_csv(DATA_MAIN)
panel["cohort_label"] = panel["cohort"].apply(cohort_label)

# ---- A. Data inspection -----------------------------------------------------
print(panel.head())

cohort_summary = (
    panel.drop_duplicates(["store_id", "cohort"])
    .groupby("cohort")["store_id"].nunique()
    .reset_index(name="n_stores")
)
cohort_summary["cohort_label"] = cohort_summary["cohort"].apply(
    lambda x: "Never Treated" if x == 0 else f"Treated in Period {x}"
)
print(cohort_summary)

# ---- F1. Cohort sales trajectories (chapter @fig-cohort-trends) -------------
cohort_means = (
    panel.groupby(["cohort", "cohort_label", "period"])["sales"]
    .mean().reset_index(name="mean_sales")
)
fig, ax = plt.subplots(figsize=(9, 5.5))
for c, color in zip([5, 7, 9, 11], ["#5C8FAF", "#D77A91", "#5E9A78", "#A07AC1"]):
    ax.axvline(c, linestyle=(0, (2, 2)), color=color, alpha=0.6, linewidth=1.6)
for label in COHORT_ORDER:
    sub = cohort_means[cohort_means["cohort_label"] == label]
    if sub.empty:
        continue
    ax.plot(sub["period"], sub["mean_sales"], "-o", color=cohort_palette[label],
            label=label, linewidth=1.6, markersize=5)
ax.set_xticks(range(1, 13))
ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
style_axes(ax, "Cohort treatment and outcome trends",
           "Dashed lines mark each cohort's first treated period",
           "Period (month)", "Average sales (R$ millions)")
save_fig(fig, "cohort_trends")

# ---- Treatment rollout heatmap (chapter Python Block 3) ---------------------
treatment_matrix = panel.pivot_table(
    index="store_id", columns="period", values="treated"
)
fig, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(treatment_matrix, cbar=False, cmap=["white", book_colors["primary"]],
            yticklabels=False, ax=ax)
style_axes(ax, "Treatment rollout: new store layout", None, "Period", "Store")
save_fig(fig, "treatment_rollout")

# ---- C. Main CS estimation (never-treated control, doubly robust) -----------
# Fit CS with never-treated and not-yet-treated controls
cs = fit_cs(panel, control_group="never_treated")
cs_notyet = fit_cs(panel, control_group="not_yet_treated")

gt = cs.group_time_effects
att57 = gt[(5, 7)]
print(f"ATT(5, 7) = {att57['effect']:.3f} (SE {att57['se']:.3f})")

# ---- B. Sensitivity: never-treated vs not-yet-treated -----------------------
sensitivity = pd.DataFrame(
    {
        "control": ["Never-treated", "Not-yet-treated"],
        "att": [cs.overall_att, cs_notyet.overall_att],
        "se": [cs.overall_se, cs_notyet.overall_se],
    }
)
# Sensitivity: never-treated vs not-yet-treated overall ATT
print(sensitivity)

# ---- D. Aggregations --------------------------------------------------------
simple_att, simple_se = cs.overall_att, cs.overall_se
group_effects = {int(g): d for g, d in cs.group_effects.items()}
# R's did::aggte(type="group") overall ATT weights each cohort effect by its
# size (number of treated units). diff-diff exposes only the per-cohort
# effects, so the cohort-size weights are applied here. The overall SE uses an
# independence approximation across cohorts -- each cohort effect is estimated
# from largely separate units, so the cross-cohort covariance is negligible;
# this matches R's analytic SE to two decimals.
cohort_sizes = dict(zip(cohort_summary["cohort"], cohort_summary["n_stores"]))
_g = sorted(group_effects)
_w = np.array([cohort_sizes[g] for g in _g], dtype=float)
_w /= _w.sum()
group_overall = float(np.sum(_w * np.array([group_effects[g]["effect"] for g in _g])))
group_overall_se = float(
    np.sqrt(np.sum((_w * np.array([group_effects[g]["se"] for g in _g])) ** 2))
)
es = event_study_df(cs)
dyn_overall = dynamic_overall(cs)
calendar = calendar_aggregation(cs)
true_att = true_overall_att(panel)

print(f"\nCS simple overall ATT  : {simple_att:.3f} (SE {simple_se:.3f})")
print(f"CS group  overall ATT  : {group_overall:.3f}")
print(f"True overall ATT (DGP) : {true_att:.3f}")

# ---- F2. ATT(g, t) heat-map -------------------------------------------------
cohorts = [5, 7, 9, 11]
periods = list(range(1, 13))
mat = np.full((len(cohorts), len(periods)), np.nan)
for i, g in enumerate(cohorts):
    for j, t in enumerate(periods):
        if (g, t) in gt:
            mat[i, j] = gt[(g, t)]["effect"]
        elif t == g - 1:  # universal base period: reference cell pinned to 0
            mat[i, j] = 0.0
fig, ax = plt.subplots(figsize=(9, 4.5))
vmax = np.nanmax(np.abs(mat))
mesh = ax.imshow(mat, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
for i, g in enumerate(cohorts):
    for j, t in enumerate(periods):
        if np.isnan(mat[i, j]):
            continue
        faded = t < g  # pre-treatment cells are faded
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=True,
                                   color="white", alpha=0.55 if faded else 0.0,
                                   zorder=1))
        # White text on dark (high-magnitude) cells keeps the labels legible.
        dark = (not faded) and abs(mat[i, j]) / vmax > 0.55
        ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8,
                color="white" if dark else book_colors["dark_gray"], zorder=2)
ax.set_xticks(range(len(periods)), periods)
ax.set_yticks(range(len(cohorts)), cohorts)
fig.colorbar(mesh, ax=ax, label="ATT(g, t)")
style_axes(ax, "Group-time average treatment effects",
           "Pre-treatment cells are faded; post-treatment cells show ATT(g, t)",
           "Calendar period (t)", "Cohort (g)")
ax.grid(False)
save_fig(fig, "attgt_matrix")

# ---- F4. Group-specific ATT -------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
crit = float(cs.cband_crit_value)
for x, g in enumerate(cohorts):
    d = group_effects[g]
    ax.errorbar(x, d["effect"], yerr=crit * d["se"], fmt="o",
                color=book_colors["primary"], markersize=8, capsize=4, linewidth=1.6)
ax.axhline(group_overall, linestyle="--", color=book_colors["accent"], linewidth=1.4,
           label="Overall treatment effect")
ax.text(0.98, 0.04,
        f"Overall treatment effect:\naverage ATT across cohorts = R$ {group_overall:.2f} M",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
        color=book_colors["accent"])
ax.set_xticks(range(4), cohorts)
ax.legend(frameon=False, loc="upper right")
style_axes(ax, "Group-specific ATT and uniform 95% bands", None,
           "Cohort (first-treated period)", "Average ATT for cohort")
save_fig(fig, "aggte_group")

# ---- F5. Calendar-time ATT --------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.fill_between(calendar["period"], calendar["lo"], calendar["hi"],
                color=book_colors["primary"], alpha=0.2)
ax.plot(calendar["period"], calendar["att"], "-o", color=book_colors["primary"],
        linewidth=1.6, markersize=5)
cal_overall = calendar.attrs["overall"]
ax.axhline(cal_overall, linestyle="--", color=book_colors["accent"], linewidth=1.4,
           label="Overall treatment effect")
ax.text(0.02, 0.96,
        f"Overall treatment effect:\naverage ATT across calendar periods "
        f"= R$ {cal_overall:.2f} M",
        transform=ax.transAxes, ha="left", va="top", fontsize=9,
        color=book_colors["accent"])
ax.set_xticks(calendar["period"])
ax.legend(frameon=False, loc="lower right")
style_axes(ax, "Calendar-time ATT and uniform 95% bands", None,
           "Calendar period (t)", "Average ATT in period t")
save_fig(fig, "aggte_calendar")

# ---- F6. Dynamic ATT (event study) ------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.axhline(0, color=book_colors["muted"], linewidth=0.9)
ax.axvline(-0.5, linestyle="--", color=book_colors["muted"], linewidth=0.9)
for _, r in es.iterrows():
    col = book_colors["secondary"] if r["e"] < 0 else book_colors["primary"]
    yerr = None if r["e"] == -1 else [[r["att"] - r["lo"]], [r["hi"] - r["att"]]]
    ax.errorbar(r["e"], r["att"], yerr=yerr, fmt="o", color=col, markersize=6,
                capsize=3, linewidth=1.4)
ax.axhline(dyn_overall, linestyle="--", color=book_colors["accent"], linewidth=1.4)
ax.text(0.02, 0.96,
        f"Overall treatment effect:\naverage post-treatment ATT (e >= 0) "
        f"= R$ {dyn_overall:.2f} M",
        transform=ax.transAxes, ha="left", va="top", fontsize=9,
        color=book_colors["accent"])
ax.set_xticks(range(MIN_E, MAX_E + 1))
style_axes(ax, "Dynamic ATT (event study) and uniform 95% bands", None,
           "Event time (periods since treatment)", "Average ATT")
save_fig(fig, "aggte_dynamic")

# ---- E. TWFE vs CS ----------------------------------------------------------
twfe_est, twfe_se = twfe_overall(panel)
attenuation_pct = 100 * (1 - twfe_est / group_overall)
# TWFE vs CS overall ATT
print(pd.DataFrame({
    "estimator": ["Naive TWFE", "Callaway-Sant'Anna"],
    "estimate": [twfe_est, group_overall],
    "se": [twfe_se, group_overall_se],
}))
print(f"Attenuation (1 - TWFE/CS) * 100 = {attenuation_pct:.1f}%")

# ---- E (figure). Overall ATT -- TWFE vs CS vs ground truth -----------------
fig, ax = plt.subplots(figsize=(7, 4.5))
labels = ["True ATT", "Naive TWFE", "Callaway-Sant'Anna"]
values = [true_att, twfe_est, group_overall]
ses = [np.nan, twfe_se, group_overall_se]
colors = [book_colors["muted"], book_colors["success"], book_colors["primary"]]
for x, (v, s, col) in enumerate(zip(values, ses, colors)):
    if np.isnan(s):
        ax.plot(x, v, "o", color=col, markersize=9)
    else:
        ax.errorbar(x, v, yerr=1.96 * s, fmt="o", color=col, markersize=9,
                    capsize=4, linewidth=2)
ax.set_xticks(range(3), labels)
style_axes(ax, "Overall ATT -- naive TWFE vs Callaway-Sant'Anna vs ground truth",
           None, None, "Overall ATT (R$ millions)")
save_fig(fig, "twfe_vs_cs_aggregated")  # writes images/twfe_vs_cs_aggregated_py.png

# ---- F7. TWFE vs CS event study: with vs without the never-treated holdout --
def plot_es_compare(ax, twfe_df, cs_df):
    """Overlay a TWFE and a CS event study on one axis.

    Each estimator also gets a dashed, colour-matched line at its overall
    effect -- the average of its own post-treatment ATT (e >= 0)."""
    ax.axhline(0, color=book_colors["muted"], linewidth=0.9)
    ax.axvline(-0.5, linestyle="--", color=book_colors["muted"], linewidth=0.9)
    for src, sub, color, off in [("TWFE", twfe_df, book_colors["success"], -0.12),
                                 ("CS", cs_df, book_colors["primary"], 0.12)]:
        for i, (_, r) in enumerate(sub.iterrows()):
            yerr = (None if np.isnan(r["se"])
                    else [[r["att"] - r["lo"]], [r["hi"] - r["att"]]])
            ax.errorbar(r["e"] + off, r["att"], yerr=yerr, fmt="o", color=color,
                        markersize=5, capsize=2, linewidth=1.2, alpha=0.65,
                        label=src if i == 0 else None)
        overall = sub.loc[sub["e"] >= 0, "att"].mean()
        ax.axhline(overall, linestyle="--", color=color, linewidth=1.4, alpha=0.65)
    ax.set_xticks(range(MIN_E, MAX_E + 1))
    handles, labels = ax.get_legend_handles_labels()
    overall_lbl = "Overall effect (avg post-treatment ATT)"
    handles.append(plt.Line2D([], [], color=book_colors["muted"],
                              linestyle="--", label=overall_lbl))
    labels.append(overall_lbl)
    ax.legend(handles, labels, frameon=False, loc="upper left")


# E.2a -- WITH the never-treated holdout: both estimators have a clean control.
twfe_hold_df = twfe_event_study(panel, with_holdout=True)
cs_hold_df = es.assign(src="CS")
fig, ax = plt.subplots(figsize=(9, 5))
plot_es_compare(ax, twfe_hold_df, cs_hold_df)
style_axes(ax,
           "TWFE vs Callaway-Sant'Anna event study -- with the never-treated holdout",
           "Both estimators use the BA never-treated stores as a clean control",
           "Event time (periods since treatment)", "Average ATT")
save_fig(fig, "twfe_vs_cs_es_holdout")

# E.2b -- holdout removed: neither estimator has a clean control.
panel_noh = panel.query("cohort != 0").copy()
twfe_noh_df = twfe_event_study(panel, with_holdout=False)
# Refit CS on the holdout-removed sample
cs_noh = fit_cs(panel_noh, control_group="not_yet_treated")
cs_noh_df = event_study_df(cs_noh).assign(src="CS")
fig, ax = plt.subplots(figsize=(9, 5))
plot_es_compare(ax, twfe_noh_df, cs_noh_df)
style_axes(ax,
           "TWFE vs Callaway-Sant'Anna event study -- holdout removed",
           "With no never-treated control, TWFE leans on already-treated cohorts",
           "Event time (periods since treatment)", "Average ATT")
save_fig(fig, "twfe_vs_cs_es_noholdout")

###############################################################################
# SECTION 2 -- COVARIATE ANALYSIS AND FIGURES (reads data/staggered_did_cov.csv)
###############################################################################
# SECTION 2 -- covariate analysis (data/staggered_did_cov.csv)

panel_cov = pd.read_csv(DATA_COV)
panel_cov["cohort_label"] = panel_cov["cohort"].apply(cohort_label)

# ---- F1. Cohort trends on the covariate dataset -----------------------------
cohort_means_cov = (
    panel_cov.groupby(["cohort", "cohort_label", "period"])["sales"]
    .mean().reset_index(name="mean_sales")
)
fig, ax = plt.subplots(figsize=(9, 5.5))
for c, color in zip([5, 7, 9, 11], ["#5C8FAF", "#D77A91", "#5E9A78", "#A07AC1"]):
    ax.axvline(c, linestyle=(0, (2, 2)), color=color, alpha=0.6, linewidth=1.6)
for label in COHORT_ORDER:
    sub = cohort_means_cov[cohort_means_cov["cohort_label"] == label]
    if sub.empty:
        continue
    ax.plot(sub["period"], sub["mean_sales"], "-o", color=cohort_palette[label],
            label=label, linewidth=1.6, markersize=5)
ax.set_xticks(range(1, 13))
ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
style_axes(ax, "Cohort trends when treatment timing depends on observables",
           "Urban, high-baseline cohorts trend up faster even before they are treated",
           "Period (month)", "Average sales (R$ millions)")
save_fig(fig, "cov_cohort_trends")

# ---- F2. Overlap diagnostic (two panels) ------------------------------------
stores = panel_cov.drop_duplicates(["store_id"])[
    ["store_id", "cohort", "cohort_label", "urban", "pre_treatment_sales"]
].copy()
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
box_data = [stores.loc[stores["cohort_label"] == lbl, "pre_treatment_sales"].values
            for lbl in COHORT_ORDER]
bp = axes[0].boxplot(box_data, patch_artist=True, widths=0.6)
for patch, lbl in zip(bp["boxes"], COHORT_ORDER):
    patch.set_facecolor(cohort_palette[lbl])
    patch.set_alpha(0.7)
axes[0].set_xticks(range(1, 6), COHORT_ORDER, rotation=25, ha="right")
style_axes(axes[0], "Pre-treatment sales by cohort",
           "Boxplots overlap -- comparable stores in every cohort",
           None, "pre_treatment_sales")
urban_share = stores.groupby("cohort_label")["urban"].mean().reindex(COHORT_ORDER)
bars = axes[1].bar(range(5), urban_share.values, width=0.65,
                   color=[cohort_palette[lbl] for lbl in COHORT_ORDER], alpha=0.85)
for rect, val in zip(bars, urban_share.values):
    axes[1].text(rect.get_x() + rect.get_width() / 2, val + 0.02, f"{val:.0%}",
                 ha="center", fontsize=9, color=book_colors["dark_gray"])
axes[1].set_xticks(range(5), COHORT_ORDER, rotation=25, ha="right")
axes[1].set_ylim(0, max(urban_share.values) * 1.2)
style_axes(axes[1], "Urban share by cohort",
           "Earlier cohorts skew urban, but every cohort has both types",
           None, "Share of urban stores")
fig.tight_layout()
save_fig(fig, "cov_overlap")

# ---- CS estimation: unconditional vs conditional ----------------------------
# Fit unconditional and conditional CS on the covariate dataset
cs_uncond = fit_cs(panel_cov, covariates=None)
cs_cond = fit_cs(panel_cov, covariates=["pre_treatment_sales", "urban"])

es_uncond = event_study_df(cs_uncond)
es_cond = event_study_df(cs_cond)
uncond_dyn_overall = dynamic_overall(cs_uncond)
cond_dyn_overall = dynamic_overall(cs_cond)
true_att_cov = true_overall_att(panel_cov)

# ---- F3. Two-panel event study: unconditional vs conditional ----------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
panels = [("Without covariates", es_uncond, uncond_dyn_overall),
          ("With covariates", es_cond, cond_dyn_overall)]
for ax, (title, es_df, dyn_ov) in zip(axes, panels):
    ax.axhline(0, color=book_colors["muted"], linewidth=0.9)
    ax.axvline(-0.5, linestyle="--", color=book_colors["muted"], linewidth=0.9)
    for _, r in es_df.iterrows():
        col = book_colors["secondary"] if r["e"] < 0 else book_colors["primary"]
        yerr = (None if r["e"] == -1
                else [[r["att"] - r["lo"]], [r["hi"] - r["att"]]])
        ax.errorbar(r["e"], r["att"], yerr=yerr, fmt="o", color=col, markersize=5,
                    capsize=3, linewidth=1.3)
    ax.axhline(dyn_ov, linestyle="--", color=book_colors["accent"], linewidth=1.4)
    ax.text(0.02, 0.97,
            f"Overall treatment effect:\naverage post-treatment ATT (e >= 0) "
            f"= R$ {dyn_ov:.2f} M",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            color=book_colors["accent"])
    ax.set_xticks(range(MIN_E, MAX_E + 1))
    style_axes(ax, title, None, "Event time (periods since treatment)",
               "Average ATT (R$ millions)")
fig.suptitle("Event study, before and after conditioning on covariates",
             fontweight="bold", color=book_colors["dark_gray"], x=0.02, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.96))
save_fig(fig, "cov_eventstudy_compare")

###############################################################################
# SECTION 3 -- PROSE ANCHORS (numbers the chapter text quotes)
###############################################################################
# SECTION 3 -- PROSE ANCHORS


def anchor(label, value):
    print(f"{label:<44} : {value}")


anchor("S3.1  never-treated overall ATT",
       f"{cs.overall_att:.2f} (SE {cs.overall_se:.2f})")
anchor("S3.1  not-yet-treated overall ATT",
       f"{cs_notyet.overall_att:.2f} (SE {cs_notyet.overall_se:.2f})")
anchor("S4    ATT(5, 7)", f"{att57['effect']:.2f} (SE {att57['se']:.2f})")
anchor("S4    overall ATT (CS simple)", f"{simple_att:.2f} (SE {simple_se:.2f})")
anchor("S4    overall ATT (CS group, headline)",
       f"{group_overall:.2f} (SE {group_overall_se:.2f})")
anchor("S4    true ATT (simulation)", f"{true_att:.2f}")
for g in cohorts:
    anchor(f"S4    group ATT(g={g})", f"{group_effects[g]['effect']:.2f}")
es_idx = es.set_index("e")
anchor("S4    dynamic ATT e=0", f"{es_idx.loc[0, 'att']:.2f}")
anchor("S4    dynamic ATT e=4", f"{es_idx.loc[4, 'att']:.2f}")
anchor("S5    TWFE overall", f"{twfe_est:.2f} (SE {twfe_se:.2f})")
anchor("S5    TWFE attenuation", f"{attenuation_pct:.1f}%")
twfe_hold_idx = twfe_hold_df.set_index("e")
twfe_noh_idx = twfe_noh_df.set_index("e")
anchor("S5    TWFE ES e=-6 (with holdout)", f"{twfe_hold_idx.loc[-6, 'att']:+.2f}")
anchor("S5    TWFE ES e=0  (with holdout)", f"{twfe_hold_idx.loc[0, 'att']:+.2f}")
anchor("S5    TWFE ES e=7  (with holdout)", f"{twfe_hold_idx.loc[7, 'att']:+.2f}")
anchor("S5    TWFE ES e=-6 (no holdout)", f"{twfe_noh_idx.loc[-6, 'att']:+.2f}")
anchor("S5    TWFE ES e=0  (no holdout)", f"{twfe_noh_idx.loc[0, 'att']:+.2f}")
anchor("S5    TWFE ES e=5  (no holdout)", f"{twfe_noh_idx.loc[5, 'att']:+.2f}")
anchor("S5    CS ES e=0", f"{es_idx.loc[0, 'att']:+.2f}")
anchor("S5    CS ES e=7", f"{es_idx.loc[7, 'att']:+.2f}")

print()
anchor("Cov   true overall ATT (DGP)", f"{true_att_cov:.2f}")
anchor("Cov   unconditional CS overall ATT",
       f"{cs_uncond.overall_att:.2f} (SE {cs_uncond.overall_se:.2f})")
anchor("Cov   conditional CS overall ATT",
       f"{cs_cond.overall_att:.2f} (SE {cs_cond.overall_se:.2f})")
anchor("Cov   bias (unconditional)",
       f"{100 * (cs_uncond.overall_att - true_att_cov) / true_att_cov:+.0f}% of true")
anchor("Cov   bias (conditional)",
       f"{100 * (cs_cond.overall_att - true_att_cov) / true_att_cov:+.1f}% of true")

