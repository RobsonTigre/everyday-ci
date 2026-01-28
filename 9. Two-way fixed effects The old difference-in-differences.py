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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearmodels.panel import PanelOLS
import statsmodels.api as sm
from statsmodels.tsa.arima_process import ArmaProcess
from scipy import stats as scipy_stats

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
# DATA GENERATING PROCESS (DGP)
# ==========================================================
# Simulating an OOH (Out-of-Home) marketing campaign in the
# Brazilian state of Paraiba (PB), with Pernambuco (PE) as control

np.random.seed(789)
n_stores = 100   # Number of stores
n_periods = 10   # Number of periods

# Create balanced panel
stores = np.repeat(np.arange(1, n_stores + 1), n_periods)
periods = np.tile(np.arange(1, n_periods + 1), n_stores)

# Create DataFrame
data = pd.DataFrame({
    'store': stores,
    'period': periods
})

# States: 60% in Pernambuco (control) and 40% in Paraiba (treatment)
data['state'] = np.where(data['store'] <= 0.6 * n_stores, 'PE', 'PB')

# Treatment indicator (1 for PB stores)
data['treatment'] = np.where(data['state'] == 'PB', 1, 0)

# Store-level fixed effects
store_effects = np.random.normal(loc=2, scale=1, size=n_stores)
data['store_effect'] = data['store'].apply(lambda x: store_effects[x - 1])

# Simulate AR(1) errors for each store (serial correlation)
ar_coef = 0.6
ar_params = np.array([1, -ar_coef])  # AR(1) process
ma_params = np.array([1])
ar_process = ArmaProcess(ar_params, ma_params)

errors = []
for store_id in range(1, n_stores + 1):
    # Generate AR(1) errors for this store
    store_errors = ar_process.generate_sample(nsample=n_periods, scale=2)
    errors.extend(store_errors)
data['error'] = errors

# Baseline sales (in millions of R$) with time trend and store effects
data['sales_base'] = (
    (1 + data['period'] / 20) *
    np.random.normal(
        loc=np.where(data['state'] == 'PE', 12, 10),
        scale=2
    ) +
    data['store_effect'] +
    data['error']
)

# Apply treatment effects with varying intensity over time
def apply_treatment(row):
    if row['treatment'] == 1:
        if row['period'] == 7:
            return row['sales_base'] * 1.10  # 10% lift
        elif row['period'] == 8:
            return row['sales_base'] * 1.25  # 25% lift
        elif row['period'] == 9:
            return row['sales_base'] * 1.20  # 20% lift
        elif row['period'] == 10:
            return row['sales_base'] * 1.15  # 15% lift
    return row['sales_base']

data['sales'] = data.apply(apply_treatment, axis=1)

# Dummy for treated units in post-treatment periods
data['treated_period'] = np.where(
    (data['treatment'] == 1) & (data['period'] >= 7), 1, 0
)

# Relative period (event time): treatment starts at period 7
data['relative_period'] = data['period'] - 7

# Sort and select columns
data = data.sort_values(['store', 'period']).reset_index(drop=True)
data = data[['store', 'period', 'state', 'treatment', 'sales', 'treated_period', 'relative_period']]

# Save the dataset
# data.to_csv("/Users/robsontigre/Desktop/everyday-ci/data/did-twfe-ooh.csv", index=False)
# Data saved to the specified path


# ==========================================================
# DESCRIPTIVE STATISTICS
# ==========================================================
data = pd.read_csv("/Users/robsontigre/Desktop/everyday-ci/data/did-twfe-ooh.csv")

# Quick summary of the dataset structure and distributions
print(data.describe())

# Check data structure
print(data.info())


# ==========================================================
# TESTING FOR SERIAL CORRELATION
# ==========================================================
# Wooldridge's test for serial correlation in panel data
# H0: No serial correlation in the idiosyncratic errors
# Since we simulated AR(1) errors, we expect to reject H0

# Step 1: Run a fixed-effects regression of sales on period
data_test = data.set_index(['store', 'period'])

fe_model = PanelOLS(
    data_test['sales'],
    sm.add_constant(data_test[['relative_period']].astype(float)),
    entity_effects=True
)

fe_resid = fe_model.fit().resids

# Step 2: First-difference the residuals within each store
resid_df = pd.DataFrame({
    'store': data['store'].values,
    'period': data['period'].values,
    'resid': fe_resid.values
}).sort_values(['store', 'period'])

resid_df['resid_diff'] = resid_df.groupby('store')['resid'].diff()
resid_df['resid_diff_lag'] = resid_df.groupby('store')['resid_diff'].shift(1)
resid_df = resid_df.dropna()

# Step 3: Regress differenced residuals on their lag
X_test = sm.add_constant(resid_df['resid_diff_lag'])
ols_test = sm.OLS(resid_df['resid_diff'], X_test).fit(cov_type='HC1')
print("\nWooldridge test for serial correlation in panel data:")
print(f"  Coefficient on lagged diff: {ols_test.params.iloc[1]:.4f}")
print(f"  t-statistic: {ols_test.tvalues.iloc[1]:.4f}")
print(f"  p-value: {ols_test.pvalues.iloc[1]:.4f}")

# Interpretation: A significant p-value indicates the presence of
# autocorrelation in the residuals, which justifies our use of
# clustered standard errors at the store level.


# ==========================================================
# EXPLORATORY DATA ANALYSIS
# Figure: raw-data-chap-9-py.png
# ==========================================================

# Average sales by group over time
avg_sales = data.groupby(['period', 'state'])['sales'].mean().reset_index()
avg_sales_pivot = avg_sales.pivot(index='period', columns='state', values='sales')

# Plot
fig, ax = plt.subplots(figsize=(10, 7))

ax.plot(avg_sales_pivot.index, avg_sales_pivot['PE'],
        marker='o', color=book_colors['secondary'], label='Control (PE)',
        linewidth=2, markersize=8)
ax.plot(avg_sales_pivot.index, avg_sales_pivot['PB'],
        marker='o', color=book_colors['primary'], label='Treatment (PB)',
        linewidth=2, markersize=8)

# Treatment start line
ax.axvline(x=7, linestyle='--', color=book_colors['muted'], linewidth=1.2, alpha=0.8)

# Annotation for treatment start
ax.annotate('OOH campaign\nstarts', xy=(7, ax.get_ylim()[1] * 0.85),
            fontsize=10, color=book_colors['muted'], ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none'))

# Labels and title
ax.set_xlabel('Period', fontsize=14, color=book_colors['dark_gray'])
ax.set_ylabel('Average sales (millions R$)', fontsize=14, color=book_colors['dark_gray'])
ax.set_title('Sales trajectories before and after the OOH campaign',
             fontsize=16, fontweight='bold', color='#333333', pad=15)

# Subtitle
ax.text(0.5, 1.02, 'Stores in Paraiba (treatment) vs Pernambuco (control)',
        transform=ax.transAxes, fontsize=11, color='grey', ha='center')

# Caption (takeaway)
fig.text(0.1, -0.02,
         'Takeaway: Both groups follow similar trends pre-treatment, then diverge after the campaign starts.',
         fontsize=9, color='grey', ha='left')

ax.set_xticks(range(1, 11))
ax.legend(loc='lower right', frameon=False)

plt.tight_layout()
plt.subplots_adjust(bottom=0.12)
plt.show()
# plt.savefig('images/raw-data-chap-9-py.png', dpi=300, bbox_inches='tight')


# ==========================================================
# AGGREGATED DIFFERENCE-IN-DIFFERENCES MODEL
# ==========================================================
# Model: Y_it = gamma_i + lambda_t + delta * D_it + epsilon_it
# Where:
#   - gamma_i: store fixed effects
#   - lambda_t: period fixed effects
#   - D_it: treatment indicator (treated_period)
#   - delta: average treatment effect (ATT)

# Set multi-index for panel data
data_panel = data.set_index(['store', 'period'])

# Prepare variables (no constant needed - absorbed by fixed effects)
X_agg = data_panel[['treated_period']]
y = data_panel['sales']

# Fit model with entity and time fixed effects
agg_did_model = PanelOLS(y, X_agg, entity_effects=True, time_effects=True)
agg_did_results = agg_did_model.fit(cov_type='clustered', cluster_entity=True)

print(agg_did_results.summary)

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

# Reset data_panel to avoid issues
data_panel = data.set_index(['store', 'period'])

# Create interaction terms for event study
# Exclude the reference period (-1)
event_study_terms = []
for p in sorted(data['relative_period'].unique()):
    if p != -1:  # Exclude reference period
        col_name = f'rel_period_{p}_treated'
        # Create interaction: (relative_period == p) * treatment
        data_panel[col_name] = ((data['relative_period'].values == p) & (data['treatment'].values == 1)).astype(int)
        event_study_terms.append(col_name)

# Prepare variables for event study (no constant - absorbed by fixed effects)
X_event = data_panel[event_study_terms]
y_event = data_panel['sales']

# Fit event study model
event_study_model = PanelOLS(y_event, X_event, entity_effects=True, time_effects=True)
event_study_results = event_study_model.fit(cov_type='clustered', cluster_entity=True)

print(event_study_results.summary)

# Joint F-test for pre-treatment coefficients (leads)
# H0: All pre-treatment effects are jointly zero
pre_treatment_terms = [t for t in event_study_terms if int(t.split('_')[2]) < -1]
pre_coefs = event_study_results.params[pre_treatment_terms].values
pre_vcov = event_study_results.cov.loc[pre_treatment_terms, pre_treatment_terms].values

# Wald statistic: beta' * V^{-1} * beta / q ~ F(q, N-k)
q = len(pre_treatment_terms)
wald_stat = pre_coefs @ np.linalg.inv(pre_vcov) @ pre_coefs / q
df_resid = event_study_results.nobs - event_study_results.params.shape[0]
p_value = 1 - scipy_stats.f.cdf(wald_stat, q, df_resid)

print(f"\nJoint F-test for pre-treatment coefficients:")
print(f"  F({q}, {df_resid}) = {wald_stat:.4f}")
print(f"  p-value = {p_value:.4f}")


# ==========================================================
# EVENT STUDY PLOT
# Figure: es-plot-chap-9-py.png
# ==========================================================

# Extract coefficients and confidence intervals
coef_names = [col for col in event_study_results.params.index if col.startswith('rel_period')]
coefs = event_study_results.params[coef_names]
std_errors = event_study_results.std_errors[coef_names]

# Parse relative periods from coefficient names
rel_periods = [int(name.split('_')[2]) for name in coef_names]

# Create DataFrame for plotting
coef_df = pd.DataFrame({
    'relative_period': rel_periods,
    'estimate': coefs.values,
    'std_error': std_errors.values
})

# Add reference period (k = -1) with zero effect
ref_row = pd.DataFrame({
    'relative_period': [-1],
    'estimate': [0],
    'std_error': [0]
})
coef_df = pd.concat([coef_df, ref_row], ignore_index=True)
coef_df = coef_df.sort_values('relative_period').reset_index(drop=True)

# Calculate confidence intervals (95%)
coef_df['ci_lower'] = coef_df['estimate'] - 1.96 * coef_df['std_error']
coef_df['ci_upper'] = coef_df['estimate'] + 1.96 * coef_df['std_error']

# Plot
fig, ax = plt.subplots(figsize=(10, 7))

# Shade pre-treatment region
ax.axvspan(-6.5, -0.5, alpha=0.15, color=book_colors['light_gray'], zorder=0)

# Reference lines
ax.axhline(y=0, linestyle='--', color=book_colors['muted'], linewidth=1, alpha=0.8)
ax.axvline(x=-0.5, linestyle='--', color=book_colors['muted'], linewidth=1, alpha=0.8)

# Error bars
ax.errorbar(
    coef_df['relative_period'],
    coef_df['estimate'],
    yerr=[coef_df['estimate'] - coef_df['ci_lower'],
          coef_df['ci_upper'] - coef_df['estimate']],
    fmt='o',
    color=book_colors['primary'],
    markersize=8,
    capsize=4,
    capthick=1.5,
    linewidth=1.5,
    ecolor=book_colors['primary'],
    zorder=3
)

# Annotations
ax.annotate('Pre-treatment\n(should be ~0)',
            xy=(-3.5, coef_df['ci_upper'].max() * 0.6),
            fontsize=10, color=book_colors['muted'], ha='center', style='italic')

ax.annotate('Post-treatment\n(treatment effect)',
            xy=(2.5, coef_df['ci_upper'].max() * 0.95),
            fontsize=10, color=book_colors['primary'], ha='center', style='italic')

# Labels and title
ax.set_xlabel('Periods relative to treatment', fontsize=14, color=book_colors['dark_gray'])
ax.set_ylabel('Estimated effect (millions R$)', fontsize=14, color=book_colors['dark_gray'])
ax.set_title('Event study: effect of OOH campaign on sales over time',
             fontsize=16, fontweight='bold', color='#333333', pad=15)

# Subtitle
ax.text(0.5, 1.02, 'Reference period: k = -1 (one period before treatment)',
        transform=ax.transAxes, fontsize=11, color='grey', ha='center')

# Caption (takeaway)
fig.text(0.1, -0.02,
         'Takeaway: No significant pre-trends; effect peaks at k=1, then gradually declines.',
         fontsize=9, color='grey', ha='left')

ax.set_xticks(range(-6, 4))
plt.tight_layout()
plt.subplots_adjust(bottom=0.12)
plt.show()
# plt.savefig('images/es-plot-chap-9-py.png', dpi=300, bbox_inches='tight')


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
