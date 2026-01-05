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
# If you haven't already, run this in your terminal to install the packages:
# pip install pandas numpy matplotlib scipy statsmodels (or use "pip3")

# You must run the lines below at the start of every new Python session.
import pandas as pd  # Data manipulation
import numpy as np  # Mathematical computing
import matplotlib.pyplot as plt  # Data visualization
from scipy.stats import gaussian_kde, ttest_ind  # Statistical functions
from statsmodels.stats.power import TTestIndPower  # Power analysis


#########################################
# Section 5.1: Hypothesis testing illustration
# Figure: hypothesis-test-panels.png
#########################################
np.random.seed(123)

# Simulate "Parallel Universes" (The Null Hypothesis)
sim_data = pd.DataFrame({'effect': np.random.normal(0, 7, 10000)})

# Calculate critical values for alpha = 0.05 (two-tailed)
critical_val_upper = np.quantile(sim_data['effect'], 0.975)
critical_val_lower = np.quantile(sim_data['effect'], 0.025)

# The observed result from the A/B test
observed_effect = 20

# Create a helper column for fill color
sim_data['region'] = np.where(
    (sim_data['effect'] > critical_val_upper) | (sim_data['effect'] < critical_val_lower),
    'Rejection region',
    'Fail to reject'
)

print("=" * 60)
print("Section 5.1: Hypothesis Testing")
print("=" * 60)
print(f"Critical value (upper): {critical_val_upper:.2f}")
print(f"Critical value (lower): {critical_val_lower:.2f}")
print(f"Observed effect: {observed_effect}")
print(f"Result: Observed effect ({observed_effect}) > critical value ({critical_val_upper:.2f}) -> Reject H0")


#########################################
# Section 5.3: Basic Power Analysis Example
# Recommendation system A/B test
#########################################
# Historical means and pooled standard deviation
mean_control = 150
mean_treatment = 165
sd_pooled = 75

# Compute effect size (Cohen's d)
effect_size = abs(mean_treatment - mean_control) / sd_pooled

# Power analysis for two-sample t-test (two-sided)
analysis = TTestIndPower()
required_n = analysis.solve_power(
    effect_size=effect_size,
    power=0.80,
    alpha=0.05,
    ratio=1.0,  # balanced groups
    alternative='two-sided'
)

# Required sample size per group
required_n_ceiling = int(np.ceil(required_n))

print("\n" + "=" * 60)
print("Section 5.3: Basic Power Analysis Example")
print("=" * 60)
print(f"Effect size (Cohen's d): {effect_size}")
print(f"Required sample size per group: {required_n_ceiling}")
print(f"Total sample size: {required_n_ceiling * 2}")


#########################################
# Section 5.3: Power vs Sample Size Curve
# Figure: power_vs_sample_size.png
#########################################
# Define parameters
effect_d = 0.2
alpha_level = 0.05
n_range = np.arange(10, 610, 10)
power_analysis = TTestIndPower()

# Calculate power for each sample size
power_values = [power_analysis.power(effect_size=effect_d, nobs1=n, alpha=alpha_level,
                                      alternative='two-sided') for n in n_range]

# Find the minimum n for 80% power
power_array = np.array(power_values)
min_n_80 = n_range[np.where(power_array >= 0.80)[0][0]]

print("\n" + "=" * 60)
print("Section 5.3: Power vs Sample Size")
print("=" * 60)
print(f"Effect size d: {effect_d}")
print(f"Minimum N per group for 80% power: {min_n_80}")


#########################################
# Section 5.4.1: Effect Size vs Required Sample Size
# Figure: effect_size_vs_n_python.png
#########################################
# Define a sequence of effect sizes to test
effect_sizes = np.arange(0.1, 0.51, 0.05)

# Calculate the required sample size for each effect size
sample_sizes = []
for d in effect_sizes:
    n = power_analysis.solve_power(
        effect_size=d,
        power=0.80,
        alpha=0.05,
        ratio=1.0,
        alternative='two-sided'
    )
    sample_sizes.append(int(np.ceil(n)))

print("\n" + "=" * 60)
print("Section 5.4.1: Effect Size vs Required Sample Size")
print("=" * 60)
for d, n in zip(effect_sizes, sample_sizes):
    print(f"  d = {d:.2f} -> N per group = {n}")


#########################################
# Section 5.4.2: Power Curves for Different Significance Levels
# Figure: power_curves_alpha_python.png
#########################################
# Define parameters
alpha_levels = [0.01, 0.05, 0.10]
n_seq = np.linspace(10, 200, 100)

# Calculate power for each alpha level at n=100 (example point)
print("\n" + "=" * 60)
print("Section 5.4.2: Power Curves for Different Alpha Levels")
print("=" * 60)
print("Power at n=100 per group (d=0.3):")
for alpha in alpha_levels:
    power_at_100 = power_analysis.power(effect_size=0.3, nobs1=100, alpha=alpha, alternative='two-sided')
    print(f"  α = {alpha} -> Power = {power_at_100:.3f}")


#########################################
# Section 5.4.3: Power analysis - allocation ratio
# Figure: power-allocation-ratio.png
#########################################
total_n = 1000
effect_size_alloc = 0.2
allocation_ratios = np.arange(0.1, 1.0, 0.1)

# Calculate power for different allocation ratios
powers_alloc = []
for ratio in allocation_ratios:
    n1 = total_n * ratio
    # ratio parameter in statsmodels is n2/n1
    power = power_analysis.power(effect_size_alloc, nobs1=n1, alpha=0.05, 
                                  ratio=(1-ratio)/ratio, alternative='two-sided')
    powers_alloc.append(power)

# Find maximum power
max_power = max(powers_alloc)
max_ratio = allocation_ratios[np.argmax(powers_alloc)]

print("\n" + "=" * 60)
print("Section 5.4.3: Treatment Allocation Ratio")
print("=" * 60)
print(f"Total N: {total_n}, Effect size d: {effect_size_alloc}")
for ratio, power in zip(allocation_ratios, powers_alloc):
    marker = " <-- MAX" if ratio == max_ratio else ""
    print(f"  Ratio {ratio:.1f} -> Power = {power:.4f}{marker}")
print(f"\nOptimal ratio: {max_ratio:.1f} with power = {max_power:.4f}")


#########################################
# Section 5.4.4: Outcome variance comparison
# Figure: variance_comparison_r.png
#########################################
np.random.seed(123)

# Parameters for both scenarios
mean_diff = 20
n = 10000

# High variance example
mean1_high_var = 100
mean2_high_var = mean1_high_var + mean_diff
sd_high_var = 20

group1_high_var = np.random.normal(mean1_high_var, sd_high_var, n)
group2_high_var = np.random.normal(mean2_high_var, sd_high_var, n)

# Low variance example
mean1_low_var = 100
mean2_low_var = mean1_low_var + mean_diff
sd_low_var = 5

group1_low_var = np.random.normal(mean1_low_var, sd_low_var, n)
group2_low_var = np.random.normal(mean2_low_var, sd_low_var, n)

print("\n" + "=" * 60)
print("Section 5.4.4: Outcome Variance Comparison")
print("=" * 60)
print("High variance scenario (SD=20):")
print(f"  Control mean: {np.mean(group1_high_var):.2f}")
print(f"  Treatment mean: {np.mean(group2_high_var):.2f}")
print("Low variance scenario (SD=5):")
print(f"  Control mean: {np.mean(group1_low_var):.2f}")
print(f"  Treatment mean: {np.mean(group2_low_var):.2f}")


#########################################
# Section 5.4.4: Power vs Outcome Variance (Simulation)
# Figure: power_vs_variance.png
#########################################
np.random.seed(42)

def simulate_power(n_total, sd_outcome, prop_treatment=0.5,
                   true_effect_percent=5, n_simulations=1000,
                   significance_level=0.05):
    """Simulate experiments and estimate power empirically."""
    
    base_value = 100
    true_effect = base_value * (true_effect_percent / 100)
    significant_tests = 0
    
    for _ in range(n_simulations):
        n_control = round(n_total * (1 - prop_treatment))
        control_data = np.random.normal(base_value, sd_outcome, n_control)
        n_treatment = n_total - n_control
        treatment_data = np.random.normal(base_value + true_effect, sd_outcome, n_treatment)
        _, p_value = ttest_ind(treatment_data, control_data)
        if p_value < significance_level:
            significant_tests += 1
    
    return significant_tests / n_simulations

# Test different standard deviation levels
sds = np.arange(15, 55, 5)
power_by_sd = [simulate_power(n_total=800, sd_outcome=sd, prop_treatment=0.5) for sd in sds]

print("\n" + "=" * 60)
print("Section 5.4.4: Power vs Outcome Variance (Simulation)")
print("=" * 60)
print("N=800, 5% effect, α=0.05, 50/50 allocation:")
for sd, power in zip(sds, power_by_sd):
    print(f"  SD = {sd} -> Power = {power:.3f}")


#########################################
# Section 5.4.5: Compliance effect
# Figure: compliance_effect_r.png and compliance_dillution_r.png
#########################################
np.random.seed(123)

# Sample sizes
n_control = 1000
n_treatment = 1000

# Proportion of compliers in the treatment group
prop_compliers = 0.1

# Generate outcome variable for control group
y_control = np.random.normal(50, 10, n_control)

# Generate outcome variable for non-compliers (same baseline as control)
# Note: Non-compliers must have the same mean as control to satisfy the
# exclusion restriction - assignment only affects Y through actual treatment.
y_non_compliers = np.random.normal(50, 10, int(n_treatment * (1 - prop_compliers)))

# Generate outcome variable for compliers (shifted significantly)
y_compliers = np.random.normal(75, 10, int(n_treatment * prop_compliers))

# Calculate means
mean_control_comp = np.mean(y_control)
mean_non_compliers = np.mean(y_non_compliers)
mean_compliers = np.mean(y_compliers)

# Create combined treatment group
y_treated = np.concatenate([y_non_compliers, y_compliers])
mean_treated_comp = np.mean(y_treated)

print("\n" + "=" * 60)
print("Section 5.4.5: Compliance Effect")
print("=" * 60)
print(f"Proportion of compliers: {prop_compliers * 100}%")
print(f"Control mean: R${mean_control_comp:.2f}")
print(f"Non-compliers mean: R${mean_non_compliers:.2f}")
print(f"Compliers mean: R${mean_compliers:.2f}")
print(f"Combined treatment mean: R${mean_treated_comp:.2f}")
print(f"\nLATE (effect on compliers): R${mean_compliers - mean_control_comp:.2f}")
print(f"ITT (diluted effect): R${mean_treated_comp - mean_control_comp:.2f}")
print(f"Dilution: {(mean_treated_comp - mean_control_comp) / (mean_compliers - mean_control_comp) * 100:.1f}% of true effect retained")


#########################################
# Section 5.5: Peeking Simulation - P-value Trajectories
# Figure: pvalue-trajectories.png and peeking-false-positive.png
#########################################
np.random.seed(123)

# Parameters for the simulation
n_experiments = 100
max_sample_size = 500
check_interval = 20
significance_level = 0.05

# Storage for results
pvalue_trajectories = []

# Run simulations
for exp in range(n_experiments):
    control_full = np.random.normal(100, 20, max_sample_size)
    treatment_full = np.random.normal(100, 20, max_sample_size)
    check_points = list(range(check_interval, max_sample_size + 1, check_interval))
    pvalues = []
    
    for n in check_points:
        _, p_value = ttest_ind(control_full[:n], treatment_full[:n])
        pvalues.append(p_value)
    
    pvalue_trajectories.append({
        'check_points': check_points,
        'pvalues': pvalues
    })

# Identify which experiments ever crossed the threshold
crossing_exps = []
for i, traj in enumerate(pvalue_trajectories):
    if any(p < 0.05 for p in traj['pvalues']):
        crossing_exps.append(i)

non_crossing_exps = [i for i in range(n_experiments) if i not in crossing_exps]

# Calculate cumulative false positive rate
check_points = list(range(check_interval, max_sample_size + 1, check_interval))
cumulative_fp = []

for i in range(len(check_points)):
    fp_count = sum(1 for traj in pvalue_trajectories 
                   if any(p < significance_level for p in traj['pvalues'][:i+1]))
    cumulative_fp.append(fp_count / n_experiments)

final_fp_rate = cumulative_fp[-1]

print("\n" + "=" * 60)
print("Section 5.5: Peeking Simulation")
print("=" * 60)
print(f"Number of A/A experiments: {n_experiments}")
print(f"Max sample size: {max_sample_size}")
print(f"Check interval: {check_interval}")
print(f"\nExperiments that crossed threshold: {len(crossing_exps)} of {n_experiments}")
print(f"Experiments that never crossed: {len(non_crossing_exps)}")
print(f"\nFinal cumulative false positive rate: {final_fp_rate:.1%}")
print(f"Inflation vs nominal 5%: {final_fp_rate / 0.05:.1f}x")


#########################################
# Section 5.5: Ignoring multiple testing
# Figure: multiple-comparisons.png
#########################################
# Set up the range for the number of metrics (hypotheses tested)
n_metrics = np.arange(1, 51)
alpha = 0.05

# Calculate the probability of at least one false positive
prob_fp = 1 - (1 - alpha) ** n_metrics

# Key points
key_points = [10, 20, 50]
key_probs = [1 - (1 - alpha) ** n for n in key_points]

print("\n" + "=" * 60)
print("Section 5.5: Multiple Comparisons")
print("=" * 60)
print("Probability of at least one false positive (α=0.05):")
for n, prob in zip(key_points, key_probs):
    print(f"  {n} metrics -> {prob:.1%} chance of false positive")


print("\n" + "=" * 60)
print("Script completed successfully!")
print("=" * 60)
