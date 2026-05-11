"""
Black Population Growth and Manufacturing Employment Decline in U.S. Cities, 1950-1990
Replication Script — Quaestio Verum Institute, 2026
AI-assisted (Claude, Anthropic)

Reproduces key regression results from the paper.
Requires: regression_dataset.csv in ../data/
Dependencies: pandas, numpy, statsmodels, scipy, matplotlib
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv('../data/regression_dataset.csv')
print(f"Loaded {len(df)} observations from {df['city'].nunique()} cities")

# Decade dummies (omit 1960s)
df['d1970s'] = (df['census_year'] == 1970).astype(int)
df['d1980s'] = (df['census_year'] == 1980).astype(int)
df['d1990s'] = (df['census_year'] == 1990).astype(int)

# Regression sample: non-missing DV and key predictors
mask = df['mfg_change_pct'].notna() & df['prev_pct_black'].notna() & df['delta_pct_black'].notna()
reg = df[mask].copy()
print(f"Regression sample: {len(reg)} observations")

# ============================================================
# TABLE 2: MAIN RESULTS
# ============================================================
print("\n" + "="*60)
print("TABLE 2: OLS — DV: Manufacturing Employment Change (%)")
print("="*60)

def run_ols(y, X_vars, data, cov_type='HC1', label=''):
    X = sm.add_constant(data[X_vars])
    model = sm.OLS(data[y], X).fit(cov_type=cov_type)
    print(f"\n{label}")
    print(f"N={model.nobs:.0f}, R²={model.rsquared:.4f}, Adj.R²={model.rsquared_adj:.4f}")
    for var in X_vars:
        coef = model.params[var]
        pval = model.pvalues[var]
        stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.10 else ''
        print(f"  {var:<30} {coef:>8.4f}{stars}  (p={pval:.4f})")
    return model

m1 = run_ols('mfg_change_pct', ['prev_pct_black'], reg, label='Column 1: Bivariate')

m2 = run_ols('mfg_change_pct', ['prev_pct_black', 'delta_pct_black'], reg,
             label='Column 2: + Change in Black share')

m3 = run_ols('mfg_change_pct',
             ['prev_pct_black', 'delta_pct_black', 'prev_mfg_ratio', 'log_pop',
              'd1970s', 'd1980s', 'd1990s'], reg,
             label='Column 3: + Controls')

full_vars = ['prev_pct_black', 'delta_pct_black', 'prev_pct_hispanic', 'delta_pct_hispanic',
             'prev_mfg_ratio', 'log_pop', 'd1970s', 'd1980s', 'd1990s']
m4 = run_ols('mfg_change_pct', full_vars, reg, label='Column 4: Full specification (HC1)')
m5 = run_ols('mfg_change_pct', full_vars, reg, cov_type='cluster',
             label='Column 5: County-clustered SEs')

# Joint F-test for Black variables
print(f"\nJoint F-test (Black share + Change in Black share):")
hypotheses = '(prev_pct_black = 0), (delta_pct_black = 0)'
ftest = m4.f_test(hypotheses)
print(f"  F = {ftest.fvalue[0][0]:.2f}, p = {ftest.pvalue:.4f}")

# ============================================================
# TABLE 3: DECADE-SPECIFIC RESULTS
# ============================================================
print("\n" + "="*60)
print("TABLE 3: Decade-Specific OLS")
print("="*60)

decade_vars = ['prev_pct_black', 'delta_pct_black', 'prev_mfg_ratio', 'log_pop']
decade_map = {1960: '1950s-60s', 1970: '1960s-70s', 1980: '1970s-80s', 1990: '1980s-90s'}

for year, label in decade_map.items():
    sub = reg[reg['census_year'] == year]
    if len(sub) > 10:
        run_ols('mfg_change_pct', decade_vars, sub, label=f'{label} (N={len(sub)})')

# ============================================================
# TABLE 5: POPULATION CHANGE
# ============================================================
print("\n" + "="*60)
print("TABLE 5: OLS — DV: Population Change (%)")
print("="*60)

pop_vars = ['mfg_change_pct', 'prev_pct_black', 'delta_pct_black',
            'prev_pct_hispanic', 'log_pop', 'd1970s', 'd1980s', 'd1990s']
pop_mask = reg['pop_change_pct'].notna()
pop_reg = reg[pop_mask]

run_ols('pop_change_pct', pop_vars, pop_reg, label='Column 1: Full')

# Interaction model
pop_reg = pop_reg.copy()
pop_reg['mfg_x_black'] = pop_reg['mfg_change_pct'] * pop_reg['prev_pct_black']
run_ols('pop_change_pct', pop_vars + ['mfg_x_black'], pop_reg, label='Column 2: With interaction')

# ============================================================
# BRITISH COMPARISON
# ============================================================
print("\n" + "="*60)
print("BRITISH COMPARISON: Manufacturing Loss 1971-1991")
print("="*60)

uk = pd.read_csv('../data/british_cities.csv')
print(f"\n{'City':<20} {'70s Loss':>10} {'80s Loss':>10} {'Total':>10}")
print("-" * 55)
for _, row in uk.iterrows():
    print(f"{row['city']:<20} {row['pct_change_70s']:>9.1f}% {row['pct_change_80s']:>9.1f}% {row['pct_change_total']:>9.1f}%")
print("-" * 55)
print(f"{'AVERAGE':<20} {uk['pct_change_70s'].mean():>9.1f}% {uk['pct_change_80s'].mean():>9.1f}% {uk['pct_change_total'].mean():>9.1f}%")

# ============================================================
# INTERNATIONAL MVA
# ============================================================
print("\n" + "="*60)
print("INTERNATIONAL: Group Averages — MVA per Capita and % GDP")
print("="*60)

intl = pd.read_csv('../data/international_mva.csv')
group_avg = intl.groupby('population_group')[['mva_per_capita_usd', 'mva_pct_gdp']].mean().sort_values('mva_per_capita_usd', ascending=False)
print(group_avg.round(1).to_string())

# ============================================================
# FIGURE 1: Manufacturing Change vs Black Share
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel A: Black share vs mfg change
ax = axes[0, 0]
ax.scatter(reg['prev_pct_black'], reg['mfg_change_pct'], alpha=0.3, s=15, color='steelblue')
m, b = np.polyfit(reg['prev_pct_black'].dropna(), reg.loc[reg['prev_pct_black'].notna(), 'mfg_change_pct'], 1)
x_line = np.linspace(0, 0.75, 100)
ax.plot(x_line, m * x_line + b, 'r-', linewidth=2)
ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax.set_xlabel('Black Share (Start of Decade)')
ax.set_ylabel('Mfg Employment Change (%)')
ax.set_title('A. Manufacturing Change vs. Black Share')

# Panel B: Change in black share vs mfg change
ax = axes[0, 1]
ax.scatter(reg['delta_pct_black'], reg['mfg_change_pct'], alpha=0.3, s=15, color='steelblue')
m, b = np.polyfit(reg['delta_pct_black'].dropna(), reg.loc[reg['delta_pct_black'].notna(), 'mfg_change_pct'], 1)
x_line = np.linspace(-0.2, 0.45, 100)
ax.plot(x_line, m * x_line + b, 'r-', linewidth=2)
ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax.set_xlabel('Change in Black Share (pp)')
ax.set_ylabel('Mfg Employment Change (%)')
ax.set_title('B. Manufacturing Change vs. Change in Black Share')

# Panel C: Binned means
ax = axes[1, 0]
reg['black_decile'] = pd.qcut(reg['prev_pct_black'], q=10, duplicates='drop')
binned = reg.groupby('black_decile')['mfg_change_pct'].agg(['mean', 'sem'])
x_vals = range(len(binned))
ax.errorbar(x_vals, binned['mean'], yerr=1.96*binned['sem'], fmt='o-', color='steelblue', capsize=4)
ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax.set_xlabel('Black Share (Decile Mean)')
ax.set_ylabel('Mean Mfg Employment Change (%)')
ax.set_title('C. Binned Means: Mfg Change by Black Share Decile')

# Panel D: Population change residual vs black share
X_resid = sm.add_constant(reg[['mfg_change_pct']].dropna())
y_resid = reg.loc[reg['mfg_change_pct'].notna(), 'pop_change_pct'].dropna()
common_idx = X_resid.index.intersection(y_resid.index)
X_resid = X_resid.loc[common_idx]
y_resid = y_resid.loc[common_idx]
resid_model = sm.OLS(y_resid, X_resid).fit()
residuals = resid_model.resid
black_share_resid = reg.loc[common_idx, 'prev_pct_black']
ax = axes[1, 1]
ax.scatter(black_share_resid, residuals, alpha=0.3, s=15, color='orange')
m, b = np.polyfit(black_share_resid.dropna(), residuals[black_share_resid.notna()], 1)
x_line = np.linspace(0, 0.8, 100)
ax.plot(x_line, m * x_line + b, 'r-', linewidth=2)
ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax.set_xlabel('Black Share')
ax.set_ylabel('Population Change (Residual after Mfg)')
ax.set_title('D. Pop Change (net of Mfg) vs. Black Share')

plt.tight_layout()
plt.savefig('../output/figures/figure1_main.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nFigure 1 saved to ../output/figures/figure1_main.png")

print("\n" + "="*60)
print("Replication complete.")
print("="*60)
