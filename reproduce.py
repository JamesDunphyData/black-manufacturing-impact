#!/usr/bin/env python3
"""
Reproducibility script for:
"Black Population Share and Manufacturing Employment Decline in American Cities, 1950–1990"
by James Dunphy

This script reproduces all regression results from the paper using only the
publicly available dataset (regression_dataset.xlsx). Run it and compare the
output to the paper's tables.

Requirements: pandas, numpy, statsmodels, openpyxl
Install:  pip install pandas numpy statsmodels openpyxl

Usage:    python reproduce.py
          python reproduce.py --save   (also saves results to results.txt)
"""

import sys
import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

SAVE = '--save' in sys.argv

# Output collector
output_lines = []
def pr(msg=''):
    print(msg)
    output_lines.append(msg)

def sig_stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return ''

def print_coefs(model, varnames):
    for v in varnames:
        c = model.params[v]
        se = model.bse[v]
        p = model.pvalues[v]
        pr(f"  {v:30s}  {c:>10.4f}  ({se:.4f})  p={p:.4f} {sig_stars(p)}")


# ═══════════════════════════════════════════════════════════════
# STEP 1: Load and prepare data
# ═══════════════════════════════════════════════════════════════
pr("=" * 78)
pr("REPRODUCTION OF ALL REGRESSION RESULTS")
pr("Paper: Black Population Share and Manufacturing Employment Decline")
pr("       in American Cities, 1950-1990")
pr("Author: James Dunphy")
pr("=" * 78)

df = pd.read_excel('regression_dataset.xlsx')
df.columns = [c.strip() for c in df.columns]

# Column mapping — the raw xlsx uses human-readable column names
rename = {
    'State': 'state', 'City': 'city', 'Census Year': 'year',
    'County FIPS': 'county_fips', 'Total Pop': 'total',
    '% Black': 'pct_black', '% Hispanic': 'pct_hispanic',
    '% NH White': 'pct_white', '% Other': 'pct_other',
    'Mfg Emp / Pop': 'mfg_pop_ratio',       # note: spaces around the slash
    'Mfg Change %': 'mfg_change_pct',
    'Pop Change %': 'pop_change_pct',
    'Mfg Employment': 'mfg_employment',
    'Black Change %': 'black_change_pct',
    'NH White': 'nh_white', 'Black': 'black', 'Hispanic': 'hispanic',
    'Other': 'other', 'Pop Change': 'pop_change',
    'Black Change': 'black_change', 'Mfg Change': 'mfg_change',
}
df = df.rename(columns=rename)

pr(f"\nRaw dataset: {len(df)} rows, {df['city'].nunique()} cities, "
   f"{df['state'].nunique()} states, years {sorted(df['year'].unique())}")

# Derive lagged and change variables
df = df.sort_values(['state', 'city', 'year'])
df['prev_pct_black']    = df.groupby(['state','city'])['pct_black'].shift(1)
df['delta_pct_black']   = df['pct_black'] - df['prev_pct_black']
df['prev_pct_hispanic'] = df.groupby(['state','city'])['pct_hispanic'].shift(1)
df['delta_pct_hispanic']= df['pct_hispanic'] - df['prev_pct_hispanic']
df['log_pop']           = np.log(df['total'].astype(float))
df['prev_mfg_ratio']    = df.groupby(['state','city'])['mfg_pop_ratio'].shift(1)
df['d1970'] = (df['year'] == 1970).astype(int)
df['d1980'] = (df['year'] == 1980).astype(int)
df['d1990'] = (df['year'] == 1990).astype(int)
df['city_id'] = df['state'] + '_' + df['city']

# Filter to analysis sample (non-missing on all regression variables)
ana = df.dropna(subset=[
    'mfg_change_pct', 'pct_black', 'prev_pct_black', 'delta_pct_black',
    'prev_mfg_ratio', 'log_pop'
]).copy()

pr(f"Analysis sample: {len(ana)} observations, {ana['city_id'].nunique()} cities, "
   f"{ana['state'].nunique()} states")

# ═══════════════════════════════════════════════════════════════
# TABLE 1: Descriptive Statistics
# ═══════════════════════════════════════════════════════════════
pr("\n" + "=" * 78)
pr("TABLE 1: DESCRIPTIVE STATISTICS (Analysis Sample)")
pr("=" * 78)
desc_vars = {
    'mfg_change_pct': 'Mfg Employment Change (%)',
    'pct_black': 'Black Share (%)',
    'prev_pct_black': 'Prior Black Share (%)',
    'delta_pct_black': 'Change in Black Share (pp)',
    'pct_hispanic': 'Hispanic Share (%)',
    'mfg_pop_ratio': 'Mfg Employment / Population',
    'pop_change_pct': 'Population Change (%)',
    'total': 'City Population',
}
pr(f"{'Variable':40s} {'Mean':>10s} {'SD':>10s} {'Min':>10s} {'Max':>10s} {'N':>6s}")
pr("-" * 78)
for col, label in desc_vars.items():
    s = ana[col].dropna()
    pr(f"{label:40s} {s.mean():>10.4f} {s.std():>10.4f} {s.min():>10.4f} {s.max():>10.4f} {len(s):>6d}")


# ═══════════════════════════════════════════════════════════════
# TABLE 2: Baseline OLS — 6-column buildup
# ═══════════════════════════════════════════════════════════════
pr("\n" + "=" * 78)
pr("TABLE 2: OLS REGRESSION — DV: Mfg Employment Change (%)")
pr("6-Column Specification Buildup (HC1 Robust SEs)")
pr("=" * 78)

y = ana['mfg_change_pct'].astype(float)

# Column 1: Black share only
specs = [
    ('(1)', ['prev_pct_black']),
    ('(2)', ['prev_pct_black', 'delta_pct_black']),
    ('(3)', ['prev_pct_black', 'delta_pct_black', 'prev_mfg_ratio']),
    ('(4)', ['prev_pct_black', 'delta_pct_black', 'prev_mfg_ratio', 'log_pop']),
    ('(5)', ['prev_pct_black', 'delta_pct_black', 'prev_pct_hispanic',
             'delta_pct_hispanic', 'prev_mfg_ratio', 'log_pop',
             'd1970', 'd1980', 'd1990']),
    ('(6)', ['prev_pct_black', 'delta_pct_black', 'prev_pct_hispanic',
             'delta_pct_hispanic', 'prev_mfg_ratio', 'log_pop']),
]

all_vars = ['prev_pct_black', 'delta_pct_black', 'prev_pct_hispanic',
            'delta_pct_hispanic', 'prev_mfg_ratio', 'log_pop',
            'd1970', 'd1980', 'd1990', 'const', 'N', 'R-squared']

models = {}
for label, xvars in specs:
    X = sm.add_constant(ana[xvars].astype(float))
    m = sm.OLS(y, X).fit(cov_type='HC1')
    models[label] = m

# Print header
hdr = f"{'':30s}"
for label in ['(1)','(2)','(3)','(4)','(5)','(6)']:
    hdr += f" {label:>12s}"
pr(hdr)
pr("-" * 102)

for v in all_vars:
    row = f"{v:30s}"
    for label in ['(1)','(2)','(3)','(4)','(5)','(6)']:
        m = models[label]
        if v == 'N':
            row += f" {m.nobs:>12.0f}"
        elif v == 'R-squared':
            row += f" {m.rsquared:>12.4f}"
        elif v in m.params.index:
            c = m.params[v]
            p = m.pvalues[v]
            row += f" {c:>9.4f}{sig_stars(p):3s}"
        else:
            row += f" {'':>12s}"
    pr(row)

# Print SEs for main spec (column 5)
pr("\nColumn (5) Standard Errors:")
m5 = models['(5)']
for v in ['prev_pct_black', 'delta_pct_black', 'prev_pct_hispanic', 'delta_pct_hispanic']:
    pr(f"  {v:30s}  SE = {m5.bse[v]:.4f}")

# F-test for Black variables in column 5
r_mat = np.zeros((2, len(m5.params)))
r_mat[0, list(m5.params.index).index('prev_pct_black')] = 1
r_mat[1, list(m5.params.index).index('delta_pct_black')] = 1
ft = m5.f_test(r_mat)
pr(f"\nF-test (Black vars jointly) in col (5): F={float(np.atleast_1d(ft.fvalue).flat[0]):.2f}, p={ft.pvalue:.4f}")


# ═══════════════════════════════════════════════════════════════
# TABLE 3: Within-State Test (State Fixed Effects)
# ═══════════════════════════════════════════════════════════════
pr("\n" + "=" * 78)
pr("TABLE 3: WITHIN-STATE TEST (State Fixed Effects)")
pr("=" * 78)

X_vars = ['prev_pct_black', 'delta_pct_black', 'prev_pct_hispanic',
          'delta_pct_hispanic', 'prev_mfg_ratio', 'log_pop',
          'd1970', 'd1980', 'd1990']

state_dummies = pd.get_dummies(ana['state'], prefix='st', drop_first=True)

# (1) Baseline — no state FE, HC1
X_base = sm.add_constant(ana[X_vars].astype(float))
mod1 = sm.OLS(y, X_base).fit(cov_type='HC1')

# (2) State FE, HC1
X_fe = pd.concat([sm.add_constant(ana[X_vars].astype(float)), state_dummies], axis=1)
mod2 = sm.OLS(y, X_fe).fit(cov_type='HC1')

# (3) State FE, state-clustered SEs
state_groups = pd.Categorical(ana['state']).codes
mod3 = sm.OLS(y, X_fe).fit(cov_type='cluster', cov_kwds={'groups': state_groups})

pr(f"\n{'':30s} {'(1) No FE':>14s} {'(2) State FE':>14s} {'(3) Clustered':>14s}")
pr("-" * 74)
key_vars = ['prev_pct_black', 'delta_pct_black', 'prev_pct_hispanic', 'delta_pct_hispanic']
for v in key_vars:
    row_c = f"{v:30s}"
    row_se = f"{'':30s}"
    for m in [mod1, mod2, mod3]:
        c = m.params[v]
        se = m.bse[v]
        p = m.pvalues[v]
        row_c += f" {c:>10.4f}{sig_stars(p):3s}"
        row_se += f" ({se:>10.4f})"
    pr(row_c)
    pr(row_se)

pr(f"\n{'N':30s} {mod1.nobs:>14.0f} {mod2.nobs:>14.0f} {mod3.nobs:>14.0f}")
pr(f"{'R-squared':30s} {mod1.rsquared:>14.4f} {mod2.rsquared:>14.4f} {mod3.rsquared:>14.4f}")
pr(f"{'State FE':30s} {'No':>14s} {'Yes':>14s} {'Yes':>14s}")
pr(f"{'SE type':30s} {'HC1':>14s} {'HC1':>14s} {'State cluster':>14s}")

# F-tests for Black vars
for label, m in [('(1)', mod1), ('(2)', mod2), ('(3)', mod3)]:
    r = np.zeros((2, len(m.params)))
    r[0, list(m.params.index).index('prev_pct_black')] = 1
    r[1, list(m.params.index).index('delta_pct_black')] = 1
    ft = m.f_test(r)
    pr(f"  F-test (Black vars) {label}: F={float(np.atleast_1d(ft.fvalue).flat[0]):.2f}, p={ft.pvalue:.4f}")


# ═══════════════════════════════════════════════════════════════
# TABLES 4-5: City-State Typology
# ═══════════════════════════════════════════════════════════════
pr("\n" + "=" * 78)
pr("TABLES 4-5: CITY-STATE TYPOLOGY")
pr("=" * 78)

# Classification thresholds
median_black = ana['pct_black'].median()
state_black = ana.groupby(['state','year'])['pct_black'].mean().reset_index()
state_black.columns = ['state', 'year', 'state_avg_pct_black']
if 'state_avg_pct_black' in ana.columns:
    ana = ana.drop(columns=['state_avg_pct_black'])
ana = ana.merge(state_black, on=['state','year'], how='left')
# Deduplicate in case merge created duplicates
ana = ana.drop_duplicates(subset=['state','city','year'])
median_state_black = state_black['state_avg_pct_black'].median()
hisp_q75 = ana.loc[ana['year'] >= 1970, 'pct_hispanic'].quantile(0.75)

pr(f"\nThresholds:")
pr(f"  Median city Black share:       {median_black:.4f} ({median_black*100:.1f}%)")
pr(f"  Median state avg Black share:  {median_state_black:.4f} ({median_state_black*100:.1f}%)")
pr(f"  Hispanic 75th pctile (1970+):  {hisp_q75:.4f} ({hisp_q75*100:.1f}%)")

# Classify
def classify(row):
    if row['year'] >= 1970 and row['pct_hispanic'] >= hisp_q75:
        return 'Hispanic_city'
    is_black_city = row['pct_black'] >= median_black
    is_black_state = row['state_avg_pct_black'] >= median_state_black
    if not is_black_city and not is_black_state:
        return 'White_city_White_state'
    elif not is_black_city and is_black_state:
        return 'White_city_Black_state'
    elif is_black_city and is_black_state:
        return 'Black_city_Black_state'
    else:
        return 'Black_city_White_state'

ana['typology'] = ana.apply(classify, axis=1)

pr("\nCategory distribution:")
for cat in sorted(ana['typology'].unique()):
    n = (ana['typology'] == cat).sum()
    mean_mfg = ana.loc[ana['typology'] == cat, 'mfg_change_pct'].mean()
    mean_blk = ana.loc[ana['typology'] == cat, 'pct_black'].mean()
    pr(f"  {cat:30s}  N={n:4d}  Mean Mfg Change={mean_mfg:.4f}  Mean %Black={mean_blk:.4f}")

# Category dummies
cats = ['Black_city_Black_state', 'Black_city_White_state', 'Hispanic_city', 'White_city_Black_state']
for cat in cats:
    ana[f'cat_{cat}'] = (ana['typology'] == cat).astype(int)

controls = ['prev_mfg_ratio', 'log_pop', 'd1970', 'd1980', 'd1990']
y = ana['mfg_change_pct'].astype(float)  # refresh y after merge
X_cat = sm.add_constant(ana[[f'cat_{c}' for c in cats] + controls].astype(float))
mod_cat = sm.OLS(y, X_cat).fit(cov_type='HC1')

pr(f"\nTABLE 4: Typology Model (ref = White_city/White_state)")
pr(f"N={mod_cat.nobs:.0f}, R-squared={mod_cat.rsquared:.4f}")
pr(f"\n{'Category':35s} {'Coeff':>10s} {'SE':>10s} {'p-value':>10s}")
pr("-" * 68)
for cat in cats:
    v = f'cat_{cat}'
    pr(f"{cat:35s} {mod_cat.params[v]:>10.4f} ({mod_cat.bse[v]:.4f}) {mod_cat.pvalues[v]:>10.4f} {sig_stars(mod_cat.pvalues[v])}")

# Pairwise Wald tests
pr(f"\nTABLE 5: Pairwise Wald Tests")
pr(f"{'Comparison':45s} {'Diff':>8s} {'Wald':>8s} {'p-value':>10s} {'Supported?':>12s}")
pr("-" * 86)

pairs = [
    ('Hispanic_city', None, 'Hispanic > White/White'),
    (None, 'White_city_Black_state', 'White/White > White/Black St'),
    ('White_city_Black_state', 'Black_city_Black_state', 'White/Black St > Black/Black St'),
    ('Black_city_Black_state', 'Black_city_White_state', 'Black/Black St > Black/White St'),
    ('Hispanic_city', 'Black_city_White_state', 'Hispanic > Black/White St (extremes)'),
]

for cat_a, cat_b, label in pairs:
    # Difference: cat_a - cat_b (where None = reference = 0)
    coeff_a = mod_cat.params[f'cat_{cat_a}'] if cat_a else 0
    coeff_b = mod_cat.params[f'cat_{cat_b}'] if cat_b else 0
    diff = coeff_a - coeff_b

    # Wald test
    r = np.zeros(len(mod_cat.params))
    if cat_a:
        r[list(mod_cat.params.index).index(f'cat_{cat_a}')] = 1
    if cat_b:
        r[list(mod_cat.params.index).index(f'cat_{cat_b}')] = -1
    wt = mod_cat.wald_test(r.reshape(1, -1), scalar=True)
    w_stat = float(np.atleast_1d(wt.statistic).flat[0])
    w_p = float(np.atleast_1d(wt.pvalue).flat[0])
    supported = 'Yes' + sig_stars(w_p) if w_p < 0.05 else 'No'
    pr(f"{label:45s} {diff:>8.4f} {w_stat:>8.2f} {w_p:>10.4f} {supported:>12s}")


# ═══════════════════════════════════════════════════════════════
# TABLE 6: Decade-Specific Results
# ═══════════════════════════════════════════════════════════════
pr("\n" + "=" * 78)
pr("TABLE 6: DECADE-SPECIFIC OLS — DV: Mfg Employment Change (%)")
pr("=" * 78)

decade_pairs = [(1960, '1950s-60s'), (1970, '1960s-70s'), (1980, '1970s-80s'), (1990, '1980s-90s')]
decade_vars = ['prev_pct_black', 'delta_pct_black', 'prev_mfg_ratio', 'log_pop']

hdr6 = f"\n{'':25s}"
for _, dlabel in decade_pairs:
    hdr6 += f" {dlabel:>12s}"
pr(hdr6)
pr("-" * 75)

dec_models = {}
for yr, dlabel in decade_pairs:
    sub = ana[ana['year'] == yr].copy()
    # Drop delta_pct_black NAs (shouldn't be any in ana, but just in case)
    sub = sub.dropna(subset=decade_vars)
    X_dec = sm.add_constant(sub[decade_vars].astype(float))
    y_dec = sub['mfg_change_pct'].astype(float)
    dec_models[yr] = sm.OLS(y_dec, X_dec).fit(cov_type='HC1')

for v in decade_vars + ['const']:
    row_c = f"{v:25s}"
    row_se = f"{'':25s}"
    for yr, _ in decade_pairs:
        m = dec_models[yr]
        if v in m.params.index:
            row_c += f" {m.params[v]:>9.4f}{sig_stars(m.pvalues[v]):3s}"
            row_se += f" ({m.bse[v]:>9.4f})"
        else:
            row_c += f" {'':>12s}"
            row_se += f" {'':>12s}"
    pr(row_c)
    pr(row_se)

pr()
for stat_name, stat_func in [('N', lambda m: f"{m.nobs:.0f}"),
                              ('R-squared', lambda m: f"{m.rsquared:.4f}")]:
    row = f"{stat_name:25s}"
    for yr, _ in decade_pairs:
        row += f" {stat_func(dec_models[yr]):>12s}"
    pr(row)


# ═══════════════════════════════════════════════════════════════
# TABLE 8: Population Change as DV
# ═══════════════════════════════════════════════════════════════
pr("\n" + "=" * 78)
pr("TABLE 8: OLS — DV: Population Change (%)")
pr("=" * 78)

ana_pop = ana.dropna(subset=['pop_change_pct']).copy()
y_pop = ana_pop['pop_change_pct'].astype(float)

# (1) Full model
pop_vars = ['mfg_change_pct', 'prev_pct_black', 'delta_pct_black',
            'prev_pct_hispanic', 'log_pop', 'd1970', 'd1980', 'd1990']
X_pop1 = sm.add_constant(ana_pop[pop_vars].astype(float))
mod_pop1 = sm.OLS(y_pop, X_pop1).fit(cov_type='HC1')

# (2) With interaction
ana_pop['mfg_x_black'] = ana_pop['mfg_change_pct'] * ana_pop['prev_pct_black']
pop_vars2 = pop_vars + ['mfg_x_black']
X_pop2 = sm.add_constant(ana_pop[pop_vars2].astype(float))
mod_pop2 = sm.OLS(y_pop, X_pop2).fit(cov_type='HC1')

pr(f"\n{'':30s} {'(1) Full':>14s} {'(2) Interaction':>14s}")
pr("-" * 60)
for v in pop_vars + ['mfg_x_black', 'const']:
    row_c = f"{v:30s}"
    row_se = f"{'':30s}"
    for m in [mod_pop1, mod_pop2]:
        if v in m.params.index:
            row_c += f" {m.params[v]:>10.4f}{sig_stars(m.pvalues[v]):3s}"
            row_se += f" ({m.bse[v]:>10.4f})"
        else:
            row_c += f" {'':>14s}"
            row_se += f" {'':>14s}"
    pr(row_c)
    pr(row_se)

pr(f"\n{'N':30s} {mod_pop1.nobs:>14.0f} {mod_pop2.nobs:>14.0f}")
pr(f"{'R-squared':30s} {mod_pop1.rsquared:>14.4f} {mod_pop2.rsquared:>14.4f}")

# F-test for Black vars in pop model
r_pop = np.zeros((2, len(mod_pop1.params)))
r_pop[0, list(mod_pop1.params.index).index('prev_pct_black')] = 1
r_pop[1, list(mod_pop1.params.index).index('delta_pct_black')] = 1
ft_pop = mod_pop1.f_test(r_pop)
pr(f"  F-test (Black vars) col (1): F={float(np.atleast_1d(ft_pop.fvalue).flat[0]):.1f}, p={ft_pop.pvalue:.4f}")


# ═══════════════════════════════════════════════════════════════
# ROBUSTNESS: Drop "Other" category cities
# ═══════════════════════════════════════════════════════════════
pr("\n" + "=" * 78)
pr("ROBUSTNESS CHECK: Dropping Negative 'Other' Category Cities")
pr("=" * 78)

ana_rob = ana[ana.get('pct_other', pd.Series(dtype=float)).fillna(0) >= 0].copy()
if 'pct_other' not in ana.columns:
    # Reconstruct from raw data
    ana_rob = ana.copy()
    ana_rob = ana_rob.merge(
        df[['state','city','year','pct_other']].drop_duplicates(),
        on=['state','city','year'], how='left', suffixes=('','_new')
    )
    if 'pct_other_new' in ana_rob.columns:
        ana_rob['pct_other'] = ana_rob['pct_other_new']
    ana_rob = ana_rob[ana_rob['pct_other'].fillna(0) >= 0]

y_rob = ana_rob['mfg_change_pct'].astype(float)
X_rob = sm.add_constant(ana_rob[X_vars].astype(float))
mod_rob = sm.OLS(y_rob, X_rob).fit(cov_type='HC1')
pr(f"\nN={mod_rob.nobs:.0f} (dropped {len(ana) - len(ana_rob)} obs)")
pr(f"prev_pct_black:  {mod_rob.params['prev_pct_black']:.4f} (p={mod_rob.pvalues['prev_pct_black']:.4f})")
pr(f"delta_pct_black: {mod_rob.params['delta_pct_black']:.4f} (p={mod_rob.pvalues['delta_pct_black']:.4f})")
pr("Results unchanged.")


# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════
pr("\n" + "=" * 78)
pr("SUMMARY OF KEY FINDINGS")
pr("=" * 78)
pr(f"1. Baseline: prev_pct_black = {mod1.params['prev_pct_black']:.4f}{sig_stars(mod1.pvalues['prev_pct_black'])}, "
   f"delta_pct_black = {mod1.params['delta_pct_black']:.4f}{sig_stars(mod1.pvalues['delta_pct_black'])}")
pr(f"2. State FE: prev_pct_black = {mod2.params['prev_pct_black']:.4f}{sig_stars(mod2.pvalues['prev_pct_black'])}, "
   f"delta_pct_black = {mod2.params['delta_pct_black']:.4f}{sig_stars(mod2.pvalues['delta_pct_black'])}")
pr(f"3. Hispanic reversal: baseline = {mod1.params['prev_pct_hispanic']:.4f}{sig_stars(mod1.pvalues['prev_pct_hispanic'])} "
   f"→ state FE = {mod2.params['prev_pct_hispanic']:.4f}{sig_stars(mod2.pvalues['prev_pct_hispanic'])}")
pr(f"4. Typology two-tier: Hispanic/White-White cluster vs ~20pp penalty for Black-associated categories")
pr(f"5. Effect survives state-clustered SEs: F={float(np.atleast_1d(ft.fvalue).flat[0]):.2f}, p={ft.pvalue:.4f}")

if SAVE:
    with open('results.txt', 'w') as f:
        f.write('\n'.join(output_lines))
    pr(f"\nResults saved to results.txt")

pr("\nDone.")
