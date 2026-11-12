Black Population Share and Manufacturing Employment Decline in American Cities, 1950–1990

Author: James Dunphy

Overview

This repository contains the data and code to independently reproduce all regression results in the paper. The analysis tests whether Black population share (both level and decade-over-decade change) predicts manufacturing employment decline in 307 US cities across four decades.

Key Findings
Black population share is a strong, robust negative predictor of manufacturing employment change (baseline: prev_pct_black = -0.43, p < 0.001; delta_pct_black = -2.23, p < 0.001).
The relationship survives state fixed effects — it is within-state, not an artifact of regional manufacturing trends.
Hispanic share initially appeared positively associated with manufacturing growth (+0.32, p = 0.019), but this reverses with state fixed effects (-0.56, p < 0.001), indicating the baseline was driven by favorable state-level manufacturing environments in Texas, California, and Florida.
A city-state typology reveals a two-tier structure: White/White and Hispanic cities cluster together (~0% penalty), while all Black-associated categories cluster at ~20 percentage points worse manufacturing outcomes.
British industrial cities lost manufacturing at comparable rates (55–64%) while remaining 95%+ white, establishing that deindustrialization alone is sufficient to destroy a city's industrial base.
Files
File	Description
regression_dataset.xlsx	Panel dataset: 1,470 obs × 21 columns. 307 cities, 51 states, 5 census years (1950–1990). Source: US Census + County Business Patterns.
reproduce.py	Self-contained Python script that reproduces all regression tables from the paper.
results.txt	Pre-computed output from reproduce.py for reference (run python reproduce.py --save to regenerate).
How to Reproduce
Requirements
Python 3.8+
pandas, numpy, statsmodels, openpyxl
Setup
bash
pip install pandas numpy statsmodels openpyxl
Run
bash
python reproduce.py

Or, to save output to a file:

bash
python reproduce.py --save
What It Reproduces

The script reproduces the following tables from the paper:

Paper Table	Content
Table 1	Descriptive statistics
Table 2	6-column OLS buildup (baseline)
Table 3	State fixed effects (3 specifications)
Table 4	City-state typology model
Table 5	Pairwise Wald tests between categories
Table 6	Decade-specific regressions
Table 8	Population change as dependent variable
—	Robustness check (drop negative "Other" cities)

Tables 7 (Selected Cities), 9 (British Comparison), and 10 (International Manufacturing) are descriptive/comparative and use external data not included in this dataset.

Data Dictionary
regression_dataset.xlsx
Column	Description
State	2-letter state abbreviation
City	City name
Census Year	Decennial census year (1950, 1960, 1970, 1980, 1990)
County FIPS	County FIPS code (manufacturing data is county-level)
Total Pop	City total population
NH White	Non-Hispanic White population
Black	Black population
Hispanic	Hispanic population
Other	Other race category
% NH White	Non-Hispanic White share (0–1)
% Black	Black share (0–1)
% Hispanic	Hispanic share (0–1)
% Other	Other share (0–1)
Mfg Employment	County-level manufacturing employment (County Business Patterns)
Mfg Emp / Pop	Manufacturing employment divided by city population
Pop Change	Absolute population change from prior decade
Pop Change %	Population percentage change
Black Change	Absolute change in Black population
Black Change %	Black population percentage change
Mfg Change	Absolute change in manufacturing employment
Mfg Change %	Manufacturing employment percentage change
Derived Variables (computed by reproduce.py)
Variable	Derivation
prev_pct_black	Prior decade's Black share (lagged)
delta_pct_black	pct_black − prev_pct_black
prev_pct_hispanic	Prior decade's Hispanic share (lagged)
delta_pct_hispanic	pct_hispanic − prev_pct_hispanic
log_pop	ln(Total Pop)
prev_mfg_ratio	Prior decade's Mfg Emp / Pop (lagged)
d1970, d1980, d1990	Decade dummies
state_avg_pct_black	Mean Black share across cities in same state-year
typology	5-category classification (see below)
Typology Classification

Cities are classified into five mutually exclusive categories using data-driven thresholds:

Hispanic City: Hispanic share ≥ 75th percentile (1970+ observations only)
White City / White State: Below-median Black share in below-median-Black state
White City / Black State: Below-median Black share in above-median-Black state
Black City / Black State: Above-median Black share in above-median-Black state
Black City / White State: Above-median Black share in below-median-Black state

Thresholds are computed from the analysis sample:

Median city Black share: ~8%
Median state average Black share: ~9%
Hispanic 75th percentile (1970+): ~10%
Methodology Notes
Standard errors: HC1 (heteroskedasticity-robust) throughout. State-clustered SEs reported as robustness.
State fixed effects: 50 state dummies (drop-first encoding). DC is the single-city "state."
No region dummies: Excluded by design — the test is whether race predicts manufacturing decline after controlling for state-level trends.
Panel structure: 1,137 city-decade observations after dropping rows with missing lagged variables (first observation per city drops out).
Manufacturing data: County Business Patterns, matched to cities. Pre-1975 data uses nearest available CBP year (e.g., 1962 CBP for 1960 census).
"Other" race category: The Census Hispanic-race overlap produces negative "Other" values for some city-decades. Robustness check drops these 24 cities; results are unchanged.
Data Sources
US Decennial Census (1950–1990): Population and racial composition
County Business Patterns (1958–1991): Manufacturing employment
UK Office for National Statistics / NOMIS: British manufacturing employment (Section 7)
World Bank / UNIDO: International manufacturing value added (Section 8)
License

Data is derived from publicly available US government sources. Code is provided under MIT license for reproducibility purposes.

Citation

Dunphy, J. (2026). Black Population Share and Manufacturing Employment Decline in American Cities, 1950–1990. Working Paper.
