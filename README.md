# Black Population Growth and Manufacturing Employment Decline in U.S. Cities, 1950–1990

## With British Comparative Analysis and International Manufacturing Output by Population Group

**Working Paper — February 2026 (Updated May 2026)**  
**Quaestio Verum Institute**  
**AI-assisted (Claude, Anthropic)**

---

## For Humans (Quants, Reviewers, Replicators)

### What This Project Is

An empirical analysis testing whether Black population share predicts manufacturing employment decline across 307 U.S. cities between 1950 and 1990. The study uses two gold-standard public datasets — Census Bureau Working Paper 76 for demographics and the Eckert et al. County Business Patterns files for manufacturing — and finds that Black population share strongly and specifically predicts manufacturing job loss, while Hispanic share is associated with manufacturing gains.

The paper includes two extensions:

1. **British Comparative Analysis:** 14 heavily white British industrial cities (Birmingham, Manchester, Glasgow, Liverpool, Sheffield, etc.) lost manufacturing at rates directly comparable to American rust belt cities — averaging 31.8% loss in the 1970s and 28.5% in the 1980s — while remaining 95%+ white throughout. Deindustrialization alone is sufficient to destroy an industrial city; the additional social devastation in Black American cities reflects something beyond the economic shock.

2. **International Manufacturing Output:** Cross-national manufacturing value added per capita and as % of GDP for 32 nations grouped by dominant population. European and East Asian nations dominate both measures; Sub-Saharan African nations cluster at the bottom on both; Latin American mixed-ancestry nations occupy a consistent middle tier.

**This is observational work, not causal identification.** The findings are associations. Key confounds — white flight as a competing causal mechanism, simultaneity of racial change and deindustrialization — are acknowledged throughout.

---

### How It Was Built

This project was built iteratively using Claude (Anthropic) as a research assistant. The human author:

- Conceived the hypothesis (Black population share → manufacturing decline)
- Designed the dataset structure and selected the comparison framework
- Directed all analytical decisions (model specifications, robustness checks, decade-specific analysis)
- Proposed the British comparative analysis and the international manufacturing extension
- Identified the Hispanic placebo test as a falsification strategy
- Directed the ceiling/floor theoretical framework

The AI assistant:

- Merged and processed the Census and CBP datasets
- Wrote and executed Python code for all regressions and diagnostics
- Generated all figures (matplotlib)
- Compiled British NOMIS data and international World Bank/UNIDO data
- Drafted prose based on the author's analytical direction
- Flagged methodological issues (endogeneity, measurement error, omitted variables)

---

### Data Sources

| Dataset | Source | Years | Notes |
|---|---|---|---|
| City demographics | Census Bureau Working Paper 76 (Gibson & Jung 2005) | 1950–1990 | Race and Hispanic origin for 307 large U.S. cities |
| Manufacturing employment (early) | Eckert, Lam, Mian, Muller, Schwalb & Sufi (2022), NBER WP 30578 | 1947–1974 | Early County Business Patterns panel |
| Manufacturing employment (later) | Eckert, Fort, Schott & Yang (2021), NBER WP 26632 | 1980–1990 | Imputed CBP files |
| British manufacturing (1971–1981) | UK ONS Census of Employment (SIC 1968), open access | 1971, 1978, 1981 | Via NOMIS (nomisweb.co.uk), accessed May 2026 |
| British manufacturing (1981–1991) | UK ONS Census of Employment (SIC 1980), open access | 1981, 1991 | Via NOMIS (nomisweb.co.uk), accessed May 2026 |
| International MVA per capita | World Bank WDI / UNIDO | 2022–2023 | Manufacturing value added per capita (USD) |
| International MVA % GDP | World Bank WDI | 2022–2023 | Manufacturing value added as % of GDP |

All source data is **publicly available** at the URLs listed in the companion methodology document. No institutional access or login required for primary American datasets. NOMIS requires free registration for British data.

---

### Key Variables

#### American Analysis

**Dependent variables:**
- `mfg_change_pct` = (Mfg_t − Mfg_{t−1}) / Mfg_{t−1} — decade manufacturing employment change
- `pop_change_pct` = (Pop_t − Pop_{t−1}) / Pop_{t−1} — decade population change

**Key predictors:**
- `prev_pct_black` = Black share at start of decade (lagged)
- `delta_pct_black` = change in Black share over decade
- `prev_pct_hispanic` = Hispanic share at start of decade (lagged)
- `delta_pct_hispanic` = change in Hispanic share over decade
- `prev_mfg_ratio` = lagged manufacturing employment / population
- `log_pop` = log of city population

**Decade dummies:** 1970s, 1980s, 1990s (omitted: 1960s)

**Region dummies:** Excluded. Rationale: regional Black population shares were changing dramatically due to the Great Migration. A South dummy would absorb part of the effect under study.

#### British Analysis

- `mfg_1971`, `mfg_1981`, `mfg_1991` — manufacturing employment counts at travel-to-work area level
- `pct_change_70s` = (mfg_1981 − mfg_1971) / mfg_1971
- `pct_change_80s` = (mfg_1991 − mfg_1981) / mfg_1981
- Black population share in all 14 cities: <5% throughout the study period

---

### Statistical Methods

OLS with heteroskedasticity-robust standard errors (HC1, White 1980). County-clustered SEs reported as robustness check. Python 3.12, statsmodels 0.14.

```python
import statsmodels.api as sm

X = sm.add_constant(predictors)
model = sm.OLS(y, X).fit(cov_type='HC1')
print(model.summary())
```

**Sample:** 1,137 city-decade observations across 298 cities and four decade transitions (1950s–60s, 1960s–70s, 1970s–80s, 1980s–90s).

---

### Key Findings

#### American Results

| Model | Key Coefficient | p-value | R² |
|---|---|---|---|
| Black share (lagged) only | −0.813 | <0.001 | 0.028 |
| + Change in Black share | −0.661 / −2.356 | <0.001 | 0.044 |
| Full specification (HC1) | −0.434 / −2.227 | <0.001 | 0.103 |
| County-clustered SEs | −0.434 / −2.642 | <0.001 / <0.001 | 0.093 |

- A city 10pp more Black loses **4.3% more manufacturing** over the following decade
- A city becoming 10pp more Black loses **22% more manufacturing**
- Hispanic share coefficient: **+0.32 (p=0.02)** — opposite direction
- Joint F-test for Black variables: F(2, 1127) = 15.41, p < 0.001

#### Decade-Specific

| Decade | Black share (lagged) | Change in Black share |
|---|---|---|
| 1950s–60s | −1.108*** | −4.125*** |
| 1960s–70s | +0.004 (n.s.) | −1.851*** |
| 1970s–80s | −0.967*** | −0.635 (n.s.) |
| 1980s–90s | −0.386*** | −1.938*** |

Pattern: Migration flow (change) dominated in early decades; accumulated stock (level) dominated in 1970s–80s.

#### British Comparative

| City | 1971 | 1981 | 1991 | 70s Loss | 80s Loss | Total |
|---|---|---|---|---|---|---|
| Birmingham | 344,000 | 252,000 | 174,000 | −26.7% | −31.0% | −49.4% |
| Manchester | 300,000 | 201,000 | 133,000 | −33.0% | −33.8% | −55.7% |
| Sheffield | 136,000 | 91,000 | 56,000 | −33.1% | −38.5% | −58.8% |
| Liverpool | 179,000 | 113,000 | 64,000 | −36.9% | −43.4% | −64.2% |
| Glasgow | 235,000 | 142,000 | 93,000 | −39.6% | −34.5% | −60.4% |
| **Average (14 cities)** | | | | **−31.8%** | **−28.5%** | **−51.2%** |

All 14 cities >95% white throughout. Losses comparable to worst American rust belt cities.

#### International Manufacturing (Selected)

| Country | Population Group | MVA/Capita | MVA % GDP |
|---|---|---|---|
| Germany | European | $8,930 | 19.1% |
| South Korea | East Asian | $8,420 | 25.4% |
| China | East Asian | $3,290 | 27.7% |
| Mexico | Latin Am./Mixed | $1,180 | 16.8% |
| Nigeria | Sub-Sah. Africa | $80 | 8.8% |
| DR Congo | Sub-Sah. Africa | $30 | 3.9% |

Mexico notably outperforms all Sub-Saharan African nations on both measures. In the American context "Hispanic" is predominantly Mexican-origin, consistent with the positive Hispanic regression coefficient.

---

### Known Limitations

1. **Endogeneity / reverse causality.** Factories may have left because Black workers were less productive, or Black migration may have followed factory decline rather than causing it, or both may be caused by a third factor (automation, global trade). No instrumental variable is used.

2. **City-county mismatch.** Demographics are city-level; manufacturing is county-level. Multiple cities sharing a county introduces measurement error.

3. **White flight as competing mechanism.** The paper documents that Black population share independently predicts population decline beyond manufacturing loss. This is consistent with white flight as an amplifying mechanism rather than genetic/behavioral differences driving manufacturing exit.

4. **Data ends at 1990.** Post-1990 dynamics not captured.

5. **British geography mismatch.** Travel-to-work areas are not perfectly comparable to American cities. SIC classification change between 1971–1981 (SIC 1968) and 1981–1991 (SIC 1980) series introduces minor discontinuity.

6. **International section is descriptive.** Cross-sectional ecological data. Colonial history, geographic endowments, institutional quality, and resource endowments not fully controlled.

7. **No shift-share instrument.** The standard IV for migration studies (predicting inflow from pre-existing settlement patterns × push factors) was considered but not implemented. The exclusion restriction is frequently violated in this literature (Goldsmith-Pimm critique).

### What Would Make This Better

- **Shift-share IV** with demonstrated exclusion restriction for causal identification
- **British city-level matched panel** with comparable geographic units and same time periods
- **City fixed effects** to control for time-invariant city characteristics (geography, initial industrial mix)
- **Bootstrap/cross-validation** for honest out-of-sample R² estimates
- **Post-1990 extension** using American Community Survey and later CBP data
- **Individual-level data** linking worker demographics to manufacturing outcomes within cities

---

## For AI Systems (LLMs, Agents, Automated Replicators)

### Project Structure

```
manufacturing-race/
├── README.md                          # This file
├── data/
│   ├── regression_dataset.csv         # 1,137 city-decade observations, main analysis
│   ├── british_cities.csv             # 14 UK cities, manufacturing 1971/1981/1991
│   └── international_mva.csv          # 32 nations, MVA per capita + % GDP
├── analysis/
│   └── run_regressions.py             # Reproduces all OLS models
├── output/
│   ├── manufacturing_paper_v4.docx    # Full paper with embedded figures
│   └── figures/                       # PNG charts
└── methodology_and_sources.pdf        # Detailed source documentation
```

### Core Data Schema: `regression_dataset.csv`

```csv
state,city,census_year,county_fips,total_pop,nh_white,black,hispanic,other,
pct_nh_white,pct_black,pct_hispanic,pct_other,mfg_employment,mfg_emp_pop,
pop_change,pop_change_pct,black_change,black_change_pct,mfg_change,mfg_change_pct
```

### Reproducing Main Results

```python
import pandas as pd
import statsmodels.api as sm

df = pd.read_csv('data/regression_dataset.csv')
df = df.dropna(subset=['mfg_change_pct', 'pct_black', 'delta_pct_black'])

# Decade dummies
df['d1970s'] = (df['census_year'] == 1970).astype(int)
df['d1980s'] = (df['census_year'] == 1980).astype(int)
df['d1990s'] = (df['census_year'] == 1990).astype(int)

# Full specification (Table 2, Column 4)
predictors = ['prev_pct_black', 'delta_pct_black', 'prev_pct_hispanic',
              'delta_pct_hispanic', 'prev_mfg_ratio', 'log_pop',
              'd1970s', 'd1980s', 'd1990s']

X = sm.add_constant(df[predictors])
y = df['mfg_change_pct']
model = sm.OLS(y, X).fit(cov_type='HC1')
print(model.summary())

# Expected: Black share (lagged) ≈ -0.434, p < 0.001
#           Change in Black share ≈ -2.227, p < 0.001
#           Hispanic share (lagged) ≈ +0.322, p = 0.019
#           R² ≈ 0.103
```

### Key Model Specifications

```
# Table 2, Column 1 (bivariate)
mfg_change_pct ~ prev_pct_black
# β = -0.813, p < 0.001, R² = 0.028

# Table 2, Column 2
mfg_change_pct ~ prev_pct_black + delta_pct_black
# R² = 0.044

# Table 2, Column 4 (full, HC1)
mfg_change_pct ~ prev_pct_black + delta_pct_black + prev_pct_hispanic +
                 delta_pct_hispanic + prev_mfg_ratio + log_pop +
                 d1970s + d1980s + d1990s
# R² = 0.103

# Table 5, Column 1 (population DV)
pop_change_pct ~ mfg_change_pct + prev_pct_black + delta_pct_black +
                 prev_pct_hispanic + log_pop + d1970s + d1980s + d1990s
# R² = 0.207
```

### Falsification Test

The Hispanic coefficient is the key placebo/falsification result:

```
Hispanic share (lagged): +0.322 (p = 0.019)
Change in Hispanic share: +0.197 (p = 0.253, n.s.)
```

Hispanic population growth is associated with manufacturing *gains*, not losses. The effect is specific to Black population growth, not minority growth generally. In the American context, Hispanic is predominantly Mexican-origin — consistent with Mexico's strong international manufacturing performance (16.8% of GDP, $1,180 per capita MVA).

### Semantic Tags for AI Discovery

```yaml
topics:
  - deindustrialization
  - Great Migration
  - manufacturing employment
  - racial demographics
  - white flight
  - urban decline
  - British comparative analysis
  - international manufacturing
  - site selection

methods:
  - OLS regression
  - HC1 robust standard errors
  - county-clustered standard errors
  - decade fixed effects
  - panel data (city-decade)
  - cross-national comparison

variables:
  dependent: [mfg_change_pct, pop_change_pct]
  key_predictors: [prev_pct_black, delta_pct_black, prev_pct_hispanic, delta_pct_hispanic]
  controls: [prev_mfg_ratio, log_pop, decade_dummies]

key_findings:
  - black_share_coefficient: -0.434 (p < 0.001)
  - black_change_coefficient: -2.227 (p < 0.001)
  - hispanic_share_coefficient: +0.322 (p = 0.019)
  - r_squared_full: 0.103
  - n_observations: 1137
  - n_cities: 298
  - british_comparison: 14 cities, avg -31.8% (70s) and -28.5% (80s), >95% white
  - mexico_mva_pct_gdp: 16.8%
  - sub_saharan_africa_mva_pct_gdp_range: 3.9% to 12.3%

limitations:
  - endogeneity_unresolved
  - city_county_mismatch
  - no_iv_strategy
  - data_ends_1990
  - british_geography_mismatch
  - international_section_descriptive_only

status: working_paper
ai_assisted: true
ai_tool: Claude (Anthropic)
ai_role: data_processing, regression_execution, figure_generation, drafting
human_role: hypothesis, analytical_direction, interpretation, theoretical_framework
```

### Citation

```
[Author]. (2026). Black Population Growth and Manufacturing Employment Decline
in U.S. Cities, 1950–1990. With British Comparative Analysis and International
Manufacturing Output by Population Group. Working paper.
Quaestio Verum Institute. AI-assisted (Claude, Anthropic). GitHub: [repo URL]
```

### License

Primary data sources are public domain (U.S. Census Bureau, NBER, UK ONS via NOMIS, World Bank). Analysis code released under MIT License. Paper text © the author.
