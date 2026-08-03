# Home Credit Default Risk — Feature Engineering, Model Optimization & Risk Threshold Analysis

> **Scope note**: This repository covers the full pipeline in four phases — from raw-table feature engineering through feature selection, robustness stress-testing, and business-driven threshold optimization. Phase 0 (feature engineering) reproduces and extends **kozodoi's public Kaggle solution** for the Home Credit Default Risk competition; Phases 1–3 are original analysis built on top of that feature set.
---

## Results at a Glance

| Question tested | Result | Business impact |
|---|---|---|
| Can the feature set shrink without losing performance? | **1,389 → 483 features (-65.2%)** at only -0.0034 AUC | Lower data-acquisition cost, same risk quality |
| What happens if the top 3 features disappear? | AUC -0.017, but **+515 missed defaults** | **-$90.2M** estimated loss |
| Can an internal proxy replace external bureau signals? | AUC **+0.0009**, but **+112 missed defaults** | **-$11.7M** despite the AUC gain |
| Do non-financial features (education, behavior, family) help? | AUC +0.0001–0.0005 | **No FN improvement** — 2 of 3 groups made it worse |
| Does moving the decision threshold help more than adding features? | Threshold 0.15 vs. 0.50: **-1,330+ False Negatives** | **+$860M** net benefit — the single largest lever found |

**Bottom line**: in this dataset, adjusting the decision threshold changed business outcomes more
than any feature-side intervention did. AUC moved by fractions of a point across every experiment;
the dollar impact moved by tens to hundreds of millions. See each phase below for methodology.

---

## What I Reproduced vs. What I Designed

| | Scope | Ownership |
|---|---|---|
| **Phase 0 — Feature Engineering** | Reproduces kozodoi's public Kaggle solution (7-table merge, ratio features, aggregation) | Reproduction; my own contribution here is limited to extending `compute_accept_reject_ratio` to multiple lag windows and documenting the leakage/missing-value/outlier decisions the original solution makes implicitly |
| **Phase 1 — Feature Optimization** | Gain-based feature elimination, 95% cumulative-importance cutoff | Designed and run by me |
| **Phase 2 — Robustness Check** | Feature stress test, KNN proxy feature, SHAP analysis | Designed and run by me |
| **Phase 3 — Non-Financial Features & Threshold** | Hypothesis groups, net-benefit threshold sweep | Designed and run by me |

The judgment calls this project is meant to demonstrate — what to test, what "success" means for
each test, and how to translate a confusion matrix into a dollar figure — live entirely in Phases
1–3.

---

## Problem Definition

Design a credit-scoring pipeline for **unbanked and underserved applicants** that goes beyond asingle AUC number: build the feature set from raw transactional and bureau data, then verify **how few features, how sensitive to its top signals, and at what decision threshold** the model actually minimizes real lending losses (False Negatives), rather than only maximizing a statistical score.

---

## Project Structure

| Phase | Focus | Question it answers |
|---|---|---|
| **0. Feature Engineering** | Build the feature set from 7 raw tables | What signals can be engineered from a client's application, bureau, and transaction history? |
| **1. Feature Optimization** | Reduce 1,389 → 483 features | How many of those features are actually load-bearing? |
| **2. Robustness Check** | Remove top features / test proxy features | How fragile is the model if its dominant signals disappear or get replaced? |
| **3. Non-Financial Features & Threshold** | Add non-financial features, sweep thresholds | Does more data help more than simply moving the decision threshold? |

---

## Phase 0: Feature Engineering

*Notebook: `01_kozodoi_data_prep_reproduced.ipynb` — a reproduction of kozodoi's public Home Creditsolution, extended in places (see below), used here to build the underlying feature set thatPhases 1–3 analyze.*

### Target Definition
`TARGET` follows the Home Credit competition's standard definition: `1` if the client had payment difficulties (late repayment) on the loan, `0` otherwise. The target column is **split off from the feature tables before any feature engineering begins** (`y = train[["SK_ID_CURR", "TARGET"]]; del train["TARGET"]`), specifically to keep it from leaking into aggregation or encoding steps.

### Data Sources
Seven raw tables are combined at the `SK_ID_CURR` (client) level: `application_{train,test}`, `bureau`, `bureau_balance`, `previous_application`, `POS_CASH_balance`, `credit_card_balance`, and `installments_payments`.

### Leakage Prevention
- Target separated from features prior to any transformation (above).
- All historical tables (`bureau`, `previous_application`, `installments_payments`, `POS_CASH_balance`, `credit_card_balance`) record **past** credit behavior relative to the current application and are aggregated to client level (mean/std/min/max/mode/count) rather than joined row-by-row — this avoids row-level duplication leakage during the merge.
- Train/test are only separated back out **after** feature engineering (`appl["SK_ID_CURR"].isin(y["SK_ID_CURR"])`), so both sets go through an identical transformation pipeline — no train-specific statistic is computed and leaked into test.
- All downstream experiments (Phases 1–3) reuse the same fixed `StratifiedKFold(n_splits=5, random_state=42)` split, so feature-selection and threshold results are comparable across experiments rather than each drawing a different lucky fold.

### Missing Values
No explicit imputation (mean/median/mode fill) is applied to the merged feature table. This is a deliberate choice, not an oversight: LightGBM natively learns an optimal split direction for missing values during training, so imputing them beforehand would discard information (the fact that a value is missing is itself often predictive, e.g. a client with no bureau record). A `create_null_flags` helper exists in the notebook to explicitly flag missingness per column, but it is not invoked in the final pipeline — noted here as a possible improvement (see Limitations).
The one place manual imputation *is* used is downstream in Phase 2, where `EXT_SOURCE_1/2/3` are mean-imputed to construct a KNN-based neighbor feature (KNN requires complete rows).

### Outliers & Skew
- Monetary variables (`AMT_CREDIT`, `AMT_INCOME_TOTAL`, `AMT_ANNUITY`, etc.) are **log-transformed** (`create_logs`) across every table, which compresses the influence of extreme values and reduces distributional skew.
- Date-like columns (`DAYS_BIRTH`, `DAYS_EMPLOYED`, `DAYS_REGISTRATION`, etc.) are unit-converted from raw day counts to years/months (`convert_days`).
- **Known gap**: this pipeline does not explicitly detect or clip domain-known sentinel values (e.g., `DAYS_EMPLOYED`'s `365243` placeholder for "not currently employed"). The log/unit conversion does not remove this anomaly — it is called out explicitly in Limitations rather than silently left unaddressed.

### Feature Engineering Highlights
Each engineered feature is tied to a specific risk hypothesis, not just generated mechanically:

- **Debt burden ratios** (`CREDIT_BY_INCOME`, `ANNUITY_BY_INCOME`, `GOODS_PRICE_BY_INCOME`, `INCOME_PER_PERSON`): hypothesis — repayment risk scales with how much of a client's income is already committed to debt service, not with loan size alone.
- **Repayment discipline features** (`DPD`, `DBD`, `PAYMENT_PERC`, `PAYMENT_DIFF` from `installments_payments`): hypothesis — how consistently a client pays *previous* installments is a more direct default signal than static demographics.
- **Acceptance/rejection ratio with lags** (`compute_accept_reject_ratio`, extended in this reproduction to support multiple lag windows [1, 3, 5] instead of only the single most recent application in the original solution): hypothesis — a client's recent approval/rejection pattern with other lenders signals current creditworthiness better than a single most-recent decision.
- **Cross-table mix ratios** (`mix_AMT_PREV_CREDIT_RATIO`, `mix_AMT_BURO_CREDIT_RATIO`): hypothesis — comparing the current application's requested amount against the client's own historical average (rather than the population average) captures individual behavioral drift.
- **Client-level aggregation** (`aggregate_data`): each historical table is collapsed to mean/volatility/most-frequent-category/diversity statistics per client, capturing not just *what* a client did but how *consistent* their financial behavior has been.

### Result
The pipeline produces **1,389 features** across all merged tables, which becomes the starting point for Phase 1.

---

## Phase 1: Feature Optimization (Efficiency)

*Focus: Reducing complexity without sacrificing predictive power.*

<img width="1005" height="553" alt="image" src="https://github.com/user-attachments/assets/d763faf0-a1b3-40ee-af0a-4efabe5626f6" />


### Methodology
- **Strategy**: Feature-importance-based (gain) elimination with **LightGBM (5-fold CV)**.
- **Goal**: Identify a "lightweight feature set" that lowers data-management cost and operational risk without meaningfully hurting predictive power.
- 494 features were initially selected to reach the 95% cumulative-importance threshold; 11 were unavailable in the reproduced dataset (attributed to encoding/versioning differences from the original solution) and were dropped for reproducibility, leaving **483 features**.

### Key Results

| Metric | Original Model | Optimized Model | Change |
|---|---|---|---|
| Feature Count | 1,389 | 483 | **-65.2%** |
| CV AUC | 0.7923 | 0.7889 | -0.0034 (retained 95%+ of importance) |

> **Business impact**: Reduced the feature footprint of the scoring engine by over 60%, significantly lowering third-party data acquisition and pipeline maintenance cost, with negligible risk-management degradation.

### Feature Dominance
The top 10 features (mostly `EXT_SOURCE` series) account for **31.25%** of total model importance, despite being just 0.7% of the feature count — a concentration that motivates Phase 2's stress test.

---

## Phase 2: Robustness Check & Stress Test (Reliability)

*Focus: Quantifying the financial risk of feature dependency.*

<img width="972" height="670" alt="image" src="https://github.com/user-attachments/assets/95c17d4e-c0ae-4980-a5c9-826752cb5f1a" />


### 1. Feature Stress Test (Top-3 Removal)
The top 3 dominant features (`EXT_SOURCE_MEAN`, `EXT_SOURCE_3`, `EXT_SOURCE_2` — ~22% of total importance) were removed one, two, and three at a time to simulate a real-world scenario where an external bureau feed becomes unavailable or degraded.

- **Statistical result**: AUC dropped by **0.017**.
- **Business result**: False Negatives (missed defaults) increased by **515 cases**.
- **Financial impact**: an estimated additional loss of **$90.2M** (assuming 60% LGD).

### 2. Viability of a Proxy Feature (KNN Approach)
Attempted to replace external bureau signals with an internal `neighbors_target_mean` feature — the average default rate among the 500 nearest neighbors by `EXT_SOURCE_1/2/3` (mean-imputed for KNN compatibility), a technique adapted from top Kaggle solutions.

- **The AUC trap**: AUC improved slightly (**+0.0009**), but False Negatives *increased* by **112
  cases**.
- **Observation (via SHAP)**: the model exhibited a "stereotyping" bias — it leaned on group-level trends (peer default rate) over individual-level granularity.
- **Risk**: an estimated potential loss of **$11.7M** despite the higher AUC — a concrete example of why AUC alone is an unsafe optimization target for this problem.

---

## Phase 3: Non-Financial Features & Threshold Optimization (Reliability)

*Focus: Testing whether non-financial data adds more value than adjusting the decision threshold.*

<img width="605" height="366" alt="image" src="https://github.com/user-attachments/assets/7fb31864-053f-4e4b-a4f7-396cfce40473" />

### 1. Non-Financial Feature Groups
Features removed during Phase 1's selection were reviewed for non-financial content and grouped into three hypotheses, then re-added to the 483-feature baseline:

- **Behavioral patterns** (document flags, application weekday): hypothesis — application timing and completeness reflect a client's diligence/organization.
- **Education / occupation**: hypothesis — occupational stability correlates with income stability.
- **Family / social structure**: hypothesis — household composition affects financial obligations.

| Group | CV AUC | ΔAUC vs. baseline | ΔFalse Negatives vs. baseline |
|---|---|---|---|
| Baseline (483 features) | 0.7889 | – | – |
| + Behavioral | 0.7890 | +0.0001 | +8 |
| + Education/Occupation | 0.7894 | +0.0005 | +21 |
| + Family/Social | 0.7889 | ~0.0000 | -1 |

**Interpretation**: none of the three groups meaningfully reduced False Negatives — two groups *increased* them despite a marginal AUC gain. These features were already present in the original 1,389-feature set and had been filtered out for low importance in Phase 1; testing low-importance features already excluded from the model tested "low-importance non-financial features already in the dataset," not genuine new external data. That distinction matters and is stated explicitly rather than implied as a general finding about non-financial data.

### 2. Structural Change: Threshold Optimization
Rather than adding features, the decision threshold was swept from 0.05 to 0.50 to measure its effect on the False Negative / False Positive trade-off and net business benefit.

| Threshold | FN reduction vs. 0.50 | Net benefit |
|---|---|---|
| 0.40 | -1,330 | positive |
| 0.30 | -3,569 | positive, FP growing faster |
| **0.15** | — | **+$860M (peak)** |
| 0.05 | — | **-$346M (FP overwhelms gains)** |

**Net benefit formula**: `(FN_reduction × avg_loan_amount) − (FP_increase × avg_loan_amount × opportunity_cost_rate)`, using an average loan amount of $175,233 and an 18% opportunity cost for false positives.

**Business framework**: the "optimal" threshold depends on what the business optimizes for — pure profit maximization favors 0.15, while operational and regulatory constraints (manual review capacity, customer experience) make 0.30–0.40 more realistic in practice.

---

## Key Insights

1. **Metric misalignment**: statistical gains (AUC) do not always translate into business profit; a small AUC dip can accompany a very large financial exposure change, and vice versa (Phase 2.2).
2. **Dependency risk**: over-reliance on a small number of external features creates a fragile model — losing 3 features cost $90.2M in this experiment (Phase 2.1).
3. **Strategy over score**: a model's decision logic (how it decides) matters as much as its score.
4. **Features are not the only lever**: adjusting the decision threshold alone reduced False Negatives by 1,330+ cases without touching a single feature — an effect feature engineering alone did not achieve in this experiment (Phase 3.2).

---

## Limitations & Future Work

Stated explicitly rather than left implicit, since this is what a reviewer would ask about first:

- **No formal outlier/anomaly treatment** for known sentinel values (e.g., `DAYS_EMPLOYED = 365243`). Log-transform and unit conversion mitigate general skew but do not clip or flag this specific anomaly. Next step: add explicit anomaly flags for such sentinel values.
- **No explicit missing-value imputation** beyond LightGBM's native handling — a deliberate modeling choice (see Phase 0), but worth validating against an imputation-based baseline to quantify the trade-off rather than assuming it's optimal.
- **`create_null_flags` helper is defined but unused** in the final pipeline; wiring it in would let missingness itself be tested as an explicit signal.
- **Non-financial feature test (Phase 3.1) only re-tested already-filtered low-importance features**, not genuinely new external/alternative data sources — a fair test of the latter would need new data, not a re-inclusion experiment.

---

## Installation & Usage

### Data Directory Setup
**Note**: the raw Kaggle dataset exceeds GitHub's size limit (~500MB per file) and is not included
in this repository.

1. Download the dataset from [Kaggle Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data).
2. Place the `.csv` files in a folder named `data/` in the project root.
3. Run `01_kozodoi_data_prep_reproduced.ipynb` first to generate `train_full_cor.csv`,
   `test_full_cor.csv`, and `y_full_cor.csv`.
4. Run `Feature_Optimization_and_Analysis_eng.ipynb`, then `Robustness_Check_eng.ipynb`, or the
   standalone scripts `combined_experiment.py` / `threshold_analysis.py`, which use **relative
   paths** (`../data`, `../models`) for cross-environment compatibility.

### Attribution
Phase 0's feature engineering reproduces and extends the public solution by Kaggle Grandmaster
**kozodoi** for the Home Credit Default Risk competition, used here for learning purposes and as
the foundation for the original analysis in Phases 1–3.
