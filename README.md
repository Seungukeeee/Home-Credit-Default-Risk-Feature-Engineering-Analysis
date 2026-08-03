# Home Credit Default Risk — Feature Engineering, Model Optimization & Risk Threshold Analysis

> **Scope note**: This repository covers the full pipeline in four phases — from raw-table feature engineering through feature selection, robustness stress-testing, and business-driven threshold optimization. Phase 0 (feature engineering) reproduces and extends **kozodoi's public Kaggle solution** for the Home Credit Default Risk competition; Phases 1–3 are original analysis built on top of that feature set.
---

## Problem Definition

Design a credit-scoring pipeline for **unbanked and underserved applicants** that goes beyond asingle AUC number: build the feature set from raw transactional and bureau data, then verify **how few features, how sensitive to its top signals, and at what decision threshold** the model actually minimizes real lending losses (False Negatives), rather than only maximizing a statistical score.

---

## Validation Setup

*What every result in this README rests on — read this before the numbers below.*

**Target**: `TARGET` follows the Home Credit competition's standard definition — `1` if the client
had payment difficulties (late repayment), `0` otherwise.

**Data split**: `application_train`/`application_test` are concatenated for feature engineering,
then split back apart by `SK_ID_CURR` membership *after* all transformations are applied — so both
sets go through an identical pipeline and no train-only statistic leaks into test. Model validation
uses a fixed **5-fold `StratifiedKFold(random_state=42)`**, reused unchanged across every phase and
every experiment below, so results are comparable to each other rather than each drawing a
different lucky fold.

**Leakage prevention**:
- `TARGET` is separated from the feature tables *before* any feature engineering step
  (`y = train[["SK_ID_CURR","TARGET"]]; del train["TARGET"]`), so it cannot leak into aggregation or
  encoding.
- Historical tables (`bureau`, `previous_application`, `installments_payments`,
  `POS_CASH_balance`, `credit_card_balance`) record **past** credit behavior and are aggregated to
  one row per client (mean/std/min/max/mode/count) before merging — this avoids row-level
  duplication leakage during the join.

**Baseline**: the reference point for every experiment in this README is the **483-feature model** from Phase 1 — **CV AUC 0.7889**, 24,825 actual defaults, 23,589 missed at the default 0.50 threshold. Every number quoted below (AUC change, FN change, dollar impact) is measured *against
this baseline*, not against the raw 1,389-feature model.

---

## Results at a Glance

My call, in one line: this dataset doesn't need more features — it needs its decision threshold moved. Every feature-side intervention below (removing top features, building a proxy, adding non-financial data) changed the dollar outcome by single-digit millions at best; moving the threshold alone changed it by hundreds of millions. That's the judgment this project is built to demonstrate, and the numbers below are the evidence for it.

| Question tested | Result | Business impact |
|---|---|---|
| Can the feature set shrink without losing performance? | **1,389 → 483 features (-65.2%)** at only -0.0034 AUC | Lower data-acquisition cost, same risk quality |
| What happens if the top 3 features disappear? | AUC -0.017, but **+515 missed defaults** | **-$90.2M** estimated loss |
| Can an internal proxy replace external bureau signals? | AUC **+0.0009**, but **+112 missed defaults** | **-$11.7M** despite the AUC gain |
| Do non-financial features (education, behavior, family) help? | Best case (Family/Social), re-optimizing threshold too: +30 FN caught | **+$15.9M** — real, but <2% of what threshold tuning alone achieves below |
| Does moving the decision threshold help more than adding features? | At threshold 0.15, Recall jumps from **4.5%** → **46.6%** (baseline's 0.50) | **+$860M** net benefit — the single largest lever found |

**Bottom line**: the baseline model, used at the standard 0.50 threshold, catches only 4.5% of actual defaulters (Recall) — it is close to useless for risk management as shipped. Every feature-side experiment below moved that needle by low single-digit percentage points at best. Threshold tuning alone moved Recall by 10x and net benefit by $860M. AUC barely moved across any experiment (±0.02 at most); the business impact moved by orders of magnitude more than AUC did — which is the core reason this project treats AUC as a poor proxy for the actual decision problem. See each phase below for full methodology and the Recall/Precision trade-off at every threshold.

---

## What I Reproduced vs. What I Designed

| | Scope | Ownership |
|---|---|---|
| **Phase 0 — Feature Engineering** | Reproduces kozodoi's public Kaggle solution (7-table merge, ratio features, aggregation) | Reproduction; my own contribution here is limited to extending `compute_accept_reject_ratio` to multiple lag windows and documenting the leakage/missing-value/outlier decisions the original solution makes implicitly |
| **Phase 1 — Feature Optimization** | Gain-based feature elimination, 95% cumulative-importance cutoff | Designed and run by me |
| **Phase 2 — Robustness Check** | Feature stress test, KNN proxy feature, SHAP analysis | Designed and run by me |
| **Phase 3 — Non-Financial Features & Threshold** | Hypothesis groups, net-benefit threshold sweep | Designed and run by me |

The judgment calls this project is meant to demonstrate — what to test, what "success" means for each test, and how to translate a confusion matrix into a dollar figure — live entirely in Phases 1–3.

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

> **Hypothesis**: features filtered out for low importance in Phase 1 might still carry business value if grouped thematically (behavior, education/occupation, family structure) rather than judged feature-by-feature.
> **Result**: partially confirmed, then reframed. At a fixed 0.50 threshold, none of the three groups meaningfully help (two make False Negatives worse). But letting each group re-optimize its own decision threshold reveals a small, genuine improvement — the more important finding is *how small* it is next to Phase 3.2 below.

Features removed during Phase 1's selection were reviewed for non-financial content and grouped
into three hypotheses, then re-added to the 483-feature baseline:

- **Behavioral patterns** (document flags, application weekday): hypothesis — application timing
  and completeness reflect a client's diligence/organization.
- **Education / occupation**: hypothesis — occupational stability correlates with income stability.
- **Family / social structure**: hypothesis — household composition affects financial obligations.

**At a fixed 0.50 threshold** (the standard classification cutoff), none of the three groups clearly
helped — AUC moved by 0.0001–0.0005 and two groups showed slightly *more* missed defaults, not
fewer.

**Letting each group find its own optimal threshold** (`combined_experiment_summary.csv`) tells a
different, more precise story:

| Group | Its own optimal threshold | FN reduction vs. baseline (at that threshold) | Net benefit |
|---|---|---|---|
| Baseline (483 features) | 0.05 | — (reference) | $0 |
| + Behavioral | 0.10 | +49 fewer missed defaults | **+$5.6M** |
| + Education/Occupation | 0.05 | +9 fewer missed defaults | **+$15.6M** |
| + Family/Social | 0.05 | +30 fewer missed defaults | **+$15.9M** |

**Interpretation**: all three groups *do* produce a small, real improvement once the threshold is also allowed to move — this is a more accurate finding than "non-financial features don't help," and it's worth stating precisely rather than rounding down to a flat no. But the scale is the point: the best of the three (Family/Social, +$15.9M) delivers **under 2% of the $860M** achieved by threshold optimization alone on the unchanged baseline (Phase 3.2, below) — with no new features, no new data collection, and no added model complexity. These features were also already present in the original 1,389-feature set and had been filtered out for low importance in Phase 1, so this
tests "low-importance non-financial features already in the dataset," not genuinely new external data — a caveat worth keeping in mind before generalizing the conclusion.

### 2. Structural Change: Threshold Optimization

> **Hypothesis**: if adding features (Phase 3.1) barely moves the needle, a lever that doesn't touch features at all — the decision threshold — might matter more.
> **Result**: confirmed, by roughly two orders of magnitude — this was the single highest-leverage change found across all four phases.

Rather than adding features, the decision threshold on the unchanged 483-feature baseline was swept from 0.05 to 0.50, tracking Recall, Precision, and net business benefit at each point(`threshold_analysis.csv`).

| Threshold | Recall | Precision | Net benefit vs. 0.50 |
|---|---|---|---|
| 0.05 | 0.830 | 0.147 | **-$346.5M** (FP overwhelms gains) |
| 0.10 | 0.624 | 0.212 | +$731.2M |
| **0.15** | 0.466 | 0.268 | **+$860.4M (peak)** |
| 0.20 | 0.351 | 0.319 | +$766.7M |
| 0.25 | 0.256 | 0.359 | +$584.9M |
| 0.30 | 0.189 | 0.404 | +$433.3M |
| 0.50 (baseline) | 0.045 | 0.577 | $0 (reference) |

**Net benefit formula**: `(FN_reduction × avg_loan_amount) − (FP_increase × avg_loan_amount × opportunity_cost_rate)`, using an average loan amount of $175,233 and an 18% opportunity cost for false positives, both relative to the 0.50 baseline.

**My recommendation**: 0.15 maximizes net benefit on paper, but at that point only **26.8%** of flagged applications are true defaults (Precision) — meaning roughly 3 in 4 flagged clients would be manually reviewed for nothing, which has a real operational cost this formula doesn't capture (reviewer headcount, customer friction, false-decline reputational risk). I'd recommend **0.20–0.25** as the practical operating point instead: it keeps ~80–90% of the theoretical net benefit ($585–767M vs. the $860M peak) while cutting the false-positive review load roughly in half
versus 0.15 (Precision 0.32–0.36 vs. 0.27). The exact choice ultimately depends on the bank's actual manual-review capacity, which isn't modeled here — but the direction (move well away from 0.50, land short of the raw net-benefit peak) is a judgment call I'd defend in a business review, not just a number I'd report.

---

## Key Insights

1. **Metric misalignment**: statistical gains (AUC) do not always translate into business profit; a small AUC dip can accompany a very large financial exposure change, and vice versa (Phase 2.2).
2. **Dependency risk**: over-reliance on a small number of external features creates a fragile model — losing 3 features cost $90.2M in this experiment (Phase 2.1).
3. **Strategy over score**: a model's decision logic (how it decides) matters as much as its score.
4. **Threshold >> features, quantified**: the best feature-side intervention tested (adding Family/Social features, Phase 3.1) delivered $15.9M in net benefit; moving the decision threshold alone on the exact same model delivered $860M (Phase 3.2) — roughly 50x more, with zero new data and zero added complexity. If I had to prioritize one lever on a real team with limited time, this is the evidence I'd bring to that conversation.

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
Phase 0's feature engineering reproduces and extends the public solution by Kaggle Grandmaster **kozodoi** for the Home Credit Default Risk competition, used here for learning purposes and as the foundation for the original analysis in Phases 1–3.
