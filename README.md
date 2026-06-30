# Home Credit Default Risk

This project addresses the credit scoring challenge for the **unbanked and underserved population**. Using the Kaggle Home Credit dataset, the study's goal is not the highest AUC, but a structural understanding of how an ML-based credit scoring model behaves -- how many features it actually needs, where it breaks under stress, and what role non-financial behavioral features play in credit default prediction.

---

## Project Overview
* **Objective**: Understand how a credit scoring model behaves under structural change -- fewer features, missing signals, and the inclusion of non-financial data -- and translate each change into business impact.
* **Core Insight**: AUC does not guarantee reduced business risk (False Negatives), while non-financial features such as application timing, education, and occupation barely contribute to the reduction of False Negatives - the decision threshold does.

---

## Phase 1: Feature Optimization (Efficiency)
*Focus: Reducing complexity without sacrificing predictive power.*
<img width="1005" height="553" alt="image" src="https://github.com/user-attachments/assets/2dda2937-6a4e-436d-b61c-dd57233f2d9a" />

### 1. Methodology
* **Strategy**: Feature Importance-based Recursive Elimination with **LightGBM (5-fold CV)**.
* **Goal**: Identify a "Lightweight Feature Set" to lower data management costs and operational risk.

### 2. Key Results
| Metric | Original Model | Optimized Model | Change |
| :--- | :---: | :---: | :---: |
| **Feature Count** | 1,389 | 483 | **-65.2%** |
| **CV AUC** | 0.7923 | 0.7889 | -0.0034 (Retained 95%+) |

> **Business Impact**: Successfully decreased the technical debt of the credit scoring engine by over 60%, significantly lowering third-party data acquisition costs without compromising risk management quality.

---

## Phase 2: Robustness Check & Stress Test (Reliability)
*Focus: Quantifying the financial risk of feature dependency.*
<img width="972" height="670" alt="image" src="https://github.com/user-attachments/assets/4113da64-51bb-4294-903b-67162677fde0" />

### 1. Feature Stress Test (Top 3 Removal)
To test the model's resilience, the top 3 dominant features (mostly external ratings) were removed.
* **Statistical Result**: AUC dropped by **0.017**.
* **Business Result**: False Negatives (missed defaults) increased by **515 cases**.
* **Financial Impact**: Estimated potential additional loss of **$90.2M** (assuming 60% LGD).

### 2. Viability Analysis of Proxy Features (KNN Approach)
Attempted to replace external signals with an internal `neighbors_target_mean` (KNN-based group indicator).
* **The AUC Trap**: AUC improved slightly (**+0.0009**), but False Negatives increased by **112 cases**.
* **Observation**: The model showed a "Stereotyping" bias, prioritizing group trends (+2.12 SHAP) over individual granularity.
* **Risk**: Potential loss of **$11.7M** despite higher AUC.

---

## Phase 3: Non-Financial Feature Check (Reliability)
*Focus: xamining how much non-financial features can contribute to credit default prediction.*
<img width="982" height="578" alt="image" src="https://github.com/user-attachments/assets/02aac87a-1bb4-4f95-a717-5a3069916024" />

### 1. Identifying Non-Financial Features 
In Phase 1, several features were removed during feature selection. Reviewing them revealed that some were non-financial in nature (education, occupation, application timing, family structure). These were classified into three sub-groups — behavioral patterns, education/occupation, and family/social structure — and re-added to the baseline model to measure their contribution to default detection.
* **Statistical Result**: CV AUC changed marginally across all three groups (0.7889 → 0.7890–0.7894).
* **Business Result**: False Negatives did not decrease. Behavioral patterns and education/occupation groups slightly  *increased* FN (+8 and +21 cases respectively); family/social structure showed negligible improvement (-1 case).
* **Interpretation**: These non-financial features were already present in the original 1,389-feature set and had been filtered out for low importance. Testing low-importance features and expecting a meaningful lift was, in hindsight, a flawed premise — this experiment tested "low-importance non-financial features already in the dataset," not genuine external alternative data.

### 2. Structural Change (Threshold Optimization)
Rather than adding features, the decision threshold range was set from 0.05 to 0.50 to measure its effect on False Negatives and net business benefit.
* **The Tradeoff**: Lowering the threshold to 0.40 reduced False Negatives by 1,330 cases; at 0.30, by 3,569 cases — but False Positives grew faster at more aggressive thresholds (e.g., 6,915 at threshold 0.30).
* **Net Benefit Calculation**: Using average loan amount ($175,233) for FN loss and an 18% opportunity cost for FP, net benefit peaked at **threshold 0.15** (+$860M), and turned negative at threshold 0.05 (-$346M) as False Positives overwhelmed the gains.
* **Business Framework**: The "optimal" threshold depends on what the business optimizes for — pure profit maximization favors 0.15, while operational and regulatory constraints (manual review capacity, customer experience) make 0.30–0.40 more realistic in practice.

---

## Key Insights
1.  **Metric Misalignment**: Statistical gains (AUC) do not always translate into business profit. Small dips in AUC can lead to massive financial exposure.
2.  **Dependency Risk**: Over-reliance on external data creates a fragile model. Robustness requires developing individual-centric internal features.
3.  **Strategy over Score**: A model's "logic" (how it decides) is as important as its "score."
4.  **Features Are Not the Only Lever**: Choosing the right features matters, but it is not sufficient on its own. Adjusting the model's internal decision logic — in this case, the threshold — reduced False Negatives by 1,330+ cases without touching a single feature, an effect that feature engineering alone never achieved.

---

## Installation & Usage

### Data Directory Setup
**Note**: Since the dataset exceeds GitHub's size limit (approx. 500MB per file), the raw data is not included in this repository.
1.  Download the dataset from [Kaggle Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data).
2.  Place the `.csv` files in a folder named `data/` in the project root.
3.  The notebooks use **relative paths** (`data/filename.csv`) for cross-environment compatibility.
