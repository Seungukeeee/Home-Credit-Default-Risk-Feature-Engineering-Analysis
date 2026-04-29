# Home Credit Default Risk: Balancing Efficiency and Robustness

This project addresses the credit scoring challenge for the **unbanked and underserved population**. Using the Kaggle Home Credit dataset, the study focuses on optimizing model complexity while rigorously testing its resilience against data dependency risks.

---

## Project Overview
* **Objective**: Minimize operational complexity (Technical Debt) through feature optimization.
* **Constraint**: Evaluate the risk of over-reliance on specific external indicators (`EXT_SOURCE`).
* **Core Insight**: Statistical performance (AUC) must be balanced with business-centric risk metrics (Potential Loss/LGD).

---

## Phase 1: Feature Optimization (Efficiency)
*Focus: Reducing complexity without sacrificing predictive power.*

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

## Key Insights
1.  **Metric Misalignment**: Statistical gains (AUC) do not always translate into business profit. Small dips in AUC can lead to massive financial exposure.
2.  **Dependency Risk**: Over-reliance on external data creates a fragile model. Robustness requires developing individual-centric internal features.
3.  **Strategy over Score**: A model's "logic" (how it decides) is as important as its "score."

---

## Installation & Usage

### Data Directory Setup
**Note**: Since the dataset exceeds GitHub's size limit (approx. 500MB per file), the raw data is not included in this repository.
1.  Download the dataset from [Kaggle Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data).
2.  Place the `.csv` files in a folder named `data/` in the project root.
3.  The notebooks use **relative paths** (`data/filename.csv`) for cross-environment compatibility.
