"""
EWS Experiment: Alternative Data Integration
============================================
Research Question:
    When non-financial individual applicant features are added to the baseline model,
    does False Negative decrease significantly?

Hypothesis:
    Behavioral pattern features (repayment willingness) will reduce FN more than
    education/occupation (repayment ability) or family structure features,
    because willingness precedes ability.

Experiment Structure:
    - Baseline : 482 features, FN = 23,589
    - Exp 1    : Baseline + Group 1 (Behavioral patterns, 39 features)
    - Exp 2    : Baseline + Group 2 (Education/Occupation, 89 features)
    - Exp 3    : Baseline + Group 3 (Family/Social structure, 7 features)

Success Criterion:
    FN reduction >= 1,000 cases (~$175M loss prevention)
"""

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix
from lightgbm import log_evaluation, early_stopping
import pickle
import warnings
import gc

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 0. Paths
# ──────────────────────────────────────────────
# Paths are relative to the repository root.
# Place train_full_cor.csv / test_full_cor.csv / y_full_cor.csv under DATA_PATH,
# and selected_features_483.pkl under MODEL_PATH before running.
DATA_PATH   = "../data"
MODEL_PATH  = "../models"
OUTPUT_PATH = "./results"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# ──────────────────────────────────────────────
# 1. Load Data & Baseline Features
# ──────────────────────────────────────────────
print("=" * 60)
print("Loading data...")
print("=" * 60)

train = pd.read_csv(os.path.join(DATA_PATH, "train_full_cor.csv"))
test  = pd.read_csv(os.path.join(DATA_PATH, "test_full_cor.csv"))
y     = pd.read_csv(os.path.join(DATA_PATH, "y_full_cor.csv"))

y_target = y["TARGET"] if "TARGET" in y.columns else y.iloc[:, 0]

with open(os.path.join(MODEL_PATH, "selected_features_483.pkl"), "rb") as f:
    baseline_features = pickle.load(f)

# filter to existing columns only
baseline_features = [f for f in baseline_features if f in train.columns]
print(f"Baseline features loaded: {len(baseline_features)}")

# ──────────────────────────────────────────────
# 2. Feature Groups
# ──────────────────────────────────────────────

# Group 1: Behavioral Patterns (repayment willingness)
# Hypothesis: These reflect intentional behavior → strongest FN reduction
group1_behavior = [
    # Document submission patterns
    "app_FLAG_DOCUMENT_2",  "app_FLAG_DOCUMENT_4",  "app_FLAG_DOCUMENT_5",
    "app_FLAG_DOCUMENT_6",  "app_FLAG_DOCUMENT_7",  "app_FLAG_DOCUMENT_8",
    "app_FLAG_DOCUMENT_9",  "app_FLAG_DOCUMENT_10", "app_FLAG_DOCUMENT_11",
    "app_FLAG_DOCUMENT_12", "app_FLAG_DOCUMENT_13", "app_FLAG_DOCUMENT_14",
    "app_FLAG_DOCUMENT_15", "app_FLAG_DOCUMENT_16", "app_FLAG_DOCUMENT_17",
    "app_FLAG_DOCUMENT_18", "app_FLAG_DOCUMENT_19", "app_FLAG_DOCUMENT_20",
    "app_FLAG_DOCUMENT_21",
    # Contact information
    "app_FLAG_PHONE", "app_FLAG_EMAIL",
    # Application timing (day of week)
    "app_WEEKDAY_APPR_PROCESS_START_MONDAY",
    "app_WEEKDAY_APPR_PROCESS_START_TUESDAY",
    "app_WEEKDAY_APPR_PROCESS_START_WEDNESDAY",
    "app_WEEKDAY_APPR_PROCESS_START_THURSDAY",
    "app_WEEKDAY_APPR_PROCESS_START_SATURDAY",
    "app_WEEKDAY_APPR_PROCESS_START_SUNDAY",
    # Previous application timing patterns
    "prev_WEEKDAY_APPR_PROCESS_START_MONDAY_unique",
    "prev_WEEKDAY_APPR_PROCESS_START_MONDAY_mode_True",
    "prev_WEEKDAY_APPR_PROCESS_START_TUESDAY_unique",
    "prev_WEEKDAY_APPR_PROCESS_START_TUESDAY_mode_True",
    "prev_WEEKDAY_APPR_PROCESS_START_WEDNESDAY_unique",
    "prev_WEEKDAY_APPR_PROCESS_START_WEDNESDAY_mode_True",
    "prev_WEEKDAY_APPR_PROCESS_START_THURSDAY_unique",
    "prev_WEEKDAY_APPR_PROCESS_START_THURSDAY_mode_True",
    "prev_WEEKDAY_APPR_PROCESS_START_SATURDAY_unique",
    "prev_WEEKDAY_APPR_PROCESS_START_SATURDAY_mode_True",
    "prev_WEEKDAY_APPR_PROCESS_START_SUNDAY_unique",
    "prev_WEEKDAY_APPR_PROCESS_START_SUNDAY_mode_True",
]

# Group 2: Education / Occupation (repayment ability)
# Hypothesis: Derived from willingness → secondary FN reduction
group2_edu_job = [
    # Education level
    "app_NAME_EDUCATION_TYPE_Higher_education",
    "app_NAME_EDUCATION_TYPE_Secondary___secondary_special",
    "app_NAME_EDUCATION_TYPE_Incomplete_higher",
    "app_NAME_EDUCATION_TYPE_Lower_secondary",
    # Income type
    "app_NAME_INCOME_TYPE_State_servant",
    "app_NAME_INCOME_TYPE_Commercial_associate",
    "app_NAME_INCOME_TYPE_Pensioner",
    "app_NAME_INCOME_TYPE_Unemployed",
    "app_NAME_INCOME_TYPE_Student",
    "app_NAME_INCOME_TYPE_Maternity_leave",
    # Occupation type (high importance ones first)
    "app_OCCUPATION_TYPE_Core_staff",
    "app_OCCUPATION_TYPE_High_skill_tech_staff",
    "app_OCCUPATION_TYPE_Sales_staff",
    "app_OCCUPATION_TYPE_Managers",
    "app_OCCUPATION_TYPE_Low_skill_Laborers",
    "app_OCCUPATION_TYPE_Medicine_staff",
    "app_OCCUPATION_TYPE_Cleaning_staff",
    "app_OCCUPATION_TYPE_Cooking_staff",
    "app_OCCUPATION_TYPE_Security_staff",
    "app_OCCUPATION_TYPE_Waiters_barmen_staff",
    "app_OCCUPATION_TYPE_Secretaries",
    "app_OCCUPATION_TYPE_IT_staff",
    "app_OCCUPATION_TYPE_Private_service_staff",
    "app_OCCUPATION_TYPE_HR_staff",
    "app_OCCUPATION_TYPE_Realty_agents",
    # Organization type (high importance ones)
    "app_ORGANIZATION_TYPE_Self_employed",
    "app_ORGANIZATION_TYPE_Business_Entity_Type_3",
    "app_ORGANIZATION_TYPE_Transport__type_3",
    "app_ORGANIZATION_TYPE_Transport__type_4",
    "app_ORGANIZATION_TYPE_Industry__type_9",
    "app_ORGANIZATION_TYPE_Military",
    "app_ORGANIZATION_TYPE_Security_Ministries",
    "app_ORGANIZATION_TYPE_School",
    "app_ORGANIZATION_TYPE_Trade__type_7",
    "app_ORGANIZATION_TYPE_Restaurant",
    "app_ORGANIZATION_TYPE_Realtor",
    "app_ORGANIZATION_TYPE_Bank",
    "app_ORGANIZATION_TYPE_Medicine",
    "app_ORGANIZATION_TYPE_Other",
    "app_ORGANIZATION_TYPE_Security",
    "app_ORGANIZATION_TYPE_Hotel",
    "app_ORGANIZATION_TYPE_Legal_Services",
    "app_ORGANIZATION_TYPE_Police",
    "app_ORGANIZATION_TYPE_Postal",
    "app_ORGANIZATION_TYPE_University",
    "app_ORGANIZATION_TYPE_Kindergarten",
    "app_ORGANIZATION_TYPE_Telecom",
    "app_ORGANIZATION_TYPE_Industry__type_3",
    "app_ORGANIZATION_TYPE_Industry__type_11",
    "app_ORGANIZATION_TYPE_Industry__type_12",
    "app_ORGANIZATION_TYPE_Industry__type_4",
    "app_ORGANIZATION_TYPE_Industry__type_1",
    "app_ORGANIZATION_TYPE_Business_Entity_Type_1",
    "app_ORGANIZATION_TYPE_Business_Entity_Type_2",
    "app_ORGANIZATION_TYPE_Transport__type_2",
    "app_ORGANIZATION_TYPE_Electricity",
    "app_ORGANIZATION_TYPE_Government",
    "app_ORGANIZATION_TYPE_Housing",
    "app_ORGANIZATION_TYPE_Services",
    "app_ORGANIZATION_TYPE_Emergency",
    "app_ORGANIZATION_TYPE_Agriculture",
    "app_ORGANIZATION_TYPE_Industry__type_5",
    "app_ORGANIZATION_TYPE_Industry__type_2",
    "app_ORGANIZATION_TYPE_Industry__type_6",
    "app_ORGANIZATION_TYPE_Industry__type_7",
    "app_ORGANIZATION_TYPE_Industry__type_8",
    "app_ORGANIZATION_TYPE_Industry__type_10",
    "app_ORGANIZATION_TYPE_Industry__type_13",
    "app_ORGANIZATION_TYPE_Mobile",
    "app_ORGANIZATION_TYPE_Trade__type_1",
    "app_ORGANIZATION_TYPE_Trade__type_2",
    "app_ORGANIZATION_TYPE_Trade__type_3",
    "app_ORGANIZATION_TYPE_Trade__type_4",
    "app_ORGANIZATION_TYPE_Trade__type_5",
    "app_ORGANIZATION_TYPE_Trade__type_6",
    "app_ORGANIZATION_TYPE_Religion",
    "app_ORGANIZATION_TYPE_Insurance",
    "app_ORGANIZATION_TYPE_Cleaning",
    "app_ORGANIZATION_TYPE_Culture",
    "app_ORGANIZATION_TYPE_XNA",
    "app_ORGANIZATION_TYPE_Transport__type_1",
    # Loan type
    "app_NAME_CONTRACT_TYPE_Revolving_loans",
    "app_ANNUITY_LENGTH",
    # Previous application contract types
    "prev_NAME_CONTRACT_TYPE_Consumer_loans_unique",
    "prev_NAME_CONTRACT_TYPE_Consumer_loans_mode_True",
    "prev_NAME_CONTRACT_TYPE_Revolving_loans_unique",
    "prev_NAME_CONTRACT_TYPE_Revolving_loans_mode_True",
    "prev_NAME_CONTRACT_TYPE_XNA_unique",
    "prev_NAME_CONTRACT_TYPE_XNA_mode_True",
]

# Group 3: Family / Social Structure
# Hypothesis: Contextual constraint → weakest direct FN reduction
group3_family = [
    "app_CNT_CHILDREN",
    "app_CNT_ADULTS",
    "app_CHILDREN_RATIO",
    "app_NAME_FAMILY_STATUS_Separated",
    "app_NAME_FAMILY_STATUS_Single___not_married",
    "app_NAME_FAMILY_STATUS_Widow",
    "app_NAME_FAMILY_STATUS_Unknown",
]

# ──────────────────────────────────────────────
# 3. LightGBM Config
# ──────────────────────────────────────────────
SEED      = 42
NUM_FOLDS = 5
THRESHOLD = 0.5  # default threshold for FN/FP measurement

lgbm_params = dict(
    n_estimators     = 10000,
    learning_rate    = 0.005,
    num_leaves       = 70,
    colsample_bytree = 0.8,
    subsample        = 0.9,
    max_depth        = 7,
    reg_alpha        = 0.1,
    reg_lambda       = 0.1,
    min_split_gain   = 0.01,
    min_child_weight = 2,
    random_state     = SEED,
    n_jobs           = -1,
)

AVG_LOAN_AMOUNT = 175_233  # USD

# ──────────────────────────────────────────────
# 4. Experiment Runner
# ──────────────────────────────────────────────
def run_experiment(exp_name, feature_list, train_df, y_target, threshold=THRESHOLD):
    """
    Run 5-fold CV with given features.
    Returns AUC, confusion matrix, and business impact.
    """
    print(f"\n{'='*60}")
    print(f"Experiment: {exp_name}")
    print(f"Features  : {len(feature_list)}")
    print(f"{'='*60}")

    # filter to existing columns
    features = [f for f in feature_list if f in train_df.columns]
    missing  = [f for f in feature_list if f not in train_df.columns]
    if missing:
        print(f"[WARNING] {len(missing)} features not found in dataset: {missing[:5]}{'...' if len(missing)>5 else ''}")
    print(f"Valid features: {len(features)}")

    folds     = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    aucs      = np.zeros(NUM_FOLDS)
    all_preds = np.zeros(len(train_df))
    all_true  = np.zeros(len(train_df))

    for fold, (trn_idx, val_idx) in enumerate(folds.split(train_df[features], y_target)):
        trn_x = train_df[features].iloc[trn_idx]
        trn_y = y_target.iloc[trn_idx]
        val_x = train_df[features].iloc[val_idx]
        val_y = y_target.iloc[val_idx]

        model = lgb.LGBMClassifier(**lgbm_params)
        model.fit(
            trn_x, trn_y,
            eval_set    = [(val_x, val_y)],
            eval_metric = "auc",
            callbacks   = [log_evaluation(period=1000), early_stopping(stopping_rounds=300)],
        )

        preds = model.predict_proba(val_x)[:, 1]
        aucs[fold] = roc_auc_score(val_y, preds)
        all_preds[val_idx] = preds
        all_true[val_idx]  = val_y.values

        print(f"  Fold {fold+1} AUC: {aucs[fold]:.6f}")
        del trn_x, trn_y, val_x, val_y, model
        gc.collect()

    # confusion matrix across all folds
    preds_binary = (all_preds >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(all_true, preds_binary).ravel()

    mean_auc = np.mean(aucs)
    loss_exposure = fn * AVG_LOAN_AMOUNT

    print(f"\n--- Results: {exp_name} ---")
    print(f"CV AUC    : {mean_auc:.6f}")
    print(f"TN        : {tn:,}")
    print(f"FP        : {fp:,}")
    print(f"FN        : {fn:,}  ← key metric")
    print(f"TP        : {tp:,}")
    print(f"Recall    : {tp/(tp+fn):.4f}")
    print(f"Precision : {tp/(tp+fp):.4f}")
    print(f"Loss Exposure (FN × $175,233): ${loss_exposure:,.0f}")

    return {
        "experiment" : exp_name,
        "n_features" : len(features),
        "cv_auc"     : round(mean_auc, 6),
        "TN"         : int(tn),
        "FP"         : int(fp),
        "FN"         : int(fn),
        "TP"         : int(tp),
        "Recall"     : round(tp/(tp+fn), 4),
        "Precision"  : round(tp/(tp+fp), 4),
        "loss_exposure_usd": int(loss_exposure),
    }


# ──────────────────────────────────────────────
# 5. Run All Experiments
# ──────────────────────────────────────────────
results = []

# Baseline
res = run_experiment(
    "Baseline (482 features)",
    baseline_features,
    train, y_target
)
results.append(res)
baseline_fn = res["FN"]

# Exp 1: Baseline + Behavioral Patterns
exp1_features = baseline_features + [f for f in group1_behavior if f not in baseline_features]
res = run_experiment(
    "Exp1: Baseline + Behavioral Patterns",
    exp1_features,
    train, y_target
)
results.append(res)

# Exp 2: Baseline + Education/Occupation
exp2_features = baseline_features + [f for f in group2_edu_job if f not in baseline_features]
res = run_experiment(
    "Exp2: Baseline + Education/Occupation",
    exp2_features,
    train, y_target
)
results.append(res)

# Exp 3: Baseline + Family/Social Structure
exp3_features = baseline_features + [f for f in group3_family if f not in baseline_features]
res = run_experiment(
    "Exp3: Baseline + Family/Social Structure",
    exp3_features,
    train, y_target
)
results.append(res)

# ──────────────────────────────────────────────
# 6. Summary
# ──────────────────────────────────────────────
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")

summary_df = pd.DataFrame(results)
summary_df["FN_reduction"]    = baseline_fn - summary_df["FN"]
summary_df["savings_usd"]     = summary_df["FN_reduction"] * AVG_LOAN_AMOUNT
summary_df["success_criteria"] = summary_df["FN_reduction"] >= 1000

print(summary_df[[
    "experiment", "n_features", "cv_auc",
    "FN", "FN_reduction", "savings_usd", "success_criteria"
]].to_string(index=False))

# Save results
summary_df.to_csv(os.path.join(OUTPUT_PATH, "ews_experiment_results.csv"), index=False)
print(f"\nResults saved to: {OUTPUT_PATH}/ews_experiment_results.csv")

# Hypothesis validation
print(f"\n{'='*60}")
print("HYPOTHESIS VALIDATION")
print(f"{'='*60}")
fn_reductions = {r["experiment"]: baseline_fn - r["FN"] for r in results[1:]}
for exp, reduction in fn_reductions.items():
    print(f"  {exp}: FN reduction = {reduction:,} cases (${reduction * AVG_LOAN_AMOUNT:,.0f})")

best_exp = max(fn_reductions, key=fn_reductions.get)
print(f"\nBest performing group: {best_exp}")
if "Behavioral" in best_exp:
    print("✓ Hypothesis SUPPORTED: Behavioral patterns reduced FN the most.")
else:
    print("✗ Hypothesis NOT supported: Review results and update explanation.")
