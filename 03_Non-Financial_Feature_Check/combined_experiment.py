"""
EWS Combined Experiment: Alternative Features + Threshold Optimization
=======================================================================
Research Question:
    Does adding non-financial features improve net benefit
    beyond threshold optimization alone?

Experiment Structure:
    - Baseline     : 482 features, threshold sweep
    - Exp1 + Thresh: Baseline + Behavioral patterns, threshold sweep
    - Exp2 + Thresh: Baseline + Education/Occupation, threshold sweep
    - Exp3 + Thresh: Baseline + Family/Social structure, threshold sweep

Key Metric:
    Net Benefit = (FN_reduction × $175,233) - (FP_increase × $175,233 × 18%)
    Optimal threshold = argmax(Net Benefit)
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
# Place train_full_cor.csv / y_full_cor.csv under DATA_PATH,
# and selected_features_483.pkl under MODEL_PATH before running.
DATA_PATH   = "../data"
MODEL_PATH  = "../models"
OUTPUT_PATH = "./results"

os.makedirs(OUTPUT_PATH, exist_ok=True)

AVG_LOAN_AMOUNT   = 175_233
AVG_INTEREST_RATE = 0.18
THRESHOLDS        = np.arange(0.05, 0.55, 0.05)

# ──────────────────────────────────────────────
# 1. Load Data
# ──────────────────────────────────────────────
print("Loading data...")
train    = pd.read_csv(os.path.join(DATA_PATH, "train_full_cor.csv"))
y        = pd.read_csv(os.path.join(DATA_PATH, "y_full_cor.csv"))
y_target = y["TARGET"] if "TARGET" in y.columns else y.iloc[:, 0]

with open(os.path.join(MODEL_PATH, "selected_features_483.pkl"), "rb") as f:
    baseline_features = pickle.load(f)
baseline_features = [f for f in baseline_features if f in train.columns]
print(f"Baseline features: {len(baseline_features)}")

# ──────────────────────────────────────────────
# 2. Feature Groups
# ──────────────────────────────────────────────
group1_behavior = [
    "app_FLAG_DOCUMENT_2",  "app_FLAG_DOCUMENT_4",  "app_FLAG_DOCUMENT_5",
    "app_FLAG_DOCUMENT_6",  "app_FLAG_DOCUMENT_7",  "app_FLAG_DOCUMENT_8",
    "app_FLAG_DOCUMENT_9",  "app_FLAG_DOCUMENT_10", "app_FLAG_DOCUMENT_11",
    "app_FLAG_DOCUMENT_12", "app_FLAG_DOCUMENT_13", "app_FLAG_DOCUMENT_14",
    "app_FLAG_DOCUMENT_15", "app_FLAG_DOCUMENT_16", "app_FLAG_DOCUMENT_17",
    "app_FLAG_DOCUMENT_18", "app_FLAG_DOCUMENT_19", "app_FLAG_DOCUMENT_20",
    "app_FLAG_DOCUMENT_21",
    "app_FLAG_PHONE", "app_FLAG_EMAIL",
    "app_WEEKDAY_APPR_PROCESS_START_MONDAY",
    "app_WEEKDAY_APPR_PROCESS_START_TUESDAY",
    "app_WEEKDAY_APPR_PROCESS_START_WEDNESDAY",
    "app_WEEKDAY_APPR_PROCESS_START_THURSDAY",
    "app_WEEKDAY_APPR_PROCESS_START_SATURDAY",
    "app_WEEKDAY_APPR_PROCESS_START_SUNDAY",
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

group2_edu_job = [
    "app_NAME_EDUCATION_TYPE_Higher_education",
    "app_NAME_EDUCATION_TYPE_Secondary___secondary_special",
    "app_NAME_EDUCATION_TYPE_Incomplete_higher",
    "app_NAME_EDUCATION_TYPE_Lower_secondary",
    "app_NAME_INCOME_TYPE_State_servant",
    "app_NAME_INCOME_TYPE_Commercial_associate",
    "app_NAME_INCOME_TYPE_Pensioner",
    "app_NAME_INCOME_TYPE_Unemployed",
    "app_NAME_INCOME_TYPE_Student",
    "app_NAME_INCOME_TYPE_Maternity_leave",
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
    "app_NAME_CONTRACT_TYPE_Revolving_loans",
    "app_ANNUITY_LENGTH",
    "prev_NAME_CONTRACT_TYPE_Consumer_loans_unique",
    "prev_NAME_CONTRACT_TYPE_Consumer_loans_mode_True",
    "prev_NAME_CONTRACT_TYPE_Revolving_loans_unique",
    "prev_NAME_CONTRACT_TYPE_Revolving_loans_mode_True",
    "prev_NAME_CONTRACT_TYPE_XNA_unique",
    "prev_NAME_CONTRACT_TYPE_XNA_mode_True",
]

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

# ──────────────────────────────────────────────
# 4. Experiment Runner
# ──────────────────────────────────────────────
def run_experiment(exp_name, feature_list, train_df, y_target):
    features = [f for f in feature_list if f in train_df.columns]
    missing  = [f for f in feature_list if f not in train_df.columns]
    if missing:
        print(f"  [WARNING] {len(missing)} features not in dataset, skipped.")
    print(f"\n{'='*60}")
    print(f"Experiment : {exp_name}")
    print(f"Features   : {len(features)}")
    print(f"{'='*60}")

    folds     = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    all_preds = np.zeros(len(train_df))
    all_true  = np.zeros(len(train_df))
    aucs      = []

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
        aucs.append(roc_auc_score(val_y, preds))
        all_preds[val_idx] = preds
        all_true[val_idx]  = val_y.values
        print(f"  Fold {fold+1} AUC: {aucs[-1]:.6f}")

        del trn_x, trn_y, val_x, val_y, model
        gc.collect()

    print(f"  CV AUC: {np.mean(aucs):.6f}")

    # threshold sweep
    ref_row    = None
    th_results = []
    for th in THRESHOLDS:
        preds_bin = (all_preds >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(all_true, preds_bin).ravel()
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        th_results.append({
            "experiment": exp_name,
            "threshold" : round(th, 2),
            "cv_auc"    : round(np.mean(aucs), 6),
            "n_features": len(features),
            "TN": int(tn), "FP": int(fp),
            "FN": int(fn), "TP": int(tp),
            "Recall"   : round(recall, 4),
            "Precision": round(precision, 4),
        })
        if abs(th - 0.50) < 0.001:
            ref_row = th_results[-1]

    return th_results, ref_row


# ──────────────────────────────────────────────
# 5. Run All Experiments
# ──────────────────────────────────────────────
all_results  = []
ref_fn_by_th = {}  # baseline FN per threshold (for net benefit calculation)

# Baseline
res, ref = run_experiment("Baseline", baseline_features, train, y_target)
all_results.extend(res)
# store baseline FN per threshold
for r in res:
    ref_fn_by_th[r["threshold"]] = {"FN": r["FN"], "FP": r["FP"]}

# Exp 1
exp1 = baseline_features + [f for f in group1_behavior if f not in baseline_features]
res, _ = run_experiment("Exp1: +Behavioral", exp1, train, y_target)
all_results.extend(res)

# Exp 2
exp2 = baseline_features + [f for f in group2_edu_job if f not in baseline_features]
res, _ = run_experiment("Exp2: +Education/Occupation", exp2, train, y_target)
all_results.extend(res)

# Exp 3
exp3 = baseline_features + [f for f in group3_family if f not in baseline_features]
res, _ = run_experiment("Exp3: +Family/Social", exp3, train, y_target)
all_results.extend(res)

# ──────────────────────────────────────────────
# 6. Net Benefit Calculation
# ──────────────────────────────────────────────
results_df = pd.DataFrame(all_results)

def calc_net_benefit(row):
    base = ref_fn_by_th.get(row["threshold"], {})
    base_fn = base.get("FN", row["FN"])
    base_fp = base.get("FP", row["FP"])
    fn_reduction = base_fn - row["FN"]
    fp_increase  = row["FP"] - base_fp
    fn_savings   = fn_reduction * AVG_LOAN_AMOUNT
    fp_cost      = fp_increase  * AVG_LOAN_AMOUNT * AVG_INTEREST_RATE
    return {
        "FN_vs_baseline"      : fn_reduction,
        "FP_vs_baseline"      : fp_increase,
        "FN_savings_usd"      : int(fn_savings),
        "FP_cost_usd"         : int(fp_cost),
        "net_benefit_usd"     : int(fn_savings - fp_cost),
    }

extra = results_df.apply(calc_net_benefit, axis=1, result_type="expand")
results_df = pd.concat([results_df, extra], axis=1)

# ──────────────────────────────────────────────
# 7. Summary — optimal threshold per experiment
# ──────────────────────────────────────────────
print(f"\n{'='*60}")
print("OPTIMAL THRESHOLD PER EXPERIMENT")
print(f"{'='*60}")

summary_rows = []
for exp_name in results_df["experiment"].unique():
    sub = results_df[results_df["experiment"] == exp_name]
    best = sub.loc[sub["net_benefit_usd"].idxmax()]
    summary_rows.append({
        "experiment"      : exp_name,
        "optimal_threshold": best["threshold"],
        "cv_auc"          : best["cv_auc"],
        "FN"              : best["FN"],
        "FP"              : best["FP"],
        "FN_vs_baseline"  : best["FN_vs_baseline"],
        "net_benefit_usd" : best["net_benefit_usd"],
    })
    print(f"\n[{exp_name}]")
    print(f"  Optimal threshold : {best['threshold']}")
    print(f"  CV AUC            : {best['cv_auc']}")
    print(f"  FN                : {best['FN']:,}  (vs baseline: {best['FN_vs_baseline']:+,})")
    print(f"  FP                : {best['FP']:,}")
    print(f"  Net Benefit       : ${best['net_benefit_usd']:,.0f}")

summary_df = pd.DataFrame(summary_rows)

# ──────────────────────────────────────────────
# 8. Hypothesis Check
# ──────────────────────────────────────────────
print(f"\n{'='*60}")
print("HYPOTHESIS VALIDATION")
print(f"{'='*60}")

non_base = summary_df[summary_df["experiment"] != "Baseline"]
best_exp = non_base.loc[non_base["net_benefit_usd"].idxmax(), "experiment"]
print(f"Best performing group (net benefit): {best_exp}")
if "Behavioral" in best_exp:
    print("✓ Hypothesis SUPPORTED: Behavioral patterns yielded highest net benefit.")
else:
    print(f"✗ Hypothesis NOT supported. Best group: {best_exp}")

# Save
results_df.to_csv(os.path.join(OUTPUT_PATH, "combined_experiment_full.csv"), index=False)
summary_df.to_csv(os.path.join(OUTPUT_PATH, "combined_experiment_summary.csv"), index=False)
print(f"\nResults saved to: {OUTPUT_PATH}/")
