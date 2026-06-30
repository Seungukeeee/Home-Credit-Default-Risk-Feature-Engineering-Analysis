"""
Threshold Analysis for EWS Experiment
======================================
Goal: Find optimal threshold that reduces FN significantly
      while keeping FP at an acceptable level.

Current baseline (threshold=0.5):
    FN = 23,700 | Recall = 0.0453 | FP = 826
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

AVG_LOAN_AMOUNT = 175_233

# ──────────────────────────────────────────────
# 1. Load Data
# ──────────────────────────────────────────────
print("Loading data...")
train = pd.read_csv(os.path.join(DATA_PATH, "train_full_cor.csv"))
y     = pd.read_csv(os.path.join(DATA_PATH, "y_full_cor.csv"))
y_target = y["TARGET"] if "TARGET" in y.columns else y.iloc[:, 0]

with open(os.path.join(MODEL_PATH, "selected_features_483.pkl"), "rb") as f:
    baseline_features = pickle.load(f)
baseline_features = [f for f in baseline_features if f in train.columns]
print(f"Baseline features: {len(baseline_features)}")

# ──────────────────────────────────────────────
# 2. Train baseline model & collect OOF predictions
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

print("\nTraining baseline model (5-fold CV)...")
folds     = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
all_preds = np.zeros(len(train))
all_true  = np.zeros(len(train))
aucs      = []

for fold, (trn_idx, val_idx) in enumerate(folds.split(train[baseline_features], y_target)):
    trn_x = train[baseline_features].iloc[trn_idx]
    trn_y = y_target.iloc[trn_idx]
    val_x = train[baseline_features].iloc[val_idx]
    val_y = y_target.iloc[val_idx]

    model = lgb.LGBMClassifier(**lgbm_params)
    model.fit(
        trn_x, trn_y,
        eval_set    = [(val_x, val_y)],
        eval_metric = "auc",
        callbacks   = [log_evaluation(period=1000), early_stopping(stopping_rounds=300)],
    )

    preds = model.predict_proba(val_x)[:, 1]
    auc   = roc_auc_score(val_y, preds)
    aucs.append(auc)
    all_preds[val_idx] = preds
    all_true[val_idx]  = val_y.values
    print(f"  Fold {fold+1} AUC: {auc:.6f}")

    del trn_x, trn_y, val_x, val_y, model
    gc.collect()

print(f"\nCV AUC: {np.mean(aucs):.6f}")

# ──────────────────────────────────────────────
# 3. Threshold Sweep
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("THRESHOLD ANALYSIS")
print("="*60)
print(f"{'Threshold':>10} {'TN':>8} {'FP':>8} {'FN':>8} {'TP':>8} {'Recall':>8} {'Precision':>10} {'FN_reduction':>13} {'Savings_USD':>14}")
print("-"*100)

thresholds   = np.arange(0.05, 0.55, 0.05)
baseline_fn  = None
threshold_results = []

for th in thresholds:
    preds_binary = (all_preds >= th).astype(int)
    tn, fp, fn, tp = confusion_matrix(all_true, preds_binary).ravel()
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    if abs(th - 0.50) < 0.001:
        baseline_fn = fn

    threshold_results.append({
        "threshold" : round(th, 2),
        "TN": int(tn), "FP": int(fp),
        "FN": int(fn), "TP": int(tp),
        "Recall"    : round(recall, 4),
        "Precision" : round(precision, 4),
    })

# compute FN reduction vs threshold=0.5
ref_fn = next(r["FN"] for r in threshold_results if r["threshold"] == 0.50)
for r in threshold_results:
    r["FN_reduction"] = ref_fn - r["FN"]
    r["savings_usd"]  = r["FN_reduction"] * AVG_LOAN_AMOUNT
    print(f"{r['threshold']:>10.2f} {r['TN']:>8,} {r['FP']:>8,} {r['FN']:>8,} {r['TP']:>8,} "
          f"{r['Recall']:>8.4f} {r['Precision']:>10.4f} {r['FN_reduction']:>13,} {r['savings_usd']:>14,.0f}")

# ──────────────────────────────────────────────
# 4. Find optimal threshold
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("OPTIMAL THRESHOLD CANDIDATES")
print("="*60)

results_df = pd.DataFrame(threshold_results)

# Candidate 1: FN reduction >= 1000 (success criterion)
candidates = results_df[results_df["FN_reduction"] >= 1000]
if len(candidates) > 0:
    # Among those, find best precision (minimize FP explosion)
    best = candidates.loc[candidates["Precision"].idxmax()]
    print(f"\n[Target: FN reduction ≥ 1,000]")
    print(f"  Best threshold : {best['threshold']}")
    print(f"  FN             : {best['FN']:,}  (reduction: {best['FN_reduction']:,})")
    print(f"  FP             : {best['FP']:,}")
    print(f"  Recall         : {best['Recall']}")
    print(f"  Precision      : {best['Precision']}")
    print(f"  Estimated savings: ${best['savings_usd']:,.0f}")
else:
    print("\n[Target: FN reduction ≥ 1,000] → No threshold achieves this with current model.")
    print("  Consider retraining with recall-focused objective.")

# Candidate 2: Recall >= 0.3
candidates2 = results_df[results_df["Recall"] >= 0.30]
if len(candidates2) > 0:
    best2 = candidates2.iloc[0]
    print(f"\n[Target: Recall ≥ 0.30]")
    print(f"  Threshold  : {best2['threshold']}")
    print(f"  FN         : {best2['FN']:,}  (reduction: {best2['FN_reduction']:,})")
    print(f"  FP         : {best2['FP']:,}")
    print(f"  Recall     : {best2['Recall']}")
    print(f"  Precision  : {best2['Precision']}")

# Save
results_df.to_csv(os.path.join(OUTPUT_PATH, "threshold_analysis.csv"), index=False)
print(f"\nResults saved to: {OUTPUT_PATH}/threshold_analysis.csv")
