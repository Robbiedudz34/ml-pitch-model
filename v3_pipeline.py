import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "vX_Models"

# --- Feature sets ---
feature_cols_v3_groups = [
    "Velo", "Spin", "Spin_theta", "HB", "IVB", "ReleaseExtension",
    "Spin_per_IVB", "Spin_per_HB", "HB_IVB_Ratio",
    "SpinX", "SpinY",
    "Velo_per_Spin", "Velo_per_RelExt",
    "Velo_delta", "Velo_z", "Spin_delta", "Spin_z",
    "Spin_theta_delta", "Spin_theta_z",
    "HB_delta", "HB_z", "IVB_delta", "IVB_z",
    "Hand_encoded"
]

# Changeup, Splitter
feature_cols_v3a = feature_cols_v3_groups[:]

# Fastball, Sinker
feature_cols_v3b = [
    "Spin_theta", "HB", "IVB", "ReleaseExtension",
    "Spin_per_IVB", "Spin_per_HB", "HB_IVB_Ratio",
    "SpinX", "SpinY",
    "Velo_per_Spin", "Velo_per_RelExt",
    "Velo_delta", "Velo_z", "Spin_delta", "Spin_z",
    "Spin_theta_delta", "Spin_theta_z",
    "HB_delta", "HB_z", "IVB_delta", "IVB_z",
    "Hand_encoded"
]

# Breaking Pitches Subgroup
feature_cols_v3c = [
    "Velo", "Spin", "Spin_theta",
    "HB", "IVB", "ReleaseExtension",
    "Spin_per_IVB", "Spin_per_HB", "HB_IVB_Ratio", "IVB_HB_Ratio",
    "SpinX", "SpinY", "Velo_per_Spin", "Velo_per_RelExt",
    "Velo_delta", "Velo_z", "Spin_delta", "Spin_z",
    "Spin_theta_delta", "Spin_theta_z",
    "HB_delta", "HB_z", "IVB_delta", "IVB_z",
    "Move_Mag", "Move_Angle", "HB_per_Velo", "IVB_per_Velo",
    "Hand_encoded"
]

# Cutter, Slider
feature_cols_v3c1 = [
    "Velo", "Spin", "Spin_theta",
    "HB", "IVB", "ReleaseExtension",
    "Spin_per_IVB", "Spin_per_HB", "HB_IVB_Ratio", "IVB_HB_Ratio",
    "SpinX", "SpinY", "Velo_per_Spin", "Velo_per_RelExt",
    "Velo_delta", "Velo_z", "Spin_delta", "Spin_z",
    "Spin_theta_delta", "Spin_theta_z",
    "HB_delta", "HB_z", "IVB_delta", "IVB_z",
    "Move_Mag", "Move_Angle", "HB_per_Velo", "IVB_per_Velo"
]

# Sweeper, Curveball
feature_cols_v3c2 = [
    "Velo", "Spin", "Spin_theta",
    "HB", "IVB", "ReleaseExtension",
    "Spin_per_IVB", "Spin_per_HB", "HB_IVB_Ratio", "IVB_HB_Ratio",
    "SpinX", "SpinY", "Velo_per_Spin", "Velo_per_RelExt",
    "Velo_delta", "Velo_z", "Spin_delta", "Spin_z",
    "Spin_theta_delta", "Spin_theta_z",
    "HB_delta", "HB_z", "IVB_delta", "IVB_z",
    "Move_Mag", "Move_Angle", "HB_per_Velo", "IVB_per_Velo"
]

# Helper Function for Changeup, Splitter Determination
def predict_with_priors(model, X, y_true, pitcher_ids, threshold_low=0.45, threshold_high=0.55):
    """
    Two-step CH/SPLIT prediction:
      - Use LightGBM model probabilities
      - High confidence → accept prediction
      - Uncertain rows → resolve using pitcher-level majority priors
    """
    # Predict probabilities for "Splitter" class (assume binary model)
    y_proba = model.predict(X)
    if y_proba.ndim == 2:  # multiclass safety check
        y_proba = y_proba[:, 1]

    # Assign confident predictions directly
    y_pred = np.full_like(y_true, fill_value=-1)  # -1 = unresolved
    y_pred[y_proba <= threshold_low] = 0  # Changeup
    y_pred[y_proba >= threshold_high] = 1  # Splitter

    # For uncertain predictions, resolve using pitcher priors
    uncertain_mask = y_pred == -1
    if uncertain_mask.any():
        df_uncertain = pd.DataFrame({
            "Pitcher": pitcher_ids[uncertain_mask],
            "True": y_true[uncertain_mask]
        })

        # Build pitcher-level priors from certain rows
        df_certain = pd.DataFrame({
            "Pitcher": pitcher_ids[y_pred != -1],
            "Pred": y_pred[y_pred != -1]
        })
        pitcher_priors = df_certain.groupby("Pitcher")["Pred"].apply(lambda x: int(x.mean() >= 0.5))

        # Apply priors
        resolved_preds = []
        for pid in df_uncertain["Pitcher"]:
            if pid in pitcher_priors.index:
                resolved_preds.append(pitcher_priors.loc[pid])
            else:
                # Default to majority class in training
                resolved_preds.append(0)  # Changeup fallback
        y_pred[uncertain_mask] = resolved_preds

    return y_pred

# --- Load Encoders and Models ---
enc_path = ROOT / "v3_encoders.pkl"
with open(enc_path, "rb") as f:
    encoders = pickle.load(f)

print("Loaded encoders:")
for k, enc in encoders.items():
    print(f"  {k}: {list(enc.classes_)}")

# Load LightGBM boosters - submodels trained already
groups_model = lgb.Booster(model_file=str(MODELS_DIR / "v3_groups_model.txt"))
v3a_model = lgb.Booster(model_file=str(MODELS_DIR / "v3a_model.txt"))
v3b_model = lgb.Booster(model_file=str(MODELS_DIR / "v3b_model.txt"))
v3c_model = lgb.Booster(model_file=str(MODELS_DIR / "v3c_model.txt"))
v3c1_model = lgb.Booster(model_file=str(MODELS_DIR / "v3c1_model.txt"))
v3c2_model = lgb.Booster(model_file=str(MODELS_DIR / "v3c2_model.txt"))

print(f"\nLoaded models: v3_groups, v3a, v3b, v3c, v3c1, v3c2")

# Load test data parquets, created from build_training_features.py
test_groups = pd.read_parquet(ROOT / "test_features_v3_groups.parquet")
test_v3a    = pd.read_parquet(ROOT / "test_features_v3a.parquet")
test_v3b    = pd.read_parquet(ROOT / "test_features_v3b.parquet")
test_v3c = pd.read_parquet(ROOT / "test_features_v3c.parquet")
test_v3c1 = pd.read_parquet(ROOT / "test_features_v3c1.parquet")
test_v3c2 = pd.read_parquet(ROOT / "test_features_v3c2.parquet")

print(f"\nTest datasets → {len(test_groups):,} (groups), {len(test_v3a):,} (CH_SPLIT), {len(test_v3b):,} (FB_SINK), {len(test_v3c):,} (BREAKING), {len(test_v3c1):,} (HARD), {len(test_v3c2):,} (SLOW)")

# --- Phase 1 - Broad Groups ---
X_groups = test_groups[feature_cols_v3_groups]
y_true_groups = test_groups["broad_groups_encoded"]

y_proba_groups = groups_model.predict(X_groups)
y_pred_groups = np.argmax(y_proba_groups, axis=1)
classes_groups = encoders["BroadGroups"].classes_

print("\n=== Broad Groups Report ===\n")
print(classification_report(y_true_groups, y_pred_groups, labels=[0,1,2], target_names=classes_groups, zero_division=0))

# --- Phase 2a - Changeup, Splitter ---
true_chsplit_count = len(test_v3a)

g_labels = encoders["BroadGroups"].inverse_transform(y_pred_groups)
mask_chsplit = g_labels == "CH_SPLIT"
predicted_chsplit_count = mask_chsplit.sum()

print(f"\nCH_SPLIT routing check:")
print(f"   True CH_SPLIT rows in parquet: {true_chsplit_count:,}")
print(f"   Predicted CH_SPLIT rows by pipeline: {predicted_chsplit_count:,}")
print(f"   Lost rows before v3a: {true_chsplit_count - predicted_chsplit_count:,}")

if mask_chsplit.any():
    df_chsplit = test_v3a.copy()
    X_v3a = df_chsplit[feature_cols_v3a]

    # Re-encode ground truth to 0/1
    y_true_v3a = df_chsplit["PitchGroup"].map({"Changeup": 0, "Splitter": 1}).values
    pitcher_ids = df_chsplit["Pitcher"].values if "Pitcher" in df_chsplit.columns else np.arange(len(df_chsplit))

    # Predict with priors
    y_pred_v3a = predict_with_priors(v3a_model, X_v3a, y_true_v3a, pitcher_ids)

    classes_v3a = ["Changeup", "Splitter"]

    print("\n=== CH vs SPLIT Report (with priors) ===\n")
    print(classification_report(y_true_v3a, y_pred_v3a, labels=[0,1], target_names=classes_v3a, zero_division=0))

else:
    print("\nNo CH_SPLIT rows were predicted — skipping v3a evaluation")

# --- Phase 2b - Fastball, Sinker ---
true_fbsink_count = len(test_v3b)

mask_fbsink = g_labels == "FB_SINK"
predicted_fbsink_count = mask_fbsink.sum()

print(f"\nFB_SINK routing check:")
print(f"   True FB_SINK rows in parquet: {true_fbsink_count:,}")
print(f"   Predicted FB_SINK rows by pipeline: {predicted_fbsink_count:,}")
print(f"   Lost rows before v3b: {true_fbsink_count - predicted_fbsink_count:,}")

if mask_fbsink.any():
    df_fbsink = test_v3b.copy()
    X_v3b = df_fbsink[feature_cols_v3b]

    # Re-encode ground truth
    y_true_v3b = df_fbsink["PitchGroup"].map({"Fastball": 0, "Sinker": 1}).values

    # Predict (plain binary)
    y_proba_v3b = v3b_model.predict(X_v3b)
    if y_proba_v3b.ndim == 2:
        y_proba_v3b = y_proba_v3b[:, 1]
    y_pred_v3b = (y_proba_v3b > 0.5).astype(int)

    classes_v3b = ["Fastball", "Sinker"]

    print("\n=== FB vs SINK Report (v3b) ===\n")
    print(classification_report(y_true_v3b, y_pred_v3b, labels=[0,1], target_names=classes_v3b, zero_division=0))

else:
    print("\nNo FB_SINK rows were predicted — skipping v3b evaluation")

# --- Phase 2c - Hard, Slow Breaking Pitches ---
true_breaking_count = len(test_v3c)

mask_breaking = g_labels == "BREAKING"
predicted_breaking_count = mask_breaking.sum()

print(f"\nBREAKING routing check:")
print(f"   True BREAKING rows in parquet: {true_breaking_count:,}")
print(f"   Predicted BREAKING rows by pipeline: {predicted_breaking_count:,}")
print(f"   Lost rows before v3c: {true_breaking_count - predicted_breaking_count:,}")

if mask_breaking.any():
    df_breaking = test_v3c.copy()
    X_v3c = df_breaking[feature_cols_v3c]

    # Re-encode ground truth: HARD=0, SLOW=1
    y_true_v3c = df_breaking["BreakGroup"].map({"HARD": 0, "SLOW": 1}).values

    # Predict
    y_proba_v3c = v3c_model.predict(X_v3c)
    if y_proba_v3c.ndim == 2:
        y_proba_v3c = y_proba_v3c[:, 1]
    y_pred_v3c = (y_proba_v3c > 0.5).astype(int)

    classes_v3c = ["HARD", "SLOW"]

    print("\n=== BREAKING Report (v3c) ===\n")
    print(classification_report(y_true_v3c, y_pred_v3c, labels = [0,1], target_names=classes_v3c, zero_division=0))

else:
    print("\nNo BREAKING rows were predicted — skipping v3c evaluation")

# --- Phase 2c1 - Cutter, Slider ---
true_hard_count = len(test_v3c1)

# Mask from v3c predictions
mask_hard = (y_pred_v3c == 0)  # 0 = HARD
predicted_hard_count = mask_hard.sum()

print(f"\nHARD breaking routing check:")
print(f"   True HARD rows in parquet: {true_hard_count:,}")
print(f"   Predicted HARD rows by pipeline: {predicted_hard_count:,}")
print(f"   Lost rows before v3c1: {true_hard_count - predicted_hard_count:,}")

if mask_hard.any():
    df_hard = test_v3c1.copy()
    X_v3c1 = df_hard[feature_cols_v3c1]

    # Re-encode ground truth: Locally Mismatched - Slider = 0, Cutter = 1
    y_true_v3c1 = df_hard["PitchGroup"].map({"Slider": 0, "Cutter": 1}).values

    # Predict
    y_proba_v3c1 = v3c1_model.predict(X_v3c1)
    if y_proba_v3c1.ndim == 2:
        y_proba_v3c1 = y_proba_v3c1[:, 1]
    y_pred_v3c1 = (y_proba_v3c1 > 0.5).astype(int)

    classes_v3c1 = ["Slider", "Cutter"]

    print("\n=== HARD Breaking Report (v3c1) ===\n")
    print(classification_report(y_true_v3c1, y_pred_v3c1, labels=[0,1], target_names=classes_v3c1, zero_division=0))

else:
    print("\nNo HARD rows were predicted — skipping v3c1 evaluation")

# --- Phase 2c2 - Curveball, Sweeper ---
true_slow_count = len(test_v3c2)

# Mask from v3c predictions
mask_slow = (y_pred_v3c == 1)  # 1 = SLOW
predicted_slow_count = mask_slow.sum()

print(f"\nSLOW breaking routing check:")
print(f"   True SLOW rows in parquet: {true_slow_count:,}")
print(f"   Predicted SLOW rows by pipeline: {predicted_slow_count:,}")
print(f"   Lost rows before v3c2: {true_slow_count - predicted_slow_count:,}")

if mask_slow.any():
    df_slow = test_v3c2.copy()
    X_v3c2 = df_slow[feature_cols_v3c2]

    # Re-encode ground truth into LOCAL 0/1 space
    y_true_v3c2 = df_slow["PitchGroup"].map({"Curveball": 0, "Sweeper": 1}).values

    # Predict
    y_proba_v3c2 = v3c2_model.predict(X_v3c2)
    if y_proba_v3c2.ndim == 2:
        y_proba_v3c2 = y_proba_v3c2[:, 1]
    y_pred_v3c2 = (y_proba_v3c2 > 0.5).astype(int)

    # Local classes
    classes_v3c2 = ["Curveball", "Sweeper"]

    print("\n=== SLOW Breaking Report (v3c2, local labels) ===\n")
    print(classification_report(
        y_true_v3c2,
        y_pred_v3c2,
        labels=[0, 1],
        target_names=classes_v3c2,
        zero_division=0
    ))

else:
    print("\nNo SLOW rows were predicted — skipping v3c2 evaluation")

# --- Phase 3 - Final Unified Eval ---
print("\n=== Final Unified Evaluation: All 8 Pitch Types ===")

# Collect subgroup predictions into their test sets
test_v3a = test_v3a.copy()
test_v3a["y_pred"] = np.where(y_pred_v3a == 0, 0, 6)  # Changeup=0, Splitter=6

test_v3b = test_v3b.copy()
test_v3b["y_pred"] = np.where(y_pred_v3b == 0, 3, 4)  # Fastball=3, Sinker=4

test_v3c1 = test_v3c1.copy()
test_v3c1["y_pred"] = np.where(y_pred_v3c1 == 0, 5, 2)  # Cutter=2, Slider=5, flipped the local back

test_v3c2 = test_v3c2.copy()
test_v3c2["y_pred"] = np.where(y_pred_v3c2 == 0, 1, 7)  # Curveball=1, Sweeper=7

# Merge all subgroups back together
test_full_final = pd.concat([test_v3a, test_v3b, test_v3c1, test_v3c2])

y_true_global = test_full_final["pitch_group_encoded"].values
y_pred_global = test_full_final["y_pred"].values

# Global class names
classes_global = list(encoders["PitchGroup"].classes_)

# Classification report
print(classification_report(
    y_true_global,
    y_pred_global,
    labels=list(range(len(classes_global))),
    target_names=classes_global,
    zero_division=0
))

# Confusion matrix
cm_global = confusion_matrix(
    y_true_global,
    y_pred_global,
    labels=list(range(len(classes_global)))
)
cm_norm_global = cm_global.astype("float") / cm_global.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(10, 8))
sns.heatmap(cm_norm_global, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=classes_global, yticklabels=classes_global)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix – Final 8-Class Pipeline")
plt.tight_layout()
plt.show()