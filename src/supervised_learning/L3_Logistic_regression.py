# ==========================================
# 1. IMPORT LIBRARIES
# ==========================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    precision_recall_curve
)

import matplotlib.pyplot as plt


# ==========================================
# 2. LOAD DATASET
# ==========================================

df = pd.read_csv(r'Datasets\SL_shoppers_purchasing_intention _dataset\online_shoppers_intention.csv')


# ==========================================
# 3. BASIC DATA UNDERSTANDING
# ==========================================

print("Dataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())


# ==========================================
# 4. CHECK TARGET VARIABLE
# ==========================================

print("\nRevenue Percentage:")
print(df["Revenue"].value_counts(normalize=True) * 100)


# ==========================================
# 5. DEFINE FEATURES (X) AND TARGET (y)
# ==========================================

X = df.drop("Revenue", axis=1)
y = df["Revenue"].astype(int)


# ==========================================
# 6. IDENTIFY NUMERICAL AND CATEGORICAL
#    FEATURES
# ==========================================

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "bool"]
).columns.tolist()


# ==========================================
# 7. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)


# ==========================================
# 8. PREPROCESSING
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numerical_features),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)


# ==========================================
# 9. MODELS: BASELINE + CLASS-BALANCED
# ==========================================

model_baseline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)

model_balanced = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ]
)


# ==========================================
# 10. CROSS-VALIDATION (on training data only)
# ==========================================
# 5-fold stratified cross-validation checks how stable each
# model's performance is across different splits of the training
# data, instead of trusting a single train/test split.

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}

print("\n========== CROSS-VALIDATION: BASELINE ==========")
cv_results_baseline = cross_validate(
    model_baseline, X_train, y_train,
    cv=cv_strategy, scoring=scoring
)
for metric in scoring:
    scores = cv_results_baseline[f"test_{metric}"]
    print(f"{metric:10s}: mean={scores.mean():.4f}  std={scores.std():.4f}")

print("\n========== CROSS-VALIDATION: BALANCED ==========")
cv_results_balanced = cross_validate(
    model_balanced, X_train, y_train,
    cv=cv_strategy, scoring=scoring
)
for metric in scoring:
    scores = cv_results_balanced[f"test_{metric}"]
    print(f"{metric:10s}: mean={scores.mean():.4f}  std={scores.std():.4f}")


# ==========================================
# 11. FIT FINAL MODELS ON FULL TRAINING SET
# ==========================================

model_baseline.fit(X_train, y_train)
model_balanced.fit(X_train, y_train)

print("\nModel Training Completed (Baseline + Balanced)!")


# ==========================================
# 12. PREDICTIONS (default 0.5 threshold)
# ==========================================

y_pred_baseline = model_baseline.predict(X_test)
y_prob_baseline = model_baseline.predict_proba(X_test)[:, 1]

y_pred_balanced = model_balanced.predict(X_test)
y_prob_balanced = model_balanced.predict_proba(X_test)[:, 1]


# ==========================================
# 13. EVALUATION FUNCTION
# ==========================================

def evaluate_model(name, y_test, y_pred, y_prob):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"\n========== {name} PERFORMANCE (threshold=0.5) ==========")
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)
    print("ROC-AUC  :", roc_auc)

    print(f"\n{name} Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print(f"\n{name} Classification Report:")
    print(classification_report(y_test, y_pred))

    return {"accuracy": accuracy, "precision": precision,
            "recall": recall, "f1": f1, "roc_auc": roc_auc}


baseline_metrics = evaluate_model("BASELINE", y_test, y_pred_baseline, y_prob_baseline)
balanced_metrics = evaluate_model("CLASS-BALANCED", y_test, y_pred_balanced, y_prob_balanced)

comparison = pd.DataFrame({"Baseline": baseline_metrics, "Balanced": balanced_metrics})
print("\n========== MODEL COMPARISON (threshold=0.5) ==========")
print(comparison)


# ==========================================
# 14. THRESHOLD TUNING (on balanced model)
# ==========================================
# Instead of the default 0.5 cutoff, find the threshold that
# maximizes F1 score on the balanced model's probabilities.

precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob_balanced)

# avoid divide-by-zero
f1_scores = np.divide(
    2 * precisions * recalls,
    precisions + recalls,
    out=np.zeros_like(precisions),
    where=(precisions + recalls) != 0
)

best_idx = np.argmax(f1_scores[:-1])  # last point has no matching threshold
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

print(f"\nBest threshold (max F1) = {best_threshold:.4f}  (F1 = {best_f1:.4f})")

y_pred_tuned = (y_prob_balanced >= best_threshold).astype(int)

print("\n========== BALANCED MODEL @ TUNED THRESHOLD ==========")
print("Accuracy :", accuracy_score(y_test, y_pred_tuned))
print("Precision:", precision_score(y_test, y_pred_tuned))
print("Recall   :", recall_score(y_test, y_pred_tuned))
print("F1 Score :", f1_score(y_test, y_pred_tuned))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_tuned))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_tuned))


# ==========================================
# 15. ROC CURVE PLOT (both models)
# ==========================================

fig, ax = plt.subplots(figsize=(7, 6))
RocCurveDisplay.from_predictions(y_test, y_prob_baseline, name="Baseline", ax=ax)
RocCurveDisplay.from_predictions(y_test, y_prob_balanced, name="Class-Balanced", ax=ax)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve_comparison.png", dpi=150)
plt.show()


# ==========================================
# 16. PRECISION-RECALL CURVE PLOT
# ==========================================
# More informative than ROC for imbalanced datasets like this one.

fig2, ax2 = plt.subplots(figsize=(7, 6))
PrecisionRecallDisplay.from_predictions(y_test, y_prob_baseline, name="Baseline", ax=ax2)
PrecisionRecallDisplay.from_predictions(y_test, y_prob_balanced, name="Class-Balanced", ax=ax2)
plt.title("Precision-Recall Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("precision_recall_curve_comparison.png", dpi=150)
plt.show()


# ==========================================
# 17. SEE ACTUAL VS PREDICTED (tuned threshold)
# ==========================================

results = X_test.copy()
results["Actual"] = y_test.values
results["Predicted_Default"] = y_pred_balanced
results["Predicted_Tuned"] = y_pred_tuned
results["Purchase_Probability"] = y_prob_balanced

print("\nPrediction Examples (Balanced Model, Default vs Tuned Threshold):")
print(results[["Actual", "Predicted_Default", "Predicted_Tuned", "Purchase_Probability"]].head(20))
