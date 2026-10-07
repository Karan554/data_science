# ==========================================
# 1. IMPORT LIBRARIES
# ==========================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
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
    RocCurveDisplay
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

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())


# ==========================================
# 4. CHECK TARGET VARIABLE
# ==========================================

print("\nRevenue Distribution:")
print(df["Revenue"].value_counts())

print("\nRevenue Percentage:")
print(df["Revenue"].value_counts(normalize=True) * 100)


# ==========================================
# 5. DEFINE FEATURES (X) AND TARGET (y)
# ==========================================

X = df.drop("Revenue", axis=1)

y = df["Revenue"].astype(int)

print(X.shape)
print(y.shape)

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

print("\nNumerical Features:")
print(numerical_features)

print("\nCategorical Features:")
print(categorical_features)


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

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)


# ==========================================
# 8. PREPROCESSING
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numerical_features
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)


# ==========================================
# 9. CREATE LOGISTIC REGRESSION MODEL
#    (BASELINE - no class balancing)
# ==========================================

model_baseline = Pipeline(
    steps=[
        (
            "preprocessing",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000
            )
        )
    ]
)


# ==========================================
# 9b. CREATE LOGISTIC REGRESSION MODEL
#     (BALANCED - handles class imbalance)
# ==========================================
# class_weight="balanced" automatically increases the
# penalty for misclassifying the minority class (Revenue=1,
# i.e. shoppers who actually purchased). This usually improves
# recall on the minority class at some cost to precision.

model_balanced = Pipeline(
    steps=[
        (
            "preprocessing",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            )
        )
    ]
)


# ==========================================
# 10. TRAIN BOTH MODELS
# ==========================================

model_baseline.fit(X_train, y_train)
model_balanced.fit(X_train, y_train)

print("\nModel Training Completed (Baseline + Balanced)!")


# ==========================================
# 11. MAKE PREDICTIONS
# ==========================================

y_pred_baseline = model_baseline.predict(X_test)
y_prob_baseline = model_baseline.predict_proba(X_test)[:, 1]

y_pred_balanced = model_balanced.predict(X_test)
y_prob_balanced = model_balanced.predict_proba(X_test)[:, 1]


# ==========================================
# 12. MODEL EVALUATION FUNCTION
# ==========================================

def evaluate_model(name, y_test, y_pred, y_prob):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"\n========== {name} PERFORMANCE ==========")
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)
    print("ROC-AUC  :", roc_auc)

    print(f"\n{name} Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print(f"\n{name} Classification Report:")
    print(classification_report(y_test, y_pred))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc
    }


baseline_metrics = evaluate_model(
    "BASELINE", y_test, y_pred_baseline, y_prob_baseline
)

balanced_metrics = evaluate_model(
    "CLASS-BALANCED", y_test, y_pred_balanced, y_prob_balanced
)


# ==========================================
# 13. SIDE-BY-SIDE METRIC COMPARISON
# ==========================================

comparison = pd.DataFrame(
    {
        "Baseline": baseline_metrics,
        "Balanced": balanced_metrics
    }
)

print("\n========== MODEL COMPARISON ==========")
print(comparison)


# ==========================================
# 14. ROC CURVE PLOT (both models)
# ==========================================

fig, ax = plt.subplots(figsize=(7, 6))

RocCurveDisplay.from_predictions(
    y_test, y_prob_baseline, name="Baseline", ax=ax
)
RocCurveDisplay.from_predictions(
    y_test, y_prob_balanced, name="Class-Balanced", ax=ax
)

plt.title("ROC Curve Comparison")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve_comparison.png", dpi=150)
plt.show()


# ==========================================
# 15. SEE ACTUAL VS PREDICTED (balanced model)
# ==========================================

results = X_test.copy()

results["Actual"] = y_test.values
results["Predicted"] = y_pred_balanced
results["Purchase_Probability"] = y_prob_balanced

print("\nPrediction Examples (Balanced Model):")

print(
    results[
        [
            "Actual",
            "Predicted",
            "Purchase_Probability"
        ]
    ].head(20)
)