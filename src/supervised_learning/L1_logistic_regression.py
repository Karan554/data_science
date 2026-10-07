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
    confusion_matrix,
    classification_report
)


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
# ==========================================

model = Pipeline(
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
# 10. TRAIN MODEL
# ==========================================

model.fit(X_train, y_train)

print("\nModel Training Completed!")


# ==========================================
# 11. MAKE PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]


# ==========================================
# 12. MODEL EVALUATION
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)


print("\n========== MODEL PERFORMANCE ==========")

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)


# ==========================================
# 13. CONFUSION MATRIX
# ==========================================

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ==========================================
# 14. COMPLETE CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ==========================================
# 15. SEE ACTUAL VS PREDICTED
# ==========================================

results = X_test.copy()

results["Actual"] = y_test.values
results["Predicted"] = y_pred
results["Purchase_Probability"] = y_probability

print("\nPrediction Examples:")

print(
    results[
        [
            "Actual",
            "Predicted",
            "Purchase_Probability"
        ]
    ].head(20)
)
