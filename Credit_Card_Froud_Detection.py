
# 1. IMPORT LIBRARIES

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


# 2. LOAD DATASET

df = pd.read_csv("credit_card_fraud_case_study_sample.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

# 3. CHECK TARGET DISTRIBUTION

print("\nTarget Distribution:")
print(df["isFraud"].value_counts())

print("\nTarget Percentage:")
print(df["isFraud"].value_counts(normalize=True) * 100)


# 4. SEPARATE FEATURES AND TARGET

X = df.drop(columns=["isFraud"])
y = df["isFraud"]


# Keep only numerical columns
X = X.select_dtypes(include=np.number)


# Remove ID column if present
if "TransactionID" in X.columns:
    X = X.drop(columns=["TransactionID"])


# 5. HANDLE MISSING VALUES

X = X.fillna(X.median())


print("\nFeatures Used:")
print(X.columns.tolist())


# 6. TRAIN / VALIDATION / TEST SPLIT

# First split:
# 70% Training
# 30% Temporary

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


# Second split:
# 15% Validation
# 15% Test

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)


print("\nData Split:")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)


# 7. FEATURE SCALING

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# 8. APPLY SMOTE ONLY ON TRAINING DATA

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train_scaled,
    y_train
)


print("\nBefore SMOTE:")
print(y_train.value_counts())

print("\nAfter SMOTE:")
print(pd.Series(y_train_smote).value_counts())


# 9. TRAIN XGBOOST MODEL

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)


xgb_model.fit(
    X_train_smote,
    y_train_smote
)


print("\nXGBoost Model Training Completed.")


# 10. VALIDATION PREDICTIONS

y_val_probability = xgb_model.predict_proba(X_val_scaled)[:, 1]


# 11. THRESHOLD OPTIMIZATION

thresholds = np.arange(0.10, 0.91, 0.05)

best_threshold = 0.50
best_f1 = 0


for threshold in thresholds:

    y_val_pred = (
        y_val_probability >= threshold
    ).astype(int)

    f1 = f1_score(
        y_val,
        y_val_pred,
        zero_division=0
    )

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold


print("\nBest Threshold:", best_threshold)
print("Best Validation F1 Score:", best_f1)


# 12. TEST SET PREDICTIONS

y_test_probability = xgb_model.predict_proba(
    X_test_scaled
)[:, 1]


y_test_pred = (
    y_test_probability >= best_threshold
).astype(int)


# 13. XGBOOST EVALUATION

xgb_precision = precision_score(
    y_test,
    y_test_pred,
    zero_division=0
)

xgb_recall = recall_score(
    y_test,
    y_test_pred,
    zero_division=0
)

xgb_f1 = f1_score(
    y_test,
    y_test_pred,
    zero_division=0
)

xgb_roc_auc = roc_auc_score(
    y_test,
    y_test_probability
)

xgb_pr_auc = average_precision_score(
    y_test,
    y_test_probability
)


print("\n==============================")
print("XGBOOST RESULTS")
print("==============================")

print("Precision :", xgb_precision)
print("Recall    :", xgb_recall)
print("F1 Score  :", xgb_f1)
print("ROC-AUC   :", xgb_roc_auc)
print("PR-AUC    :", xgb_pr_auc)


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_test_pred,
        zero_division=0
    )
)


# 14. XGBOOST CONFUSION MATRIX


cm = confusion_matrix(
    y_test,
    y_test_pred
)

print("\nConfusion Matrix:")
print(cm)


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Not Fraud", "Fraud"]
)

disp.plot()
plt.title("XGBoost Confusion Matrix")
plt.show()



# 15. XGBOOST FEATURE IMPORTANCE


feature_importance = pd.Series(
    xgb_model.feature_importances_,
    index=X.columns
)

feature_importance = feature_importance.sort_values(
    ascending=False
)


print("\nTop 10 Important Features:")
print(feature_importance.head(10))


plt.figure(figsize=(10, 6))

feature_importance.head(10).sort_values().plot(
    kind="barh"
)

plt.title("Top 10 XGBoost Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")

plt.tight_layout()
plt.show()


# 16. TRAIN SVM MODEL

svm_model = SVC(
    kernel="rbf",
    C=1.0,
    probability=True,
    class_weight="balanced",
    random_state=42
)


svm_model.fit(
    X_train_scaled,
    y_train
)


print("\nSVM Model Training Completed.")


# 17. SVM PREDICTIONS

y_svm_probability = svm_model.predict_proba(
    X_test_scaled
)[:, 1]


y_svm_pred = (
    y_svm_probability >= 0.50
).astype(int)


# 18. SVM EVALUATION

svm_precision = precision_score(
    y_test,
    y_svm_pred,
    zero_division=0
)

svm_recall = recall_score(
    y_test,
    y_svm_pred,
    zero_division=0
)

svm_f1 = f1_score(
    y_test,
    y_svm_pred,
    zero_division=0
)

svm_roc_auc = roc_auc_score(
    y_test,
    y_svm_probability
)

svm_pr_auc = average_precision_score(
    y_test,
    y_svm_probability
)


print("\n==============================")
print("SVM RESULTS")
print("==============================")

print("Precision :", svm_precision)
print("Recall    :", svm_recall)
print("F1 Score  :", svm_f1)
print("ROC-AUC   :", svm_roc_auc)
print("PR-AUC    :", svm_pr_auc)


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_svm_pred,
        zero_division=0
    )
)


# 19. MODEL COMPARISON

comparison = pd.DataFrame({

    "Metric": [
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC",
        "PR-AUC"
    ],

    "XGBoost": [
        xgb_precision,
        xgb_recall,
        xgb_f1,
        xgb_roc_auc,
        xgb_pr_auc
    ],

    "SVM": [
        svm_precision,
        svm_recall,
        svm_f1,
        svm_roc_auc,
        svm_pr_auc
    ]
})


print("\n==============================")
print("MODEL COMPARISON")
print("==============================")

print(comparison)


# 20. FINAL CONFUSION MATRIX

print("\nFinal XGBoost Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_test_pred
    )
)


# 21. FINAL SUMMARY

print("\n==============================")
print("PROJECT COMPLETED")
print("==============================")

print("Dataset Shape       :", df.shape)
print("Number of Features  :", X.shape[1])
print("Best XGBoost Threshold :", best_threshold)

print("\nXGBoost F1 Score :", xgb_f1)
print("XGBoost ROC-AUC  :", xgb_roc_auc)
print("XGBoost PR-AUC   :", xgb_pr_auc)

print("\nSVM F1 Score :", svm_f1)
print("SVM ROC-AUC  :", svm_roc_auc)
print("SVM PR-AUC   :", svm_pr_auc)

print("\nCredit Card Fraud Detection Pipeline Completed Successfully.")