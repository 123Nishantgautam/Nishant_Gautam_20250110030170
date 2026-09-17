
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


# 1. Load Dataset

df = pd.read_csv("creditcard.csv")

print(df.head())
print(df["Class"].value_counts())


# 2. Separate Features and Target

X = df.drop("Class", axis=1)
y = df["Class"]


# 3. Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 4. Scale Features

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 5. Apply SMOTE

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("\nBefore SMOTE:")
print(y_train.value_counts())

print("\nAfter SMOTE:")
print(pd.Series(y_train_smote).value_counts())


# 6. XGBoost Model

model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train_smote, y_train_smote)


# 7. Predict Fraud Probability

y_prob = model.predict_proba(X_test)[:, 1]


# 8. ROC-AUC

auc = roc_auc_score(y_test, y_prob)

print("\nXGBoost ROC-AUC:", auc)


# 9. Threshold Tuning

threshold = 0.30

y_pred = (y_prob >= threshold).astype(int)

print("\nThreshold:", threshold)

print("Precision:",
      precision_score(y_test, y_pred))

print("Recall:",
      recall_score(y_test, y_pred))

print("F1 Score:",
      f1_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# 10. Classification Report


print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# 11. Feature Importance

importance = pd.Series(
    model.feature_importances_,
    index=df.drop("Class", axis=1).columns
)

importance = importance.sort_values(ascending=False)

print("\nFeature Importance:")
print(importance.head(10))


# 12. Baseline SVM

svm = SVC(
    kernel="rbf",
    probability=True,
    random_state=42
)

svm.fit(X_train_smote, y_train_smote)

svm_prob = svm.predict_proba(X_test)[:, 1]

svm_auc = roc_auc_score(y_test, svm_prob)

print("\nSVM ROC-AUC:", svm_auc)


# 13. SVM Threshold

svm_threshold = 0.30

svm_pred = (
    svm_prob >= svm_threshold
).astype(int)

print("\nSVM Results")

print("Precision:",
      precision_score(y_test, svm_pred))

print("Recall:",
      recall_score(y_test, svm_pred))

print("F1 Score:",
      f1_score(y_test, svm_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, svm_pred))