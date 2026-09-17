# 1. Import libraries

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report


# 2. Load dataset

df = pd.read_csv("hospital_readmission.csv")

print(df.head())
print(df.shape)


# 3. Separate features and target

X = df.drop(columns=["readmitted_30d", "patient_id"])
y = df["readmitted_30d"]

numeric = X.select_dtypes(include=["int64", "float64"]).columns
categorical = X.select_dtypes(include=["object"]).columns

print("Numeric columns:")
print(list(numeric))

print("\nCategorical columns:")
print(list(categorical))


# 4. Preprocess data

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric),
    ("categorical", categorical_pipeline, categorical)
])


# 5. Build model

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        penalty="l2",
        C=1.0,
        max_iter=1000
    ))
])


# 6. Split data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# 7. Train model

model.fit(X_train, y_train)

print("Model training completed")


# 8. Make predictions

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Predictions completed")


# 9. Evaluate model

auc = roc_auc_score(y_test, y_prob)

print("ROC-AUC:", auc)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# 10. Analyze errors

cm = confusion_matrix(y_test, y_pred)

false_positive = cm[0][1]
false_negative = cm[1][0]

print("False Positives:", false_positive)
print("False Negatives:", false_negative)

print("\nFalse Negative:")
print("A high-risk patient is predicted as low-risk.")

print("\nFalse Positive:")
print("A low-risk patient is predicted as high-risk.")