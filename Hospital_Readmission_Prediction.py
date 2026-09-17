
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

# 1. Load dataset
df = pd.read_csv("hospital_readmission.csv")

# 2. Select features and target
X = df[[
    "age",
    "heart_rate",
    "blood_pressure",
    "prior_visits",
    "diagnosis_code"
]]

y = df["readmitted"]

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Logistic Regression with L2 regularization
model = LogisticRegression(
    penalty="l2",
    C=1.0,
    max_iter=1000
)

# 6. Train model
model.fit(X_train, y_train)

# 7. Predict probability
y_prob = model.predict_proba(X_test)[:, 1]

# 8. ROC-AUC
auc = roc_auc_score(y_test, y_prob)

print("ROC-AUC:", auc)

# 9. Classification report
y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))