"""
train_model.py
==============
FYP: Student Performance Prediction System
Author: FYP Student

This script:
  1. Loads the synthetic student dataset
  2. Preprocesses features using StandardScaler (data normalization)
  3. Trains a Logistic Regression classifier
  4. Evaluates model performance (accuracy, classification report, confusion matrix)
  5. Saves the trained model and scaler as .pkl files for the Flask API
"""

import pandas as pd
import numpy as np
import pickle
import json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)

# ─────────────────────────────────────────────
# 1. Load Dataset
# ─────────────────────────────────────────────
print("=" * 55)
print("  Student Performance Prediction — Model Training")
print("=" * 55)

df = pd.read_csv("student_data.csv")
print(f"\n[INFO] Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print(df.head(5).to_string())

# ─────────────────────────────────────────────
# 2. Feature / Target Split
# ─────────────────────────────────────────────
FEATURE_COLS = [
    "Study_Hours",
    "Attendance_Rate",
    "Previous_Grades",
    "Participation_Score",
    "Parental_Involvement",
]
TARGET_COL = "Passed"

X = df[FEATURE_COLS].values   # shape (200, 5)
y = df[TARGET_COL].values     # shape (200,)

# ─────────────────────────────────────────────
# 3. Train / Test Split  (80 / 20)
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\n[INFO] Training samples : {X_train.shape[0]}")
print(f"[INFO] Testing  samples : {X_test.shape[0]}")

# ─────────────────────────────────────────────
# 4. Feature Scaling — StandardScaler
#    WHY: Features have very different ranges
#    (Study_Hours: 1–10 vs Attendance_Rate: 60–100).
#    StandardScaler transforms each feature to
#    μ=0, σ=1 so Logistic Regression converges
#    correctly and coefficients are comparable.
# ─────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit on train only
X_test_scaled  = scaler.transform(X_test)        # apply same transform

# ─────────────────────────────────────────────
# 5. Logistic Regression Training
#    max_iter=1000 ensures convergence on our
#    scaled data; C=1.0 is the default
#    regularisation strength.
# ─────────────────────────────────────────────
model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
model.fit(X_train_scaled, y_train)
print("\n[INFO] Logistic Regression model trained successfully.")

# ─────────────────────────────────────────────
# 6. Evaluation
# ─────────────────────────────────────────────
y_pred      = model.predict(X_test_scaled)
y_proba     = model.predict_proba(X_test_scaled)[:, 1]

accuracy    = accuracy_score(y_test, y_pred)
report      = classification_report(y_test, y_pred,
                                    target_names=["Fail", "Pass"])
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"\n{'─'*40}")
print(f"  Model Accuracy : {accuracy * 100:.2f}%")
print(f"{'─'*40}")
print("\n  Classification Report:\n")
print(report)
print("  Confusion Matrix:")
print(conf_matrix)

# ─────────────────────────────────────────────
# 7. Coefficient Analysis
#    (used later by Flask for justification)
#    model.coef_[0] gives one weight per feature
#    after scaling, so magnitudes are comparable.
# ─────────────────────────────────────────────
coef_dict = dict(zip(FEATURE_COLS, model.coef_[0].tolist()))
print("\n  Feature Coefficients (after scaling):")
for feat, coef in sorted(coef_dict.items(), key=lambda x: abs(x[1]), reverse=True):
    bar = "█" * int(abs(coef) * 5)
    sign = "+" if coef >= 0 else "-"
    print(f"    {feat:<25} {sign}{abs(coef):.4f}  {bar}")

# ─────────────────────────────────────────────
# 8. Save artefacts
# ─────────────────────────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Save feature metadata (coefficients + names) for the API justification logic
metadata = {
    "feature_names": FEATURE_COLS,
    "coefficients": coef_dict,
    "accuracy": round(accuracy * 100, 2),
}
with open("model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\n[SUCCESS] Saved: model.pkl | scaler.pkl | model_metadata.json")
print("=" * 55)
print("  Run 'python app.py' to start the Flask server.")
print("=" * 55)
