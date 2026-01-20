import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
import joblib

# =========================
# LOAD DATASETS
# =========================
benign = pd.read_csv("../data/Benign-Monday.csv")
ddos = pd.read_csv("../data/DDoS-Friday.csv")
botnet = pd.read_csv("../data/Botnet-Friday.csv")
infiltration = pd.read_csv("../data/Infiltration-Thursday.csv")

# =========================
# ADD LABELS
# =========================
benign["label"] = 0          # Normal
ddos["label"] = 1            # DDoS
botnet["label"] = 2          # Botnet
infiltration["label"] = 3    # Infiltration

# =========================
# COMBINE DATA
# =========================
data = pd.concat([benign, ddos, botnet, infiltration], ignore_index=True)

# Remove non-numeric columns (safe for CIC datasets)
data = data.select_dtypes(include=["number"])

X = data.drop("label", axis=1)
y = data["label"]

# =========================
# SPLIT DATA
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

from xgboost import XGBClassifier

temp_model = XGBClassifier(
    objective="multi:softmax",
    num_class=4,
    eval_metric="mlogloss",
    random_state=42
)

temp_model.fit(X_train, y_train)
# adding another one 
import numpy as np

importance = temp_model.feature_importances_
top_features = np.argsort(importance)[-25:]

X_selected = X.iloc[:, top_features]

# Re-split with selected features
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.25, random_state=42, stratify=y
)


# =========================
# SCALE FEATURES
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# XGBOOST MODEL
# =========================
model = XGBClassifier(
    objective="multi:softmax",
    num_class=4,
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    random_state=42
)
print("\nTraining XGBoost model...")
model.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=["Benign", "DDoS", "Botnet", "Infiltration"]
))

# =========================
# SAVE MODEL
# =========================
joblib.dump(model, "xgb_attack_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n✅ Model and scaler saved successfully!")
