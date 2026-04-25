import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import os
import datetime
import json

# -------------------------------
# Paths
# -------------------------------
DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "churn.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------------
# Load Dataset
# -------------------------------
data = pd.read_csv(DATA_FILE)

# Convert TotalCharges to numeric & drop missing
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
data = data.dropna()

# Convert target column
data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})

# Features and target
X = data[["tenure", "MonthlyCharges", "TotalCharges"]]
y = data["Churn"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Train Model
# -------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -------------------------------
# Save Model (no extension) with timestamp
# -------------------------------
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_FILE = os.path.join(MODEL_DIR, f"model_{timestamp}")  # no .pkl
joblib.dump(model, MODEL_FILE, compress=0)

# -------------------------------
# Calculate Metrics
# -------------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_prob),
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
}

# Save metrics to JSON
METRICS_FILE = os.path.join(MODEL_DIR, f"metrics_{timestamp}.json")
try:
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"✅ Metrics saved successfully as '{METRICS_FILE}'")
except Exception as e:
    print(f"❌ Failed to save metrics: {e}")
# -------------------------------
# Success Message
# -------------------------------
print(f"✅ Model trained and saved successfully as '{MODEL_FILE}'")
print(f"✅ Metrics saved successfully as '{METRICS_FILE}'")
print("You can now run the Streamlit app to make predictions and visualize results!")