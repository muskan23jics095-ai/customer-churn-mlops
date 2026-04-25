import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import json

st.set_page_config(page_title="Customer Churn Dashboard", page_icon="📊", layout="wide")

st.title("Customer Churn Dashboard")

st.markdown(
"""
This dashboard predicts whether a telecom customer is likely to churn using a machine learning model.
It also provides data exploration and model evaluation insights.
"""
)

# -------------------------------
# Paths
# -------------------------------
PROJECT_ROOT = os.path.dirname(__file__) + "/../"
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_FILE = os.path.join(PROJECT_ROOT, "data", "churn.csv")

# -------------------------------
# Load dataset
# -------------------------------
try:
    data = pd.read_csv(DATA_FILE)
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data = data.dropna()
    data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})
except Exception as e:
    st.error(f"Dataset not found or error loading: {e}")
    st.stop()

# Keep only numeric columns for correlation and plots
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

# -------------------------------
# Load latest model
# -------------------------------
model_files = sorted([f for f in os.listdir(MODEL_DIR) if f.startswith("model")])
if model_files:
    latest_model = os.path.join(MODEL_DIR, model_files[-1])
    try:
        model = joblib.load(latest_model)
        st.success(f"✅ Loaded model: {model_files[-1]}")
    except:
        model = None
        st.warning("⚠️ Model exists but could not be loaded.")
else:
    model = None
    st.warning("⚠️ No model found in models/ folder.")

# -------------------------------
# Load latest metrics
# -------------------------------
metrics_files = sorted([f for f in os.listdir(MODEL_DIR) if f.startswith("metrics") and f.endswith(".json")])
metrics = {}
if metrics_files:
    latest_metrics_file = os.path.join(MODEL_DIR, metrics_files[-1])
    try:
        with open(latest_metrics_file) as f:
            metrics = json.load(f)
        st.success(f"✅ Loaded metrics: {metrics_files[-1]}")
    except Exception as e:
        st.warning(f"⚠️ Could not load metrics: {e}")

# -------------------------------
# Tabs
# -------------------------------
tab1, tab2, tab3 = st.tabs(["Prediction", "Data Exploration", "Model Info"])

# -------- Prediction Tab --------
with tab1:
    st.header("Customer Churn Prediction")
    tenure = st.slider("Tenure (months)", 0, 100, 12)
    monthly = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
    total = st.number_input("Total Charges", min_value=0.0, value=2000.0)

    if st.button("Predict"):
        if model:
            input_df = pd.DataFrame([[tenure, monthly, total]],
                                    columns=["tenure", "MonthlyCharges", "TotalCharges"])
            prediction = model.predict(input_df)
            probability = model.predict_proba(input_df)[0][1]

            st.write(f"Churn probability: **{probability*100:.2f}%**")
            st.progress(int(probability*100))

            if prediction[0] == 1:
                st.error("❌ Customer will churn")
            else:
                st.success("✅ Customer will stay")
        else:
            st.warning("⚠️ Model not loaded. Cannot predict.")

# -------- Data Exploration Tab --------
# -------- Data Exploration Tab --------
with tab2:
    st.header("Dataset Overview")
    st.write("Dataset Shape:", data.shape)
    st.write("Number of Customers:", len(data))
    st.dataframe(data.head(10))

    st.subheader("Churn Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="Churn", data=data, ax=ax)
    st.pyplot(fig)

    st.subheader("Correlation Heatmap (Numeric Features Only)")
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()  # only numeric
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(data[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("Feature Distributions")
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        fig, ax = plt.subplots()
        sns.histplot(data[col], kde=True, color="green", ax=ax)
        st.pyplot(fig)
    csv = data.to_csv(index=False).encode("utf-8")

    st.download_button(
    label="Download Dataset",
    data=csv,
    file_name="churn_data.csv",
    mime="text/csv",
)   
# -------- Model Info Tab --------
with tab3:
    
    st.header("Model Evaluation Metrics")

    if metrics:
        st.write(metrics)

        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(np.array(metrics["confusion_matrix"]), annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

    else:
        st.info("Metrics not found. Run train.py to generate metrics.")

    # ROC Curve
    st.subheader("ROC Curve")

    from sklearn.metrics import roc_curve, auc

    if model:
        X = data[["tenure","MonthlyCharges","TotalCharges"]]
        y = data["Churn"]

        y_prob = model.predict_proba(X)[:,1]

        fpr, tpr, _ = roc_curve(y, y_prob)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        ax.plot([0,1],[0,1],"--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()

        st.pyplot(fig)
    