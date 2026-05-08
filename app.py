import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from xgboost import XGBRegressor

st.set_page_config(page_title="Insurance Premium Predictor", layout="centered")

st.title("💰 Insurance Premium Predictor")

# -----------------------------
# Load Model (FINAL FIX)
# -----------------------------
try:
    from xgboost import XGBRegressor

    BASE_DIR = os.path.dirname(__file__)

    # ✅ Load pipeline (ONLY preprocessing)
    pipeline = joblib.load(os.path.join(BASE_DIR, "model", "pipeline.pkl"))

    # ✅ Load trained XGBoost
    xgb = XGBRegressor()
    xgb.load_model(os.path.join(BASE_DIR, "model", "xgb_model.json"))

    st.success("✅ Model loaded successfully")

except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.stop()

# -----------------------------
# ✅ USER INPUTS
# -----------------------------
age = st.number_input("Age", 18, 100, 30)
gender = st.selectbox("Gender", ["Male", "Female"])

income = st.number_input("Annual Income", 10000, 1000000, 50000)
marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

dependents = st.number_input("Number of Dependents", 0, 10, 1)

education = st.selectbox(
    "Education Level",
    ["High School", "Bachelor's", "Master's", "PhD"]
)

occupation = st.selectbox("Occupation", ["Employed", "Self-Employed", "Unemployed"])

health = st.slider("Health Score", 0, 100, 70)

location = st.selectbox("Location", ["Urban", "Rural", "Suburban"])
policy = st.selectbox("Policy Type", ["Basic", "Premium", "Comprehensive"])

claims = st.number_input("Previous Claims", 0, 20, 0)

vehicle_age = st.number_input("Vehicle Age", 0, 30, 5)
credit = st.number_input("Credit Score", 300, 900, 650)

insurance_duration = st.number_input("Insurance Duration", 0, 20, 5)

smoking = st.selectbox("Smoking Status", ["Yes", "No"])
exercise = st.selectbox("Exercise Frequency", ["Daily", "Weekly", "Monthly", "Rarely"])

property_type = st.selectbox(
    "Property Type",
    ["House", "Apartment", "Condo"]
)

policy_date = st.date_input("Policy Start Date")

# ✅ Feature engineering
policy_year = policy_date.year
policy_month = policy_date.month

# -----------------------------
# ✅ PREDICT
# -----------------------------
if st.button("🚀 Predict Premium"):
    try:
        data = pd.DataFrame({
            "Age": [age],
            "Gender": [gender],
            "Annual Income": [income],
            "Marital Status": [marital],
            "Number of Dependents": [dependents],
            "Education Level": [education],
            "Occupation": [occupation],
            "Health Score": [health],
            "Location": [location],
            "Policy Type": [policy],
            "Previous Claims": [claims],
            "Vehicle Age": [vehicle_age],
            "Credit Score": [credit],
            "Insurance Duration": [insurance_duration],
            "policy_year": [policy_year],
            "policy_month": [policy_month],
            "Smoking Status": [smoking],
            "Exercise Frequency": [exercise],
            "Property Type": [property_type]
        })

        # ✅ Align features exactly like training
        data = data.reindex(columns=pipeline.feature_names_in_, fill_value=0)
        
        # ✅ Step 1: Preprocess features using pipeline
        X_processed = pipeline.transform(data)

        # ✅ Step 2: Predict using XGBoost directly
        log_pred = xgb.predict(X_processed)


        # ✅ Ensure valid values
        log_pred = np.maximum(log_pred, 0)

        # ✅ Convert log → actual premium
        prediction = np.expm1(log_pred)

        # ✅ Extract scalar
        if isinstance(prediction, np.ndarray):
            prediction = prediction.item()

        # ✅ Display result
        st.subheader("✅ Prediction Result")

        st.metric(
            label="💰 Estimated Premium",
            value=f"₹ {prediction:,.2f}"
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")