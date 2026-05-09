import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import xgboost as xgb

st.set_page_config(page_title="Insurance Premium Predictor", layout="centered")

st.title("💰 Insurance Premium Predictor")

# -----------------------------
# ✅ LOAD MODEL
# -----------------------------
try:
    BASE_DIR = os.path.dirname(__file__)

    # ✅ Load preprocessor
    preprocessor = joblib.load(
        os.path.join(BASE_DIR, "model", "preprocessor_v2.pkl")
    )

    # ✅ Load XGBoost Booster (FINAL SAFE)
    booster = xgb.Booster()
    booster.load_model(
        os.path.join(BASE_DIR, "model", "xgb_model_v2.json")
    )

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

        # ✅ ALIGN FEATURES
        data = data.reindex(columns=preprocessor.feature_names_in_, fill_value=0)

        # ✅ TRANSFORM
        X_processed = preprocessor.transform(data)

        # ✅ CONVERT TO DMATRIX
        dmatrix = xgb.DMatrix(X_processed)

        # ✅ PREDICT
        prediction = booster.predict(dmatrix)

        prediction = prediction.item()

        st.subheader("✅ Prediction Result")
        st.metric("💰 Estimated Premium", f"₹ {prediction:,.2f}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")