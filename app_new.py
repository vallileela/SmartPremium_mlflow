# ============================================
# 📦 IMPORTS
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
# ============================================
# ⚙️ PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="Insurance Premium Predictor",
    layout="centered"
)
st.title(" Insurance Premium Predictor")
# ============================================
# 📂 LOAD SAVED PIPELINE
# ============================================

BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
     "final_pipeline.pkl"
)

pipeline = joblib.load(MODEL_PATH)

# ============================================
# 📝 USER INPUTS
# ============================================

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

income = st.number_input(
    "Annual Income",
    min_value=10000,
    max_value=1000000,
    value=50000
)

marital = st.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced"]
)

dependents = st.number_input(
    "Number of Dependents",
    min_value=0,
    max_value=10,
    value=1
)

education = st.selectbox(
    "Education Level",
    [
        "High School",
        "Bachelor's",
        "Master's",
        "PhD"
    ]
)

occupation = st.selectbox(
    "Occupation",
    [
        "Employed",
        "Self-Employed",
        "Unemployed"
    ]
)

health = st.slider(
    "Health Score",
    min_value=0,
    max_value=100,
    value=70
)

location = st.selectbox(
    "Location",
    ["Urban", "Suburban", "Rural"]
)

policy = st.selectbox(
    "Policy Type",
    ["Basic", "Premium", "Comprehensive"]
)

claims = st.number_input(
    "Previous Claims",
    min_value=0,
    max_value=20,
    value=0
)

vehicle_age = st.number_input(
    "Vehicle Age",
    min_value=0,
    max_value=30,
    value=5
)

credit = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=900,
    value=650
)

insurance_duration = st.number_input(
    "Insurance Duration (Years)",
    min_value=0,
    max_value=20,
    value=5
)

smoking = st.selectbox(
    "Smoking Status",
    ["Yes", "No"]
)

exercise = st.selectbox(
    "Exercise Frequency",
    [
        "Daily",
        "Weekly",
        "Monthly",
        "Rarely"
    ]
)

property_type = st.selectbox(
    "Property Type",
    [
        "House",
        "Apartment",
        "Condo"
    ]
)

policy_date = st.date_input(
    "Policy Start Date"
)


# ============================================
# ⚙️ FEATURE ENGINEERING
# ============================================

policy_year = policy_date.year

policy_month = policy_date.month

# ============================================
# 🚀 PREDICTION
# ============================================

if st.button("🚀 Predict Premium"):

    try:

        # ====================================
        # CREATE INPUT DATAFRAME
        # ====================================

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

            "Smoking Status": [smoking],

            "Exercise Frequency": [exercise],

            "Property Type": [property_type],

            "policy_year": [policy_year],

            "policy_month": [policy_month]
        })


        # ====================================
        # PREDICT
        # ====================================

        prediction = pipeline.predict(data)

        prediction = float(prediction[0])


        # ====================================
        # DISPLAY RESULT
        # ====================================

        st.success(
            "✅ Prediction Generated Successfully"
        )

        st.metric(
            label="💰 Estimated Premium",
            value=f"₹ {prediction:,.2f}"
        )


    except Exception as e:

        st.error(
            f"❌ Prediction Failed: {e}"
        )