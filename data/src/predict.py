import pandas as pd
import numpy as np
import joblib
import os

# Load model
 #Get project root (smart_premium)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# Build correct path
model_path = os.path.join(BASE_DIR, "model", "best_model.pkl")

print("Loading model from:", model_path)

model = joblib.load(model_path)

# Load test data
df_test = pd.read_csv(r"F:\streamlit_session\guvi_projects\smart_premium\data\raw\test.csv")

def clean_and_impute(df):

    if "Policy Start Date" in df.columns:
        df["Policy Start Date"] = pd.to_datetime(df["Policy Start Date"], errors='coerce')
        df["policy_year"] = df["Policy Start Date"].dt.year
        df["policy_month"] = df["Policy Start Date"].dt.month

    df.drop(columns=["Policy Start Date", "Customer Feedback", "id"], inplace=True, errors='ignore')

    num_cols = df.select_dtypes(include="number").columns
    cat_cols = df.select_dtypes(exclude="number").columns

    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df

df_test = clean_and_impute(df_test)

pred_log = model.predict(df_test)
pred = np.expm1(pred_log)

df_test["Predicted Premium"] = pred

df_test.to_csv(r"F:\streamlit_session\guvi_projects\smart_premium\data\processed\test_predictions.csv", index=False)

print("----- FINAL OUTPUT -----")
print(df_test.head())