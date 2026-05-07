#  Insurance Premium Prediction using Machine Learning and MLflow

## 📌 Project Overview

This project predicts insurance premium amounts using Machine Learning techniques. The application helps estimate insurance premiums based on customer and policy details.

The project includes:

* Data Cleaning & Preprocessing
* Machine Learning Model Building
* Model Evaluation
* MLflow Experiment Tracking
* Streamlit Web Application
* Model Deployment Ready Structure

---

# 🚀 Features

✅ End-to-End Machine Learning Pipeline
✅ Data Cleaning and Preprocessing
✅ Multiple Regression Models
✅ MLflow Experiment Tracking
✅ Best Model Selection
✅ Streamlit User Interface
✅ Deployment Ready

---

# 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* MLflow
* Streamlit
* Joblib

---

# 📂 Project Structure

```plaintext
smart_premium/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── model/
│   └── best_model.pkl
│
├── mlruns/
│
├── app.py
├── modelbuilding.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 📊 Machine Learning Models Used

The following regression models were trained and evaluated:

1. Linear Regression
2. Random Forest Regressor
3. XGBoost Regressor

---

# 📈 Model Evaluation Metrics

The models were evaluated using:

* RMSE (Root Mean Squared Error)
* RMSLE (Root Mean Squared Logarithmic Error)
* R² Score

## Final Results

| Model             | RMSE   | R2 Score | RMSLE  |
| ----------------- | ------ | -------- | ------ |
| Linear Regression | 1.0896 | 0.0124   | 0.1645 |
| Random Forest     | 1.0635 | 0.0591   | 0.1609 |
| XGBoost           | 1.0545 | 0.0751   | 0.1594 |

### 🏆 Best Model: XGBoost

---

# ⚙️ MLflow Experiment Tracking

MLflow was used for:

* Tracking model experiments
* Logging parameters
* Logging metrics
* Saving trained models

## Run MLflow UI

Open terminal from project root and run:

python -m mlflow ui --backend-store-uri file:./mlruns --port 5001

Open browser:

http://127.0.0.1:5001


---

# 🌐 Streamlit Application

The project includes a Streamlit web application for predicting insurance premiums.

## Run Streamlit App

Open terminal from project root and run:


streamlit run app.py









