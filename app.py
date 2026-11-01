import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved model, scaler and column order
model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.title("Loan Approval Prediction")
st.write("Enter the applicant details to predict whether the loan will be approved.")

col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", [0, 1, 2, 3])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["No", "Yes"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
with col2:
    applicant_income = st.number_input("Applicant Income", min_value=0, value=3500, step=100)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0, step=100)
    loan_amount = st.number_input("Loan Amount (in thousands)", min_value=1, value=100, step=5)
    loan_term = st.selectbox("Loan Amount Term (months)",
                             [12, 36, 60, 84, 120, 180, 240, 300, 360, 480], index=8)
    credit_history = st.selectbox("Credit History (1 = good, 0 = bad)", [1, 0])

# The model only saw values in these ranges (after outlier removal)
if applicant_income > 6783 or coapplicant_income > 5063 or loan_amount > 150 or loan_amount < 36:
    st.warning("Some values are outside the range the model was trained on, so the prediction may be unreliable.")

if st.button("Predict"):
    # Same preprocessing as the notebook
    row = pd.DataFrame([{
        "Gender": 1 if gender == "Male" else 0,
        "Married": 1 if married == "Yes" else 0,
        "Dependents": dependents,
        "Education": 1 if education == "Graduate" else 0,
        "Self_Employed": 1 if self_employed == "Yes" else 0,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "TotalIncome_log": np.log1p(applicant_income + coapplicant_income),
        "Property_Area_Semiurban": 1 if property_area == "Semiurban" else 0,
        "Property_Area_Urban": 1 if property_area == "Urban" else 0,
    }])
    row = row[columns].astype(float)                       # same column order as training
    scale_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount",
                  "Loan_Amount_Term", "TotalIncome_log"]
    row[scale_cols] = scaler.transform(row[scale_cols])    # same scaler as training

    pred = model.predict(row)[0]
    prob = model.predict_proba(row)[0][1]

    if pred == 1:
        st.success(f"Loan likely to be APPROVED (probability {prob:.0%})")
    else:
        st.error(f"Loan likely to be REJECTED (approval probability {prob:.0%})")
