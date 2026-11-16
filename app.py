import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("log_model.joblib")
scaler = joblib.load("scaler.joblib")

st.set_page_config(page_title="Credit-Wise Loan Approval", page_icon="💳")
st.title("Credit-Wise Loan Approval Predictor")
st.write("Enter applicant details to get loan approval prediction")

# Input form
with st.form("loan_form"):
    col1, col2 = st.columns(2)

    with col1:
        Applicant_Income = st.number_input("Applicant Income", value=15000.0)
        Coapplicant_Income = st.number_input("Coapplicant Income", value=8000.0)
        Age = st.number_input("Age", value=34, min_value=18)
        Dependents = st.number_input("Dependents", value=1, min_value=0)
        Credit_Score = st.number_input("Credit Score", value=750, min_value=300, max_value=900)
        Existing_Loans = st.number_input("Existing Loans", value=1, min_value=0)
        DTI_Ratio = st.number_input("DTI Ratio", value=0.25, min_value=0.0, max_value=1.0)
        Savings = st.number_input("Savings", value=12000.0)
        Collateral_Value = st.number_input("Collateral Value", value=30000.0)

    with col2:
        Loan_Amount = st.number_input("Loan Amount", value=18000.0)
        Loan_Term = st.number_input("Loan Term (months)", value=36, min_value=12)
        Education_Level = st.number_input("Education Level (0 or 1)", value=1, min_value=0, max_value=1)
        Employment_Status = st.selectbox("Employment Status", ["Salaried", "Self-employed", "Unemployed"])
        Marital_Status = st.selectbox("Marital Status", ["Married", "Single"])
        Loan_Purpose = st.selectbox("Loan Purpose", ["Home", "Personal", "Car", "Education"])
        Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
        Employer_Category = st.selectbox("Employer Category", ["MNC", "Private", "Government", "Unemployed"])
        Gender = st.selectbox("Gender", ["Male", "Female"])

    submitted = st.form_submit_button("Predict")

if submitted:
    # One-hot encoding
    emp_salaried = 1 if Employment_Status == "Salaried" else 0
    emp_self = 1 if Employment_Status == "Self-employed" else 0
    emp_unemp = 1 if Employment_Status == "Unemployed" else 0

    marital_single = 1 if Marital_Status == "Single" else 0

    purpose_car = 1 if Loan_Purpose == "Car" else 0
    purpose_edu = 1 if Loan_Purpose == "Education" else 0
    purpose_home = 1 if Loan_Purpose == "Home" else 0
    purpose_personal = 1 if Loan_Purpose == "Personal" else 0

    area_semi = 1 if Property_Area == "Semiurban" else 0
    area_urban = 1 if Property_Area == "Urban" else 0

    emp_cat_gov = 1 if Employer_Category == "Government" else 0
    emp_cat_mnc = 1 if Employer_Category == "MNC" else 0
    emp_cat_private = 1 if Employer_Category == "Private" else 0
    emp_cat_unemp = 1 if Employer_Category == "Unemployed" else 0

    gender_male = 1 if Gender == "Male" else 0

    features = np.array([[
        Applicant_Income, Coapplicant_Income, Age, Dependents, Credit_Score,
        Existing_Loans, DTI_Ratio, Savings, Collateral_Value, Loan_Amount,
        Loan_Term, Education_Level,
        emp_salaried, emp_self, emp_unemp,
        marital_single,
        purpose_car, purpose_edu, purpose_home, purpose_personal,
        area_semi, area_urban,
        emp_cat_gov, emp_cat_mnc, emp_cat_private, emp_cat_unemp,
        gender_male
    ]])

    features_scaled = scaler.transform(features)
    proba = model.predict_proba(features_scaled)[0][1]
    prediction = "Approved ✅" if proba >= 0.5 else "Rejected ❌"

    st.subheader("Result")
    st.metric("Prediction", prediction)
    st.metric("Approval Probability", f"{proba*100:.1f}%")