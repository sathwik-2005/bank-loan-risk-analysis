import streamlit as st
import joblib
import numpy as np

# Load the trained model, scaler, and column order
rf_model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')
model_columns = joblib.load('model_columns.pkl')

st.title("Bank Loan Risk Predictor")
st.write("Enter applicant details to predict loan approval likelihood.")

gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0, value=5000)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0)
loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0, value=100)
loan_term = st.selectbox("Loan Amount Term (days)", [360, 180, 120, 60])
credit_history = st.selectbox("Credit History", ["Good (1)", "Bad (0)"])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

if st.button("Predict"):
    # Encode inputs the same way as training
    gender_val = 1 if gender == "Male" else 0
    married_val = 1 if married == "Yes" else 0
    dependents_val = {"0": 0, "1": 1, "2": 2, "3+": 3}[dependents]
    education_val = 0 if education == "Graduate" else 1
    self_employed_val = 1 if self_employed == "Yes" else 0
    credit_history_val = 1.0 if credit_history == "Good (1)" else 0.0
    property_area_val = {"Rural": 0, "Semiurban": 1, "Urban": 2}[property_area]

    input_data = np.array([[gender_val, married_val, dependents_val, education_val,
                             self_employed_val, applicant_income, coapplicant_income,
                             loan_amount, loan_term, credit_history_val, property_area_val]])

    prediction = rf_model.predict(input_data)[0]
    probability = rf_model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"✅ Loan likely to be APPROVED (confidence: {probability:.1%})")
    else:
        st.error(f"❌ Loan likely to be REJECTED (confidence: {1-probability:.1%})")